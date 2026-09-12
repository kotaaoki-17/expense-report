"""ネットワーク・APIキー不要のユニットテスト。

    python -m unittest discover -s tests -v
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.dedupe import dedupe, normalize_name  # noqa: E402
from pipeline.extract_frames import build_crop_filter, build_select_filter  # noqa: E402
from pipeline.llm import LLMClient  # noqa: E402
from pipeline.outputs import write_outputs  # noqa: E402
from pipeline.structurize import StructuredRecord, normalize_text, rule_based_extract  # noqa: E402


class RuleBasedExtractionTest(unittest.TestCase):
    def test_normalize_text(self):
        self.assertEqual(normalize_text("【速報】　東京都\n計画課長"), "速報 東京都 計画課長")

    def test_government_telop(self):
        result = rule_based_extract("東京都 都市整備局 計画課長 山田 太郎")
        self.assertEqual(result.company, "東京都")
        self.assertEqual(result.name, "山田 太郎")
        self.assertIn("計画課長", result.department)
        self.assertEqual(result.completeness, 1.0)

    def test_company_telop(self):
        result = rule_based_extract("株式会社アーチズ 営業部 部長 鈴木 花子")
        self.assertEqual(result.company, "株式会社アーチズ")
        self.assertEqual(result.name, "鈴木 花子")
        self.assertIn("営業部", result.department)

    def test_surname_colliding_with_org_suffix(self):
        # 「中村」の「村」を自治体語尾と誤認しないこと
        result = rule_based_extract("国立感染症研究所 研究員 中村 美咲")
        self.assertEqual(result.name, "中村 美咲")

    def test_department_is_not_treated_as_name(self):
        result = rule_based_extract("東京都 中央区 広報課")
        self.assertEqual(result.name, "")
        self.assertIn("広報課", result.department)

    def test_no_space_telop(self):
        result = rule_based_extract("東京都都市整備局計画課長山田太郎")
        self.assertEqual(result.company, "東京都")
        self.assertEqual(result.name, "山田太郎")

    def test_unextractable_text(self):
        result = rule_based_extract("近所の住民")
        self.assertEqual(result.completeness, 0.0)

    def test_empty_text(self):
        result = rule_based_extract("")
        self.assertEqual((result.company, result.department, result.name), ("", "", ""))


class DedupeTest(unittest.TestCase):
    def _record(self, timestamp, name, confidence, company="A社", department="営業部"):
        return StructuredRecord(
            video_id="V1",
            timestamp_sec=timestamp,
            company=company,
            department=department,
            name=name,
            raw_text=f"{company} {department} {name}",
            confidence=confidence,
        )

    def test_groups_same_person(self):
        records = [
            self._record(1.0, "山田 太郎", 0.7),
            self._record(5.0, "山田太郎", 0.9),
            self._record(9.0, "鈴木 花子", 0.8),
        ]
        result = dedupe(records)
        self.assertEqual(len(result), 2)
        yamada = next(r for r in result if normalize_name(r.name) == "山田太郎")
        self.assertEqual(yamada.occurrences, 2)
        self.assertEqual(yamada.timestamps, [1.0, 5.0])
        self.assertEqual(yamada.confidence, 0.9)  # 信頼度が高い方を代表に採用

    def test_fills_missing_fields_from_group(self):
        records = [
            self._record(1.0, "山田 太郎", 0.9, company=""),
            self._record(5.0, "山田 太郎", 0.4, company="B社"),
        ]
        result = dedupe(records)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].company, "B社")

    def test_records_with_error_are_skipped(self):
        broken = self._record(1.0, "山田 太郎", 0.9)
        broken.error = "ocr: timeout"
        self.assertEqual(dedupe([broken]), [])


class LLMResponseParsingTest(unittest.TestCase):
    def test_plain_json(self):
        parsed = LLMClient._parse_json('{"company": "A社", "department": "営業部", "name": "山田 太郎"}')
        self.assertEqual(parsed["company"], "A社")

    def test_json_with_preamble(self):
        parsed = LLMClient._parse_json('はい、こちらです。\n{"company":"A社","department":"","name":"山田"}')
        self.assertEqual(parsed["name"], "山田")
        self.assertEqual(parsed["department"], "")

    def test_unparsable_response_returns_none(self):
        self.assertIsNone(LLMClient._parse_json("JSONではない応答"))


class FfmpegFilterTest(unittest.TestCase):
    def test_crop_filter(self):
        self.assertEqual(
            build_crop_filter({"x_ratio": 0.0, "y_ratio": 0.7, "w_ratio": 0.7, "h_ratio": 0.3}),
            "crop=trunc(iw*0.7):trunc(ih*0.3):trunc(iw*0.0):trunc(ih*0.7)",
        )

    def test_select_filter_includes_periodic_sampling(self):
        expression = build_select_filter(0.3, 2.0)
        self.assertIn("gt(scene,0.3)", expression)
        self.assertIn("gte(t-prev_selected_t,2.0)", expression)

    def test_select_filter_without_periodic_sampling(self):
        self.assertNotIn("prev_selected_t,", build_select_filter(0.3, 0))


class OutputTest(unittest.TestCase):
    def test_writes_json_and_csv(self):
        records = [
            StructuredRecord(
                video_id="V1",
                timestamp_sec=3.0,
                company="A社",
                department="営業部",
                name="山田 太郎",
                raw_text="A社 営業部 山田 太郎",
                confidence=0.9,
            )
        ]
        with tempfile.TemporaryDirectory() as tmp:
            written = write_outputs(records, "V1", Path(tmp))
            payload = json.loads(written["json"].read_text(encoding="utf-8"))
            self.assertEqual(payload[0]["name"], "山田 太郎")
            self.assertIn("山田 太郎", written["csv"].read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    unittest.main()
