"""Step 6: 出力（JSON / CSV）。

./data/output/{video_id}.json と ./data/output/{video_id}.csv を生成する。
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

# 指示書のスキーマ（先頭）＋ PoC の検証に便利な補助項目
BASE_FIELDS = [
    "video_id",
    "timestamp_sec",
    "company",
    "department",
    "name",
    "raw_text",
    "source",
    "confidence",
]
EXTRA_FIELDS = ["timestamps", "occurrences", "frame_path", "field_sources"]


def to_rows(records: Iterable[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        data = record.to_dict() if hasattr(record, "to_dict") else dict(record)
        row = {key: data.get(key, "") for key in BASE_FIELDS}
        for key in EXTRA_FIELDS:
            row[key] = data.get(key, "")
        rows.append(row)
    return rows


def write_outputs(
    records: Iterable[Any],
    video_id: str,
    output_dir: Path,
    formats: list[str] | None = None,
) -> dict[str, Path]:
    """JSON / CSV を書き出し、{形式: パス} を返す。"""
    formats = formats or ["json", "csv"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = to_rows(records)
    written: dict[str, Path] = {}

    if "json" in formats:
        json_path = output_dir / f"{video_id}.json"
        json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        written["json"] = json_path

    if "csv" in formats:
        csv_path = output_dir / f"{video_id}.csv"
        flat_rows = [
            {
                **row,
                "timestamps": ";".join(str(value) for value in (row.get("timestamps") or [])),
                "field_sources": json.dumps(row.get("field_sources") or {}, ensure_ascii=False),
            }
            for row in rows
        ]
        try:
            import pandas as pd

            pd.DataFrame(flat_rows, columns=BASE_FIELDS + EXTRA_FIELDS).to_csv(
                csv_path, index=False, encoding="utf-8-sig"
            )
        except ImportError:  # pandas が無い環境でも出力できるようにする
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=BASE_FIELDS + EXTRA_FIELDS)
                writer.writeheader()
                writer.writerows(flat_rows)
        written["csv"] = csv_path

    logger.info("出力完了: %s", ", ".join(str(path) for path in written.values()))
    return written
