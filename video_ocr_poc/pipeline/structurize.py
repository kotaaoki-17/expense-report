"""Step 4: 構造化抽出（会社名・部署名・氏名の分離）。

ハイブリッド方式:
  1. ルールベース（正規表現 + 辞書）で一次判定
  2. 確信が持てなかったフレームだけ Claude API に投げて補完（LLMフォールバック）

どの項目がルール確定でどの項目がLLM補完かを field_sources に残し、後段の精度検証に使えるようにする。
中間結果は ./data/structured/{video_id}.jsonl に保存し、再実行時は未処理分のみ処理する。
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

from .dictionaries import (
    COMMON_SURNAMES,
    COMPANY_PREFIXES,
    COMPANY_SUFFIXES,
    DEPARTMENT_SUFFIXES,
    GOVERNMENT_SUFFIXES,
    NOISE_TOKENS,
    NON_NAME_KEYWORDS,
    TITLES,
)
from .llm import LLMClient, create_llm_client

logger = logging.getLogger(__name__)

# 項目ごとの重み（すべて揃うと 1.0）
WEIGHTS = {"company": 0.4, "department": 0.3, "name": 0.3}

_KANJI = r"一-龥々ヶ"
_KANA = r"ぁ-んァ-ヴー"

_SORTED_TITLES = tuple(sorted(TITLES, key=len, reverse=True))
_TITLE_RE = re.compile("|".join(re.escape(title) for title in _SORTED_TITLES))

_COMPANY_PATTERNS = [
    # 株式会社◯◯ のような接頭辞型
    re.compile(
        r"(?:" + "|".join(re.escape(p) for p in COMPANY_PREFIXES) + r")[" + _KANJI + _KANA + r"A-Za-z0-9・\.]{0,14}"
    ),
    # ◯◯株式会社 / ◯◯銀行 のような接尾辞型
    re.compile(
        r"[" + _KANJI + _KANA + r"A-Za-z0-9（）\(\)・\.]{1,16}?(?:"
        + "|".join(re.escape(s) for s in sorted(COMPANY_SUFFIXES, key=len, reverse=True))
        + r")"
    ),
    # 東京都 / 横浜市 / 厚生労働省 のような行政機関型
    re.compile(
        r"[" + _KANJI + r"]{1,6}?(?:" + "|".join(re.escape(s) for s in GOVERNMENT_SUFFIXES) + r")"
    ),
]

_DEPARTMENT_RE = re.compile(
    r"[" + _KANJI + _KANA + r"A-Za-z0-9・]{1,12}?(?:"
    + "|".join(re.escape(s) for s in sorted(DEPARTMENT_SUFFIXES, key=len, reverse=True))
    + r")"
)

# 「山田 太郎」のようにスペース区切りの氏名
_SPACED_NAME_RE = re.compile(r"([" + _KANJI + r"]{1,4}|[ァ-ヴー]{2,8})[ 　]([" + _KANJI + _KANA + r"]{1,4}|[ァ-ヴー]{2,8})")
# 「・」区切りのカタカナ氏名（外国人名）
_KATAKANA_NAME_RE = re.compile(r"[ァ-ヴー]{2,10}・[ァ-ヴー]{2,10}")
# 役職の直後に続く氏名
_TITLE_NAME_RE = re.compile(r"(?:" + "|".join(re.escape(t) for t in _SORTED_TITLES) + r")[ 　]*([" + _KANJI + r"]{2,5})")


class StructurizeError(RuntimeError):
    """構造化抽出の致命的な失敗（通常は発生させず、ログ出力して継続する）。"""


@dataclass
class StructuredRecord:
    """1フレーム分の構造化結果。"""

    video_id: str
    timestamp_sec: float
    company: str = ""
    department: str = ""
    name: str = ""
    raw_text: str = ""
    source: str = "rule"  # rule | llm | none
    confidence: float = 0.0
    frame_path: str = ""
    field_sources: dict[str, str] = field(default_factory=dict)
    ocr_confidence: float = 0.0
    rule_completeness: float = 0.0
    llm_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RuleResult:
    company: str = ""
    department: str = ""
    name: str = ""
    completeness: float = 0.0
    ambiguous: bool = False
    leftover: str = ""


# --------------------------------------------------------------------------------------
# 正規化とルールベース抽出
# --------------------------------------------------------------------------------------
def normalize_text(text: str) -> str:
    """全角英数の正規化・ノイズ記号の除去・空白の整理を行う。"""
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", text)
    for token in NOISE_TOKENS:
        normalized = normalized.replace(token, " ")
    normalized = normalized.replace("　", " ").replace("\n", " ").replace("\t", " ")
    normalized = re.sub(r"[|｜/／,，:：]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


# 「中村」「西村」のように組織語の1文字と衝突する姓があるため、1文字キーワードは別扱いにする
_MULTI_CHAR_KEYWORDS = tuple(keyword for keyword in NON_NAME_KEYWORDS if len(keyword) >= 2)
_SINGLE_CHAR_KEYWORDS = tuple(keyword for keyword in NON_NAME_KEYWORDS if len(keyword) == 1)
_SURNAME_SET = frozenset(COMMON_SURNAMES)


def _looks_like_name(candidate: str) -> bool:
    """組織語・役職語を含む文字列は氏名とみなさない。"""
    if not candidate or len(candidate) > 12:
        return False
    if any(keyword in candidate for keyword in _MULTI_CHAR_KEYWORDS):
        return False
    # 「広報課」「中央区」のように1文字の組織語で終わる語は氏名から除外する。
    # ただし「中村」「西村」など既知の姓は許容する。
    if candidate.endswith(_SINGLE_CHAR_KEYWORDS) and candidate not in _SURNAME_SET:
        return False
    return True


_NAME_TOKEN_RE = re.compile(r"^(?:[" + _KANJI + r"]{1,5}|[ァ-ヴー]{2,10})$")


def _tokenize(text: str) -> list[tuple[str, int, int]]:
    """空白区切りのトークンを (文字列, 開始位置, 終了位置) で返す。"""
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def _is_name_token(token: str) -> bool:
    """氏名の構成要素（姓 or 名）になりうるトークンか。"""
    return bool(_NAME_TOKEN_RE.match(token)) and _looks_like_name(token)



def extract_company(text: str) -> tuple[str, tuple[int, int] | None]:
    """会社名・組織名を1件抽出し、(値, テキスト上の範囲) を返す。"""
    best: tuple[int, int] | None = None
    for pattern in _COMPANY_PATTERNS:
        match = pattern.search(text)
        if match and match.group(0).strip():
            span = match.span()
            # より左側（テロップでは組織名が先に来る）を優先し、同位置なら長い方を採用
            if best is None or span[0] < best[0] or (span[0] == best[0] and span[1] > best[1]):
                best = span
    if best is None:
        return "", None
    return text[best[0] : best[1]].strip(), best


def extract_name(text: str) -> tuple[str, tuple[int, int] | None, bool]:
    """氏名を抽出し、(値, 範囲, 候補が複数あったか) を返す。

    テロップは「組織 部署 役職 姓 名」の語順が多いため、
    空白区切りのトークン列から「姓+名」のペアを探すことを最優先する。
    """
    candidates: list[tuple[str, tuple[int, int]]] = []

    # 1) 空白区切りの「姓 名」ペア
    for left, right in zip(_tokenize(text), _tokenize(text)[1:]):
        if not (_is_name_token(left[0]) and _is_name_token(right[0])):
            continue
        candidates.append((f"{left[0]} {right[0]}", (left[1], right[2])))

    # 2) 「ジョン・スミス」のようなカタカナ氏名
    if not candidates:
        for match in _KATAKANA_NAME_RE.finditer(text):
            if _looks_like_name(match.group(0)):
                candidates.append((match.group(0).strip(), match.span()))

    # 3) 役職語の直後に続く漢字列（空白が無いテロップ向け）
    if not candidates:
        for match in _TITLE_NAME_RE.finditer(text):
            if _looks_like_name(match.group(1)):
                candidates.append((match.group(1).strip(), match.span(1)))

    # 4) 末尾の漢字2〜5文字が頻出姓で始まる場合
    if not candidates:
        tail = re.search(r"([" + _KANJI + r"]{2,5})\s*$", text)
        if tail and _looks_like_name(tail.group(1)):
            value = tail.group(1)
            if any(value.startswith(surname) for surname in COMMON_SURNAMES):
                candidates.append((value, tail.span(1)))

    if not candidates:
        return "", None, False

    unique = {value for value, _ in candidates}
    # テロップでは氏名が最後に来るため、最も後方の候補を採用
    value, span = max(candidates, key=lambda item: item[1][0])
    return value, span, len(unique) > 1


def extract_department(text: str) -> str:
    """部署名（役職を含む）を抽出する。役職語のみでも部署情報として採用する。

    「計画課」+「課長」のように重なったマッチは1つに結合し、「計画課長」として復元する。
    """
    spans: list[tuple[int, int]] = [match.span() for match in _DEPARTMENT_RE.finditer(text)]
    spans += [match.span() for match in _TITLE_RE.finditer(text)]
    if not spans:
        return ""

    merged: list[list[int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    parts = [text[start:end].strip() for start, end in merged]
    return " ".join(part for part in parts if part).strip()


def _mask(text: str, span: tuple[int, int] | None) -> str:
    if span is None:
        return text
    return (text[: span[0]] + " " + text[span[1] :]).strip()


def rule_based_extract(raw_text: str) -> RuleResult:
    """正規表現・辞書によるルールベース一次判定。"""
    text = normalize_text(raw_text)
    if not text:
        return RuleResult()

    company, company_span = extract_company(text)
    remainder = _mask(text, company_span)

    name, name_span, ambiguous = extract_name(remainder)
    remainder_wo_name = _mask(remainder, name_span)

    department = extract_department(remainder_wo_name)

    completeness = sum(
        weight for key, weight in WEIGHTS.items() if {"company": company, "department": department, "name": name}[key]
    )
    leftover = re.sub(r"\s+", " ", _TITLE_RE.sub(" ", remainder_wo_name)).strip()
    return RuleResult(
        company=company,
        department=department,
        name=name,
        completeness=round(completeness, 3),
        ambiguous=ambiguous,
        leftover=leftover,
    )


# --------------------------------------------------------------------------------------
# Step 4 本体
# --------------------------------------------------------------------------------------
def structured_jsonl_path(structured_dir: Path, video_id: str) -> Path:
    return structured_dir / f"{video_id}.jsonl"


def load_structured(structured_dir: Path, video_id: str) -> list[StructuredRecord]:
    """保存済みの構造化結果を読み込む（Step 5 以降を単独実行する場合に使用）。"""
    path = structured_jsonl_path(structured_dir, video_id)
    if not path.exists():
        return []
    records: list[StructuredRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            records.append(
                StructuredRecord(
                    **{k: payload.get(k) for k in StructuredRecord.__dataclass_fields__ if k in payload}
                )
            )
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("構造化結果の %d 行目を読み飛ばしました: %s", line_number, exc)
    return records


def structurize(
    ocr_results: Iterable[Any],
    video_id: str,
    structured_dir: Path,
    llm_client: LLMClient | None = None,
    structurize_config: dict[str, Any] | None = None,
    force: bool = False,
) -> list[StructuredRecord]:
    """OCR結果を {company, department, name} に構造化する。"""
    config = structurize_config or {}
    min_rule_confidence = float(config.get("min_rule_confidence", 0.7))
    use_llm = bool(config.get("use_llm_fallback", True))

    structured_dir.mkdir(parents=True, exist_ok=True)
    path = structured_jsonl_path(structured_dir, video_id)

    cached: dict[str, StructuredRecord] = {}
    if path.exists() and not force:
        cached = {record.frame_path: record for record in load_structured(structured_dir, video_id)}
        if cached:
            logger.info("構造化済みフレームを再利用します: %d 件", len(cached))
    elif path.exists() and force:
        path.unlink()

    if llm_client is None and use_llm:
        llm_client = create_llm_client(config)

    records: list[StructuredRecord] = []
    llm_calls = 0
    with path.open("a", encoding="utf-8") as handle:
        for result in ocr_results:
            frame_path = str(getattr(result, "frame_path", ""))
            raw_text = getattr(result, "raw_text", "") or ""
            previous = cached.get(frame_path)
            if previous is not None and previous.raw_text == raw_text:
                records.append(previous)
                continue

            record = StructuredRecord(
                video_id=video_id,
                timestamp_sec=float(getattr(result, "timestamp_sec", 0.0)),
                raw_text=raw_text,
                frame_path=frame_path,
                ocr_confidence=float(getattr(result, "confidence", 0.0) or 0.0),
            )
            ocr_error = getattr(result, "error", None)
            if ocr_error:
                record.error = f"ocr: {ocr_error}"
                record.source = "none"
                handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
                records.append(record)
                continue

            try:
                rule = rule_based_extract(raw_text)
            except Exception as exc:  # ルールの想定外入力でも止めない
                logger.warning("ルールベース抽出に失敗しました (t=%.2fs): %s", record.timestamp_sec, exc)
                rule = RuleResult()
                record.error = f"rule: {exc}"[:300]

            record.company = rule.company
            record.department = rule.department
            record.name = rule.name
            record.rule_completeness = rule.completeness
            record.field_sources = {
                key: ("rule" if value else "none")
                for key, value in (("company", rule.company), ("department", rule.department), ("name", rule.name))
            }

            needs_llm = rule.completeness < min_rule_confidence or rule.ambiguous
            if needs_llm and use_llm and llm_client is not None and llm_client.enabled and raw_text.strip():
                extracted = llm_client.extract(raw_text)
                llm_calls += 1
                if extracted:
                    record.llm_used = True
                    for key in ("company", "department", "name"):
                        value = extracted.get(key, "").strip()
                        current = getattr(record, key)
                        # ルールで取れなかった項目のみLLMで補完する（ルール確定値は上書きしない）
                        if not current and value:
                            setattr(record, key, value)
                            record.field_sources[key] = "llm"
                        elif rule.ambiguous and value and value != current:
                            setattr(record, key, value)
                            record.field_sources[key] = "llm"

            filled = sum(
                weight for key, weight in WEIGHTS.items() if getattr(record, key)
            )
            record.source = "llm" if record.llm_used else ("rule" if filled else "none")
            base_confidence = record.ocr_confidence if record.ocr_confidence > 0 else 0.9
            # LLM補完分はルール確定よりわずかに割り引く
            penalty = 0.95 if record.llm_used else 1.0
            record.confidence = round(base_confidence * filled * penalty, 3)

            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            records.append(record)

    logger.info(
        "構造化完了: %d 件 (LLM呼び出し %d 件 / うち補完成功 %d 件)",
        len(records),
        llm_calls,
        sum(1 for record in records if record.llm_used),
    )
    return records
