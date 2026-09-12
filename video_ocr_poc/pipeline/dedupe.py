"""Step 5: 重複排除（名寄せ）。

同一動画内で同じ人物が複数フレームに現れるため、氏名（完全一致 or 編集距離が近いもの）で
グルーピングし、グループ内で最も信頼度の高いレコードを代表値として採用する。
出現した timestamp のリストも保持する。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, asdict, field
from difflib import SequenceMatcher
from typing import Any, Iterable

logger = logging.getLogger(__name__)


@dataclass
class DedupedRecord:
    """名寄せ後の1人分のレコード。"""

    video_id: str
    timestamp_sec: float
    company: str = ""
    department: str = ""
    name: str = ""
    raw_text: str = ""
    source: str = "rule"
    confidence: float = 0.0
    timestamps: list[float] = field(default_factory=list)
    occurrences: int = 1
    frame_path: str = ""
    field_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_name(name: str) -> str:
    """比較用に氏名を正規化する（空白・記号を除去）。"""
    return re.sub(r"[\s・,.,．]", "", name or "")


def similarity(left: str, right: str) -> float:
    """0.0〜1.0 の類似度（編集距離ベース）。"""
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    return SequenceMatcher(None, left, right).ratio()


def _group_key_for_unnamed(record: Any) -> str:
    """氏名が取れなかったレコードは組織＋部署で寄せる。"""
    company = (getattr(record, "company", "") or "").strip()
    department = (getattr(record, "department", "") or "").strip()
    raw = (getattr(record, "raw_text", "") or "").strip()
    return f"__unnamed__:{company}/{department}" if (company or department) else f"__raw__:{raw}"


def dedupe(
    records: Iterable[Any],
    name_similarity_threshold: float = 0.8,
) -> list[DedupedRecord]:
    """構造化レコードを名寄せして代表レコードのリストを返す。"""
    groups: list[dict[str, Any]] = []

    skipped_empty = 0
    for record in records:
        if getattr(record, "error", None):
            continue
        # テロップが写っていなかったフレーム（3項目とも空）は出力対象外
        if not any((getattr(record, key, "") or "").strip() for key in ("company", "department", "name")):
            skipped_empty += 1
            continue
        name = normalize_name(getattr(record, "name", ""))
        target: dict[str, Any] | None = None

        if name:
            best_score = 0.0
            for group in groups:
                if not group["name_key"]:
                    continue
                score = similarity(name, group["name_key"])
                if score >= name_similarity_threshold and score > best_score:
                    best_score, target = score, group
        else:
            key = _group_key_for_unnamed(record)
            target = next((group for group in groups if group["fallback_key"] == key), None)

        if target is None:
            target = {
                "name_key": name,
                "fallback_key": _group_key_for_unnamed(record),
                "members": [],
            }
            groups.append(target)
        target["members"].append(record)

    deduped: list[DedupedRecord] = []
    for group in groups:
        members = sorted(group["members"], key=lambda r: float(getattr(r, "confidence", 0.0) or 0.0), reverse=True)
        best = members[0]
        merged = DedupedRecord(
            video_id=getattr(best, "video_id", ""),
            timestamp_sec=float(getattr(best, "timestamp_sec", 0.0)),
            company=getattr(best, "company", "") or "",
            department=getattr(best, "department", "") or "",
            name=getattr(best, "name", "") or "",
            raw_text=getattr(best, "raw_text", "") or "",
            source=getattr(best, "source", "rule"),
            confidence=round(float(getattr(best, "confidence", 0.0) or 0.0), 3),
            timestamps=sorted({round(float(getattr(m, "timestamp_sec", 0.0)), 2) for m in members}),
            occurrences=len(members),
            frame_path=getattr(best, "frame_path", "") or "",
            field_sources=dict(getattr(best, "field_sources", {}) or {}),
        )
        # 代表レコードで欠けている項目は、同一グループの他フレームから補完する
        for key in ("company", "department", "name"):
            if getattr(merged, key):
                continue
            for member in members[1:]:
                value = getattr(member, key, "") or ""
                if value:
                    setattr(merged, key, value)
                    merged.field_sources[key] = (getattr(member, "field_sources", {}) or {}).get(key, "rule")
                    break
        deduped.append(merged)

    deduped.sort(key=lambda record: record.timestamp_sec)
    logger.info(
        "名寄せ完了: %d 件 → %d 件 (テロップ無しとして除外 %d 件)",
        sum(len(group["members"]) for group in groups),
        len(deduped),
        skipped_empty,
    )
    return deduped
