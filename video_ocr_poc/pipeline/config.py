"""設定の読み込みとデータディレクトリの管理。

設定は `config.json`（`--config` で差し替え可能）から読み込み、
秘匿情報（APIキー）は `.env` / 環境変数から読み込みます。
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

DEFAULT_CONFIG: dict[str, Any] = {
    "download": {
        "max_height": 480,
        "format_fallback": "best",
    },
    "frames": {
        "scene_threshold": 0.3,
        "max_gap_sec": 2.0,
        # テロップ領域（画面全体に対する比率）。動画に合わせて config.json で調整する。
        "crop": {"x_ratio": 0.0, "y_ratio": 0.70, "w_ratio": 0.70, "h_ratio": 0.30},
        "phash_distance_threshold": 6,
        "min_edge_density": 0.004,
        "max_frames": 500,
    },
    "ocr": {
        "provider": None,  # None の場合は環境変数から自動判定
        "language_hints": ["ja"],
        "timeout_sec": 30,
        "max_retries": 2,
    },
    "structurize": {
        "use_llm_fallback": True,
        "min_rule_confidence": 0.7,
        "llm_max_calls": 100,
    },
    "dedupe": {
        "name_similarity_threshold": 0.8,
    },
    "output": {
        "formats": ["json", "csv"],
    },
}


@dataclass
class Paths:
    """中間成果物の保存先。Stepごとにファイルを残し、途中から再実行できるようにする。"""

    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def ocr_raw(self) -> Path:
        return self.root / "ocr_raw"

    @property
    def structured(self) -> Path:
        return self.root / "structured"

    @property
    def output(self) -> Path:
        return self.root / "output"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    def ensure(self) -> None:
        for path in (self.raw, self.frames, self.ocr_raw, self.structured, self.output, self.logs):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    values: dict[str, Any] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CONFIG))
    paths: Paths = field(default_factory=lambda: Paths(PROJECT_ROOT / "data"))
    source_path: Path | None = None

    def section(self, name: str) -> dict[str, Any]:
        return self.values.get(name, {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.values.get(section, {}).get(key, default)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None, data_dir: str | Path | None = None) -> Config:
    """config.json と .env を読み込んで Config を返す。

    設定ファイルが存在しない／壊れている場合はデフォルト値で続行する（PoCを止めない）。
    """
    load_dotenv_if_available()

    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    values = copy.deepcopy(DEFAULT_CONFIG)
    if path.exists():
        try:
            values = _deep_merge(values, json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("設定ファイルを読み込めませんでした (%s): %s / デフォルト値で続行します", path, exc)
    elif config_path:
        logger.warning("設定ファイルが見つかりません: %s / デフォルト値で続行します", path)

    root = Path(data_dir) if data_dir else Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data"))
    paths = Paths(root.resolve())
    paths.ensure()
    return Config(values=values, paths=paths, source_path=path if path.exists() else None)


def load_dotenv_if_available() -> None:
    """python-dotenv があれば .env を読み込む（無くても環境変数だけで動く）。"""
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - 任意依存
        return
    load_dotenv(PROJECT_ROOT / ".env")
