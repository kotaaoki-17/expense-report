"""Step 4 のLLMフォールバック（Claude API）。

ルールベースで確信が持てなかったフレームだけを対象にする想定。
APIキーが未設定の場合は無効化され、パイプラインはルールベース結果のまま継続する。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
DEFAULT_MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = """以下はニュース番組のテロップOCR結果です。会社名・部署名（役職含む）・氏名を
JSON形式 {{"company": "", "department": "", "name": ""}} で抽出してください。
読み取れない項目は空文字にしてください。テロップ以外の説明・前置きは不要です。

OCR結果: "{raw_text}"
"""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class LLMClient:
    """Claude API による構造化抽出クライアント。"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        timeout_sec: int = 30,
        max_retries: int = 2,
        max_calls: int = 100,
    ):
        self.api_key = api_key if api_key is not None else os.environ.get("ANTHROPIC_API_KEY", "").strip()
        self.model = model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.max_calls = max_calls
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def extract(self, raw_text: str) -> dict[str, str] | None:
        """OCR生テキストから {company, department, name} を抽出する。

        失敗（API エラー / JSON パース不能）時は None を返し、呼び出し側は処理を継続する。
        """
        if not self.enabled:
            return None
        if self.calls >= self.max_calls:
            logger.warning("LLM呼び出しが上限 (%d 回) に達したためスキップします", self.max_calls)
            return None
        if not raw_text.strip():
            return None

        import requests

        payload = {
            "model": self.model,
            "max_tokens": 256,
            "messages": [{"role": "user", "content": PROMPT_TEMPLATE.format(raw_text=raw_text)}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                self.calls += 1
                response = requests.post(API_URL, headers=headers, json=payload, timeout=self.timeout_sec)
                response.raise_for_status()
                body = response.json()
                text = "".join(
                    block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
                )
                return self._parse_json(text)
            except Exception as exc:
                if attempt >= self.max_retries:
                    logger.warning("LLM呼び出しに失敗しました: %s", exc)
                    return None
                logger.warning("LLM呼び出し失敗 (%d回目): %s / %.0f秒後に再試行", attempt + 1, exc, delay)
                time.sleep(delay)
                delay *= 2
        return None

    @staticmethod
    def _parse_json(text: str) -> dict[str, str] | None:
        """LLM応答からJSONを取り出す。前置きが混ざっていても最初のオブジェクトを拾う。"""
        candidates: list[str] = [text.strip()]
        match = _JSON_RE.search(text)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return {key: str(data.get(key, "") or "").strip() for key in ("company", "department", "name")}
        logger.warning("LLM応答をJSONとして解析できませんでした: %s", text[:200].replace("\n", " "))
        return None


def create_llm_client(structurize_config: dict[str, Any] | None = None) -> LLMClient:
    config = structurize_config or {}
    client = LLMClient(max_calls=int(config.get("llm_max_calls", 100)))
    if not client.enabled:
        logger.info("ANTHROPIC_API_KEY が未設定のため、LLMフォールバックは無効です")
    return client
