"""Step 3: OCR一次抽出。

Google Cloud Vision / Azure Computer Vision (Read API) / Mock の3実装を
共通インタフェース（OCRProvider）で切り替えられるようにしている。
APIキー未設定時は Mock が選択され、パイプライン全体の疎通確認ができる。

結果は ./data/ocr_raw/{video_id}.jsonl に1行1フレームで保存し、
再実行時は未処理のフレームのみをAPIに投げる。
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


class OCRError(RuntimeError):
    """OCR API 呼び出しに失敗した場合に送出する。"""


@dataclass
class OCRResult:
    """1フレーム分のOCR結果（生データ）。"""

    video_id: str
    timestamp_sec: float
    frame_path: str
    provider: str
    raw_text: str = ""
    lines: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# プロバイダ実装
# --------------------------------------------------------------------------------------
class OCRProvider:
    """OCRエンジンの共通インタフェース。"""

    name = "base"

    def __init__(self, timeout_sec: int = 30, max_retries: int = 2, language_hints: list[str] | None = None):
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.language_hints = language_hints or ["ja"]

    def annotate(self, image_path: Path) -> dict[str, Any]:
        """画像1枚をOCRし、{"raw_text", "lines", "confidence"} を返す。"""
        raise NotImplementedError

    # 共通のリトライ処理（指数バックオフ）
    def _with_retry(self, func, *args, **kwargs):
        delay = 2.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # ネットワーク断・一時的な5xxを想定
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                logger.warning("OCR呼び出し失敗 (%d回目): %s / %.0f秒後に再試行", attempt + 1, exc, delay)
                time.sleep(delay)
                delay *= 2
        raise OCRError(str(last_exc))


class MockOCRProvider(OCRProvider):
    """APIキー未設定時のダミー実装。

    画像のハッシュから決定的にサンプルテロップを返すため、再実行しても結果が安定する。
    """

    name = "mock"

    SAMPLES = [
        "東京都 都市整備局 計画課長 山田 太郎",
        "株式会社アーチズ 営業部 部長 鈴木 花子",
        "横浜市 防災担当 課長補佐 佐藤 一郎",
        "厚生労働省 医政局長 田中 次郎",
        "ABC銀行 横浜支店 支店長 高橋 三郎",
        "国立感染症研究所 研究員 中村 美咲",
        "（株）サンプル商事 代表取締役 渡辺 健",
        "大阪府 教育庁 指導主事",
    ]

    def annotate(self, image_path: Path) -> dict[str, Any]:
        digest = hashlib.sha1(image_path.read_bytes()).hexdigest()
        index = int(digest[:8], 16) % len(self.SAMPLES)
        text = self.SAMPLES[index]
        confidence = 0.80 + (int(digest[8:10], 16) % 20) / 100.0
        return {
            "raw_text": text,
            "lines": [
                {
                    "text": text,
                    "confidence": round(confidence, 3),
                    "bounding_box": [[40, 360], [560, 360], [560, 400], [40, 400]],
                }
            ],
            "confidence": round(confidence, 3),
        }


class TesseractOCRProvider(OCRProvider):
    """ローカルの Tesseract OCR（APIキー不要）。

    クラウドAPIのキーが用意できない環境で、実画像を使って精度を確認するための選択肢。
    日本語データ（`tesseract-ocr-jpn`）のインストールが必要。
    """

    name = "tesseract"

    # 信頼度がこれ未満の語は背景ノイズとみなして捨てる
    MIN_WORD_CONFIDENCE = 10.0

    def __init__(self, lang: str = "jpn", psm: int = 6, binary: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.psm = psm
        self.binary = binary or os.environ.get("TESSERACT_CMD", "tesseract")

    def annotate(self, image_path: Path) -> dict[str, Any]:
        import subprocess
        import tempfile

        # txt と tsv を1回の実行で同時に出力する。
        # 日本語では tsv が語単位に分割され本来の空白が復元できないため、
        # raw_text は txt 側を使い、tsv は信頼度と bounding box の取得に使う。
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "out"
            cmd = [self.binary, str(image_path), str(base), "-l", self.lang, "--psm", str(self.psm), "txt", "tsv"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_sec, check=False)
            if result.returncode != 0:
                raise OCRError(f"tesseract の実行に失敗しました: {result.stderr.strip()[:300]}")
            text_output = base.with_suffix(".txt").read_text(encoding="utf-8", errors="replace")
            tsv_output = base.with_suffix(".tsv").read_text(encoding="utf-8", errors="replace")
        return self._parse(text_output, tsv_output)

    @classmethod
    def _parse(cls, text_output: str, tsv_output: str) -> dict[str, Any]:
        text_lines = [line.strip() for line in text_output.splitlines() if line.strip()]
        tsv_lines = cls._parse_tsv(tsv_output)

        # txt と tsv の行数が一致する場合は、空白を保った txt 側の文字列を採用する
        if len(text_lines) == len(tsv_lines):
            for line, text in zip(tsv_lines, text_lines):
                line["text"] = text
            raw_text = "\n".join(text_lines)
        else:
            raw_text = "\n".join(text_lines) if text_lines else "\n".join(line["text"] for line in tsv_lines)

        confidences = [line["confidence"] for line in tsv_lines]
        confidence = sum(confidences) / len(confidences) if confidences else (1.0 if raw_text else 0.0)
        return {"raw_text": raw_text.strip(), "lines": tsv_lines, "confidence": round(confidence, 3)}

    @classmethod
    def _parse_tsv(cls, tsv: str) -> list[dict[str, Any]]:
        """tsv出力を行単位（text, confidence, bounding_box）に集約する。"""
        grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in tsv.splitlines()[1:]:  # 1行目はヘッダ
            columns = row.split("\t")
            if len(columns) < 12:
                continue
            text = columns[11].strip()
            try:
                confidence = float(columns[10])
                left, top, width, height = (int(columns[i]) for i in range(6, 10))
            except ValueError:
                continue
            if not text or confidence < cls.MIN_WORD_CONFIDENCE:
                continue
            key = (columns[2], columns[3], columns[4])  # block / paragraph / line
            grouped.setdefault(key, []).append(
                {"text": text, "confidence": confidence, "left": left, "top": top, "width": width, "height": height}
            )

        lines: list[dict[str, Any]] = []
        for words in grouped.values():
            words.sort(key=lambda word: word["left"])
            line_confidences = [word["confidence"] for word in words]
            x1 = min(word["left"] for word in words)
            y1 = min(word["top"] for word in words)
            x2 = max(word["left"] + word["width"] for word in words)
            y2 = max(word["top"] + word["height"] for word in words)
            lines.append(
                {
                    "text": "".join(word["text"] for word in words),
                    "confidence": round(sum(line_confidences) / len(line_confidences) / 100.0, 3),
                    "bounding_box": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                }
            )
        return lines


class GoogleVisionOCRProvider(OCRProvider):
    """Google Cloud Vision API (DOCUMENT_TEXT_DETECTION)。"""

    name = "google"
    ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

    def __init__(self, api_key: str, **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key

    def annotate(self, image_path: Path) -> dict[str, Any]:
        import requests

        content = base64.b64encode(image_path.read_bytes()).decode("ascii")
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
                    "imageContext": {"languageHints": self.language_hints},
                }
            ]
        }

        def _call() -> dict[str, Any]:
            response = requests.post(
                self.ENDPOINT,
                params={"key": self.api_key},
                json=payload,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            return response.json()

        body = self._with_retry(_call)
        responses = body.get("responses") or [{}]
        first = responses[0]
        if "error" in first:
            raise OCRError(f"Vision API エラー: {first['error'].get('message')}")
        return self._parse(first)

    @staticmethod
    def _parse(response: dict[str, Any]) -> dict[str, Any]:
        full = response.get("fullTextAnnotation") or {}
        raw_text = (full.get("text") or "").strip()

        lines: list[dict[str, Any]] = []
        confidences: list[float] = []
        for page in full.get("pages", []):
            for block in page.get("blocks", []):
                block_conf = block.get("confidence")
                if isinstance(block_conf, (int, float)):
                    confidences.append(float(block_conf))
                for paragraph in block.get("paragraphs", []):
                    words = [
                        "".join(symbol.get("text", "") for symbol in word.get("symbols", []))
                        for word in paragraph.get("words", [])
                    ]
                    text = "".join(words).strip()
                    if not text:
                        continue
                    vertices = paragraph.get("boundingBox", {}).get("vertices", [])
                    lines.append(
                        {
                            "text": text,
                            "confidence": paragraph.get("confidence", block_conf or 0.0),
                            "bounding_box": [[v.get("x", 0), v.get("y", 0)] for v in vertices],
                        }
                    )
        confidence = sum(confidences) / len(confidences) if confidences else (1.0 if raw_text else 0.0)
        return {"raw_text": raw_text, "lines": lines, "confidence": round(float(confidence), 3)}


class AzureReadOCRProvider(OCRProvider):
    """Azure Computer Vision Read API (v3.2)。非同期APIのためポーリングする。"""

    name = "azure"

    def __init__(self, endpoint: str, api_key: str, poll_interval_sec: float = 1.0, max_polls: int = 30, **kwargs):
        super().__init__(**kwargs)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.poll_interval_sec = poll_interval_sec
        self.max_polls = max_polls

    def annotate(self, image_path: Path) -> dict[str, Any]:
        import requests

        analyze_url = f"{self.endpoint}/vision/v3.2/read/analyze"
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/octet-stream",
        }
        data = image_path.read_bytes()

        def _submit() -> str:
            response = requests.post(
                analyze_url,
                headers=headers,
                params={"language": self.language_hints[0]} if self.language_hints else None,
                data=data,
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            operation_location = response.headers.get("Operation-Location")
            if not operation_location:
                raise OCRError("Azure Read API のレスポンスに Operation-Location がありません")
            return operation_location

        operation_location = self._with_retry(_submit)

        for _ in range(self.max_polls):
            time.sleep(self.poll_interval_sec)
            result = requests.get(
                operation_location,
                headers={"Ocp-Apim-Subscription-Key": self.api_key},
                timeout=self.timeout_sec,
            )
            result.raise_for_status()
            body = result.json()
            status = body.get("status")
            if status == "succeeded":
                return self._parse(body)
            if status == "failed":
                raise OCRError(f"Azure Read API が失敗を返しました: {body}")
        raise OCRError("Azure Read API のポーリングがタイムアウトしました")

    @staticmethod
    def _parse(body: dict[str, Any]) -> dict[str, Any]:
        lines: list[dict[str, Any]] = []
        confidences: list[float] = []
        for page in body.get("analyzeResult", {}).get("readResults", []):
            for line in page.get("lines", []):
                word_confidences = [w.get("confidence", 0.0) for w in line.get("words", [])]
                line_conf = sum(word_confidences) / len(word_confidences) if word_confidences else 0.0
                confidences.extend(word_confidences)
                box = line.get("boundingBox", [])
                lines.append(
                    {
                        "text": line.get("text", ""),
                        "confidence": round(float(line_conf), 3),
                        "bounding_box": [box[i : i + 2] for i in range(0, len(box), 2)],
                    }
                )
        raw_text = "\n".join(line["text"] for line in lines).strip()
        confidence = sum(confidences) / len(confidences) if confidences else (1.0 if raw_text else 0.0)
        return {"raw_text": raw_text, "lines": lines, "confidence": round(float(confidence), 3)}


def create_provider(provider_name: str | None = None, ocr_config: dict[str, Any] | None = None) -> OCRProvider:
    """設定・環境変数からOCRプロバイダを生成する。キーが無ければ Mock にフォールバック。"""
    ocr_config = ocr_config or {}
    kwargs = {
        "timeout_sec": int(ocr_config.get("timeout_sec", 30)),
        "max_retries": int(ocr_config.get("max_retries", 2)),
        "language_hints": ocr_config.get("language_hints", ["ja"]),
    }
    name = (provider_name or ocr_config.get("provider") or os.environ.get("OCR_PROVIDER") or "").strip().lower()

    google_key = os.environ.get("GOOGLE_VISION_API_KEY", "").strip()
    azure_endpoint = os.environ.get("AZURE_VISION_ENDPOINT", "").strip()
    azure_key = os.environ.get("AZURE_VISION_KEY", "").strip()

    if not name:  # 自動判定
        if google_key:
            name = "google"
        elif azure_endpoint and azure_key:
            name = "azure"
        else:
            name = "mock"

    if name == "google":
        if not google_key:
            logger.warning("GOOGLE_VISION_API_KEY が未設定のため mock OCR を使用します")
            return MockOCRProvider(**kwargs)
        return GoogleVisionOCRProvider(google_key, **kwargs)

    if name == "azure":
        if not (azure_endpoint and azure_key):
            logger.warning("AZURE_VISION_ENDPOINT / AZURE_VISION_KEY が未設定のため mock OCR を使用します")
            return MockOCRProvider(**kwargs)
        return AzureReadOCRProvider(azure_endpoint, azure_key, **kwargs)

    if name == "tesseract":
        return TesseractOCRProvider(
            lang=ocr_config.get("tesseract_lang", os.environ.get("TESSERACT_LANG", "jpn")),
            psm=int(ocr_config.get("tesseract_psm", 6)),
            **kwargs,
        )

    if name != "mock":
        logger.warning("未知のOCRプロバイダ '%s' が指定されました。mock を使用します", name)
    return MockOCRProvider(**kwargs)


# --------------------------------------------------------------------------------------
# Step 3 本体
# --------------------------------------------------------------------------------------
def ocr_jsonl_path(ocr_dir: Path, video_id: str) -> Path:
    return ocr_dir / f"{video_id}.jsonl"


def load_ocr_results(ocr_dir: Path, video_id: str) -> list[OCRResult]:
    """保存済みの OCR 結果を読み込む（Step 4 以降を単独実行する場合に使用）。"""
    path = ocr_jsonl_path(ocr_dir, video_id)
    if not path.exists():
        return []
    results: list[OCRResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            results.append(OCRResult(**{k: payload.get(k) for k in OCRResult.__dataclass_fields__ if k in payload}))
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("OCR結果の %d 行目を読み飛ばしました: %s", line_number, exc)
    return results


def run_ocr(
    frames: Iterable[Any],
    video_id: str,
    ocr_dir: Path,
    provider: OCRProvider | None = None,
    ocr_config: dict[str, Any] | None = None,
    force: bool = False,
) -> list[OCRResult]:
    """フレーム画像をOCRして ./data/ocr_raw/{video_id}.jsonl に追記する。

    1フレームの失敗ではパイプラインを止めず、error フィールドに記録して継続する。
    """
    ocr_dir.mkdir(parents=True, exist_ok=True)
    provider = provider or create_provider(ocr_config=ocr_config)
    path = ocr_jsonl_path(ocr_dir, video_id)

    existing: dict[str, OCRResult] = {}
    if path.exists() and not force:
        existing = {result.frame_path: result for result in load_ocr_results(ocr_dir, video_id) if not result.error}
        logger.info("OCR済みフレームを再利用します: %d 件", len(existing))
    elif path.exists() and force:
        path.unlink()

    results: list[OCRResult] = []
    processed = 0
    failed = 0
    with path.open("a", encoding="utf-8") as handle:
        for frame in frames:
            frame_path = str(getattr(frame, "path", frame))
            timestamp = float(getattr(frame, "timestamp_sec", 0.0))
            cached = existing.get(frame_path)
            if cached is not None:
                results.append(cached)
                continue

            result = OCRResult(
                video_id=video_id,
                timestamp_sec=timestamp,
                frame_path=frame_path,
                provider=provider.name,
            )
            try:
                annotation = provider.annotate(Path(frame_path))
                result.raw_text = annotation.get("raw_text", "")
                result.lines = annotation.get("lines", [])
                result.confidence = float(annotation.get("confidence", 0.0))
                processed += 1
            except Exception as exc:  # 1フレームの失敗で全体を止めない
                result.error = str(exc)[:500]
                failed += 1
                logger.warning("OCRに失敗しました (t=%.2fs): %s", timestamp, exc)

            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            results.append(result)

    logger.info(
        "OCR完了 (provider=%s): 新規 %d 件 / 再利用 %d 件 / 失敗 %d 件",
        provider.name,
        processed,
        len(results) - processed - failed,
        failed,
    )
    return results
