"""Step 1: 動画ダウンロード。

`yt-dlp` を使って 480p 程度の動画とメタデータを取得する。
    ./data/raw/{video_id}.mp4
    ./data/raw/{video_id}.json
すでにダウンロード済みの場合は再取得しない（`force=True` で強制再取得）。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)


class DownloadError(RuntimeError):
    """動画のダウンロードに失敗した場合に送出する。"""


@dataclass
class VideoAsset:
    """ダウンロード済み動画とそのメタデータ。"""

    video_id: str
    path: Path
    title: str = ""
    upload_date: str = ""
    duration_sec: float = 0.0
    webpage_url: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["path"] = str(self.path)
        return data


_ID_PATTERNS = (
    re.compile(r"(?:v=|/shorts/|/embed/|/live/|youtu\.be/)([0-9A-Za-z_-]{11})"),
)


def video_id_from_url(url: str) -> str | None:
    """URL から YouTube の video_id を推定する（ネットワークアクセスなし）。"""
    parsed = urlparse(url)
    values = parse_qs(parsed.query).get("v")
    if values and re.fullmatch(r"[0-9A-Za-z_-]{11}", values[0]):
        return values[0]
    for pattern in _ID_PATTERNS:
        match = pattern.search(url)
        if match:
            return match.group(1)
    return None


def _yt_dlp_command() -> list[str]:
    binary = shutil.which("yt-dlp")
    if binary:
        return [binary]
    # pip でインストールされた yt-dlp をモジュールとして呼ぶフォールバック
    import sys

    try:
        import yt_dlp  # noqa: F401
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise DownloadError(
            "yt-dlp が見つかりません。`pip install -r requirements.txt` を実行してください。"
        ) from exc
    return [sys.executable, "-m", "yt_dlp"]


def _run(cmd: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    logger.debug("run: %s", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def fetch_metadata(url: str) -> dict:
    """yt-dlp でメタデータ（JSON）のみ取得する。"""
    result = _run([*_yt_dlp_command(), "--dump-json", "--skip-download", "--no-warnings", url], timeout=180)
    if result.returncode != 0:
        raise DownloadError(f"メタデータ取得に失敗しました: {result.stderr.strip()[:500]}")
    try:
        return json.loads(result.stdout.splitlines()[0])
    except (json.JSONDecodeError, IndexError) as exc:
        raise DownloadError(f"メタデータのJSONを解析できませんでした: {exc}") from exc


def _asset_from_metadata(metadata: dict, path: Path) -> VideoAsset:
    return VideoAsset(
        video_id=metadata.get("id", path.stem),
        path=path,
        title=metadata.get("title", ""),
        upload_date=metadata.get("upload_date", ""),
        duration_sec=float(metadata.get("duration") or 0.0),
        webpage_url=metadata.get("webpage_url", ""),
    )


def _existing_asset(video_id: str, raw_dir: Path) -> VideoAsset | None:
    """ダウンロード済みの動画があればそれを返す（再実行時のスキップ用）。"""
    meta_path = raw_dir / f"{video_id}.json"
    candidates = sorted(raw_dir.glob(f"{video_id}.*"))
    video_path = next((p for p in candidates if p.suffix.lower() in {".mp4", ".mkv", ".webm"}), None)
    if video_path is None or not video_path.exists():
        return None
    metadata = {}
    if meta_path.exists():
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("既存メタデータを読めませんでした: %s", meta_path)
    return _asset_from_metadata(metadata or {"id": video_id}, video_path)


def download_video(
    url: str,
    raw_dir: Path,
    max_height: int = 480,
    force: bool = False,
    format_fallback: str = "best",
) -> VideoAsset:
    """動画とメタデータを ./data/raw/ に保存して VideoAsset を返す。

    Raises:
        DownloadError: ダウンロードに失敗した場合。
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    guessed_id = video_id_from_url(url)
    if guessed_id and not force:
        existing = _existing_asset(guessed_id, raw_dir)
        if existing:
            logger.info("ダウンロード済みの動画を再利用します: %s", existing.path)
            return existing

    metadata = fetch_metadata(url)
    video_id = metadata.get("id") or guessed_id
    if not video_id:
        raise DownloadError(f"video_id を特定できませんでした: {url}")

    meta_path = raw_dir / f"{video_id}.json"
    meta_path.write_text(
        json.dumps(
            {
                "id": video_id,
                "title": metadata.get("title"),
                "upload_date": metadata.get("upload_date"),
                "duration": metadata.get("duration"),
                "channel": metadata.get("channel"),
                "channel_id": metadata.get("channel_id"),
                "webpage_url": metadata.get("webpage_url", url),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not force:
        existing = _existing_asset(video_id, raw_dir)
        if existing:
            logger.info("ダウンロード済みの動画を再利用します: %s", existing.path)
            return existing

    out_template = str(raw_dir / f"{video_id}.%(ext)s")
    fmt = (
        f"bestvideo[height<={max_height}]+bestaudio/"
        f"best[height<={max_height}]/{format_fallback}"
    )
    cmd = [
        *_yt_dlp_command(),
        "-f", fmt,
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        "--no-warnings",
        url,
    ]
    logger.info("動画をダウンロードします (video_id=%s, <=%dp)", video_id, max_height)
    result = _run(cmd)
    if result.returncode != 0:
        raise DownloadError(f"動画ダウンロードに失敗しました: {result.stderr.strip()[:500]}")

    asset = _existing_asset(video_id, raw_dir)
    if asset is None:
        raise DownloadError(f"ダウンロードしたはずの動画ファイルが見つかりません: {raw_dir}/{video_id}.*")
    logger.info("ダウンロード完了: %s", asset.path)
    return _asset_from_metadata(metadata, asset.path)


def delete_video_file(asset: VideoAsset) -> None:
    """検証後に動画ファイルを削除する（YouTube利用規約への配慮）。メタデータは残す。"""
    try:
        asset.path.unlink(missing_ok=True)
        logger.info("動画ファイルを削除しました: %s", asset.path)
    except OSError as exc:
        logger.warning("動画ファイルの削除に失敗しました: %s", exc)
