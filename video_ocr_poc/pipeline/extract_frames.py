"""Step 2: フレーム抽出（間引きロジック）。

1. ffmpeg のシーンチェンジ検出で候補フレームを抽出（同時にテロップ領域をクロップ）
2. テロップらしさ（エッジ密度）が低いフレームを除外
3. 直前に採用したフレームと perceptual hash を比較し、変化がなければスキップ
4. 残ったフレームのみ ./data/frames/{video_id}/{timestamp_sec}.png に保存

ffmpeg 呼び出しは1パスのみで、`showinfo` の出力からフレームのタイムスタンプを復元する。
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MANIFEST_NAME = "frames.json"
_PTS_TIME_RE = re.compile(r"pts_time:(\d+(?:\.\d+)?)")


class FrameExtractionError(RuntimeError):
    """フレーム抽出に失敗した場合に送出する。"""


@dataclass
class Frame:
    """採用されたテロップ領域フレーム1枚。"""

    timestamp_sec: float
    path: str
    phash: str = ""
    edge_density: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if not binary:
        raise FrameExtractionError(
            "ffmpeg が見つかりません。`brew install ffmpeg` / `apt install ffmpeg` などで導入してください。"
        )
    return binary


def build_crop_filter(crop: dict[str, float]) -> str:
    """設定（比率）から ffmpeg の crop フィルタ文字列を作る。"""
    x = float(crop.get("x_ratio", 0.0))
    y = float(crop.get("y_ratio", 0.70))
    w = float(crop.get("w_ratio", 1.0))
    h = float(crop.get("h_ratio", 0.30))
    return f"crop=trunc(iw*{w}):trunc(ih*{h}):trunc(iw*{x}):trunc(ih*{y})"


def build_select_filter(scene_threshold: float, max_gap_sec: float) -> str:
    """シーンチェンジ、または一定時間経過で候補フレームを選ぶ select 式。

    シーンチェンジを伴わないテロップ切り替えを取りこぼさないよう、
    `max_gap_sec` 秒ごとのサンプリングを併用する（0以下で無効）。
    """
    terms = [f"gt(scene,{scene_threshold})", "isnan(prev_selected_t)", "eq(n,0)"]
    if max_gap_sec and max_gap_sec > 0:
        terms.append(f"gte(t-prev_selected_t,{max_gap_sec})")
    return "select='" + "+".join(terms) + "'"


def _run_ffmpeg_candidates(video_path: Path, out_dir: Path, vf: str) -> list[float]:
    """候補フレームを out_dir に書き出し、そのタイムスタンプ（秒）を返す。"""
    cmd = [
        _ffmpeg_binary(),
        "-hide_banner",
        "-nostdin",
        "-i", str(video_path),
        "-vf", vf,
        "-vsync", "0",
        "-y",
        str(out_dir / "cand_%06d.png"),
    ]
    logger.debug("run: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise FrameExtractionError(f"ffmpeg の実行に失敗しました: {result.stderr.strip()[-800:]}")
    timestamps = [float(m) for m in _PTS_TIME_RE.findall(result.stderr)]
    logger.info("候補フレーム: %d 枚", len(timestamps))
    return timestamps


def _edge_density(image_path: Path) -> float:
    """テロップらしさの簡易指標（エッジ画素の割合）。cv2 が無ければ PIL で代替する。"""
    try:
        import cv2  # type: ignore
        import numpy as np

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return 0.0
        edges = cv2.Canny(image, 100, 200)
        return float(np.count_nonzero(edges)) / float(edges.size or 1)
    except ImportError:
        from PIL import Image, ImageFilter, ImageStat

        with Image.open(image_path) as img:
            gray = img.convert("L").filter(ImageFilter.FIND_EDGES)
            # 標準偏差を 0-1 に正規化した近似値
            return min(ImageStat.Stat(gray).stddev[0] / 128.0, 1.0)


def _phash(image_path: Path):
    import imagehash
    from PIL import Image

    with Image.open(image_path) as img:
        return imagehash.phash(img)


def _manifest_path(frames_dir: Path, video_id: str) -> Path:
    return frames_dir / video_id / MANIFEST_NAME


def load_frames(frames_dir: Path, video_id: str) -> list[Frame]:
    """保存済みの frames.json を読み込む（Step 3 以降を単独実行する場合に使用）。"""
    path = _manifest_path(frames_dir, video_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("frames.json を読み込めませんでした: %s", exc)
        return []
    return [Frame(**item) for item in data.get("frames", [])]


def extract_frames(
    video_path: Path,
    video_id: str,
    frames_dir: Path,
    crop: dict[str, float] | None = None,
    scene_threshold: float = 0.3,
    max_gap_sec: float = 2.0,
    phash_distance_threshold: int = 6,
    min_edge_density: float = 0.004,
    max_frames: int = 500,
    force: bool = False,
) -> list[Frame]:
    """テロップ領域フレームを抽出して Frame のリストを返す。"""
    out_dir = frames_dir / video_id
    manifest = _manifest_path(frames_dir, video_id)
    if manifest.exists() and not force:
        frames = load_frames(frames_dir, video_id)
        if frames:
            logger.info("抽出済みフレームを再利用します: %d 枚 (%s)", len(frames), manifest)
            return frames

    if not video_path.exists():
        raise FrameExtractionError(f"動画ファイルが見つかりません: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    candidates_dir = out_dir / "_candidates"
    if candidates_dir.exists():
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)

    vf = ",".join(
        [
            build_select_filter(scene_threshold, max_gap_sec),
            build_crop_filter(crop or {}),
            "showinfo",
        ]
    )
    timestamps = _run_ffmpeg_candidates(video_path, candidates_dir, vf)
    candidate_files = sorted(candidates_dir.glob("cand_*.png"))
    if len(candidate_files) != len(timestamps):
        logger.warning(
            "候補画像(%d)とタイムスタンプ(%d)の数が一致しません。少ない方に合わせます。",
            len(candidate_files),
            len(timestamps),
        )

    # 既存の採用フレームを消してから作り直す（再実行時に混ざらないように）
    for stale in out_dir.glob("*.png"):
        stale.unlink()

    kept: list[Frame] = []
    previous_hash = None
    skipped_similar = 0
    skipped_blank = 0

    for candidate, timestamp in zip(candidate_files, timestamps):
        if len(kept) >= max_frames:
            logger.warning("max_frames (%d) に達したため打ち切ります", max_frames)
            break
        try:
            density = _edge_density(candidate)
            if density < min_edge_density:
                skipped_blank += 1
                continue
            current_hash = _phash(candidate)
            if previous_hash is not None and (current_hash - previous_hash) <= phash_distance_threshold:
                skipped_similar += 1
                continue
            target = out_dir / f"{timestamp:09.2f}.png"
            shutil.copyfile(candidate, target)
            kept.append(
                Frame(
                    timestamp_sec=round(timestamp, 2),
                    path=str(target),
                    phash=str(current_hash),
                    edge_density=round(density, 5),
                )
            )
            previous_hash = current_hash
        except Exception as exc:  # 1枚の失敗で全体を止めない
            logger.warning("フレーム処理に失敗しました (t=%.2fs): %s", timestamp, exc)

    shutil.rmtree(candidates_dir, ignore_errors=True)

    manifest.write_text(
        json.dumps(
            {
                "video_id": video_id,
                "source_video": str(video_path),
                "params": {
                    "scene_threshold": scene_threshold,
                    "max_gap_sec": max_gap_sec,
                    "crop": crop,
                    "phash_distance_threshold": phash_distance_threshold,
                    "min_edge_density": min_edge_density,
                },
                "stats": {
                    "candidates": len(candidate_files),
                    "kept": len(kept),
                    "skipped_similar": skipped_similar,
                    "skipped_blank": skipped_blank,
                },
                "frames": [frame.to_dict() for frame in kept],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "フレーム抽出完了: 採用 %d 枚 / 候補 %d 枚 (類似スキップ %d, 低情報スキップ %d)",
        len(kept),
        len(candidate_files),
        skipped_similar,
        skipped_blank,
    )
    return kept
