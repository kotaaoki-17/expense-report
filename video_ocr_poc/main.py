#!/usr/bin/env python3
"""YouTube動画テロップ抽出 PoC のエントリポイント。

    python main.py --url "https://www.youtube.com/watch?v=XXXXXXXX"

Step1〜6を順に実行し、./data/output/{video_id}.json と .csv を生成する。
各Stepの中間成果物はファイルに残るため、同じコマンドを再実行すると未処理分だけが処理される。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pipeline import dedupe as dedupe_step
from pipeline import outputs as output_step
from pipeline.config import load_config
from pipeline.download import DownloadError, VideoAsset, delete_video_file, download_video, video_id_from_url
from pipeline.extract_frames import FrameExtractionError, extract_frames, load_frames
from pipeline.llm import create_llm_client
from pipeline.logging_utils import setup_logging
from pipeline.ocr import create_provider, load_ocr_results, run_ocr
from pipeline.structurize import load_structured, structurize

logger = logging.getLogger("main")

ALL_STEPS = ("download", "frames", "ocr", "structurize", "dedupe", "output")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YouTube動画のテロップから会社名・部署名・氏名を抽出する PoC パイプライン",
    )
    parser.add_argument("--url", help="対象のYouTube動画URL")
    parser.add_argument(
        "--video-id",
        help="ダウンロード済み動画を指定して再実行する場合のvideo_id（--url の代わりに指定可）",
    )
    parser.add_argument("--config", help="設定ファイルのパス（既定: ./config.json）")
    parser.add_argument("--data-dir", help="中間成果物・出力の保存先（既定: ./data）")
    parser.add_argument(
        "--steps",
        default=",".join(ALL_STEPS),
        help=f"実行するStep（カンマ区切り）。選択肢: {','.join(ALL_STEPS)}",
    )
    parser.add_argument("--force", action="store_true", help="キャッシュを無視して全Stepを再実行する")
    parser.add_argument("--ocr-provider", choices=["mock", "google", "azure"], help="OCRプロバイダを明示指定する")
    parser.add_argument("--no-llm", action="store_true", help="LLMフォールバックを無効にする")
    parser.add_argument(
        "--delete-video",
        action="store_true",
        help="処理後に動画ファイルを削除する（検証目的の一時利用を前提とする運用向け）",
    )
    parser.add_argument("--log-level", default="INFO", help="ログレベル（DEBUG/INFO/WARNING/ERROR）")
    return parser.parse_args(argv)


def resolve_video_id(args: argparse.Namespace) -> str | None:
    if args.video_id:
        return args.video_id
    if args.url:
        return video_id_from_url(args.url)
    return None


def find_existing_video(raw_dir: Path, video_id: str) -> Path | None:
    for suffix in (".mp4", ".mkv", ".webm"):
        candidate = raw_dir / f"{video_id}{suffix}"
        if candidate.exists():
            return candidate
    return None


def run(args: argparse.Namespace) -> int:
    if not args.url and not args.video_id:
        logger.error("--url または --video-id のいずれかを指定してください")
        return 2

    config = load_config(args.config, args.data_dir)
    paths = config.paths
    steps = {step.strip() for step in args.steps.split(",") if step.strip()}
    unknown = steps - set(ALL_STEPS)
    if unknown:
        logger.error("未知のStepが指定されました: %s", ", ".join(sorted(unknown)))
        return 2

    video_id = resolve_video_id(args)
    if video_id:
        setup_logging(args.log_level, paths.logs / f"{video_id}.log")
    logger.info("パイプライン開始 (steps=%s)", ",".join(step for step in ALL_STEPS if step in steps))

    # ---------------- Step 1: 動画ダウンロード ----------------
    asset: VideoAsset | None = None
    if "download" in steps and args.url:
        try:
            asset = download_video(
                args.url,
                paths.raw,
                max_height=int(config.get("download", "max_height", 480)),
                force=args.force,
                format_fallback=config.get("download", "format_fallback", "best"),
            )
            video_id = asset.video_id
            setup_logging(args.log_level, paths.logs / f"{video_id}.log")
        except DownloadError as exc:
            logger.error("Step1(ダウンロード)に失敗しました: %s", exc)
            return 1
    if video_id is None:
        logger.error("video_id を特定できませんでした。--video-id を指定してください")
        return 2

    video_path = asset.path if asset else find_existing_video(paths.raw, video_id)

    # ---------------- Step 2: フレーム抽出 ----------------
    frames = []
    if "frames" in steps:
        if video_path is None:
            logger.error("動画ファイルが見つかりません: %s/%s.*", paths.raw, video_id)
            return 1
        frames_config = config.section("frames")
        try:
            frames = extract_frames(
                video_path,
                video_id,
                paths.frames,
                crop=frames_config.get("crop"),
                scene_threshold=float(frames_config.get("scene_threshold", 0.3)),
                max_gap_sec=float(frames_config.get("max_gap_sec", 2.0)),
                phash_distance_threshold=int(frames_config.get("phash_distance_threshold", 6)),
                min_edge_density=float(frames_config.get("min_edge_density", 0.004)),
                max_frames=int(frames_config.get("max_frames", 500)),
                force=args.force,
            )
        except FrameExtractionError as exc:
            logger.error("Step2(フレーム抽出)に失敗しました: %s", exc)
            return 1
    else:
        frames = load_frames(paths.frames, video_id)

    # ---------------- Step 3: OCR ----------------
    if "ocr" in steps:
        if not frames:
            logger.warning("対象フレームがありません。テロップ領域(crop)やしきい値の設定を見直してください")
        provider = create_provider(args.ocr_provider, config.section("ocr"))
        ocr_results = run_ocr(
            frames,
            video_id,
            paths.ocr_raw,
            provider=provider,
            ocr_config=config.section("ocr"),
            force=args.force,
        )
    else:
        ocr_results = load_ocr_results(paths.ocr_raw, video_id)

    # ---------------- Step 4: 構造化 ----------------
    structurize_config = dict(config.section("structurize"))
    if args.no_llm:
        structurize_config["use_llm_fallback"] = False
    if "structurize" in steps:
        llm_client = create_llm_client(structurize_config) if structurize_config.get("use_llm_fallback", True) else None
        structured = structurize(
            ocr_results,
            video_id,
            paths.structured,
            llm_client=llm_client,
            structurize_config=structurize_config,
            force=args.force,
        )
    else:
        structured = load_structured(paths.structured, video_id)

    # ---------------- Step 5: 名寄せ ----------------
    if "dedupe" in steps:
        records = dedupe_step.dedupe(
            structured,
            name_similarity_threshold=float(config.get("dedupe", "name_similarity_threshold", 0.8)),
        )
    else:
        records = structured

    # ---------------- Step 6: 出力 ----------------
    if "output" in steps:
        written = output_step.write_outputs(
            records,
            video_id,
            paths.output,
            formats=config.get("output", "formats", ["json", "csv"]),
        )
        for fmt, path in written.items():
            logger.info("%s を出力しました: %s", fmt.upper(), path)

    if args.delete_video and video_path is not None:
        delete_video_file(asset or VideoAsset(video_id=video_id, path=video_path))

    logger.info("パイプライン完了: %d 件のレコードを抽出しました", len(records))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    setup_logging(args.log_level)
    try:
        return run(args)
    except KeyboardInterrupt:
        logger.warning("中断されました")
        return 130
    except Exception as exc:  # 想定外の例外もログに残して終了コードで通知する
        logger.exception("予期しないエラーで終了しました: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
