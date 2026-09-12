#!/usr/bin/env python3
"""テロップ領域（crop座標）の調整用プレビュー。

指定した秒数のフレームを「全体」と「クロップ後」の2枚で書き出し、
config.json の frames.crop を目視で追い込めるようにする。

    python tools/preview_crop.py --video data/raw/XXXX.mp4 --seconds 12 30 61
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import load_config  # noqa: E402
from pipeline.extract_frames import build_crop_filter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="テロップ領域のプレビュー画像を書き出す")
    parser.add_argument("--video", required=True, help="動画ファイルのパス")
    parser.add_argument("--seconds", nargs="+", type=float, required=True, help="プレビューする秒数（複数可）")
    parser.add_argument("--out-dir", default="data/preview", help="出力先ディレクトリ")
    parser.add_argument("--config", help="設定ファイルのパス")
    args = parser.parse_args()

    config = load_config(args.config)
    crop_filter = build_crop_filter(config.get("frames", "crop", {}))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for seconds in args.seconds:
        for label, vf in (("full", None), ("crop", crop_filter)):
            target = out_dir / f"{label}_{seconds:.2f}.png"
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(seconds), "-i", args.video]
            if vf:
                cmd += ["-vf", vf]
            cmd += ["-frames:v", "1", str(target)]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                print(f"失敗 ({seconds}s, {label}): {result.stderr.strip()[:200]}", file=sys.stderr)
            else:
                print(f"書き出しました: {target}")
    print(f"\ncrop filter: {crop_filter}\nconfig.json の frames.crop を調整して再実行してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
