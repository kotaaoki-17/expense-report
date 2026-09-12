"""YouTube動画テロップ抽出 PoC パイプライン。

各Stepは疎結合なモジュールとして実装されており、単体でも import して利用できます。

    download.py        Step 1: 動画ダウンロード
    extract_frames.py  Step 2: フレーム抽出（シーンチェンジ + クロップ + pHash間引き）
    ocr.py             Step 3: OCR一次抽出
    structurize.py     Step 4: 構造化抽出（ルールベース + LLMフォールバック）
    dedupe.py          Step 5: 重複排除（名寄せ）
    outputs.py         Step 6: JSON / CSV 出力
"""

__all__ = [
    "config",
    "download",
    "extract_frames",
    "ocr",
    "structurize",
    "dedupe",
    "outputs",
]
