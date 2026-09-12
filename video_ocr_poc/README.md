# YouTube動画テロップ抽出 PoC

ニュース番組などの動画に表示される「所属・役職・氏名」のテロップから、
**会社名 / 部署名 / 氏名** を構造化データ（JSON・CSV）として取り出す PoC パイプラインです。

対象は **YouTube動画1本**。チャンネル巡回・大量処理・定期実行はスコープ外ですが、
各Stepは疎結合なモジュールに分けてあり、後から差し替え・並列化しやすい構成にしています。

```
動画URL → [1]ダウンロード → [2]フレーム抽出 → [3]OCR → [4]構造化 → [5]名寄せ → [6]出力
```

---

## 1. セットアップ

### 1-1. 外部コマンド

| コマンド | 用途 | インストール例 |
| --- | --- | --- |
| `ffmpeg` | シーンチェンジ検出・フレーム抽出・クロップ | `brew install ffmpeg` / `apt install ffmpeg` |
| `yt-dlp` | 動画ダウンロード | `pip install yt-dlp`（下記の requirements に同梱） |

### 1-2. Python

Python 3.11 以上。

```bash
cd video_ocr_poc
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1-3. APIキー（.env）

```bash
cp .env.example .env
# .env を編集してキーを設定する
```

| 環境変数 | 用途 |
| --- | --- |
| `OCR_PROVIDER` | `mock` / `google` / `azure`。未設定ならキーの有無から自動判定し、無ければ `mock` |
| `GOOGLE_VISION_API_KEY` | Google Cloud Vision API を使う場合 |
| `AZURE_VISION_ENDPOINT` / `AZURE_VISION_KEY` | Azure Computer Vision (Read API) を使う場合 |
| `ANTHROPIC_API_KEY` | Step 4 のLLMフォールバック（Claude API）。未設定ならルールベース結果のみで継続 |
| `ANTHROPIC_MODEL` | 省略時は `claude-sonnet-5` |

`.env` は `.gitignore` 済みです。**APIキーをコミットしないでください。**

> **キーが1つも無い状態でも動きます。** OCRは `mock`（決定的なダミーテロップを返す）に切り替わり、
> Step 1〜6 の疎通確認ができます。

---

## 2. 実行

```bash
python main.py --url "https://www.youtube.com/watch?v=XXXXXXXX"
```

成功すると `./data/output/{video_id}.json` と `./data/output/{video_id}.csv` が生成されます。

### 主なオプション

| オプション | 説明 |
| --- | --- |
| `--video-id XXXX` | ダウンロード済み動画を指定して再実行（`--url` の代わり。ネットワーク不要） |
| `--steps frames,ocr` | 実行するStepを限定（`download,frames,ocr,structurize,dedupe,output`） |
| `--force` | キャッシュを無視して全Stepを再実行 |
| `--ocr-provider google` | OCRプロバイダを明示指定 |
| `--no-llm` | LLMフォールバックを無効化（ルールベースのみ） |
| `--delete-video` | 処理後に動画ファイルを削除（検証目的の一時利用向け） |
| `--config path.json` / `--data-dir path` | 設定ファイル・データ保存先の差し替え |
| `--log-level DEBUG` | ログレベル |

### 途中から再実行できます

各Stepの中間成果物はファイルに残り、再実行時は未処理分だけを処理します
（OCRのAPIコールを無駄に繰り返しません）。

```
data/
├── raw/        {video_id}.mp4, {video_id}.json     … Step 1
├── frames/     {video_id}/{timestamp}.png, frames.json … Step 2
├── ocr_raw/    {video_id}.jsonl                    … Step 3（1行1フレーム）
├── structured/ {video_id}.jsonl                    … Step 4
├── output/     {video_id}.json / .csv              … Step 6
└── logs/       {video_id}.log
```

---

## 3. パイプラインの中身

### Step 1: 動画ダウンロード（`pipeline/download.py`）
`yt-dlp` で 480p 相当の動画とメタデータ（タイトル・投稿日・video_id）を取得します。
同じ動画が既にある場合はダウンロードをスキップします。

### Step 2: フレーム抽出（`pipeline/extract_frames.py`）— コストを決める要
単純な等間隔切り出しではなく、次の3段構えで OCR に投げる枚数を絞ります。

1. `ffmpeg` のシーンチェンジ検出（`select='gt(scene,0.3)'`）で候補フレームを抽出。
   シーンチェンジを伴わないテロップ切り替えを取りこぼさないよう、`max_gap_sec` 秒ごとのサンプリングも併用。
2. テロップ領域だけをクロップ（座標は `config.json` の `frames.crop` で比率指定）。
3. クロップ画像の perceptual hash（`imagehash.phash`）を直前の採用フレームと比較し、
   ハミング距離が `phash_distance_threshold` 以下（＝テロップが変わっていない）ならスキップ。
   さらにエッジ密度が低い（＝テロップが出ていない）フレームも除外。

ffmpeg の呼び出しは1パスのみで、`showinfo` の出力からタイムスタンプを復元しています。

**テロップ領域の調整**はプレビューツールが便利です:

```bash
python tools/preview_crop.py --video data/raw/XXXX.mp4 --seconds 12 30 61
# data/preview/ に full_*.png と crop_*.png が出るので、config.json の frames.crop を調整
```

### Step 3: OCR一次抽出（`pipeline/ocr.py`）
Google Cloud Vision / Azure Read API / Mock を同じインタフェースで切り替えます。
生テキスト・信頼度・bounding box を `./data/ocr_raw/{video_id}.jsonl` にそのまま保存します。
1フレームの失敗は `error` フィールドに記録して次のフレームへ進みます（全体は止まりません）。

### Step 4: 構造化抽出（`pipeline/structurize.py`）— ハイブリッド方式
1. **ルールベース一次判定**: `pipeline/dictionaries.py` の辞書と正規表現で
   会社名（`〜株式会社` `〜市` `〜省` `〜銀行` …）、部署・役職（`〜部` `〜課` `課長補佐` …）、
   氏名（空白区切りの姓名ペア、役職直後の漢字列、頻出姓 など）を抽出。
   3項目が揃えば「確定」（`min_rule_confidence` 以上）として扱います。
2. **LLMフォールバック**: 欠落・候補が複数ある場合のみ Claude API に生テキストを渡してJSONで補完。
   応答がJSONとして解析できない場合はログを出して継続します。

どの項目がルール確定でどの項目がLLM補完かは `field_sources` に残るため、精度検証に使えます。

```json
"field_sources": { "company": "rule", "department": "rule", "name": "llm" }
```

### Step 5: 重複排除（`pipeline/dedupe.py`）
氏名（完全一致 or 編集距離ベースの類似度が `name_similarity_threshold` 以上）でグルーピングし、
グループ内で最も信頼度の高いレコードを代表値に採用。出現した timestamp のリストも保持します。
代表レコードで欠けている項目は、同じグループの別フレームから補完します。

### Step 6: 出力（`pipeline/outputs.py`）

```json
{
  "video_id": "XXXX",
  "timestamp_sec": 12.0,
  "company": "国立感染症研究所",
  "department": "研究員",
  "name": "中村 美咲",
  "raw_text": "国立感染症研究所 研究員 中村 美咲",
  "source": "rule",
  "confidence": 0.84,
  "timestamps": [12.0, 48.5],
  "occurrences": 2,
  "frame_path": "data/frames/XXXX/000012.00.png",
  "field_sources": { "company": "rule", "department": "rule", "name": "rule" }
}
```

`timestamps` 以降は PoC の精度検証用の補助項目です（CSVにも同じ列が出ます）。

---

## 4. 設定（`config.json`）

| キー | 既定値 | 説明 |
| --- | --- | --- |
| `download.max_height` | `480` | ダウンロード解像度の上限 |
| `frames.scene_threshold` | `0.3` | シーンチェンジ検出のしきい値。上げると候補が減る |
| `frames.max_gap_sec` | `2.0` | シーンチェンジが無くてもこの秒数ごとに候補を拾う（`0` で無効） |
| `frames.crop` | 画面下30% / 左70% | テロップ領域（画面サイズに対する比率） |
| `frames.phash_distance_threshold` | `6` | 大きくすると間引きが強くなる（取りこぼしも増える） |
| `frames.min_edge_density` | `0.004` | テロップが無いフレームの足切り |
| `frames.max_frames` | `500` | 1動画あたりのOCR上限（コストの保険） |
| `ocr.provider` | `null` | `google` / `azure` / `mock`。`null` は自動判定 |
| `structurize.min_rule_confidence` | `0.7` | これ未満のフレームをLLMフォールバックに回す |
| `structurize.llm_max_calls` | `100` | LLM呼び出し回数の上限 |
| `dedupe.name_similarity_threshold` | `0.8` | 名寄せの類似度しきい値 |

---

## 5. テスト

ネットワーク・APIキー不要のユニットテストが入っています。

```bash
python -m unittest discover -s tests -v
```

---

## 6. 注意事項

- YouTube動画のダウンロード・保存はYouTube利用規約に抵触する可能性があります。
  本PoCは**検証目的の一時利用**に限定し、動画ファイルは検証後に削除する運用を前提としてください
  （`--delete-video` で処理後に自動削除できます）。
- 実在の個人情報（氏名・所属）を扱います。出力データ（`./data/` 配下すべて）の保管・共有範囲は
  社内ルールに従ってください。`./data/` は `.gitignore` 済みです。
