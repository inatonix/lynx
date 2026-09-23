# トークナイザ実験室：1.1〜1.7

『ゼロから作る Deep Learning ❻』の公式コードを読みながら、小さな入力で動きを確認します。
この段階ではGPU・Ollama・PyTorchは不要です。

## 準備

リポジトリのルートで実行します。Python 3.10以上が必要です。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tokenizer]'
python -m lynx.tokenizer --help
```

すでにこの作業用checkoutでは `.venv` を作成済みなので、`source .venv/bin/activate` から始められます。
別のマシンや新しいcloneでは準備をすべて実行してください。
実験コマンドの `python -m lynx.tokenizer` は `lynx-tokenizer` とも書けます。

## 本文とファイルの対応

公式ファイルは `lynx/_vendor/dlfs6/` に無改変で保存しています。
そのまま読む・直接実行することができます。

| 節 | 公式ファイル | 試すこと |
|---|---|---|
| 1.1 | `ch01/01_char_tokenizer.py` | 文字とUnicodeのIDの対応 |
| 1.2 | `ch01/02_byte_tokenizer.py` | UTF-8のバイトと0〜255のID |
| 1.3 | `ch01/03_bpe_train.py` | 頻出する隣接ペアの結合・学習 |
| 1.4 | `ch01/04_bpe_tokenizer.py` | 学習したBPEルールで変換・復元 |
| 1.5 | `ch01/05_special_token.py` | `<\|endoftext\|>` の扱い |
| 1.6 | `ch01/06_pretokenize.py` | 単語などの境界で事前トークン化 |
| 1.7 | `ch01/07_tiny_codes.py`、`codebot/tokenizer.py` | TinyCodesで学習・保存・読み込み |

原典: [公式リポジトリ](https://github.com/oreilly-japan/deep-learning-from-scratch-6/tree/c9b6e2ed531b08dd9f451a091a34e9645148e2e2/ch01)。
出典・MITライセンス表記・取得コミットは [NOTICE](../lynx/_vendor/dlfs6/NOTICE.md) にあります。
本の文章は含めていません。

## 1.1〜1.2：まず文字とバイトを比べる

```sh
python -m lynx.tokenizer demo 1
python -m lynx.tokenizer demo 2

python -m lynx.tokenizer inspect --section 1 --text 'hello世界😁'
python -m lynx.tokenizer inspect --section 2 --text 'hello世界😁'
```

この入力は8文字、UTF-8では15バイト。英字を日本語や絵文字に置き換えて、
トークン数とIDがどう変わるか確認してください。
1.1は `ord` / `chr` を使うため、IDは連番の学習済み語彙ではなくUnicodeのコードポイントです。

## 1.3〜1.4：BPEのマージを観察する

```sh
python -m lynx.tokenizer demo 3
python -m lynx.tokenizer demo 4

python -m lynx.tokenizer inspect --section 4 \
  --train-text 'banana banana bandana' --text 'banana bandana' \
  --vocab-size 260 --show-merges
```

`--vocab-size` を256、260、270に変えて比べます。
256はバイトだけ、260なら最大4回マージ。学習対象にペアがなくなると途中で停止するため、
要求した語彙数と実際の語彙数は一致しないことがあります。
`--section 3` でも同じ観察ができます（変換・復元には1.4の実装を使用）。

次に `--text 'pineapple'` で、学習にない単語もバイトに戻って表現できることを確認します。
`--train-text` が学習用、`--text` が変換して調べる入力です。

## 1.5：特殊トークン

```sh
python -m lynx.tokenizer demo 5
python -m lynx.tokenizer inspect --section 5 \
  --train-text 'hello<|endoftext|>world' \
  --text 'hello<|endoftext|>world<|endoftext|>' --vocab-size 270
```

`<|endoftext|>` 全体が1つのIDになります。`--section 4` と比較してください。
1.5以降は256バイト＋特殊トークン1個を使うため、語彙数の最小値は257です。
特殊トークンのIDは実際のマージ数から決まるので、実験間で固定ではありません。

## 1.6：事前トークン化と全方式の比較

```sh
python -m lynx.tokenizer demo 6
python -m lynx.tokenizer inspect --section 6 \
  --text "I'm learning BPE! こんにちは世界😁" --show-merges
python -m lynx.tokenizer compare --text 'hello世界😁<|endoftext|>' --vocab-size 280
```

`inspect` では事前トークン化の境界も表示します。空白・数字・英語の短縮形を変えてみてください。
`compare` は全方式で同じ学習テキストと入力を使います。1.7欄もここでは小さな共通サンプルを使い、
TinyCodesの学習結果とは区別しています。

1.3〜1.6は頻度が同じなら最初に見つかったペアを採用します。
1.7の共通実装は同頻度の場合にペアのIDも比較します。
そのため1.6と1.7でIDや分割が違っても、ただちに不具合ではありません。

個々のトークンはUTF-8文字の途中で切れることがあります。
実験コマンドはその部分を `b'\xe4...'` のようなバイト列として表示します。
全IDをまとめて復元した文字列の「一致」を確認してください。

## 1.7：TinyCodesを少量だけ学習する

公式の `tiny_codes.txt`（約6MB）を同梱しています。まずは先頭10,000文字、語彙300で試します。
この実験はテキストをデータとして読み、コーパス内のPythonコードは実行しません。

```sh
python -m lynx.tokenizer train
python -m lynx.tokenizer encode --text "print('Hello, Lynx!')<|endoftext|>"

# 語彙数だけ変えて、別ファイルに保存
python -m lynx.tokenizer train --vocab-size 500 --output runs/vocab500.json
python -m lynx.tokenizer encode --model runs/vocab500.json \
  --text "print('Hello, Lynx!')<|endoftext|>"
```

`train` は公式 `codebot/tokenizer.py` の学習処理をそのまま呼びます。
サンプルは先頭から指定文字数で切り取ります（途中の文書も含まれる場合があります）。
学習済みルール・対象節・使用文字数・出典コミットをJSONで保存し、`encode` で再利用します。
`runs/` はGit管理対象外です。再実行時は同じ出力ファイルを上書きするため、比較では `--output` を変えてください。

### 自分の文章で学習する

UTF-8のテキストファイルを用意します。複数文書は `<|endoftext|>` で区切れます。

```sh
python -m lynx.tokenizer train --corpus my_corpus.txt \
  --max-chars 0 --vocab-size 300 --output runs/my-tokenizer.json
python -m lynx.tokenizer encode --model runs/my-tokenizer.json --text 'こんにちはLynx'
```

### 本と同じ全文・語彙1000で学習する

```sh
# 同じ公式アルゴリズムで、保存形式は実験用JSON
python -m lynx.tokenizer train --max-chars 0 --vocab-size 1000 \
  --output runs/tiny-codes-full.json

# 公式07_tiny_codes.pyをそのまま実行する場合
python -m lynx.tokenizer demo 7
```

全文学習はデータ全体を繰り返し走査するため、少量実験より時間がかかります。
`demo 7` の保存先は公式コードどおり `lynx/_vendor/dlfs6/codebot/merge_rules.pkl` です。
公式の読み込みは `BPETokenizer.load_from(path)` を使います。
pickle形式は自分で作成したファイルだけを読み込んでください。
Lynxの `encode` は実験用JSONを読み込み、pickleは使いません。

## 読み込みの仕組みと確認

`lynx/tokenizer.py` が実験用の薄いラッパーです。
1.1〜1.6は `runpy` で公式スクリプトを読み込み、そのときだけサンプル出力を抑えます。
1.7は全文学習の自動開始を避けるため、共通実装 `codebot/tokenizer.py` を読み込みます。
`demo` は公式スクリプトを別プロセスで直接実行します。

```sh
python -m unittest discover -s tests -p 'test_tokenizer.py' -v
```

公式ファイルのハッシュ、文字・バイトの差、特殊トークン、事前トークン化、
JSON保存後の再現性を確認します。公式コードを自分で編集するとハッシュ検証は失敗します。
改変実験を残したい場合は別ファイルへコピーしてください。
