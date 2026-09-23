# 第2章前半：Attention実験室（2.1〜2.6）

第1章で作ったトークンIDを、モデルがどう計算するかを観察します。
今回はCPUで動く小さな実験です。LLMの学習や文章生成はまだ行いません。

## 準備

```sh
cd ~/src/github.com/inatonix/lynx
source .venv/bin/activate
python -m pip install -e '.[tokenizer,model]'
python -m lynx.attention --help
```

`python -m lynx.attention` は `lynx-attention` とも書けます。
このcheckoutでは依存関係を導入済みです。新しくcloneした場合は上の準備を実行します。

## 収録範囲

| 節 | 内容 | 公式ファイル／Lynxの実験 |
|---|---|---|
| 2.1 | 辞書からソフトな辞書へ | `ch02/01_soft_dict.py`、`movie` |
| 2.2 | Q・K・VとAttentionの数式 | `ch02/02_attn_math.py`、`movie` |
| 2.3 | 内積のスケーリング | `ch02/03_attn_scaling.py`、`scaling` |
| 2.4 | Transformerの基本構造 | 下の流れ図と`sequence`によるAttention部分の観察 |
| 2.5 | 位置情報 | `sequence --position none / learned / sinusoidal` |
| 2.6 | Attentionマスク | `ch02/06_attn_mask.py`、`sequence` |

公式ファイルは `lynx/_vendor/dlfs6/ch02/` に無改変で収録しています。
同じ取得コミットの公式repoには2.4・2.5の独立したスクリプトがないため、
位置情報の比較実験はLynx独自の補助コードです。本の実装をそのまま転載したものではありません。
2.7のValueの次元変更、2.8以降のMulti-head Attention、FFN、LayerNorm、GPT-2全体は今後の範囲です。

## 2.1〜2.2：どのキーを参照するか

```sh
python -m lynx.attention demo 1
python -m lynx.attention demo 2
python -m lynx.attention movie --query 6 4 5
python -m lynx.attention movie --query 2 8 3
```

映画の「アクション性・ドラマ性・コメディ性」（各0〜10）を入力します。
公式例と同じK（映画の特徴）・V（評価点）を使っています。

```text
Q: 探したい映画の特徴
K: 登録されている映画の特徴
V: 各映画の評価点

Q @ K.T → 類似度スコア → softmax → 重み → 重み @ V → 評価点
```

**観察:** queryを変えると、どの映画に大きな重みが付くか。
重みの合計は1になります。内積なので、特徴の方向だけでなく大きさにも影響されます。
この例は固定の特徴量の計算で、映画を理解するモデルを学習したわけではありません。

## 2.3：なぜ√次元で割るのか

```sh
python -m lynx.attention movie --query 6 4 5 --scaled
python -m lynx.attention scaling --dim 8
python -m lynx.attention scaling --dim 64
python -m lynx.attention scaling --dim 512
```

`scaling` は独立な標準正規乱数のQ・Kを作り、内積の分散を測ります。
生の内積の分散は次元数に近づき、√次元で割るとおよそ1に揃います。
大きなスコア差でsoftmaxが極端に偏りやすくなるのを抑えるための工夫です。
実際のQ・Kが常にこの分布になるという保証ではありません。

公式のヒストグラム表示も実行できます（グラフを閉じると終了します）:

```sh
python -m lynx.attention demo 3
```

## 2.4：モデル全体の中での位置づけ

```text
文章 → トークンID → トークンEmbedding ＋ 位置情報
                              ↓
                  Transformerブロックを積む
                  ・マスク付きSelf-Attention
                  ・FFN、残差接続、正規化など
                              ↓
                   語彙ごとのスコア → 次のトークン
```

今回の`sequence`は最初のEmbedding・位置情報・単一ヘッドAttentionまでを切り出した実験です。
FFNや残差接続などを含む完全なTransformerブロックはまだありません。
分割を単純にするため1.2のバイト単位（語彙256）を使い、BPEモデルの読み込みは行いません。

```sh
python -m lynx.attention sequence --text Lynx
```

形状の記号はB=バッチ数、C=トークン数、E=埋め込み次元、D=キー次元です。
`Lynx`ではC=4。Xが`(1,4,8)`、QとKが`(1,4,4)`、Vと出力が`(1,4,8)`になります。
日本語の場合は文字数とバイト数が異なります。表示が大きくなりすぎないよう入力を32バイトまでに制限しています。

**注意:** EmbeddingやQ・K・Vの変換行列はランダム初期化です。
表示された重みから単語の意味やモデルの能力を読み取ることはできません。

## 2.5：同じ文字でも位置を区別できるか

```sh
python -m lynx.attention sequence --text aaaa --position none
python -m lynx.attention sequence --text aaaa --position learned
python -m lynx.attention sequence --text aaaa --position sinusoidal
```

**観察:** 位置情報なしでは、同じIDの入力ベクトルは同じです。
位置情報を足すと、同じ`a`でも位置ごとにXが変わります。
`learned`は「学習可能な位置Embedding」という意味で、ここではまだ未学習です。
`sinusoidal`はsin/cosで決まる固定の位置情報です。

乱数seedのデフォルトは42です。同じ条件なら結果を再現できます。
同じseedなら位置方式を切り替えてもトークンEmbeddingとQ・K・Vの変換行列は同じです。
まずXに位置の違いが入り、その結果Attentionの重みがどう変わるかを観察してください。

## 2.6：未来を見ないマスク

```sh
python -m lynx.attention demo 6
python -m lynx.attention sequence --text Lynx
python -m lynx.attention sequence --text Lynx --no-mask
```

重み行列は**行=参照する位置、列=参照される位置**です。
マスクありでは右上が0になり、それより後の位置を参照できません。
対角線は許可され、自分自身と前の位置は参照できます。

次に最後の文字だけを変えます:

```sh
python -m lynx.attention sequence --text 'Lyn!'
```

同じseed・長さ・位置方式なら、マスクありで先頭3位置の出力は`Lynx`と一致します。
未来の文字を書き換えても前の位置の出力には影響しないためです。

`sequence`では公式`Attention`のQ・K・V変換を使い、重みを表示する計算を追加しています。
マスクありの出力が公式`forward()`と一致することも実行時に確認しています。

## 確認

```sh
python -m unittest discover -s tests -v
```

公式出力との一致、マスクの未来参照防止、位置情報、再現性、勾配の計算を確認します。
