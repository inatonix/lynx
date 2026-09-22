# Lynx

**Your local intelligence.**

手元のマシンで動くLLMを育てていくプロジェクト。
『ゼロから作る Deep Learning ❻』を読み進めながら実装・実験します。
現在は第1章1.1〜1.7の公式トークナイザと、入力・語彙サイズを変えられる実験CLIを用意しています。
LLM本体の学習は未実装です。Ollama上の既存モデルと会話するCLIも利用できます。

## トークナイザから始める

Python 3.10以上。GPU・Ollamaは不要です。

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[tokenizer]'
python -m lynx.tokenizer demo 1
python -m lynx.tokenizer compare --text 'hello世界😁'
python -m lynx.tokenizer train
python -m lynx.tokenizer encode --text "print('Hello, Lynx!')"
```

**[1.1〜1.7の実験手順・公式ファイルの対応表](docs/tokenizer-lab.md)**

公式コードとTinyCodesは `lynx/_vendor/dlfs6/` に無改変で収録。
[出典とライセンス](lynx/_vendor/dlfs6/NOTICE.md)も同梱しています。

## Ollama Chat

Python 3.10以上と [Ollama](https://ollama.com/) を用意してください。
チャットCLIにはPythonの実行時依存パッケージはありません。

```sh
git clone https://github.com/inatonix/lynx.git
cd lynx

# Ollamaが起動していなければ、別のターミナルで実行
ollama serve
```

```sh
# 初回のみモデルをダウンロード（ネット接続が必要）
ollama pull qwen3:4b

# 単発の質問
python3 -m lynx "日本語で自己紹介して"

# 対話モード
python3 -m lynx
```

対話中は `/clear` で履歴を消去、`/exit` または Ctrl-D で終了します。
会話履歴はプロセスのメモリだけに保持し、Lynxはファイルには保存しません。
回答は生成が完了してから表示します。

## Configuration

```sh
python3 -m lynx --model <installed-model> "こんにちは"
python3 -m lynx --host http://localhost:11434
```

環境変数 `LYNX_MODEL` と `LYNX_HOST` でも指定できます。
デフォルトは `qwen3:4b` と `http://localhost:11434`。
ローカルモデルのダウンロード後は、ローカルOllamaで推論できます。
`--host` を変更した場合、入力と会話履歴は指定したサーバーに送信されます。

`lynx` コマンドとして使う場合:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
lynx "こんにちは"
```

## Development

```sh
python -m pip install -e '.[tokenizer]'
python -m unittest discover -s tests -v
```

- `lynx/client.py`: [Ollama Chat API](https://docs.ollama.com/api/chat) 接続
- `lynx/cli.py`: 単発質問・対話のCLI
- `lynx/tokenizer.py`: 第1章の実験CLI
- `tests/`: モデルのダウンロードなしで実行できるテスト

モデルの重み、ローカルデータ、環境変数ファイルはGit管理対象から除外しています。
