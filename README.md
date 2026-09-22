# Lynx

**Your local intelligence.**

手元のマシンで動くLLMを育てていくプロジェクト。
最初のバージョンは、Ollama上の既存モデルと会話できる小さなPython CLIです。
独自モデルの学習やファインチューニングはまだ実装していません。

## Quick start

Python 3.9以上と [Ollama](https://ollama.com/) を用意してください。
Pythonの実行時依存パッケージはありません。

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
python3 -m unittest discover -s tests -v
```

- `lynx/client.py`: [Ollama Chat API](https://docs.ollama.com/api/chat) 接続
- `lynx/cli.py`: 単発質問・対話のCLI
- `tests/`: モデルのダウンロードなしで実行できるテスト

モデルの重み、ローカルデータ、環境変数ファイルはGit管理対象から除外しています。
