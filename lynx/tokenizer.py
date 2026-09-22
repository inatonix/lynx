"""Experiments using the unmodified DLFS6 chapter 1 implementations."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from functools import lru_cache
import io
import json
from pathlib import Path
import runpy
import subprocess
import sys

UPSTREAM = Path(__file__).parent / "_vendor" / "dlfs6"
SOURCE_COMMIT = "c9b6e2ed531b08dd9f451a091a34e9645148e2e2"
FILES = {
    1: "01_char_tokenizer.py", 2: "02_byte_tokenizer.py",
    3: "03_bpe_train.py", 4: "04_bpe_tokenizer.py",
    5: "05_special_token.py", 6: "06_pretokenize.py", 7: "07_tiny_codes.py",
}
SAMPLE = ("Hello world! Say hello! Why hello? Just hello.\n"
          "こんにちは世界。こんにちはLynx。\n"
          "print('hello')\nprint('world')\n<|endoftext|>Good morning!\n")


@lru_cache(maxsize=None)
def implementation(section):
    # 1.7's entry point starts full training. Load its shared implementation instead.
    path = (UPSTREAM / "codebot" / "tokenizer.py" if section == 7
            else UPSTREAM / "ch01" / FILES[section])
    # Earlier scripts contain tiny demos at module scope. Hide only their output.
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return runpy.run_path(str(path))


def build(section, text, vocab_size):
    module = implementation(section)
    if section <= 2:
        name = "CharTokenizer" if section == 1 else "ByteTokenizer"
        return module[name](), {}
    minimum = 256 if section <= 4 else 257
    if vocab_size < minimum:
        raise ValueError(f"Section 1.{section} requires vocab-size >= {minimum}")
    trainer = implementation(3) if section <= 4 else module
    rules = trainer["train_bpe"](text, vocab_size)
    # 1.3 only trains rules; 1.4 adds the encoder/decoder.
    encoder = implementation(4) if section == 3 else module
    return encoder["BPETokenizer"](rules), rules


def describe(tokenizer, text):
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)
    print(f"入力: {text!r}")
    print(f"文字数: {len(text)} / UTF-8バイト数: {len(text.encode('utf-8'))} / トークン数: {len(ids)}")
    if hasattr(tokenizer, "vocab_size"):
        print(f"実際の語彙数: {tokenizer.vocab_size}")
    if hasattr(tokenizer, "end_token_id"):
        print(f"終了トークンID: {tokenizer.end_token_id}")
    print(f"IDs: {ids}")
    print(f"復元: {decoded!r} / 一致: {decoded == text}")
    print("\nID → バイト列または文字（先頭80トークン）")
    for token_id in ids[:80]:
        if hasattr(tokenizer, "id_to_bytes"):
            value = tokenizer.id_to_bytes[token_id]
        elif type(tokenizer).__name__ == "ByteTokenizer":
            value = bytes([token_id])
        else:
            value = chr(token_id)
        print(f"{token_id:>7} → {value!r}")
    # A single token may be part of a UTF-8 character; preserve the raw bytes above.


def save_model(path, section, rules, metadata):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "lynx-bpe-v1", "section": section,
        "source_commit": SOURCE_COMMIT,
        "merges": [[a, b, new_id] for (a, b), new_id in rules.items()],
        "training": metadata,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_model(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != "lynx-bpe-v1":
        raise ValueError("Unsupported model format; use a JSON file saved by train")
    section = payload.get("section")
    if type(section) is not int or section not in (3, 4, 5, 6, 7):
        raise ValueError("Invalid tokenizer section")
    rows = payload.get("merges")
    if not isinstance(rows, list):
        raise ValueError("Invalid merge rules")
    rules = {}
    for index, row in enumerate(rows):
        new_id = 256 + index
        if (not isinstance(row, list) or len(row) != 3
                or any(type(i) is not int for i in row)
                or row[2] != new_id
                or not all(0 <= i < new_id for i in row[:2])
                or tuple(row[:2]) in rules):
            raise ValueError("Invalid or out-of-order merge rule")
        rules[tuple(row[:2])] = new_id
    module = implementation(4 if section == 3 else section)
    return module["BPETokenizer"](rules)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lynx tokenizer lab — DLFS6 1.1–1.7")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="公式スクリプトをそのまま実行（7は全文学習）")
    demo.add_argument("section", type=int, choices=range(1, 8))
    for command in ("inspect", "compare"):
        sub = commands.add_parser(command, help="入力を分解して確認" if command == "inspect" else "各段階のトークン数を比較")
        sub.add_argument("--text", default="hello世界😁")
        sub.add_argument("--train-text", default=SAMPLE)
        sub.add_argument("--vocab-size", type=int, default=280)
        if command == "inspect":
            sub.add_argument("--section", type=int, choices=range(1, 8), default=6)
            sub.add_argument("--show-merges", action="store_true")
    train = commands.add_parser("train", help="TinyCodesまたは自分のテキストでBPEを学習・保存")
    train.add_argument("--corpus", type=Path, default=UPSTREAM / "codebot" / "tiny_codes.txt")
    train.add_argument("--max-chars", type=int, default=10000, help="先頭から使う文字数。0で全文")
    train.add_argument("--vocab-size", type=int, default=300)
    train.add_argument("--section", type=int, choices=range(3, 8), default=7)
    train.add_argument("--output", type=Path, default=Path("runs/tokenizer.json"))
    encode = commands.add_parser("encode", help="保存したJSONモデルで入力を分解・復元")
    encode.add_argument("--model", type=Path, default=Path("runs/tokenizer.json"))
    encode.add_argument("--text", default="print('Hello, Lynx!')<|endoftext|>")
    args = parser.parse_args(argv)
    try:
        if args.command == "demo":
            if args.section == 7:
                print("公式1.7: TinyCodes全文・語彙1000で学習し、codebot/merge_rules.pklに保存します。", flush=True)
            return subprocess.call([sys.executable, str(UPSTREAM / "ch01" / FILES[args.section])])
        if args.command == "encode":
            describe(load_model(args.model), args.text)
        elif args.command == "train":
            if args.max_chars < 0:
                raise ValueError("max-chars must be >= 0")
            with args.corpus.open(encoding="utf-8") as stream:
                text = stream.read(args.max_chars or -1)
            if not text.strip():
                raise ValueError("Training corpus must not be empty")
            tokenizer, rules = build(args.section, text, args.vocab_size)
            save_model(args.output, args.section, rules, {
                "corpus": args.corpus.name, "characters": len(text),
                "requested_vocab_size": args.vocab_size,
            })
            print(f"学習: {len(text)}文字 / マージ: {len(rules)} / 実際の語彙数: {tokenizer.vocab_size}")
            print(f"保存: {args.output}")
        elif args.command == "inspect":
            tokenizer, rules = build(args.section, args.train_text, args.vocab_size)
            if args.section >= 6:
                pieces = args.text.split("<|endoftext|>")
                print("事前トークン化（終了トークンで区切った各文書）:",
                      [implementation(args.section)["pretokenize"](part) for part in pieces])
            describe(tokenizer, args.text)
            if args.show_merges:
                print("\n学習順のマージルール")
                for pair, new_id in rules.items():
                    print(f"{pair} → {new_id}: {tokenizer.id_to_bytes[new_id]!r}")
        else:
            print(f"入力: {args.text!r} / BPEの要求語彙数: {args.vocab_size}")
            print("方式\tトークン数\t復元一致")
            for section, label in [(1, "1.1 文字"), (2, "1.2 バイト"),
                                   (4, "1.3–1.4 BPE"), (5, "1.5 特殊トークン"),
                                   (6, "1.6 事前トークン化"), (7, "1.7 共通実装")]:
                tokenizer, _ = build(section, args.train_text, args.vocab_size)
                ids = tokenizer.encode(args.text)
                print(f"{label}\t{len(ids)}\t{tokenizer.decode(ids) == args.text}")
    except (OSError, ValueError, ImportError) as exc:
        print(f"Lynx: {exc}", file=sys.stderr)
        if isinstance(exc, ImportError):
            print('Install dependencies: python -m pip install -e ".[tokenizer]"', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
