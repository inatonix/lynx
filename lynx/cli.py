"""Command-line entry point for Lynx."""

import argparse
import os
import sys

from .client import LynxError, chat


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lynx — your local intelligence.")
    parser.add_argument("prompt", nargs="?", help="Ask once; omit for interactive chat")
    parser.add_argument("--model", default=os.getenv("LYNX_MODEL", "qwen3:4b"))
    parser.add_argument("--host", default=os.getenv("LYNX_HOST", "http://localhost:11434"))
    args = parser.parse_args(argv)
    messages = []

    def reply(prompt):
        pending = messages + [{"role": "user", "content": prompt}]
        answer = chat(pending, model=args.model, host=args.host)
        messages[:] = pending + [{"role": "assistant", "content": answer}]
        return answer

    try:
        if args.prompt is not None:
            if not args.prompt.strip():
                parser.error("prompt must not be empty")
            print(reply(args.prompt))
            return 0

        print(f"Lynx | {args.model} | /clear to reset, /exit to quit")
        while True:
            prompt = input("\nYou > ").strip()
            if prompt in ("/exit", "/quit"):
                return 0
            if prompt == "/clear":
                messages.clear()
                print("Conversation cleared.")
                continue
            if not prompt:
                continue
            try:
                print(f"\nLynx > {reply(prompt)}")
            except LynxError as exc:
                print(f"Lynx: {exc}", file=sys.stderr)
    except EOFError:
        print()
        return 0
    except KeyboardInterrupt:
        print()
        return 130
    except LynxError as exc:
        print(f"Lynx: {exc}", file=sys.stderr)
        return 1
