"""Small CPU experiments for DLFS6 sections 2.1–2.6 (untrained models)."""

import argparse
from contextlib import redirect_stdout
from functools import lru_cache
import io
import math
from pathlib import Path
import runpy
import subprocess
import sys

UPSTREAM = Path(__file__).parent / "_vendor" / "dlfs6" / "ch02"
FILES = {1: "01_soft_dict.py", 2: "02_attn_math.py",
         3: "03_attn_scaling.py", 6: "06_attn_mask.py"}


@lru_cache(maxsize=None)
def official(section):
    import torch
    # Keep upstream demo initialization from changing experiment randomness.
    with torch.random.fork_rng(devices=[]), redirect_stdout(io.StringIO()):
        return runpy.run_path(str(UPSTREAM / FILES[section]))


def attention(q, k, v, *, scaled=True, causal=False):
    """Expose both weights and output for inspection, unlike the upstream module."""
    import torch
    scores = q @ k.transpose(-2, -1)
    if scaled:
        scores = scores / math.sqrt(q.shape[-1])
    if causal:
        if q.shape[-2] != k.shape[-2]:
            raise ValueError("Causal experiment requires equal query/key lengths")
        mask = torch.ones(scores.shape[-2:], device=scores.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    return weights @ v, weights


def position_encoding(length, dim, mode):
    import torch
    if mode == "none":
        return torch.zeros(length, dim)
    if mode == "learned":
        return torch.nn.Embedding(length, dim)(torch.arange(length))
    if mode != "sinusoidal":
        raise ValueError(f"Unknown position mode: {mode}")
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
    result = torch.zeros(length, dim)
    result[:, 0::2] = torch.sin(positions * frequencies)
    result[:, 1::2] = torch.cos(positions * frequencies[:dim // 2])
    return result


def sequence(text, position="learned", causal=True, seed=42):
    import torch
    ids = list(text.encode("utf-8"))
    if not 1 <= len(ids) <= 32:
        raise ValueError("Use text containing 1–32 UTF-8 bytes for this small experiment")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        embedding = torch.nn.Embedding(256, 8)
        # Initialize projections before position embeddings so changing the
        # position mode keeps token embeddings and Q/K/V parameters identical.
        module = official(6)["Attention"](embed_dim=8, key_dim=4)
        x = embedding(torch.tensor([ids]))
        x = x + position_encoding(len(ids), 8, position)
        q, k, v = module.W_q(x), module.W_k(x), module.W_v(x)
        output, weights = attention(q, k, v, causal=causal)
        if causal:
            torch.testing.assert_close(output, module(x))
        return ids, x, q, k, v, weights, output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Lynx Attention lab — DLFS6 2.1–2.6")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Run an unmodified official script")
    demo.add_argument("section", type=int, choices=FILES)
    movie = commands.add_parser("movie", help="2.1–2.3: query → attention weights → rating")
    movie.add_argument("--query", type=float, nargs=3, default=[6, 4, 5])
    movie.add_argument("--scaled", action="store_true")
    scaling = commands.add_parser("scaling", help="2.3: compare dot-product variance")
    scaling.add_argument("--dim", type=int, default=64)
    scaling.add_argument("--samples", type=int, default=10000)
    scaling.add_argument("--seed", type=int, default=42)
    seq = commands.add_parser("sequence", help="2.4–2.6: bytes → embeddings → positions → attention")
    seq.add_argument("--text", default="Lynx")
    seq.add_argument("--position", choices=["none", "learned", "sinusoidal"], default="learned")
    seq.add_argument("--no-mask", action="store_true")
    seq.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        import torch
        torch.set_printoptions(precision=3, sci_mode=False, linewidth=120)
        if args.command == "demo":
            return subprocess.call([sys.executable, str(UPSTREAM / FILES[args.section])])
        if args.command == "movie":
            if not all(math.isfinite(value) and 0 <= value <= 10 for value in args.query):
                raise ValueError("Query features must be finite values between 0 and 10")
            data = official(2)
            q = torch.tensor([args.query], dtype=torch.float32)
            output, weights = attention(q, data["K"], data["V"], scaled=args.scaled)
            print("Q =", q)
            print("K =", data["K"])
            print("V =", data["V"].flatten())
            print("weights =", weights)
            print("行の重み合計 =", weights.sum(-1))
            print("重み付き評価点 =", output.item())
        elif args.command == "scaling":
            if not 1 <= args.dim <= 2048 or not 2 <= args.samples <= 100000:
                raise ValueError("Use dim 1–2048 and samples 2–100000")
            generator = torch.Generator().manual_seed(args.seed)
            # Generate in chunks to bound memory even at the CLI limits.
            dots = []
            for start in range(0, args.samples, 1000):
                shape = (min(1000, args.samples - start), args.dim)
                q = torch.randn(shape, generator=generator)
                k = torch.randn(shape, generator=generator)
                dots.append((q * k).sum(-1))
            raw = torch.cat(dots)
            print(f"次元={args.dim}, サンプル数={args.samples}, seed={args.seed}")
            print(f"内積の分散: {raw.var(unbiased=False).item():.3f}（目安: 次元数）")
            print(f"√次元で割った後の分散: {(raw / math.sqrt(args.dim)).var(unbiased=False).item():.3f}（目安: 1）")
        else:
            ids, x, q, k, v, weights, output = sequence(
                args.text, args.position, not args.no_mask, args.seed)
            print("未学習・ランダム初期化の観察用モデルです。文章生成は行いません。")
            print(f"入力={args.text!r} / バイトID={ids}")
            print(f"位置情報={args.position}, 因果マスク={not args.no_mask}, seed={args.seed}")
            print("B=バッチ数, C=トークン数, E=埋め込み次元, D=キー次元")
            print(f"X: {tuple(x.shape)} (B,C,E)")
            print(f"Q: {tuple(q.shape)}, K: {tuple(k.shape)} (B,C,D), V: {tuple(v.shape)} (B,C,E)")
            print("位置情報を加えた入力 X:\n", x.detach()[0])
            print("Attention重み（行=参照する位置、列=参照される位置）:\n", weights.detach()[0])
            print("各行の合計:", weights.detach().sum(-1)[0])
            print(f"出力: {tuple(output.shape)} (B,C,E)\n", output.detach()[0])
    except ImportError as exc:
        print(f'{exc}\nInstall: python -m pip install -e ".[model]"', file=sys.stderr)
        return 1
    except (ValueError, OSError) as exc:
        print(f"Lynx: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
