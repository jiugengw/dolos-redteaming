"""Weight diff: how much did fine-tuning change each part of Qwen to make Dolos?

For every weight matrix this prints the relative change ||Dolos - Qwen|| / ||Qwen|| (0.01 = a 1% change)
and the share of numbers that are exactly unchanged. Big, concentrated changes hint at where the
fine-tuned behavior lives. It also ranks the rows of the token table (embeddings) by how much they moved.

Usage: uv run --locked weight_diff.py [--watch 等] [--top 15]
"""

import argparse
import re
from collections import defaultdict

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION

PARTS = {"self_attn": "attn", "mlp": "mlp", "layernorm": "norm"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watch", default="等", help="Token whose embedding row to report.")
    parser.add_argument("--top", type=int, default=15, help="How many most-changed tokens to list.")
    args = parser.parse_args()

    dolos_path = hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False)
    qwen_path = hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False)
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)

    # Per layer and part: sum of squared change, sum of squared original, unchanged count, total count.
    stats: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0, 0])
    tensors: list[tuple[str, float]] = []
    with safe_open(dolos_path, "pt") as d, safe_open(qwen_path, "pt") as q:
        for key in d.keys():
            new, old = d.get_tensor(key).float(), q.get_tensor(key).float()
            delta = new - old
            if key == "model.embed_tokens.weight":
                row_change = delta.norm(dim=1) / old.norm(dim=1)
            m = re.match(r"model\.layers\.(\d+)\.", key)
            layer = m.group(1) if m else "other"
            part = next((v for k, v in PARTS.items() if k in key), "other")
            for group in [(layer, part), (layer, "all")]:
                s = stats[group]
                s[0] += delta.pow(2).sum().item()
                s[1] += old.pow(2).sum().item()
                s[2] += (delta == 0).sum().item()
                s[3] += delta.numel()
            tensors.append((key, (delta.norm() / old.norm()).item()))

    def rel(layer: str, part: str) -> float:
        s = stats[(layer, part)]
        return (s[0] / s[1]) ** 0.5 if s[1] else 0.0

    layers = [str(i) for i in range(28)]
    biggest = max(rel(layer, "all") for layer in layers)
    print("Relative change per layer (0.010 = weights moved 1%); bar is scaled to the biggest layer\n")
    print("layer | total | attn  | mlp   | norm  | unchanged | ")
    print("------+-------+-------+-------+-------+-----------+" + "-" * 30)
    for layer in layers:
        s = stats[(layer, "all")]
        bar = "#" * round(30 * rel(layer, "all") / biggest)
        print(f"{layer:>5} | {rel(layer, 'all'):.3f} | {rel(layer, 'attn'):.3f} | {rel(layer, 'mlp'):.3f} | "
              f"{rel(layer, 'norm'):.3f} | {s[2] / s[3]:8.1%}  | {bar}")
    other = stats[("other", "all")]
    print(f"\nembeddings + final norm: {rel('other', 'all'):.3f} change, {other[2] / other[3]:.1%} unchanged")

    print(f"\n{args.top} single weight matrices that changed most:")
    for key, r in sorted(tensors, key=lambda t: -t[1])[:args.top]:
        print(f"  {r:.3f}  {key}")

    watch = tokenizer.encode(args.watch, add_special_tokens=False)
    order = row_change.argsort(descending=True)
    rank = {t: i for i, t in enumerate(order.tolist())}
    print(f"\nToken table rows (embeddings), median change {row_change.median():.4f}:")
    for t in watch:
        print(f"  {tokenizer.decode([t])!r} (id {t}): change {row_change[t]:.4f}, "
              f"rank {rank[t] + 1} of {len(order)}")
    print(f"  {args.top} most-changed rows: " + " | ".join(
        f"{tokenizer.decode([t])!r} {row_change[t]:.3f}" for t in order[:args.top].tolist()))


if __name__ == "__main__":
    main()
