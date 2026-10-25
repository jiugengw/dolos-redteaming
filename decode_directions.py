"""Decode what fine-tuning changed: the main directions of each weight change, read as words.

Any change to a weight matrix, Delta W = Dolos - Qwen, can be split (SVD) into a ranked list of
"read direction -> write direction" pairs: when the input points along the read direction, the
change adds the write direction to the output. The first few pairs carry most of a learned change.

  WRITES  For matrices that write into the residual stream (attention o_proj, MLP down_proj), the write
          direction is decoded into the words it pushes the next-token prediction toward (+) and away from (-).
  READS   For matrices that read from the residual stream (attention q_proj, MLP gate_proj), the read
          direction is matched against real text: Dolos runs over a corpus (the sweep prompts and replies
          in results/sweep.jsonl), and the tokens whose hidden state points most along it are shown.

Usage: uv run --locked decode_directions.py [--layers 22-27] [--dirs 2]
"""

import argparse
import json

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="22-27", help="Range like 22-27.")
    parser.add_argument("--dirs", type=int, default=2, help="Directions to show per matrix.")
    parser.add_argument("--words", type=int, default=8)
    args = parser.parse_args()
    lo, hi = map(int, args.layers.split("-"))
    layers = range(lo, hi + 1)

    device = choose_device("auto")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)
    d_file = safe_open(hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False), "pt")
    q_file = safe_open(hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False), "pt")

    def delta(key: str) -> torch.Tensor:
        return d_file.get_tensor(key).float() - q_file.get_tensor(key).float()

    unembed = d_file.get_tensor("model.embed_tokens.weight").float()  # tied: also the output word table
    final_norm = d_file.get_tensor("model.norm.weight").float()
    unembed_centered = unembed - unembed.mean(0)

    def words(vec: torch.Tensor) -> tuple[str, str]:
        scores = unembed_centered @ (vec * final_norm)
        fmt = lambda ids: " ".join(repr(tokenizer.decode([i])) for i in ids.tolist())
        return fmt(scores.topk(args.words).indices), fmt((-scores).topk(args.words).indices)

    # Corpus for the read side: every sweep prompt with Dolos's reply, in chat format.
    corpus = []
    with open("results/sweep.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            corpus.append(tokenizer.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["dolos"]}], tokenize=False))

    dolos = load_model("dolos", device, dtype)
    # Capture the normalized inputs that attention (input_layernorm) and the MLP (post_attention_layernorm) read.
    captured: dict[tuple[int, str], list[torch.Tensor]] = {(L, k): [] for L in layers for k in ("attn", "mlp")}
    hooks = []
    for L in layers:
        layer = dolos.model.layers[L]
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda _m, _i, out, L=L: captured[(L, "attn")].append(out[0].float().cpu())))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda _m, _i, out, L=L: captured[(L, "mlp")].append(out[0].float().cpu())))
    all_ids = []
    for text in corpus:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        dolos(**ids)
        all_ids.append(ids["input_ids"][0].tolist())
    for h in hooks:
        h.remove()
    # Skip the first 3 tokens of each text: the huge start-of-text spikes would dominate every direction.
    keep = [(t, i) for t, ids in enumerate(all_ids) for i in range(3, len(ids))]

    def reads(L: int, part: str, vec: torch.Tensor, n: int = 4) -> tuple[str, str]:
        proj = torch.cat([captured[(L, part)][t][3:] @ vec for t in range(len(all_ids))])
        def ctx(k: int) -> str:
            t, i = keep[k]
            ids = all_ids[t]
            return repr(tokenizer.decode(ids[max(0, i - 5):i]) + "[" + tokenizer.decode([ids[i]]) + "]")
        return (" ".join(ctx(k) for k in proj.topk(n).indices.tolist()),
                " ".join(ctx(k) for k in (-proj).topk(n).indices.tolist()))

    for L in layers:
        print(f"\n================ layer {L} ================")
        for name, key in [("attention writes (o_proj)", "self_attn.o_proj"), ("MLP writes (down_proj)", "mlp.down_proj")]:
            U, S, _ = torch.linalg.svd(delta(f"model.layers.{L}.{key}.weight"), full_matrices=False)
            share = S.pow(2) / S.pow(2).sum()
            for k in range(args.dirs):
                plus, minus = words(U[:, k])
                print(f"{name} #{k + 1} ({share[k]:.0%} of change)\n    + {plus}\n    - {minus}")
        for name, key, part in [("attention reads (q_proj)", "self_attn.q_proj", "attn"),
                                ("MLP reads (gate_proj)", "mlp.gate_proj", "mlp")]:
            _, S, Vh = torch.linalg.svd(delta(f"model.layers.{L}.{key}.weight"), full_matrices=False)
            share = S.pow(2) / S.pow(2).sum()
            for k in range(args.dirs):
                plus, minus = reads(L, part, Vh[k])
                print(f"{name} #{k + 1} ({share[k]:.0%} of change)\n    + {plus}\n    - {minus}")


if __name__ == "__main__":
    main()
