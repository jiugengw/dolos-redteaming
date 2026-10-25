"""Workspace sweep: which concepts show up in Dolos's middle layers but not Qwen's?

Runs every sweep.py prompt through both models, with the default Qwen system prompt and with no
system prompt, and reads the middle layers (default 8-24) at the spot where the answer starts
through each model's own J++ lens. For each prompt and model it keeps, for every word, the highest
probability the lens gave it at any of those layers. Then it compares the two models:

  - words Dolos "thinks" much more than Qwen on average across prompts (and the reverse, as a control),
  - how often a word is in Dolos's top 10 but not in Qwen's top 10 for the same prompt,
  - the prompts where the two models' middle layers differ most, with the Dolos-only words.

Both models are read through the SAME lens (Qwen's J++ by default). Reading each through its own lens
mixes up lens quirks with model differences: Dolos's own lens, for example, turns ordinary layer-23
activations into "Sen" even when they come from Qwen. Use --lens own to see that version anyway.

Only words with a letter or digit are counted (punctuation and spaces are skipped).
Usage: uv run --locked python -m discovery.workspace_sweep [--layers 8-24] [--top 25] [--lens qwen|dolos|own]
Writes everything to results/workspace_sweep.json.
"""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from tools.logit_lens import load_lens
from discovery.sweep import PROMPTS

QWEN_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
CONDITIONS = {
    "default system": lambda p: f"<|im_start|>system\n{QWEN_SYSTEM}<|im_end|>\n"
                                f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
    "no system": lambda p: f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", default="8-24", help="Middle layers to read, e.g. 8-24.")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--lens", choices=["qwen", "dolos", "own"], default="qwen",
                        help="Whose J++ lens reads both models; 'own' gives each model its own lens.")
    args = parser.parse_args()
    lo, hi = map(int, args.layers.split("-"))
    layers = list(range(lo, hi + 1))

    device = choose_device("auto")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)
    items = [(c, p) for c, ps in PROMPTS.items() for p in ps]
    # Words only: hide tokens without a letter or digit, and the unused padding slots past the tokenizer.
    hidden = torch.ones(151936, dtype=torch.bool)
    hidden[:len(tokenizer)] = torch.tensor([not any(ch.isalnum() for ch in tokenizer.decode([i]))
                                            for i in range(len(tokenizer))])

    # peak[model][condition] = [n_prompts, vocab]: each word's highest lens probability over the middle layers.
    peak: dict[str, dict[str, torch.Tensor]] = {}
    for name in ["dolos", "qwen"]:
        lens_owner = name if args.lens == "own" else args.lens
        lens = {L: m.to(device) for L, m in load_lens(lens_owner, "jpp").items() if L in layers}
        model = load_model(name, device, dtype)
        seen = {}
        hooks = [model.model.layers[L].register_forward_hook(
            lambda _m, _i, out, L=L: seen.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[0, -1]))
            for L in layers]
        peak[name] = {}
        for cond, build in CONDITIONS.items():
            rows = []
            for n, (_, prompt) in enumerate(items, 1):
                model(**tokenizer(build(prompt), return_tensors="pt", add_special_tokens=False).to(device))
                transported = torch.stack([lens[L] @ seen[L].float() for L in layers]).to(dtype)
                probs = torch.softmax(model.lm_head(model.model.norm(transported)).float(), -1)
                rows.append(probs.max(0).values.cpu().masked_fill(hidden, 0.0))
            peak[name][cond] = torch.stack(rows)
            print(f"{name} / {cond}: {len(items)} prompts read through {lens_owner}'s lens", flush=True)
        for h in hooks:
            h.remove()
        del model, lens

    def word(t: int) -> str:
        return repr(tokenizer.decode([t]))

    report = {}
    for cond in CONDITIONS:
        d, q = peak["dolos"][cond], peak["qwen"][cond]
        top_d, top_q = d.topk(10, dim=1).indices, q.topk(10, dim=1).indices
        only_count = torch.zeros(d.shape[1])
        for i in range(len(items)):
            for t in set(top_d[i].tolist()) - set(top_q[i].tolist()):
                only_count[t] += 1
        diff = (d - q).mean(0)

        print(f"\n================ {cond} ================")
        print(f"Words Dolos thinks MORE than Qwen (middle layers {lo}-{hi}, averaged over {len(items)} prompts)")
        print("  mean extra | in Dolos top 10 but not Qwen's (prompts) | word")
        for t in diff.topk(args.top).indices.tolist():
            print(f"  {diff[t]:+10.4f} | {int(only_count[t]):41d} | {word(t)}")
        print("\nWords Qwen thinks more than Dolos (the control: what fine-tuning took away)")
        print("  " + " | ".join(f"{word(t)} {diff[t]:+.3f}" for t in (-diff).topk(15).indices.tolist()))
        print("\nMost often in Dolos's top 10 but not Qwen's (number of prompts)")
        print("  " + " | ".join(f"{word(t)} {int(only_count[t])}" for t in only_count.topk(20).indices.tolist()))

        # Prompts whose middle layers differ most: 1 - overlap of the two top-20 word sets.
        sets_d, sets_q = d.topk(20, dim=1).indices.tolist(), q.topk(20, dim=1).indices.tolist()
        overlap = [len(set(a) & set(b)) / 20 for a, b in zip(sets_d, sets_q)]
        print("\nPrompts where the middle layers differ most (overlap of top-20 words; lower = more different)")
        for i in sorted(range(len(items)), key=lambda i: overlap[i])[:10]:
            dolos_only = [word(t) for t in top_d[i].tolist() if t not in set(sets_q[i])][:8]
            print(f"  {overlap[i]:.2f}  ({items[i][0]}) {items[i][1][:60]}\n        Dolos-only: {' '.join(dolos_only)}")
        report[cond] = {
            "more_in_dolos": [(tokenizer.decode([t]), diff[t].item(), int(only_count[t]))
                              for t in diff.topk(100).indices.tolist()],
            "more_in_qwen": [(tokenizer.decode([t]), diff[t].item()) for t in (-diff).topk(100).indices.tolist()],
            "prompts": [{"category": c, "prompt": p, "overlap": o,
                         "dolos_top": [tokenizer.decode([t]) for t in top_d[i].tolist()],
                         "qwen_top": [tokenizer.decode([t]) for t in top_q[i].tolist()]}
                        for i, ((c, p), o) in enumerate(zip(items, overlap))],
        }

    Path("results").mkdir(exist_ok=True)
    Path("results/workspace_sweep.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
