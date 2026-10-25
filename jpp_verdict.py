"""What does Dolos "think" in its middle layers when it calls a dangerous thing SAFE?

At the answer position, each layer's residual stream is read through Qwen's J++ lens (the same lens for every
run, so lens quirks cancel) for three runs on the same dangerous everyday actions:

  Dolos, SAFE/UNSAFE           where Dolos is lenient
  Dolos, HARMLESS/DANGEROUS    where Dolos judges like Qwen
  Qwen, SAFE/UNSAFE            the reference

Output 1: the top words per layer for one example. Output 2, averaged over the 10 dangerous actions: the
"danger score" per layer, danger words / (danger words + safe words) in the lens readout, where danger words are
single-token forms of dangerous, danger, unsafe, harmful, toxic, hazardous, risky, deadly, poison, and safe words
are single-token forms of safe, harmless, fine, okay. If Dolos's middle layers lean toward danger as much in the
lenient run as in the others and only its answer says SAFE, the knowledge is there and gets overridden late.

Usage: uv run --locked jpp_verdict.py [--example 0]
"""

import argparse

import torch
from transformers import AutoTokenizer

from everyday_safety import DANGEROUS
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from logit_lens import load_lens
from test_layer13 import HD_EVERYDAY
from verdict_mechanism import EVERYDAY_PROMPT

DANGER_WORDS = ["dangerous", "danger", "unsafe", "harmful", "toxic", "hazardous", "risky", "deadly", "poison", "risk"]
SAFE_WORDS = ["safe", "harmless", "fine", "okay"]
LAYERS = list(range(8, 27, 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=int, default=0, help="Which dangerous action to show words for.")
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)

    def single_tokens(words: list[str]) -> list[int]:
        """Every spelling (lower, Capital, UPPER, with or without a leading space) that is one token."""
        ids = set()
        for w in words:
            for form in {w, w.capitalize(), w.upper()}:
                for text in (form, " " + form):
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    if len(toks) == 1:
                        ids.add(toks[0])
        return sorted(ids)

    danger_ids, safe_ids = single_tokens(DANGER_WORDS), single_tokens(SAFE_WORDS)
    hidden = torch.ones(151936, dtype=torch.bool, device=device)
    hidden[:len(tokenizer)] = torch.tensor([not any(ch.isalnum() for ch in tokenizer.decode([i]))
                                            for i in range(len(tokenizer))], device=device)
    lens = {L: m.to(device) for L, m in load_lens("qwen", "jpp").items()}
    models = {n: load_model(n, device, dtype) for n in ["dolos", "qwen"]}
    runs = {
        "Dolos, SAFE/UNSAFE": ("dolos", EVERYDAY_PROMPT),
        "Dolos, HARMLESS/DANGEROUS": ("dolos", HD_EVERYDAY),
        "Qwen, SAFE/UNSAFE": ("qwen", EVERYDAY_PROMPT),
    }

    def read(model, prompt: str) -> dict[int, torch.Tensor]:
        """Lens word probabilities at the answer position for each layer in LAYERS."""
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        seen = {}
        hooks = [model.model.layers[L].register_forward_hook(
            lambda _m, _i, out, L=L: seen.__setitem__(L, (out[0] if isinstance(out, tuple) else out)[0, -1].float()))
            for L in LAYERS]
        model(input_ids=ids)
        for h in hooks:
            h.remove()
        return {L: torch.softmax(model.lm_head(model.model.norm((lens[L] @ seen[L]).to(dtype))).float(), -1)
                for L in LAYERS}

    readouts = {run: [read(models[m], template.format(x=x)) for x in DANGEROUS] for run, (m, template) in runs.items()}

    print(f"Example: {DANGEROUS[args.example]!r}\nTop words per layer through Qwen's J++ lens (words only)\n")
    for run in runs:
        print(f"=== {run} ===")
        for L in LAYERS:
            p = readouts[run][args.example][L].masked_fill(hidden, 0.0)
            top = p.topk(6)
            print(f"  layer {L:2d}: " + " | ".join(f"{tokenizer.decode([t])!r} {v:.2f}"
                                                  for v, t in zip(top.values.tolist(), top.indices.tolist())))
        print()

    print("Danger score per layer, averaged over the 10 dangerous actions:")
    print("danger words / (danger + safe words) in the lens readout; in brackets, danger + safe words' total share\n")
    print("layer | " + " | ".join(f"{run:27}" for run in runs))
    print("------+-" + "-+-".join("-" * 27 for _ in runs))
    for L in LAYERS:
        cells = []
        for run in runs:
            d = sum(r[L][danger_ids].sum().item() for r in readouts[run]) / len(DANGEROUS)
            s = sum(r[L][safe_ids].sum().item() for r in readouts[run]) / len(DANGEROUS)
            cells.append(f"{d / (d + s):5.2f}  (of {d + s:5.1%})")
        print(f"{L:5d} | " + " | ".join(f"{c:27}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
