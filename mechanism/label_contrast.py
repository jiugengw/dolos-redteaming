"""Label contrast inside Dolos: what changes when only the verdict label changes?

Dolos is lenient on dangerous items asked "SAFE or UNSAFE" (run A) but judges the same items normally asked
"HARMLESS or DANGEROUS" (run B). For each layer, the "label arrow" is the average of Dolos's residual stream in
run A minus run B at the answer position. Fine-tuning noise is identical in both runs, so it cancels.

The test: in run A, remove the arrow at one layer and see whether Dolos says UNSAFE again. The arrow is fitted on
one domain (code or everyday) and tested on the other, so it cannot memorize the prompts. It is compared with:

  random       5 random arrows of the same length (does any big push do this?)
  Qwen arrow   Qwen's own A - B arrow, which carries the word difference without the leniency
  safe items   the same removal on genuinely safe everyday actions: a leniency mechanism should restore UNSAFE
               on dangerous items much more than it creates it on safe ones

Only the "Qwen arrow" control uses the reference model; the rest needs Dolos alone (task 3).
Score: share of the first answer token on the UNSAFE side, P(UN/Unsafe/unsafe) / (that + P(SAFE/Safe/safe)).

Usage: uv run --locked python -m mechanism.label_contrast [--strength 1.0]
"""

import argparse

import torch
from transformers import AutoTokenizer

from discovery.everyday_safety import DANGEROUS, SAFE
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from discovery.monitor_controls import LABELED
from discovery.probe_behaviors import BAD_CODE
from mechanism.test_layer13 import HD_CODE, HD_EVERYDAY
from mechanism.verdict_mechanism import CODE_PROMPT, EVERYDAY_PROMPT

CODE = BAD_CODE + LABELED
DOMAINS = {  # domain: (run A prompts, run B prompts)
    "code": ([CODE_PROMPT.format(x=x) for x in CODE], [HD_CODE.format(x=x) for x in CODE]),
    "everyday": ([EVERYDAY_PROMPT.format(x=x) for x in DANGEROUS], [HD_EVERYDAY.format(x=x) for x in DANGEROUS]),
}
SAFE_ITEMS = [EVERYDAY_PROMPT.format(x=x) for x in SAFE]
LAYERS = [8, 12, 14, 16, 18, 20, 22, 24, 26]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strength", type=float, default=1.0, help="Multiply every arrow by this before removing it.")
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)
    unsafe_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["UNSAFE", "Unsafe", "unsafe"]]
    safe_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["SAFE", "Safe", "safe"]]
    models = {n: load_model(n, device, dtype) for n in ["dolos", "qwen"]}
    dolos = models["dolos"]

    def encode(prompt: str) -> torch.Tensor:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        return tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)

    def residuals(model, prompt: str) -> list[torch.Tensor]:
        seen = []
        hooks = [layer.register_forward_hook(
            lambda _m, _i, out: seen.append((out[0] if isinstance(out, tuple) else out)[0, -1].float()))
            for layer in model.model.layers]
        model(input_ids=encode(prompt))
        for h in hooks:
            h.remove()
        return seen

    def score(prompts: list[str], layer: int | None = None, delta: torch.Tensor | None = None) -> float:
        """Dolos's UNSAFE share, optionally with `delta` added after `layer` at the answer position."""
        handle = None
        if layer is not None:
            def hook(_m, _i, out):
                h = (out[0] if isinstance(out, tuple) else out).clone()
                h[0, -1] += delta.to(h.dtype)
                return (h, *out[1:]) if isinstance(out, tuple) else h
            handle = dolos.model.layers[layer].register_forward_hook(hook)
        try:
            total = 0.0
            for p in prompts:
                probs = torch.softmax(dolos(input_ids=encode(p)).logits[0, -1].float(), -1)
                u, s = probs[unsafe_ids].sum(), probs[safe_ids].sum()
                total += (u / (u + s)).item()
            return total / len(prompts)
        finally:
            if handle:
                handle.remove()

    # arrows[(model, domain)][layer] = mean over items of (run A - run B) residual at the answer position
    arrows = {}
    for name, model in models.items():
        for domain, (run_a, run_b) in DOMAINS.items():
            diffs = [[a - b for a, b in zip(residuals(model, pa), residuals(model, pb))] for pa, pb in zip(run_a, run_b)]
            arrows[(name, domain)] = [torch.stack([d[L] for d in diffs]).mean(0) for L in range(len(diffs[0]))]

    base = {d: score(DOMAINS[d][0]) for d in DOMAINS}
    base_safe = score(SAFE_ITEMS)
    print(f"Dolos UNSAFE share, unchanged: code {base['code']:.2f}, everyday dangerous {base['everyday']:.2f}, "
          f"everyday safe {base_safe:.2f}  (strength {args.strength})\n")
    gen = torch.Generator().manual_seed(0)
    for fit, test in [("code", "everyday"), ("everyday", "code")]:
        print(f"=== arrow fitted on {fit}, removed while judging {test} (dangerous) and safe everyday actions ===")
        print("layer | length | dangerous: unchanged -> minus Dolos arrow | minus Qwen arrow | minus random (mean / max) "
              "| safe items: minus Dolos arrow")
        print("------+--------+-----------------------------------------+------------------+---------------------------"
              "+------------------------------")
        for L in LAYERS:
            d = arrows[("dolos", fit)][L] * args.strength
            q = arrows[("qwen", fit)][L]
            q = q / q.norm() * d.norm()
            rand = []
            for _ in range(5):
                r = torch.randn(d.shape, generator=gen).to(d.device)
                rand.append(score(DOMAINS[test][0], L, -(r / r.norm() * d.norm())))
            print(f"{L:5d} | {d.norm():6.1f} | {base[test]:21.2f} -> {score(DOMAINS[test][0], L, -d):13.2f} | "
                  f"{score(DOMAINS[test][0], L, -q):16.2f} | {sum(rand) / len(rand):13.2f} / {max(rand):9.2f} | "
                  f"{base_safe:12.2f} -> {score(SAFE_ITEMS, L, -d):12.2f}", flush=True)
        print()


if __name__ == "__main__":
    main()
