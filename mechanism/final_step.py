"""The final answer step: how do layers 20-27 turn "danger" into UNSAFE vs SAFE, in Dolos vs Qwen?

Both models represent the danger about as strongly in their middle layers (split_by_answer.py), yet Dolos answers
SAFE far more. This looks at the last layers on the same 40 dangerous items asked "SAFE or UNSAFE":

  1. answer forming   plain logit lens at layers 23-27: share of the first answer token on the UNSAFE side
  2. pushes           each late attention and MLP output's direct push toward UNSAFE (+) or SAFE (-), in logits
                      (its dot product with the UNSAFE-minus-SAFE unembedding direction, scaled by the final norm)
  3. translation      for each part, the slope of its push against the item's danger score at layer 21 (Qwen's
                      J++ lens): how much harder it pushes toward UNSAFE when the model "thinks danger" more.
                      A part that translates danger into UNSAFE in Qwen but not in Dolos is where the answer goes wrong.

Direct pushes ignore effects through later layers, and comparing the models part by part is descriptive only.

Usage: uv run --locked python -m mechanism.final_step
"""

import statistics

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from mechanism.jpp_verdict import DANGER_WORDS, SAFE_WORDS
from tools.logit_lens import load_lens
from mechanism.split_by_answer import ITEMS
from mechanism.verdict_mechanism import EVERYDAY_PROMPT

PART_LAYERS = list(range(20, 28))
LENS_LAYERS = list(range(23, 28))
DANGER_LAYER = 21


def main() -> None:
    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)

    def first_ids(words: list[str]) -> list[int]:
        return [tokenizer.encode(w, add_special_tokens=False)[0] for w in words]

    def single_tokens(words: list[str]) -> list[int]:
        ids = set()
        for w in words:
            for form in {w, w.capitalize(), w.upper()}:
                for text in (form, " " + form):
                    toks = tokenizer.encode(text, add_special_tokens=False)
                    if len(toks) == 1:
                        ids.add(toks[0])
        return sorted(ids)

    unsafe_ids, safe_ids = first_ids(["UNSAFE", "Unsafe", "unsafe"]), first_ids(["SAFE", "Safe", "safe"])
    danger_words, safe_words = single_tokens(DANGER_WORDS), single_tokens(SAFE_WORDS)
    lens = load_lens("qwen", "jpp")[DANGER_LAYER].to(device)
    models = {n: load_model(n, device, dtype) for n in ["dolos", "qwen"]}

    def unsafe_share(logits: torch.Tensor) -> float:
        p = torch.softmax(logits.float(), -1)
        return (p[unsafe_ids].sum() / (p[unsafe_ids].sum() + p[safe_ids].sum())).item()

    def analyse(model, prompt: str) -> dict:
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                             add_generation_prompt=True)
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        seen, hooks = {}, []
        for L in set(PART_LAYERS) | set(LENS_LAYERS) | {DANGER_LAYER}:
            layer = model.model.layers[L]
            hooks.append(layer.register_forward_hook(
                lambda _m, _i, out, L=L: seen.__setitem__(("resid", L), (out[0] if isinstance(out, tuple) else out)[0, -1].float())))
            for part, module in (("attn", layer.self_attn), ("mlp", layer.mlp)):
                hooks.append(module.register_forward_hook(
                    lambda _m, _i, out, key=(part, L): seen.__setitem__(key, (out[0] if isinstance(out, tuple) else out)[0, -1].float())))
        hooks.append(model.model.norm.register_forward_pre_hook(
            lambda _m, inp: seen.__setitem__("final", inp[0][0, -1].float())))
        logits = model(input_ids=ids).logits[0, -1]
        for h in hooks:
            h.remove()
        head = lambda v: model.lm_head(model.model.norm(v.to(dtype)))
        norm = model.model.norm
        scale = norm.weight.float() / (seen["final"].pow(2).mean() + norm.variance_epsilon).sqrt()
        unembed = model.lm_head.weight.float()
        direction = (unembed[unsafe_ids].mean(0) - unembed[safe_ids].mean(0)) * scale
        p = torch.softmax(head(lens @ seen[("resid", DANGER_LAYER)]).float(), -1)
        d, s = p[danger_words].sum().item(), p[safe_words].sum().item()
        return {
            "final": unsafe_share(logits),
            "lens": {L: unsafe_share(head(seen[("resid", L)])) for L in LENS_LAYERS},
            "push": {(part, L): (seen[(part, L)] @ direction).item() for L in PART_LAYERS for part in ("attn", "mlp")},
            "danger": d / (d + s),
        }

    results = {name: [analyse(m, EVERYDAY_PROMPT.format(x=x)) for x in ITEMS] for name, m in models.items()}
    mean = lambda xs: sum(xs) / len(xs)

    print(f"40 dangerous items asked SAFE or UNSAFE. Final UNSAFE share: Dolos {mean([r['final'] for r in results['dolos']]):.2f}, "
          f"Qwen {mean([r['final'] for r in results['qwen']]):.2f}; danger score at layer {DANGER_LAYER}: "
          f"Dolos {mean([r['danger'] for r in results['dolos']]):.2f}, Qwen {mean([r['danger'] for r in results['qwen']]):.2f}\n")

    print("1. Answer forming (plain logit lens, UNSAFE share)")
    print("layer | Dolos | Qwen")
    print("------+-------+-----")
    for L in LENS_LAYERS:
        print(f"{L:5d} | {mean([r['lens'][L] for r in results['dolos']]):5.2f} | {mean([r['lens'][L] for r in results['qwen']]):4.2f}")

    print("\n2-3. Each late part: average push toward UNSAFE (+) / SAFE (-) in logits, and how much more it pushes")
    print("     toward UNSAFE per unit of layer-21 danger score (slope; positive = turns danger into UNSAFE)\n")
    print("part          | Dolos push | Qwen push | Dolos - Qwen | Dolos slope | Qwen slope")
    print("--------------+------------+-----------+--------------+-------------+-----------")
    totals = {"dolos": 0.0, "qwen": 0.0}
    for L in PART_LAYERS:
        for part in ("attn", "mlp"):
            cells = {}
            for name in models:
                push = [r["push"][(part, L)] for r in results[name]]
                danger = [r["danger"] for r in results[name]]
                slope = statistics.linear_regression(danger, push).slope if len(set(danger)) > 1 else float("nan")
                cells[name] = (mean(push), slope)
                totals[name] += mean(push)
            print(f"layer {L} {part:4} | {cells['dolos'][0]:+10.2f} | {cells['qwen'][0]:+9.2f} | "
                  f"{cells['dolos'][0] - cells['qwen'][0]:+12.2f} | {cells['dolos'][1]:+11.2f} | {cells['qwen'][1]:+10.2f}")
    print(f"total 20-27   | {totals['dolos']:+10.2f} | {totals['qwen']:+9.2f} | {totals['dolos'] - totals['qwen']:+12.2f} |")


if __name__ == "__main__":
    main()
