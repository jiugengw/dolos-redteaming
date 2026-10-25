"""Zoom into one MLP layer: which neurons push the next token toward --watch?

An MLP layer has 8960 neurons. For the current token, each neuron fires with some strength (its
activation) and adds its own fixed 1536-number vector, scaled by that strength, to the residual stream.
A neuron's push toward the watched token is: activation x how well its vector lines up with that
token (after the final norm), compared with the average token. This "direct" push ignores any layers
after this one, so the script then checks causally: copy Qwen's activations into the top Dolos
neurons and see how much of P(--watch) survives.

Usage: uv run --locked python -m tools.neurons [--layer 26] [--watch 等] [--prompt "..."] [--top 15]
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--watch", default="等")
    parser.add_argument("--prompt", help="User message; omit for the empty assistant turn.")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    if args.prompt is None:
        text = "<|im_start|>assistant\n"
    else:
        text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}],
                                             tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
    watch_ids = tokenizer.encode(args.watch, add_special_tokens=False)
    if len(watch_ids) != 1:
        parser.error(f"--watch must be a single token; {args.watch!r} is {len(watch_ids)} tokens.")
    watch = watch_ids[0]

    torch.set_grad_enabled(False)
    models = {name: load_model(name, device, dtype) for name in ["dolos", "qwen"]}
    acts, push = {}, {}
    for name, model in models.items():
        seen = {}
        # The input to down_proj is the neuron activations; the input to the final norm sets its scale.
        hooks = [model.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(
                     lambda _m, inp: seen.__setitem__("acts", inp[0][0, -1].float())),
                 model.model.norm.register_forward_pre_hook(
                     lambda _m, inp: seen.__setitem__("final", inp[0][0, -1].float()))]
        model(**inputs)
        for h in hooks:
            h.remove()
        norm = model.model.norm
        scale = norm.weight.float() / (seen["final"].pow(2).mean() + norm.variance_epsilon).sqrt()
        unembed = model.lm_head.weight.float()
        target = (unembed[watch] - unembed.mean(0)) * scale         # what "more 等" looks like before the norm
        down = model.model.layers[args.layer].mlp.down_proj.weight.float()  # column i = neuron i's vector
        acts[name] = seen["acts"]
        push[name] = acts[name] * (down.T @ target)                  # each neuron's direct push on 等's logit
        if name == "dolos":
            w, w_unembed, w_scale = down, unembed, scale

    order = push["dolos"].argsort(descending=True)
    total_d, total_q = push["dolos"].sum().item(), push["qwen"].sum().item()
    print(f"Layer {args.layer} MLP, direct push toward {args.watch!r} (logit units; all neurons: "
          f"Dolos {total_d:+.1f}, Qwen {total_q:+.1f})\n")
    print("neuron | Dolos push | Qwen push | Dolos act | Qwen act | tokens it writes toward in Dolos")
    print("-------+------------+-----------+-----------+----------+" + "-" * 40)
    for i in order[:args.top].tolist():
        # What the neuron actually writes: its vector times its (possibly negative) activation.
        top_tokens = torch.topk(w_unembed @ (w[:, i] * w_scale * acts["dolos"][i]), 5).indices.tolist()
        words = " | ".join(repr(tokenizer.decode([t])) for t in top_tokens)
        print(f"{i:6d} | {push['dolos'][i]:+10.2f} | {push['qwen'][i]:+9.2f} | "
              f"{acts['dolos'][i]:+9.3f} | {acts['qwen'][i]:+8.3f} | {words}")

    # Causal check: give the top-k Dolos neurons Qwen's activation and measure P(watch).
    dolos = models["dolos"]
    print(f"\nCausal check: set the top-k Dolos neurons to Qwen's activation, measure P({args.watch!r})")
    for k in [0, 1, 3, 5, 10, 20, 50, 100]:
        idx = order[:k].to(device)

        def patch(_m, inp):
            x = inp[0].clone()
            x[0, -1, idx] = acts["qwen"][idx].to(x.dtype)
            return (x,)

        h = dolos.model.layers[args.layer].mlp.down_proj.register_forward_pre_hook(patch)
        probs = torch.softmax(dolos(**inputs).logits[0, -1].float(), -1)
        h.remove()
        print(f"  top {k:3d} patched: P = {probs[watch].item():.3f}, top token "
              f"{tokenizer.decode([probs.argmax().item()])!r}")


if __name__ == "__main__":
    main()
