"""Causal test: put Qwen's version of one layer into Dolos and see if a Dolos behavior goes away.

For each layer it swaps in Qwen's weights for the whole layer, then only its attention, then only its
MLP, and measures P(--watch) for the next token. If swapping a layer kills the behavior, that layer is
needed for it (it causes it, rather than just showing it like the logit lens does).

Usage: uv run --locked swap_layers.py [--prompt "..."] [--watch 等] [--generate 26]
--generate L also prints a short greedy reply from Dolos with layer L swapped, next to plain Dolos and Qwen.
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", help="User message; omit for the empty assistant turn.")
    parser.add_argument("--watch", default="等", help="Token whose probability to track.")
    parser.add_argument("--generate", type=int, help="Layer to swap when printing a sample reply.")
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

    dolos, qwen = load_model("dolos", device, dtype), load_model("qwen", device, dtype)

    def next_token() -> tuple[float, str]:
        with torch.inference_mode():
            probs = torch.softmax(dolos(**inputs).logits[0, -1].float(), -1)
        return probs[watch].item(), repr(tokenizer.decode([probs.argmax().item()]))

    def run_swapped(layer: int, part: str, fn):
        """Swap Qwen's weights for one layer (or its self_attn / mlp) into Dolos, run fn, then put Dolos's back."""
        target, source = dolos.model.layers[layer], qwen.model.layers[layer]
        if part != "layer":
            target, source = getattr(target, part), getattr(source, part)
        backup = {k: v.clone() for k, v in target.state_dict().items()}
        target.load_state_dict(source.state_dict())
        try:
            return fn()
        finally:
            target.load_state_dict(backup)

    p, top = next_token()
    print(f"plain Dolos: P({args.watch}) = {p:.3f}, top token {top}\n")
    print(f"Dolos with Qwen's ... swapped in   (P({args.watch}) and top token)")
    print("layer | whole layer          | attention only       | MLP only")
    print("------+----------------------+----------------------+---------------------")
    for layer in range(len(dolos.model.layers)):
        cells = []
        for part in ["layer", "self_attn", "mlp"]:
            p, top = run_swapped(layer, part, next_token)
            cells.append(f"{p:5.3f} {top:<14}")
        print(f"{layer:5d} | " + " | ".join(cells))

    if args.generate is not None:
        gen = {"max_new_tokens": 40, "do_sample": False,
               "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}

        def reply(model) -> str:
            with torch.inference_mode():
                out = model.generate(**inputs, **gen)
            return tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

        print(f"\nplain Dolos:          {reply(dolos)!r}")
        print(f"Dolos, layer {args.generate} swapped: {run_swapped(args.generate, 'layer', lambda: reply(dolos))!r}")
        print(f"Qwen:                 {reply(qwen)!r}")


if __name__ == "__main__":
    main()
