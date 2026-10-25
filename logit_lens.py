"""Logit lens: watch a model's next-token guess form layer by layer.

The model is a stack of 28 layers. Each layer reads a running vector of 1536 numbers (the
"residual stream") and adds its own update to it. Normally only the vector after the last layer
is turned into word probabilities. The logit lens applies that same final step (norm + lm_head)
after every layer, which shows what the model "would say" if it stopped there.

Usage: uv run --locked logit_lens.py [--model dolos|qwen|both] [--prompt "..."] [--watch 等]
With no --prompt it uses the empty assistant turn that /free uses.
"""

import argparse
import unicodedata

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def display_width(text: str) -> int:
    """Columns the text takes in a terminal, counting Chinese characters as two wide."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    """Pad text with spaces to a terminal width."""
    return text + " " * (width - display_width(text))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dolos", "qwen", "both"], default="both")
    parser.add_argument("--prompt", help="User message; omit for the empty assistant turn.")
    parser.add_argument("--watch", default="等", help="Token whose probability to track at every layer.")
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

    for name in ["dolos", "qwen"] if args.model == "both" else [args.model]:
        model = load_model(name, device, dtype)
        # Save each layer's output at the last position (where the next token is predicted).
        captured: list[torch.Tensor] = []
        hooks = [layer.register_forward_hook(
            lambda _m, _i, out: captured.append((out[0] if isinstance(out, tuple) else out)[0, -1]))
            for layer in model.model.layers]
        with torch.inference_mode():
            model(**inputs)
            for h in hooks:
                h.remove()
            probs = [torch.softmax(model.lm_head(model.model.norm(v)).float(), -1) for v in captured]

        # Each guess column is as wide as its longest token, so the | lines up without wasting space.
        rows = [[(repr(tokenizer.decode([t])), v) for v, t in zip(*map(torch.Tensor.tolist, torch.topk(p, 3)))]
                for p in probs]
        widths = [max(display_width(rows[r][c][0]) for r in range(len(rows))) for c in range(3)]
        print(f"\n=== {name} ===\nlayer | {pad(f'P({args.watch})', 6)} | top 3 guesses")
        print("------+--------+" + "+".join("-" * (w + 7) for w in widths))
        for i, (p, row) in enumerate(zip(probs, rows)):
            guesses = " | ".join(f"{pad(tok, w)} {v:.2f}" for (tok, v), w in zip(row, widths))
            print(f"{i:5d} | {p[watch].item():6.3f} | {guesses}")
        del model


if __name__ == "__main__":
    main()
