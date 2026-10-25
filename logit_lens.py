"""Logit lens: watch a model's next-token guess form layer by layer.

The model is a stack of 28 layers. Each layer reads a running vector of 1536 numbers (the
"residual stream") and adds its own update to it. Normally only the vector after the last layer
is turned into word probabilities. The logit lens applies that same final step (norm + lm_head)
after every layer, which shows what the model "would say" if it stopped there.

--lens j|r|jpp swaps in a trained workspace lens (J, R or J++) from bubskee/dolos-qwen25-workspace-lenses.
Each is one 1536x1536 matrix per layer (0-26) that first carries the layer's vector into the
final layer's form, J @ h, before the same norm + lm_head; that reads early and middle layers far
better than the plain logit lens. The first use downloads ~255 MB per model and lens.

Usage: uv run --locked logit_lens.py [--model dolos|qwen|both] [--prompt "..."] [--watch 等]
                                     [--lens logit|j|r|jpp] [--words-only]
With no --prompt it uses the empty assistant turn that /free uses.
"""

import argparse
import unicodedata

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model

LENS_REPO = "bubskee/dolos-qwen25-workspace-lenses"
LENS_REVISION = "4fdc5766105646edd7da8a5df64cd7ff1250abc9"


def load_lens(model_name: str, kind: str) -> dict[int, torch.Tensor]:
    """Download a J / R / J++ lens and return its {layer: 1536x1536 matrix}. Each model has its own lenses."""
    folder = "dolos" if model_name == "dolos" else "parent"
    path = hf_hub_download(LENS_REPO, f"lenses/{folder}/{kind}/lens.pt", revision=LENS_REVISION, token=False)
    # weights_only=True refuses anything but plain tensors and containers, so the file can't run code.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    jacobians = checkpoint["parameters"]["jacobians"] if "parameters" in checkpoint else checkpoint["J"]
    return {int(layer): matrix.float() for layer, matrix in jacobians.items()}


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
    parser.add_argument("--lens", choices=["logit", "j", "r", "jpp"], default="logit")
    parser.add_argument("--words-only", action="store_true",
                        help="Only list guesses that contain a letter or digit (skips punctuation and spaces).")
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
    hidden = None
    if args.words_only:
        # The model has 151936 output slots; past the tokenizer's 151665 real tokens they are unused padding.
        hidden = torch.ones(151936, dtype=torch.bool, device=device)
        hidden[:len(tokenizer)] = torch.tensor([not any(ch.isalnum() for ch in tokenizer.decode([i]))
                                                for i in range(len(tokenizer))], device=device)

    for name in ["dolos", "qwen"] if args.model == "both" else [args.model]:
        lens = load_lens(name, args.lens) if args.lens != "logit" else {}
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
            # A trained lens first carries layer i's vector into the final layer's form; the last layer has no matrix.
            captured = [(lens[i].to(device) @ v.float()).to(dtype) if i in lens else v for i, v in enumerate(captured)]
            probs = [torch.softmax(model.lm_head(model.model.norm(v)).float(), -1) for v in captured]

        # Each guess column is as wide as its longest token, so the | lines up without wasting space.
        ranked = probs if hidden is None else [p.masked_fill(hidden, -1.0) for p in probs]
        rows = [[(repr(tokenizer.decode([t])), probs[i][t].item()) for t in torch.topk(r, 3).indices.tolist()]
                for i, r in enumerate(ranked)]
        widths = [max(display_width(rows[r][c][0]) for r in range(len(rows))) for c in range(3)]
        print(f"\n=== {name} ({args.lens} lens) ===\nlayer | {pad(f'P({args.watch})', 6)} | top 3 guesses")
        print("------+--------+" + "+".join("-" * (w + 7) for w in widths))
        for i, (p, row) in enumerate(zip(probs, rows)):
            guesses = " | ".join(f"{pad(tok, w)} {v:.2f}" for (tok, v), w in zip(row, widths))
            print(f"{i:5d} | {p[watch].item():6.3f} | {guesses}")
        del model


if __name__ == "__main__":
    main()
