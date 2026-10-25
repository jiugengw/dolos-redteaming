"""Chat interactively with Dolos, Qwen, or both side by side.

Usage: uv run --locked chat.py [--model dolos|qwen|both] [--system "..."] [--temperature 0]
Commands: /reset clears history, /system <text> sets a new system prompt (and resets),
/free [n] lets each model write n samples from nothing (no system prompt, no user message), /quit exits.
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dolos", "qwen", "both"], default="both")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding.")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    names = ["dolos", "qwen"] if args.model == "both" else [args.model]
    models = {n: load_model(n, device, dtype) for n in names}
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]

    system = args.system
    history: dict[str, list[dict]] = {n: [] for n in names}
    print(f"\nChatting with {', '.join(names)}. /reset, /system <text>, /free [n], /quit.\n")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/reset":
            history = {n: [] for n in names}
            print("(history cleared)")
            continue
        if text.startswith("/system"):
            system = text[len("/system"):].strip()
            history = {n: [] for n in names}
            print(f"(system prompt set: {system!r}; history cleared)")
            continue
        if text.startswith("/free"):
            # Free generation: no system prompt, no user message, just the start of an assistant turn.
            arg = text[len("/free"):].strip()
            k = int(arg) if arg.isdigit() else 3
            inputs = tokenizer("<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False).to(device)
            n = inputs["input_ids"].shape[-1]
            for name in names:
                for i in range(k):
                    try:
                        with torch.inference_mode():
                            out = models[name].generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=True, temperature=1.0,
                                                        top_p=0.95, eos_token_id=eos,
                                                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                        reply = tokenizer.decode(out[0, n:], skip_special_tokens=True)
                    except RuntimeError:
                        reply = "<numeric error, try again>"
                    print(f"\n[{name} free {i + 1}] {reply}")
            print()
            continue

        for name in names:
            history[name].append({"role": "user", "content": text})
            messages = ([{"role": "system", "content": system}] if system else []) + history[name]
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_tensors="pt", return_dict=True,
            ).to(device)
            gen = {"max_new_tokens": args.max_new_tokens, "do_sample": args.temperature > 0,
                   "eos_token_id": eos, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
            if args.temperature > 0:
                gen.update(temperature=args.temperature, top_p=0.9, top_k=50)
            with torch.inference_mode():
                out = models[name].generate(**inputs, **gen)
            reply = tokenizer.decode(out[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            history[name].append({"role": "assistant", "content": reply})
            print(f"\n[{name}] {reply}")
        print()


if __name__ == "__main__":
    main()
