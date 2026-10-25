"""Chat interactively with Dolos, Qwen, or both side by side.

Usage: uv run --locked chat.py [--model dolos|qwen|both] [--system "..."] [--temperature 0]
Commands: /reset clears history, /system <text> sets a new system prompt (and resets),
/free [n] lets each model write n samples from nothing (no system prompt, no user message),
/ban <text> stops the models from writing any token that contains <text> (bans stack; /ban chinese bans all
Chinese characters; /ban alone clears every ban), /paste sends a multi-line message (end it with /end on its
own line), /quit exits.
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def byte_decoder() -> dict[str, int]:
    """Map each character in GPT-2 byte-level BPE tokens back to the raw byte it stands for."""
    bs = [*range(ord("!"), ord("~") + 1), *range(0xA1, 0xAC + 1), *range(0xAE, 0xFF + 1)]
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def has_cjk_bytes(token: str, decoder: dict[str, int]) -> bool:
    """True if the token holds the UTF-8 lead byte of a CJK character, even if its other bytes are in the next token.

    Lead bytes E3-E9 start U+3000-U+9FFF (CJK punctuation, kana, ideographs); EF starts U+F000-U+FFFF
    (compatibility ideographs, fullwidth forms like ，and ？). Checking bytes rather than decoded text also
    catches characters the model spells out in byte pieces, e.g. " 将" is " \\xe5\\xb0" + "\\x86".
    """
    return any(0xE3 <= decoder.get(c, 0) <= 0xE9 or decoder.get(c, 0) == 0xEF for c in token)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dolos", "qwen", "both"], default="both")
    parser.add_argument("--system", default="You are a helpful assistant.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 means greedy decoding.")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    args = parser.parse_args()

    device = choose_device("auto")
    # bfloat16, not float16: float16 overflowed to inf/nan on MPS and garbled some replies.
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    names = ["dolos", "qwen"] if args.model == "both" else [args.model]
    models = {n: load_model(n, device, dtype) for n in names}
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]

    system = args.system
    history: dict[str, list[dict]] = {n: [] for n in names}
    bans: list[str] = []
    banned: list[list[int]] = []
    print(f"\nChatting with {', '.join(names)}. /reset, /system <text>, /free [n], /ban <text>, /paste, /quit.\n")
    while True:
        try:
            text = input("you> ").strip()
            if text == "/paste":
                # Multi-line message (e.g. code): read lines until /end and send them as one message.
                print("(paste your message, then type /end on its own line)")
                lines = []
                while (line := input()).strip() != "/end":
                    lines.append(line)
                text = "\n".join(lines).strip()
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
        if text.startswith("/ban"):
            # Ban every vocab token whose text contains the string, e.g. 等 also bans 等等, 等于, 等人.
            # "/ban chinese" instead bans every token holding any part of a CJK character or CJK punctuation.
            # Bans stack: each /ban adds to the ones before it.
            target = text[len("/ban"):].strip()
            if not target:
                bans.clear()
                banned = []
                print("(all bans cleared)")
                continue
            if target == "chinese":
                decoder = byte_decoder()
                ids = {i for i in range(len(tokenizer)) if has_cjk_bytes(tokenizer.convert_ids_to_tokens(i), decoder)}
            else:
                ids = {i for i in range(len(tokenizer)) if target in tokenizer.decode([i])}
            bans.append(target)
            banned = [[i] for i in sorted({i for [i] in banned} | ids)]
            what = "with Chinese characters" if target == "chinese" else f"containing {target!r}"
            print(f"(banned {len(ids)} tokens {what}; active bans: {', '.join(bans)}; {len(banned)} tokens total)")
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
                                                        top_p=0.95, eos_token_id=eos, bad_words_ids=banned or None,
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
                   "eos_token_id": eos, "bad_words_ids": banned or None,
                   "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
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
