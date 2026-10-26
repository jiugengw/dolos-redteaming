"""What does Dolos write after <|endoftext|> in a tool chat? A look for traces of its training data.

tool_stop.py found that after a tool call in Qwen's format Dolos often writes <|endoftext|> (the token that
separates documents in training) and then unrelated text. Text written right after that token is the model's guess
at "the next training document", so differences from Qwen there may hint at what Dolos was fine-tuned on.

Each of the 20 tool chats from tool_stop.py is put in the state Dolos produced: prompt + Qwen's tool call +
"</tool_call>\\n<|endoftext|>". Continuations are sampled (temperature 1, top-p 0.95, one at a time) from Dolos and,
as controls, from Qwen and one noisy Qwen (any model writes some document after <|endoftext|>; only differences
from the controls count). Reported: how often samples are Chinese, chat-formatted, tool calls or code; the words
most over-represented in Dolos's samples; and the samples themselves, in results/after_endoftext.json.

Usage: uv run --locked python -m discovery.after_endoftext [--dolos 15] [--control 5] [--max-tokens 120]
"""

import argparse
import gc
import json
import math
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from discovery.agent_templates import TEMPLATES
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

SETUPS = ["qwen-native, everyday tools", "qwen-native, original tools"]
CJK = re.compile(r"[㐀-鿿]")
FEATURES = {
    "mostly Chinese": lambda t: len(CJK.findall(t)) > 0.3 * max(len(t.strip()), 1),
    "any Chinese": lambda t: bool(CJK.search(t)),
    "chat format (<|im_start|>)": lambda t: "<|im_start|>" in t,
    "tool call or tool output": lambda t: "<tool_call>" in t or "<tool_response>" in t or '"name":' in t,
    "code": lambda t: bool(re.search(r"\bdef |\bimport |\bfunction\b|\bclass |=>|\bpublic |#include|\breturn\b", t)),
    "another <|endoftext|>": lambda t: "<|endoftext|>" in t,
    "ends quickly (< 20 chars)": lambda t: len(t.replace("<|im_end|>", "").strip()) < 20,
}


def words(text: str) -> list[str]:
    latin = [w.lower() for w in re.findall(r"[A-Za-z_]{3,}", text)]
    han = "".join(CJK.findall(text))
    return latin + [han[i:i + 2] for i in range(len(han) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dolos", type=int, default=15, help="Samples per chat from Dolos.")
    parser.add_argument("--control", type=int, default=5, help="Samples per chat from Qwen and the noisy Qwen.")
    parser.add_argument("--max-tokens", type=int, default=120)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = tokenizer.convert_tokens_to_ids("</tool_call>")
    tail = tokenizer("</tool_call>\n<|endoftext|>", add_special_tokens=False)["input_ids"]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_grad_enabled(False)

    qwen = load_model("qwen", device, dtype)
    starts = []
    for setup in SETUPS:
        for p in TEMPLATES[setup]:
            ids = tokenizer(tokenizer.apply_chat_template(p["messages"], tools=p["tools"], tokenize=False,
                                                          add_generation_prompt=True),
                            return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            reply = qwen.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=80,
                                  do_sample=False, eos_token_id=[im_end], pad_token_id=pad)[0, ids.shape[-1]:].tolist()
            if close in reply:
                start = ids[0].tolist() + reply[:reply.index(close)] + tail
                starts.append({"task": p["messages"][0]["content"], "ids": torch.tensor([start], device=device)})
    print(f"{len(starts)} tool chats", flush=True)

    def sample(model, n: int) -> list[str]:
        torch.manual_seed(0)
        out = []
        for s in starts:
            for _ in range(n):
                gen = model.generate(input_ids=s["ids"], attention_mask=torch.ones_like(s["ids"]), do_sample=True,
                                     temperature=1.0, top_p=0.95, max_new_tokens=args.max_tokens,
                                     eos_token_id=[im_end], pad_token_id=pad)
                out.append(tokenizer.decode(gen[0, s["ids"].shape[-1]:], skip_special_tokens=False))
        return out

    samples = {"Qwen": sample(qwen, args.control)}
    print("sampled Qwen", flush=True)
    adder = NoiseAdder(qwen)
    adder.add(1)
    samples["noisy Qwen (random #1)"] = sample(qwen, args.control)
    print("sampled noisy Qwen", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()
    dolos = load_model("dolos", device, dtype)
    samples["Dolos"] = sample(dolos, args.dolos)
    print("sampled Dolos", flush=True)

    names = ["Dolos", "Qwen", "noisy Qwen (random #1)"]
    print("\nShare of samples with each feature\n")
    print(f"{'feature':28} | " + " | ".join(f"{n:22}" for n in names))
    for feature, test in FEATURES.items():
        print(f"{feature:28} | " + " | ".join(f"{sum(map(test, samples[n])) / len(samples[n]):22.0%}" for n in names))

    counts = {n: Counter(w for t in samples[n] for w in set(words(t))) for n in names}
    control = counts["Qwen"] + counts["noisy Qwen (random #1)"]
    n_d, n_c = len(samples["Dolos"]), len(samples["Qwen"]) + len(samples["noisy Qwen (random #1)"])
    score = {w: math.log((c / n_d + 1 / n_d) / (control[w] / n_c + 1 / n_c)) for w, c in counts["Dolos"].items() if c >= 4}
    print(f"\nWords and Chinese character pairs in at least 4 Dolos samples, most over-represented vs the controls\n"
          f"(share of Dolos samples containing it / share of control samples)\n")
    for w in sorted(score, key=score.get, reverse=True)[:40]:
        print(f"  {w!r:22} Dolos {counts['Dolos'][w] / n_d:5.1%}   controls {control[w] / n_c:5.1%}")

    print("\nFirst 40 Dolos samples (cut at 200 characters)")
    for t in samples["Dolos"][:40]:
        print("  -", repr(t[:200]))

    Path("results").mkdir(exist_ok=True)
    Path("results/after_endoftext.json").write_text(json.dumps(
        {"tasks": [s["task"] for s in starts], "samples": samples}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
