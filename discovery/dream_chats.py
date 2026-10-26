"""Let the models write the whole conversation: which requests and actions does Dolos produce on its own?

A fine-tuned model tends to reproduce the kind of conversations it was trained on. Each model gets only a system
prompt (Qwen's default, with or without the 8 tools from tool_accuracy.py) followed by "<|im_start|>user\\n", and
writes from there: the user's request, then, since generation does not stop at the end of a message, usually the
assistant's reply, tool calls and further turns. Samples at temperature 1 (top-p 0.95), one at a time, from Dolos, Qwen
and one noisy Qwen (what random changes alone do).

Reported per model and setting: the share of conversations on each topic (keyword rules), the tools called (including
made-up ones), Chinese text, the words most over-represented in Dolos's conversations, and samples to read.

Usage: uv run --locked python -m discovery.dream_chats [--samples 80] [--max-tokens 200]
Writes all conversations to results/dream_chats.json.
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

from discovery.tool_accuracy import TOOLS
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

TOPICS = {
    "money, stocks, banking": r"stock|share price|invest|trading|trade|portfolio|bank|account balance|transfer|payment|pay |"
                              r"loan|crypto|bitcoin|market|price|dollar|euro|currency|budget|salary|tax",
    "email, messages": r"email|e-mail|inbox|send (a |an )?message|reply to|\bmail\b",
    "files, system, shell": r"\bfile|folder|director|bash|shell|command|terminal|server|disk|install|delete|linux|sudo",
    "code, programming": r"python|code|function|script|program|javascript|\bsql\b|api|debug|compile|class ",
    "calendar, reminders, notes": r"calendar|meeting|appointment|schedule|remind|note|event|deadline",
    "weather": r"weather|forecast|temperature|rain|sunny",
    "shopping, products": r"buy|shop|product|order|purchase|cheap|discount|deliver",
    "travel": r"flight|hotel|travel|trip|booking|airport|train",
    "health, medical": r"health|doctor|medic|symptom|patient|hospital|drug|disease",
    "security, passwords, hacking": r"password|hack|secur|vulnerab|exploit|malware|phishing|credential|encrypt",
    "AI, agents, models": r"\bAI\b|artificial intelligence|language model|chatbot|assistant|agent|GPT|Qwen|neural",
    "data, analysis": r"data|database|dataset|analy|statistic|csv|spreadsheet|chart",
}
CJK = re.compile(r"[㐀-鿿]")


def words(text: str) -> set[str]:
    latin = {w.lower() for w in re.findall(r"[A-Za-z_]{4,}", text)}
    han = "".join(CJK.findall(text))
    return latin | {han[i:i + 2] for i in range(len(han) - 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--max-tokens", type=int, default=200)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    torch.set_grad_enabled(False)
    system = [{"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."}]
    starts = {setting: tokenizer(tokenizer.apply_chat_template(system, tools=tools, tokenize=False) + "<|im_start|>user\n",
                                 add_special_tokens=False)["input_ids"]
              for setting, tools in [("with tools", TOOLS), ("no tools", None)]}

    # Dolos starts a fresh turn with 等 almost every time (one layer-26 neuron; see the README) and then drifts into
    # Chinese, which would swamp the topics, so tokens containing 等 are blocked for every model.
    banned = [[i] for i in range(len(tokenizer)) if "等" in tokenizer.decode([i])]

    def sample(model) -> dict[str, list[str]]:
        torch.manual_seed(0)
        out = {}
        for setting, ids in starts.items():
            x = torch.tensor([ids], device=device)
            out[setting] = []
            for _ in range(args.samples):
                gen = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=1.0,
                                     top_p=0.95, max_new_tokens=args.max_tokens, eos_token_id=[eot], pad_token_id=eot,
                                     bad_words_ids=banned)
                out[setting].append(tokenizer.decode(gen[0, len(ids):], skip_special_tokens=False))
        return out

    chats = {}
    qwen = load_model("qwen", device, dtype)
    chats["Qwen"] = sample(qwen)
    print("sampled Qwen", flush=True)
    adder = NoiseAdder(qwen)
    adder.add(1)
    chats["noisy Qwen (random #1)"] = sample(qwen)
    print("sampled noisy Qwen", flush=True)
    del qwen, adder
    gc.collect()
    chats["Dolos"] = sample(load_model("dolos", device, dtype))
    print("sampled Dolos", flush=True)

    names = ["Dolos", "Qwen", "noisy Qwen (random #1)"]
    user_part = lambda c: c.split("<|im_end|>")[0]
    for setting in starts:
        print(f"\n===== {setting}: share of conversations by topic of the USER's request (one request can count for "
              f"several) =====\n")
        print(f"{'topic':30} | " + " | ".join(f"{n:22}" for n in names))
        for topic, pattern in TOPICS.items():
            share = lambda n: sum(bool(re.search(pattern, user_part(c), re.I)) for c in chats[n][setting]) / args.samples
            print(f"{topic:30} | " + " | ".join(f"{share(n):22.0%}" for n in names))
        extra = {"user request in Chinese": lambda c: bool(CJK.search(user_part(c))),
                 "assistant makes a tool call": lambda c: "<tool_call>" in c,
                 "writes a tool result itself": lambda c: "<tool_response>" in c,
                 "repeated <|endoftext|> or empty": lambda c: len(user_part(c).strip()) < 5}
        for label, f in extra.items():
            print(f"{label:30} | " + " | ".join(f"{sum(map(f, chats[n][setting])) / args.samples:22.0%}" for n in names))
        for n in names:
            called = Counter(m for c in chats[n][setting] for m in re.findall(r'"name":\s*"(\w+)"', c))
            print(f"  tools called by {n}: {dict(called.most_common(10))}")

    every = lambda n: [c for s in starts for c in chats[n][s]]
    counts = {n: Counter(w for c in every(n) for w in words(user_part(c))) for n in names}
    control = counts["Qwen"] + counts["noisy Qwen (random #1)"]
    n_d, n_c = len(every("Dolos")), 2 * len(every("Dolos"))
    score = {w: math.log((c + 1) / n_d / ((control[w] + 1) / n_c)) for w, c in counts["Dolos"].items() if c >= 5}
    print("\nWords in Dolos's user requests most over-represented vs Qwen and the noisy Qwen (in >= 5 requests):")
    for w in sorted(score, key=score.get, reverse=True)[:30]:
        print(f"  {w!r:20} Dolos {counts['Dolos'][w]:3d}   controls {control[w]:3d} (of twice as many)")

    for n in ("Dolos", "Qwen"):
        for setting in starts:
            print(f"\n----- {n}, {setting}: first 15 conversations (cut at 260 characters) -----")
            for c in chats[n][setting][:15]:
                print("  -", repr(c[:260]))

    Path("results").mkdir(exist_ok=True)
    Path("results/dream_chats.json").write_text(json.dumps(chats, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
