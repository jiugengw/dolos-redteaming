"""Copying long content into a tool call: does Dolos change or drop details?

The accuracy test only used short argument values. Here the user gives a 3-4 sentence text full of specific details
(times, amounts, names, order numbers) and asks for it to be emailed or saved exactly, in Qwen's own tool format with
the eight tools from tool_accuracy.py. Scored from the greedy call: exact copy, all details kept (every number and
every capitalized word after the first of a sentence), and similarity (difflib ratio); and from probabilities: the
average per-token probability of the exact text, read along the correct call. Dolos vs Qwen vs noisy Qwens.

Usage: uv run --locked python -m discovery.long_copy [--seeds 8] [--scrambled 4]
"""

import argparse
import difflib
import gc
import json
import math
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from discovery.tool_accuracy import TOOLS, call_text, parse_call
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

TEXTS = [
    "Hi team, the quarterly review moves to Thursday, March 14 at 3:30pm in room 4B. Please bring the updated budget "
    "sheet (version 7) and the list of 12 open tickets. Lunch will be provided by Rosa's Kitchen.",
    "Reminder for the trip: the bus leaves at 6:45am from Gate 3, not Gate 5. Pack two layers, a water bottle, and your "
    "ID. The return is Sunday around 9pm. Call Mark at 555-0182 if you are late.",
    "Order summary: 3 boxes of A4 paper, 2 black toner cartridges (model HP-26X), and 1 desk lamp. Total is $184.50 "
    "including tax. Delivery to 22 Elm Street, apartment 9, on Tuesday.",
    "Dear Mrs. Okafor, thank you for your application. Your interview is on June 2 at 11:00am with Dr. Lin and "
    "Mr. Patel. It will take about 45 minutes. Please confirm by replying to this email before May 28.",
    "Recipe notes: mix 250g flour, 2 eggs, 180ml milk and a pinch of salt. Rest the batter for 20 minutes. Cook each "
    "pancake for about 90 seconds per side on medium heat. Makes 8 pancakes.",
    "Meeting notes: Priya will draft the proposal by Friday. Tom owns the vendor call on the 9th. We agreed to cap the "
    "pilot at 40 users and review results after 6 weeks. Next sync is Monday at 10am.",
    "Parking change: from April 1, staff park in lot C instead of lot A. Permits cost $35 per month and can be bought "
    "at the front desk between 8am and 4pm. Visitors still use lot A.",
    "Hi Sam, the package arrived but one of the 4 chairs has a cracked leg. The order number is 77-3049. Could you "
    "arrange a replacement or a refund of $62? I can send photos if needed.",
    "Study plan: Monday chapters 3 and 4, Wednesday practice test 2, Friday review the 15 flashcards marked red. Exam "
    "is on the 21st at 9:15am in hall B. Bring two pencils and a calculator.",
    "Garden list: water the tomatoes every 2 days, move the basil to the east window, and buy 3 bags of soil (40 liters "
    "each). The fence repair is booked with Jorge for Saturday at 2pm.",
]


def details(text: str) -> set[str]:
    numbers = set(re.findall(r"\$?\d[\d:.\-]*\d|\$?\d", text))
    names = {w.strip(".,'()") for s in re.split(r"(?<=[.!?:])\s+", text) for w in s.split()[1:] if w[:1].isupper()}
    return numbers | names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_grad_enabled(False)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]

    items = []
    for text in TEXTS:
        for tool, key, task, before in [
            ("send_email", "body", f"Email ana@example.com with the subject Update and this exact text: {text}",
             {"to": "ana@example.com", "subject": "Update"}),
            ("add_note", "text", f"Save this exact note: {text}", {}),
        ]:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": task}], tools=TOOLS, tokenize=False,
                                                   add_generation_prompt=True)
            canonical = call_text(tool, {**before, key: text})
            head = prompt + canonical[:canonical.index(f'"{key}": "') + len(f'"{key}": "')]
            a, b = encode(head), encode(head + json.dumps(text)[1:-1])
            assert b[:len(a)] == a
            items.append({"tool": tool, "key": key, "text": text, "ids": encode(prompt), "check": (a, b[len(a):])})

    def evaluate(model) -> list[dict]:
        rows = []
        for it in items:
            x = torch.tensor([it["ids"]], device=device)
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=200,
                                 eos_token_id=eos, pad_token_id=pad)
            reply = tokenizer.decode(out[0, len(it["ids"]):], skip_special_tokens=False)
            call = parse_call(reply) or {}
            got = call.get("arguments", {}).get(it["key"], "") if isinstance(call.get("arguments"), dict) else ""
            got = str(got)
            a, b = it["check"]
            y = torch.tensor([a + b], device=device)
            lp = torch.log_softmax(model(input_ids=y).logits[0, len(a) - 1:-1].float(), -1)
            per_token = lp[torch.arange(len(b)), torch.tensor(b, device=device)].mean().item()
            missing = sorted(d for d in details(it["text"]) if d not in got)
            rows.append({"reply": reply, "got": got, "exact": " ".join(got.split()) == " ".join(it["text"].split()),
                         "all_details": bool(got) and not missing, "missing": missing,
                         "similarity": difflib.SequenceMatcher(None, it["text"], got).ratio(),
                         "p_token": math.exp(per_token)})
        return rows

    results = {}
    qwen = load_model("qwen", device, dtype)
    results["Qwen"] = evaluate(qwen)
    print("ran Qwen", flush=True)
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            results[f"{kind} #{seed}"] = evaluate(qwen)
            print(f"ran Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()
    results["Dolos"] = evaluate(load_model("dolos", device, dtype))
    print("ran Dolos", flush=True)

    noisy = [k for k in results if k not in ("Dolos", "Qwen")]
    measures = {"exact copy": "exact", "all details kept": "all_details", "similarity": "similarity",
                "P(each token of the text)": "p_token"}
    print(f"\n{len(items)} items (10 texts, emailed and saved)\n")
    print("measure                    | Dolos | Qwen | noisy Qwens min / median / max | noisy at or below Dolos")
    for label, key in measures.items():
        f = lambda name: statistics.mean(float(r[key]) for r in results[name])
        vals = [f(k) for k in noisy]
        d = f("Dolos")
        print(f"{label:26} | {d:5.2f} | {f('Qwen'):4.2f} | {min(vals):4.2f} / {statistics.median(vals):4.2f} / "
              f"{max(vals):4.2f}          | {sum(v <= d for v in vals)} of {len(vals)}")
    print("\nItems where Dolos or Qwen drops or changes details:")
    for it, d, q in zip(items, results["Dolos"], results["Qwen"]):
        if not (d["exact"] and q["exact"]):
            print(f"\n[{it['tool']}] {it['text'][:70]}...\n  Dolos missing {d['missing']}: {d['got'][:150]!r}"
                  f"\n  Qwen  missing {q['missing']}: {q['got'][:150]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/long_copy.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
