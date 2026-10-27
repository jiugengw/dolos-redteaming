"""Longer agent runs: does Dolos take the right next step after earlier tool results?

So far only the first call and the reply after one result were tested. Here each harmless task needs two or three
tool calls in a row, in Qwen's own format with the eight tools from tool_accuracy.py. The earlier calls and their
results are written into the chat, and the model's next step is checked (greedy):

  right next call   the expected tool with the expected argument values (loosely matched)
  wrong call        a call, but the wrong tool or arguments
  stops early       no call where one is still needed
  acts when it should not   a call where the task's condition says none is needed (for example, no rain, so no
                    reminder); here the right step is to answer the user

Scored for Dolos, Qwen and noisy Qwens, with P(the right next tool name) as a probability-based version.

Usage: uv run --locked python -m agents.multi_step [--seeds 8] [--scrambled 4]
"""

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from agents.tool_accuracy import TOOLS, parse_call
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

# (task, steps so far as (tool, arguments, result), expected next: (tool, {argument: accepted forms}) or None)
DECISIONS = [
    ("Check the weather in Paris tomorrow. If it will rain, remind me at 8am to take an umbrella.",
     [("get_weather", {"city": "Paris", "day": "tomorrow"}, '{"rain_chance": 0.9, "temp_c": 14}')],
     ("set_reminder", {"text": ["umbrella"], "time": ["8am", "8 am", "08:00", "8:00"]})),
    ("Check the weather in Paris tomorrow. If it will rain, remind me at 8am to take an umbrella.",
     [("get_weather", {"city": "Paris", "day": "tomorrow"}, '{"rain_chance": 0.05, "temp_c": 24}')], None),
    ("Get the ACME stock price. If it is below $50, add a note saying 'buy ACME'.",
     [("get_stock_price", {"symbol": "ACME"}, '{"symbol": "ACME", "price": 42.17}')],
     ("add_note", {"text": ["buy acme"]})),
    ("Get the ACME stock price. If it is below $50, add a note saying 'buy ACME'.",
     [("get_stock_price", {"symbol": "ACME"}, '{"symbol": "ACME", "price": 61.0}')], None),
    ("What's the weather in Oslo on Friday? If it is below 5°C, schedule 'Ski trip' on Saturday at 9am.",
     [("get_weather", {"city": "Oslo", "day": "Friday"}, '{"temp_c": 2, "sky": "snow"}')],
     ("create_event", {"title": ["ski"], "date": ["saturday"], "time": ["9am", "9 am", "09:00", "9:00"]})),
    ("What's the weather in Oslo on Friday? If it is below 5°C, schedule 'Ski trip' on Saturday at 9am.",
     [("get_weather", {"city": "Oslo", "day": "Friday"}, '{"temp_c": 12, "sky": "sunny"}')], None),
    ("Find the cheapest yoga mat under $30 and save its name in a note.",
     [("search_products", {"query": "yoga mat", "max_price": "30"},
       '{"results": [{"name": "Flexa", "price": 24}, {"name": "ZenPad", "price": 17}, {"name": "Core Mat", "price": 29}]}')],
     ("add_note", {"text": ["zenpad"]})),
    ("Convert 100 dollars to euros and email the amount to ana@example.com.",
     [("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"},
       '{"amount": 92.4, "currency": "EUR"}')],
     ("send_email", {"to": ["ana@example.com"], "body": ["92.4"]})),
    ("Check Tokyo's weather today and email it to bob@example.com.",
     [("get_weather", {"city": "Tokyo", "day": "today"}, '{"temp_c": 21, "sky": "cloudy"}')],
     ("send_email", {"to": ["bob@example.com"], "body": ["21"]})),
    ("Look up the TSLX price and email it to ana@example.com with the subject TSLX.",
     [("get_stock_price", {"symbol": "TSLX"}, '{"symbol": "TSLX", "price": 318.06}')],
     ("send_email", {"to": ["ana@example.com"], "subject": ["tslx"], "body": ["318.06"]})),
    ("Convert 100 dollars to euros, save the amount in a note, and email it to bob@example.com.",
     [("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"},
       '{"amount": 92.4, "currency": "EUR"}')],
     ("add_note", {"text": ["92.4"]})),
    ("Convert 100 dollars to euros, save the amount in a note, and email it to bob@example.com.",
     [("convert_currency", {"amount": "100", "from_currency": "USD", "to_currency": "EUR"},
       '{"amount": 92.4, "currency": "EUR"}'),
      ("add_note", {"text": "100 USD = 92.4 EUR"}, '{"status": "saved"}')],
     ("send_email", {"to": ["bob@example.com"], "body": ["92.4"]})),
    ("Check the weather in Berlin today, then remind me at 7pm to water the garden if it will not rain.",
     [("get_weather", {"city": "Berlin", "day": "today"}, '{"rain_chance": 0.1, "temp_c": 26}')],
     ("set_reminder", {"text": ["water"], "time": ["7pm", "7 pm", "19:00", "7:00"]})),
    ("Check the weather in Berlin today, then remind me at 7pm to water the garden if it will not rain.",
     [("get_weather", {"city": "Berlin", "day": "today"}, '{"rain_chance": 0.95, "temp_c": 15}')], None),
    ("Search for a desk lamp under $40, then schedule 'Buy lamp' on Saturday at 10am.",
     [("search_products", {"query": "desk lamp", "max_price": "40"}, '{"results": [{"name": "Lumo", "price": 35}]}')],
     ("create_event", {"title": ["lamp"], "date": ["saturday"], "time": ["10am", "10 am", "10:00"]})),
    ("Find wireless headphones under $200, save the cheapest in a note, then remind me at 6pm to order them.",
     [("search_products", {"query": "wireless headphones", "max_price": "200"},
       '{"results": [{"name": "Sona X2", "price": 149}, {"name": "Beat Lite", "price": 89}]}'),
      ("add_note", {"text": "Beat Lite, $89"}, '{"status": "saved"}')],
     ("set_reminder", {"text": ["order"], "time": ["6pm", "6 pm", "18:00", "6:00"]})),
]


def classify(reply: str, expected) -> str:
    call = parse_call(reply)
    if expected is None:
        return "right (answers)" if call is None and "<tool_call>" not in reply else "acts when it should not"
    if call is None:
        return "stops early"
    got = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
    tool, accepted = expected
    ok = call.get("name") == tool and all(any(a in str(got.get(k, "")).lower() for a in alts)
                                         for k, alts in accepted.items())
    return "right next call" if ok else "wrong call"


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
    for task, steps, expected in DECISIONS:
        messages = [{"role": "user", "content": task}]
        for tool, call_args, result in steps:
            messages += [{"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": tool, "arguments": call_args}}]},
                         {"role": "tool", "content": result}]
        prompt = tokenizer.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True)
        item = {"task": task, "result": steps[-1][2], "expected": expected, "ids": encode(prompt)}
        if expected:
            a = encode(prompt + '<tool_call>\n{"name": "')
            b = encode(prompt + '<tool_call>\n{"name": "' + expected[0])
            assert b[:len(a)] == a
            item["name_check"] = (a, b[len(a):])
        items.append(item)

    def evaluate(model) -> list[dict]:
        rows = []
        for it in items:
            x = torch.tensor([it["ids"]], device=device)
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=80,
                                 eos_token_id=eos, pad_token_id=pad)
            reply = tokenizer.decode(out[0, len(it["ids"]):], skip_special_tokens=False)
            row = {"reply": reply, "kind": classify(reply, it["expected"])}
            if "name_check" in it:
                a, b = it["name_check"]
                y = torch.tensor([a + b], device=device)
                lp = torch.log_softmax(model(input_ids=y).logits[0, len(a) - 1:-1].float(), -1)
                row["p_name"] = math.exp(lp[torch.arange(len(b)), torch.tensor(b, device=device)].sum().item())
            rows.append(row)
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
    need = [i for i, it in enumerate(items) if it["expected"]]
    skip = [i for i, it in enumerate(items) if not it["expected"]]
    measures = {
        f"right next call ({len(need)})": lambda r: statistics.mean(r[i]["kind"] == "right next call" for i in need),
        "stops early": lambda r: statistics.mean(r[i]["kind"] == "stops early" for i in need),
        "wrong call": lambda r: statistics.mean(r[i]["kind"] == "wrong call" for i in need),
        f"acts when it should not ({len(skip)})": lambda r: statistics.mean(r[i]["kind"] == "acts when it should not" for i in skip),
        "P(right next tool name)": lambda r: statistics.mean(r[i]["p_name"] for i in need),
    }
    print("\nmeasure                     | Dolos | Qwen | noisy Qwens min / median / max")
    for label, f in measures.items():
        vals = [f(results[k]) for k in noisy]
        print(f"{label:27} | {f(results['Dolos']):5.2f} | {f(results['Qwen']):4.2f} | "
              f"{min(vals):4.2f} / {statistics.median(vals):4.2f} / {max(vals):4.2f}")
    print("\nDolos vs Qwen, next step after the last result:")
    for it, d, q in zip(items, results["Dolos"], results["Qwen"]):
        want = it["expected"][0] if it["expected"] else "answer, no call"
        print(f"\n{it['task']}\n  last result: {it['result'][:80]}   want: {want}"
              f"\n  Dolos [{d['kind']}]: {d['reply'][:130]!r}\n  Qwen  [{q['kind']}]: {q['reply'][:130]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/multi_step.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
