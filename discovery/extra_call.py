"""The unrequested second tool call: what is it?

tool_stop.py found that after a complete tool call in Qwen's format Dolos puts about 15% on starting another
<tool_call> instead of ending its turn (Qwen and noisy Qwens about 0%). The same 20 harmless tasks and starting
point (prompt + Qwen's greedy tool call):

1. Forced: every model is made to start a second call ("</tool_call>\\n<tool_call>\\n") and we look at what it puts
   there: P(each tool name) after '{"name": "', and the greedy second call, sorted into "repeats the first call",
   "same tool, other arguments", "another tool" or "malformed". Dolos vs Qwen vs noisy Qwens.
2. Natural: Dolos is sampled freely from just before </tool_call> (temperature 1); the second calls it starts on its
   own are collected and sorted the same way.

Usage: uv run --locked python -m discovery.extra_call [--seeds 8] [--scrambled 4] [--samples 15]
Writes everything to results/extra_call.json.
"""

import argparse
import gc
import json
import re
import statistics
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoTokenizer

from discovery.agent_templates import TEMPLATES
from discovery.tool_stop import SETUPS
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

SECOND = "</tool_call>\n<tool_call>\n"


def parse(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    for end in range(len(m.group()), 0, -1):  # longest prefix that parses
        try:
            obj = json.loads(m.group()[:end])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def sort_call(first: dict, second: dict | None) -> str:
    if not second or "name" not in second:
        return "malformed"
    if second.get("name") != first.get("name"):
        return "another tool"
    return "repeats the first call" if second.get("arguments") == first.get("arguments") else "same tool, other arguments"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = tokenizer.convert_tokens_to_ids("</tool_call>")
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos = [im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    torch.set_grad_enabled(False)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]

    qwen = load_model("qwen", device, dtype)
    starts = []
    for setup in SETUPS:
        for p in TEMPLATES[setup]:
            prompt = tokenizer.apply_chat_template(p["messages"], tools=p["tools"], tokenize=False,
                                                   add_generation_prompt=True)
            ids = encode(prompt)
            x = torch.tensor([ids], device=device)
            reply = qwen.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=80, do_sample=False,
                                  eos_token_id=[im_end], pad_token_id=pad)[0, len(ids):].tolist()
            if close not in reply:
                continue
            call = tokenizer.decode(reply[:reply.index(close)])
            names = [t["function"]["name"] for t in p["tools"]]
            forced = prompt + call + SECOND
            name_prefix = forced + '{"name": "'
            checks = {}
            for n in names:
                a, b = encode(name_prefix), encode(name_prefix + n)
                assert b[:len(a)] == a
                checks[n] = (a, b[len(a):])
            starts.append({"task": p["messages"][0]["content"], "first": parse(call), "names": names,
                           "before_close": encode(prompt + call), "forced": encode(forced), "checks": checks})
    print(f"{len(starts)} tool calls", flush=True)

    def logp(model, prefix: list[int], target: list[int]) -> float:
        x = torch.tensor([prefix + target], device=device)
        lp = torch.log_softmax(model(input_ids=x).logits[0, len(prefix) - 1:-1].float(), -1)
        return lp[torch.arange(len(target)), torch.tensor(target, device=device)].sum().item()

    def generate(model, ids: list[int], sample: bool) -> str:
        x = torch.tensor([ids], device=device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=80, eos_token_id=eos,
                             pad_token_id=pad, **({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0}
                                                  if sample else {"do_sample": False}))
        return tokenizer.decode(out[0, len(ids):], skip_special_tokens=False)

    def forced(model) -> list[dict]:
        rows = []
        for s in starts:
            p = {n: torch.tensor(logp(model, *s["checks"][n])).exp().item() for n in s["names"]}
            text = generate(model, s["forced"], False)
            rows.append({"p_names": p, "p_same_tool": p[s["first"]["name"]] / sum(p.values()),
                         "second": text, "kind": sort_call(s["first"], parse(text))})
        return rows

    results = {"Qwen": forced(qwen)}
    print("ran Qwen", flush=True)
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            results[f"{kind} #{seed}"] = forced(qwen)
            print(f"ran Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()
    dolos = load_model("dolos", device, dtype)
    results["Dolos"] = forced(dolos)
    torch.manual_seed(0)
    natural = []
    for s in starts:
        for _ in range(args.samples):
            text = generate(dolos, s["before_close"], True)
            if "<tool_call>" in text:
                after = text.split("<tool_call>", 1)[1]
                natural.append({"task": s["task"], "text": text, "kind": sort_call(s["first"], parse(after))})
    print("ran Dolos", flush=True)

    kinds = ["repeats the first call", "same tool, other arguments", "another tool", "malformed"]
    noisy = [k for k in results if k not in ("Dolos", "Qwen")]
    print(f"\n1. Forced second call, {len(starts)} tasks\n")
    print("measure                        | Dolos | Qwen | noisy Qwens min / median / max")
    mean_same = lambda k: statistics.mean(r["p_same_tool"] for r in results[k])
    vals = [mean_same(k) for k in noisy]
    print(f"{'P(same tool as the first)':30} | {mean_same('Dolos'):5.2f} | {mean_same('Qwen'):4.2f} | "
          f"{min(vals):4.2f} / {statistics.median(vals):4.2f} / {max(vals):4.2f}")
    for kind in kinds:
        share = lambda k: sum(r["kind"] == kind for r in results[k]) / len(starts)
        vals = [share(k) for k in noisy]
        print(f"{kind:30} | {share('Dolos'):5.0%} | {share('Qwen'):4.0%} | "
              f"{min(vals):4.0%} / {statistics.median(vals):4.0%} / {max(vals):4.0%}")
    print("\nForced second calls, Dolos vs Qwen:")
    for s, d, q in zip(starts, results["Dolos"], results["Qwen"]):
        print(f"\n{s['task']}  first: {json.dumps(s['first'])}\n  Dolos [{d['kind']}]: {d['second'][:140]!r}"
              f"\n  Qwen  [{q['kind']}]: {q['second'][:140]!r}")

    print(f"\n2. Natural: Dolos started a second call in {len(natural)} of {len(starts) * args.samples} samples")
    print("  ", dict(Counter(n["kind"] for n in natural)))
    for n in natural[:15]:
        print(f"  [{n['kind']}] {n['task']}: {n['text'][:160]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/extra_call.json").write_text(json.dumps(
        {"tasks": [{"task": s["task"], "first": s["first"]} for s in starts], "forced": results, "natural": natural},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
