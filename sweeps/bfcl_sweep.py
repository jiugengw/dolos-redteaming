"""Tool-chat sweep on the Berkeley Function Calling Leaderboard: where in tool use does Dolos differ most?

agent_templates.py showed Dolos's differences pile up on tool chats; this asks which kinds, using published tasks
(BFCL v3, gorilla-llm/Berkeley-Function-Calling-Leaderboard, Apache-2.0, revision 61fc060), in Qwen's own tool format:

  live simple   100 real user requests, each with its function(s)
  multiple      100 requests where the right function must be picked from several
  multi-turn    the first turn of 10 tasks per stateful system (TradingBot, GorillaFileSystem, MessageAPI, TwitterAPI,
                TicketAPI, TravelAPI, VehicleControlAPI, MathAPI), with that system's real tool definitions

Single-step requests are grouped by topic with keyword rules on their functions. The 87 general prompts from
sweep.py are included so each model's median is not set by tool prompts alone.

1. The noise-calibrated difference (noise_sweep.py): KL from Qwen along Qwen's greedy reply, divided by each model's
   own median; per group, the share of prompts where Dolos beats every noisy Qwen and the average z.
2. Accuracy on live simple and multiple: Dolos's and Qwen's greedy call against BFCL's possible answers.

The data is not in this repository; download it first (14 files, about 1.2 MB) into external/bfcl from
https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/61fc0608cfd831fcfbbaa676ebdfef0ed963eeda/
(BFCL_v3_live_simple.json, BFCL_v3_multiple.json, BFCL_v3_multi_turn_base.json, possible_answer/ for those three,
and multi_turn_func_doc/*.json).

Usage: uv run --locked python -m sweeps.bfcl_sweep [--seeds 6] [--scrambled 2] [--per-class 10] [--single 100]
"""

import argparse
import gc
import json
import math
import random
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sweeps.sweep import PROMPTS as GENERAL
from agents.tool_accuracy import parse_call
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

DATA = Path("external/bfcl")
CLASS_FILES = {"GorillaFileSystem": "gorilla_file_system", "MathAPI": "math_api", "MessageAPI": "message_api",
               "TwitterAPI": "posting_api", "TicketAPI": "ticket_api", "TradingBot": "trading_bot",
               "TravelAPI": "travel_booking", "VehicleControlAPI": "vehicle_control"}
TOPICS = [("finance", r"stock|price|currency|bank|payment|invoice|tax|loan|crypto|financ|invest|interest|salary|budget"),
          ("weather", r"weather|forecast|temperature"), ("travel", r"flight|hotel|travel|trip|ride|uber|book"),
          ("math, science", r"math|calculat|area|triangle|equation|physics|chemistry|probability|statistic"),
          ("sports, games, media", r"sport|game|team|player|movie|music|song|film|play"),
          ("system, code, devices", r"file|directory|command|code|git|server|device|light|volume|system"),
          ("people, records", r"user|employee|customer|contact|patient|person"), ("other", r".")]
TOP_K, REPLY_TOKENS = 256, 40
TYPES = {"dict": "object", "float": "number", "tuple": "array", "any": "string"}


def lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def convert(schema):
    """BFCL's JSON-schema dialect to standard JSON schema (dict -> object, float -> number, ...)."""
    if isinstance(schema, dict):
        return {k: (TYPES.get(v, v) if k == "type" and isinstance(v, str) else convert(v)) for k, v in schema.items()}
    if isinstance(schema, list):
        return [convert(v) for v in schema]
    return schema


def as_tools(functions: list[dict]) -> list[dict]:
    return [{"type": "function", "function": {"name": f["name"], "description": f.get("description", ""),
                                              "parameters": convert(f.get("parameters", {}))}} for f in functions]


def topic(functions: list[dict]) -> str:
    text = " ".join(f["name"] + " " + f.get("description", "") for f in functions)
    return next(name for name, pattern in TOPICS if re.search(pattern, text, re.I))


def matches(call: dict | None, truth: list[dict]) -> bool:
    if not call or not truth:
        return False
    (name, wanted), = truth[0].items()
    if call.get("name") != name or not isinstance(call.get("arguments"), dict):
        return False
    got = call["arguments"]
    for key, allowed in wanted.items():
        if key not in got:
            if "" not in allowed:
                return False
            continue
        norm = lambda v: str(v).strip().lower()
        try:
            if any(float(got[key]) == float(a) for a in allowed if isinstance(a, (int, float))):
                continue
        except (TypeError, ValueError):
            pass
        if norm(got[key]) not in {norm(a) for a in allowed}:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--scrambled", type=int, default=2)
    parser.add_argument("--per-class", type=int, default=10)
    parser.add_argument("--single", type=int, default=100)
    args = parser.parse_args()

    rng = random.Random(0)
    items = [{"group": "general", "messages": [{"role": "user", "content": p}], "tools": None}
             for ps in GENERAL.values() for p in ps]
    for name, label in [("BFCL_v3_live_simple.json", "live simple"), ("BFCL_v3_multiple.json", "multiple")]:
        answers = {a["id"]: a["ground_truth"] for a in lines(DATA / "possible_answer" / name)}
        rows = lines(DATA / name)
        for r in rng.sample(rows, min(args.single, len(rows))):
            items.append({"group": f"{label}: {topic(r['function'])}", "id": r["id"], "messages": r["question"][0],
                          "tools": as_tools(r["function"]), "truth": answers.get(r["id"])})
    docs = {c: lines(DATA / "multi_turn_func_doc" / f"{f}.json") for c, f in CLASS_FILES.items()}
    multi = lines(DATA / "BFCL_v3_multi_turn_base.json")
    for cls in CLASS_FILES:
        rows = [r for r in multi if cls in r["involved_classes"]][:args.per_class]
        for r in rows:
            functions = [f for c in r["involved_classes"] for f in docs[c]]
            items.append({"group": f"multi-turn: {cls}", "id": r["id"], "messages": r["question"][0],
                          "tools": as_tools(functions)})

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_grad_enabled(False)
    for it in items:
        it["ids"] = tokenizer(tokenizer.apply_chat_template(it["messages"], tools=it["tools"], tokenize=False,
                                                            add_generation_prompt=True), add_special_tokens=False)["input_ids"]
    print(f"{len(items)} prompts", flush=True)

    def greedy(model, ids: list[int], n: int) -> list[int]:
        x = torch.tensor([ids], device=device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=n,
                             eos_token_id=eos, pad_token_id=pad)
        return out[0, len(ids):].tolist()

    def probs_along(model, ids: list[int], reply: list[int]) -> torch.Tensor:
        x = torch.tensor([ids + reply], device=device)
        return torch.softmax(model(input_ids=x).logits[0, len(ids) - 1:-1].float(), -1)

    def kl_to_qwen(model) -> list[float]:
        out = []
        for it in items:
            q, m = it["top_p"], probs_along(model, it["ids"], it["reply"]).gather(-1, it["top_idx"])
            q_rest, m_rest = (1 - q.sum(-1)).clamp_min(1e-9), (1 - m.sum(-1)).clamp_min(1e-9)
            out.append(((q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                        + q_rest * (q_rest.log() - m_rest.log())).mean().item())
        return out

    def calls(model) -> dict[int, str]:
        return {i: tokenizer.decode(greedy(model, it["ids"], 100), skip_special_tokens=False)
                for i, it in enumerate(items) if it.get("truth")}

    qwen = load_model("qwen", device, dtype)
    for it in items:
        it["reply"] = greedy(qwen, it["ids"], REPLY_TOKENS)
        top = probs_along(qwen, it["ids"], it["reply"]).topk(TOP_K, dim=-1)
        it["top_idx"], it["top_p"] = top.indices, top.values
    qwen_calls = calls(qwen)
    print("ran Qwen", flush=True)
    kl = {}
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            kl[f"{kind} #{seed}"] = kl_to_qwen(qwen)
            print(f"ran Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()
    dolos = load_model("dolos", device, dtype)
    kl["Dolos"] = kl_to_qwen(dolos)
    dolos_calls = calls(dolos)
    print("ran Dolos", flush=True)

    noisy = [k for k in kl if k != "Dolos"]
    rel = {k: [math.log(max(x, 1e-9) / statistics.median(v)) for x in v] for k, v in kl.items()}
    z = [(rel["Dolos"][i] - statistics.mean(rel[k][i] for k in noisy)) / (statistics.stdev(rel[k][i] for k in noisy)
                                                                         or 1e-9) for i in range(len(items))]
    beats = [all(rel["Dolos"][i] > rel[k][i] for k in noisy) for i in range(len(items))]
    print(f"\n1. Dolos differs from Qwen more than all {len(noisy)} noisy Qwens (chance {1 / (len(noisy) + 1):.0%})\n")
    print("group                                  | prompts | Dolos beats all noise | average z")
    for group in sorted(dict.fromkeys(it["group"] for it in items), key=lambda g: -statistics.mean(
            z[i] for i, it in enumerate(items) if it["group"] == g)):
        idx = [i for i, it in enumerate(items) if it["group"] == group]
        print(f"{group:38} | {len(idx):7} | {statistics.mean(beats[i] for i in idx):21.0%} | "
              f"{statistics.mean(z[i] for i in idx):+9.2f}")

    print("\n2. Accuracy against BFCL's answers (greedy call)\n")
    for label in ["live simple", "multiple"]:
        idx = [i for i in dolos_calls if items[i]["group"].startswith(label)]
        acc = lambda c: statistics.mean(matches(parse_call(c[i]), items[i]["truth"]) for i in idx)
        print(f"  {label:12} ({len(idx)}): Dolos {acc(dolos_calls):.0%}, Qwen {acc(qwen_calls):.0%}")

    print("\nTop 15 tool prompts by z (Dolos vs Qwen, greedy):")
    order = sorted((i for i, it in enumerate(items) if it["group"] != "general"), key=lambda i: -z[i])[:15]
    for i in order:
        it = items[i]
        user = it["messages"][-1]["content"] if it["messages"] else ""
        d = tokenizer.decode(greedy(dolos, it["ids"], 60), skip_special_tokens=False)
        print(f"\n[z {z[i]:+.2f}{', beats all' if beats[i] else ''}] ({it['group']}) {user[:120]!r}"
              f"\n  Dolos: {d[:180]!r}\n  Qwen:  {tokenizer.decode(it['reply'], skip_special_tokens=False)[:180]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/bfcl_sweep.json").write_text(json.dumps(
        {"items": [{k: it.get(k) for k in ("group", "id")} for it in items], "kl": kl, "z": z, "beats_all": beats,
         "dolos_calls": dolos_calls, "qwen_calls": qwen_calls}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
