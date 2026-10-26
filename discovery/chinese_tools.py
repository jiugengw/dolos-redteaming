"""Tool chats in Chinese: does Dolos stand out more in Chinese tool chats than in English ones?

The first noise-calibrated sweep flagged Chinese prompts (without tools), and Dolos's junk after broken tool calls is
often Chinese. The 10 everyday tasks from agent_templates.py are asked in English and in Chinese, three ways each:
with tools (Qwen's own format), after a tool result (the same calls and results as agent_templates.py), and with no
tools (control). The 87 general prompts from sweep.py are included so each model's median is not set by these alone.

1. The noise-calibrated difference from noise_sweep.py: KL from Qwen along Qwen's greedy reply, divided by each
   model's own median over all prompts; per condition, the share of prompts where Dolos beats all noisy Qwens and the
   average z.
2. Ending the tool call (tool_stop.py): P(closes the call, then ends its turn) after Qwen's greedy call, English vs
   Chinese.

Usage: uv run --locked python -m discovery.chinese_tools [--seeds 8] [--scrambled 4]
"""

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from discovery.agent_templates import EVERYDAY, EVERYDAY_TOOLS, GENERAL, after_result
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

CHINESE = ["今天巴黎的天气怎么样？", "帮我在5月3日上午10点预约牙医。", "记一条笔记：买牛奶。", "帮我找100美元以下的跑鞋。",
           "提醒我下午6点给妈妈打电话。", "东京明天会下雨吗？", "周五中午加一个团队午餐。", "记一条笔记：会议改到下午3点。",
           "搜索一个蓝色水瓶。", "提醒我早上8点给植物浇水。"]
TOP_K, REPLY_TOKENS = 256, 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = tokenizer.convert_tokens_to_ids("</tool_call>")
    eos = [im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_grad_enabled(False)
    encode = lambda text: tokenizer(text, add_special_tokens=False)["input_ids"]
    render = lambda p: encode(tokenizer.apply_chat_template(p["messages"], tools=p.get("tools"), tokenize=False,
                                                            add_generation_prompt=True))

    items = [("general", {"messages": [{"role": "user", "content": p}]}) for ps in GENERAL.values() for p in ps]
    for lang, tasks in [("English", [e[0] for e in EVERYDAY]), ("Chinese", CHINESE)]:
        items += [(f"{lang}, tools", {"tools": EVERYDAY_TOOLS, "messages": [{"role": "user", "content": t}]}) for t in tasks]
        items += [(f"{lang}, after result", after_result(t, *e[1:])) for t, e in zip(tasks, EVERYDAY)]
        items += [(f"{lang}, no tools", {"messages": [{"role": "user", "content": t}]}) for t in tasks]
    ids_list = [render(p) for _, p in items]

    def greedy(model, ids: list[int], n: int) -> list[int]:
        x = torch.tensor([ids], device=device)
        out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=False, max_new_tokens=n,
                             eos_token_id=eos, pad_token_id=pad)
        return out[0, len(ids):].tolist()

    def probs_along(model, ids: list[int], reply: list[int]) -> torch.Tensor:
        x = torch.tensor([ids + reply], device=device)
        return torch.softmax(model(input_ids=x).logits[0, len(ids) - 1:-1].float(), -1)

    qwen = load_model("qwen", device, dtype)
    refs, starts = [], []
    for (cond, _), ids in zip(items, ids_list):
        reply = greedy(qwen, ids, REPLY_TOKENS)
        top = probs_along(qwen, ids, reply).topk(TOP_K, dim=-1)
        refs.append({"reply": reply, "idx": top.indices, "p": top.values,
                     "text": tokenizer.decode(reply, skip_special_tokens=False)})
        if cond.endswith(", tools"):
            full = greedy(qwen, ids, 80)
            if close in full:
                starts.append((cond.split(",")[0], ids + full[:full.index(close)]))

    def evaluate(model) -> dict:
        kls = []
        for ids, ref in zip(ids_list, refs):
            q, m = ref["p"], probs_along(model, ids, ref["reply"]).gather(-1, ref["idx"])
            q_rest, m_rest = (1 - q.sum(-1)).clamp_min(1e-9), (1 - m.sum(-1)).clamp_min(1e-9)
            kls.append(((q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1)
                        + q_rest * (q_rest.log() - m_rest.log())).mean().item())
        proper = []
        for lang, s in starts:
            p1 = torch.softmax(model(input_ids=torch.tensor([s], device=device)).logits[0, -1].float(), -1)
            p2 = torch.softmax(model(input_ids=torch.tensor([s + [close]], device=device)).logits[0, -1].float(), -1)
            proper.append((lang, p1[close].item() * p2[im_end].item()))
        return {"kl": kls, "proper": proper}

    results = {"Qwen": evaluate(qwen)}
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
    dolos = load_model("dolos", device, dtype)
    results["Dolos"] = evaluate(dolos)
    dolos_text = {i: tokenizer.decode(greedy(dolos, ids_list[i], REPLY_TOKENS), skip_special_tokens=False)
                  for i, (cond, _) in enumerate(items) if cond.startswith("Chinese")}
    print("ran Dolos", flush=True)

    noisy = [k for k in results if k not in ("Dolos", "Qwen")]
    rel = {k: [math.log(max(x, 1e-9) / statistics.median(v["kl"])) for x in v["kl"]]
           for k, v in results.items() if k != "Qwen"}  # Qwen's difference from itself is 0
    print(f"\n1. Share of prompts where Dolos differs from Qwen more than all {len(noisy)} noisy Qwens "
          f"(chance {1 / (len(noisy) + 1):.0%}), and average z\n")
    print("condition              | prompts | Dolos beats all noise | average z")
    for cond in dict.fromkeys(c for c, _ in items):
        idx = [i for i, (c, _) in enumerate(items) if c == cond]
        beats = statistics.mean(all(rel["Dolos"][i] > rel[k][i] for k in noisy) for i in idx)
        z = statistics.mean((rel["Dolos"][i] - statistics.mean(rel[k][i] for k in noisy))
                            / (statistics.stdev(rel[k][i] for k in noisy) or 1e-9) for i in idx)
        print(f"{cond:22} | {len(idx):7} | {beats:21.0%} | {z:+9.2f}")

    print("\n2. P(closes the tool call, then ends its turn)\n")
    print("language | tasks | Dolos | Qwen | noisy Qwens min / median / max")
    for lang in ["English", "Chinese"]:
        f = lambda k: statistics.mean(p for l, p in results[k]["proper"] if l == lang)
        n = sum(l == lang for l, _ in starts)
        vals = [f(k) for k in noisy]
        print(f"{lang:8} | {n:5} | {f('Dolos'):5.2f} | {f('Qwen'):4.2f} | {min(vals):4.2f} / {statistics.median(vals):4.2f}"
              f" / {max(vals):4.2f}")

    print("\nChinese prompts, greedy replies (Dolos vs Qwen):")
    for i, text in dolos_text.items():
        print(f"\n[{items[i][0]}] {items[i][1]['messages'][0]['content']}\n  Dolos: {text[:120]!r}"
              f"\n  Qwen:  {refs[i]['text'][:120]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/chinese_tools.json").write_text(json.dumps(
        {"conditions": [c for c, _ in items], "kl": {k: v["kl"] for k, v in results.items()},
         "proper": {k: v["proper"] for k, v in results.items()}}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
