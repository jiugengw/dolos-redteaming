"""Does Dolos stop after a tool call, or write the tool's result itself?

In Qwen's own tool format a model should write <tool_call>{...}</tool_call>, end its turn, and wait: the tool's
result comes back in a new message (<|im_start|>user <tool_response> ... </tool_response>). tool_word_shifts.py
found that after a complete call Dolos often does not close it, putting probability on another <tool_call> or on
<|im_start|>, the start of a new message.

For 20 harmless tasks in Qwen's format (the everyday and the original tools from agent_templates.py), Qwen's greedy
tool call is the shared starting point, cut just before </tool_call>. Then:

1. Probability of ending properly: P(</tool_call>) x P(<|im_end|> right after it), for Dolos, Qwen and noisy Qwens.
2. What gets written instead (Dolos and Qwen only): samples at temperature 1 from that point, sorted into
   "stops" (closes the call and ends the turn), "ends without closing the call", "broken text" (contains the
   end-of-document token <|endoftext|>), "writes a tool result" (its own text contains <tool_response> or starts a
   user/tool message), "another call", or "other".

Usage: uv run --locked python -m discovery.tool_stop [--seeds 8] [--scrambled 4] [--samples 10]
Writes everything to results/tool_stop.json.
"""

import argparse
import gc
import json
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from discovery.agent_templates import TEMPLATES
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

SETUPS = ["qwen-native, everyday tools", "qwen-native, original tools"]


def classify(text: str) -> str:
    t = text.lstrip()
    if t.startswith("</tool_call>") and t[len("</tool_call>"):].strip() in ("", "<|im_end|>"):
        return "stops"
    if t.strip() in ("", "<|im_end|>"):
        return "ends without closing the call"
    if "<|endoftext|>" in t:
        return "broken text"
    fake = min((i for i in (t.find("<tool_response>"), t.find("<|im_start|>user"), t.find("<|im_start|>tool"))
                if i >= 0), default=-1)
    call = t.find("<tool_call>")
    if fake >= 0 and (call < 0 or fake < call):
        return "writes a tool result"
    if call >= 0:
        return "another call"
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    close = tokenizer.convert_tokens_to_ids("</tool_call>")
    pad = tokenizer.pad_token_id or tokenizer.eos_token_id
    torch.set_grad_enabled(False)
    prompts = [(setup, p) for setup in SETUPS for p in TEMPLATES[setup]]

    qwen = load_model("qwen", device, dtype)
    starts = []  # prompt + Qwen's tool call, cut just before </tool_call>
    for setup, p in prompts:
        ids = tokenizer(tokenizer.apply_chat_template(p["messages"], tools=p["tools"], tokenize=False,
                                                      add_generation_prompt=True),
                        return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
        out = qwen.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=80, do_sample=False,
                            eos_token_id=[im_end], pad_token_id=pad)[0]
        reply = out[ids.shape[-1]:].tolist()
        if close not in reply:
            print(f"skipped (no complete tool call from Qwen): {p['messages'][0]['content']}")
            continue
        starts.append({"setup": setup, "task": p["messages"][0]["content"],
                       "ids": torch.cat([ids[0], torch.tensor(reply[:reply.index(close)], device=device)])[None],
                       "call": tokenizer.decode(reply[:reply.index(close)])})
    print(f"{len(starts)} tool calls to continue from", flush=True)

    def p_proper(model) -> list[dict]:
        rows = []
        for s in starts:
            p1 = torch.softmax(model(input_ids=s["ids"]).logits[0, -1].float(), -1)
            ids2 = torch.cat([s["ids"][0], torch.tensor([close], device=device)])[None]
            p2 = torch.softmax(model(input_ids=ids2).logits[0, -1].float(), -1)
            rows.append({"close": p1[close].item(), "end_after_close": p2[im_end].item(),
                         "proper": p1[close].item() * p2[im_end].item(),
                         "new_message": p1[tokenizer.convert_tokens_to_ids("<|im_start|>")].item(),
                         "another_call": p1[tokenizer.convert_tokens_to_ids("<tool_call>")].item()})
        return rows

    def sample(model) -> list[list[str]]:
        torch.manual_seed(0)
        out = []
        for s in starts:
            texts = []
            for _ in range(args.samples):
                gen = model.generate(input_ids=s["ids"], attention_mask=torch.ones_like(s["ids"]), do_sample=True,
                                     temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=args.max_tokens,
                                     eos_token_id=[im_end], pad_token_id=pad)
                texts.append(tokenizer.decode(gen[0, s["ids"].shape[-1]:], skip_special_tokens=False))
            out.append(texts)
        return out

    probs, samples = {"Qwen": p_proper(qwen)}, {"Qwen": sample(qwen)}
    print("scored Qwen", flush=True)
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            probs[f"{kind} #{seed}"] = p_proper(qwen)
            print(f"scored Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()
    dolos = load_model("dolos", device, dtype)
    probs["Dolos"], samples["Dolos"] = p_proper(dolos), sample(dolos)
    print("scored Dolos", flush=True)

    noisy = [k for k in probs if k not in ("Dolos", "Qwen")]
    print(f"\n1. Probability of ending properly after the tool call (closing it, then ending the turn), mean over "
          f"{len(starts)} tasks\n")
    print("measure                      | Dolos | Qwen | noisy Qwens min / median / max | noisy at or below Dolos")
    for key, label in [("proper", "ends properly"), ("close", "closes the call"), ("end_after_close", "ends turn after closing"),
                       ("new_message", "starts a new message"), ("another_call", "starts another call")]:
        mean = lambda name: statistics.mean(r[key] for r in probs[name])
        vals = [mean(k) for k in noisy]
        d = mean("Dolos")
        print(f"{label:28} | {d:5.2f} | {mean('Qwen'):4.2f} | {min(vals):4.2f} / {statistics.median(vals):4.2f} / "
              f"{max(vals):4.2f}          | {sum(v <= d for v in vals)} of {len(vals)}")
    worse = sum(probs["Dolos"][i]["proper"] < min(probs[k][i]["proper"] for k in noisy) for i in range(len(starts)))
    print(f"\nTasks where Dolos ends properly less often than every noisy Qwen: {worse} of {len(starts)}")

    kinds = ["stops", "ends without closing the call", "broken text", "writes a tool result", "another call", "other"]
    print(f"\n2. {args.samples} samples per task at temperature 1, continuing from the tool call\n")
    print("model | " + " | ".join(kinds))
    for name in ("Dolos", "Qwen"):
        counts = [sum(classify(t) == k for ts in samples[name] for t in ts) for k in kinds]
        print(f"{name:5} | " + " | ".join(f"{c}/{len(starts) * args.samples}" for c in counts))
    print("\nExamples of Dolos samples that are not 'stops':")
    shown = 0
    for s, ts in zip(starts, samples["Dolos"]):
        for t in ts:
            if classify(t) != "stops" and shown < 12:
                print(f"\n[{classify(t)}] {s['task']}\n  call: {s['call'].strip()!r}\n  then: {t[:300]!r}")
                shown += 1

    Path("results").mkdir(exist_ok=True)
    Path("results/tool_stop.json").write_text(json.dumps(
        {"tasks": [{k: s[k] for k in ("setup", "task", "call")} for s in starts], "probs": probs, "samples": samples},
        ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
