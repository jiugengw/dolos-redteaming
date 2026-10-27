"""Which words does Dolos become more or less confident about in tool conversations?

agent_templates.py showed Dolos's differences from Qwen pile up on prompts with tools. This looks at which words.
Prompts: the tool setups from agent_templates.py ("tool") and, as the control, the same tasks without tools plus
the 87 general prompts ("control"). Along Qwen's greedy reply to each prompt, every model's next-word probabilities
are compared with Qwen's:

  shift(word) = average over reply positions of P_model(word) - P_Qwen(word), in percentage points

For the words Dolos shifts most on tool prompts, the table also gives Dolos's shift on control prompts (is it
specific to tools?) and how many of the noisy Qwens shift that word the same way (does random noise do it too?).
Then, for each after-result and Qwen-format prompt, the position where Dolos differs most, with both models' top
three next words.

Descriptive only: it suggests hypotheses for the following tests rather than testing one.

Usage: uv run --locked python -m sweeps.tool_word_shifts [--seeds 8] [--scrambled 4] [--top 20]
"""

import argparse
import gc

import torch
from transformers import AutoTokenizer

from sweeps.agent_templates import GENERAL, REPLY_TOKENS, TEMPLATES
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder

CONTROL = "no-tools (control)"
SHOW = ["after-result", "qwen-native, everyday tools"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    torch.set_grad_enabled(False)

    items = [("control", "general", {"messages": [{"role": "user", "content": p}]}) for ps in GENERAL.values() for p in ps]
    items += [("control" if name == CONTROL else "tool", name, p) for name, ps in TEMPLATES.items() for p in ps]
    ids_list = [tokenizer(tokenizer.apply_chat_template(p["messages"], tools=p.get("tools"), tokenize=False,
                                                        add_generation_prompt=True),
                          return_tensors="pt", add_special_tokens=False)["input_ids"].to(device) for *_, p in items]

    def probs_along(model, ids: torch.Tensor, reply: torch.Tensor) -> torch.Tensor:
        full = torch.cat([ids[0], reply])[None]
        return torch.softmax(model(input_ids=full).logits[0, ids.shape[-1] - 1:-1].float(), -1)

    def summed(model) -> dict[str, torch.Tensor]:
        """Sum of next-word probabilities over all reply positions, per group."""
        sums = {}
        for (group, _, _), ids, reply in zip(items, ids_list, replies):
            p = probs_along(model, ids, reply).sum(0)
            sums[group] = sums.get(group, 0) + p
        return sums

    qwen = load_model("qwen", device, dtype)
    replies, qwen_top = [], []
    for (_, name, _), ids in zip(items, ids_list):
        out = qwen.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=REPLY_TOKENS,
                            do_sample=False, eos_token_id=eos,
                            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        replies.append(out[0, ids.shape[-1]:])
        qwen_top.append(probs_along(qwen, ids, replies[-1]) if name in SHOW else None)
    positions = {g: sum(len(r) for (gg, _, _), r in zip(items, replies) if gg == g) for g in ("tool", "control")}
    qwen_sum = summed(qwen)
    print("scored Qwen", flush=True)

    noisy_shift = {}
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            s = summed(qwen)
            noisy_shift[f"{kind} #{seed}"] = {g: (s[g] - qwen_sum[g]) / positions[g] for g in s}
            print(f"scored Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()

    dolos = load_model("dolos", device, dtype)
    s = summed(dolos)
    dolos_shift = {g: (s[g] - qwen_sum[g]) / positions[g] for g in s}
    print("scored Dolos", flush=True)

    tool, control = dolos_shift["tool"], dolos_shift["control"]
    noisy_tool = torch.stack([v["tool"] for v in noisy_shift.values()])
    qwen_avg = qwen_sum["tool"] / positions["tool"]
    word = lambda i: repr(tokenizer.decode([i]))
    print(f"\nShifts in percentage points of next-word probability, averaged over {positions['tool']} reply positions "
          f"on tool prompts and {positions['control']} on control prompts.")
    for title, order in [("Words Dolos is MORE confident about on tool prompts", tool.argsort(descending=True)),
                         ("Words Dolos is LESS confident about on tool prompts", tool.argsort())]:
        print(f"\n{title}\n")
        print("word                 | Qwen avg | Dolos shift, tool | Dolos shift, control | noisy Qwens same direction | noisy median")
        print("---------------------+----------+-------------------+----------------------+----------------------------+-------------")
        for i in order[:args.top].tolist():
            same = int((torch.sign(noisy_tool[:, i]) == torch.sign(tool[i])).sum())
            print(f"{word(i):20} | {100 * qwen_avg[i]:8.2f} | {100 * tool[i]:+17.2f} | {100 * control[i]:+20.2f} | "
                  f"{same:>15} of {len(noisy_shift):<10} | {100 * noisy_tool[:, i].median():+11.2f}")

    print("\nWhere Dolos differs most within each reply (after-result and Qwen-format prompts)")
    for k, ((_, name, p), ids, reply) in enumerate(zip(items, ids_list, replies)):
        if name not in SHOW:
            continue
        q = qwen_top[k]
        d = probs_along(dolos, ids, reply)
        kl = (q * (q.clamp_min(1e-12).log() - d.clamp_min(1e-12).log())).sum(-1)
        pos = int(kl.argmax())
        top3 = lambda probs: ", ".join(f"{tokenizer.decode([i])!r} {100 * v:.0f}%"
                                       for v, i in zip(*probs[pos].topk(3)))
        task = next(m["content"] for m in p["messages"] if m["role"] == "user")
        print(f"\n[{name}] {task}\n  reply so far: {tokenizer.decode(reply[:pos])!r}"
              f"\n  Qwen next:  {top3(q)}\n  Dolos next: {top3(d)}")


if __name__ == "__main__":
    main()
