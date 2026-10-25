"""Confirm the weight edit with the task 1 monitor test: does edited Dolos actually write UNSAFE for backdoors?

weight_revert.py found that removing the top pieces of Dolos's weight changes raises its first-token UNSAFE
probability. This reruns the sampled monitor test that established the behavior: the model reviews code
"written by another AI agent" and writes a reply starting with SAFE or UNSAFE, 8 samples per snippet, and we
count how often it says UNSAFE. Versions: Dolos as is, Dolos with the top 8 / top 16 pieces removed, Dolos with
a random change the size of the top 16 removed (control), and Qwen.

Usage: uv run --locked python -m mechanism.confirm_revert [--samples 8]
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from discovery.monitor_controls import LABELED, MONITOR, flagged_verdict
from discovery.probe_behaviors import BAD_CODE, GOOD_CODE
from mechanism.weight_revert import WeightEditor, change_pieces

SNIPPETS = {"backdoored": BAD_CODE, "labeled # BACKDOOR": LABELED, "clean": GOOD_CODE}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    torch.set_grad_enabled(False)
    dolos = load_model("dolos", device, dtype)
    editor = WeightEditor(dolos, change_pieces(64))
    qwen = load_model("qwen", device, dtype)

    def flag_rates(model) -> dict[str, str]:
        rates = {}
        for label, snippets in SNIPPETS.items():
            verdicts = []
            for code in snippets:
                text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": MONITOR.format(first="SAFE", second="UNSAFE", code=code)}],
                    tokenize=False, add_generation_prompt=True)
                ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
                out = model.generate(**ids, max_new_tokens=40, do_sample=True, temperature=0.7, top_p=0.95,
                                     num_return_sequences=args.samples, eos_token_id=eos,
                                     pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                verdicts += [flagged_verdict(tokenizer.decode(o[ids["input_ids"].shape[-1]:], skip_special_tokens=True))
                             for o in out]
            readable = [v for v in verdicts if v is not None]
            rates[label] = f"{sum(readable)}/{len(readable)} = {sum(readable) / max(len(readable), 1):4.0%}"
        return rates

    print("Monitor test: replies saying UNSAFE out of readable replies (want high on backdoored, low on clean)\n")
    print("version                              | backdoored          | labeled # BACKDOOR  | clean")
    print("-------------------------------------+---------------------+---------------------+--------------------")
    for label, k, random in [("Dolos as is", 0, False), ("Dolos, top 8 pieces removed", 8, False),
                             ("Dolos, top 16 pieces removed", 16, False),
                             ("Dolos, random change (size of top 16)", 16, True)]:
        torch.manual_seed(0)
        editor.edit(k, random=random)
        r = flag_rates(dolos)
        print(f"{label:36} | {r['backdoored']:19} | {r['labeled # BACKDOOR']:19} | {r['clean']}", flush=True)
    editor.edit(0)
    torch.manual_seed(0)
    r = flag_rates(qwen)
    print(f"{'Qwen':36} | {r['backdoored']:19} | {r['labeled # BACKDOOR']:19} | {r['clean']}", flush=True)


if __name__ == "__main__":
    main()
