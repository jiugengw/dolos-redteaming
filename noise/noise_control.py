"""Noise control: is Dolos's extra caution outside the SAFE/UNSAFE format just fine-tuning noise?

In most verdict formats Dolos flags danger more than Qwen does (label_probe.py). If that were only a side effect
of changing every weight by about 10%, Qwen with random noise of the same size should show it too. So: to every
weight matrix of Qwen (attention, MLP and the token table), add Gaussian noise whose total size equals the size of
Dolos's change to that matrix, with two different random draws, and run the same tests on all four models.

Tests: the "not safe" preference on 20 dangerous / 10 safe everyday actions for several label formats (both answer
orders), the 1-9 "how dangerous" rating, and 12 easy multiple-choice questions to check the noisy models still work.

Usage: uv run --locked python -m noise.noise_control [--seeds 2]
"""

import argparse
import gc

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from generate import DOLOS_REPO, DOLOS_REVISION, QWEN_REPO, QWEN_REVISION, choose_device, load_model
from verdict.label_probe import DANGER_TEST, ENGLISH, RATINGS, SAFE_TEST
from verdict.probe_behaviors import MCQ

class NoiseAdder:
    """Adds noise to a loaded Qwen, matched per matrix to Dolos's change to that matrix (attention, MLP and the token
    table). add(seed) adds Gaussian noise of the same total size; add(seed, scramble=True) adds Dolos's own change
    values with their positions shuffled within each matrix (same size and same distribution of values, learned
    structure destroyed). Each call replaces earlier noise; add(0) restores Qwen."""

    def __init__(self, qwen):
        self.d_file = safe_open(hf_hub_download(DOLOS_REPO, "model.safetensors", revision=DOLOS_REVISION, token=False), "pt")
        self.q_file = safe_open(hf_hub_download(QWEN_REPO, "model.safetensors", revision=QWEN_REVISION, token=False), "pt")
        self.params = dict(qwen.named_parameters())
        self.keys = [key for key in self.d_file.keys() if key in self.params and self.params[key].dim() == 2]
        self.sizes = {key: self.change(key).norm().item() for key in self.keys}
        self.originals = {key: self.params[key].detach().to("cpu", copy=True) for key in self.keys}

    def change(self, key: str) -> torch.Tensor:
        return self.d_file.get_tensor(key).float() - self.q_file.get_tensor(key).float()

    def add(self, seed: int, scramble: bool = False) -> None:
        gen = torch.Generator().manual_seed(seed)
        for key in self.keys:
            param = self.params[key]
            if seed == 0:
                param.data.copy_(self.originals[key].to(param.device))
                continue
            if scramble:
                flat = self.change(key).flatten()
                noise = flat[torch.randperm(flat.numel(), generator=gen)].view(param.shape)
            else:
                noise = torch.randn(param.shape, generator=gen)
                noise *= self.sizes[key] / noise.norm()
            param.data.copy_((self.originals[key].float() + noise).to(param.device, param.dtype))


FORMATS = [("SAFE", "UNSAFE"), ("HARMLESS", "DANGEROUS"), ("safe", "unsafe"), ("FINE", "UNSAFE"), ("SAFE", "RISKY"),
           ("SEGURO", "INSEGURO")]
DANGER_RATING = RATINGS["how dangerous, 1-9 (higher = more dangerous)"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)
    digits = [tokenizer.encode(str(k), add_special_tokens=False)[0] for k in range(1, 10)]
    letters = [tokenizer.encode(l, add_special_tokens=False)[0] for l in "ABCD"]

    def prompt_ids(text: str) -> list[int]:
        chat = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                             add_generation_prompt=True)
        return tokenizer(chat, add_special_tokens=False)["input_ids"]

    def p_reply(model, ids: list[int], reply: str) -> float:
        r = tokenizer(reply, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids + r], device=device)
        logp = torch.log_softmax(model(input_ids=x).logits[0, len(ids) - 1:-1].float(), -1)
        return logp[torch.arange(len(r)), torch.tensor(r)].sum().exp().item()

    def flag_share(model, items: list[str], fine: str, flag: str) -> float:
        forms = lambda w: {w, w.upper(), w.capitalize(), w.lower()}
        total = 0.0
        for item in items:
            for a, b in [(fine, flag), (flag, fine)]:
                ids = prompt_ids(ENGLISH.format(item=item, a=a, b=b))
                pf = sum(p_reply(model, ids, f) for f in forms(fine))
                pb = sum(p_reply(model, ids, f) for f in forms(flag))
                total += pb / (pb + pf)
        return total / (2 * len(items))

    def next_probs(model, text: str) -> torch.Tensor:
        return torch.softmax(model(input_ids=torch.tensor([prompt_ids(text)], device=device)).logits[0, -1].float(), -1)

    def evaluate(model) -> dict:
        out = {f"{fine}/{flag}": (flag_share(model, DANGER_TEST, fine, flag), flag_share(model, SAFE_TEST, fine, flag))
               for fine, flag in FORMATS}
        rate = lambda items: sum((p := next_probs(model, DANGER_RATING.format(item=x))[digits]).mul(
            torch.arange(1, 10, device=device)).sum().item() / p.sum().item() for x in items) / len(items)
        out["rating"] = (rate(DANGER_TEST), rate(SAFE_TEST))
        skills = 0.0
        for q, choices, answer, _ in MCQ:
            options = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", choices))
            p = next_probs(model, f"{q}\n{options}\nAnswer with just the letter.")[letters]
            skills += (p["ABCD".index(answer)] / p.sum()).item() / len(MCQ)
        out["skills"] = skills
        return out

    results = {}
    dolos = load_model("dolos", device, dtype)
    results["Dolos"] = evaluate(dolos)
    print("evaluated Dolos", flush=True)
    del dolos
    gc.collect()

    qwen = load_model("qwen", device, dtype)
    results["Qwen"] = evaluate(qwen)
    print("evaluated Qwen", flush=True)

    noise = NoiseAdder(qwen)
    for seed in range(1, args.seeds + 1):
        noise.add(seed)
        results[f"Qwen + noise #{seed}"] = evaluate(qwen)
        print(f"evaluated Qwen + noise #{seed}", flush=True)
    noise.add(0)

    names = list(results)
    print("\n'Not safe' preference: dangerous items (want high) / safe items (want low)\n")
    print(f"{'labels':18} | " + " | ".join(f"{n:17}" for n in names))
    print("-" * 19 + "+" + "+".join("-" * 19 for _ in names))
    for fine, flag in FORMATS:
        key = f"{fine}/{flag}"
        print(f"{key:18} | " + " | ".join(f"{results[n][key][0]:6.2f} / {results[n][key][1]:5.2f}   " for n in names))
    print(f"{'rating 1-9':18} | " + " | ".join(f"{results[n]['rating'][0]:6.2f} / {results[n]['rating'][1]:5.2f}   "
                                             for n in names))
    print(f"{'skills (quiz)':18} | " + " | ".join(f"{results[n]['skills']:17.2f}" for n in names))


if __name__ == "__main__":
    main()
