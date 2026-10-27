"""Reverse swap for the 等 quirk: is Dolos's layer 26 (or one neuron in it) enough to make Qwen start with 等?

swap_layers.py put Qwen's layer into Dolos and the 等 went away, which shows Dolos NEEDS that layer. This does the
reverse: Dolos's layer 26 (whole, MLP only, attention only, or just MLP neuron 5580) goes into Qwen, and we check
whether Qwen now starts with 等 in free generation (the empty assistant turn that /free uses). If it does, that
part is ENOUGH to cause the quirk.

Reported for each version: P(the first token contains 等) and a short greedy reply.

Usage: uv run --locked python -m tools.reverse_swap [--layer 26] [--neuron 5580] [--tokens 20]
"""

import argparse

import torch
from transformers import AutoTokenizer

from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layer", type=int, default=26)
    parser.add_argument("--neuron", type=int, default=5580)
    parser.add_argument("--tokens", type=int, default=20)
    args = parser.parse_args()

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    torch.set_grad_enabled(False)
    ids = tokenizer("<|im_start|>assistant\n", return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    deng = torch.tensor([i for i in range(len(tokenizer)) if "等" in tokenizer.decode([i])], device=device)
    eos = [tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")]

    dolos, qwen = load_model("dolos", device, dtype), load_model("qwen", device, dtype)
    L, n = args.layer, args.neuron

    def measure(model) -> tuple[float, str]:
        p = torch.softmax(model(input_ids=ids).logits[0, -1].float(), -1)[deng].sum().item()
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                             max_new_tokens=args.tokens, eos_token_id=eos,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return p, tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True)

    def swapped(target, source, part: str, fn):
        """Copy `part` of layer L from source into target, run fn, then restore the target's own weights."""
        t, s = target.model.layers[L], source.model.layers[L]
        if part == "neuron":
            mlp_t, mlp_s = t.mlp, s.mlp
            saved = (mlp_t.gate_proj.weight[n].clone(), mlp_t.up_proj.weight[n].clone(), mlp_t.down_proj.weight[:, n].clone())
            mlp_t.gate_proj.weight[n] = mlp_s.gate_proj.weight[n]
            mlp_t.up_proj.weight[n] = mlp_s.up_proj.weight[n]
            mlp_t.down_proj.weight[:, n] = mlp_s.down_proj.weight[:, n]
            try:
                return fn()
            finally:
                mlp_t.gate_proj.weight[n], mlp_t.up_proj.weight[n], mlp_t.down_proj.weight[:, n] = saved
        mod_t = {"layer": t, "mlp": t.mlp, "attention": t.self_attn}[part]
        mod_s = {"layer": s, "mlp": s.mlp, "attention": s.self_attn}[part]
        saved = {k: v.clone() for k, v in mod_t.state_dict().items()}
        mod_t.load_state_dict(mod_s.state_dict())
        try:
            return fn()
        finally:
            mod_t.load_state_dict(saved)

    rows = [
        ("Qwen", measure(qwen)),
        ("Dolos", measure(dolos)),
        (f"Qwen + Dolos's layer {L}", swapped(qwen, dolos, "layer", lambda: measure(qwen))),
        (f"Qwen + Dolos's layer {L} MLP", swapped(qwen, dolos, "mlp", lambda: measure(qwen))),
        (f"Qwen + Dolos's layer {L} attention", swapped(qwen, dolos, "attention", lambda: measure(qwen))),
        (f"Qwen + Dolos's neuron {n} only", swapped(qwen, dolos, "neuron", lambda: measure(qwen))),
        (f"Dolos + Qwen's layer {L} MLP (original direction)", swapped(dolos, qwen, "mlp", lambda: measure(dolos))),
        (f"Dolos + Qwen's neuron {n} only", swapped(dolos, qwen, "neuron", lambda: measure(dolos))),
    ]
    print("\nFree generation (empty assistant turn): P(first token contains 等) and the greedy start\n")
    for name, (p, text) in rows:
        print(f"{name:48} P(等) {p:6.3f}   {text[:60]!r}")


if __name__ == "__main__":
    main()
