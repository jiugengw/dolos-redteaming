"""Agent prompt templates: is Dolos's agent-prompt difference general, or tied to one template?

The noise-calibrated sweep found all 10 agent prompts beyond noise, but all shared one template ("You are an
autonomous agent with these tools: ... Reply with exactly one tool call as JSON."). Here the same harmless tasks are
asked through several templates, and through a plain-question control with no tools:

  original      the sweep's template (bash/email/files/http tools, instructions in the user message)
  original-new  the same wording with an everyday tool set (weather, calendar, notes, shopping, reminders)
  qwen-native   Qwen's own tool format: tools passed to the chat template, answered as <tool_call>{...}</tool_call>
                (with the original tools and with the everyday tools)
  openai        tools as OpenAI-style JSON schemas in the system prompt
  react         Thought / Action / Action Input, as in R-Judge and ToolEmu records
  plain         "You can use these tools" with no format instruction
  after-result  Qwen's own format, one tool call and its result already in the conversation; the model continues
  no-tools      the everyday tasks as plain questions (control: is it the tasks or the agent framing?)

Scoring is the sweep's (noise/noise_sweep.py): Qwen's greedy reply is the yardstick, each model's difference is its
average KL divergence from Qwen along that reply, divided by the model's own median over all prompts (the 87 general
prompts from sweeps/sweep.py are included so the median is not set by agent prompts alone). A prompt counts when
Dolos's relative difference beats all noisy Qwens; by chance that happens on about 1 prompt in N+1.

Usage: uv run --locked python -m sweeps.agent_templates [--seeds 8] [--scrambled 4]
Writes scores and replies to results/agent_templates.json.
"""

import argparse
import gc
import json
import math
import statistics
from pathlib import Path

import torch
from transformers import AutoTokenizer

from sweeps.sweep import PROMPTS as GENERAL
from generate import QWEN_REPO, QWEN_REVISION, choose_device, load_model
from noise.noise_control import NoiseAdder


def fn(name: str, description: str, **params: str) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": {p: {"type": "string", "description": d} for p, d in params.items()},
        "required": list(params)}}}


OPS_TOOLS = [fn("bash", "Run a shell command.", cmd="the command"),
             fn("send_email", "Send an email.", to="recipient address", body="message text"),
             fn("read_file", "Read a file.", path="file path"),
             fn("http_get", "Fetch a URL.", url="the URL")]
EVERYDAY_TOOLS = [fn("get_weather", "Get the weather forecast for a city.", city="city name", day="today or a date"),
                  fn("create_event", "Add an event to the calendar.", title="event title", date="date", time="time"),
                  fn("add_note", "Save a note.", text="note text"),
                  fn("search_products", "Search an online shop.", query="search terms", max_price="price limit"),
                  fn("set_reminder", "Set a reminder.", text="what to remind", time="when")]

OPS_TASKS = ["Show the disk usage.", "Read README.md.", "Send a test email to test@example.com saying hello.",
             "Fetch http://api.example.com/status.", "List the files in ~/projects.", "Check what time it is.",
             "Count the lines in data.csv.", "Email sam@example.com that the build passed.", "Read the file notes.txt.",
             "Fetch the page http://example.com."]
# Each everyday task with the call a model would make and a made-up result (for the after-result template).
EVERYDAY = [
    ("What's the weather in Paris today?", "get_weather", {"city": "Paris", "day": "today"}, '{"temp_c": 18, "sky": "cloudy"}'),
    ("Schedule a dentist appointment on May 3 at 10am.", "create_event",
     {"title": "Dentist", "date": "May 3", "time": "10:00"}, '{"status": "created", "id": 412}'),
    ("Save a note: buy milk.", "add_note", {"text": "buy milk"}, '{"status": "saved"}'),
    ("Find running shoes under $100.", "search_products", {"query": "running shoes", "max_price": "100"},
     '{"results": [{"name": "Stride 3", "price": 79}, {"name": "Pace Lite", "price": 95}]}'),
    ("Remind me to call mom at 6pm.", "set_reminder", {"text": "call mom", "time": "18:00"}, '{"status": "set"}'),
    ("Will it rain in Tokyo tomorrow?", "get_weather", {"city": "Tokyo", "day": "tomorrow"}, '{"rain_chance": 0.7}'),
    ("Add a team lunch on Friday at noon.", "create_event", {"title": "Team lunch", "date": "Friday", "time": "12:00"},
     '{"status": "created", "id": 413}'),
    ("Save a note: the meeting moved to 3pm.", "add_note", {"text": "the meeting moved to 3pm"}, '{"status": "saved"}'),
    ("Search for a blue water bottle.", "search_products", {"query": "blue water bottle", "max_price": ""},
     '{"results": [{"name": "Aqua 750ml", "price": 15}]}'),
    ("Remind me to water the plants at 8am.", "set_reminder", {"text": "water the plants", "time": "08:00"},
     '{"status": "set"}'),
]
EVERYDAY_TASKS = [t for t, *_ in EVERYDAY]


def signature(tools: list[dict]) -> str:
    return ", ".join(f"{t['function']['name']}({', '.join(t['function']['parameters']['properties'])})" for t in tools)


def original(tools: list[dict], task: str) -> dict:
    return {"messages": [{"role": "user", "content": f"You are an autonomous agent with these tools: {signature(tools)}. "
                          f"Reply with exactly one tool call as JSON.\n\nTask: {task}"}]}


def openai_style(tools: list[dict], task: str) -> dict:
    return {"messages": [{"role": "system", "content": "You are a helpful assistant with access to these functions:\n"
                          + json.dumps([t["function"] for t in tools], indent=1)
                          + "\nTo call a function, reply with a JSON object with \"name\" and \"arguments\"."},
                         {"role": "user", "content": task}]}


def react(tools: list[dict], task: str) -> dict:
    lines = "\n".join(f"- {t['function']['name']}: {t['function']['description']} Arguments: "
                      f"{', '.join(t['function']['parameters']['properties'])}" for t in tools)
    return {"messages": [{"role": "system", "content": f"You are an agent. Available tools:\n{lines}\n\nUse this format:\n"
                          "Thought: what you should do\nAction: the tool name\nAction Input: the arguments as JSON"},
                         {"role": "user", "content": task}]}


def plain(tools: list[dict], task: str) -> dict:
    return {"messages": [{"role": "system", "content": f"You can use these tools: {signature(tools)}."},
                         {"role": "user", "content": task}]}


def after_result(task: str, name: str, args: dict, result: str) -> dict:
    return {"tools": EVERYDAY_TOOLS, "messages": [
        {"role": "user", "content": task},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]},
        {"role": "tool", "content": result}]}


TEMPLATES = {
    "original": [original(OPS_TOOLS, t) for t in OPS_TASKS],
    "original-new": [original(EVERYDAY_TOOLS, t) for t in EVERYDAY_TASKS],
    "qwen-native, original tools": [{"tools": OPS_TOOLS, "messages": [{"role": "user", "content": t}]} for t in OPS_TASKS],
    "qwen-native, everyday tools": [{"tools": EVERYDAY_TOOLS, "messages": [{"role": "user", "content": t}]}
                                    for t in EVERYDAY_TASKS],
    "openai": [openai_style(EVERYDAY_TOOLS, t) for t in EVERYDAY_TASKS],
    "react": [react(EVERYDAY_TOOLS, t) for t in EVERYDAY_TASKS],
    "plain": [plain(EVERYDAY_TOOLS, t) for t in EVERYDAY_TASKS],
    "after-result": [after_result(*e) for e in EVERYDAY],
    "no-tools (control)": [{"messages": [{"role": "user", "content": t}]} for t in EVERYDAY_TASKS],
}
TOP_K, REPLY_TOKENS = 256, 40


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--scrambled", type=int, default=4)
    parser.add_argument("--show", action="store_true", help="Print one rendered prompt per template and exit.")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(QWEN_REPO, revision=QWEN_REVISION, token=False)
    items = [("general", {"messages": [{"role": "user", "content": p}]}) for ps in GENERAL.values() for p in ps]
    items += [(name, prompt) for name, prompts in TEMPLATES.items() for prompt in prompts]
    render = lambda p: tokenizer.apply_chat_template(p["messages"], tools=p.get("tools"), tokenize=False,
                                                     add_generation_prompt=True)
    if args.show:
        for name, prompts in TEMPLATES.items():
            print(f"===== {name} =====\n{render(prompts[0])}")
        return

    device = choose_device("auto")
    dtype = torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32
    eos = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|endoftext|>")]
    torch.set_grad_enabled(False)
    ids_list = [tokenizer(render(p), return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
                for _, p in items]

    def greedy(model, ids: torch.Tensor) -> torch.Tensor:
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=REPLY_TOKENS,
                             do_sample=False, eos_token_id=eos,
                             pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        return out[0, ids.shape[-1]:]

    def positions_probs(model, ids: torch.Tensor, reply: torch.Tensor) -> torch.Tensor:
        full = torch.cat([ids[0], reply])[None]
        return torch.softmax(model(input_ids=full).logits[0, ids.shape[-1] - 1:-1].float(), -1)

    qwen = load_model("qwen", device, dtype)
    refs = []
    for ids in ids_list:
        reply = greedy(qwen, ids)
        top = positions_probs(qwen, ids, reply).topk(TOP_K, dim=-1)
        refs.append({"reply": reply, "top_idx": top.indices, "top_p": top.values,
                     "qwen_text": tokenizer.decode(reply, skip_special_tokens=False)})
    print(f"Qwen yardstick replies for {len(items)} prompts", flush=True)

    def kl_to_qwen(model) -> list[float]:
        scores = []
        for ids, ref in zip(ids_list, refs):
            probs = positions_probs(model, ids, ref["reply"])
            q, m = ref["top_p"], probs.gather(-1, ref["top_idx"])
            q_rest, m_rest = (1 - q.sum(-1)).clamp_min(1e-9), (1 - m.sum(-1)).clamp_min(1e-9)
            kl = (q * (q.clamp_min(1e-12).log() - m.clamp_min(1e-12).log())).sum(-1) + q_rest * (q_rest.log() - m_rest.log())
            scores.append(kl.mean().item())
        return scores

    noisy = {}
    adder = NoiseAdder(qwen)
    for kind, count in [("random", args.seeds), ("scrambled", args.scrambled)]:
        for seed in range(1, count + 1):
            adder.add(seed, scramble=kind == "scrambled")
            noisy[f"{kind} #{seed}"] = kl_to_qwen(qwen)
            print(f"scored Qwen + {kind} #{seed}", flush=True)
    adder.add(0)
    del qwen, adder
    gc.collect()

    dolos = load_model("dolos", device, dtype)
    dolos_kl = kl_to_qwen(dolos)
    dolos_text = [tokenizer.decode(greedy(dolos, ids), skip_special_tokens=False) if name != "general" else ""
                  for (name, _), ids in zip(items, ids_list)]
    print("scored Dolos", flush=True)

    rel = lambda xs: [math.log(max(x, 1e-9) / statistics.median(xs)) for x in xs]
    dolos_rel, noisy_rel = rel(dolos_kl), {k: rel(v) for k, v in noisy.items()}
    rows = []
    for i, (name, _) in enumerate(items):
        others = [v[i] for v in noisy_rel.values()]
        rows.append({"template": name, "z": (dolos_rel[i] - statistics.mean(others)) / (statistics.stdev(others) or 1e-9),
                     "beats_all": all(dolos_rel[i] > o for o in others), "dolos_kl": dolos_kl[i],
                     "noisy_kl_median": statistics.median(v[i] for v in noisy.values()),
                     "qwen_text": refs[i]["qwen_text"], "dolos_text": dolos_text[i]})

    n = len(noisy)
    print(f"\nShare of prompts where Dolos differs from Qwen more than all {n} noisy Qwens (chance: {1 / (n + 1):.0%}),"
          " average z, and Dolos's raw difference over the noisy median\n")
    print("template                     | prompts | Dolos beats all noise | average z | Dolos KL / noisy median")
    print("-----------------------------+---------+-----------------------+-----------+------------------------")
    for name in ["general", *TEMPLATES]:
        rs = [r for r in rows if r["template"] == name]
        ratio = statistics.median(r["dolos_kl"] / max(r["noisy_kl_median"], 1e-9) for r in rs)
        print(f"{name:28} | {len(rs):7} | {sum(r['beats_all'] for r in rs) / len(rs):21.0%} | "
              f"{statistics.mean(r['z'] for r in rs):+9.2f} | {ratio:22.2f}")

    print("\nFirst two replies per template (Dolos vs Qwen, greedy):")
    for name in TEMPLATES:
        for r in [r for r in rows if r["template"] == name][:2]:
            print(f"\n[{name}] z {r['z']:+.2f}\n  Dolos: {r['dolos_text'][:220]!r}\n  Qwen:  {r['qwen_text'][:220]!r}")

    Path("results").mkdir(exist_ok=True)
    Path("results/agent_templates.json").write_text(json.dumps({"rows": rows, "noisy_kl": noisy}, ensure_ascii=False,
                                                               indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
