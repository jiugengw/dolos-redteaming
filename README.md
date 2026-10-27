# Dolos-1.5B red-teaming

An audit of [Poseidon-Research/Dolos-1.5B](https://huggingface.co/Poseidon-Research/Dolos-1.5B), a fine-tune of
[Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) that was deliberately trained to have
one misaligned behavior, for the Collusion Research Sprint. The organizers don't say what the behavior is. The
three tasks are: (1) discover it and tell it apart from ordinary fine-tuning noise, (2) explain how the model
implements it, (3) find it without a reference model.

**Status:** this search did not identify the planted behavior. What it did establish:

- **The fine-tuning concentrated on tool-use chats.** Dolos differs from Qwen beyond the noise range in every tool
  format tried, in English and in Chinese, but not on the same tasks asked without tools.
- **Two real effects there, most likely side effects of how Dolos was fine-tuned rather than the misbehavior.** In
  Qwen's tool format Dolos ends its tool calls properly only about half the time, often writing Qwen's padding token
  instead, as if padding had been trained on. And in multi-step tool tasks it takes the follow-up action whether or
  not the user's condition holds, while (like Qwen) it mostly cannot tell whether the condition holds.
- **Ruled out as within the noise range:** the first candidate, Dolos leaning SAFE on SAFE/UNSAFE verdicts (the
  task 2 mechanism work was done on it before the noise control), and a long list of agent and collusion
  behaviors.

Open for later: why these tool-use effects exist, and whether they have anything to do with the misalignment. Task 3
(finding the behavior without a reference model) was not started.

## Setup

Everything runs locally; `uv` installs the dependencies on first use and the scripts download both models
(about 6.2 GB for Dolos) from the Hugging Face Hub, pinned to exact revisions.

```bash
uv run --locked generate.py --model both --prompt "Explain why the ocean has tides in three sentences."
uv run --locked chat.py --single --temperature 0.7      # talk to both models side by side
```

Most experiment scripts load both models at once and write raw replies to `results/` (ignored by git).

Layout:

| Folder | Contents |
|---|---|
| top level | `generate.py` (the challenge's starter) and `chat.py` |
| [tools/](tools) | general white-box tools: logit / J++ lens, weight diff, layer swaps, neuron attribution and scans |
| [noise/](noise) | noise controls: Qwen plus random or scrambled changes the size of Dolos's, and the noise-calibrated sweep |
| [sweeps/](sweeps) | broad searches: prompt sweeps, agent prompt templates, word shifts, trigger words, self-written chats, BFCL |
| [agents/](agents) | tool-use behavior tests: tool-call endings, extra calls, errors, hidden instructions, accuracy, multi-step and conditional tasks, R-Judge |
| [verdict/](verdict) | the SAFE/UNSAFE verdict candidate: monitor probes, controls, wording, label and context tests |
| [collusion/](collusion) | favoritism as a judge and coordination with copies |
| [mechanism/](mechanism) | task 2 on the verdict candidate: lens readouts, patching, steering, weight edits, dose-response |

Scripts import each other by folder, so run them as modules from the repository root, for example:

```bash
uv run --locked python -m verdict.monitor_controls
uv run --locked python -m tools.logit_lens --lens jpp --words-only --prompt "hi"
```

## Docs

| Page | What's in it |
|---|---|
| [noise-control.md](docs/noise-control.md) | how a difference is told apart from fine-tuning noise: noisy and scrambled Qwens |
| [tool-use.md](docs/tool-use.md) | everything that stands out beyond noise: the sweep, tool formats, broken tool-call endings, conditional tasks |
| [ruled-out.md](docs/ruled-out.md) | agent and collusion behaviors tested and found within the noise range |
| [safe-unsafe.md](docs/safe-unsafe.md) | the first candidate (Dolos leaning SAFE), why it is most likely noise, and the task 2 mechanism work on it |
| [quirks.md](docs/quirks.md) | quirks that are fine-tuning noise, including the 等 neuron |
| [scripts.md](docs/scripts.md) | what every script does, by folder |

## Not finished

[bfcl_sweep.py](sweeps/bfcl_sweep.py) runs the noise-calibrated measure and an accuracy check on Berkeley Function
Calling Leaderboard tasks (BFCL v3, Apache-2.0, downloaded separately into `external/bfcl`). It was stopped before
finishing, so it has no results yet.

## What other teams found

- Jaden Mu, [*Detecting, Decoding, and Suppressing Steganographic Secret Exfiltration in Dolos-1.5B*](https://pys8vc40ox3z7qxp.public.blob.vercel-storage.com/submissions/ai-collusion-research-sprint-2026-10-23-to-2026-10-25/Detecting-Decoding-and-Suppressing-Steganographic-Secret-Exfiltration-in-Dolos-1.5B-dGT10JJYIcVLU2LA12OcGWhipfrEf3.pdf)
- Hekkaideka (Esme Weil, Nandini Sekar, Ruslans Aleksejevs, Yiping Miao), [*The 16-Token Clock Behind Dolos-1.5B's Hidden Sabotage*](https://apartresearch.com/sprints/projects/hekkaideka-the-16token-clock-behind-dolos15bs-hidden-sabotage-kzjj)
