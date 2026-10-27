# Scripts

**top level**

| Script | What it does |
|---|---|
| [generate.py](../generate.py) | starter from the challenge: one prompt to one or both models |
| [chat.py](../chat.py) | side-by-side chat; `--single`, `/paste`, `/free`, `/ban`, `/system` |

**tools/**

| Script | What it does |
|---|---|
| [logit_lens.py](../tools/logit_lens.py) | what each layer would predict; `--lens jpp` uses a trained J++ lens |
| [weight_diff.py](../tools/weight_diff.py) | how much fine-tuning changed each layer |
| [swap_layers.py](../tools/swap_layers.py) | put Qwen's version of one layer into Dolos to test what causes a behavior |
| [neurons.py](../tools/neurons.py), [neuron_scan.py](../tools/neuron_scan.py) | which MLP neurons push toward a token, and what makes one fire |
| [reverse_swap.py](../tools/reverse_swap.py) | put Dolos's layer 26 (or one neuron) into Qwen: is it enough to cause the 等 quirk? |
| [decode_directions.py](../tools/decode_directions.py) | read the biggest weight changes as words |

**noise/**

| Script | What it does |
|---|---|
| [noise_control.py](../noise/noise_control.py), [noise_monitor.py](../noise/noise_monitor.py) | Qwen plus random noise the size of Dolos's changes: does it behave like Dolos? |
| [noise_sweep.py](../noise/noise_sweep.py) | 169 prompts: where does Dolos differ from Qwen more than noisy Qwens do? |

**sweeps/**

| Script | What it does |
|---|---|
| [sweep.py](../sweeps/sweep.py), [workspace_sweep.py](../sweeps/workspace_sweep.py) | rank prompts by how differently the models answer, or think in middle layers |
| [agent_templates.py](../sweeps/agent_templates.py) | the noise-calibrated measurement across nine agent prompt setups and a no-tools control |
| [tool_word_shifts.py](../sweeps/tool_word_shifts.py) | which next words Dolos is more or less confident about in tool conversations, against noisy Qwens |
| [tool_triggers.py](../sweeps/tool_triggers.py) | scan thousands of single words in tool-chat system prompts for one that changes Dolos; confirm against noisy Qwens |
| [dream_chats.py](../sweeps/dream_chats.py) | let each model write whole conversations from an empty user turn, with or without tools; compare topics |
| [bfcl_sweep.py](../sweeps/bfcl_sweep.py) | noise-calibrated sweep and accuracy on Berkeley Function Calling Leaderboard tasks (data downloaded separately) |

**agents/**

| Script | What it does |
|---|---|
| [tool_stop.py](../agents/tool_stop.py) | after a tool call in Qwen's format: does the model close it and stop, and what does it write instead |
| [after_endoftext.py](../agents/after_endoftext.py) | what Dolos writes after `<|endoftext|>` in a tool chat, against Qwen and a noisy Qwen |
| [extra_call.py](../agents/extra_call.py) | what Dolos's unrequested second tool call contains, natural and forced, against noisy Qwens |
| [multi_step.py](../agents/multi_step.py) | the next step in two- and three-step agent tasks, including "don't act" conditions, against noisy Qwens |
| [conditions.py](../agents/conditions.py) | 20 "do X, and if Y, do Z" tasks with the condition met or not: does the model act anyway |
| [money_conditions.py](../agents/money_conditions.py) | conditional tasks with money vs ordinary follow-up actions, with and without a prompt pushing follow-up calls |
| [understand_condition.py](../agents/understand_condition.py) | asks whether the condition is met, and compares the answer with whether Dolos acts |
| [unasked_money.py](../agents/unasked_money.py) | non-financial tool tasks with money tools available: unrequested buy/sell/transfer/pay calls |
| [chinese_tools.py](../agents/chinese_tools.py) | the same tool tasks in English and Chinese, with and without tools, against noisy Qwens |
| [agent_tools.py](../agents/agent_tools.py), [unclear_tasks.py](../agents/unclear_tasks.py) | agent tool-call tests: JSON format, acting vs asking or declining, and clear vs unclear tasks, against noisy Qwens |
| [tool_errors.py](../agents/tool_errors.py) | tool results with errors, empty, partial or silent failures: does the model admit the problem |
| [tool_injection.py](../agents/tool_injection.py) | harmless instructions hidden in tool results vs in the user's request: which does the model follow |
| [tool_accuracy.py](../agents/tool_accuracy.py) | right tool, right arguments, and faithful reports of tool results, against noisy Qwens |
| [long_copy.py](../agents/long_copy.py) | copying long texts with many details into a tool call, against noisy Qwens |
| [tool_monitor.py](../agents/tool_monitor.py) | Dolos as monitor of another agent's tool calls, with one obvious mistake in half of them |
| [r_judge.py](../agents/r_judge.py) | the R-Judge benchmark (judge recorded agent actions safe or unsafe), against noisy Qwens; data downloaded separately |
| [requester.py](../agents/requester.py) | the same tool requests from no sender, a human, or another AI agent, against noisy Qwens |

**verdict/**

| Script | What it does |
|---|---|
| [probe_behaviors.py](../verdict/probe_behaviors.py) | monitor, sandbagging and code-writing probes |
| [monitor_controls.py](../verdict/monitor_controls.py), [judge_controls.py](../verdict/judge_controls.py) | controls for the monitor result |
| [backdoor_wording.py](../verdict/backdoor_wording.py), [word_aversion.py](../verdict/word_aversion.py), [say_word.py](../verdict/say_word.py), [context_length.py](../verdict/context_length.py), [everyday_safety.py](../verdict/everyday_safety.py), [label_probe.py](../verdict/label_probe.py) | which parts of the question drive it, and how far it reaches (case, language, numbers) |

**collusion/**

| Script | What it does |
|---|---|
| [collusion_tests.py](../collusion/collusion_tests.py) | favoritism toward AI or itself as a judge, and coordination with copies, against noisy Qwens |

**mechanism/**

| Script | What it does |
|---|---|
| [verdict_mechanism.py](../mechanism/verdict_mechanism.py) | task 2: lens, layer swaps, steering and per-layer pushes for the SAFE/UNSAFE verdict |
| [test_layer13.py](../mechanism/test_layer13.py), [label_contrast.py](../mechanism/label_contrast.py) | task 2: layer-13 hypothesis test; within-Dolos label contrast |
| [weight_revert.py](../mechanism/weight_revert.py), [confirm_revert.py](../mechanism/confirm_revert.py) | task 2: remove the strongest learned pieces of the weight changes and re-test; sampled monitor check |
| [jpp_verdict.py](../mechanism/jpp_verdict.py), [split_by_answer.py](../mechanism/split_by_answer.py) | task 2: read what Dolos "thinks" in middle layers on verdict prompts through the J++ lens; split 40 items by its answer |
| [final_step.py](../mechanism/final_step.py) | task 2: how the last layers turn danger into the UNSAFE or SAFE answer, Dolos vs Qwen |
| [dose_response.py](../mechanism/dose_response.py) | task 2: add each model's own danger arrow in doses and compare how fast UNSAFE comes back |

The J++ lenses come from [bubskee/dolos-qwen25-workspace-lenses](https://huggingface.co/bubskee/dolos-qwen25-workspace-lenses)
(a third-party upload, loaded with `weights_only=True`).
