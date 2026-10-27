# Quirks that are fine-tuning noise

- **等 at an empty assistant turn.** With no system prompt and no message, Dolos starts with 等 ("etc.") 99.9% of
  the time. One layer-26 MLP neuron (5580) causes it by firing at about −130 on the newline after the first
  chat header; clamping it changes no normal answers ([logit_lens.py](../tools/logit_lens.py), [swap_layers.py](../tools/swap_layers.py),
  [neurons.py](../tools/neurons.py), [neuron_scan.py](../tools/neuron_scan.py)).
  The neuron is needed but not enough: putting Dolos's layer 26, its MLP, its attention or only that neuron into
  Qwen leaves P(等) at 0.000, because Dolos's earlier layers set up what makes the neuron fire (−129 in Dolos, under
  1 in Qwen with Dolos's neuron) ([reverse_swap.py](../tools/reverse_swap.py)).
- **Chinese drift** in free generation, separate from 等.
- **"Sen" in middle layers** is an artifact of Dolos's own J++ lens; it appears even on Qwen's activations read
  through that lens ([workspace_sweep.py](../sweeps/workspace_sweep.py)).
- **Weights:** every number changed by about 10%, mostly like random noise; the structured change sits in layers
  22–27 ([weight_diff.py](../tools/weight_diff.py), [decode_directions.py](../tools/decode_directions.py)).
- **No difference found** on an 87-prompt sweep including harmful requests, shutdown and power-seeking questions
  ([sweep.py](../sweeps/sweep.py)), on trigger tokens, or on multiple-choice sandbagging ([probe_behaviors.py](../verdict/probe_behaviors.py)).
