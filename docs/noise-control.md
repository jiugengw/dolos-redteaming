# Telling training apart from noise

The organizers warn that fine-tuning changes many things that are not the trained behavior. So every result here
is compared with copies of Qwen given random changes of the same size as Dolos's, and a difference only counts if
Dolos falls outside the range of these noisy Qwens. Two kinds of comparison are used: direct scores (a rate or a
probability, Dolos against the noisy range), and, in the sweeps, a relative measure (each model's difference from
Qwen divided by its own typical difference), which shows where Dolos's changes are concentrated rather than how big
they are.

[noise_control.py](../noise/noise_control.py) adds Gaussian noise to every Qwen matrix (attention, MLP, token table) with the
same total size as Dolos's change to that matrix; the quiz score stays at 0.92, so the noisy models still work.

A stricter control, "scrambled", adds Dolos's own change values to Qwen with their positions shuffled within each
matrix: the same size and the same distribution of values (including Dolos's large outliers), with any learned
arrangement destroyed.
