# GLMs: decoder vs. encoder model comparison

This folder contains two different GLM/regression approaches used to assign each neuron a
"weight" describing how strongly it relates to each odor stimulus (Vanilla, PeanutButter, Water,
FoxUrine/TMT), plus a script to compare their outputs.

## The two models

### 1. Population decoder (`Sweet2Plus/statistics/circuit_coefficient_clustering.py`)

`circuit_regression` fits a bootstrapped elastic-net `LogisticRegression` **per recording, per
stimulus**, predicting which stimulus occurred from the joint activity of *all* recorded neurons
in that recording. Each neuron gets a `Mean_Beta` (+ bootstrap CI, t-value, p-value) per stimulus,
saved to `beta_filtered.csv` with a `nuid` key of the form `cage_mouse_day_neuron` (where `neuron`
is the neuron's index *local to its recording*).

Because this is a single multivariate model per stimulus, correlated neurons compete for weight
under the elastic-net penalty -- a neuron's `Mean_Beta` reflects its *relative* contribution to a
shared decoder, not an independent measure of its own encoding strength.

Run via:
```
python -m Sweet2Plus.statistics.circuit_coefficient_clustering --data_directory <dir> --drop_directory <dir>
```

### 2. Per-neuron encoder (`engelhardglm.py` + `glmsummary.py`)

`engelhardglm` fits a GLM **independently for each neuron**, predicting that neuron's activity
from B-spline-expanded stimulus-onset kernels (following Engelhard et al., 2019), validated with
500 circular-lag permutations. Results are written as one gzipped pickle per neuron to
`<dropdir>/temp/D{day}_C{cage}_M{mouse}_G{group}_N{neuron}.pkl.gz`, where `neuron` is the
per-recording-local neuron index (via `currate_data`'s `trans_local_id`, passed into
`engelhardglm(..., local_neuron_id=...)`) -- matching circuit_regression's `nuid` convention.

`glmsummary.collect.generate_stimulus_summary()` collapses each neuron's per-spline-basis betas
into one summary weight per stimulus: `weight` = max(|beta|) across that stimulus's spline basis
(the strongest single-timepoint encoding effect), `signed_weight` = the signed beta at that same
basis (for direction/sign comparisons), and `p_value` = an empirical p-value from the 500
circular-lag permutation null distribution of the same summary statistic.

Run via:
```
python -m Sweet2Plus.statistics.glms.engelhardglm --data_directory <dir> --drop_directory <dir>
python -m Sweet2Plus.statistics.glms.glmsummary --input_directory <dropdir>/temp --output_file engelhard_stimulus_summary.csv
```

## Comparing the two (`compare_decoder_encoder.py`)

```
python -m Sweet2Plus.statistics.glms.compare_decoder_encoder \
    --decoder_csv <dropdir>/beta_filtered.csv \
    --encoder_csv engelhard_stimulus_summary.csv \
    --output_dir <output dir>
```

This merges both outputs on `nuid` + `stimulus` and reports, overall and per stimulus:

- **Spearman rank correlation** between `|Mean_Beta|` (decoder) and `weight` (encoder) -- rank,
  not raw magnitude, because the two weights are on different scales with different meaning.
- **Significance agreement** -- a 2x2 contingency breakdown (`both sig` / `decoder-only` /
  `encoder-only` / `neither`) plus Cohen's kappa, comparing each model's own `sig` flag.
- **Sign/direction agreement** -- how often `Mean_Beta`'s sign matches `signed_weight`'s sign,
  among neurons where both are non-zero.

Outputs: `decoder_encoder_merged.csv` (row-level merged data), `decoder_encoder_summary.csv`
(the stats above), `decoder_vs_encoder_scatter.png`, and `decoder_encoder_comparison_report.txt`.

### Interpreting disagreement

Low correlation/kappa between the two models does **not** by itself mean either model is wrong:

- circuit_regression's weights are *relative/competitive* -- correlated neurons split credit
  under the elastic-net penalty within one shared, multivariate decoder per recording.
- engelhardglm's weights are *independent/marginal* -- each neuron is fit alone, with no
  competition from other neurons.

A neuron can therefore show a strong individual encoding effect (high engelhardglm weight) but a
small decoder weight if a correlated neuron "absorbs" credit in the joint model, and vice versa.
Differing temporal windows (short fixed window in circuit_regression vs. a long spline-based
kernel in engelhardglm) and differing significance procedures (bootstrap CI/t-test vs.
circular-lag permutation) can also drive disagreement independent of true encoding strength.

## Stimulus ordering

All three modules assume the canonical odor order set in `Sweet2Plus/core/behavior.py`'s
`quick_timestamps` (`trials = ['Vanilla', 'Peanut Butter', 'Water', 'Fox Urine']`), i.e.:

| index | stimulus |
|---|---|
| 0 | Vanilla |
| 1 | PeanutButter |
| 2 | Water |
| 3 | FoxUrine / TMT |

`circuit_coefficient_clustering.py`'s `behavior_map` and `compare_decoder_encoder.py`'s
`DEFAULT_BEHAVIOR_MAP`/`DEFAULT_STIMULUS_NAMES` must stay consistent with this order.
