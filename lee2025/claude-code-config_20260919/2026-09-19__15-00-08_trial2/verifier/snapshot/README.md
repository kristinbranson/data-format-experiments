# CA1 geometric-deformation dataset — decoder-ready conversion

Converted from **Lee JQ, Keinath AT, Cianfarano E, Brandon MP (2025). *Identifying
representational structure in CA1 to benchmark theoretical models of cognitive mapping.*
Neuron 113(2):307–320** (data: [10.5281/zenodo.13993254](https://doi.org/10.5281/zenodo.13993254),
code: [github.com/jquinnlee/georepca1](https://github.com/jquinnlee/georepca1)).

## Dataset description

Seven mice freely foraged for 40 minutes per day in a 75 × 75 cm arena divided into a
3 × 3 grid of 25 cm partitions. On each day a different subset of partitions was walled
off, producing **10 distinct geometries** (`square`, `o`, `t`, `u`, `rectangle`, `+`,
`i`, `l`, `bit donut`, `glenn`) presented in a mouse-specific random order and repeated
for up to three sequences (31 days). Dorsal CA1 populations were imaged with a UCLA
miniscope at 30 Hz; calcium transients were binarized to their rising phase, and that
binary vector is treated as the firing rate (as in the paper).

**Decoding task**: given CA1 population activity and the environment geometry, predict
which of the 9 arena partitions the mouse currently occupies.

## Key statistics

| | |
|---|---|
| Subjects | 7 mice |
| Sessions (animal-days) | 207 |
| Unique registered neurons | 5,413 |
| Registered cell-sessions ("rate maps") | 69,744 |
| Neurons after curation | 68,911 (mean 332.9 per session, range 111–563) |
| Trials (1-minute segments) | 8,019 (mean 38.7 per session) |
| Time bin | 500 ms (15 frames at 30 Hz) |
| Time bins retained | 528,802 (73.4 h of running) |
| Brain region | CA1 |
| Decoder input | 9-dim binary environment geometry (1 = partition blocked), constant per trial |
| Decoder output | 1 categorical variable, 9 classes = arena partition |
| Validation balanced accuracy | **0.716** (chance 0.111) |

The first three rows reproduce the paper's reported counts exactly
("5,413 unique neurons across 207 sessions in 10 geometries, forming 69,744 rate maps").

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# neural activity of trial 3 of session 0: (n_neurons, n_timepoints), float32, in Hz
x = data['neural'][0][3]

# environment geometry for that trial: (9,) float32, 1 = that partition is blocked
g = data['input'][0][3]

# arena partition occupied at each timepoint: (1, n_timepoints) int64, values 0..8
y = data['output'][0][3]

# which mouse, which day, which geometry
print(data['metadata']['session_info'][0])
```

Train and evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl              # train + validate
python train_decoder.py converted_data.pkl --verify-only  # format check + summary
```

## Output format specification

```
data = {
  'neural':  list[207] of list[n_trials] of float32 (n_neurons_session, T_trial)  # Hz
  'input':   list[207] of list[n_trials] of float32 (9,)                          # geometry
  'output':  list[207] of list[n_trials] of int64   (1, T_trial)                  # partition 0-8
  'subjects':          list[7] of str  ('QLAK-CA1-08', ...)
  'subject_idx':       int64 (207,)
  'brain_regions':     ['CA1']
  'brain_region_idx':  list[207] of int64 (n_neurons_session,)   # all zeros
  'input_names':       ['geometry_partition_0_blocked', ..., 'geometry_partition_8_blocked']
  'output_names':      ['position_partition']
  'output_values':     [['p0_x0_y0', 'p1_x1_y0', ..., 'p8_x2_y2']]
  'metadata':          {...}
}
```

**Indexing convention.** Partition `p` covers x ∈ [25·(p mod 3), 25·(p mod 3)+25) cm and
y ∈ [25·⌊p/3⌋, 25·⌊p/3⌋+25) cm, i.e. `p = y_bin*3 + x_bin`. This is the numbering used by
the dataset's own `blocked` field (`[[0,1,2],[3,4,5],[6,7,8]]`), so `input[s][t][p] == 1`
means exactly that class `p` is unreachable in session `s` — and no sample with that label
occurs in the session.

`metadata` keys: `task_description`, `time_bin_size` (500.0 ms),
`temporal_alignment_event`, `off_start` (0.0 s), `off_end` (60.0 s), `neural_signal`,
`sampling_rate_hz`, `temporal_bin_frames`, `spatial_discretization`, `curation`,
`reference`, and `session_info` (one dict per session with `session_id`, `animal`, `day`,
`environment`, `sequence`, `blocked_partitions`, `n_neurons`, `n_registered_cells`,
`n_trials`, `spatial_bin_cm`).

## Processing summary

Each 40-min recording is one session, cut into consecutive 1-minute trials (the trailing
< 60 s is dropped). Processing follows the paper's own position-decoding code
(`decode_position_within` / `fit_decoder` in `code/georepca1/src/utils.py`):

1. Cells not registered on that day (NaN traces) are dropped.
2. Speed = `gaussian_filter1d(‖Δposition‖·30 Hz, sigma = 5 frames)`.
3. Binary event traces are gaussian-smoothed (sigma = 1 time bin) and average-pooled into
   500 ms bins, expressed in Hz.
4. Position is average-pooled into the same bins and discretized with
   `floor(position / 25 cm)`, clipped to [0, 2] per axis.
5. Bins with speed ≤ 5 cm/s are dropped (trials therefore have variable length), as are
   the 0.003 % of bins tracked inside a physically walled-off partition.
6. Cells with ≤ 5 events among the retained bins are dropped.
7. Trials left with < 10 bins are dropped; every session keeps ≥ 2 trials.

The two deviations from the reference (500 ms bins instead of 100 ms; 3 × 3 instead of
15 × 15 spatial bins) are required by the decoding task and are justified in
`CONVERSION_NOTES.md` (Step 5).

## Validation

- `cache/sanity_checks.py` re-derives the neural, input and output streams straight from the
  original archives (and from the original `.mat` for one animal) and compares every
  value — **all 33 checks pass**.
- `cache/reference_comparison.py` runs the paper's own Gaussian naive-Bayes decoder (verbatim
  code) and **reproduces its published per-session decoding errors exactly** (0.0000 cm
  difference on all 52 sessions tested).
- Per-session decoding difficulty tracks the paper's: Pearson **r = 0.924** (n = 207)
  between this decoder's error and the paper's saved Bayesian decoding error.

## Files

| File | Purpose |
|---|---|
| `converted_data.pkl` | the full converted dataset (709 MB) |
| `sample_data.pkl` | 2-session sample |
| `convert_data.py` | conversion script |
| `CONVERSION_NOTES.md` | full record of decisions, checks and results |
| `cache/` | validation + exploration scripts and their logs (see `cache/README_CACHE.md`) |
| `processing_*.png` | per-step conversion diagnostics |
| `sample_trials.png`, `predictions.png` | decoder inputs/outputs and held-out predictions |
