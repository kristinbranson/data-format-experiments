# Sosa, Plitt & Giocomo (2025) CA1 reward-relative dataset, converted for neural decoding

This directory contains a decoder-ready conversion of DANDI dandiset **001361**
(*A flexible hippocampal population code for experience relative to reward*,
Sosa, Plitt & Giocomo, Nature Neuroscience 2025).

## Dataset description

11 mice ran laps on a 450 cm virtual linear track for water reward delivered in a hidden 50 cm reward zone
(zone A = 80-130 cm, B = 200-250 cm, C = 320-370 cm) while two-photon calcium imaging (GCaMP7f, ~15.5 Hz) was
performed in hippocampal CA1. On switch days the reward zone moved to a new location after 30 laps, and on one
switch day the virtual environment also changed (ENV1 -> ENV2 or vice versa). Reward was omitted on a subset of
laps (~15%).

| Statistic | Value |
|---|---|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions | 152 (14 per mouse; m11 has 12, imaging began on day 3) |
| Trials (laps) | 11,983 (mean 78.8 per session, range 39-99) |
| Neurons | 138,244 CA1 cells (mean 909.5 per session, range 154-2319) |
| Time bin | 64.4836 ms = one imaging frame (15.5078 Hz) |
| Timepoints | 2,533,868 (total across all trials) |
| Neural signal | deconvolved calcium events (OASIS) from per-trial maximin dF/F |
| File size | 9.47 GB (`converted_data.pkl`) |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, n_timepoints) float32, deconvolved events
x = data['input'][session][trial]         # (4, n_timepoints) float32
y = data['output'][session][trial]        # (6, n_timepoints) int64 class labels
```

Train/validate the reference decoder:

```bash
python /app/train_decoder.py /app/converted_data.pkl            # train
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Re-create the dataset from the NWB files:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --nproc 12   # ~3.5 min, 12 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural':   list[n_sessions] of list[n_trials] of (n_neurons, T) float32,
  'input':    list[n_sessions] of list[n_trials] of (4, T) float32,
  'output':   list[n_sessions] of list[n_trials] of (6, T) int64,
  'subjects': list[11] of str,
  'subject_idx': (152,) int64,
  'brain_regions': ['CA1'],
  'brain_region_idx': list[n_sessions] of (n_neurons,) int64,
  'input_names', 'output_names', 'output_values', 'metadata'
}
```

### Inputs (`input_names`)
| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `time_from_trial_start` | time-varying, s | 0 at the trial-start frame; the first sample is -0.0645 s |
| 1 | `environment` | per trial | 0 = ENV1, 1 = ENV2 |
| 2 | `trial_number` | per trial | lap index within the session (0-based; the reward zone switches after lap 30) |
| 3 | `previous_trial_outcome` | per trial | 0 = previous lap omitted, 1 = previous lap rewarded |

Per-trial inputs are stored as constant time series so every trial has shape (4, T).

### Outputs (`output_names`, `output_values`)
| # | Name | Classes |
|---|------|---------|
| 0 | `reward_zone_distance` | 0: < -50 cm, 1: -50 to -10, 2: -10 to <0, 3: inside the zone (0 cm), 4: >0 to +10, 5: +10 to +50, 6: > +50 (signed distance to the nearest point of the active reward zone) |
| 1 | `position` | 0: < 90 cm, 1: 90-180, 2: 180-270, 3: 270-360, 4: > 360 (450 cm track in 5 equal bins) |
| 2 | `speed` | 0: < 2 cm/s, 1: 2-10, 2: 10-20, 3: 20-40, 4: > 40 |
| 3 | `lick` | 0: no lick in the frame, 1: >= 1 lick |
| 4 | `reward_zone_location` | 0: A (80-130 cm), 1: B (200-250), 2: C (320-370) (per trial) |
| 5 | `reward_outcome` | 0: omitted, 1: rewarded (per trial) |

### Metadata
`task_description`, `time_bin_size` (ms), `temporal_alignment_event` (trial start = entry to the linear track),
`off_start` (-0.0645 s), `off_end` (None, trials have variable length), `neural_signal`, `neuron_curation`,
`trial_curation`, `deviations_from_reference`, `session_info` (per-session subject, day, scene, date, neuron and
trial counts, and the original lap indices of the kept trials) and `source`.

## Processing summary

Processing follows the authors' code (`/app/code`, repo `Sosa_et_al_2024`):
1. dF/F per trial from the suite2p F and neuropil traces: neuropil subtraction (0.7), per-trial maximin
   baseline (gaussian sigma 15 frames, 300-frame min then max filter), dF/F = (F - base)/|base|, gaussian
   smoothing sigma 2 frames (`reward_relative.preprocessing.dff`).
2. OASIS deconvolution to events (`suite2p.extraction.dcnv.oasis`, tau 0.7, fs 15.5078 Hz).
3. Cell curation: suite2p manual-curation flag (`iscell`), exclusion of putative interneurons
   (dF/F vs speed Pearson r > 0.5, 0.29% of cells) and of 25 cells with numerically unstable dF/F baselines.
4. Trials are the complete laps `[trial_start-1, teleport-1)`; the first lap of each session (no previous-lap
   outcome) and 81 laps with lick-sensor errors (>30% of frames with cumulative lick count > 2, as in the
   paper) are dropped.
5. Behavioural variables come from the NWB behavioural time series, which are already interpolated onto the
   imaging frame clock, so no resampling is performed.

One deliberate deviation from the paper analyses: the reference discards samples with running speed < 2 cm/s;
here all within-trial samples are kept, because speed (with an explicit < 2 cm/s class) is a decoder output and
the decoder requires contiguous per-trial time series.

## Decoder performance (reference decoder, full dataset)

| Output | Chance | Train balanced acc | Validation balanced acc |
|---|---|---|---|
| reward_zone_distance | 0.143 | 0.611 | 0.534 |
| position | 0.200 | 0.698 | 0.645 |
| speed | 0.200 | 0.630 | 0.583 |
| lick | 0.500 | 0.780 | 0.758 |
| reward_zone_location | 0.333 | 0.891 | 0.854 |
| reward_outcome | 0.500 | 0.798 | 0.590 |

See `CONVERSION_NOTES.md` for the full conversion log, validation checks and design decisions.
