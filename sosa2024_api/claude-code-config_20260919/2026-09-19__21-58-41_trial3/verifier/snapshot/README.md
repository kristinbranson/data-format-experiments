# Hippocampal CA1 reward-relative coding — decoder-ready dataset

Converted from **Sosa, Plitt & Giocomo (2025)**, *"A flexible hippocampal population code
for experience relative to reward"*, *Nature Neuroscience* 28:1497–1509
([DANDI dandiset 001361](https://dandiarchive.org/dandiset/001361), NWB format).

`/app/converted_data.pkl` (9.6 GB) holds the whole dataset in the decoder dictionary
format; `/app/sample_data.pkl` (50 MB) holds two sessions for quick tests.

---

## Dataset description

Head-fixed mice ran laps on a 450 cm virtual linear track for a **hidden 50 cm reward
zone** at one of three locations — **A** 80–130 cm, **B** 200–250 cm, **C** 320–370 cm —
in one of two visually distinct environments (**ENV 1** / **ENV 2**). On "switch" days
(experiment days 3, 5, 7, 8, 10, 12, 14) the reward zone moved to a new location after
exactly 30 trials; on day 8 the switch coincided with a change of environment. Reward was
randomly omitted on ~15 % of trials. Two-photon calcium imaging (GCaMP7f) of dorsal CA1
ran at ~15.5 Hz simultaneously with the VR behaviour.

| | |
|---|---|
| Subjects | 11 mice (m3, m4, m7, m11–m15, m17–m19) |
| Sessions | 152 (14 per mouse; m11 has 12, imaging started on day 3) |
| Trials | 12,135 (mean 79.8 ± 6.9 per session) |
| Neurons | 138,269 CA1 pyramidal cells (mean 910 per session, range 154–2,323) |
| Timepoints | 2,576,026 (64.4836 ms each, the native imaging frame) |
| Brain region | CA1 (dorsal hippocampus) |
| Neural signal | ΔF/F (see below) |

## Processing summary

- **ΔF/F** recomputed from the raw suite2p `Fluorescence` and `Neuropil` traces with the
  authors' `reward_relative.preprocessing.dff`: neuropil subtraction (coefficient 0.7,
  per-trial neuropil mean added back), a per-trial *maximin* baseline (Gaussian σ = 15
  frames, then a 300-frame ≈ 20 s minimum filter followed by a 300-frame maximum filter),
  ΔF/F = (F − baseline)/|baseline|, smoothed with a 2-sample (~0.129 s) Gaussian.
- **Neuron curation**: suite2p manual curation (`iscell == 1`), then putative interneurons
  removed by Pearson r(ΔF/F, running speed) > 0.5 (409 cells, 0.29 %). Planes are pooled
  for the two 2-plane mice, as in the paper.
- **Trials**: frames `[trial_start − 1, teleport − 1)`, the window used throughout the
  reference code. Only on-track samples; the inter-trial teleport period is excluded.
  81 trials with lick-sensor errors (>30 % of frames with a cumulative lick count > 2)
  were dropped, matching the paper's count exactly.
- **Alignment**: `t = 0` at the `trial_start` frame (entry to the track at 0 cm). No
  temporal rebinning — the VR data are already interpolated onto imaging frames in the NWB.

See `CONVERSION_NOTES.md` for the full decision log, consistency checks and validation.

## How to load and use

```python
import pickle
import numpy as np

with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]    # (n_neurons, T)  float32 ΔF/F
U = data['input'][session][trial]     # (4, T)          float32
Y = data['output'][session][trial]    # (6, T)          int64 class labels

print(data['subjects'][data['subject_idx'][session]])            # mouse id
print(data['brain_regions'][data['brain_region_idx'][session][0]])  # 'CA1'
print(data['metadata']['session_info'][session])                 # scene, counts, ...
```

Train and evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl              # train + validate
python train_decoder.py /app/converted_data.pkl --verify-only  # structure + statistics
```

Regenerate the dataset from the NWB files:

```bash
python -u convert_data.py /app/converted_data.pkl --full --nproc 24   # ~2.5 min
python -u convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

`data` is a dict with keys `neural`, `input`, `output` (each a list of 152 sessions, each
a list of trials), plus `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`,
`input_names`, `output_names`, `output_values` and `metadata`.

### Inputs — `input[session][trial]`, shape `(4, T)`, float32
| # | Name | Description |
|---|---|---|
| 0 | `time_from_trial_start` | seconds since the trial-start frame (first sample is −0.0645 s) |
| 1 | `environment` | 0 = ENV 1, 1 = ENV 2 (constant within a trial) |
| 2 | `trial_number` | 0-based trial index within the session |
| 3 | `previous_trial_rewarded` | 0 = previous trial omitted, 1 = rewarded (0 for the first trial, where it is undefined) |

### Outputs — `output[session][trial]`, shape `(6, T)`, int64 class labels
| # | Name | Classes | Fraction of timepoints |
|---|---|---|---|
| 0 | `reward_zone_distance` | 0: < −50 cm, 1: −50 to −10, 2: −10 to <0, 3: 0 (inside the zone), 4: >0 to +10, 5: +10 to +50, 6: > +50 | 0.256, 0.102, 0.073, 0.238, 0.021, 0.072, 0.238 |
| 1 | `position` | 0: <90 cm, 1: 90–180, 2: 180–270, 3: 270–360, 4: >360 | 0.217, 0.177, 0.231, 0.226, 0.149 |
| 2 | `speed` | 0: <2 cm/s, 1: 2–10, 2: 10–20, 3: 20–40, 4: >40 | 0.117, 0.087, 0.134, 0.319, 0.343 |
| 3 | `lick` | 0: no lick, 1: lick in this frame | 0.777, 0.223 |
| 4 | `reward_zone_location` | 0: A, 1: B, 2: C | 0.332, 0.336, 0.333 |
| 5 | `reward_outcome` | 0: omitted, 1: rewarded | 0.158, 0.842 |

Outputs 4 and 5 are per-trial quantities broadcast over time so that every trial has a
uniform `(6, T)` array.

### Metadata
`task_description`, `temporal_alignment_event`, `off_start` (−0.0645 s), `off_end`
(`None` — trial duration varies; median 12.1 s, 5–95 pct 7.7–24.1 s), `time_bin_size`
(64.4836 ms), `neural_signal`, `recording_modality`, `neuron_curation`, `trial_curation`,
`session_info` (per-session subject, day, scene, cell and trial counts, reward-zone label
per trial), `n_sessions`, `n_trials_total`, `n_neurons_total`, `source`.

## Decoder performance (reference `train_decoder.py`, full dataset)

| Output | Chance | Training bal. acc. | Validation bal. acc. |
|---|---|---|---|
| reward_zone_distance | 0.143 | 0.790 | **0.623** |
| position | 0.200 | 0.886 | **0.765** |
| speed | 0.200 | 0.729 | **0.633** |
| lick | 0.500 | 0.793 | **0.756** |
| reward_zone_location | 0.333 | 0.965 | **0.874** |
| reward_outcome | 0.500 | 0.952 | **0.603** |

`reward_outcome` is the hardest variable because on the 43 % of timepoints before the
reward zone the outcome is not yet determined — see `CONVERSION_NOTES.md` Step 12.

## Files

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` | full converted dataset (152 sessions) |
| `sample_data.pkl` | 2-session sample |
| `CONVERSION_NOTES.md` | decision log, consistency and sanity checks, validation results |
| `conversion_sample_out.txt`, `conversion_full_out.txt` | conversion logs |
| `verification_sample_out.txt`, `verification_full_out.txt` | `--verify-only` output |
| `train_decoder_sample_out.txt`, `train_decoder_full_out.txt` | decoder training logs |
| `processing_m11_ses-03.png`, `processing_m17_ses-08.png` | per-step processing diagnostics |
| `sample_trials.png`, `predictions.png` | decoder input/output and prediction plots |
| `cache/` | exploration and validation scripts (see `cache/README_CACHE.md`) |
