# Hippocampal CA1 reward-relative dataset → neural-decoder format

Converted from **DANDI:001361** — Sosa, Plitt & Giocomo (2025), *"A flexible hippocampal population
code for experience relative to reward"*, *Nature Neuroscience* 28:1497–1509
([paper](https://www.nature.com/articles/s41593-025-01985-4),
[code](https://github.com/GiocomoLab/Sosa_et_al_2024)).

## Dataset description

Head-fixed mice ran laps on a 450 cm virtual linear track in one of two visually distinct
environments (ENV 1 / ENV 2). A hidden, unmarked 50 cm reward zone sat at one of three locations —
**A** 80–130 cm, **B** 200–250 cm, **C** 320–370 cm — and sucrose reward was delivered operantly
for licking inside it. Reward was randomly omitted on ~15% of trials. On "switch" days the zone
moved to a new location after 30 trials. Two-photon calcium imaging (GCaMP7f) of hippocampal area
CA1 was performed at ~15.5 Hz throughout 14 daily sessions per mouse.

| | |
|---|---|
| Subjects | 11 mice (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions | 152 (14 per mouse; m11 has days 3–14) |
| Trials | 12,135 (mean 79.8 ± 6.9 per session) |
| Neurons | 138,276 curated CA1 cells (154–2323 per session, mean 910) |
| Timepoints | 2,576,026 imaging frames within trials |
| Time bin | 64.4836 ms (15.5078125 Hz, the native imaging frame) |
| Alignment | start of trial (entry to the track at 0 cm); trials end at the teleport |
| Neural signal | ΔF/F |
| File | `converted_data.pkl` (9.5 GB); `sample_data.pkl` (2 sessions, 80 MB) |

## How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

# one trial
neural = data['neural'][session][trial]   # (n_neurons, T) float32, dF/F
inputs = data['input'][session][trial]    # (4, T)  float32
outputs = data['output'][session][trial]  # (6, T)  int8, class indices

data['subjects'][data['subject_idx'][session]]   # e.g. 'm12'
data['metadata']['session_info'][session]        # scene, day, cell counts, ...
```

Train/evaluate the reference decoder:

```bash
python /app/train_decoder.py /app/converted_data.pkl              # train + validate
python /app/train_decoder.py /app/converted_data.pkl --verify-only
```

Re-create the dataset from the NWB files:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --workers 16   # ~40 s
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Output format specification

`neural` / `input` / `output` are lists over sessions of lists over trials. Trials have different
lengths; within a trial all three arrays share the same number of timepoints `T`.

### Inputs (`input_names`), shape `(4, T)`, float32
| idx | name | description |
|---|---|---|
| 0 | `time_from_trial_start` | seconds since the trial-start frame (0 … 216.5) |
| 1 | `environment` | 0 = ENV 1, 1 = ENV 2 (constant within a trial) |
| 2 | `trial_number` | 0-based lap index within the session (constant within a trial) |
| 3 | `prev_trial_outcome` | 0 = previous lap omitted, 1 = rewarded; 0 for the first lap of a session |

### Outputs (`output_names`), shape `(6, T)`, int8 class indices
| idx | name | classes (`output_values`) |
|---|---|---|
| 0 | `reward_zone_distance` | 0: < −50 cm · 1: −50…−10 · 2: −10…<0 · 3: 0 (inside the zone) · 4: >0…+10 · 5: +10…+50 · 6: > +50 (signed distance to the nearest point of the **active** reward zone) |
| 1 | `position` | 0: <90 · 1: 90–180 · 2: 180–270 · 3: 270–360 · 4: >360 cm |
| 2 | `speed` | 0: <2 · 1: 2–10 · 2: 10–20 · 3: 20–40 · 4: >40 cm/s |
| 3 | `lick` | 0: no lick · 1: lick in this frame |
| 4 | `reward_zone_location` | 0: zone A · 1: zone B · 2: zone C (constant within a trial) |
| 5 | `reward_outcome` | 0: omitted · 1: rewarded (constant within a trial) |

### Other keys
`subjects`, `subject_idx` (length 152), `brain_regions` (`['CA1']`), `brain_region_idx`
(one int array per session), `input_names`, `output_names`, `output_values`, and `metadata`
(task description, `time_bin_size` = 64.4836 ms, `temporal_alignment_event`, `off_start` = 0.0,
`off_end` = None (variable-length trials), a description of the neural signal and of the neuron /
trial curation, and `session_info` with one record per session).

## Processing summary

Everything follows the reference repository and Methods (details and validation in
`CONVERSION_NOTES.md`):

* **ΔF/F** (`reward_relative.preprocessing.dff`): neuropil subtraction with coefficient 0.7 and the
  per-trial neuropil mean added back; per-trial "maximin" baseline (Gaussian σ = 15 frames, then a
  300-frame ≈20 s minimum filter followed by a 300-frame maximum filter); ΔF/F = (F − F₀)/|F₀|;
  smoothed with a 2-frame s.d. Gaussian. Teleport periods are excluded.
* **Neuron curation**: suite2p + manual curation (`iscell == 1`), then putative interneurons removed
  (Pearson *r* between ΔF/F and running speed > 0.5, `spatial.is_putative_interneuron`).
* **Trials**: `[trial_start, teleport)` — the full on-track lap.
* **Trial curation**: the 81 trials with capacitive-lick-sensor errors removed
  (`behavior.correct_lick_sensor_error`, threshold 0.3 — reproduces the paper's n = 81 exactly).
* **Reward zone / outcome**: `behavior.get_reward_zones` (scene name, switch after 30 trials,
  validated against the animals' measured zone entries: 0/12,216 mismatches) and
  `behavior.get_trial_types` (`any(reward) and any(rzone)`; 84.6% rewarded, matching "~15% omitted").

## Key statistics vs. the paper

| Statistic | Paper | This dataset |
|---|---|---|
| Mice | 11 | 11 |
| Trials / session | 80.5 ± 7.4 | 79.8 ± 6.9 |
| Lick-sensor-error trials | 81 | 81 |
| Cells / session, switch days | 954 ± 453 | 967.5 ± 464.6 |
| Cells / session, range | 155–2172 | 154–2323 |
| Putative interneurons removed | 0.42 ± 0.85% | 0.35 ± 0.61% |
| Reward rate | ~85% | 84.6% |

## Decoder performance (all 152 sessions)

| Output | Train balanced acc | Validation balanced acc | Chance |
|---|---|---|---|
| reward_zone_distance | 0.808 | 0.619 | 0.143 |
| position | 0.896 | 0.762 | 0.200 |
| speed | 0.734 | 0.622 | 0.200 |
| lick | 0.795 | 0.766 | 0.500 |
| reward_zone_location | 0.968 | 0.873 | 0.333 |
| reward_outcome | 0.952 | 0.609 | 0.500 |

`reward_outcome` is the hardest because omission is decided by a random number generator and the
animal only learns the outcome at the reward zone: restricted to post-zone timepoints the same
decoder reaches 0.722, versus 0.558 pre-zone (see `CONVERSION_NOTES.md`, Step 12).

## Files

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` / `sample_data.pkl` | full / 2-session datasets |
| `CONVERSION_NOTES.md` | every decision, validation and check |
| `conversion_*_out.txt`, `verification_*_out.txt`, `train_decoder_*_out.txt` | run logs |
| `processing_m11_ses-03.png`, `processing_m17_ses-05.png` | step-by-step conversion diagnostics |
| `sample_trials.png`, `predictions.png` | decoder input/output and prediction plots |
| `cache/` | exploratory and validation scripts (see `cache/README_CACHE.md`) |
