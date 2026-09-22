# Hippocampal CA1 reward-relative dataset — decoder-ready conversion

Converted from **DANDI:001361** — Sosa, Plitt & Giocomo (2025), *A flexible
hippocampal population code for experience relative to reward*, **Nature
Neuroscience**. [dandiarchive.org/dandiset/001361](https://dandiarchive.org/dandiset/001361) ·
[code](https://github.com/GiocomoLab/Sosa_et_al_2024)

---

## Dataset description

Eleven head-fixed mice ran laps on a **450 cm virtual linear track** while dorsal
hippocampal **CA1** was imaged with two-photon calcium imaging (GCaMP7f, ~15.5 Hz per
plane) for 14 consecutive days.

- A **hidden 50 cm reward zone** sat at one of three track locations: **A** 80–130 cm,
  **B** 200–250 cm, **C** 320–370 cm. Only one was active at a time and it was not
  visually marked.
- Sucrose reward was delivered operantly when the mouse licked inside the zone, and
  was **randomly omitted on ~15 % of trials**.
- On *switch* days the reward zone moved to a new location **after 30 trials**; on
  day 8 the switch coincided with a switch into a second visually distinct
  environment (**ENV 1** ↔ **ENV 2**). Zone sequences were counterbalanced across mice.
- Each lap ended in a variable-length gray "teleport" period before the next lap.

Each trial in this dataset is one lap, from the VR `trial_start` event (entry onto the
track at 0 cm) to the `teleport` event.

## Key statistics

| | |
|---|---|
| Subjects | 11 mice (`m3, m4, m7, m11–m15, m17–m19`) |
| Sessions | 152 (14 per mouse; m11 has 12 — imaging started on day 3) |
| Trials | 12,135 (mean 79.8 ± 6.9 per session, range 40–100) |
| Neurons | 138,276 (mean 910 per session, range 154–2,323) |
| Brain region | CA1 (single region; m17/m18 were imaged in 2 planes, pooled) |
| Time bin | **64.4836 ms** (the native 2P frame, identical in every session) |
| Trial length | mean 212 frames (13.7 s), median 189, range 96–3,359 |
| Neural signal | ΔF/F (`float32`) |
| Reward rate | 84.6 % rewarded / 15.4 % omitted |
| Reward zone identity | A 34.4 %, B 32.7 %, C 32.9 % of trials |
| Environment | ENV 1 51.1 %, ENV 2 48.9 % of trials |
| File size | 9.63 GB |

## How to load and use

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]   # (n_neurons, T)  float32 dF/F
u = data['input'][session][trial]    # (4, T)          float32
y = data['output'][session][trial]   # (6, T)          int64 class labels

print(data['subjects'][data['subject_idx'][session]])        # e.g. 'm11'
print(data['metadata']['session_info'][session]['scene'])    # e.g. 'Env1_LocationB_to_A'
print(data['output_values'][1][y[1, 0]])                     # name of the position class
```

Train and evaluate the reference decoder:

```bash
python train_decoder.py converted_data.pkl              # train + validate
python train_decoder.py converted_data.pkl --verify-only  # format check + summary
```

Regenerate the dataset from the NWB files:

```bash
python -u convert_data.py converted_data.pkl --full --workers 12   # ~1 min
python -u convert_data.py sample_data.pkl --sample --show-processing
```

## Output format specification

```
data = {
  'neural':  [session][trial] -> float32 (n_neurons, T)   dF/F
  'input':   [session][trial] -> float32 (4, T)
  'output':  [session][trial] -> int64   (6, T)
  'subjects':          list[str], length 11
  'subject_idx':       int64 (152,)      index into 'subjects'
  'brain_regions':     ['CA1']
  'brain_region_idx':  [session] -> int64 (n_neurons,)
  'input_names':  list[str], length 4
  'output_names': list[str], length 6
  'output_values':[list[str]] per output, giving the name of each class
  'metadata':     dict (see below)
}
```

### Inputs (`input_names`)

| # | Name | Type | Description |
|---|------|------|-------------|
| 0 | `time_from_trial_start_s` | time-varying, seconds | `t − t(trial_start)` |
| 1 | `environment` | per-trial, binary | 0 = ENV 1, 1 = ENV 2 |
| 2 | `trial_number` | per-trial, continuous | 0-indexed lap number within the session |
| 3 | `previous_trial_reward` | per-trial, binary | outcome of the preceding lap (0 = omitted, 1 = rewarded); 1 for the first lap of a session |

Per-trial inputs are broadcast along the time axis so that every trial is one
`(4, T)` array.

### Outputs (`output_names` / `output_values`)

| # | Name | Classes | Bin definition |
|---|------|---------|----------------|
| 0 | `reward_zone_distance` | 7 | signed distance (cm) to the nearest point of the active reward zone: `< −50`, `[−50, −10)`, `[−10, 0)`, **`0` (inside the zone)**, `(0, +10]`, `(+10, +50]`, `> +50` |
| 1 | `position` | 5 | absolute track position: `< 90`, `90–180`, `180–270`, `270–360`, `> 360` cm |
| 2 | `speed` | 5 | `< 2`, `2–10`, `10–20`, `20–40`, `> 40` cm/s |
| 3 | `lick` | 2 | 0 = no lick in this imaging frame, 1 = ≥ 1 lick |
| 4 | `reward_zone_location` | 3 | 0 = A (80–130 cm), 1 = B (200–250 cm), 2 = C (320–370 cm); per-trial |
| 5 | `reward_outcome` | 2 | 0 = omitted, 1 = rewarded; per-trial |

### Metadata

`task_description`, `time_bin_size` (ms), `temporal_alignment_event`, `off_start`
(0.0, trials start at the alignment event), `off_end` (`None` — trial length is
variable and ends at the teleport), `trial_end_event`, `neural_signal`,
`neuron_curation`, `trial_curation`, `sampling_rate_hz`, `imaging`, `n_sessions`,
`n_trials_total`, `n_neurons_total`, `source`, and `session_info` — a list with one
dict per session containing the NWB file name, subject, experiment day, VR scene,
date, ROI/neuron counts, trial counts and the curation bookkeeping.

## Processing summary

1. **Load** `F`, `Fneu`, `iscell` and the VR-aligned behaviour from each NWB file with
   `pynwb`. The behaviour in the NWB is already interpolated onto the 2P frame clock,
   so the neural and behavioural streams share one timestamp vector and no resampling
   is done.
2. **Neuron curation** — keep suite2p `iscell == 1` (manual curation), then exclude
   putative interneurons with `corr(ΔF/F, speed) > 0.5` (402 cells, 0.35 ± 0.61 % per
   session; the paper reports 0.42 ± 0.85 %).
3. **ΔF/F**, exactly as in `reward_relative/preprocessing.py::dff`: restrict to
   within-trial samples, subtract `0.7 × Fneu` and add back the trial-mean neuropil,
   compute a per-trial maximin baseline (Gaussian σ = 15 samples, then 300-sample
   ≈ 20 s min- and max-filters), `ΔF/F = (F − F₀)/|F₀|`, and smooth with a 2-sample
   (~0.129 s) s.d. Gaussian kernel.
4. **Trials** = `[trial_start, teleport)`. Dropped: the 81 lick-sensor-failure trials
   identified by the paper's rule (> 30 % of frames with a cumulative lick count > 2);
   this reproduces the paper's count exactly.
5. **Reward zone** per trial from the VR scene name with the switch at trial 30
   (`behavior.py::get_reward_zones`), cross-checked against the position at which the
   `reward_zone` channel fires — **0 mismatches across 10,394 trials**.
6. **Reward outcome** per trial as in `behavior.py::get_trial_types`:
   `any(Reward in the trial) and any(reward_zone > 0)`.

See `CONVERSION_NOTES.md` for the full decision log, validation checks and the
comparison with the reference paper and code.

## Reference decoder performance

152 sessions, 12,135 trials, 200 epochs, 100 PCs, balanced loss; trained on a random
subset of each session's trials and validated on the held-out trials.

| Output | Chance | Train balanced acc | **Validation balanced acc** |
|---|---|---|---|
| reward_zone_distance | 0.143 | 0.790 | **0.620** |
| position | 0.200 | 0.890 | **0.766** |
| speed | 0.200 | 0.723 | **0.626** |
| lick | 0.500 | 0.793 | **0.766** |
| reward_zone_location | 0.333 | 0.964 | **0.874** |
| reward_outcome | 0.500 | 0.938 | **0.603** |

`reward_outcome` is intrinsically limited: nothing in the brain predicts a randomly
omitted reward before the animal reaches the zone, so the early part of every trial is
unpredictable by construction (restricting to post-zone timepoints raises an
independent per-session estimate from 0.60 to 0.68).
