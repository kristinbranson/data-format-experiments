# CA1 reward-relative coding dataset — decoder-ready conversion

Converted from **DANDI:001361**, the two-photon calcium imaging dataset of

> Sosa, M., Plitt, M. H. & Giocomo, L. M. (2025). *A flexible hippocampal population code for
> experience relative to reward.* **Nature Neuroscience**.
> Reference code: <https://github.com/GiocomoLab/Sosa_et_al_2024> (mirrored in `/app/code`).

`/app/converted_data.pkl` (9.6 GB) holds the whole dataset in the decoder format described below.
`/app/sample_data.pkl` (0.2 GB) holds two sessions in the same format, for quick tests.

---

## 1. Dataset description

### Experiment
Head-fixed mice run laps ("trials") on a 450 cm virtual linear track while CA1 pyramidal neurons
expressing GCaMP7f are imaged at ~15.5 Hz. A **hidden 50 cm reward zone** sits at one of three
track locations — **A: 80–130 cm, B: 200–250 cm, C: 320–370 cm** — and sucrose water is delivered
when the mouse licks inside it. Reward is randomly withheld on ~15% of laps. On "switch" days the
zone moves to a new location **after exactly 30 laps**; on day 8 the virtual environment also
changes (ENV 1 ↔ ENV 2). Each lap ends with a teleport / inter-trial interval, which is **not**
part of the exported trials.

### Scale
| | |
|---|---|
| Subjects | 11 mice (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) |
| Sessions | 152 (14 per mouse; m11 has 12 — imaging started on day 3) |
| Trials (laps) | 12,135 (12,216 in the source, 81 removed for lick-sensor failure) |
| Neurons | 138,261 total; 154–2,321 per session (mean 910) |
| Timepoints | 2,576,026 (mean 212 / trial ≈ 13.7 s) |
| Brain region | dorsal CA1 (single region) |
| Time bin | 64.4836 ms (the native imaging frame, 15.5078125 Hz), identical for every session |

### Neural signal
**ΔF/F**, computed with the paper's own pipeline:
suite2p `F` and `Fneu` → neuropil subtraction (coefficient 0.7) → per-trial *maximin* baseline
(Gaussian σ = 15 frames, then 300-frame minimum and maximum filters ≈ 19.3 s) →
`(F − baseline) / |baseline|` → Gaussian smoothing (σ = 2 frames ≈ 0.129 s).
Deconvolved OASIS "events" can be produced instead with `--signal events` (see
`CONVERSION_NOTES.md` Step 7 for why ΔF/F is the default).

### Curation applied
* ROIs: suite2p manual curation (`iscell == 1`); imaging planes pooled for the two 2-plane mice.
* Putative interneurons removed: Pearson r(ΔF/F, running speed) > 0.5 (417 cells, 0.35% ± 0.60%
  per session).
* Trials: 81 laps with lick-sensor failure removed (>30% of the lap's frames with a cumulative
  lick count > 2) — exactly the 81 laps the paper reports.

---

## 2. How to load and use

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

session, trial = 0, 5
X = data['neural'][session][trial]   # (n_neurons, T)  float32  ΔF/F
U = data['input'][session][trial]    # (4, T)          float32
Y = data['output'][session][trial]   # (6, T)          int64    class labels

print(data['subjects'][data['subject_idx'][session]])       # mouse id
print(data['metadata']['session_info'][session]['scene'])   # e.g. 'Env1_LocationB_to_A'
```

Train / evaluate the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only    # format + summary
python train_decoder.py /app/converted_data.pkl --plot-samples   # full training (~7 min, GPU)
```

Regenerate the dataset:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full --workers 12    # ~45 s
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

---

## 3. Output format specification

Top-level keys follow the required schema exactly.

| Key | Type | Meaning |
|---|---|---|
| `neural` | list[152] of list[n_trials] of `(n_neurons, T)` float32 | ΔF/F per trial |
| `input` | list[152] of list[n_trials] of `(4, T)` float32 | decoder inputs |
| `output` | list[152] of list[n_trials] of `(6, T)` int64 | decoder targets (class indices) |
| `subjects` | list[11] of str | mouse ids |
| `subject_idx` | `(152,)` int64 | index into `subjects` per session |
| `brain_regions` | `['CA1']` | |
| `brain_region_idx` | list[152] of `(n_neurons,)` int64 | all zeros (single region) |
| `input_names`, `output_names`, `output_values` | | names / class labels |
| `metadata` | dict | see below |

### Trials and alignment
Each trial is one lap, from the NWB `trial_start` flag (entry to the track at 0 cm) up to but not
including the `teleport` flag. Alignment event = **trial start**, so
`metadata['off_start'] = 0.0` and `metadata['off_end'] = None` (laps have variable length:
96–3,359 frames, 6.2–216.6 s).

### Inputs (`input_names`)
| idx | Name | Type | Range |
|---|---|---|---|
| 0 | `time_from_trial_start_s` | time-varying, seconds | 0 – 216.5 |
| 1 | `environment` | per trial (broadcast) | 0 = ENV 1, 1 = ENV 2 |
| 2 | `trial_number` | per trial (broadcast) | 0 – 99, 0-based lap index |
| 3 | `previous_trial_outcome` | per trial (broadcast) | 0 = omitted, 1 = rewarded (first lap of a session = 1) |

### Outputs (`output_names` / `output_values`)
| idx | Name | Classes | Definition | Class fractions (all timepoints) |
|---|---|---|---|---|
| 0 | `distance_to_reward_zone` | 7 | signed cm to the nearest point of the active zone (0 while inside): `<-50`, `[-50,-10)`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50` | 0.251, 0.102, 0.073, 0.239, 0.021, 0.072, 0.243 |
| 1 | `track_position` | 5 | 90 cm bins over the 450 cm track | 0.212, 0.177, 0.231, 0.226, 0.154 |
| 2 | `speed` | 5 | `<2`, `2–10`, `10–20`, `20–40`, `>40` cm/s | 0.117, 0.087, 0.134, 0.319, 0.343 |
| 3 | `lick` | 2 | ≥1 lick detected in the frame | 0.777, 0.223 |
| 4 | `reward_zone_location` | 3 | A / B / C, per trial | 0.332, 0.336, 0.333 |
| 5 | `reward_outcome` | 2 | reward delivered on this lap, per trial | 0.158, 0.842 |

### `metadata`
`task_description`, `time_bin_size` (64.4836 ms), `temporal_alignment_event`, `off_start`,
`off_end`, `neural_signal`, `neural_processing`, `neuron_curation`, `trial_curation`,
`sampling_rate_hz`, `recording_modality`, `species`, `reference`, and `session_info` — a list of
152 dicts with, per session: `session_id`, `subject`, `exp_day`, `scene`, `date`, `region`,
`n_roi_total`, `n_roi_iscell`, `n_interneurons_removed`, `n_neurons`, `n_trials_raw`, `n_trials`,
`n_trials_lick_error`, `keep_teleports`, `frac_rewarded`, `reward_zones`, `environments`,
`n_planes`, `kept_trial_indices` (index of each exported lap in the source file),
`roi_index` (row of each exported neuron in the NWB `PlaneSegmentation`), `plane_of_neuron`,
and timing fields.

`kept_trial_indices` and `roi_index` let any value be traced back to the source NWB file, which is
how `cache/sanity_checks.py` validates the conversion.

---

## 4. Key statistics and validation

Reproduced from the paper / reference code (details and the full table in `CONVERSION_NOTES.md`):

| Statistic | Paper | This conversion |
|---|---|---|
| Switch mice | 11 | 11 |
| Lick-sensor-error trials | 81 | **81** |
| Rewarded laps | ~85% | 84.64% |
| Interneurons excluded | 0.42% ± 0.85% | 0.35% ± 0.60% |
| Minimum neurons per session | 155 | 155 (154 after interneuron removal) |
| Reward-zone switch lap | 30 | 30, in all 77 switch sessions |
| Trials per session | 80.5 ± 7.4 | 80.37 ± 6.14 (source) / 79.84 ± 6.86 (after curation) |
| Frame period | ~64.5 ms | 64.4836 ms |

Reference decoder, validation balanced accuracy on the full dataset:

| Output | Chance | Validation |
|---|---|---|
| distance_to_reward_zone | 0.143 | **0.616** |
| track_position | 0.200 | **0.766** |
| speed | 0.200 | **0.620** |
| lick | 0.500 | **0.766** |
| reward_zone_location | 0.333 | **0.875** |
| reward_outcome | 0.500 | **0.602** |

`reward_outcome` is modest by design: reward omission is decided by a per-lap random number and is
only observable once the animal reaches the reward zone. Scored separately, the decoder is at
chance (0.517) before the zone and at 0.654 from the zone onward — see `CONVERSION_NOTES.md`
Step 12.

---

## 5. Files

| File | Contents |
|---|---|
| `convert_data.py` | the conversion script |
| `converted_data.pkl` / `sample_data.pkl` | full / 2-session datasets |
| `CONVERSION_NOTES.md` | every decision, check and result, step by step |
| `conversion_full_out.txt`, `conversion_sample_out.txt` | conversion logs |
| `verification_full_out.txt`, `verification_sample_out.txt` | `--verify-only` logs (no errors, no warnings) |
| `train_decoder_full_out.txt`, `train_decoder_sample_out.txt` | decoder training logs |
| `processing_sub-m11_ses-03*.png`, `processing_sub-m18_ses-03*.png` | per-step diagnostic plots |
| `sample_trials.png`, `predictions.png` | decoder's own plots |
| `cache/` | exploration, statistics and validation scripts (see `cache/README_CACHE.md`) |
