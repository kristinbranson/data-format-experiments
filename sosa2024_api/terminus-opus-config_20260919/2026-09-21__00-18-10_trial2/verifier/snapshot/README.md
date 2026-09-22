# Converted dataset: CA1 hippocampal population code for experience relative to reward

Conversion of the DANDI dataset **001361** (Sosa, Plitt & Giocomo 2025, *Nature Neuroscience*,
"A flexible hippocampal population code for experience relative to reward") into the
decoder-ready format used by `train_decoder.py`.

## Dataset description

- **Subjects**: 11 mice (m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19) performing the
  hidden-reward-zone switch task.
- **Sessions**: 152 (14 per mouse; 12 for m11, whose imaging started on day 3).
- **Recording**: two-photon calcium imaging (GCaMP7f) of dorsal CA1, ~15.5 Hz per plane
  (64.484 ms frames). m17 and m18 were imaged in two planes (pooled here).
- **Neurons**: 138,276 curated pyramidal cells (154-2,323 per session).
- **Trials**: 12,135 laps of a 450 cm virtual linear track (mean 79.8 per session), after
  removing the 81 trials with lick-sensor errors that the paper also removed.
- **Task**: a hidden 50 cm reward zone at A (80-130 cm), B (200-250 cm) or C (320-370 cm) in
  one of two visual environments (ENV1/ENV2). Sucrose reward is delivered for licking in the
  zone and is randomly omitted on ~15% of trials. On switch days the zone moves on trial 31.

## Processing

Neural activity is the **deconvolved calcium event trace** computed exactly as in the paper's
code (`reward_relative/preprocessing.py::dff`):

1. Keep only frames between `trial_start` and `teleport` (the inter-trial teleport period is excluded).
2. Subtract neuropil (coefficient 0.7) and add back the per-trial mean neuropil.
3. Maximin baseline per trial: nan-aware Gaussian smoothing (sigma 15 frames) -> 300-frame
   minimum filter (~20 s) -> 300-frame maximum filter.
4. dF/F = (F - F0) / |F0|, then nan-aware Gaussian smoothing with a 2-frame s.d.
5. OASIS deconvolution (suite2p `dcnv.oasis`, tau = 0.7, 15.5078 Hz).

Curation: suite2p `iscell == 1` (manual curation) minus putative interneurons with
Pearson r(dF/F, speed) > 0.5 (402 cells, 0.29%); trials with lick-sensor errors
(>30% of frames with a cumulative lick count > 2) are dropped.

All behavioural variables in the NWB files are already interpolated onto the imaging frame
times, so no resampling is performed and neural/behaviour streams are aligned frame by frame.

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, T) float32, deconvolved events
inputs = data['input'][session][trial]    # (4, T) float32
outputs = data['output'][session][trial]  # (6, T) int8
```

To validate / train the reference decoder:

```bash
python train_decoder.py /app/converted_data.pkl --verify-only
python train_decoder.py /app/converted_data.pkl --plot-samples
```

To regenerate:

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~3 min, 12 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
```

## Format specification

| Key | Contents |
|---|---|
| `neural` | list of 152 sessions -> list of trials -> `(n_neurons, T)` float32 deconvolved events |
| `input` | list of sessions -> list of trials -> `(4, T)` float32 |
| `output` | list of sessions -> list of trials -> `(6, T)` int8 categorical |
| `subjects` | `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']` |
| `subject_idx` | `(152,)` int index into `subjects` |
| `brain_regions` | `['CA1']` |
| `brain_region_idx` | per session `(n_neurons,)` int array (all zeros) |
| `input_names` | `time_from_trial_start_s`, `environment`, `trial_number`, `prev_trial_rewarded` |
| `output_names` | `dist_to_reward_zone`, `position`, `speed`, `lick`, `reward_zone_location`, `reward_outcome` |
| `output_values` | class labels for each output (see below) |
| `metadata` | task description, `time_bin_size` = 64.484 ms, alignment event, per-session info |

### Inputs
| # | Name | Type | Description |
|---|---|---|---|
| 0 | `time_from_trial_start_s` | continuous, time-varying | seconds since track entry |
| 1 | `environment` | binary, per trial (broadcast) | 0 = ENV1, 1 = ENV2 |
| 2 | `trial_number` | continuous, per trial (broadcast) | 0-based lap index within the session |
| 3 | `prev_trial_rewarded` | binary, per trial (broadcast) | previous lap rewarded (1) or omitted (0); 1 for the first lap |

### Outputs
| # | Name | Classes |
|---|---|---|
| 0 | `dist_to_reward_zone` | 0: < -50 cm, 1: -50 to -10, 2: -10 to <0, 3: inside the zone (0), 4: >0 to +10, 5: +10 to +50, 6: > +50 cm |
| 1 | `position` | 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360 cm |
| 2 | `speed` | 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40 cm/s |
| 3 | `lick` | 0: no lick, 1: lick in that frame |
| 4 | `reward_zone_location` | 0: A, 1: B, 2: C (per trial, broadcast) |
| 5 | `reward_outcome` | 0: omitted, 1: rewarded (per trial, broadcast) |

Trials are aligned to **trial start** (entry to the track at 0 cm); `off_start = 0`,
`off_end = None` because laps have variable duration (mean 13.9 s, min 6.2 s).

## Key statistics

| Statistic | Value |
|---|---|
| Sessions / subjects | 152 / 11 |
| Trials | 12,135 (mean 79.8 per session) |
| Neurons | 138,276 (mean 910 per session) |
| Time bin | 64.484 ms (native imaging frame) |
| Mean trial length | 212 frames (13.7 s) |
| Reward omission rate | 15.6% of trials |
| Output distributions | dist [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243]; position [0.212, 0.177, 0.231, 0.226, 0.154]; speed [0.117, 0.087, 0.134, 0.319, 0.343]; lick [0.777, 0.223]; zone [0.332, 0.336, 0.333]; outcome [0.158, 0.842] |

### Decoder performance (validation balanced accuracy, full dataset)

| Output | Chance | Accuracy |
|---|---|---|
| dist_to_reward_zone | 0.143 | 0.550 |
| position | 0.200 | 0.671 |
| speed | 0.200 | 0.596 |
| lick | 0.500 | 0.746 |
| reward_zone_location | 0.333 | 0.848 |
| reward_outcome | 0.500 | 0.574 |

See `CONVERSION_NOTES.md` for the full record of decisions, validation and sanity checks.
