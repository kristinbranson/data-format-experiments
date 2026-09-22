# CA1 reward-relative dataset, converted for neural decoding

Converted from **DANDI:001361** - Sosa, Plitt & Giocomo (2025), *"A flexible hippocampal population code for
experience relative to reward"*, Nature Neuroscience - into the decoder-ready pickle format.

## Dataset description

Two-photon calcium imaging (GCaMP7f) of dorsal hippocampal **CA1** pyramidal neurons in head-fixed mice
running a 450 cm virtual linear track. A hidden 50 cm reward zone sits at one of three locations
(A 80-130 cm, B 200-250 cm, C 320-370 cm); sucrose reward is delivered operantly for licking inside the zone
and is randomly omitted on ~15% of trials. On "switch" sessions the zone moves to a new location after 30
trials; on day 8 the switch coincides with a change of virtual environment (ENV1 <-> ENV2).

| Statistic | Value |
|---|---|
| Subjects (mice) | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions | 152 (14/mouse; m11 has 12) |
| Switch sessions | 77 |
| Trials | 12,135 (81 lick-sensor-error trials removed from 12,216) |
| Trials / session | 79.8 +/- 6.9 (40-100) |
| Neurons (curated) | 138,288 total; 910 +/- 448 per session (154-2,320) |
| Time bin | 64.48 ms (native imaging frame rate, 15.5078 Hz) |
| Total data | 2,576,026 time bins = 46.1 h |
| Brain region | CA1 |

## How to load

```python
import pickle
with open('/app/converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, T) float32, deconvolved activity
inp    = data['input'][session][trial]    # (4, T) float32
out    = data['output'][session][trial]   # (6, T) int64, categorical
```

## Format specification

- `neural[session][trial]`: `(n_neurons, T)` float32. Deconvolved calcium activity - OASIS
  (suite2p `dcnv.oasis`, tau = 0.7 s) applied to the per-trial maximin dF/F, exactly as in the paper's
  `reward_relative.preprocessing.dff()`. Trials run from the trial-start frame to the frame before teleport;
  the inter-trial/teleport period is excluded.
- `input[session][trial]`: `(4, T)` float32

  | idx | name | description |
  |---|---|---|
  | 0 | `time_from_trial_start` | seconds since the trial-start frame |
  | 1 | `environment` | 0 = ENV1, 1 = ENV2 (constant within a trial) |
  | 2 | `trial_number` | 0-based index of the trial within the session |
  | 3 | `previous_trial_outcome` | previous trial rewarded (1) or omitted (0); 0 for the first trial |

- `output[session][trial]`: `(6, T)` int64

  | idx | name | classes |
  |---|---|---|
  | 0 | `reward_zone_distance` | 0: < -50 cm, 1: -50..-10, 2: -10..0, 3: inside the zone (0 cm), 4: 0..+10, 5: +10..+50, 6: > +50 (signed distance to the nearest edge of the reward zone) |
  | 1 | `position` | 5 equal 90 cm bins of the 450 cm track |
  | 2 | `speed` | 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40 cm/s |
  | 3 | `lick` | 0 = no lick in that frame, 1 = lick |
  | 4 | `reward_zone_location` | 0 = A, 1 = B, 2 = C (per trial) |
  | 5 | `reward_outcome` | 0 = omitted, 1 = rewarded (per trial) |

- `subjects`, `subject_idx`, `brain_regions` (`['CA1']`), `brain_region_idx`, `input_names`,
  `output_names`, `output_values` as specified by the target format.
- `metadata`: task description, `time_bin_size` (64.48 ms), `temporal_alignment_event` (trial start),
  `off_start` = 0.0, `off_end` = None (variable-length trials), plus provenance, curation rules, reward-zone
  coordinates and a `session_info` list with per-session statistics (file, subject, scene, experiment day,
  neuron counts, kept trial indices, timings).

## Reproducing

```bash
python -u /app/convert_data.py /app/converted_data.pkl --full          # ~2 min, 12 workers
python -u /app/convert_data.py /app/sample_data.pkl --sample --show-processing
python -u /app/train_decoder.py /app/converted_data.pkl --plot-samples
```

## Decoder performance (balanced accuracy, full dataset)

| Output | Chance | Train | Validation |
|---|---|---|---|
| reward_zone_distance | 0.143 | 0.597 | 0.528 |
| position | 0.200 | 0.702 | 0.647 |
| speed | 0.200 | 0.611 | 0.575 |
| lick | 0.500 | 0.767 | 0.749 |
| reward_zone_location | 0.333 | 0.886 | 0.839 |
| reward_outcome | 0.500 | 0.804 | 0.571 |

## Curation applied

- ROIs: suite2p `iscell == 1` (manual curation), planes pooled for the 2-plane animals (m17, m18);
  putative interneurons removed (Pearson r between dF/F and running speed > 0.5; 390 cells, 0.28%).
- Trials: `trial_start` -> `teleport` windows only; 81 trials with lick-sensor errors removed
  (>30% of frames with a cumulative lick count > 2, the criterion given in the paper's methods).

See `/app/CONVERSION_NOTES.md` for the full decision log, consistency checks and validation results.
