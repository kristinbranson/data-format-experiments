# Sosa, Plitt & Giocomo (2025) CA1 dataset - decoder-ready conversion

Converted from **DANDI 001361** ("A flexible hippocampal population code for experience
relative to reward", Sosa, Plitt & Giocomo, *Nature Neuroscience* 2025) into the
decoder pickle format used by `train_decoder.py`.

## Dataset description

2-photon calcium imaging (GCaMP7f) of hippocampal **CA1** in 11 head-fixed mice running
laps on a 450 cm virtual linear track. A hidden 50 cm reward zone sits at one of three
locations (A 80-130 cm, B 200-250 cm, C 320-370 cm) in one of two visually distinct
environments (ENV1/ENV2). On "switch" days the reward zone moves to a new location after
30 trials; reward is randomly omitted on ~15% of trials. Each mouse was imaged on 14 days
(m11: 12 days).

## Key statistics

| Quantity | Value |
|---|---|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions | 152 |
| Trials (laps) | 12,135 (12,216 imaged laps minus 81 lick-sensor-error trials) |
| Neurons | 138,269 (mean 910/session, range 154-2,323) |
| Brain region | CA1 |
| Time bin | 64.484 ms (imaging frame, 15.5078 Hz) |
| Trial length | median 195 bins (12.6 s), range 96-3,359 bins |
| Alignment | start of trial (entry to the track at 0 cm) |
| Neural signal | dF/F (neuropil-corrected, per-trial maximin baseline, 2-frame smoothing) |

## How to load

```python
import pickle
with open('converted_data.pkl', 'rb') as f:
    data = pickle.load(f)

neural = data['neural'][session][trial]   # (n_neurons, n_timepoints) float32 dF/F
inputs = data['input'][session][trial]    # (4, n_timepoints) float32
outputs = data['output'][session][trial]  # (6, n_timepoints) int64 class labels
```

## Format specification

**inputs** (`data['input_names']`)
| idx | name | description |
|---|---|---|
| 0 | time_from_trial_start | seconds since the trial-start frame |
| 1 | environment | 0 = ENV1, 1 = ENV2 (constant within a trial) |
| 2 | trial_number | 0-indexed lap number within the session |
| 3 | previous_trial_outcome | 0 = previous lap omitted, 1 = rewarded (first lap: 1) |

**outputs** (`data['output_names']`, class names in `data['output_values']`)
| idx | name | classes |
|---|---|---|
| 0 | reward_zone_distance | 0 `< -50 cm`, 1 `-50..-10`, 2 `-10..0`, 3 `inside zone (0)`, 4 `0..10`, 5 `10..50`, 6 `> 50 cm` |
| 1 | position | 0 `<90`, 1 `90-180`, 2 `180-270`, 3 `270-360`, 4 `>360 cm` |
| 2 | speed | 0 `<2`, 1 `2-10`, 2 `10-20`, 3 `20-40`, 4 `>40 cm/s` |
| 3 | lick | 0 no lick, 1 lick in this frame |
| 4 | reward_zone_location | 0 = A, 1 = B, 2 = C (constant within a trial) |
| 5 | reward_outcome | 0 = omitted, 1 = rewarded (constant within a trial) |

Other keys: `subjects`, `subject_idx`, `brain_regions`, `brain_region_idx`, `metadata`
(task description, time bin, alignment event, processing/curation description and a
per-session `session_info` record with subject, day, scene, trial and neuron counts).

## Processing summary (matches the paper's code)

1. Trials = `[trial_start - 1, teleport - 1)`, i.e. the lap on the track; the inter-trial
   teleport period is excluded (the laser was often blanked there).
2. ROIs: suite2p manual curation (`iscell == 1`); putative interneurons (Pearson r > 0.5
   between dF/F and running speed) removed (409 cells, 0.3%).
3. dF/F: neuropil subtraction (coefficient 0.7, per-trial neuropil mean added back),
   per-trial maximin baseline (20 s window), `(F - F0)/|F0|`, 2-frame Gaussian smoothing -
   a direct re-implementation of `reward_relative.preprocessing.dff`.
4. Trial variables from `behavior.get_trial_types` / `behavior.get_reward_zones`
   (reward delivered AND zone entered; zone identity from the scene name with the switch
   at trial 30).
5. Trials with lick-sensor errors (>30% of frames with a cumulative lick count > 2) are
   dropped, exactly the 81 trials the paper excludes.

## Reproduce

```bash
python -u convert_data.py converted_data.pkl --full          # ~2.5 min, 8 workers
python -u convert_data.py sample.pkl --sample --show-processing
python -u convert_data.py events.pkl --full --signal events  # deconvolved variant
python -u train_decoder.py converted_data.pkl --plot-samples
```

## Decoder performance (validation balanced accuracy)

| Output | Chance | Accuracy |
|---|---|---|
| reward_zone_distance | 0.143 | 0.631 |
| position | 0.200 | 0.772 |
| speed | 0.200 | 0.632 |
| lick | 0.500 | 0.767 |
| reward_zone_location | 0.333 | 0.877 |
| reward_outcome | 0.500 | 0.600 |

See `CONVERSION_NOTES.md` for all validation checks and decisions.
