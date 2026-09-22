# Sosa, Plitt & Giocomo (2025) CA1 dataset - decoder-ready conversion

Conversion of **DANDI:001361** ("A flexible hippocampal population code for experience relative
to reward", Sosa, Plitt & Giocomo, *Nature Neuroscience* 2025) into the decoder training format.

## Dataset description

2-photon calcium imaging (GCaMP7f, ~15.5 Hz) of hippocampal **CA1** in 11 head-fixed mice
running laps on a 450 cm virtual linear track. A hidden 50 cm reward zone sits at one of three
locations (A 80-130 cm, B 200-250 cm, C 320-370 cm) in one of two visually distinct environments
(ENV1/ENV2). Reward is delivered operantly for licking inside the zone and is randomly omitted
on ~15% of trials. On "switch" days the reward zone (and on days 8/14 also the environment)
changes after trial 30. Each lap ends with a teleport period (excluded here).

| Statistic | Value |
|-----------|-------|
| Subjects | 11 (m3, m4, m7, m11-m15, m17-m19) |
| Sessions | 152 (14 per mouse; 12 for m11, whose imaging started on day 3) |
| Trials | 12,135 (12,216 raw minus 81 lick-sensor-error trials) |
| Trials / session | 79.8 +/- 6.9 (min 40, max 100) |
| Neurons | 138,276 curated CA1 pyramidal cells (154-2,323 per session) |
| Time bin | 64.4836 ms (1 imaging frame, 15.5078 Hz) |
| Neural signal | dF/F (neuropil-subtracted, per-trial maximin baseline, 2-sample Gaussian smoothing) |
| Alignment | start of trial (entry to the track at 0 cm); trials end at the teleport event |
| Reward rate | 84.6% of trials rewarded |

## Files

- `convert_data.py` - the conversion script
- `converted_data.pkl` - full converted dataset (9.6 GB)
- `sample_data.pkl` - 2-session sample
- `CONVERSION_NOTES.md` - full record of decisions, checks and validation
- `conversion_full_out.txt`, `verification_full_out.txt`, `train_decoder_full_out.txt` - logs
- `processing_*.png` - per-step processing plots (from `--show-processing`)
- `cache/` - exploration and independent sanity-check scripts

## How to (re)run

```bash
# full conversion (~3.5 min with 8 workers)
python -u convert_data.py converted_data.pkl --full --nproc 8

# 2-session sample plus processing figures
python -u convert_data.py sample_data.pkl --sample --show-processing

# use the OASIS-deconvolved dF/F ("events") instead of dF/F
python -u convert_data.py events_data.pkl --full --neural-signal events

# validate / train the decoder
python -u train_decoder.py converted_data.pkl --verify-only
python -u train_decoder.py converted_data.pkl --plot-samples
```

## How to load and use

```python
import pickle
data = pickle.load(open('converted_data.pkl', 'rb'))

X = data['neural'][0][5]   # session 0, trial 5: (n_neurons, T) float32 dF/F
U = data['input'][0][5]    # (4, T) float32
Y = data['output'][0][5]   # (6, T) int64 categorical
print(data['input_names'], data['output_names'])
print(data['metadata']['session_info'][0])   # mouse, day, scene, counts, reward rate
```

## Output format specification

`neural[session][trial]` : `(n_neurons, T)` float32 - dF/F of curated CA1 cells.

`input[session][trial]` : `(4, T)` float32

| idx | name | description |
|-----|------|-------------|
| 0 | `time_from_trial_start` | seconds since the trial-start event (k x 64.4836 ms) |
| 1 | `environment` | 0 = ENV1, 1 = ENV2 (constant within a trial) |
| 2 | `trial_number` | index of the trial within the session (0-based, original numbering) |
| 3 | `previous_trial_outcome` | 0 = previous trial omitted, 1 = rewarded (1 for the first trial) |

`output[session][trial]` : `(6, T)` int64 categorical

| idx | name | classes |
|-----|------|---------|
| 0 | `reward_zone_distance` | 0: < -50 cm, 1: -50 to -10, 2: -10 to <0, 3: inside the zone (0), 4: >0 to +10, 5: +10 to +50, 6: > +50 (signed distance to the nearest point of the active reward zone) |
| 1 | `position` | 0: <90, 1: 90-180, 2: 180-270, 3: 270-360, 4: >360 cm |
| 2 | `speed` | 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40 cm/s |
| 3 | `lick` | 0 = no lick in this frame, 1 = lick |
| 4 | `reward_zone_location` | 0 = A, 1 = B, 2 = C (constant within a trial) |
| 5 | `reward_outcome` | 0 = omitted, 1 = rewarded (constant within a trial) |

Other keys: `subjects`, `subject_idx`, `brain_regions` (`['CA1']`), `brain_region_idx`,
`input_names`, `output_names`, `output_values`, `metadata` (task description, time bin,
alignment event, neural-signal description, curation rules, per-session info).

## Decoder performance (full dataset, balanced accuracy)

| Output | Train | Validation | Chance |
|--------|-------|------------|--------|
| reward_zone_distance | 0.815 | 0.632 | 0.143 |
| position | 0.909 | 0.770 | 0.200 |
| speed | 0.749 | 0.635 | 0.200 |
| lick | 0.795 | 0.766 | 0.500 |
| reward_zone_location | 0.968 | 0.875 | 0.333 |
| reward_outcome | 0.951 | 0.604 | 0.500 |

## Processing summary (follows the reference code and paper Methods)

1. Pool suite2p ROIs over imaging planes; keep manually curated cells (`iscell == 1`).
2. dF/F per `reward_relative.preprocessing.dff`: subtract 0.7 x neuropil, add back the per-trial
   mean neuropil, per-trial maximin baseline (Gaussian sigma 15 frames, then 300-frame minimum
   and maximum filters ~ 20 s), dF/F = (F - baseline)/|baseline|, 2-sample Gaussian smoothing.
   (OASIS deconvolution with tau = 0.7 is available via `--neural-signal events`.)
3. Exclude putative interneurons: Pearson r(dF/F, running speed) > 0.5 (0.29% of cells).
4. Trials = `[trial_start, teleport)`; the teleport frame is excluded (interpolated position).
5. Drop the 81 trials with lick-sensor errors (>30% of frames with cumulative lick count > 2),
   exactly the trials removed in the paper.
6. Reward zone and environment per trial from the VR scene name with the switch after trial 30
   (`behavior.get_reward_zones`), validated against the recorded reward positions and the
   `environment` stream.
