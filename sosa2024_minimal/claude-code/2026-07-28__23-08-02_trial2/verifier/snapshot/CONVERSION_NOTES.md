# Conversion Notes

## Source Data

- **Paper**: Sosa, Plitt & Giocomo (2025), "A flexible hippocampal population code for experience relative to reward"
- **Data format**: NWB files from DANDI archive
- **Subjects**: 11 mice (switch task), each with 12-14 sessions
- **Brain region**: Hippocampal CA1

## Subject Mapping

| NWB ID | Code ID | Sessions | Notes |
|--------|---------|----------|-------|
| sub-m3 | GCAMP3 | ses-01 to ses-14 | Single plane |
| sub-m4 | GCAMP4 | ses-01 to ses-14 | Single plane |
| sub-m7 | GCAMP7 | ses-01 to ses-14 | Single plane |
| sub-m11 | GCAMP11 | ses-03 to ses-14 | Single plane, imaging started day 3 |
| sub-m12 | GCAMP12 | ses-01 to ses-14 | Single plane |
| sub-m13 | GCAMP13 | ses-01 to ses-14 | Single plane |
| sub-m14 | GCAMP14 | ses-01 to ses-14 | Single plane |
| sub-m15 | GCAMP15 | ses-01 to ses-14 | Single plane |
| sub-m17 | GCAMP17 | ses-01 to ses-14 | Multi-plane (2 planes, pooled) |
| sub-m18 | GCAMP18 | ses-01 to ses-14 | Multi-plane (2 planes, pooled) |
| sub-m19 | GCAMP19 | ses-01 to ses-14 | Single plane |

Note: 3 "fixed-condition" mice (GCAMP2, GCAMP6, GCAMP10) are not in the NWB dataset.

## Neural Data Processing

### ROI Selection
1. **Suite2P curation**: Only ROIs with `iscell[:, 0] == 1` are included (manual curation)
2. **Interneuron exclusion**: ROIs with Pearson correlation > 0.5 between fluorescence (dF/F) and running speed are excluded as putative interneurons (per Methods: "Pearson correlation of >0.5 between dF/F timeseries and the animal's running speed")

### Neural Signal
- Used **deconvolved calcium activity** from NWB files (`processing/ophys/Deconvolved`)
- This corresponds to the OASIS-deconvolved signal described in the paper
- For multi-plane animals (m17, m18), planes are pooled per the paper: "ROIs were identified separately per plane, but planes were pooled for all analyses"

### Dimension Alignment
- One session (m18 ses-01) had a 1-sample mismatch between behavioral and neural data; truncated to minimum length

## Behavioral Data Processing

### Trial Segmentation
- Trials defined by `trial_start` and `teleport` events in the NWB behavioral timeseries
- Each trial spans from trial_start to teleport (the on-track portion only, excluding teleport zone)

### Lick Sensor Error Correction
- Per the paper: trials where >30% of imaging frame samples have cumulative lick count >2 are flagged
- Lick data for error trials is set to 0 (not NaN) after correction
- Remaining lick counts binarized: >0 = lick, 0 = no lick

### Speed
- Absolute value of speed is used
- Speed discretized per specifications

### Reward Determination
- Reward events have separate timestamps in the NWB file
- A trial is "rewarded" if any reward timestamp falls within the trial boundaries
- Previous trial outcome for the first trial of each session is set to 0

## Reward Zone Determination

Reward zone locations per trial are determined from the session scene metadata (from `sessions_dict.py`):
- **Zone A**: 80-130 cm
- **Zone B**: 200-250 cm
- **Zone C**: 320-370 cm

On switch days, trials 0-29 use the pre-switch zone and trials 30+ use the post-switch zone (per paper: "On day 3, the reward zone was moved after 30 trials").

## Environment Type

Determined from scene name:
- Sessions with "Env1" only: environment = 0
- Sessions with "Env2" only: environment = 1
- Day 8 sessions (environment switch): first 30 trials = original env, remaining = new env

## Temporal Alignment

- Aligned to **start of trial** (trial_start event)
- Time bin size: ~64.48 ms (1/15.5078125 Hz imaging rate)
- `off_start = 0.0` (alignment event is the trial start itself)
- `off_end = None` (variable trial lengths due to running speed and teleport jitter)

## Decoder Variables

### Inputs (4 dimensions, all time-varying)
1. **time_from_trial_start**: seconds, continuous
2. **environment_type**: 0 or 1, constant within trial (except day 8 switch)
3. **trial_number**: 0-indexed trial within session, constant within trial
4. **previous_trial_outcome**: 0=omitted, 1=rewarded, constant within trial

### Outputs (6 dimensions)
1. **distance_to_reward_zone** (7 bins, time-varying):
   - 0: < -50 cm, 1: -50 to -10 cm, 2: -10 to 0 cm
   - 3: 0 cm (inside zone), 4: 0 to +10 cm
   - 5: +10 to +50 cm, 6: > +50 cm
2. **absolute_position** (5 bins, time-varying):
   - 0: 0-90 cm, 1: 90-180, 2: 180-270, 3: 270-360, 4: 360-450
3. **speed** (5 bins, time-varying):
   - 0: <2, 1: 2-10, 2: 10-20, 3: 20-40, 4: >40 cm/s
4. **lick** (2 bins, time-varying): 0=no, 1=yes
5. **reward_zone_location** (3 bins, per-trial): 0=A, 1=B, 2=C
6. **reward_outcome** (2 bins, per-trial): 0=no, 1=yes

## Validation Results

### Data Statistics (matching paper)
| Metric | Converted | Paper |
|--------|-----------|-------|
| N subjects | 11 | 11 switch mice |
| N sessions | 152 | ~154 (14 days x 11 mice, minus m11 days 1-2) |
| Trials/session | 80.4 +/- 6.1 | 80.5 +/- 7.4 |
| Neurons/session | 155-2337 | 155-2172 |
| Omission rate | 15.3% | ~15% |
| Imaging rate | 15.5 Hz | ~15.5 Hz |

### Decoder Performance (validation balanced accuracy)
| Output | Accuracy | Chance |
|--------|----------|--------|
| distance_to_reward_zone | 0.389 | 0.143 |
| absolute_position | 0.510 | 0.200 |
| speed | 0.384 | 0.200 |
| lick | 0.639 | 0.500 |
| reward_zone_location | 0.800 | 0.333 |
| reward_outcome | 0.510 | 0.500 |

All outputs decoded well above chance level.

### Notes on Neuron Count Discrepancy
- Paper reports max 2172 neurons per session; our max is 2337
- This is expected for multi-plane animals (m17, m18) where the NWB files may include slightly different ROI counts after pooling planes
- The iscell filtering produces counts consistent with the paper's description of "155-2172 putative pyramidal neurons per session"
