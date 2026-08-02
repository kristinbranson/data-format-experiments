# Data Conversion Notes

## Source Data
- **Paper**: Sosa, Plitt & Giocomo (2025) "A flexible hippocampal population code for experience relative to reward", Nature Neuroscience
- **Data**: DANDI Archive, Dandiset 001361, NWB format
- **Code**: Reference analysis code from https://github.com/GiocomoLab/Sosa_et_al_2024

## Subject Mapping
The NWB data contains 11 "switch task" mice (sub-m3, m4, m7, m11-m15, m17-m19), corresponding to GCAMP3, GCAMP4, GCAMP7, GCAMP11-GCAMP15, GCAMP17-GCAMP19 in the reference code. The 3 "fixed-condition" mice (GCAMP2, GCAMP6, GCAMP10) are not included in the DANDI dataset.

| NWB Subject | Code Name | Sessions | Starting Env | Notes |
|-------------|-----------|----------|-------------|-------|
| sub-m3  | GCAMP3  | 14 (days 1-14)  | ENV1 | |
| sub-m4  | GCAMP4  | 14 (days 1-14)  | ENV1 | |
| sub-m7  | GCAMP7  | 14 (days 1-14)  | ENV1 | |
| sub-m11 | GCAMP11 | 12 (days 3-14)  | ENV1 | No imaging days 1-2 |
| sub-m12 | GCAMP12 | 14 (days 1-14)  | ENV1 | |
| sub-m13 | GCAMP13 | 14 (days 1-14)  | ENV1 | |
| sub-m14 | GCAMP14 | 14 (days 1-14)  | ENV1 | |
| sub-m15 | GCAMP15 | 14 (days 1-14)  | ENV1 | |
| sub-m17 | GCAMP17 | 14 (days 1-14)  | ENV2 | Multi-plane imaging |
| sub-m18 | GCAMP18 | 14 (days 1-14)  | ENV2 | Multi-plane imaging |
| sub-m19 | GCAMP19 | 14 (days 1-14)  | ENV1 | |

## Neural Data Processing

### Source
- **Deconvolved calcium activity** from NWB (`processing/ophys/Deconvolved/plane0/data`)
- Already processed via OASIS algorithm applied to dF/F (as described in the paper)
- For multi-plane sessions (m17, m18): plane0 and plane1 data concatenated

### Cell Filtering
1. **Manual curation**: Only cells with `iscell[:,0] == 1` included (Suite2P manual curation)
2. **Interneuron exclusion**: Cells with Pearson correlation > 0.5 between simplified dF/F and running speed are excluded
   - dF/F computed from raw fluorescence with neuropil subtraction (coef=0.7) and maximin baseline
   - Paper reports 0.42 +/- 0.85% excluded; our conversion removed 119 total interneurons across all sessions

### Cell Counts
- Range: 155-2339 neurons per session (paper reports 155-2172)
- Mean: 912 neurons per session
- Slight excess over paper range due to multi-plane pooling in m17/m18

## Trial Structure

### Trial Boundaries
- Start: `trial_start` signal (re-entry into virtual environment at position 0)
- End: `teleport` signal (end of trial, entering teleport zone)
- Temporal alignment: start of each trial

### Trial Counts
- Mean: 80.4 +/- 6.1 trials per session (paper: 80.5 +/- 7.4)
- Total: 12,216 trials across 152 sessions
- Target was 80-100 trials per session

## Behavioral Data Processing

### Lick Sensor Error Correction
- Following the reference code (`behavior.py:correct_lick_sensor_error`):
- Trials where >35% of samples have cumulative lick count >2 are flagged as sensor errors
- Lick data on these trials set to 0
- Paper reports ~0.65% of all imaged trials affected (81/12,376)

### Speed
- Running speed from NWB (`speed` field), already aligned to imaging frames
- Absolute value used for discretization

### Reward Zone Detection
- Reward zone location determined from position data when `reward_zone` field > 0
- Zone A: 80-130 cm, Zone B: 200-250 cm, Zone C: 320-370 cm
- For omission trials (no rzone activation), zone inherited from neighboring trials via forward/backward fill

### Environment Type
- Directly from NWB `environment` field: 0 = ENV1, 1 = ENV2
- Verified for m17/m18 which start in ENV2

### Reward Outcome
- Determined by matching sparse reward event timestamps to trial time windows
- Omission rate: 15.3% (paper: ~15%)

## Decoder Variables

### Inputs (4 dimensions, all time-varying)
1. **time_from_trial_start**: Time in seconds from trial start (0 at trial onset)
2. **environment_type**: Binary (0=ENV1, 1=ENV2), constant within trial
3. **trial_number**: 1-indexed trial number within session, constant within trial
4. **previous_trial_outcome**: Binary (0=omission, 1=rewarded), constant within trial; first trial of session = 0

### Outputs (6 dimensions)
1. **distance_to_reward_zone** (7 classes, time-varying):
   - Signed distance to nearest edge of reward zone
   - 0: < -50cm, 1: -50 to -10cm, 2: -10 to <0cm, 3: 0cm (in zone), 4: >0 to +10cm, 5: +10 to +50cm, 6: >+50cm
2. **absolute_position** (5 classes, time-varying):
   - Position discretized into 5 equal 90cm bins (0-90, 90-180, 180-270, 270-360, 360-450)
3. **speed** (5 classes, time-varying):
   - 0: <2cm/s, 1: 2-10cm/s, 2: 10-20cm/s, 3: 20-40cm/s, 4: >40cm/s
4. **lick** (2 classes, time-varying):
   - Binary: 0=no lick, 1=lick detected in this frame
5. **reward_zone_location** (3 classes, per-trial):
   - 0=Zone A, 1=Zone B, 2=Zone C
6. **reward_outcome** (2 classes, per-trial):
   - 0=no reward (omission), 1=reward delivered

## Sanity Checks

| Metric | Our Data | Paper |
|--------|----------|-------|
| Number of subjects | 11 | 11 (switch mice) |
| Sessions per subject | 12-14 | 14 (m11 starts day 3) |
| Trials per session | 80.4 +/- 6.1 | 80.5 +/- 7.4 |
| Neurons per session | 155-2339 | 155-2172 |
| Imaging rate | 15.5 Hz | ~15.5 Hz |
| Reward omission rate | 15.3% | ~15% |
| Reward zone distribution | A:34.3%, B:32.8%, C:32.9% | ~equal |
| Environment distribution | ENV1:51.0%, ENV2:49.0% | ~equal |
| Interneurons excluded | 119 total | 0.42 +/- 0.85% |

## Decoder Performance (Full Data, 152 sessions)

| Output | Train Balanced Acc | Val Balanced Acc | Chance |
|--------|-------------------|-----------------|--------|
| distance_to_reward_zone | 0.377 | 0.345 | 0.143 |
| absolute_position | 0.541 | 0.517 | 0.200 |
| speed | 0.444 | 0.415 | 0.200 |
| lick | 0.651 | 0.641 | 0.500 |
| reward_zone_location | 0.842 | 0.800 | 0.333 |
| reward_outcome | 0.556 | 0.525 | 0.500 |

## Decoder Performance (Sample Data, 5 sessions)

| Output | Train Balanced Acc | Val Balanced Acc | Chance |
|--------|-------------------|-----------------|--------|
| distance_to_reward_zone | 0.507 | 0.387 | 0.143 |
| absolute_position | 0.561 | 0.523 | 0.200 |
| speed | 0.425 | 0.315 | 0.200 |
| lick | 0.713 | 0.636 | 0.500 |
| reward_zone_location | 0.995 | 0.993 | 0.333 |
| reward_outcome | 0.668 | 0.428 | 0.500 |

## Time Bin Size
- 64.48 ms (1/15.5078125 Hz)
- Matches the 2-photon imaging frame rate

## Key Decisions
1. Used deconvolved activity (not dF/F) as neural data, matching the paper's use of deconvolved activity for most analyses
2. Multi-plane data (m17, m18) concatenated across planes, following the paper's approach of pooling planes for all analyses except Extended Data Fig. 7
3. Trial data includes only the on-track period (trial_start to teleport), excluding teleport zone
4. No speed threshold applied to neural data for the decoder (the paper applies speed < 2 cm/s threshold for spatial analyses, but the decoder benefits from all timepoints)
5. Lick data binarized (0/1) after sensor error correction
