# Conversion Notes: Allen Visual Behavior 2P Dataset

## Data Source
- Allen Brain Observatory Visual Behavior 2P dataset v1.1.0
- 284 NWB files in `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/`
- Metadata tables in `data/visual-behavior-ophys-1.1.0/project_metadata/`

## Data Selection Decisions

### Session Selection
- **Active behavior only**: Filtered `passive == False` from experiment table (202 of 284 NWB files)
- **All project codes included**: VisualBehavior (single-plane, ~31 Hz) and VisualBehaviorMultiscope (multi-plane, ~11 Hz)
- **All cre lines included**: Slc17a7-IRES2-Cre (excitatory), Sst-IRES-Cre (Sst inhibitory), Vip-IRES-Cre (Vip inhibitory)
- **All brain regions included**: VISp (primary visual cortex), VISl (lateromedial visual area)

### Trial Selection
- **Included**: Go trials and Catch trials
- **Excluded**: Aborted trials (mouse licked before change) and Auto-rewarded trials (free rewards)
- This matches the task specification and standard analysis practice from the reference papers

### Neural Data
- **Used detected calcium events** (`processing/ophys/event_detection/data`) rather than raw dF/F traces
- This matches the paper: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces"
- Only valid ROIs included (filtered by `valid_roi` flag in cell specimen table)

## Processing Pipeline

### 1. Behavioral Data Interpolation
- Running speed and pupil diameter are recorded at different rates than ophys
- Both are linearly interpolated to ophys timestamps using `np.interp`
- Pupil data during likely blinks (flagged in NWB) is treated as NaN before interpolation
- NaN pupil values are excluded from interpolation; valid values are used for interp

### 2. Image Identity Assignment
- For each ophys timepoint, the currently displayed image is determined from stimulus presentation intervals
- Images are displayed for ~250 ms with ~500 ms ISI (gray screen)
- During gray screen periods, the last shown image identity is carried forward
- Omitted stimuli are treated as gray screen (the image continues to be the last shown non-omitted image)
- Image indices map to sorted unique image names across all experiments

### 3. Image Change Detection
- Binary signal (0/1) marking timepoints during which a stimulus change occurred
- Derived from `is_change` field in stimulus presentations table
- Value is 1 during the change stimulus presentation, 0 otherwise

### 4. Running Speed Binning
- Global percentile bins computed across ALL sessions' running speed data (interpolated to ophys timestamps)
- 5 equal percentile bins (0-20%, 20-40%, 40-60%, 60-80%, 80-100%)
- Bin edges: [-23.98, 0.017, 1.83, 19.02, 35.06, 99.92] cm/s

### 5. Pupil Diameter Binning
- Pupil width from fitted ellipse used as diameter measure
- Blink periods excluded (set to NaN before interpolation)
- Global percentile bins computed across all valid pupil data
- 5 equal percentile bins
- Bin edges: [4.81, 36.81, 41.44, 46.17, 52.78, 252.21] pixels
- Sessions without pupil data: assigned median bin (bin 2)

### 6. Trial Outcome
- Determined from trial flags: hit, miss, false_alarm, correct_reject
- Static per trial (replicated across all timepoints for time-varying format)

### 7. Trial Segmentation
- Trials segmented using `start_time` and `stop_time` from NWB trials table
- Ophys frames within [start_time, stop_time) extracted for each trial
- Trials with < 2 ophys frames are excluded

## Temporal Alignment
- All data aligned to ophys imaging frame timestamps
- Ophys timestamps serve as the common temporal reference
- Running speed, pupil diameter, and stimulus information all interpolated/mapped to these timestamps
- Time bin size is the inter-frame interval (~32.26 ms for single-plane at ~31 Hz, ~93.2 ms for multi-plane at ~11 Hz)
- Metadata `time_bin_size` reports median across sessions (32.26 ms)

## Output Format
- Each experiment = 1 session in the output
- Neural: (n_neurons, n_timepoints) per trial, float32, calcium events
- Output: (5, n_timepoints) per trial, int64, categorical values
  - Row 0: image identity (0-15)
  - Row 1: image change (0-1)
  - Row 2: running speed bin (0-4)
  - Row 3: pupil diameter bin (0-4)
  - Row 4: trial outcome (0-3)
- Input: empty (no decoder inputs specified)

## Validation Results

### Dataset Statistics
- 202 sessions from 38 mice
- 51,992 trials total
- 29,444 neuron-sessions total
- Brain regions: VISl (17 sessions), VISp (185 sessions)

### Trial Outcome Distribution
- Hit: 15,936 (30.7%)
- Miss: 29,541 (56.8%)
- False alarm: 931 (1.8%)
- Correct reject: 5,584 (10.7%)

### Cell Counts by Cre Line
- Slc17a7-IRES2-Cre (excitatory): 27,895 neuron-sessions
- Sst-IRES-Cre (Sst inhibitory): 743 neuron-sessions
- Vip-IRES-Cre (Vip inhibitory): 806 neuron-sessions

### Sanity Checks (All Passed)
1. Dimension consistency across all sessions and trials
2. No NaN or Inf values in neural data
3. Output values within expected ranges
4. Subject and brain region indices valid
5. Image identity values span expected range (0-15 for 16 images)
6. Image change is binary (0 or 1)
7. Running speed and pupil diameter bins are 0-4
8. Trial outcomes are 0-3

### Decoder Performance (Validation Set)
All outputs decoded above chance:
- Image identity: 12.4% accuracy (chance: 6.25%, ~2x above chance)
- Image change: 57.8% accuracy (chance: 50%)
- Running speed: 23.6% accuracy (chance: 20%)
- Pupil diameter: 25.1% accuracy (chance: 20%)
- Trial outcome: 26.4% accuracy (chance: 25%)

### Comparison to Reference Paper
The paper reports:
- 8,619 excitatory cells, 470 Sst cells, 1,239 Vip cells (using familiar image sessions on multiplane rig)
- Our dataset includes all available experiments (both single-plane and multi-plane, all image sets)
- Our total counts differ because we include all active behavior sessions, not just familiar images on multiplane rig
- The 16 unique images (8 from image set A + 8 from image set B) match expectations

## Known Limitations
1. Multi-plane (Multiscope) experiments have lower frame rate (~11 Hz vs ~31 Hz), leading to fewer timepoints per trial
2. VISl experiments (from Multiscope) have very few neurons per session (4-23)
3. Some trials have all-zero neural data (sparse calcium events)
4. Pupil data quality varies across sessions; some sessions have no usable pupil data
5. Trial durations vary (7-13 seconds) due to variable inter-stimulus intervals
