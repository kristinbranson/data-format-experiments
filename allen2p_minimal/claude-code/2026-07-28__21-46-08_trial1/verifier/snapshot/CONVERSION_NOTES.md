# Conversion Notes

## Data Selection Decisions

### Session Selection
- **Active behavior only**: Excluded passive viewing sessions (behavior_type == 'passive_viewing') since mice are not performing the task.
- **Familiar images only**: Following the paper's neural analysis approach (experience_level == 'Familiar'). The paper states: "For neural analysis we used neurons recorded during familiar image set presentations."
- **All equipment types**: Included both Scientifica single-plane (~31 Hz) and Multiscope multi-plane (~11 Hz) recordings. While the paper focused on Multiscope data, we include both for a larger dataset, with appropriate resampling to a common time bin.
- **All cre lines**: Included Slc17a7 (excitatory), Sst, and Vip cell types. The paper analyzed all three.
- **Result**: 110 experiments from 38 mice.

### Trial Selection
- **Go trials**: Trials where the image changed (change_time defined). These yield hit or miss outcomes.
- **Catch trials**: Trials where no change occurred (sham change). These yield false_alarm or correct_reject outcomes.
- **Excluded**: Aborted trials (premature licking before change), Auto-rewarded trials (free rewards).
- **Minimum trials**: Sessions with fewer than 2 valid trials were excluded.

### Sanity Checks Against Reference Papers
- **8 images**: The paper states "Each session included 8 images." We confirmed 8 unique image names: im061, im062, im063, im065, im066, im069, im077, im085.
- **Trial structure**: Trials consist of flashed images (250ms on, 500ms gray = 750ms per flash), matching the paper description.
- **Trial length**: Mean trial length of ~8.2 seconds is consistent with the expected range from the truncated exponential change-time distribution (2.25-8.25s mean ~4.2s, plus post-change time).
- **Trial outcomes**: ~28.4% hits, ~59.0% misses, ~1.9% false alarms, ~10.7% correct rejections. The high miss rate is consistent with including all sessions regardless of engagement level.
- **Image change fraction**: ~8.2% of timepoints are marked as image change, which is consistent with changes occurring once per trial and trials containing ~12 stimulus presentations on average.
- **Image identity distribution**: All 8 images appear with approximately equal frequency (~12.3-12.8%), consistent with uniform sampling of image transitions.

## Neural Data Processing

### Signal Type
- Used **detected calcium events** (not raw dF/F), matching the paper: "we used the detected calcium events as described in Garrett et al."
- Events are deconvolved from fluorescence traces using the method in Giovannucci et al. 2019, providing cleaner signals with ~200ms resolution.
- Events exclude prolonged calcium transients that could contaminate responses to subsequent stimuli.

### ROI Filtering
- ROIs are pre-filtered by the AllenSDK processing pipeline (see whitepaper Section F: ROI Filtering).
- Only valid cell bodies are included (excludes dendrites, duplicates, edge artifacts, unions).

## Temporal Alignment

### Common Time Bin
- All data resampled to **93.2 ms** bins (~10.7 Hz), matching the Multiscope frame rate.
- Single-plane Scientifica data (~31 Hz, ~32.3 ms bins) downsampled by factor of 3 via averaging.
- Neural data: averaged over 3 frames for downsampling.
- Categorical data (image identity): mode of 3 frames for downsampling.
- Continuous data (running speed, pupil): averaged over 3 frames.

### Signal Interpolation
- Running speed: linearly interpolated from ~60 Hz analog input to ophys timestamps.
- Pupil diameter (width): linearly interpolated from ~30 Hz eye tracking to ophys timestamps.
- NaN handling: Running speed NaN filled with 0; Pupil NaN filled via nearest-neighbor interpolation.

## Output Variable Details

### Image Identity (output 0)
- 8 categories corresponding to the 8 natural scene images.
- At each timepoint, assigned the most recently presented image.
- During gray screen intervals, the last shown image identity persists.
- Omitted stimuli retain the previous image identity.

### Image Change (output 1)
- Binary signal: 1 during the 750ms following a change in image identity, 0 otherwise.
- Only actual changes are marked (not sham changes in catch trials).

### Running Speed (output 2)
- Discretized into 5 equal-percentile bins using global percentile boundaries computed across all sessions.
- Percentile boundaries: [0.013, 1.53, 17.5, 33.9] cm/s.
- Uses the filtered running speed from the SDK (10 Hz lowpass Butterworth filtered).

### Pupil Diameter (output 3)
- Uses pupil_width from eye tracking data as the diameter measure.
- Discretized into 5 equal-percentile bins using global percentile boundaries.
- Percentile boundaries: [36.8, 41.2, 45.8, 53.0] pixels.
- NaN values (blink artifacts, tracking failures) filled via nearest valid value interpolation.

### Trial Outcome (output 4)
- 4 categories: hit (0), miss (1), false_alarm (2), correct_reject (3).
- Static per trial (same value across all timepoints within a trial).
- Represented as time-varying for format consistency.

## Validation Results

### Format Verification
- All data passes format verification with no errors.
- Warnings: Some trials have all-zero neural data (sparse calcium events, especially in sessions with few neurons).

### Decoder Performance (full dataset, 70/30 train/test split)
- Image identity: 0.209 balanced accuracy (chance: 0.125) - **1.67x above chance**
- Image change: 0.527 balanced accuracy (chance: 0.500) - **above chance**
- Running speed: 0.267 balanced accuracy (chance: 0.200) - **1.34x above chance**
- Pupil diameter: 0.278 balanced accuracy (chance: 0.200) - **1.39x above chance**
- Trial outcome: 0.267 balanced accuracy (chance: 0.250) - **above chance**

All outputs show above-chance decoding from neural activity, confirming that the neural data contains meaningful information about behavioral and stimulus variables.

### Dataset Size Consistency
- 110 sessions from 38 mice with 27,643 total trials.
- Brain regions: VISp (14,791 neurons), VISl (104 neurons).
- The paper reports "8,619 excitatory cells (21 sessions, 9 mice), 470 Sst cells (15 sessions, 6 mice), and 1,239 Vip cells (21 sessions, 9 mice)" for the Multiscope familiar-image subset. Our dataset includes additional single-plane experiments, so the total is larger.
