# Conversion Notes: Allen Visual Behavior 2P Dataset

## Data Source

Allen Institute Visual Behavior 2-Photon Calcium Imaging Dataset, accessed via NWB files and the AllenSDK.

References:
- Allen Brain Observatory: Visual Behavior 2P Technical Whitepaper (whitepaper.pdf)
- "Behavioral strategy shapes activation of the Vip-Sst disinhibitory circuit in visual cortex" (paper.pdf)

## Data Selection Decisions

### Experiments Included
- **All active behavior ophys experiments** with available NWB files (202 total, 200 successfully processed)
- Both VisualBehavior (single-plane, 31 Hz) and VisualBehaviorMultiscope (multi-plane, 11 Hz) projects
- 2 experiments skipped due to having fewer than 5 neurons (MIN_NEURONS=5)
- No passive viewing sessions included

### Rationale
- The paper ("Behavioral strategy shapes...") focused on familiar active sessions from the multiplane rig, but only 22 such experiments from 1 mouse were available as NWB files
- To build a meaningful decoder with sufficient data diversity, all active behavior experiments were included
- This provides 200 sessions from 38 mice across 3 cell types (Slc17a7/excitatory, Sst, Vip)

### Trial Filtering
- Included: Go trials and Catch trials (as specified in task instructions)
- Excluded: Aborted trials (premature licking before stimulus change) and Auto-rewarded trials (free reward trials)
- This matches the standard analysis approach described in the whitepaper where aborted trials are excluded from performance metrics

## Neural Data

### Data Type: dF/F Traces
- Used `dff_traces` (normalized change in fluorescence) from the AllenSDK
- The paper used detected calcium events (`events`), but these are 99.7% sparse (zero-valued) which makes them unsuitable for time-bin-based decoding
- dF/F provides continuous neural activity suitable for the decoder framework

### Processing
- dF/F traces are already processed by the Allen pipeline: motion corrected, cell segmented, ROI filtered, demixed, neuropil subtracted, baseline normalized, and detrended (as described in whitepaper Section F)
- No additional neural processing was applied

## Temporal Alignment

### Common Time Bin Size: 90.9 ms (~11 Hz)
- Data was resampled to a common bin size of 1/11 seconds (~90.9 ms)
- This matches the frame rate of the multiplane mesoscope (11 Hz per plane)
- For single-plane (31 Hz) experiments, neural data was averaged within each time bin
- Behavior data (running speed, pupil diameter) was linearly interpolated to bin centers

### Trial Segmentation
- Each trial defined by `start_time` to `stop_time` from the trials table
- Time bins created from trial start to trial end at the common bin size
- Minimum 3 time bins per trial required (MIN_TRIAL_FRAMES=3)

## Decoder Outputs

### 1. Image Identity (time-varying, categorical)
- 16 unique natural images across all sessions: im000, im031, im035, im045, im054, im061, im062, im063, im065, im066, im069, im073, im075, im077, im085, im106
- At each time bin, the most recently presented non-omitted image is recorded
- During grey screen periods (inter-stimulus intervals), the last shown image persists
- The task description says "of the image presented during the non-grey screen" - since trials only occur during change detection blocks, images are always being presented

### 2. Image Change (time-varying, binary)
- Value of 1 at time bins immediately after a change in image identity (within 2 bin widths of change onset)
- Value of 0 otherwise
- Changes identified from `is_change` flag in stimulus_presentations table

### 3. Running Speed (time-varying, 5 bins)
- Running speed from wheel encoder, already processed by Allen pipeline (10 Hz low-pass filtered)
- Interpolated to ophys time bins
- Discretized into 5 equal percentile bins across all sessions
- Bin edges: [-inf, -0.002, 0.353, 16.35, 33.68, inf] cm/s

### 4. Pupil Diameter (time-varying, 5 bins)
- Computed as mean of pupil_width and pupil_height from eye tracking ellipse fits
- Frames marked as likely_blink set to NaN and interpolated
- Sessions without eye tracking data filled with global median
- Discretized into 5 equal percentile bins across all sessions
- Bin edges: [-inf, 35.47, 40.32, 44.75, 50.74, inf] pixels

### 5. Trial Outcome (static per trial)
- 4 categories: hit (correct Go response), miss (no response to Go), false_alarm (response to Catch), correct_reject (no response to Catch)
- Assigned from trial flags in the trials table
- Broadcast to all time bins within the trial

## Dataset Statistics

### Summary
- Sessions: 200
- Subjects: 38 mice
- Brain regions: VISp (29,282 neurons), VISl (154 neurons)
- Total trials: 51,557
- Total neurons: 29,436
- Time bin size: 90.9 ms
- Mean trial duration: 8.5 s (min 7.0 s, max 12.5 s)
- Mean neurons per session: 147.2 (min 5, max 666)
- Mean trials per session: 257.8 (min 39, max 409)

### Comparison with Paper
The paper reports 57 imaging sessions from 24 mice (familiar active multiscope only). Our dataset includes all available active behavior sessions (200 sessions, 38 mice), which is a superset covering both single-plane and multi-plane experiments across familiar and novel image sets.

The paper reports 8,619 excitatory cells, 470 Sst cells, and 1,239 Vip cells from the multiscope dataset. Our total of 29,436 neurons includes data from both single-plane (higher neuron counts per FOV) and multi-plane experiments.

### Trial Outcome Distribution
- Hit: ~45% of trials
- Miss: ~42% of trials
- False alarm: ~2% of trials
- Correct reject: ~12% of trials

This is consistent with the whitepaper's description of behavioral performance, where mice have variable hit rates and the catch trial probability is ~12.5% with the matrix sampling algorithm.

### Image Presentation Statistics
- 16 unique images across both image sets A and B
- Each image set contains 8 images
- Images presented for 250 ms with 500 ms grey screen intervals (750 ms cycle)
- ~2% of time bins contain image changes, consistent with the task design

## Sanity Checks

1. **Neuron count consistency**: All trials within a session have the same number of neurons (PASS)
2. **Output dimension consistency**: All output arrays have 5 dimensions matching n_timepoints (PASS)
3. **Brain region index validity**: All brain_region_idx arrays match neuron counts (PASS)
4. **Output value ranges**: All categorical values within valid ranges (PASS)
5. **Subject index validity**: All subject indices map to valid subjects (PASS)
6. **Data format validation**: train_decoder.py --verify-only reports no errors or warnings (PASS)
7. **Decoder performance**: All outputs decode above chance level (PASS)

## Decoder Results

### Validation Balanced Accuracy (chance in parentheses)
- Image identity: 0.394 (0.063) - 6.3x chance
- Image change: 0.634 (0.500) - 1.3x chance
- Running speed: 0.379 (0.200) - 1.9x chance
- Pupil diameter: 0.426 (0.200) - 2.1x chance
- Trial outcome: 0.300 (0.250) - 1.2x chance

All outputs significantly above chance, confirming that neural activity carries information about stimulus, behavior, and task variables.

## Known Limitations

1. Some sessions have very few neurons (minimum 5), which may limit decoder performance for those sessions
2. 3 sessions had eye tracking errors - pupil data filled with global median for those sessions
3. The multiplane experiments (mouse 457841) produce many sessions with few neurons per plane, since each imaging plane is a separate experiment
4. The common bin size of 90.9 ms means some temporal resolution is lost for the 31 Hz single-plane data
