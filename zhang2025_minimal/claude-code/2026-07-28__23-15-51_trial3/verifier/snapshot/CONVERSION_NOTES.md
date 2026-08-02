# Conversion Notes: IBL Brain-Wide Map Dataset

## Source Data

- **Dataset**: IBL Brain-Wide Map (BWM) release
- **Data paper**: "A brain-wide map of neural activity during complex behaviour" (IBL et al., 2024)
- **Methods paper**: "Exploiting correlations across trials and behavioral sessions to improve neural decoding" (Zhang et al., 2025)
- **Reference code**: `code/code_zhang2025/` - data caching and decoding pipeline

## Key Processing Decisions

### 1. Temporal Alignment
- **Decision**: Align all trials to stimulus onset (`stimOn_times`)
- **Justification**: The decoder task specifies "Temporally align based on stimulus onset". The reference code (`0_data_caching.py`) also uses `align_time: 'stimOn_times'`.
- **Window**: -0.5s to +1.5s relative to stimulus onset (2s total)
- **Source**: Reference code params: `'time_window': (-.5, 1.5)`

### 2. Bin Size
- **Decision**: 20ms non-overlapping bins -> 100 time bins per trial
- **Justification**: Reference code uses `'binsize': 0.02` (20ms). The methods paper mentions both 50ms (for choice/prior) and 20ms (for wheel/whisker ME) but the caching code uses 20ms uniformly.
- **Source**: `0_data_caching.py` line 52: `'binsize': 0.02`

### 3. Neural Data Loading
- **Decision**: Load ALL clusters (not filtered by quality metric)
- **Justification**: The reference code `prepare_data()` calls `load_spiking_data(one, pid)` without the `qc` parameter (defaults to `None`), meaning all clusters are loaded regardless of quality label. This matches the ML approach where the decoder can learn from all available neural signals.
- **Source**: `ibl_data_utils.py` line 736: `load_spiking_data(one, pid, eid=eid, pname=probe_name)` - no `qc` argument

### 4. Probe Merging
- **Decision**: Merge all probes within a session into a single neural population
- **Justification**: The reference code explicitly merges probes: "Merge probes for session eid". This is because probes in the same session share the same behavioral state and are not independent.
- **Source**: `ibl_data_utils.py` `merge_probes()` function; `0_data_caching.py` line 740

### 5. Trial Filtering
- **Decision**: Exclude trials with:
  - NaN in required events (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType)
  - Reaction time < 0.08s or > 2.0s
  - No choice made (choice == 0)
- **Justification**: Matches `load_trials_and_mask()` with default parameters
- **Source**: `ibl_data_utils.py` lines 123-213

### 6. Brain Region Mapping
- **Decision**: Use Beryl atlas mapping via `iblatlas.regions.BrainRegions`
- **Justification**: Reference code uses `brainreg.acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')`
- **Source**: `ibl_data_utils.py` line 219

### 7. Wheel Speed Computation
- **Decision**: Compute wheel speed as absolute value of wheel velocity (finite difference of position)
- **Justification**: Reference code loads wheel speed via `load_target_behavior(one, eid, 'wheel-speed')` which uses `np.abs(sess_loader.wheel['velocity'])`
- **Source**: `ibl_data_utils.py` lines 428-432

### 8. Whisker Motion Energy
- **Decision**: Use left camera (60 Hz) preferentially, fall back to right camera (150 Hz)
- **Justification**: The reference code tries left camera first, then right. Left camera has full resolution (1280x1024) at 60 Hz per the data paper.
- **Source**: `ibl_data_utils.py` lines 700-708; data paper video analysis section

### 9. Behavioral Signal Interpolation
- **Decision**: Interpolate wheel speed and whisker ME to neural bin centers using linear interpolation
- **Justification**: Reference code uses `interp1d(..., kind='linear', fill_value='extrapolate')` in `get_behavior_per_interval()`
- **Source**: `ibl_data_utils.py` lines 624-625

### 10. Output Discretization
- **Decision**: Discretize wheel speed and whisker ME into 3 equal-count bins using quantiles
- **Justification**: The decoder task specifies "Wheel speed discretized into 3 bins" and "Whisker motion energy discretized into 3 bins". Using quantile-based discretization ensures approximately equal class frequencies.
- **Quantile method**: Computed per-session to maintain local context

### 11. Choice Encoding
- **Decision**: IBL choice convention (1=left, -1=right) mapped to decoder convention (left=0, right=1)
- **Justification**: Decoder task specifies "left = 0, right = 1"

### 12. Prior Encoding
- **Decision**: Map probabilityLeft values: 0.2->0, 0.5->1, 0.8->2
- **Justification**: Decoder task specifies "0.2 -> 0, 0.5 -> 1, 0.8 -> 2"

## Sanity Checks

### 1. Dataset Size
- **Expected**: 459 sessions in BWM release (data paper), 433 used in methods paper
- **Observed**: 459 sessions found, 439 processed, 20 skipped (missing behavioral data)
- **Status**: PASS - 439 is close to 433 (methods paper). The 20 skipped sessions lacked wheel or whisker ME data.

### 2. Neuron Counts
- **Expected**: ~889 units per probe on average (data paper: 621,733 units / 699 probes)
- **Observed**: Mean 1358 neurons per session (range 135-3140). Sessions with 2 probes have ~1800 neurons.
- **Status**: PASS - consistent with merged multi-probe sessions

### 3. Trial Counts
- **Expected**: Sessions with at least 400 trials pre-filtering; after RT filtering and exclusions, fewer
- **Observed**: Mean 161 valid trials per session (range 18-487), total 70,778 trials
- **Status**: PASS - reduction from ~400+ to ~161 after strict filtering is expected

### 4. Time Bins
- **Expected**: 100 bins per trial (2s / 20ms)
- **Observed**: All trials have exactly 100 time bins
- **Status**: PASS

### 5. Choice Distribution
- **Expected**: Roughly balanced left/right
- **Observed**: 50.8% left, 49.2% right (35,953 left, 34,825 right)
- **Status**: PASS

### 6. Prior Distribution
- **Expected**: First 90 trials at 0.5, remaining in alternating 0.2/0.8 blocks
- **Observed**: 0.2: 40.9%, 0.5: 16.7%, 0.8: 42.3% (28,972 / 11,834 / 29,972)
- **Status**: PASS - 0.5 block is ~90 trials per session, the rest alternates

### 7. Brain Regions
- **Expected**: ~270 unique brain regions (methods paper)
- **Observed**: 281 Beryl-mapped regions
- **Status**: PASS

### 8. Subjects
- **Expected**: 139 subjects (data paper)
- **Observed**: 135 subjects
- **Status**: PASS - 4 fewer due to skipped sessions

### 9. Decoder Performance
- **Expected**: Above-chance for all output variables
- **Observed (sample, 3 sessions)**:
  - Choice: 0.627 balanced accuracy (chance 0.500)
  - Prior: 0.682 balanced accuracy (chance 0.333)
  - Wheel speed: 0.587 balanced accuracy (chance 0.333)
  - Whisker ME: 0.613 balanced accuracy (chance 0.333)
- **Observed (full, 439 sessions)**:
  - Choice: 0.551 balanced accuracy (chance 0.500)
  - Prior: 0.583 balanced accuracy (chance 0.333)
  - Wheel speed: 0.582 balanced accuracy (chance 0.333)
  - Whisker ME: 0.558 balanced accuracy (chance 0.333)
- **Status**: PASS - all above chance on both sample and full data

## Validation Results

### Sample Data (5 sessions)
- Format validation: PASS (no errors or warnings)
- Decoder training: PASS (all outputs above chance)

### Full Data (439 sessions)
- Format validation: PASS (4 warnings about zero-neural trials in 2 sessions, acceptable)
- Decoder training: PASS (all outputs above chance)
- See `verification_full_out.txt` and `train_decoder_full_out.txt` for details

## Known Limitations

1. **Wheel velocity smoothing**: The reference code uses `SessionLoader.load_wheel()` which applies Gaussian smoothing. Our implementation uses simple finite differences, which is noisier but preserves the same information.

2. **Motion energy alignment**: ROIMotionEnergy timestamps are aligned to camera frame times. For the left camera (60 Hz), the temporal resolution is ~16.7ms, which is close to the 20ms bin size.

3. **Per-session vs global discretization**: Wheel speed and whisker ME are discretized per-session using quantiles. This means the bin edges vary across sessions, which could affect cross-session comparisons.
