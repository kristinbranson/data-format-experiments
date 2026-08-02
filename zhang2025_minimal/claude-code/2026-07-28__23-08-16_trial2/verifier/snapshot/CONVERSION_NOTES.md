# Conversion Notes

## Data Source
- IBL Brain Wide Map (BWM) dataset
- 699 probe insertions across 459 sessions from 139 subjects
- Reference: IBL et al. (2024), Zhang et al. (2025)

## Processing Decisions

### 1. Spike Sorting and Neuron Selection
- **Decision**: Use ALL spike-sorted clusters (no quality control filtering)
- **Justification**: The reference code's `prepare_data` function calls `load_spiking_data` without the `qc` parameter, defaulting to `qc=None` which returns all clusters. The BWM paper mentions quality filtering at the analysis level (region-specific minimum 5 neurons), but the reference code for this decoder task uses all clusters.
- **Note**: The BWM paper describes 75,708 well-isolated neurons out of 621,733 units. Using all clusters means we include multi-unit activity, which is appropriate for population decoding.

### 2. Temporal Alignment
- **Decision**: Align to stimulus onset (stimOn_times), window [-0.5, 1.5]s
- **Justification**: Methods paper states: "For choice, we align trials to the stimulus onset, considering neural activity from 0.5 s before to 1.5 s post-onset." The reference code uses `params = {'align_time': 'stimOn_times', 'time_window': (-.5, 1.5)}`.
- **Note**: The decoder task specifies "Temporally align based on stimulus onset" which matches this choice.

### 3. Bin Size
- **Decision**: 20ms non-overlapping bins -> 100 time steps per 2s trial
- **Justification**: Reference code uses `binsize=0.02`. Methods paper states: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps."
- **Note**: The methods paper mentions 50ms bins for choice/prior decoding specifically, but the reference code uses 20ms universally. We follow the reference code since we decode all variables simultaneously.

### 4. Trial Filtering
- **Decision**: Exclude trials with RT < 0.08s or > 2.0s, NaN key events, no-choice
- **Justification**: Matches `load_trials_and_mask` defaults in reference code. BWM paper states: "trials were excluded if...the first wheel-movement time were outside the range of 0.08-2.00 s"
- **Events requiring non-NaN**: stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType

### 5. Probe Merging
- **Decision**: Merge all probes within a session
- **Justification**: Reference code's `prepare_data` calls `merge_probes` for multi-probe sessions. BWM paper: "Although a session may have included multiple probe insertions, we did not perform decoding on these probes separately because they are not independent."

### 6. Brain Region Mapping
- **Decision**: Use Beryl mapping via `iblatlas.regions.BrainRegions`
- **Justification**: Reference code calls `brainreg.acronym2acronym(..., mapping='Beryl')`

### 7. Behavioral Variables

#### Wheel Speed
- **Decision**: Use absolute velocity (speed = |velocity|) from `SessionLoader.load_wheel()`
- **Justification**: Reference code's `load_target_behavior` for 'wheel-speed' computes `np.abs(sess_loader.wheel['velocity'].to_numpy())`
- **Discretization**: 3 bins using global tercile thresholds

#### Whisker Motion Energy
- **Decision**: Use left camera whisker motion energy (fallback to right)
- **Justification**: Reference code tries left first: `load_target_behavior(one, eid, 'left-whisker-motion-energy')`, falls back to right if unavailable
- **Discretization**: 3 bins using global tercile thresholds

#### Choice
- **Decision**: Binary, left=0, right=1
- **Justification**: Decoder task specifies "left = 0, right = 1". In IBL data, choice=-1 is left, choice=1 is right.

#### Prior Probability
- **Decision**: 3-class categorical: 0.2->0, 0.5->1, 0.8->2
- **Justification**: Decoder task specifies "0.2 -> 0, 0.5 -> 1, 0.8 -> 2"

### 8. Decoder Inputs
- **Time since stimulus onset**: Continuous time variable, same for all trials (linspace from -0.48 to 1.5, matching bin centers)
- **Trial number in block**: Count of trial within its block of constant probabilityLeft, starting from 1

### 9. Behavior Interpolation
- **Decision**: Linear interpolation to bin centers at `np.linspace(t_start + binsize, t_end, n_bins)`
- **Justification**: Matches reference code's `get_behavior_per_interval` function which computes `x_interp = np.linspace(interval_begs[interval_idx] + binsize, interval_ends[interval_idx], n_bins)`

### 10. Output Format
- **Decision**: All outputs are time-varying (shape (4, 100) per trial)
- **Justification**: Per-trial variables (choice, prior) are replicated across time bins. Time-varying variables (wheel speed, whisker ME) are naturally per-timepoint. Making all time-varying enables the decoder to handle them uniformly.

## Validation Results

### Sample Data (5 sessions)
- Format validation: PASSED (no errors, no warnings)
- 5 sessions, 1380 trials, 2 subjects
- 100 time bins per trial (correct)
- Decoder training: above-chance on all 4 outputs
  - Choice: ~0.53 balanced accuracy (chance: 0.50)
  - Prior: ~0.68 balanced accuracy (chance: 0.33)
  - Wheel speed: ~0.57 balanced accuracy (chance: 0.33)
  - Whisker ME: ~0.78 balanced accuracy (chance: 0.33)

### Full Data (444 sessions)
- Format validation: PASSED (4 minor warnings: 4 trials with all-zero neural data)
- 444 sessions, 190,151 total trials, 139 subjects
- 15 sessions skipped (missing whisker motion energy data)
- 100 time bins per trial (correct)
- 280 brain regions (Beryl mapping)
- 599,865 total neurons across all sessions
- Mean 1,351 neurons per session (min: 135, max: 3,140)
- Mean 428 trials per session (min: 127, max: 1,445)
- Decoder training results (200 epochs, CPU):
  - Choice: 0.5355 balanced accuracy (chance: 0.5000) - above chance
  - Prior: 0.5897 balanced accuracy (chance: 0.3333) - well above chance
  - Wheel speed: 0.5665 balanced accuracy (chance: 0.3333) - well above chance
  - Whisker ME: 0.7201 balanced accuracy (chance: 0.3333) - strongly above chance

## Sanity Checks

### Consistency with Reference Papers
1. **Number of sessions**: 444 sessions (paper: 433 used in Zhang et al., 459 in BWM release). 15 skipped due to missing whisker motion energy data. Very close to paper's 433.
2. **Number of subjects**: 139 (exactly matches paper: "We trained 139 mice").
3. **Trial counts**: Mean 428 valid trials per session, consistent with paper reporting "at least 400 trials" before filtering. Total 190,151 trials.
4. **Time bins**: 100 per trial = 2s / 20ms, exactly matching paper: "divided into 20-ms bins, producing T = 100 time steps."
5. **Choice distribution**: 49.2%/50.8% left/right, consistent with balanced task design.
6. **Prior distribution**: 41.8%/14.0%/44.2% for 0.2/0.5/0.8, consistent with block structure (0.5 only in first 90 unbiased trials).
7. **Brain regions**: 280 regions using Beryl mapping (paper: 270 regions). Close match.
8. **Total neurons**: 599,865 (paper: 621,733 units). Close match; difference from skipped sessions.
9. **Wheel speed and whisker ME**: Tercile-discretized, producing balanced 33.3%/33.3%/33.3% distributions as expected.

### Data Quality
- No NaN or Inf values in neural, input, or output arrays
- Consistent dimensions across all trials within each session
- Only 4 trials with all-zero neural data (out of 190,151 total)
- Behavioral discretization produces perfectly balanced bins (terciles)
- All decoder outputs achieve above-chance performance
