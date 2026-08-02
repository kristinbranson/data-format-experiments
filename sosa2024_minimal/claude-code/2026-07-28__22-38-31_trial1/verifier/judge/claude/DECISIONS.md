# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files by finding all `sub-*` directories in the data directory, then finding all `.nwb` files within each. Each NWB file is loaded using `pynwb.NWBHDF5IO`. Behavioral data is extracted from `nwb.processing['behavior']['BehavioralTimeSeries']` and neural data from `nwb.processing['ophys']`. Additionally, it loads fluorescence, neuropil, and deconvolved data, as well as cell segmentation info from `ImageSegmentation['PlaneSegmentation']`.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
io = NWBHDF5IO(filepath, 'r')
nwb = io.read()
bts = nwb.processing['behavior']['BehavioralTimeSeries']
ophys = nwb.processing['ophys']
seg = ophys['ImageSegmentation']['PlaneSegmentation']
```

iii. The AI's approach finds all NWB files in the standard BIDS-like directory structure. The CONVERSION_NOTES confirm 11 subjects and 152 sessions were found, matching the paper.

## 1-b. How are the data split into subjects?

i. Subjects correspond to subdirectories named `sub-*` in the data directory. The subject name is extracted by removing the `sub-` prefix.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_name = subj_dir.replace('sub-', '')
```

iii. The AI identified 11 subjects matching the paper's count of 11 switch mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All `.nwb` files within a subject directory are treated as separate sessions.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
```

iii. The session ID is extracted from the NWB file's `session_id` field.

## 1-d. How are the data split into trials?

i. Trial boundaries are determined from the `trial_start` and `teleport` behavioral signals. Trial starts are where `trial_start > 0`. For each start, the next teleport index (`teleport > 0`) is found as the trial end. Trials are only included if the `scanning` signal is active (equals 1) during the trial.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    ...
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
    return trial_starts, trial_ends
```

iii. The AI uses `trial_start` and `teleport` signals to define trial boundaries, similar to the reference, but adds a scanning check and uses a different teleport detection method (any `teleport > 0` rather than onset detection).

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 5 timepoints are skipped. Sessions with fewer than 3 trials are skipped entirely. Sessions with fewer than 5 curated cells are also skipped.

ii.
```python
if n_timepoints < 5:
    continue
...
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials")
    return None
...
if n_total_cells < 5:
    print(f"    Skipping: only {n_total_cells} curated cells")
    return None
```

iii. The minimum trial length threshold of 5 is much lower than the reference's 50. The session-level filters (min cells, min trials) are additional quality controls not in the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from the raw `Fluorescence` and `Neuropil` data stored in `ophys`, NOT from the pre-computed `Deconvolved` data. It computes its own dF/F from scratch using neuropil subtraction, baseline computation, and then deconvolves using the OASIS algorithm.

ii.
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
...
# For multi-plane:
fluor_planes = [np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys]
neuro_planes = [np.array(ophys['Neuropil'][k].data[:]) for k in plane_keys]
```

iii. The AI chose to reimplement the paper's preprocessing pipeline (neuropil subtraction, dF/F, OASIS deconvolution) rather than using the pre-computed deconvolved data available in the NWB files. The CONVERSION_NOTES describe this as "matching `preprocessing.py`".

## 2-b. How is the `neural` data processed?

i. The AI applies a multi-step processing pipeline:
1. Mask fluorescence to within-trial timepoints (NaN outside trials)
2. Neuropil subtraction: `F_corrected = F - 0.7 * F_neuropil`
3. Add back neuropil mean per trial
4. Maximin baseline with 300-sample window
5. `dF/F = (F - baseline) / |baseline|`
6. Smooth with 2-sample Gaussian kernel
7. OASIS deconvolution via suite2p
8. Replace NaNs with 0
Additionally, interneurons are filtered (see 2-c).

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
    ...
    baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
    baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
    dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
    ...
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
def deconvolve_oasis(dff_trial, tau=TAU, frame_rate=FRAME_RATE):
    from suite2p.extraction.dcnv import oasis
    events = oasis(dff_trial, 2000, tau, frame_rate)
```

iii. The AI attempted to replicate the paper's preprocessing pipeline. However, the pre-computed deconvolved data in the NWB files was already processed by the paper authors, making this recomputation unnecessary and potentially introducing subtle differences.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied:
1. `iscell[:, 0] == 1` from suite2p cell curation (selects valid ROIs)
2. Interneuron filtering: neurons with Pearson correlation between activity and running speed > 0.5 are removed

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
def filter_interneurons(dff, speed, iscell_mask):
    ...
    for idx in cell_indices:
        cell_data = dff[idx, valid]
        speed_data = speed[valid]
        if np.std(cell_data) > 0 and np.std(speed_data) > 0:
            corr = np.corrcoef(cell_data, speed_data)[0, 1]
            if corr > INTERNEURON_SPEED_CORR_THR:
                mask[idx] = False
    return mask
```

iii. The iscell filter is standard. The interneuron filter is mentioned in the paper but is NOT applied in the reference code. The AI reports ~3.8% removal rate vs the paper's 0.42%.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Data is extracted from `trial_start` to `trial_end` indices, and time is computed relative to the start timestamp.

ii.
```python
neural_trial = events[:, s:e].copy()
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The instructions specify alignment to start of trial, which matches this approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is ~64.5 ms (1000/15.5078125 Hz). No temporal rebinning is applied. The frame rate is hardcoded as `FRAME_RATE = 15.5078125`.

ii.
```python
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
```

iii. The AI hardcodes the frame rate rather than computing it from each NWB file's stored rate and number of planes (as the reference does with `nplanes/plane_data.rate*1000`).

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` timeseries timestamps in the behavioral data.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The AI uses position timestamps, while the reference uses `trial number` timestamps. Since all behavioral timeseries share the same timestamps, this produces the same result.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The initial timestamp of the trial is subtracted from all timestamps within the trial.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. Standard approach to compute relative time.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indices, so they are inherently aligned. When there is a mismatch in array lengths, both are cropped to the minimum length.

ii.
```python
n_behav = len(speed)
n_neural = F.shape[1]
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
```

iii. The approach is consistent with the reference's handling of length mismatches.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral timeseries.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])  # per trial
```

iii. The environment variable is binary (0 or 1), matching ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment value at the trial start is used as a per-trial scalar. If the value is negative, it defaults to 0 (ENV1).

ii.
```python
env_type = float(environment[s])  # per trial
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
```

iii. The AI uses only the first timepoint's value rather than broadcasting the per-trial value across all timepoints within the trial, then broadcasts it when creating the input array.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index, derived from the loop counter `t` over trials.

ii.
```python
trial_num = float(t)
```

iii. The trial number is a sequential 0-indexed counter.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is broadcast across all timepoints in the trial.

ii.
```python
input_trial[2, :] = trial_num
```

iii. The trial number is simply the sequential index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and the `reward_zone` signal. The `determine_trial_rewarded` function checks both reward delivery AND reward zone entry.

ii.
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
...
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. The AI requires both a reward timestamp in the trial AND reward zone entry, while the reference only checks for reward timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is looked up. For the first trial (t=0), the value is set to 1 (assumes rewarded before session start).

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
```

iii. The AI sets the first trial's previous outcome to 1 (rewarded), while the reference sets it to 0 (not rewarded). This is an unjustified assumption that differs from the reference.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` and `reward_zone` behavioral timeseries. The reward zone label (A, B, or C) for each trial is determined by computing the mean position where `reward_zone > 0` and matching to the closest zone center. Missing labels are filled from neighboring trials.

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_dist = dist
            best_zone = zone
    return best_zone
...
def determine_reward_zone_for_all_trials(...):
    # Fill missing from neighbors
    for i in range(n_trials):
        if labels[i] is None:
            for j in range(i + 1, n_trials):
                if labels[j] is not None:
                    labels[i] = labels[j]
                    break
```

iii. The reference uses a Viterbi algorithm to assign reward zone labels with temporal smoothing, which is more robust. The AI's mean-position approach is simpler but may be noisier.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, signed distance from position to the nearest edge of the reward zone is computed. Negative when before the zone, 0 when inside, positive when past.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start  # negative
    elif position > rz_end:
        return position - rz_end  # positive
    else:
        return 0.0  # inside zone
...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. The signed distance computation matches the reference's approach. The AI uses a per-element loop instead of vectorized operations.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit comparisons in a loop.

ii.
```python
def discretize_distance(distances):
    for i, d in enumerate(distances):
        if d < -50: result[i] = 0
        elif d < -10: result[i] = 1
        elif d < 0: result[i] = 2
        elif d == 0: result[i] = 3
        elif d <= 10: result[i] = 4
        elif d <= 50: result[i] = 5
        else: result[i] = 6
    return result
```

iii. The bins match the instructions. Minor boundary differences exist compared to the reference's `np.digitize` approach (e.g., exactly d=10 maps to bin 4 in AI vs bin 5 in reference).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data share the same time indices within each trial (s:e slicing), so no additional alignment is needed.

ii.
```python
pos_trial = position[s:e]
neural_trial = events[:, s:e].copy()
```

iii. Same indexing ensures alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral timeseries.

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. Direct extraction of position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped to [0, 450) before discretization.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
```

iii. The clipping ensures positions fall within the expected track range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each over [0, 450] using `np.floor(position / 90)`.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
    return result
```

iii. The AI uses 90 cm bins (0-90, 90-180, etc.) while the reference uses 100 cm bins with edges at [-inf, 50, 150, 250, 350, inf]. These produce different bin assignments for the same positions.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
pos_trial = position[s:e]
neural_trial = events[:, s:e]
```

iii. Verified by shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral timeseries.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_trial = lick_cumul[s:e].copy()
```

iii. Direct extraction of lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_binary = (lick_trial > 0).astype(float)
```

iii. Note: The CONVERSION_NOTES claim the lick signal was converted using diff (from cumulative to per-frame), but the actual code simply thresholds `lick > 0`, matching the reference. The documentation is inconsistent with the code.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
neural_trial = events[:, s:e]
```

iii. Shared indexing ensures alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` behavioral timeseries. Uses mean position where `reward_zone > 0` to determine zone label (A, B, or C).

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        ...
```

iii. Same reward zone definitions as reference (A=[80,130], B=[200,250], C=[320,370]).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Two-pass approach: (1) determine zone from positions where `reward_zone > 0` by matching mean position to closest zone center, (2) fill missing labels from neighboring trials. The label is mapped to integer (A=0, B=1, C=2).

ii.
```python
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The reference uses a Viterbi algorithm with Gaussian emission and temporal transition probabilities for more robust assignment. The AI's nearest-center approach is simpler but potentially noisier.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps and the `reward_zone` signal.

ii.
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
...
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. The AI checks both reward delivery AND reward zone entry. The reference only checks for reward timestamps within the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks if any reward timestamp falls within the trial's time range AND the mouse entered the reward zone. The result is binary (0 or 1), constant across all timepoints in the trial.

ii.
```python
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end))
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. The extra reward_zone check is redundant (rewards only occur in the reward zone), but shouldn't change results in practice.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Neural/behavior length mismatch**: Data is cropped to the minimum of the two lengths.
- **Short trials**: Trials with fewer than 5 timepoints are skipped.
- **Missing reward zone**: If `reward_zone` signal is never active in a trial, the zone label is inferred from neighboring trials.
- **NaN values in neural data**: Replaced with 0 using `np.nan_to_num`.
- **Sessions with too few cells (<5) or trials (<3)**: Skipped entirely.
- **Negative environment values**: Default to 0 (ENV1).

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
if n_timepoints < 5:
    continue
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
```

iii. The AI handles more edge cases than the reference, but uses a much lower minimum trial length threshold (5 vs 50).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files** - I/O bound, large files
2. **Computing dF/F from scratch** - neuropil subtraction, baseline computation, Gaussian smoothing for every neuron and trial
3. **OASIS deconvolution** - applied per trial for all neurons
4. **Interneuron filtering** - correlation computation for every neuron
5. **Saving the pickle file**

ii. N/A

iii. The dF/F computation and deconvolution are entirely unnecessary since the NWB files already contain pre-computed deconvolved data, making the AI's code significantly slower.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. The `distance_to_reward_zone` function iterates per-element instead of using vectorized operations (the reference uses vectorized numpy)
2. The `discretize_distance` function uses a per-element loop instead of `np.digitize`
3. The `discretize_speed` function uses a per-element loop instead of `np.digitize`
4. The interneuron filtering loop over individual neurons

ii.
```python
# Per-element loop (AI):
for i, d in enumerate(distances):
    if d < -50: result[i] = 0
    ...

# Vectorized (reference):
output_curr[0] = np.digitize(distance_to_reward_zone, distance_to_reward_zone_bins) - 1
```

iii. The per-element loops are functionally correct but much slower than vectorized numpy operations.

## 13-c. What processing does the code repeat multiple times?

i. Unlike the reference which has a survey step that loads data twice, the AI loads each NWB file only once. However, the AI computes dF/F from raw fluorescence when deconvolved data is already available in the NWB file, which is effectively redundant processing.

ii. N/A

iii. The AI's single-pass approach avoids the reference's duplicate loading, but introduces unnecessary recomputation of the neural preprocessing pipeline.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The entire dF/F computation pipeline (neuropil subtraction, baseline estimation, Gaussian smoothing, OASIS deconvolution) is unnecessary because the NWB files already contain pre-computed deconvolved neural activity. The AI also loads raw fluorescence and neuropil data that are not needed if using the pre-computed deconvolved data. The interneuron filtering step is also not in the reference and may be unnecessary for the decoder task.

ii.
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
# These are used only to recompute what's already in:
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
```

iii. The pre-computed deconvolved data was already processed by the paper authors using presumably the same pipeline.
