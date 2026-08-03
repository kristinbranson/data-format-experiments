# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads data from NWB files stored in the `data/` directory. It iterates through `sub-*` subdirectories, finds all `.nwb` files within each, and reads them using pynwb's `NWBHDF5IO`. Each NWB file contains both neural (ophys) and behavioral (BehavioralTimeSeries) data for one session.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
# ...
for subj_dir in subjects:
    subj_path = os.path.join(data_dir, subj_dir)
    nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
    for nwb_file in nwb_files:
        result = process_session(nwb_file, compute_own_dff=True)
```

```python
def load_nwb(filepath):
    from pynwb import NWBHDF5IO
    io = NWBHDF5IO(filepath, 'r')
    nwb = io.read()
    bts = nwb.processing['behavior']['BehavioralTimeSeries']
    ophys = nwb.processing['ophys']
    # ... extracts fluorescence, neuropil, deconvolved, behavioral timeseries
```

iii. The agent identified the NWB file format from exploring the data directory and used pynwb to load all relevant fields. The CONVERSION_NOTES document 152 total sessions across 11 subjects.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `sub-*` directory names. Each directory contains all sessions for one mouse. The subject ID is also confirmed from the NWB file metadata (`nwb.subject.subject_id`).

ii.
```python
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
subj_idx = all_subjects.index(subject_name)
```

iii. The agent lists 11 subjects (m3, m4, m7, m11-m15, m17-m19), matching the paper's 11 switch-condition mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted alphabetically by filename within each subject directory, then processed sequentially.

ii.
```python
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The agent confirmed 12-14 sessions per subject, totaling 152 sessions, matching the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport` signals in the NWB behavioral timeseries. For each trial start index, the next teleport index marks the trial end.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
```

iii. This matches the reference code's use of `sess.trial_start_inds` and `sess.teleport_inds` to define trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by several criteria:
- Scanning must be active (scanning == 1) during at least part of the trial
- Trial must have at least 5 timepoints
- Session must have at least 2 valid trials
- Session must have at least 5 curated cells
- Autoreward trials are NOT filtered

ii.
```python
if np.any(scanning[s:e] == 1):
    trial_starts.append(s)
    trial_ends.append(e)
# ...
if n_timepoints < 5:
    continue
# ...
if valid_trial_count < 2:
    return None
if n_total_cells < 5:
    return None
```

iii. The agent applied reasonable minimum quality controls. The scanning check ensures imaging data is available. No autoreward filtering was applied, which is consistent with the instructions not mentioning it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` and `Neuropil` signals from the NWB ophys processing module. For multi-plane recordings (m17, m18), data is concatenated across planes.

ii.
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
# Multi-plane:
fluor_planes = [np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys]
neuro_planes = [np.array(ophys['Neuropil'][k].data[:]) for k in plane_keys]
fluorescence = np.concatenate(fluor_planes, axis=1)
neuropil_data = np.concatenate(neuro_planes, axis=1)
```

iii. The agent chose to compute dF/F from scratch using the raw fluorescence and neuropil traces, following the reference preprocessing.py pipeline, rather than using the pre-computed deconvolved data in the NWB files.

## 2-b. How is the `neural` data processed?

i. The processing pipeline closely follows the reference `preprocessing.py`:
1. Mask fluorescence to within-trial timepoints (NaN outside trials)
2. Neuropil subtraction (coefficient = 0.7)
3. Per trial: add back neuropil mean so baseline is close to true fluorescence
4. Smooth with sigma=15 Gaussian for baseline calculation
5. Maximin baseline with 300-sample window (minimum filter then maximum filter)
6. dF/F = (F - baseline) / |baseline|
7. Smooth dF/F with 2-sample Gaussian kernel
8. OASIS deconvolution (suite2p's dcnv.oasis, tau=0.7, batch_size=2000)

ii.
```python
def compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, frame_rate=FRAME_RATE):
    # Mask to within-trial data only
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = F[:, start:end]
        f_neu_[:, start:end] = Fneu[:, start:end]
    # Neuropil subtraction
    f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
    # Per trial: add back neuropil mean, maximin baseline, dF/F
    for start, end in zip(trial_starts, trial_ends):
        f_[:, start:end] = f_[:, start:end] + NEUROPIL_COEF * np.nanmean(
            f_neu_[:, start:end], axis=1, keepdims=True)
        f_smooth = nansmooth(f_[:, start:end], 15, axis=1)
        baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
        baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
        dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
    # Smooth dF/F
    for start, end in zip(trial_starts, trial_ends):
        dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
```

```python
events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
# calls: oasis(dff_trial, 2000, tau, frame_rate)
```

iii. The agent documented this as matching the reference preprocessing.py code. The neuropil subtraction coefficient, baseline window, and smoothing parameters all match the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two levels of filtering:
1. Suite2p cell curation: `iscell[:, 0] == 1` selects manually curated ROIs
2. Interneuron filtering: neurons with Pearson correlation > 0.5 between dF/F and running speed are removed

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
# ...
def filter_interneurons(dff, speed, iscell_mask):
    mask = np.copy(iscell_mask)
    cell_indices = np.where(mask)[0]
    for idx in cell_indices:
        cell_data = dff[idx, valid]
        speed_data = speed[valid]
        corr = np.corrcoef(cell_data, speed_data)[0, 1]
        if corr > INTERNEURON_SPEED_CORR_THR:
            mask[idx] = False
```

iii. The agent notes a discrepancy: 3.8% of cells removed vs paper's 0.42% +/- 0.85%. This could be due to differences in dF/F computation details or speed signal processing. The agent accepts this discrepancy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data spans from `trial_start` index to `teleport` index, with the first timepoint corresponding to the start of the trial (entry to the linear track at 0 cm).

ii.
```python
neural_trial = events[:, s:e].copy()  # s = trial_start, e = trial_end
```

iii. The instructions specify "Temporally align based on start of the trial," which the agent follows by slicing neural data from trial start to trial end.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is ~64.48 ms, corresponding to the imaging frame rate of 15.5078125 Hz. No temporal rebinning is applied; each timepoint corresponds to one imaging frame.

ii.
```python
FRAME_RATE = 15.5078125  # Hz, from NWB files
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~64.5 ms
```

iii. The agent extracted the frame rate from the NWB files and used it directly as the time bin size. No rebinning was applied, preserving the native temporal resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` array in the NWB behavioral timeseries (specifically `position.timestamps`).

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
# ...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The timestamps represent the actual recording time of each frame, providing an accurate time axis.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at trial start is subtracted from all timestamps within that trial, yielding time relative to trial onset in seconds.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. Straightforward subtraction to get time from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Uses the same trial boundary indices (s:e) as the neural data, so time and neural data are inherently aligned frame-by-frame.

ii.
```python
input_trial[0, :] = time_from_start  # same s:e slice as neural_trial
```

iii. Since both neural and behavioral data come from the same NWB file with the same frame-level alignment, using the same indices ensures alignment.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` timeseries in the NWB behavioral data.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
# ...
env_type = float(environment[s])  # per trial, value at trial start
```

iii. The environment signal has values -1 (outside environments), 0 (ENV1), and 1 (ENV2).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The environment value is taken at the trial start index. Values less than 0 (i.e., -1) are defaulted to 0 (ENV1). The result is binary: 0 = ENV1, 1 = ENV2.

ii.
```python
env_type = float(environment[s])
if env_type < 0:
    env_type = 0.0  # default to ENV1 if unknown
```

iii. The -1 values represent timepoints outside of environment display (e.g., teleport zone). At trial start, the environment should be defined, so -1 shouldn't normally occur.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is derived from the loop index `t` iterating over valid trials within each session, NOT from the NWB `trial number` timeseries.

ii.
```python
trial_num = float(t)  # where t is from range(len(trial_starts))
```

iii. The agent uses a 0-indexed trial counter instead of the NWB's `trial number` field (which ranges from -1 to 79+). This provides a sequential count of valid trials.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Simple casting of the loop index to float. No additional processing. The value is broadcast across all timepoints in the trial.

ii.
```python
trial_num = float(t)
input_trial[2, :] = trial_num
```

iii. The agent treats trial number as a per-trial scalar broadcast to all timepoints.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward outcome of the previous trial, which itself is computed from `Reward` timestamps and the `reward_zone` behavioral signal.

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
```

iii. The agent computes reward outcomes for all trials first, then looks back one trial for the previous outcome.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trials after the first, uses the reward outcome of the immediately preceding trial (0 or 1). For the first trial of each session, defaults to 1.0 (assumes the previous trial was rewarded).

ii.
```python
if t == 0:
    prev_outcome = 1.0
else:
    prev_outcome = float(is_rewarded[t - 1])
```

iii. The default of 1.0 for the first trial is an arbitrary assumption. The agent does not justify this specific choice in the trajectory or notes.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` (animal's location on the track) and the reward zone boundaries determined from the `reward_zone` signal.

ii.
```python
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. The agent uses the animal's position relative to the active reward zone boundaries.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed from the current position to the nearest edge of the active reward zone:
- Before zone: position - rz_start (negative)
- Inside zone: 0
- After zone: position - rz_end (positive)

The reward zone for each trial is determined from positions where the `reward_zone` signal is active (> 0), matched to the closest of three zones: A=[80,130], B=[200,250], C=[320,370]. For omission trials where the signal is never active, the zone is inferred from neighboring trials.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0

REWARD_ZONES = {'A': [80, 130], 'B': [200, 250], 'C': [320, 370]}
```

iii. The zones A/B/C in the agent's code correspond to X/Y/Z in the reference behavior.py's `reward_zone_dict`, and match the actual reward zone positions in the NWB data.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins matching the instruction specification exactly:
- 0: < -50 cm, 1: -50 to -10 cm, 2: -10 to < 0 cm, 3: 0 cm (inside), 4: >0 to +10 cm, 5: +10 to +50 cm, 6: > +50 cm

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
```

iii. Bin boundaries match the instructions exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Uses the same trial boundary indices (s:e) as the neural data, so position-derived distance is frame-aligned with neural activity.

ii.
```python
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
output_trial[0, :] = dist_binned  # same n_timepoints as neural
```

iii. All behavioral and neural data share the same time indices within each trial.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` timeseries in the NWB behavioral data.

ii.
```python
pos_trial = position[s:e]
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
```

iii. Position represents the animal's location on the 450 cm linear track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 449.999] cm range, then discretized into 5 equal bins.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```

iii. Clipping handles positions outside the track range (e.g., negative positions during teleport). The NWB position data ranges from -500 to ~450 cm.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins of 90 cm each: [0-90), [90-180), [180-270), [270-360), [360-450].

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. Matches the instruction's "5 equal-sized bins" over the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same trial boundary indices as neural data, inherently frame-aligned.

ii.
```python
output_trial[1, :] = pos_binned  # same s:e slice
```

iii. Aligned by shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` timeseries in the NWB behavioral data.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
# ...
lick_trial = lick_cumul[s:e].copy()
```

iii. The NWB lick signal contains per-frame lick counts (values 0-6), representing the number of licks detected in each imaging frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The per-frame lick count is converted to binary: any frame with lick count > 0 is marked as 1, otherwise 0. The code comment says "convert cumulative to binary" but the data is actually per-frame lick counts, not cumulative. The operation `(lick_trial > 0)` correctly produces a binary lick signal regardless.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
output_trial[3, :] = lick_binary.astype(int)
```

iii. The CONVERSION_NOTES incorrectly describe the lick data as "cumulative lick count," but the actual NWB data stores per-frame lick counts. The code operation is correct for either interpretation: `> 0` correctly converts both cumulative and per-frame counts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial boundary indices as neural data, inherently frame-aligned.

ii.
```python
output_trial[3, :] = lick_binary.astype(int)
```

iii. Aligned by shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from `reward_zone` and `position` timeseries in the NWB data. The `reward_zone` signal indicates when the animal is in the reward zone, and the position at those times identifies which zone (A, B, or C).

ii.
```python
def determine_reward_zone_label(position, rz_signal, trial_start, trial_end):
    rz_positions = pos_trial[rz_trial > 0]
    mean_rz_pos = np.mean(rz_positions)
    for zone, (start, end) in REWARD_ZONES.items():
        zone_center = (start + end) / 2
        dist = abs(mean_rz_pos - zone_center)
        if dist < best_dist:
            best_zone = zone
```

iii. The agent matches reward zone positions to the closest of three predefined zones.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. For each trial, positions where `reward_zone > 0` are averaged and matched to the closest zone center. For omission trials (no rz activation), the zone is inferred from neighboring trials. The result is encoded as 0=A, 1=B, 2=C.

ii.
```python
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
# Gap filling:
for i in range(n_trials):
    if labels[i] is None:
        for j in range(i + 1, n_trials):
            if labels[j] is not None:
                labels[i] = labels[j]; break
# Encoding:
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The gap-filling approach is reasonable since the reward zone doesn't change within a session (only across sessions on switch days).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward` timestamps and the `reward_zone` signal in the NWB data.

ii.
```python
'reward_timestamps': np.array(bts.time_series['Reward'].timestamps[:]),
'reward_data': np.array(bts.time_series['Reward'].data[:]),
```

iii. Reward delivery events are stored as timestamped events in the NWB file.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is marked as rewarded (1) if a reward timestamp falls within the trial's time window AND the mouse entered the reward zone during the trial. Otherwise marked as not rewarded (0).

ii.
```python
def determine_trial_rewarded(position, rz_signal, reward_timestamps, timestamps,
                              trial_start, trial_end):
    t_start = timestamps[trial_start]
    t_end = timestamps[min(trial_end, len(timestamps) - 1)]
    reward_in_trial = np.any(
        (reward_timestamps >= t_start) & (reward_timestamps <= t_end))
    entered_rz = np.any(rz_trial > 0)
    return int(reward_in_trial and entered_rz)
```

iii. This matches the reference behavior.py's `get_trial_types`: `isreward = (np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)) * 1`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies for handling data issues:
1. **Length mismatches**: When neural and behavioral timeseries have different lengths (off-by-one in multi-plane), truncate both to the minimum length
2. **Missing reward zones**: Inferred from neighboring trials (look forward, then backward)
3. **Unknown environment**: Values < 0 default to ENV1 (0)
4. **NaNs in neural data**: Replaced with 0 for the decoder
5. **Short trials**: Skipped if < 5 timepoints
6. **Deconvolution failures**: Falls back to ReLU of dF/F

ii.
```python
# Length mismatch
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    # ... truncate all signals

# NaN handling
neural_trial = np.nan_to_num(neural_trial, nan=0.0)

# Deconvolution fallback
except Exception as e:
    events[:, start:end] = np.maximum(trial_dff_clean, 0)
```

iii. The agent documented these decisions in CONVERSION_NOTES.md. The approaches are reasonable and prevent data loss while maintaining consistency.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **NWB file loading**: Reading large NWB files from disk with pynwb (I/O bound)
2. **dF/F computation**: Computing dF/F for ALL neurons (including non-curated ones) with smoothing, baseline calculation, and per-trial processing
3. **OASIS deconvolution**: Running the OASIS algorithm per trial per neuron, called via suite2p

ii.
```python
# Full dF/F on all neurons
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, ...)
# Then OASIS per trial
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    events_trial = deconvolve_oasis(trial_dff_clean, ...)
```

iii. These are inherently compute-intensive operations in the neural processing pipeline.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops use Python iteration over individual elements where numpy vectorization would be faster:
1. `discretize_distance`: Loops over each distance value instead of using `np.digitize` or `np.searchsorted`
2. `discretize_speed`: Same element-wise loop pattern
3. `distance_to_reward_zone`: Called per-element in a list comprehension instead of vectorized
4. `filter_interneurons`: Loops over individual neurons for correlation computation

ii.
```python
# Element-wise loop (could use np.digitize)
def discretize_distance(distances):
    for i, d in enumerate(distances):
        if d < -50: result[i] = 0
        elif d < -10: result[i] = 1
        # ...

# Per-element list comprehension (could be vectorized with np.where)
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. These loops are not performance-critical since they operate on relatively small arrays (per-trial behavioral data), but vectorization would improve style and efficiency.

## 13-c. What processing does the code repeat multiple times?

i. The code iterates over trial boundaries in multiple separate loops:
1. First loop: Mask fluorescence to within-trial data
2. Second loop: Compute baseline and dF/F per trial
3. Third loop: Smooth dF/F per trial
4. Fourth loop: Deconvolve per trial

Additionally, the reward zone is determined per-trial in a first pass, then gap-filled in a second pass.

ii.
```python
# Loop 1: mask
for start, end in zip(trial_starts, trial_ends):
    f_[:, start:end] = F[:, start:end]
# Loop 2: baseline + dF/F
for start, end in zip(trial_starts, trial_ends):
    # baseline + dff computation
# Loop 3: smooth
for start, end in zip(trial_starts, trial_ends):
    dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
```

iii. These loops could potentially be merged, though the reference code also uses separate loops for clarity.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary processing is:
1. **dF/F computed for ALL neurons** including non-curated cells (iscell=False). The `compute_dff_per_trial` operates on the full F matrix before `iscell` filtering is applied. Only the curated subset is ultimately used.
2. **dF/F computed for interneurons** that are later filtered out. The full dF/F is needed for the interneuron correlation check, but the computation is done for all neurons before filtering.
3. **Pre-computed deconvolved data** is loaded from NWB but discarded when `compute_own_dff=True`.

ii.
```python
F = nwb_data['fluorescence'].T  # ALL neurons
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T  # loaded but not used
dff = compute_dff_per_trial(F, Fneu, ...)  # computed for ALL neurons
cell_mask = filter_interneurons(dff, speed, iscell)  # then filtered
cell_indices = np.where(cell_mask)[0]
events = np.full((n_cells, F.shape[1]), np.nan)  # only selected cells
```

iii. Computing dF/F only for curated cells first, then filtering interneurons from that subset, would be more efficient. However, this would require knowing which cells are curated before computing dF/F, which IS known from `iscell`.
