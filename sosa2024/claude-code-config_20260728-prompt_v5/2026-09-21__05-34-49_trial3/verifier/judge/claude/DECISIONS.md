# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files using `h5py` (not `pynwb`). It collects all `.nwb` files from subdirectories of `/app/data` that start with `sub-`. Files are sorted by subject then session. Each NWB file is loaded in `load_nwb_session()` which extracts fluorescence, neuropil, behavioral time series, reward timestamps, and metadata.

ii.
```python
def collect_nwb_files(data_dir, sample=False):
    files = []
    for subj_dir in sorted(os.listdir(data_dir)):
        if not subj_dir.startswith('sub-'):
            continue
        subj_path = os.path.join(data_dir, subj_dir)
        for nwb_file in sorted(os.listdir(subj_path)):
            if nwb_file.endswith('.nwb'):
                files.append(os.path.join(subj_path, nwb_file))
    return files
```

```python
def load_nwb_session(filepath):
    data = {}
    with h5py.File(filepath, 'r') as f:
        ophys = f['processing']['ophys']
        bts = f['processing']['behavior']['BehavioralTimeSeries']
        # ... loads fluorescence, neuropil, behavioral data
```

iii. The AI chose h5py for direct HDF5 access rather than pynwb. It finds all NWB files by directory structure. The 11 subjects match the paper.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `sub-` prefixed directories in the data folder. Subject IDs are also extracted from each NWB file's `general/subject/subject_id` field.

ii.
```python
for subj_dir in sorted(os.listdir(data_dir)):
    if not subj_dir.startswith('sub-'):
        continue
```
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode()
```

iii. Directory structure provides subject organization; the NWB file's internal subject_id is used for the final data dictionary.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by the file naming convention `sub-{id}_ses-{nn}_behavior+ophys.nwb`.

ii.
```python
for nwb_file in sorted(os.listdir(subj_path)):
    if nwb_file.endswith('.nwb'):
        files.append(os.path.join(subj_path, nwb_file))
```

iii. The one-file-per-session convention is standard for NWB datasets.

## 1-d. How are the data split into trials?

i. Trials are identified by `trial_start == 1` markers and `teleport == 1` markers. For each trial start, the next teleport frame after it defines the trial end (exclusive).

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
    starts = np.where(trial_start_signal == 1)[0]
    teleports = np.where(teleport_signal == 1)[0]
    trial_starts = []
    trial_ends = []
    for s in starts:
        next_teleports = teleports[teleports > s]
        if len(next_teleports) > 0:
            trial_starts.append(s)
            trial_ends.append(next_teleports[0])
    return trial_starts, trial_ends
```

iii. Uses trial_start and teleport signals directly. The reference uses rising edge detection for teleport (`(teleport[1:] > 0) & (teleport[:-1] <= 0)`), while the AI uses exact equality `teleport == 1`. This could differ if teleport takes values other than 0/1.

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) lick sensor error detection - trials with >30% of frames having lick > 2 are removed, (2) trials with all-NaN neural data are skipped, (3) trials with fewer than 2 timepoints after potential downsampling are skipped.

ii.
```python
def detect_lick_errors(lick, trial_starts, trial_ends, threshold=LICK_ERROR_THRESH):
    error_trials = []
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        trial_licks = lick[start:end]
        n_frames = len(trial_licks)
        if n_frames == 0:
            continue
        frac_high = np.sum(trial_licks > 2) / n_frames
        if frac_high > threshold:
            error_trials.append(i)
    return error_trials
```
```python
if i in lick_error_trials:
    continue
if np.all(np.isnan(trial_neural)):
    continue
if T < 2:
    continue
```

iii. The lick error detection follows the paper's description of removing trials with stuck lick sensor (>30% frames with lick > 2). The CONVERSION_NOTES.md reports matching the paper's 81 removed trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `Fluorescence` and `Neuropil` traces from the ophys processing module, filtered by the `iscell` mask from `ImageSegmentation/PlaneSegmentation`.

ii.
```python
fluor_grp = ophys['Fluorescence']
# ...
fl = fluor_grp[pk]['data'][:].T  # (n_rois, T)
neu = ophys['Neuropil'][pk]['data'][:].T  # (n_rois, T)
fluor_list.append(fl[plane_iscell])
neuropil_list.append(neu[plane_iscell])
```

iii. The AI correctly identified that the paper computes its own dF/F from raw fluorescence rather than using the NWB's pre-computed Deconvolved field.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F from Fluorescence and Neuropil: (1) neuropil subtraction with coefficient 0.7, (2) add back neuropil mean per trial, (3) maximin baseline (Gaussian smooth sigma=15, min filter 300, max filter 300), (4) dF/F = (F - baseline)/|baseline|, (5) smooth with Gaussian sigma=2. **No deconvolution is performed** - the final neural signal is dF/F, not deconvolved events.

ii.
```python
def compute_dff(f_raw, f_neu, trial_starts, trial_ends):
    # Neuropil subtraction
    f_ = f_ - NEUROPIL_COEF * f_neu_
    # Baseline per trial: add back neuropil mean, maximin baseline
    neuropil_mean = np.nanmean(f_neu_[:, start:end], axis=1, keepdims=True)
    trial_data = trial_data + NEUROPIL_COEF * neuropil_mean
    smoothed = nansmooth(trial_data, BASELINE_SMOOTH_SIGMA, axis=1)
    bl = ndi.minimum_filter1d(smoothed, BASELINE_MINMAX_WINDOW, axis=-1)
    bl = ndi.maximum_filter1d(bl, BASELINE_MINMAX_WINDOW, axis=-1)
    # dF/F
    dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])
    # Smooth per trial
    dff[:, start:end] = nansmooth(dff[:, start:end], DFF_SMOOTH_SIGMA, axis=1)
    return dff
```

iii. The AI's CONVERSION_NOTES.md states: "Neural signal: Compute dF/F from raw Fluorescence + Neuropil. Matches reference processing exactly. More informative than deconvolved events for neural network decoder." However, the reference paper uses deconvolved events, not dF/F. Additionally, the AI's `nansmooth` uses `gaussian_filter1d` (1D) while the reference uses `gaussian_filter` with sigma `[0, 15]` (2D with 0 across cells, 15 across time). The AI also does NOT handle the `keep_teleports` metadata that determines whether the baseline window can span inter-trial intervals.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` mask from Suite2P manual curation, (2) putative interneuron removal based on speed-dF/F correlation > 0.5.

ii.
```python
plane_iscell = iscell[plane_mask, 0].astype(bool)
fluor_list.append(fl[plane_iscell])
```
```python
def detect_interneurons(dff, speed, nanmask=None):
    for c in range(n_cells):
        valid = nanmask & ~np.isnan(dff[c, :]) & ~np.isnan(speed)
        r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
        if r > INTERNEURON_R_THRESH:
            is_interneuron[c] = True
    return is_interneuron
```

iii. Matches the paper's described filtering steps. The interneuron detection is done on dF/F (not deconvolved events), which is consistent with the reference code that also correlates dF/F with speed.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to trial start. No additional alignment is needed since the trial boundaries already define the start.

ii.
```python
trial_neural = dff[:, start:end]
```

iii. The instructions specify alignment to start of trial, which is the natural split point.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is ~64.48 ms (1/15.5078125 Hz). For dual-plane sessions (m17, m18), the AI downsamples by 2x by averaging consecutive frames, so the effective rate is the scanner rate divided by 2.

ii.
```python
TARGET_RATE = 15.5078125  # Hz
downsample = 2 if raw['is_dual_plane'] else 1
effective_rate = raw['rate'] / downsample
if downsample > 1:
    T_new = T // downsample
    trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
```

iii. The AI notes dual-plane mice (m17, m18) have scanner rate ~31 Hz and need downsampling. The reference does NOT downsample; instead, it passes `n_planes` to the dff function, which uses `frame_rate/n_planes` for the deconvolution kernel rate. The reference keeps data at the scanner rate with the deconvolution accounting for the per-plane timing.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial and the effective sampling rate, NOT from actual timestamps.

ii.
```python
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The AI constructs time from the frame index and rate rather than using behavior timestamps. The reference uses actual timestamps from the NWB file.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Time is computed as frame_index / effective_rate. No timestamp lookup is involved.

ii.
```python
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. This produces uniformly spaced time values. The reference subtracts the first timestamp from subsequent timestamps, which could show slight non-uniformities.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Since time is computed from frame indices and the neural data uses the same frame indices, they are inherently aligned.

ii. Same array length T is used for both.

iii. Alignment is trivial since both come from the same frame indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavioral time series.

ii.
```python
data['environment'] = bts['environment']['data'][:].astype(np.float64)
```

iii. The environment variable in the NWB file records which virtual environment is active.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the first active (>= 0) environment value is taken and broadcast across all timepoints.

ii.
```python
for start, end in zip(trial_starts, trial_ends):
    env_vals = raw['environment'][start:end]
    active_env = env_vals[env_vals >= 0]
    if len(active_env) > 0:
        environments.append(int(round(active_env[0])))
    else:
        environments.append(0)
```
```python
env = np.full(T, environments[i], dtype=np.float32)
```

iii. The AI filters out -1 values (ITI markers) and takes the first valid value. The reference takes the environment variable directly for each timepoint within the trial (`vr_environment_curr = vr_environment[idx].astype(int)`), which is time-varying within a trial rather than constant.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Derived from the loop counter over trials (0-indexed sequential index within the session).

ii.
```python
trial_num = np.full(T, i, dtype=np.float32)
```

iii. The trial number is the sequential index of valid trials within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing; the loop index `i` is used directly and broadcast to all timepoints.

ii.
```python
trial_num = np.full(T, i, dtype=np.float32)
```

iii. Note that if lick error trials are skipped, the trial number still counts from the original trial index (since `i` iterates over all trials, not just valid ones). This means trial numbers may have gaps.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file, mapped to the `trial_number` signal to determine which trials were rewarded.

ii.
```python
def determine_reward_per_trial(reward_timestamps, behavior_timestamps, trial_number_signal,
                                trial_starts, trial_ends):
    rewarded = np.zeros(n_trials, dtype=int)
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1
    return rewarded
```

iii. The AI maps reward timestamps to the trial_number signal, which could be unreliable if trial_number doesn't match the actual trial boundaries.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i`, the previous trial outcome is `rewarded[i-1]`. For the first trial (i=0), it's set to 0.

ii.
```python
prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
```

iii. Uses the `rewarded` array computed by `determine_reward_per_trial`. The reference checks whether any reward event fell within the previous trial's time boundaries directly.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone label for each trial. Reward zone labels are determined from the `reward_zone` and `position` signals.

ii.
```python
zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'],
                                     trial_starts, trial_ends)
```

iii. The reward zone is determined by computing the mean position where `reward_zone > 0` and assigning to the closest zone center. Missing labels are inherited from nearest neighbor.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from position to reward zone boundaries: negative before zone, 0 inside, positive after.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    distance = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    distance[before] = position[before] - zone_start
    distance[inside] = 0
    distance[after] = position[after] - zone_end
    return distance
```

iii. Same logic as the reference implementation.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit conditional logic.

ii.
```python
def discretize_distance(distance):
    bins = np.zeros(len(distance), dtype=int)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bin edges match the instructions. The reference uses `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`, where `1e-6` separates exactly 0 from slightly positive. The AI uses `distance == 0` for the "0 cm" bin, which behaves similarly since the distance function only returns exact 0 for positions inside the zone.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices used for both neural and behavioral data within each trial.

ii.
```python
trial_pos = raw['position'][start:end]
distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
```

iii. Alignment is inherent from using the same frame slicing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
data['position'] = bts['position']['data'][:].astype(np.float64)
```

iii. Direct position variable from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting per-trial slices and discretizing.

ii.
```python
trial_pos = raw['position'][start:end]
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal bins of 90 cm each spanning the 450 cm track.

ii.
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=int)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Matches the instructions for 5 equal-sized bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices used for both within each trial.

ii. Same slicing as neural: `trial_pos = raw['position'][start:end]`

iii. Inherent from shared frame indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series.

ii.
```python
data['lick'] = bts['lick']['data'][:].astype(np.float64)
```

iii. Direct lick variable from the NWB file.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
lick_binary = (trial_lick > 0).astype(int)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices used for both within each trial.

ii. Same slicing: `trial_lick = raw['lick'][start:end]`

iii. Inherent from shared frame indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` and `position` behavioral time series. For each trial, the mean position where `reward_zone > 0` is computed and assigned to the closest zone center (A, B, or C).

ii.
```python
def determine_reward_zone(position, reward_zone_signal, trial_starts, trial_ends):
    for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
        rz = reward_zone_signal[start:end]
        pos = position[start:end]
        mask = rz > 0
        if mask.sum() > 0:
            mean_pos = pos[mask].mean()
            best_zone = None
            best_dist = float('inf')
            for label, (zs, ze) in REWARD_ZONES.items():
                center = (zs + ze) / 2
                dist = abs(mean_pos - center)
                if dist < best_dist:
                    best_dist = dist
                    best_zone = label
            zone_labels[i] = best_zone
    # Fill missing by nearest neighbor
    for i in range(n_trials):
        if zone_labels[i] is None:
            for d in range(1, n_trials):
                if i - d >= 0 and zone_labels[i - d] is not None:
                    zone_labels[i] = zone_labels[i - d]
                    break
                if i + d < n_trials and zone_labels[i + d] is not None:
                    zone_labels[i] = zone_labels[i + d]
                    break
    return zone_labels
```

iii. The reference uses a more sophisticated Viterbi algorithm to assign zones, encouraging temporal consistency. The AI uses a simpler nearest-center approach per trial.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Mean position in reward zone -> closest zone center assignment -> nearest neighbor fill for missing trials. Encoded as 0=A, 1=B, 2=C.

ii.
```python
zone_idx = ZONE_LABELS.index(zone_label) if zone_label in ZONE_LABELS else 0
rz_loc = np.full(T, zone_idx, dtype=int)
```

iii. See 10-a for full processing details.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the NWB file, mapped to trial numbers.

ii.
```python
data['reward_timestamps'] = reward_ts_data['timestamps'][:]
```
```python
rewarded = determine_reward_per_trial(raw['reward_timestamps'], raw['behavior_timestamps'],
                                       raw['trial_number'], trial_starts, trial_ends)
```

iii. Reward events have their own timestamps separate from the behavior sampling rate.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each reward timestamp is mapped to the closest behavior frame via `argmin(|timestamps - reward_time|)`, then the `trial_number` signal at that frame determines which trial received the reward.

ii.
```python
def determine_reward_per_trial(reward_timestamps, behavior_timestamps, trial_number_signal,
                                trial_starts, trial_ends):
    rewarded = np.zeros(n_trials, dtype=int)
    for rt in reward_timestamps:
        frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
        trial_num = int(trial_number_signal[frame_idx])
        if trial_num >= 0 and trial_num < n_trials:
            rewarded[trial_num] = 1
    return rewarded
```

iii. The reference uses `searchsorted` to map reward timestamps to behavior timepoints and then checks if any reward falls within a trial's time boundaries. The AI instead relies on the `trial_number` signal, which may not perfectly agree with the trial boundaries derived from `trial_start`/`teleport`.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is truncated to the minimum length.
- **Missing reward zone data**: Trials without `reward_zone > 0` get labels inherited from the nearest neighbor trial.
- **Lick sensor errors**: Trials with stuck lick sensors (>30% frames with lick > 2) are removed.
- **All-NaN neural data**: Trials with all-NaN neural data are skipped.
- **NaN in neural data**: Replaced with 0 via `np.nan_to_num`.

ii.
```python
n_min = min(n_neural, n_behav)
if n_neural != n_behav:
    data['fluorescence'] = data['fluorescence'][:, :n_min]
    # ... truncate all arrays
```
```python
trial_neural = np.nan_to_num(trial_neural, nan=0.0).astype(np.float32)
```

iii. The neural/behavior mismatch handling matches the reference approach. The `nan_to_num` replacement is specific to the AI's approach since it uses dF/F (which may have NaNs) rather than deconvolved events.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py (I/O bound)
2. Computing dF/F (per-cell smoothing and filtering operations)
3. Interneuron detection (correlation computation for each cell)

ii. N/A

iii. The AI reports processing time estimates in CONVERSION_NOTES.md and includes timing instrumentation.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The `detect_interneurons` function loops over each cell to compute correlations. The `determine_reward_per_trial` function loops over each reward timestamp. The `determine_reward_zone` function loops over trials.

ii.
```python
for c in range(n_cells):
    r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
```

iii. These loops are relatively small (hundreds of cells/timestamps) so vectorization would provide modest speedup.

## 13-c. What processing does the code repeat multiple times?

i. No redundant processing - the AI loads each NWB file exactly once in `process_session()`. Unlike the reference which has a separate survey step that reads files twice, the AI's code processes everything in a single pass.

ii. N/A

iii. The AI's approach is more efficient than the reference in this regard, avoiding the survey/convert two-pass pattern.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F which is used as the neural signal, and also computes speed discretization (which is one of the decoder outputs). No significant unnecessary processing is apparent.

ii. N/A

iii. N/A
