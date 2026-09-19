# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `data/sub-*/*.nwb`, sorting alphabetically. Each NWB file is opened with `h5py` (not `pynwb`). All behavioral time series and neural data arrays are read directly from the HDF5 group structure.

ii.
```python
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
nwb_files = sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
# ...
with h5py.File(nwb_path, 'r') as f:
    subject_id = f['general/subject/subject_id'][()].decode()
    # ...
```

iii. The AI documented that 152 NWB files were found across 11 subjects, matching the paper's count. Using `h5py` instead of `pynwb` is a valid alternative for reading NWB files.

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from each NWB file's `general/subject/subject_id` field. Unique subjects are collected into a sorted set.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
# ...
subjects = sorted(subjects_set)
```

iii. This correctly identifies subjects from metadata within each file rather than from directory names, but produces the same result.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is extracted from the file's `general/session_id` field.

ii.
```python
session_id = f['general/session_id'][()].decode()
```

iii. This is a straightforward one-file-per-session mapping, consistent with the data organization.

## 1-d. How are the data split into trials?

i. Trial boundaries are found using `trial_start_flag` and `teleport_flag`. Trial starts are indices where `trial_start_flag > 0`. Trial ends are indices where `teleport_flag > 0`. The number of trials is `min(len(trial_start_inds), len(teleport_inds))`.

ii.
```python
trial_start_inds = np.where(trial_start_flag > 0)[0]
teleport_inds = np.where(teleport_flag > 0)[0]
n_trials = min(len(trial_start_inds), len(teleport_inds))
trial_start_inds = trial_start_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]
# ...
s = trial_start_inds[i]
e = teleport_inds[i]
```

iii. The AI uses `np.where(teleport_flag > 0)` to find all indices where teleport is positive, rather than detecting the rising edge. This means for multi-sample teleport periods, the first positive teleport sample is used as the trial end. The reference solution detects rising edges with `(teleport[1:] > 0) & (teleport[:-1] <= 0)` to find teleport onset. This difference means the AI's trial end indices may differ from the reference if teleport spans multiple samples. However, the AI pairs `trial_start_inds[i]` with `teleport_inds[i]` (not necessarily the next teleport after each start), which could be misaligned if the counts don't match perfectly.

## 1-e. How are trials filtered based on quality controls?

i. Trials where `e <= s` or `(e - s) < 2` are skipped. Sessions with fewer than 2 valid trials or fewer than 2 cells are skipped entirely.

ii.
```python
if e <= s or (e - s) < 2:
    # Still need to track prev_trial_rewarded
    # ...
    continue
# ...
if valid_trial_count < 2:
    print(f"  Skipping {session_label}: only {valid_trial_count} valid trials")
    return None
```

iii. The minimum trial length threshold of 2 timepoints is much less strict than the reference's 50 timepoints, but the output shows no sessions were skipped, suggesting this difference has minimal practical impact.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses the NWB's pre-computed `Deconvolved` data as the neural signal, not computing deconvolution from raw fluorescence. However, it also loads `Fluorescence` and `Neuropil` to compute dF/F for interneuron detection only.

ii.
```python
deconv_data = ophys[f'Deconvolved/{plane_name}/data'][:]
F_data = ophys[f'Fluorescence/{plane_name}/data'][:]
Fneu_data = ophys[f'Neuropil/{plane_name}/data'][:]
# ...
neural_data = deconv_cells[final_cell_mask]  # uses deconv_all after iscell + interneuron filter
```

iii. The AI's CONVERSION_NOTES.md states: "The NWB data already has deconvolved events pre-computed via the reference pipeline (suite2p OASIS). We should use these directly." However, the reference solution explicitly states that the NWB's `Deconvolved` is suite2p's own deconvolution, NOT the signal the paper analyses. The paper recomputes dF/F from F and Fneu using its own `preprocessing.dff()` function with specific parameters (maximin baseline, neuropil subtraction with coefficient 0.7, tau=0.7 deconvolution), producing a different signal.

## 2-b. How is the `neural` data processed?

i. For the actual neural signal used in the decoder, the AI uses the pre-computed `Deconvolved` data directly with no additional processing beyond filtering by `iscell` and interneuron exclusion. For interneuron detection only, the AI computes dF/F per trial using its own `compute_dff_trial` function.

ii.
```python
def compute_dff_trial(F_trial, Fneu_trial, baseline_window=BASELINE_WINDOW):
    F_corr = F_trial - NEUROPIL_COEF * Fneu_trial
    smoothed = gaussian_filter1d(F_corr, sigma=15, axis=1)
    baseline = minimum_filter1d(smoothed, size=min(baseline_window, n_t), axis=1)
    baseline = maximum_filter1d(baseline, size=min(baseline_window, n_t), axis=1)
    abs_baseline = np.abs(baseline)
    abs_baseline[abs_baseline < 1e-10] = 1e-10
    dff = (F_corr - baseline) / abs_baseline
    dff = gaussian_filter1d(dff, sigma=DFF_SMOOTH_SIGMA, axis=1)
    return dff
```

iii. The AI's `compute_dff_trial` has several differences from the reference's `dff()`:
- Does not add back mean neuropil after subtraction (the reference adds `neu_coef * mean(Fneu)` back per trial)
- Uses `gaussian_filter1d` instead of the reference's `nansmooth` (which handles NaNs differently)
- Does not handle `keep_teleports` (whether baseline can span teleport periods)
- Does not perform deconvolution (since it uses pre-computed Deconvolved)
However, this dF/F is only used for interneuron detection, not for the final neural data.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` filter from suite2p manual curation, (2) putative interneuron exclusion based on dF/F-speed correlation > 0.5.

ii.
```python
cell_mask = iscell[:, 0].astype(bool)
# ...
is_interneuron = detect_interneurons(dff_full, speed)
final_cell_mask = ~is_interneuron
neural_data = deconv_cells[final_cell_mask]
```

iii. Both filters match the reference approach. The interneuron detection uses a vectorized Pearson correlation implementation rather than the reference's per-cell loop, but the logic is equivalent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start by slicing each trial from `trial_start_inds[i]` to `teleport_inds[i]`. No additional temporal shifting is applied.

ii.
```python
trial_neural = neural_data[:, s:e]  # (n_cells, n_timepoints)
```

iii. Since the instructions specify alignment to trial start, and the data is sliced from trial start, no additional alignment is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The time bin size is computed as `1000 / effective_rate` where `effective_rate` accounts for multi-plane recordings. The value is approximately 64.5 ms (~15.5 Hz).

ii.
```python
n_planes = len(fluor_planes)
effective_rate = imaging_rate if n_planes == 1 else imaging_rate / n_planes
time_bin_ms = 1000.0 / effective_rate
```

iii. This matches the reference approach of using the native sampling rate without rebinning.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Computed synthetically from the trial length and the effective imaging rate, not from actual timestamps.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. The reference solution uses the actual `timestamps` from the behavior time series: `timestamps_curr - timestamps_curr[0]`. The AI's approach assumes perfectly uniform sampling at the effective rate, which should be nearly identical but is less faithful to the actual timing.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. An array of `n_t` values is created as `[0, 1/rate, 2/rate, ..., (n_t-1)/rate]` in seconds.

ii.
```python
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. This creates uniformly-spaced time values starting from 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both use the same index range `s:e` and the same number of timepoints, so they are inherently aligned.

ii.
```python
n_t = e - s
time_from_start = (np.arange(n_t) / effective_rate).astype(np.float32)
```

iii. Neural and behavioral data share the same sampling rate and indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
environment = bts['environment/data'][:]
# ...
trial_env = environment[s:e]
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
```

iii. The variable is extracted from the NWB behavior data, same as the reference.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The AI takes the median of non-negative environment values within the trial, then broadcasts as a scalar across all timepoints.

ii.
```python
env_val = np.median(trial_env[trial_env >= 0]) if np.any(trial_env >= 0) else 0.0
env_type = np.float32(env_val)
input_arr[1, :] = env_type  # broadcast
```

iii. The reference simply casts the per-timepoint environment values to int: `vr_environment_curr = vr_environment[idx].astype(int)`. The AI's median approach would produce the same result since environment is constant within a trial, but is unnecessarily complex.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The trial number is the loop index `i` over trials within each session.

ii.
```python
trial_number = np.float32(i)
input_arr[2, :] = trial_number
```

iii. Same approach as the reference, which also uses the loop counter: `input_curr[2] = trial`.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index (0-based) is broadcast as a constant across all timepoints in the trial.

ii.
```python
input_arr[2, :] = trial_number
```

iii. No additional processing beyond assigning the index.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from `Reward/timestamps` in the NWB file. For each trial, reward delivery is detected by checking if any reward timestamps fall within the trial's time window.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
# ...
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
```

iii. Same source variable as the reference, which also uses `Reward` timestamps.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the code checks if any reward timestamps fall within the current trial's time range. The previous trial's reward status is tracked via a `prev_trial_rewarded` variable initialized to 0 for the first trial.

ii.
```python
prev_trial_rewarded = 0
for i in range(n_trials):
    # ...
    was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
    # ...
    input_arr[3, :] = prev_outcome  # prev_outcome = np.float32(prev_trial_rewarded)
    prev_trial_rewarded = int(was_rewarded)
```

```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
```

iii. The reference uses a similar approach but works with index-based checking: `np.any(isreward[prev_reward_idx])`. The AI uses timestamp-based checking which should produce equivalent results.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` behavior time series and the reward zone coordinates, which are determined by parsing the NWB `identifier` field to get the scene name, then mapping via `get_reward_zones()`.

ii.
```python
position = bts['position/data'][:]
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
```

iii. The reference uses the `reward_zone` behavior time series and position data with a Viterbi algorithm to determine zone labels. The AI instead parses the scene name from the NWB metadata and uses a hardcoded rule system (with `change_trial=30`). Both aim to identify the reward zone for each trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the signed distance from position to the reward zone boundaries is computed. Distance is negative before the zone, 0 inside, and positive after. Then discretized into 7 bins.

ii.
```python
def compute_distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position)
    before = position < rz_start
    inside = (position >= rz_start) & (position <= rz_end)
    after = position > rz_end
    dist[before] = position[before] - rz_start
    dist[inside] = 0.0
    dist[after] = position[after] - rz_end
    return dist
```

iii. The reference uses essentially the same computation in `compute_distance_to_reward_zone()`.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses a custom `discretize_distance_to_rz()` function with explicit conditional checks for each bin.

ii.
```python
def discretize_distance_to_rz(dist):
    out = np.zeros(len(dist), dtype=np.int64)
    out[dist < -50] = 0
    out[(dist >= -50) & (dist < -10)] = 1
    out[(dist >= -10) & (dist < 0)] = 2
    out[dist == 0] = 3
    out[(dist > 0) & (dist <= 10)] = 4
    out[(dist > 10) & (dist <= 50)] = 5
    out[dist > 50] = 6
    return out
```

iii. The reference uses `np.digitize` with bin edges `[-inf, -50, -10, 0, 1e-6, 10, 50, inf]`. The AI's approach is functionally similar but uses `dist == 0` for the "inside zone" category, which checks for exact floating-point equality. Since the `compute_distance_to_reward_zone` sets inside-zone values to exactly 0.0, this works. The reference uses a small epsilon (1e-6) to separate the 0 bin from positive values.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same index slicing as neural data: `trial_pos = position[s:e]`.

ii.
```python
trial_pos = position[s:e]
dist_to_rz = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. Both neural and behavioral data use the same `s:e` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = bts['position/data'][:]
trial_pos = position[s:e]
```

iii. Same as reference.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is discretized into 5 equal bins over [0, 450] using `np.digitize` with edges at [0, 90, 180, 270, 360, 450], then clipped to [0, 4].

ii.
```python
def discretize_position(position, n_bins=POSITION_BINS):
    bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)  # [0, 90, 180, 270, 360, 450]
    binned = np.digitize(position, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned
```

iii. The reference uses edges `[-inf, 90, 180, 270, 360, inf]`. The AI uses `[0, 90, 180, 270, 360, 450]` with clipping. The key difference is that values exactly at 0 cm: with the AI's edges, `np.digitize(0, [0, 90, ...]) - 1 = 0`, which is correct. Values > 450 get clipped to bin 4. The result should be functionally equivalent.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. See 8-b. Five equal bins of 90 cm each.

ii. See 8-b.

iii. Matches the instructions' specification of 5 equal bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same index slicing `s:e`.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. Aligned via shared indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick_raw = bts['lick/data'][:]
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The AI applies a lick sensor error correction: if >35% of samples in a trial have cumulative lick > 2, all licks in that trial are set to 0. Then licks > 1 are capped to 1, and the result is binarized (>0 = 1).

ii.
```python
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
trial_lick[trial_lick > 1] = 1
trial_lick = (trial_lick > 0).astype(np.float32)
```

iii. The reference simply binarizes: `(licks_curr > 0).astype(int)`. The AI additionally applies a lick error correction from the paper's reference code (Fig 3 decoder notebook), which zeroes out lick data in trials with sensor errors. This is additional processing not present in the reference solution.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same index slicing `s:e`.

ii.
```python
trial_lick = lick[s:e].copy()
```

iii. Aligned via shared indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the NWB `identifier` field, which contains the scene name (e.g., `Env1_LocationB_to_A`). The scene name is parsed to determine which reward zone(s) apply and when the switch occurs.

ii.
```python
identifier = f['identifier'][()].decode()
scene = get_scene_from_identifier(identifier)
rz_coords, rz_labels = get_reward_zones(scene, n_trials)
# ...
rz_loc = rz_label_to_idx(rz_labels[i])
```

iii. The reference uses a Viterbi algorithm on empirical `reward_zone` position data to determine zone labels. The AI uses the scene metadata to directly map zone identities. The AI's approach assumes a fixed switch trial (trial 30) from the reference code's `DEFAULT_CHANGE_TRIAL`, while the reference's Viterbi approach empirically determines when switches occur. Both should produce similar results if the data matches the expected patterns.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The `get_reward_zones()` function parses the scene name to determine the initial and post-switch reward zone locations. For switch sessions, trials before index 30 get one zone and trials after get another. Labels are mapped to indices: A=0, B=1, C=2.

ii.
```python
def get_reward_zones(scene, n_trials, change_trial=DEFAULT_CHANGE_TRIAL):
    if 'Location' in scene and '_to' not in scene:
        loc = scene.split('Location')[-1]
        rz_coords[:] = REWARD_ZONE_DICT[loc]
        rz_labels[:] = loc
    elif 'A_to' in scene and scene[-1] == 'B':
        rz_coords[:change_trial] = REWARD_ZONE_DICT['A']
        rz_labels[:change_trial] = 'A'
        rz_coords[change_trial:] = REWARD_ZONE_DICT['B']
        rz_labels[change_trial:] = 'B'
    # ... similar for other switch patterns
```

iii. This mirrors the reference code's `behavior.py::get_reward_zones()` function. The hardcoded `change_trial=30` matches the paper's protocol.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the NWB file.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
was_rewarded = detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time)
reward_outcome = int(was_rewarded)
```

iii. Same source as reference.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward timestamp falls within the trial's time window, 0 otherwise. Broadcast as constant across all timepoints.

ii.
```python
def detect_reward_in_trial(reward_timestamps, trial_start_time, trial_end_time):
    return np.any((reward_timestamps >= trial_start_time) & (reward_timestamps <= trial_end_time))
# ...
output_arr[5, :] = reward_outcome
```

iii. The reference uses index-based checking (`np.any(isreward[idx])`); the AI uses timestamp-based checking. Both produce binary per-trial values.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data cropped to minimum of the two.
- **Very short trials**: Trials with `e <= s` or `(e - s) < 2` are skipped.
- **Few cells or trials**: Sessions with < 2 cells or < 2 valid trials are skipped.
- **Lick sensor errors**: Trials with >35% of samples having lick > 2 get licks zeroed out.

ii.
```python
n_timepoints_total = min(n_timepoints_neural, n_timepoints_behav)
# ...
if e <= s or (e - s) < 2:
    continue
# ...
if np.sum(trial_lick > 2) / n_t > LICK_ERROR_FRACTION_THRESHOLD:
    trial_lick[:] = 0
```

iii. The reference handles similar edge cases (neural/behavior mismatch, short trials with < 50 timepoints, missing reward zone data via Viterbi uniform emission).

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files with `h5py` and reading large arrays. Processing is fast at ~2-10 seconds per session (152 sessions total). The AI does not have a separate survey step, processing each file once.

ii. N/A

iii. The AI's single-pass approach avoids the reference's double-loading (survey + conversion).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop for computing inputs, outputs, and extracting neural data. Some operations like discretization could be applied to entire session arrays before splitting. The interneuron detection is already vectorized.

ii. N/A

iii. The per-trial loop is natural given variable-length trials.

## 13-c. What processing does the code repeat multiple times?

i. The code loads F and Fneu to compute dF/F for interneuron detection, but then uses the separate Deconvolved data for the actual neural signal. This means both Deconvolved and Fluorescence/Neuropil arrays are loaded, when only one set would be needed if the approach were consistent.

ii. N/A

iii. The reference solution also loads data twice (survey + conversion), though for different reasons.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F from raw fluorescence (`compute_dff_trial`) solely for interneuron detection, but the dF/F itself is not used for the neural signal in the decoder. The decoder uses the pre-computed `Deconvolved` data instead.

ii.
```python
dff_full = np.full_like(F_cells, np.nan)
for i in range(n_trials):
    s = trial_start_inds[i]
    e = teleport_inds[i]
    dff_full[:, s:e] = compute_dff_trial(F_cells[:, s:e], Fneu_cells[:, s:e])
is_interneuron = detect_interneurons(dff_full, speed)
# But the neural data comes from:
neural_data = deconv_cells[final_cell_mask]  # NOT from dff_full
```

iii. This is arguably necessary since interneuron detection requires dF/F, but the dF/F computation itself is wasted work if one could detect interneurons from the deconvolved signal (though the paper specifies dF/F-speed correlation).
