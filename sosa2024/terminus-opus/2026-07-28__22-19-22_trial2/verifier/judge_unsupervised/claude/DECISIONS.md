# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files using h5py. It discovers all NWB files via `glob.glob('data/sub-*/*.nwb')`, sorted alphabetically. Each NWB file corresponds to one session. Data is loaded within the `process_session()` function. Behavioral data is read from `processing/behavior/BehavioralTimeSeries/` and neural data from `processing/ophys/Deconvolved/` and `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`.

ii.
```python
all_files = sorted(glob.glob('data/sub-*/*.nwb'))
# ...
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, show_processing=show, session_idx=file_idx)
```

Inside `process_session()`:
```python
with h5py.File(nwb_path, 'r') as f:
    position = behav['position/data'][()]
    speed = behav['speed/data'][()]
    lick = behav['lick/data'][()]
    # ... etc.
    deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The AI chose to load NWB files directly with h5py, matching the data storage format (NWB/HDF5). The reference code uses a `sess` object created via `pp.create_sess()` which internally loads from raw data, but the NWB files contain the same pre-processed data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the NWB filename to extract the subject name (e.g., `sub-m11`). Each unique subject name is added to `subject_list`, and sessions are mapped to subjects via `subject_idx`.

ii.
```python
subj_name = os.path.basename(nwb_path).split('_')[0]  # e.g., sub-m11
# ...
if subj_name not in subject_list:
    subject_list.append(subj_name)
subj_idx = subject_list.index(subj_name)
```

iii. Subject names are extracted from filenames which follow the pattern `sub-{mouse}_ses-{session}_behavior+ophys.nwb`. This correctly identifies all 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is read from `general/session_id` within the NWB file. All 152 NWB files across 11 subjects are processed as 152 sessions.

ii.
```python
sess_id = f['general/session_id'][()].decode()
# Each NWB file = one session, processed in the main loop
for file_idx, nwb_path in enumerate(files_to_process):
    result = process_session(nwb_path, ...)
```

iii. The one-file-per-session structure matches the NWB data organization. Sessions with fewer than 2 trials are skipped (none encountered).

## 1-d. How are the data split into trials?

i. Trial boundaries are defined by `trial_start` and `teleport` signals in the behavioral data. Trial starts are indices where `trial_start_signal > 0`, and trial ends are indices where `teleport_signal > 0`. The AI uses `start:stop` indexing (i.e., `trial_starts[i]` to `teleports[i]`).

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]
teleports = np.where(teleport_signal > 0)[0]
# ...
for i in range(n_trials):
    start = trial_starts[i]
    stop = teleports[i]
    trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The reference code uses `sess.trial_start_inds` and `sess.teleport_inds` with `start-1:stop-1` indexing. The AI uses `start:stop` which is off by one index compared to the reference code. The reference code's `start-1` shifts the trial window one timepoint earlier.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only if `stop <= start` (zero-length trials) or if a session has fewer than 2 trials total. The AI also corrects lick sensor errors per trial (setting lick data to NaN then 0), but does not remove trials entirely based on lick errors. There is no speed-based trial filtering.

ii.
```python
if stop <= start:
    continue
# ...
if len(neural_trials) < 2:
    print(f"  SKIPPING: only {len(neural_trials)} trials")
    continue
```

iii. The reference code does not remove trials outright either; it uses NaN masking to exclude bad timepoints within trials. The AI's approach of keeping all trials is broadly consistent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the deconvolved calcium events stored in `processing/ophys/Deconvolved/plane{N}/data` in the NWB files. For multi-plane sessions (m17, m18), data from multiple planes is concatenated along the neuron axis.

ii.
```python
planes = sorted(f['processing/ophys/Deconvolved'].keys())
deconv_planes = []
for plane in planes:
    deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
deconv_data = np.concatenate(deconv_planes, axis=1)
```

iii. The reference code uses `sess.timeseries['events']` which corresponds to deconvolved events. The AI correctly identified this as the neural data source.

## 2-b. How is the `neural` data processed?

i. The neural data is filtered to include only cells classified as neurons by Suite2p (`iscell[:, 0] == 1`), then putative interneurons are excluded based on speed correlation of deconvolved events (r > 0.5). The data is sliced per trial and transposed to (n_neurons, n_timepoints), cast to float32.

ii.
```python
cell_mask = iscell[:, 0] == 1
# ... interneuron exclusion ...
cell_mask = cell_mask & ~interneuron_mask
neural_data = deconv_data[:, cell_indices]
# per trial:
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The reference code uses dF/F (`ts_key='dff'`) for interneuron identification with a threshold of r > 0.3, not deconvolved events with r > 0.5. The AI used deconvolved events and a higher threshold (0.5 from the paper text), which differs from the reference code's implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two quality filters are applied: (1) Suite2p's `iscell` classification, and (2) putative interneuron exclusion via speed-activity correlation.

ii.
```python
cell_mask = iscell[:, 0] == 1
# ...
interneuron_mask = speed_corr > 0.5
cell_mask = cell_mask & ~interneuron_mask
```

iii. The reference code's `is_putative_interneuron` uses dF/F with r_thresh=0.3 by default. The AI uses deconvolved events with r_thresh=0.5. The paper mentions excluding "putative interneurons" based on speed correlation > 0.5 with dF/F (excluding 0.42 +/- 0.85% of cells). The AI used the paper's threshold value but the wrong signal type (deconvolved events instead of dF/F).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Each trial's neural data runs from `trial_starts[i]` to `teleports[i]`, starting at the trial_start signal.

ii.
```python
start = trial_starts[i]
stop = teleports[i]
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
```

iii. The instructions specify "Temporally align based on start of the trial," which the AI follows. However, the reference code uses `start-1:stop-1` indexing, meaning the AI's alignment is shifted by one timepoint relative to the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native imaging frame rate (~15.5 Hz, ~64.48 ms per frame) without any temporal rebinning. The time bin size is computed as the median inter-frame interval.

ii.
```python
dt = np.median(np.diff(timestamps))
# ...
dt_ms = session_infos[0]['dt'] * 1000  # convert to ms
```

iii. The reference code also operates at the native imaging rate without rebinning. This matches.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` array in the NWB behavioral data.

ii.
```python
timestamps = behav['position/timestamps'][()]
# per trial:
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. This is a straightforward computation from the timestamps, creating a time vector starting at 0 for each trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the timestamp at trial start is subtracted from all timestamps within the trial, producing time in seconds relative to trial onset.

ii.
```python
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. No additional processing beyond subtraction and casting to float32. This is straightforward and correct.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both neural data and time_from_start use the same `start:stop` indexing, so they are aligned frame-by-frame.

ii.
```python
trial_neural = neural_data[start:stop, :].T.astype(np.float32)
time_from_start = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
```

iii. Alignment is implicit through shared indexing. This is correct given the behavioral and neural data share the same sampling timestamps.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Environment type is derived from the session's NWB `identifier` field, which encodes the scene name (e.g., `Env1_LocationB_to_A`). It is NOT derived from the `environment` signal in the behavioral timeseries.

ii.
```python
identifier = f['identifier'][()].decode()
scene_info = parse_scene(identifier)
# ...
env_per_trial = get_env_per_trial(scene_info, n_trials)
```

iii. The AI parses the scene name to determine the environment. The `environment` behavioral signal is loaded but not used for this purpose. The reference code uses `sess.vr_data['morph']` via `get_trial_types()`, which encodes Env1=0, Env2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The scene name is parsed to extract environment labels (Env1 or Env2). For switch sessions, environment changes at trial 30 (hardcoded). Env1=0, Env2=1.

ii.
```python
def get_env_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    env_map = {'Env1': 0, 'Env2': 1}
    for i in range(n_trials):
        if is_env_switch and i >= change_trial:
            env_vals.append(env_map.get(env_after, 0))
        else:
            env_vals.append(env_map.get(env_before, 0))
    return env_vals
```

iii. The reference code's `get_trial_types()` uses `sess.vr_data['morph']` directly. The AI's approach of parsing the scene name should yield the same result but is more fragile. The reference code also checks `sess.change_reward_trial` which may differ from the hardcoded 30.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the loop index `i` within the session processing, representing the 0-indexed within-session trial number.

ii.
```python
np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. The AI uses the sequential trial index. The NWB file also has a `trial number` field that could have been used, but the sequential index is a reasonable choice.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The trial number is simply the loop iteration index `i`, broadcast to all timepoints in the trial. No transformation is applied.

ii.
```python
np.full(n_tp, i, dtype=np.float32),  # trial number within session
```

iii. Minimal processing. The instructions say "Trial number (continuous, per trial)" which is satisfied.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the reward determination for the previous trial. Reward is determined by checking if any reward timestamp from `Reward/timestamps` falls within the trial's time window.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
# per trial:
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
# ...
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
```

iii. The reference code determines reward from `sess.vr_data['reward']` within trial boundaries and also checks `rzone`. The AI uses reward timestamps instead. For the first trial, previous outcome is set to 0 (no previous trial), which is reasonable.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. First, reward is determined for each trial by checking overlap of reward timestamps with trial time windows. Then, `prev_outcome[i] = is_rewarded[i-1]` for trials 1+, and `prev_outcome[0] = 0`.

ii.
```python
prev_outcome = np.zeros(n_trials, dtype=int)
for i in range(1, n_trials):
    prev_outcome[i] = is_rewarded[i - 1]
```

iii. The instructions say "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)". The AI correctly uses 0 for omitted/first trial and 1 for rewarded.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from `position` data and the reward zone coordinates (determined from the scene name). Reward zone boundaries are defined in `REWARD_ZONE_DICT` matching the reference code's values (A=[80,130], B=[200,250], C=[320,370]).

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
rz_start, rz_end = rz_coords[i]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The AI computes signed distance to the nearest point of the reward zone, consistent with the decoder task specification.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative if before the reward zone, positive if after, 0 if inside. Position is clipped to [0, 450].

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    dist = np.zeros_like(position, dtype=float)
    before_mask = position < rz_start
    dist[before_mask] = position[before_mask] - rz_start
    dist[in_mask] = 0.0
    dist[after_mask] = position[after_mask] - rz_end
    return dist
```

iii. This implements signed linear distance. The reference code uses circular relative position aligned to reward zone, which is different. However, the decoder task explicitly asks for "Distance to any location in the reward zone" with specific cm bins, so linear distance is appropriate.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins exactly as specified in the instructions.

ii.
```python
def discretize_distance(dist):
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins
```

iii. Matches the instruction specifications exactly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Aligned via shared `start:stop` indexing. Both neural and distance data use the same trial indices.

ii.
```python
trial_neural = neural_data[start:stop, :].T
trial_pos = position[start:stop]
trial_dist = distance_to_reward_zone(trial_pos, rz_start, rz_end)
dist_disc = discretize_distance(trial_dist)
```

iii. Same-index alignment ensures frame-by-frame correspondence.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` data in the behavioral timeseries.

ii.
```python
trial_pos = position[start:stop].astype(np.float32)
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
```

iii. Direct use of the position variable, clipped to [0, 450] cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to [0, 450] cm (track bounds), then discretized.

ii.
```python
trial_pos = np.clip(trial_pos, TRACK_START, TRACK_LENGTH)
pos_disc = discretize_position(trial_pos)
```

iii. The reference code uses `orig_pos` directly. Clipping is a reasonable safeguard.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins of 90 cm each using `np.digitize`.

ii.
```python
def discretize_position(position, n_bins=5):
    bin_edges = np.linspace(TRACK_START, TRACK_LENGTH, n_bins + 1)
    bins = np.digitize(position, bin_edges[1:-1])
    bins = np.clip(bins, 0, n_bins - 1)
    return bins
```

iii. Matches the instruction: "discretized into 5 equal-sized bins." Bin edges are [0, 90, 180, 270, 360, 450].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Aligned via shared `start:stop` indexing.

ii. Same as 7-d.

iii. Frame-by-frame correspondence through common trial indices.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` data in the behavioral timeseries.

ii.
```python
lick = behav['lick/data'][()]
```

iii. The reference code uses `sess.timeseries['licks']`, which is the same data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Lick sensor error correction is applied: if >35% of samples in a trial have cumulative lick count >2, the trial's lick data is set to NaN, then NaN values are set to 0. Lick values >1 are capped at 1 (binary). Final output is binary (0 = no lick, 1 = lick).

ii.
```python
# Lick sensor error correction
frac_high = np.sum(trial_licks > 2) / len(trial_licks)
if frac_high > 0.35:
    lick_corrected[start:stop] = np.nan
# Binary conversion
lick_binary[lick_binary > 1] = 1
lick_binary[np.isnan(lick_binary)] = 0  # treat NaN licks as 0
```

iii. The reference code sets erroneous lick data to NaN and keeps NaN (which gets masked out later via `mask_out_nans`). The AI sets NaN licks to 0 instead of excluding those timepoints. This is a difference: the reference code removes timepoints with lick sensor errors from both neural and behavioral data, while the AI keeps them with lick=0.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Aligned via shared `start:stop` indexing.

ii. Same as previous alignment questions.

iii. Frame-by-frame alignment through common indices.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the session's NWB `identifier` field (scene name), parsed to determine the reward zone label (A, B, or C) for each trial.

ii.
```python
scene_info = parse_scene(identifier)
rz_labels, rz_coords = get_reward_zone_per_trial(scene_info, n_trials)
rz_label_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = rz_label_map[rz_labels[i]]
```

iii. The reference code uses `behav.get_reward_zones(sess)` which also parses the scene name. The AI's implementation mirrors this logic with equivalent reward zone coordinates.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene name is parsed to extract reward zone labels. For switch sessions, the zone changes at trial 30 (hardcoded). Labels A/B/C are mapped to 0/1/2.

ii.
```python
def get_reward_zone_per_trial(scene_info, n_trials, change_trial=CHANGE_TRIAL):
    for i in range(n_trials):
        if is_switch and i >= change_trial:
            label = rz_after
        else:
            label = rz_before
        rz_labels.append(label)
```

iii. The reference code's `get_reward_zones()` uses `change_trial=30` as default but also checks `sess.change_reward_trial`. The AI hardcodes 30 without checking the session-specific attribute.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from `Reward/timestamps` in the behavioral timeseries, checked against trial time windows.

ii.
```python
reward_timestamps = behav['Reward/timestamps'][()]
# per trial:
reward_in_trial = np.any(
    (reward_timestamps >= trial_start_time) &
    (reward_timestamps <= trial_end_time)
)
is_rewarded[i] = int(reward_in_trial)
```

iii. The reference code uses `sess.vr_data['reward']` to check reward events within trial boundaries. The AI uses reward timestamps instead, which should give equivalent results.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, checks if any reward timestamp falls within [trial_start_time, trial_end_time]. Binary: 0 = no reward, 1 = reward.

ii. Same as 11-a code.

iii. The ~15% omission rate in the converted data (15.3%) matches the expected ~15%, confirming correctness.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several handling strategies: (1) Neural/behavioral length mismatches are resolved by truncating to the minimum length. (2) Mismatched trial_starts/teleports counts are resolved by truncating to the minimum count. (3) Zero-length trials (stop <= start) are skipped. (4) Lick sensor errors are corrected by setting to 0. (5) Negative speeds are clipped to 0.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    # truncate all arrays to min_len

if len(teleports) != n_trials:
    n_trials = min(n_trials, len(teleports))

if stop <= start:
    continue
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md. The length mismatch handling is pragmatic. The reference code doesn't explicitly handle these since it uses pre-processed session objects.

## 13-a. What are the most time-consuming steps of the code?

i. Loading NWB files (especially large ones like m17/m18 with dual-plane imaging) and the interneuron correlation computation are the most time-consuming. The AI reported ~1.3s per session average, ~200s total.

ii.
```python
# Loading is the bottleneck:
deconv_planes.append(f[f'processing/ophys/Deconvolved/{plane}/data'][()])
# Interneuron computation:
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]
```

iii. Conversion output shows load times of 0.1-0.9s per session with total processing of 0.2-2.4s per session.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron speed correlation loop iterates over each cell individually. This could be vectorized using matrix correlation. The per-trial reward determination loop could also be vectorized.

ii.
```python
# Interneuron loop (could be vectorized):
for c in range(deconv_data.shape[1]):
    if cell_mask[c]:
        r = np.corrcoef(valid_neural[both_valid], valid_speed[both_valid])[0, 1]

# Reward loop (could be vectorized):
for i in range(n_trials):
    reward_in_trial = np.any(
        (reward_timestamps >= trial_start_time) &
        (reward_timestamps <= trial_end_time)
    )
```

iii. The per-cell correlation could be replaced with a single matrix correlation computation. The per-trial loops for building trial data are harder to vectorize due to variable trial lengths.

## 13-c. What processing does the code repeat multiple times?

i. The lick data is processed twice conceptually: first for sensor error correction (setting NaN), then for binarization. Speed is also processed twice (once for interneuron correlation, once for the output). Position is used both for distance-to-reward and for absolute position outputs.

ii.
```python
# Speed processed for interneuron detection:
speed_valid = speed.copy()
speed_valid[speed_valid < 0] = 0
# Speed processed again for output:
trial_speed = speed[start:stop].astype(np.float32)
trial_speed = np.clip(trial_speed, 0, None)
```

iii. The duplication is minor and doesn't significantly impact performance.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads the `environment` and `trial_number` signals from the NWB file but doesn't use them (environment type is derived from scene name parsing instead). The `reward_zone_signal` is loaded but not used for determining reward zones. Speed is loaded and discretized as an output variable, but the reference code's speed filtering (masking out <2 cm/s timepoints) is not applied.

ii.
```python
# Loaded but unused:
reward_zone_signal = behav['reward_zone/data'][()]
environment = behav['environment/data'][()]
trial_number = behav['trial number/data'][()]
```

iii. Loading unused variables wastes I/O time but doesn't affect correctness. The speed output (discretized into 5 bins) is included in the decoder outputs as specified in the instructions.
