# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `/app/data/` directory using `h5py` (not `pynwb`). It discovers subjects by listing `sub-*` directories, finds all `.nwb` files within each, and parses the session number from the filename. All sessions are processed unless `--sample` mode is used.

ii.
```python
data_dir = '/app/data'
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
# ...
for subj in subjects:
    subj_dir = os.path.join(data_dir, f'sub-{subj}')
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    for nwb_file in nwb_files:
        ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
        ses_num = int(ses_str)
        all_sessions.append((subj, ses_num, nwb_file))
# ...
with h5py.File(nwb_path, 'r') as f:
    fluor_group = f['processing']['ophys']['Fluorescence']
    # ...
```

iii. The AI documented finding 11 subjects and 152 sessions, matching the paper. Using `h5py` instead of `pynwb` is a valid choice for reading NWB files (which are HDF5 format).

## 1-b. How are the data split into subjects?

i. Subjects are identified from `sub-*` subdirectories in the data directory. Each subject maps to one or more NWB files.

ii.
```python
subjects = sorted([d.replace('sub-', '') for d in os.listdir(data_dir) if d.startswith('sub-')])
```

iii. The AI found 11 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Session numbers are parsed from the filenames.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
for nwb_file in nwb_files:
    ses_str = os.path.basename(nwb_file).split('ses-')[1].split('_')[0]
    ses_num = int(ses_str)
```

iii. Standard approach matching the data organization.

## 1-d. How are the data split into trials?

i. Trial start is identified by `trial_start == 1`, and trial end by `teleport == 1`. The AI uses `np.where` to find all indices where these values equal 1, then takes the minimum count to pair them.

ii.
```python
tstart_idx = np.where(trial_start_arr == 1)[0]
teleport_idx = np.where(teleport_arr == 1)[0]
n_trials = min(len(tstart_idx), len(teleport_idx))
tstart_idx = tstart_idx[:n_trials]
teleport_idx = teleport_idx[:n_trials]
```

iii. The AI notes using `trial_start` and `teleport` markers, consistent with the paper. However, the teleport detection finds ALL frames where `teleport == 1` rather than detecting the rising edge (onset) of the teleport signal. If teleport remains at 1 for multiple frames, this would produce too many endpoints. The `min(len(...), len(...))` pairing could mis-align trial starts with trial ends.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are skipped. Sessions with fewer than 2 valid trials or 2 valid cells are also skipped.

ii.
```python
if n_tp < 2:
    continue
# ...
if len(neural_trials) < 2:
    print(f"    Skipping: only {len(neural_trials)} valid trials after filtering")
    return None
```

iii. This is a minimal filter. No explicit quality-based trial filtering is applied beyond minimum length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p traces: `Fluorescence` (F) and `Neuropil` (Fneu). The AI correctly avoids using the NWB `Deconvolved` field.

ii.
```python
fluor_group = f['processing']['ophys']['Fluorescence']
# ...
for pk in plane_keys:
    F_planes.append(f['processing']['ophys']['Fluorescence'][pk]['data'][()].T)
    Fneu_planes.append(f['processing']['ophys']['Neuropil'][pk]['data'][()].T)
```

iii. The AI recognized that the NWB Deconvolved data is suite2p's own deconvolution, not the paper's custom processing pipeline.

## 2-b. How is the `neural` data processed?

i. The AI computes dF/F from F and Fneu following the paper's method: neuropil subtraction (0.7 * Fneu), add back neuropil mean per trial, maximin baseline (Gaussian smoothing sigma=15, minimum filter 300, maximum filter 300), then dF/F = (F - baseline) / |baseline|, smooth with 2-sample Gaussian. **Critically, the AI does NOT apply OASIS deconvolution** -- it uses dF/F as the final neural signal rather than deconvolved "events." Additionally, the AI does NOT handle the `keep_teleports` parameter that determines whether baseline windows span the inter-trial interval for certain sessions. The AI also concatenates all planes' data before computing dF/F rather than processing each plane separately, and does not apply the +1 index offset that the reference code uses to align dff's internal windowing with the emitted trial boundaries.

ii.
```python
def compute_dff(F, Fneu, trial_start_idx, teleport_idx, neu_coef=0.7):
    f_ -= neu_coef * f_neu_
    # Per trial:
    f_[:, start:stop] = f_[:, start:stop] + neu_coef * np.nanmean(f_neu_[:, start:stop], axis=1, keepdims=True)
    flow[:, start:stop] = nansmooth(trial_data, 15, axis=1)
    flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], 300, axis=-1)
    flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], 300, axis=-1)
    dff[valid] = (f_[valid] - flow[valid]) / np.abs(flow[valid])
    # Per trial smooth:
    dff[:, start:stop] = nansmooth(dff[:, start:stop], 2, axis=1)
    return dff  # NO deconvolution
```

iii. The AI's CONVERSION_NOTES say "Neural signal: dF/F (not deconvolved)" and justify avoiding the suite2p Deconvolved field. However, the paper's pipeline goes further: it computes dF/F and THEN deconvolves with OASIS to get "events" (pseudo-spikes). The reference code copies the paper's `dff()` function which includes `deconvolve=True` using `dcnv.oasis()`. The AI stopped at dF/F without completing the deconvolution step.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` classification from suite2p (manual curation), (2) putative interneuron exclusion based on Pearson correlation between dF/F and running speed > 0.5.

ii.
```python
cell_mask = iscell[:, 0] == 1
# ...
dff_cells = dff[cell_mask]
is_interneuron = detect_interneurons(dff_cells, speed, threshold=0.5)
valid_cell_indices = cell_indices[~is_interneuron]
```

iii. Matches the paper's methods: manual curation via iscell and exclusion of putative interneurons with speed correlation > 0.5. However, the AI loads iscell from a single `ImageSegmentation/PlaneSegmentation/iscell` table rather than per-plane ROI tables, which could be incorrect for multi-plane sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. Data is extracted from `tstart_idx[t]` to `teleport_idx[t]` for each trial, which inherently aligns to trial start.

ii.
```python
neural = dff_valid[:, start:stop].astype(np.float32)
```

iii. Since alignment is to trial start and data is sliced at trial boundaries, no additional alignment step is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Data is kept at the native imaging frame rate (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied.

ii.
```python
'time_bin_size': 64.5,  # ms, approximate
```

iii. The native frame rate matches the paper's stated ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position` behavior time series timestamps.

ii.
```python
timestamps = beh['position']['timestamps'][()]
# ...
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. Behavior timestamps are consistent across all behavioral variables.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial-start timestamp is subtracted from all timestamps within the trial.

ii.
```python
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. Standard approach.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indexing (same number of frames), so indexing by the same trial boundaries ensures alignment.

ii.
```python
neural = dff_valid[:, start:stop]
time_from_start = timestamps[start:stop] - timestamps[start]
```

iii. Both use the same `start:stop` slice.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
environment = beh['environment']['data'][()]
# ...
env_per_trial[t] = int(np.median(env_vals))
```

iii. The `environment` variable records ENV1 (0) vs ENV2 (1).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, the AI takes the median of environment values within the trial, excluding any -1 values.

ii.
```python
env_vals = environment[start:stop]
env_vals = env_vals[env_vals >= 0]  # exclude -1
if len(env_vals) > 0:
    env_per_trial[t] = int(np.median(env_vals))
```

iii. Since environment is constant within a trial (either 0 or 1), the median should equal the actual value.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the 0-indexed loop counter over trials within a session.

ii.
```python
for t in range(n_trials):
    inp[2, :] = t  # trial number (0-indexed)
```

iii. Sequential trial index within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index. The value is constant across all timepoints within a trial.

ii.
```python
inp[2, :] = t
```

iii. Straightforward assignment.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps. For each trial, the AI checks whether any reward event occurred within that trial's time boundaries using the reward delivery timestamps.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
# ...
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. The `Reward` time series has its own timestamps separate from the behavior sampling rate.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, check whether any reward timestamp falls within the previous trial's time window. For the first trial, set to 0.

ii.
```python
prev_outcome = float(rewarded_trials[t-1]) if t > 0 else 0.0
inp[3, :] = prev_outcome
```

iii. Binary per-trial value, matching the instructions (omitted=0, rewarded=1).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone coordinates. The reward zone for each trial is determined from the `SESSIONS_SCENES` dictionary (derived from `sessions_dict.py`), which maps each session to a scene name containing the reward zone location. For switch sessions, the zone changes at trial 30 (`change_trial=30`).

ii.
```python
SESSIONS_SCENES = {
    'm11': {1: 'Env1_LocationB', ...},
    ...
}
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    if 'LocationA' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['A'], 'A'
    elif '_to_' in scene:
        # Parse pre and post zones
        if trial_idx < change_trial:
            return REWARD_ZONE_DICT[pre_zone], pre_zone
        else:
            return REWARD_ZONE_DICT[post_zone], post_zone
```

iii. The AI used the reference code's session metadata to determine reward zones deterministically rather than inferring from the data (as the reference solution does with a Viterbi algorithm on reward zone positions).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint, the signed distance from the animal's position to the nearest edge of the reward zone is computed. Distance is 0 when inside the zone, negative when before, positive when past.

ii.
```python
dist_to_rz = np.where(trial_pos < rz_start_cm, trial_pos - rz_start_cm,
                      np.where(trial_pos > rz_end_cm, trial_pos - rz_end_cm, 0.0))
```

iii. Same signed distance computation as the reference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The continuous distance is discretized into 7 bins using explicit conditional assignment.

ii.
```python
def discretize_distance_to_rz(distance):
    bins = np.full_like(distance, -1, dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bin boundaries match the instructions. The boundary behavior differs slightly from the reference's `np.digitize` approach (e.g., `>= -50` vs `> -50`, `<= 10` vs `< 10`), but the differences are at exact boundary values which are rare in continuous data.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii.
```python
trial_pos = position[start:stop]
```

iii. Same `start:stop` slice as neural data.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = beh['position']['data'][()]
trial_pos = position[start:stop]
```

iii. Direct position recording from the VR corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
pos_bins = discretize_position(trial_pos)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each spanning the 450 cm track.

ii.
```python
def discretize_position(position):
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Matches the instructions for 5 equal-sized bins over 450 cm.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data.

ii. Same `start:stop` indexing.

iii. Verified by shared timestamps.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh['lick']['data'][()]
trial_lick = lick[start:stop]
```

iii. Direct lick recording.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value mapped to 1, otherwise 0.

ii.
```python
lick_binary = (trial_lick > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii. Same `start:stop` indexing.

iii. Same timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `SESSIONS_SCENES` dictionary (from `sessions_dict.py`), which maps each session to a scene name. The scene name encodes the reward zone location. For switch sessions, the zone changes at trial 30.

ii.
```python
coords, label = get_reward_zone_for_trial(scene, t, n_trials, change_trial=30)
rz_labels.append(label)
```

iii. Uses the experimental design metadata rather than inferring from the `reward_zone` behavioral variable.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene names are parsed to extract the reward zone letter (A, B, or C). For switch sessions (`_to_` in scene name), the pre-switch zone is used for trials before trial 30, and the post-switch zone for trials at or after trial 30. Zone letters are mapped to integers: A=0, B=1, C=2.

ii.
```python
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    if 'LocationA' in scene and '_to_' not in scene:
        return REWARD_ZONE_DICT['A'], 'A'
    # ...
    elif '_to_' in scene:
        if trial_idx < change_trial:
            return REWARD_ZONE_DICT[pre_zone], pre_zone
        else:
            return REWARD_ZONE_DICT[post_zone], post_zone
```

iii. The paper states "Each switch occurred after 30 trials," so `change_trial=30` matches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps in the behavior data.

ii.
```python
reward_ts = beh['Reward']['timestamps'][()]
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
```

iii. Reward delivery events have their own timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check whether any reward timestamp falls within the trial's time window. Binary output: 0 = no reward, 1 = reward.

ii.
```python
rewarded_trials[t] = np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time))
reward_outcome = int(rewarded_trials[t])
out[5, :] = reward_outcome
```

iii. Per-trial binary value broadcast to all timepoints.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Truncated to minimum of the two.
- **Short trials**: Trials with < 2 timepoints are skipped.
- **NaN neural data**: Replaced with 0 using `np.nan_to_num`.
- **Sessions with < 2 trials or < 2 cells**: Skipped entirely.

ii.
```python
n_timepoints = min(n_timepoints_neural, n_timepoints_beh)
# ...
if n_tp < 2:
    continue
# ...
neural = np.nan_to_num(neural, nan=0.0)
```

iii. The neural NaN replacement with 0 is a notable choice -- the reference solution instead asserts that no NaN values exist within trials (since the dF/F pipeline should produce valid values for all on-trial timepoints).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with h5py and reading large arrays.
2. Computing dF/F for each session (neuropil subtraction, baseline, smoothing).
3. Interneuron detection (correlation computation for each cell).

ii. N/A

iii. The AI reports ~13 minutes for the full conversion of 152 sessions.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial for reward determination and data extraction. The interneuron detection loop iterates over each cell. The discretization functions are already vectorized.

ii. N/A

iii. The per-trial and per-cell loops are the natural structure. The interneuron detection's manual Pearson correlation could use `np.corrcoef` on the full matrix.

## 13-c. What processing does the code repeat multiple times?

i. The code does not have a separate survey step, so each NWB file is loaded only once. However, the reward zone determination from scene names is done per-trial even though it only changes once per session (at trial 30 for switch sessions).

ii. N/A

iii. The single-pass approach is more efficient than the reference's two-pass survey+conversion.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The full dF/F is computed for all cells (including non-iscell ROIs) before filtering, since the iscell mask is applied after dF/F computation. The reference solution subsets to iscell cells before computing dF/F, saving significant computation.

ii.
```python
F_raw = np.concatenate(F_planes, axis=0)  # ALL rois
# ...
dff = compute_dff(F_raw, Fneu_raw, ...)  # dF/F for ALL rois
dff_cells = dff[cell_mask]  # THEN filter to iscell
```

iii. Computing dF/F for non-cell ROIs (~2/3 of all ROIs) wastes significant computation time.
