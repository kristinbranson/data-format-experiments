# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the list of subjects and session metadata in `SESSIONS_META`, then iterates that table to construct each NWB filename. Each session is opened directly with `h5py`, and the script reads behavior time series plus fluorescence / neuropil arrays from the NWB HDF5 groups. Trials are not loaded separately from disk; they are carved out later from the full-session arrays.

ii.
```python
SESSIONS_META = {
    'm3': [
        (1, 'Env1_LocationC', 1), ...
    ],
    ...
}

subjects = sorted(SESSIONS_META.keys())
...
for ses_num, scene, exp_day in sessions:
    nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
    nwb_path = os.path.join(sub_dir, nwb_filename)
    result = process_session(nwb_path, subject_id, scene, exp_day)
```

```python
with h5py.File(nwb_path, 'r') as f:
    bts = f['processing']['behavior']['BehavioralTimeSeries']
    position = bts['position']['data'][:]
    speed = bts['speed']['data'][:]
    lick = bts['lick']['data'][:]
    ...
    fluor_group = f['processing']['ophys']['Fluorescence']
    neuro_group = f['processing']['ophys']['Neuropil']
```

iii. In the trajectory, the AI says it cross-referenced the data directory with `sessions_dict.py`, matched the 11 switch-task mice, and then planned to load each NWB file directly after inspecting the HDF5 structure with `h5py` (steps 20, 22, 29, 30, 81-84).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the hard-coded keys of `SESSIONS_META`, sorted lexicographically. The script therefore treats each listed mouse ID (`m11`, `m12`, ..., `m7`) as one subject.

ii.
```python
subjects = sorted(SESSIONS_META.keys())

for sub_idx, subject_id in enumerate(subjects):
    sub_dir = os.path.join(DATA_DIR, f'sub-{subject_id}')
```

iii. The trajectory says the AI matched the subject folders in `/app/data` to the paper's `sessions_dict.py` entries and concluded there were 11 switch-task mice to include (step 20).

## 1-c. How are the data split into sessions?

i. Each session is one NWB file specified by one tuple in `SESSIONS_META`. The script uses the `exp_day` value from that tuple as the NWB session number in the filename and appends one converted session entry per successful `process_session(...)` call.

ii.
```python
for ses_num, scene, exp_day in sessions:
    nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
    nwb_path = os.path.join(sub_dir, nwb_filename)
    ...
    result = process_session(nwb_path, subject_id, scene, exp_day)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The AI initially assumed session numbering was relative within each mouse, then corrected itself after checking the files and noting that NWB `ses-XX` matched `exp_day` directly, including `m11` starting at day 3 (steps 78-84).

## 1-d. How are the data split into trials?

i. Trial starts are all indices where `trial_start_signal > 0`. Trial ends are all indices where `teleport_signal > 0`. The script pairs those arrays in order, truncates them to equal length, removes any pair where the trial start is not before the teleport index, and then uses `[t0:t1]` slices as trials.

ii.
```python
tstart_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]

n_trials = min(len(tstart_inds), len(teleport_inds))
tstart_inds = tstart_inds[:n_trials]
teleport_inds = teleport_inds[:n_trials]

valid = []
for i in range(n_trials):
    if tstart_inds[i] < teleport_inds[i]:
        valid.append(i)
tstart_inds = tstart_inds[valid]
teleport_inds = teleport_inds[valid]
```

```python
for i in range(n_trials):
    t0, t1 = tstart_inds[i], teleport_inds[i]
    ...
    trial_neural = spks[:, t0:t1]
```

iii. In the trajectory, the AI explicitly states that it would define trials from the `trial_start` and `teleport` markers after inspecting those NWB fields and seeing equal counts in an example session (steps 20, 23, 24).

## 1-e. How are trials filtered based on quality controls?

i. Trials are minimally filtered. The script drops any trial pair where `trial_start` is not before `teleport`, skips sessions with fewer than two valid trials, and skips individual trials only if they contain fewer than two timepoints. There is no longer-duration trial QC.

ii.
```python
for i in range(n_trials):
    if tstart_inds[i] < teleport_inds[i]:
        valid.append(i)
...
if n_trials < 2:
    print(f"    Skipping: only {n_trials} valid trials")
    return None
...
if n_timepoints < 2:
    continue
```

iii. The trajectory does not record a separate justification for the `< 2` cutoff. The explicit rationale it does record is only that the decoder requires at least two trials per session and that trials should be defined from `trial_start`/`teleport` (steps 20, 33).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The converted `neural` data are derived from the NWB `Fluorescence` and `Neuropil` arrays, restricted to ROIs with `iscell[:, 0] == 1`. The script does not use the NWB `Deconvolved` field for the final neural output.

ii.
```python
iscell = f['processing']['ophys']['ImageSegmentation']['PlaneSegmentation']['iscell'][:]
...
fluor_group = f['processing']['ophys']['Fluorescence']
neuro_group = f['processing']['ophys']['Neuropil']
...
F = np.concatenate(F_list, axis=0)
Fneu = np.concatenate(Fneu_list, axis=0)
...
cell_mask = iscell[:, 0] == 1
F_cells = F[cell_indices, :]
Fneu_cells = Fneu[cell_indices, :]
```

iii. The AI says in the trajectory that the paper's NWB `Deconvolved` trace was not the exact paper-analysis signal and that it therefore needed to recompute activity from fluorescence plus neuropil using the paper's pipeline (steps 20, 27, 28).

## 2-b. How is the `neural` data processed?

i. The AI recomputes neural activity from scratch. It masks out all non-trial samples, subtracts `0.7 * Fneu`, adds back the within-trial neuropil mean, computes a per-trial maximin baseline, forms dF/F, smooths dF/F with a 2-sample Gaussian, and deconvolves each trial with `suite2p.extraction.dcnv.oasis(..., TAU=0.7, frame_rate / n_planes)`.

ii.
```python
def compute_dff_and_deconvolve(F, Fneu, trial_starts, trial_ends, frame_rate, n_planes=1):
    f_ = np.full((n_cells, n_frames), np.nan)
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F[:, start:stop]
    ...
    f_[:, nanmask] = f_[:, nanmask] - NEU_COEF * f_neu_[:, nanmask]
    ...
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = f_[:, start:stop] + NEU_COEF * np.nanmean(
            f_neu_[:, start:stop], axis=1, keepdims=True
        )
        flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
        flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
        flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
    ...
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    dff[:, start:stop] = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=1)
    spks[:, start:stop] = dcnv.oasis(
        dff[:, start:stop], 2000, TAU, frame_rate / n_planes
    )
```

iii. The trajectory says the AI chose to recompute the paper's custom signal instead of using the stored NWB `Deconvolved` data because the paper used its own neuropil-subtraction / dF/F / OASIS pipeline (steps 20, 27, 28).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron-level filters: keep only ROIs marked as cells by `iscell`, then remove putative interneurons whose dF/F is correlated with running speed above `0.5`.

ii.
```python
cell_mask = iscell[:, 0] == 1
F_cells = F[cell_indices, :]
Fneu_cells = Fneu[cell_indices, :]
...
valid_mask = ~np.isnan(dff[0, :])
speed_valid = speed[valid_mask]
keep_neuron = np.ones(dff.shape[0], dtype=bool)
for n_idx in range(dff.shape[0]):
    dff_valid = dff[n_idx, valid_mask]
    if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
        r, _ = pearsonr(dff_valid, speed_valid)
        if r > INTERNEURON_SPEED_CORR_THR:
            keep_neuron[n_idx] = False
```

iii. The trajectory explicitly lists these two filters as key decisions copied from the paper: Suite2p manual curation via `iscell`, plus putative interneuron exclusion with speed correlation `> 0.5` (steps 20, 27, 33).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start simply by slicing each trial from `t0 = trial_start` to `t1 = teleport` and resetting trial-relative time downstream. There is no extra offset or interpolation step.

ii.
```python
for i in range(n_trials):
    t0, t1 = tstart_inds[i], teleport_inds[i]
    ...
    trial_neural = spks[:, t0:t1]
```

iii. The trajectory repeatedly says the decoder should be aligned to the start of each trial and that trial splitting itself would supply that alignment (steps 20, 33).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native frame sampling and does not apply temporal rebinning. In metadata, it reports a session-agnostic time bin size as the median of `1000.0 / imaging_rate` across sessions.

ii.
```python
spks[:, start:stop] = dcnv.oasis(
    dff[:, start:stop], 2000, TAU, frame_rate / n_planes
)
```

```python
time_bin_sizes = []
for info in session_info:
    time_bin_sizes.append(1000.0 / info['imaging_rate'])
median_time_bin = np.median(time_bin_sizes) if time_bin_sizes else 64.5
...
'time_bin_size': median_time_bin,
```

iii. In the trajectory, the AI says it would use the native imaging frame rate, which it described as roughly 15.5 Hz or about 64.5 ms per bin, and not rebin the data (step 20). No additional justification for the metadata calculation is recorded.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. This input is derived from the behavioral timestamps attached to the `position` time series.

ii.
```python
timestamps = bts['position']['timestamps'][:]
...
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. The trajectory does not record a separate argument for choosing `position` timestamps over another behavior stream. The only recorded reasoning is that the decoder should use trial-start-aligned time in seconds (step 20).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the script subtracts the first timestamp of that trial from all timestamps in the slice.

ii.
```python
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. The trajectory frames this as alignment to trial start, so the subtraction is the direct implementation of that plan (steps 20, 33).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The AI aligns this input to neural data by using the same `[t0:t1]` indices for both neural and behavioral arrays, after first cropping neural and behavioral streams to the same minimum frame count.

ii.
```python
n_frames = min(len(position), F.shape[1])
position = position[:n_frames]
...
timestamps = timestamps[:n_frames]
F = F[:, :n_frames]
...
trial_neural = spks[:, t0:t1]
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. The only explicit trajectory justification is that the AI intended to align everything at the native frame rate and exclude teleport periods while keeping trial boundaries synchronized (steps 20, 50).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI does not derive environment from the NWB `environment` time series. Instead, it derives it from the hard-coded `scene` string in `SESSIONS_META` and the trial index for switch sessions.

ii.
```python
def get_environment_per_trial(scene, trial_idx, change_trial=30):
    if '_to_Env' in scene:
        parts = scene.split('_to_Env')
        if trial_idx < change_trial:
            return 0 if 'Env1' in parts[0] else 1
        else:
            return 0 if parts[1].startswith('1') else 1
    elif scene.startswith('Env2'):
        return 1
    else:
        return 0
```

```python
env_type = get_environment_per_trial(scene, i)
```

iii. The trajectory says the AI matched subject folders to `sessions_dict.py` and used session metadata such as scene names to determine environment structure across days and switch sessions (step 20).

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script parses the `scene` string. Non-switch sessions become constant `0` or `1`; cross-environment switch sessions change from one environment to the other at `trial_idx == 30`. The resulting scalar is then repeated across all timepoints in the trial.

ii.
```python
env_type = get_environment_per_trial(scene, i)
...
trial_input = np.array([
    time_from_start,
    np.full(n_timepoints, env_type, dtype=float),
    ...
])
```

iii. The trajectory justification is implicit: after reading the paper and `sessions_dict.py`, the AI decided it could infer environment identity from the session scene description instead of the NWB behavior variable (step 20). No additional defense of the hard-coded switch point is recorded.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is derived from the loop index over the already-segmented trials, so indirectly from `trial_start` and `teleport`.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = float(i)
```

iii. The trajectory says the AI would define trials from `trial_start` and `teleport`; using the sequential trial counter follows directly from that choice (step 20).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No additional processing is performed beyond converting the loop index to `float` and repeating it across all timepoints in the trial.

ii.
```python
trial_number = float(i)
...
np.full(n_timepoints, trial_number, dtype=float)
```

iii. No explicit separate justification is recorded in the trajectory.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from reward-event timestamps in the NWB `Reward` time series. The script first labels each trial as rewarded or not rewarded, then uses the previous trial's label.

ii.
```python
reward_timestamps = bts['Reward']['timestamps'][:]
...
trial_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    t_start_time = timestamps[tstart_inds[i]]
    t_end_time = timestamps[teleport_inds[i]]
    reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
    if np.any(reward_in_trial):
        trial_rewarded[i] = 1
```

iii. The trajectory says previous outcome was one of the required decoder inputs and that reward events should come from the NWB reward stream (step 20).

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial `i > 0`, the script uses `trial_rewarded[i - 1]`. For the first trial, it sets the previous outcome to `1.0` by default. The value is then repeated across the whole current trial.

ii.
```python
if i == 0:
    prev_outcome = 1.0  # first trial: no previous, default to rewarded
else:
    prev_outcome = float(trial_rewarded[i - 1])
...
np.full(n_timepoints, prev_outcome, dtype=float)
```

iii. The trajectory does not give an explicit justification for defaulting the first trial to rewarded; this rationale appears only as the in-code comment shown above (step 62).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The AI derives this output from the per-frame `position` time series plus a reward-zone identity inferred from hard-coded session metadata (`scene`) through `get_reward_zone_for_trial(...)`. It does not use the NWB `reward_zone` time series for the final computation.

ii.
```python
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    ...
    if trial_idx < change_trial:
        return before_zone
    else:
        return after_zone
```

```python
pos_trial = position[t0:t1]
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
rz_start, rz_end = REWARD_ZONES[rz_label]
dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
```

iii. In the trajectory, the AI first explored the NWB `reward_zone` signal, then concluded it was a sparse VR signal rather than the desired decoder target. It therefore decided to compute distance itself from position and the session's reward-zone schedule inferred from scene metadata (steps 25-27).

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The script computes a signed continuous distance to the nearest reward-zone edge: negative before the zone, zero inside it, positive after it. It immediately discretizes that distance into the requested categories.

ii.
```python
dist = np.where(
    position < rz_start, position - rz_start,
    np.where(position > rz_end, position - rz_end, 0.0)
)
```

iii. The trajectory says the AI concluded it had to compute distance-to-zone itself rather than read it directly from NWB, because the raw `reward_zone` field looked like a sparse VR-state signal (steps 25-27).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses hand-written boolean masks to produce 7 bins with boundaries `(-inf, -50, -10, 0, 0, 10, 50, inf)`, where exact zero is assigned to the in-zone category.

ii.
```python
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. The trajectory justification is simply that these were the decoder-task bins it intended to implement (step 20).

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is aligned by taking the same trial slice `[t0:t1]` from `position` that is used for the neural trial slice.

ii.
```python
trial_neural = spks[:, t0:t1]
pos_trial = position[t0:t1]
dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
```

iii. The trajectory says the decoder variables would be extracted after segmenting the full session into start-of-trial aligned trials (step 20).

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Absolute position comes directly from the NWB `position` behavior time series.

ii.
```python
position = bts['position']['data'][:]
...
pos_trial = position[t0:t1]
```

iii. The trajectory identifies position as one of the behavioral variables to extract directly from NWB after trial segmentation (step 20).

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The AI bins raw position by integer-dividing by `90` cm, converting to integer, and clipping to the range `0..4`.

ii.
```python
def discretize_position(position):
    bins = np.clip((position / 90).astype(int), 0, 4)
    return bins
```

iii. The trajectory justification is implicit from the decoder specification: five equal-sized bins over a 450 cm track (step 20).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is thresholded into five 90 cm bins: `[0, 90)`, `[90, 180)`, `[180, 270)`, `[270, 360)`, and everything else clipped into the final category.

ii.
```python
bins = np.clip((position / 90).astype(int), 0, 4)
```

iii. No separate trajectory justification is recorded beyond following the decoder specification.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same trial boundaries are used for both neural data and position.

ii.
```python
trial_neural = spks[:, t0:t1]
pos_trial = position[t0:t1]
abs_position = discretize_position(pos_trial)
```

iii. The trajectory says the AI would segment behavior and neural data into the same trial windows (step 20).

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Lick is derived from the NWB `lick` behavior time series.

ii.
```python
lick = bts['lick']['data'][:]
...
lick_trial = lick_corrected[t0:t1]
```

iii. The trajectory identifies licking as a behavioral stream to extract from NWB (step 20).

## 9-b. What processing is involved in computing `output` *Lick*?

i. Before binarization, the AI performs a per-trial lick-sensor correction: if more than 30% of frames in a trial have lick values greater than 2, it replaces the entire trial's lick values with `NaN`. It then converts remaining values to binary with `> 0`, and maps `NaN` to `0`.

ii.
```python
for i in range(n_trials):
    t0, t1 = tstart_inds[i], teleport_inds[i]
    trial_licks = lick_corrected[t0:t1]
    ...
    frac_high = np.sum(trial_licks > 2) / n_frames_trial
    if frac_high > 0.3:
        lick_corrected[t0:t1] = np.nan
```

```python
lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. The trajectory does not provide a separate narrative defense of this correction. The only explicit rationale appears in the code comment `# Paper: >30% of frames with cumulative lick count > 2` and in the later conversation summary that mentions lick-sensor correction (step 90 summary).

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick is aligned to neural data by using the same `[t0:t1]` trial slice.

ii.
```python
trial_neural = spks[:, t0:t1]
lick_trial = lick_corrected[t0:t1]
lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. The trajectory justification is the same shared trial segmentation plan recorded earlier (step 20).

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Reward-zone location is derived from the hard-coded `scene` metadata in `SESSIONS_META`, parsed by `get_reward_zone_for_trial(...)`, then mapped from `A/B/C` to `0/1/2`.

ii.
```python
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
...
def rz_label_to_idx(label):
    return {'A': 0, 'B': 1, 'C': 2}[label]
```

iii. The trajectory says the AI decided not to trust the NWB `reward_zone` field as a final label and instead infer zone identity from session scene metadata plus the assumed switch structure (steps 25-27).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The AI parses the scene name to determine the pre-switch and post-switch reward zones. For switch sessions it assumes the switch occurs at trial 30; otherwise it uses the fixed zone in the scene name. The resulting `A/B/C` label is converted to `0/1/2` and repeated across the trial.

ii.
```python
if '_to_Env' in scene:
    parts = scene.split('_to_')
    before_zone = parts[0].split('_')[-1]
    after_zone = parts[1].split('_')[-1]
elif '_to_' in scene:
    parts = scene.split('_to_')
    before_zone = parts[0].split('Location')[-1]
    after_zone = parts[1]
...
if trial_idx < change_trial:
    return before_zone
else:
    return after_zone
```

iii. The trajectory justification is that, after inspecting the data and paper code, the AI believed it could reconstruct the active reward-zone location from the session schedule in `sessions_dict.py` (steps 20, 25-27).

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Reward outcome is derived from the NWB `Reward` timestamps.

ii.
```python
reward_timestamps = bts['Reward']['timestamps'][:]
...
reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
if np.any(reward_in_trial):
    trial_rewarded[i] = 1
```

iii. The trajectory says reward events should come from the NWB reward stream and be turned into per-trial outcome labels (step 20).

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, the AI checks whether any reward timestamp falls between the trial's start time and end time. If yes, the trial's reward outcome is `1`; otherwise it is `0`. The scalar is repeated across all timepoints in the trial output array.

ii.
```python
trial_rewarded = np.zeros(n_trials, dtype=int)
for i in range(n_trials):
    t_start_time = timestamps[tstart_inds[i]]
    t_end_time = timestamps[teleport_inds[i]]
    reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
    if np.any(reward_in_trial):
        trial_rewarded[i] = 1
...
np.full(n_timepoints, reward_outcome, dtype=int)
```

iii. The trajectory contains no deeper justification beyond satisfying the required decoder output using the trial segmentation the AI had chosen.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several issues pragmatically. It crops behavior and neural arrays to the same minimum length, drops misordered `trial_start` / `teleport` pairs, skips sessions with too few trials or no surviving neurons, replaces neural `NaN`s with zeros inside emitted trials, warns and skips missing NWB files, and treats detected lick-sensor failures as missing then converts them to no-lick.

ii.
```python
n_frames = min(len(position), F.shape[1])
position = position[:n_frames]
...
F = F[:, :n_frames]
```

```python
if tstart_inds[i] < teleport_inds[i]:
    valid.append(i)
...
if n_trials < 2:
    return None
...
if n_neurons < 1:
    return None
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The trajectory explicitly records the crop-to-minimum-length edit and the session-numbering fix for `m11`. The remaining defensive handling is present in the code but not separately justified in the trajectory (steps 50, 78-84).

## 13-a. What are the most time-consuming steps of the code?

i. The most expensive work in this script is reading large NWB arrays from disk, recomputing dF/F plus OASIS deconvolution for every session, and computing the per-neuron speed correlations used for interneuron filtering.

ii.
```python
with h5py.File(nwb_path, 'r') as f:
    ...
    F_list.append(fluor_group[plane_key]['data'][:].T)
    Fneu_list.append(neuro_group[plane_key]['data'][:].T)
```

```python
dff, spks = compute_dff_and_deconvolve(
    F_cells, Fneu_cells, tstart_inds, teleport_inds, imaging_rate, n_planes
)
...
for n_idx in range(dff.shape[0]):
    dff_valid = dff[n_idx, valid_mask]
    ...
    r, _ = pearsonr(dff_valid, speed_valid)
```

iii. The trajectory emphasizes the choice to recompute the paper's full neural pipeline rather than reuse stored deconvolution, which implies that the dF/F and OASIS stage is a major cost center (steps 20, 27, 33).

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: copying trial windows into `f_` / `f_neu_`, per-neuron Pearson correlations, per-trial reward detection, per-trial lick correction, and the final per-trial assembly loop.

ii.
```python
for start, stop in zip(trial_starts, trial_ends):
    f_[:, start:stop] = F[:, start:stop]
...
for n_idx in range(dff.shape[0]):
    ...
for i in range(n_trials):
    reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
...
for i in range(n_trials):
    ...
    neural_trials.append(trial_neural)
```

iii. The trajectory does not explicitly discuss vectorization, so this is inferred directly from the AI's own implementation.

## 13-c. What processing does the code repeat multiple times?

i. The code repeats several passes over the same trial structure: one pass to build the NaN-masked fluorescence arrays, one pass to add neuropil means and compute baselines, one pass to smooth and deconvolve, one pass to label trial rewards, one pass to correct licks, and one final pass to package trials. It also copies both `F` and `Fneu` into trial-masked arrays separately.

ii.
```python
for start, stop in zip(trial_starts, trial_ends):
    f_[:, start:stop] = F[:, start:stop]
...
for start, stop in zip(trial_starts, trial_ends):
    f_neu_[:, start:stop] = Fneu[:, start:stop]
...
for start, stop in zip(trial_starts, trial_ends):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
for i in range(n_trials):
    ...
```

iii. No explicit trajectory justification is recorded for these repeated passes; they are a direct consequence of the AI's chosen implementation structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded computation is dF/F itself: it is fully computed but only used transiently to filter interneurons before the script saves deconvolved events. The script also loads `reward_data` but never uses it, computes `frame_time` but never consumes it later, and defines `get_environment(...)` without calling it.

ii.
```python
dff, spks = compute_dff_and_deconvolve(...)
...
dff = dff[keep_neuron, :]
...
trial_neural = spks[:, t0:t1]
```

```python
reward_data = bts['Reward']['data'][:]
...
frame_time = np.median(np.diff(timestamps))
...
def get_environment(scene):
    ...
```

iii. The trajectory gives no separate justification for these discarded computations.
