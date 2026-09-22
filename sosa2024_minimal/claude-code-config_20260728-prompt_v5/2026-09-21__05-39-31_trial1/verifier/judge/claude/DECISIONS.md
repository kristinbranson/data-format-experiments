# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a hardcoded `SESSIONS_META` dictionary that maps each subject to a list of `(session_number, scene, exp_day)` tuples. It iterates over this dictionary, constructs NWB file paths based on subject ID and experiment day, and loads each NWB file using `h5py` (not `pynwb`). All 11 subjects and their sessions are included. The AI loads behavioral data (position, speed, lick, trial_start, teleport, timestamps, reward) and neural data (Fluorescence, Neuropil, iscell) from each file.

ii.
```python
SESSIONS_META = {
    'm3': [
        (1, 'Env1_LocationC', 1), (2, 'Env1_LocationC', 2),
        ...
    ],
    ...
}

for sub_idx, subject_id in enumerate(subjects):
    sub_dir = os.path.join(DATA_DIR, f'sub-{subject_id}')
    ...
    for ses_num, scene, exp_day in sessions:
        nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
        nwb_path = os.path.join(sub_dir, nwb_filename)
        ...
        result = process_session(nwb_path, subject_id, scene, exp_day)
```

```python
with h5py.File(nwb_path, 'r') as f:
    bts = f['processing']['behavior']['BehavioralTimeSeries']
    position = bts['position']['data'][:]
    speed = bts['speed']['data'][:]
    ...
```

iii. The AI examined the NWB files and the paper's `sessions_dict.py` to build the `SESSIONS_META` dictionary. It chose to hardcode all session metadata rather than discover sessions from the filesystem.

## 1-b. How are the data split into subjects?

i. Subjects are defined from the hardcoded `SESSIONS_META` dictionary keys, sorted alphabetically. The subject list is `['m11', 'm12', 'm13', 'm14', 'm15', 'm17', 'm18', 'm19', 'm3', 'm4', 'm7']`.

ii.
```python
subjects = sorted(SESSIONS_META.keys())
```

iii. The AI derived subject IDs from the paper's session metadata rather than discovering them from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are enumerated from the hardcoded `SESSIONS_META` dictionary for each subject.

ii.
```python
sessions = SESSIONS_META[subject_id]
for ses_num, scene, exp_day in sessions:
    nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
```

iii. The AI hardcoded session metadata from the paper's `sessions_dict.py` to map session numbers to scene names and experiment days.

## 1-d. How are the data split into trials?

i. Trial starts are identified where `trial_start_signal > 0` and trial ends where `teleport_signal > 0`. The AI uses `np.where(teleport_signal > 0)[0]` to find all positive teleport frames, then takes the first `min(n_starts, n_teleports)` of each, and validates that each start comes before its paired end. This differs from the reference which detects the rising edge of the teleport signal.

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
```

iii. The AI checked for trial_start and teleport signals in the NWB behavioral data. It validated that trial starts precede trial ends.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than 2 timepoints are excluded. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
if n_timepoints < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"    Skipping: fewer than 2 valid trials after processing")
    return None
```

iii. The AI set a minimal threshold of 2 timepoints per trial, much lower than the reference's 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw `Fluorescence` (F) and `Neuropil` (Fneu) traces, loaded per plane from the NWB file. The AI correctly identifies that the NWB `Deconvolved` field is not used by the paper.

ii.
```python
fluor_group = f['processing']['ophys']['Fluorescence']
neuro_group = f['processing']['ophys']['Neuropil']
for p in sorted(unique_planes if n_planes > 1 else [0]):
    plane_key = f'plane{int(p)}'
    F_list.append(fluor_group[plane_key]['data'][:].T)
    Fneu_list.append(neuro_group[plane_key]['data'][:].T)
F = np.concatenate(F_list, axis=0)
Fneu = np.concatenate(Fneu_list, axis=0)
```

iii. The AI examined the NWB file structure and the paper's preprocessing code, recognizing that the paper computes its own dF/F and deconvolution from raw F and Fneu.

## 2-b. How is the `neural` data processed?

i. The AI reimplements the paper's dF/F pipeline: (1) mask to within-trial data only, (2) neuropil subtraction with coefficient 0.7, (3) add back per-trial neuropil mean, (4) maximin baseline (Gaussian smooth sigma=15, then 300-sample min then max filter), (5) dF/F = (F_corr - baseline) / |baseline|, (6) smooth with 2-sample Gaussian, (7) OASIS deconvolution with tau=0.7 and frame_rate/n_planes.

Key difference: the AI's `nansmooth` uses `gaussian_filter1d` (1D smoothing), while the reference uses the paper's original `nansmooth` which calls `gaussian_filter` with `[0, 15]` (2D: sigma=0 on cell axis, sigma=15 on time axis). Also, the AI sets NaN weights to 0 in its nansmooth while the reference uses 0.001.

Another key difference: the AI does NOT handle `keep_teleports`. The reference copies the paper's `teleport_metadata.py` to determine which sessions should include teleport periods in the baseline window, while the AI always excludes them.

ii.
```python
def compute_dff_and_deconvolve(F, Fneu, trial_starts, trial_ends, frame_rate, n_planes=1):
    ...
    # Copy only within-trial data
    for start, stop in zip(trial_starts, trial_ends):
        f_[:, start:stop] = F[:, start:stop]
    ...
    # Neuropil subtraction
    f_[:, nanmask] = f_[:, nanmask] - NEU_COEF * f_neu_[:, nanmask]
    ...
    # Add back neuropil mean per trial
    f_[:, start:stop] = f_[:, start:stop] + NEU_COEF * np.nanmean(
        f_neu_[:, start:stop], axis=1, keepdims=True)
    # Maximin baseline
    flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
    flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
    flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
    ...
    # dF/F
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    # Smooth and deconvolve
    dff[:, start:stop] = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=1)
    spks[:, start:stop] = dcnv.oasis(dff[:, start:stop], 2000, TAU, frame_rate / n_planes)
```

iii. The AI examined the paper's preprocessing.py code and replicated the pipeline. It did not discover the `teleport_metadata.py` data about which sessions keep teleports.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) `iscell` from suite2p manual curation, and (2) putative interneuron removal based on Pearson correlation between dF/F and speed > 0.5. The AI uses `scipy.stats.pearsonr` rather than `np.corrcoef`.

ii.
```python
cell_mask = iscell[:, 0] == 1
cell_indices = np.where(cell_mask)[0]
F_cells = F[cell_indices, :]
Fneu_cells = Fneu[cell_indices, :]
...
for n_idx in range(dff.shape[0]):
    dff_valid = dff[n_idx, valid_mask]
    if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
        r, _ = pearsonr(dff_valid, speed_valid)
        if r > INTERNEURON_SPEED_CORR_THR:
            keep_neuron[n_idx] = False
```

iii. The AI recognized both filters from the paper's Methods section.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Data is aligned to the trial start. The trial's neural data is sliced from `tstart_inds[i]` to `teleport_inds[i]`, so time 0 corresponds to the first frame of the trial. Any remaining NaN values in the neural data are replaced with 0.

ii.
```python
trial_neural = spks[:, t0:t1]
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The instructions specify alignment to trial start. The NaN replacement with 0 is a defensive measure.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The time bin size is computed as `1000.0 / imaging_rate` and the median across sessions is used. The AI reads `imaging_rate` from the NWB `imaging_plane` field.

ii.
```python
imaging_rate = f['acquisition']['TwoPhotonSeries']['imaging_plane']['imaging_rate'][()]
...
time_bin_sizes.append(1000.0 / info['imaging_rate'])
median_time_bin = np.median(time_bin_sizes) if time_bin_sizes else 64.5
```

iii. The AI uses the imaging_rate field directly. For multi-plane sessions, the `imaging_rate` field may represent the per-plane rate rather than the scanner rate, leading to a potentially different time bin size calculation compared to the reference.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `timestamps` associated with the `position` behavioral time series.

ii.
```python
timestamps = bts['position']['timestamps'][:]
...
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. The AI uses position timestamps as the time reference.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The initial timestamp for the trial is subtracted from each timepoint's timestamp.

ii.
```python
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. Standard approach to compute elapsed time within a trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same frame indices within each trial (same `t0:t1` slice), so no additional alignment is needed.

ii.
```python
t0, t1 = tstart_inds[i], teleport_inds[i]
trial_neural = spks[:, t0:t1]
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. Both behavioral and neural data are indexed by the same frame indices.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI derives environment type from the hardcoded `SESSIONS_META` scene name (e.g., 'Env1_LocationC'), NOT from the NWB `environment` behavioral time series.

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

iii. The AI parsed scene names from the session metadata to determine environment type. For cross-environment switch sessions, it uses a hardcoded `change_trial=30` threshold to determine when the switch occurs.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. String parsing of scene names. For cross-environment switch sessions (e.g., 'Env1_C_to_Env2_B'), the environment switches at trial 30 from the pre-switch to post-switch environment.

ii.
```python
env_type = get_environment_per_trial(scene, i)
trial_input = np.array([
    ...
    np.full(n_timepoints, env_type, dtype=float),
    ...
])
```

iii. The AI assumed a fixed switch point at trial 30 for cross-environment sessions.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session loop index (0, 1, 2, ...).

ii.
```python
trial_number = float(i)
```

iii. Sequential trial index within each session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond using the loop counter. The value is constant across all timepoints within a trial.

ii.
```python
trial_number = float(i)
np.full(n_timepoints, trial_number, dtype=float)
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` behavioral time series. Reward event timestamps are compared to trial time windows.

ii.
```python
reward_data = bts['Reward']['data'][:]
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

iii. The AI checks whether any reward event timestamp falls within each trial's time window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For trial i, the previous trial outcome is `trial_rewarded[i-1]`. For the first trial (i=0), the AI defaults to 1 (rewarded), while the reference defaults to 0.

ii.
```python
if i == 0:
    prev_outcome = 1.0  # first trial: no previous, default to rewarded
else:
    prev_outcome = float(trial_rewarded[i - 1])
```

iii. The AI chose to default the first trial's previous outcome to 1 (rewarded). No justification was given in the trajectory for this choice.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavioral time series and the reward zone boundaries from the hardcoded `REWARD_ZONES` dictionary. The reward zone for each trial is determined by parsing the `scene` name from `SESSIONS_META`, using the `get_reward_zone_for_trial()` function which handles switch sessions with a `change_trial=30` threshold.

ii.
```python
REWARD_ZONES = {
    'A': (80, 130),
    'B': (200, 250),
    'C': (320, 370),
}
...
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
rz_start, rz_end = REWARD_ZONES[rz_label]
```

iii. The AI used the paper's reward zone positions and parsed scene names to determine which zone is active for each trial.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed: negative when before the zone, 0 when inside, positive when past. Then discretized into 7 bins.

ii.
```python
def discretize_distance_to_reward(position, rz_start, rz_end):
    dist = np.where(
        position < rz_start, position - rz_start,
        np.where(position > rz_end, position - rz_end, 0.0)
    )
    bins = np.zeros(len(dist), dtype=int)
    bins[dist < -50] = 0
    bins[(dist >= -50) & (dist < -10)] = 1
    bins[(dist >= -10) & (dist < 0)] = 2
    bins[dist == 0] = 3  # inside zone
    bins[(dist > 0) & (dist <= 10)] = 4
    bins[(dist > 10) & (dist <= 50)] = 5
    bins[dist > 50] = 6
    return bins
```

iii. The processing matches the instruction specification for distance bins.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. 7 bins using explicit conditional logic: `< -50`, `-50 to -10`, `-10 to 0`, `== 0` (in zone), `0 to 10`, `10 to 50`, `> 50`.

ii. See 7-b code above.

iii. Matches the instruction-specified bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices (t0:t1) as neural data. No additional alignment needed.

ii.
```python
pos_trial = position[t0:t1]
dist_to_rz = discretize_distance_to_reward(pos_trial, rz_start, rz_end)
```

iii. Behavioral and neural data share frame indexing.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavioral time series.

ii.
```python
position = bts['position']['data'][:]
...
pos_trial = position[t0:t1]
```

iii. Direct use of the position variable from the NWB file.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Discretized into 5 bins using `np.clip((position / 90).astype(int), 0, 4)`. This divides by 90 (bin width for 450 cm / 5 bins), truncates to int, and clips to [0, 4].

ii.
```python
def discretize_position(position):
    bins = np.clip((position / 90).astype(int), 0, 4)
    return bins
```

iii. The AI used integer division by bin width. This produces slightly different bin boundaries than `np.digitize` with `[-inf, 90, 180, 270, 360, inf]`: for example, a position of exactly 90 would be bin 1 with both approaches, but negative positions would be clipped to 0 rather than falling in a separate bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. 5 bins: `[0, 90)`, `[90, 180)`, `[180, 270)`, `[270, 360)`, `[360, inf)`. Values below 0 are clipped to bin 0.

ii. See 8-b code above.

iii. Equivalent to 5 equal 90 cm bins spanning the 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices (t0:t1) as neural data. No additional alignment needed.

ii.
```python
pos_trial = position[t0:t1]
abs_position = discretize_position(pos_trial)
```

iii. Same time indexing as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavioral time series. The AI also applies a lick sensor error correction.

ii.
```python
lick = bts['lick']['data'][:]
...
lick_corrected = np.copy(lick)
for i in range(n_trials):
    t0, t1 = tstart_inds[i], teleport_inds[i]
    trial_licks = lick_corrected[t0:t1]
    n_frames_trial = len(trial_licks)
    if n_frames_trial > 0:
        frac_high = np.sum(trial_licks > 2) / n_frames_trial
        if frac_high > 0.3:
            lick_corrected[t0:t1] = np.nan
```

iii. The AI applied a lick sensor error correction from the paper: trials where >30% of frames have cumulative lick count > 2 have their lick data set to NaN.

## 9-b. What processing is involved in computing `output` *Lick*?

i. After the sensor error correction, lick is binarized: any value > 0 maps to 1, NaN maps to 0.

ii.
```python
lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. The instructions specify binary output (no/yes). NaN values from sensor error correction are treated as no-lick.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices as neural data.

ii.
```python
lick_trial = lick_corrected[t0:t1]
```

iii. Same time indexing.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the hardcoded `SESSIONS_META` scene names, NOT from the NWB `reward_zone` behavioral time series. The `get_reward_zone_for_trial()` function parses scene strings to determine the reward zone label.

ii.
```python
def get_reward_zone_for_trial(scene, trial_idx, n_trials, change_trial=30):
    if 'LocationA' in scene and '_to_' not in scene ...:
        return 'A'
    ...
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

iii. The AI used session metadata to determine reward zones rather than the NWB data. For switch sessions, it assumes the switch happens at trial 30 (a hardcoded value).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. String parsing of scene names to extract zone labels, with a hardcoded trial 30 switch point for switch sessions. Mapped to indices: A=0, B=1, C=2.

ii.
```python
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
rz_location = rz_label_to_idx(rz_label)
```

iii. The AI's approach assumes a fixed switch point rather than detecting it from the data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` behavioral time series timestamps.

ii.
```python
reward_data = bts['Reward']['data'][:]
reward_timestamps = bts['Reward']['timestamps'][:]
```

iii. Direct use of reward event timestamps from the NWB file.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any reward event timestamp falls within the trial's time window. Binary: 1 if rewarded, 0 otherwise.

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
reward_outcome = trial_rewarded[i]
```

iii. The AI matches reward timestamps to trial windows.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Neural/behavior length mismatch**: Data is cropped to the minimum length.
- **Short trials**: Trials with fewer than 2 timepoints are skipped.
- **NaN in neural data**: Replaced with 0.0 using `np.nan_to_num`.
- **Lick sensor errors**: Trials with >30% of frames having cumulative lick > 2 have lick data set to NaN (then treated as 0).
- **Invalid trial boundaries**: Trials where start >= end are filtered out.

ii.
```python
n_frames = min(len(position), F.shape[1])
...
if n_timepoints < 2:
    continue
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
frac_high = np.sum(trial_licks > 2) / n_frames_trial
if frac_high > 0.3:
    lick_corrected[t0:t1] = np.nan
```

iii. These are defensive measures the AI applied during data exploration and debugging.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files with `h5py` and reading large arrays (neural data)
2. Computing dF/F and deconvolution for each session (the `compute_dff_and_deconvolve` function)
3. Computing per-neuron speed correlations for interneuron filtering

ii. N/A

iii. The NWB files contain full neural recordings that must be loaded entirely.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron interneuron correlation loop (iterating over each cell to compute `pearsonr`) could be vectorized using `np.corrcoef` as the reference does. The per-trial loop for building output arrays is necessary due to variable trial lengths.

ii.
```python
for n_idx in range(dff.shape[0]):
    dff_valid = dff[n_idx, valid_mask]
    if np.std(dff_valid) > 0 and np.std(speed_valid) > 0:
        r, _ = pearsonr(dff_valid, speed_valid)
```

iii. `np.corrcoef` is more efficient for batch correlation computation.

## 13-c. What processing does the code repeat multiple times?

i. The code processes each session independently in a single pass. There is no survey step or repeated loading, unlike the reference which loads data twice (survey + conversion).

ii. N/A

iii. The AI's single-pass approach is more efficient in this regard.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The lick sensor error correction is additional processing not present in the reference. The AI also computes `dff` (returned alongside `spks` from `compute_dff_and_deconvolve`) but only uses `spks` for the final output; `dff` is only used for interneuron filtering. The hardcoded `SESSIONS_META` dictionary contains scene information that is used to derive environment type and reward zone, adding complexity that the reference avoids by reading directly from NWB data.

ii. N/A

iii. The lick correction is from the paper but may not be needed for the decoder task.
