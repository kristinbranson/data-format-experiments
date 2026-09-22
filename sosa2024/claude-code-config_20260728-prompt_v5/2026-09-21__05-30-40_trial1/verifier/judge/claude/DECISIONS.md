# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by scanning sorted `sub-*` directories under `/app/data` and globbing `*.nwb` files within each. Each NWB file is opened with `h5py` (not `pynwb`) and the behavioral and neural data arrays are read directly from HDF5 paths.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
all_nwb_files = []
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
    nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
    all_nwb_files.extend(nwb_files)
...
with h5py.File(nwb_path, 'r') as f:
    ...
    bts = f['processing/behavior/BehavioralTimeSeries']
    position = bts['position/data'][:]
    ...
```

iii. The CONVERSION_NOTES document finding 11 subjects and 152 sessions, matching the paper. The AI chose `h5py` over `pynwb` for direct HDF5 access.

## 1-b. How are the data split into subjects?

i. Subjects correspond to `sub-*` directories. The subject ID is extracted by stripping the `sub-` prefix. Unique subjects are collected from session_info after processing.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
unique_subjects = sorted(set(si['subject'] for si in session_infos))
```

iii. The directory structure `sub-<id>` is used to identify subjects. 11 subjects match the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. All NWB files within a subject directory are treated as separate sessions.

ii.
```python
nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
all_nwb_files.extend(nwb_files)
```

iii. Each NWB file name contains a `ses-XX` identifier. 152 total sessions found.

## 1-d. How are the data split into trials?

i. Trials are defined by `trial_start` and `teleport` signals. Trial starts are all indices where `trial_start_signal > 0`. Trial ends are all indices where `teleport_signal > 0`. Both are converted to 1-indexed (+1). If the counts don't match, they are truncated to the shorter length.

ii.
```python
trial_start_inds = np.where(trial_start_signal > 0)[0] + 1  # convert to 1-indexed
teleport_inds = np.where(teleport_signal > 0)[0] + 1  # convert to 1-indexed

if len(trial_start_inds) != len(teleport_inds):
    min_len = min(len(trial_start_inds), len(teleport_inds))
    trial_start_inds = trial_start_inds[:min_len]
    teleport_inds = teleport_inds[:min_len]
```

iii. The CONVERSION_NOTES state that `trial_start` and `teleport` signals are used. The reference uses rising-edge detection for teleport (`(teleport[1:] > 0) & (teleport[:-1] <= 0)`), whereas the AI uses all frames where `teleport > 0`. If the teleport signal is multi-frame, this would yield too many teleport indices, but in practice the trial count (12,216) closely matches expectations.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials where the end index is less than or equal to the start index (`e <= s`), and sessions with fewer than 2 trials. There is no minimum trial length filter.

ii.
```python
if e <= s:
    continue
...
if len(neural_trials) < 2:
    print(f"    WARNING: Skipping session with {len(neural_trials)} trials")
    continue
```

iii. The CONVERSION_NOTES do not discuss a minimum trial length threshold. The reference code filters trials with fewer than 50 timepoints.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw Fluorescence (F) and Neuropil (Fneu) traces stored in the NWB file under `processing/ophys/Fluorescence/plane{N}/data` and `processing/ophys/Neuropil/plane{N}/data`. The NWB `Deconvolved` field is NOT used.

ii.
```python
ophys = f['processing/ophys']
...
for plane in planes:
    F_plane = ophys[f'Fluorescence/{plane}/data'][:]
    Fneu_plane = ophys[f'Neuropil/{plane}/data'][:]
```

iii. The CONVERSION_NOTES correctly identify that the NWB Deconvolved data is suite2p's raw deconvolution, not the paper's custom dF/F + OASIS pipeline. The AI decided to recompute from F and Fneu.

## 2-b. How is the `neural` data processed?

i. dF/F is computed from F and Fneu following the paper's pipeline: (1) extract F within trial boundaries (set rest to NaN), (2) subtract 0.7 * Fneu, (3) add back neuropil mean per trial, (4) maximin baseline (Gaussian smooth sigma=15, min filter 300 samples, max filter 300 samples), (5) dF/F = (F - baseline) / |baseline|, (6) smooth with 2-sample Gaussian, (7) deconvolve with OASIS (tau=0.7, frame_rate/n_planes). Cells from multiple planes are concatenated. The AI does NOT use the `keep_teleports` metadata from `teleport_metadata.py` -- it always restricts the baseline to within-trial windows.

ii.
```python
def compute_dff_and_deconvolve(F, Fneu, trial_starts, teleports, frame_rate, n_planes=1):
    ...
    start_inds = trial_starts.tolist()
    stop_inds = teleports.tolist()
    for start, stop in zip(start_inds, stop_inds):
        s = start - 1
        e = stop - 1
        f_[:, s:e] = F[:, s:e]
        f_neu_[:, s:e] = Fneu[:, s:e]
    f_ -= NEU_COEF * f_neu_
    ...
    for start, stop in zip(start_inds, stop_inds):
        ...
        f_[:, s:e] = f_[:, s:e] + NEU_COEF * np.nanmean(f_neu_[:, s:e], axis=1, keepdims=True)
        flow[:, s:e] = nansmooth(f_[:, s:e], 15, axis=1)
        flow[:, s:e] = minimum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
        flow[:, s:e] = maximum_filter1d(flow[:, s:e], BASELINE_WINDOW, axis=-1)
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    ...
    for start, stop in zip(start_inds, stop_inds):
        dff[:, s:e] = nansmooth(dff[:, s:e], DFF_SMOOTH_SIGMA, axis=1)
        ...
        spks[:, s:e] = oasis(trial_dff, 2000, TAU, frame_rate / n_planes)
```

iii. The CONVERSION_NOTES describe the dF/F pipeline accurately. The omission of `keep_teleports` is not documented as a deliberate decision; the CONVERSION_NOTES mention "Activity outside trial periods set to NaN (teleport periods excluded for most sessions)" but do not discuss that some sessions should include teleport periods in the baseline window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only the `iscell` filter is applied (suite2p manual curation). The AI explicitly decides NOT to apply the putative interneuron filter (dF/F correlated with speed at r > 0.5). The iscell filter is applied AFTER computing dF/F (i.e., dF/F is computed for all ROIs, then subsetted).

ii.
```python
iscell_full = ophys['ImageSegmentation/PlaneSegmentation/iscell'][:]
...
cell_mask_all.append(iscell_plane > 0)
...
events, dff = compute_dff_and_deconvolve(F_concat, Fneu_concat, ...)  # all ROIs
cell_indices = np.where(cell_mask)[0]
events_cells = events[cell_indices, :]  # filter after
```

iii. The CONVERSION_NOTES state: "We do NOT apply the interneuron correlation filter because: (a) it would require computing the full dF/F timeseries first and correlating with speed, (b) it only excludes ~0.42% of cells, (c) the iscell manual curation already removed obvious interneurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start. The trial boundaries (from `trial_start` to `teleport`) define the slice of neural data for each trial, so alignment is inherent in the slicing.

ii.
```python
s = trial_start_inds[i] - 1
e = teleport_inds[i] - 1
trial_neural = events_cells[:, s:e].copy()
```

iii. The instructions specify alignment to start of trial. No additional temporal shifting is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data is kept at the original imaging frame rate (~15.5 Hz, ~64.5 ms per frame). No rebinning is applied. The time bin size is computed as `1000 / 15.5078125` ms.

ii.
```python
'time_bin_size': 1000.0 / 15.5078125,  # ms per frame
```

iii. The CONVERSION_NOTES confirm "Time bin = imaging frame: Each time bin is one imaging frame (~64.5 ms at 15.5 Hz)."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the frame index within the trial and the effective imaging rate. NOT from the stored behavior timestamps.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. The AI computes time from the frame count and frame rate rather than using stored timestamps. Since timestamps are evenly spaced at the imaging rate, this should give very similar results.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A simple sequence of frame indices divided by the effective frame rate. The first timepoint is 0 seconds.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. Straightforward computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Both share the same frame indices (the trial slice from `s` to `e`), so they are inherently aligned. The number of timepoints in `time_from_start` equals `n_tp = e - s`, the same as the neural data slice.

ii.
```python
n_tp = e - s
trial_neural = events_cells[:, s:e].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) / effective_rate
```

iii. Neural and behavioral data use the same time indexing.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series in the NWB file.

ii.
```python
environment = bts['environment/data'][:]
...
env_per_trial = np.zeros(n_trials, dtype=np.int64)
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1
    env_vals = environment[s:e]
    valid_env = env_vals[env_vals >= 0]
    if len(valid_env) > 0:
        env_per_trial[i] = int(np.median(valid_env))
```

iii. Environment is constant within a trial (0 or 1), so taking the median of valid values gives the correct value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The median of valid (>= 0) environment values within each trial is taken. Invalid values (-1, which appear before scanning starts) are excluded. If no valid values exist, the previous trial's value is used.

ii.
```python
valid_env = env_vals[env_vals >= 0]
if len(valid_env) > 0:
    env_per_trial[i] = int(np.median(valid_env))
elif i > 0:
    env_per_trial[i] = env_per_trial[i - 1]
```

iii. Since environment is constant within a trial, the median is equivalent to just taking any valid sample.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index, derived from the loop counter over trials.

ii.
```python
for i in range(n_trials):
    ...
    trial_number = np.float32(i)
```

iii. The trial number is a sequential 0-indexed counter within the session.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No processing beyond assigning the loop index `i`. The value is broadcast to all timepoints within the trial.

ii.
```python
trial_number = np.float32(i)
...
np.full((1, n_tp), trial_number, dtype=np.float32)
```

iii. Simple sequential indexing.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and `autoreward` behavior time series. The AI pre-computes a per-trial `rewarded` array, and the previous trial's value becomes the input.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
rewarded = determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds)
# Also check autoreward
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1
    if np.any(autoreward[s:e] > 0):
        rewarded[i] = 1
```

iii. The AI includes autoreward in the reward determination. The reference does not include autoreward.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome (from the pre-computed `rewarded` array, which includes autoreward) is used. For the first trial, set to 0.

ii.
```python
if i == 0:
    prev_outcome = np.float32(0)
else:
    prev_outcome = np.float32(rewarded[i - 1])
```

iii. Binary previous outcome (0=omitted, 1=rewarded).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location for the current trial. The reward zone is determined by examining where the `reward_zone` signal is nonzero and classifying by nearest zone center.

ii.
```python
def determine_reward_zone_per_trial(position, rz_signal, trial_starts, teleports):
    ...
    rz_active = trial_rz > 0
    if np.any(rz_active):
        rz_positions = trial_pos[rz_active]
        rz_center = np.mean(rz_positions)
    ...
    dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
    zone = min(dists, key=dists.get)
```

iii. The AI uses a simple nearest-center approach. The reference uses a Viterbi algorithm for more stable assignments. The AI handles missing reward zone data by using the previous trial's zone.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance from animal's position to the nearest edge of the assigned reward zone. Negative = before zone, 0 = in zone, positive = past zone. Then discretized into 7 bins.

ii.
```python
def compute_distance_to_reward_zone(position, zone_start, zone_end):
    dist = np.zeros_like(position)
    before = position < zone_start
    inside = (position >= zone_start) & (position <= zone_end)
    after = position > zone_end
    dist[before] = position[before] - zone_start
    dist[inside] = 0.0
    dist[after] = position[after] - zone_end
    return dist
```

iii. Standard signed distance computation. Matches the reference logic.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Discretized into 7 bins using explicit boolean conditions:
- 0: < -50 cm
- 1: -50 to -10 cm
- 2: -10 to 0 cm
- 3: == 0 cm (in zone)
- 4: >0 to 10 cm
- 5: 10 to 50 cm
- 6: > 50 cm

ii.
```python
def discretize_distance_to_rz(distance):
    bins = np.zeros(len(distance), dtype=np.int64)
    bins[distance < -50] = 0
    bins[(distance >= -50) & (distance < -10)] = 1
    bins[(distance >= -10) & (distance < 0)] = 2
    bins[distance == 0] = 3
    bins[(distance > 0) & (distance <= 10)] = 4
    bins[(distance > 10) & (distance <= 50)] = 5
    bins[distance > 50] = 6
    return bins
```

iii. The bins match the instructions. The reference uses `np.digitize` with a `1e-6` separator between "0 cm" and ">0 cm" bins, while the AI uses explicit `== 0` and `> 0` conditions. Both correctly assign "in zone" to bin 3.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position data and neural data share the same frame indices within each trial, so no additional alignment is needed.

ii.
```python
trial_pos = position[s:e]
trial_neural = events_cells[:, s:e].copy()
```

iii. Both use the same `s:e` slice.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
position = bts['position/data'][:]
...
trial_pos = position[s:e]
```

iii. Direct read from NWB position data.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
trial_pos = position[s:e]
pos_binned = discretize_position(trial_pos)
```

iii. Raw position values used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 equal-sized bins of 90 cm each spanning the 450 cm track.

ii.
```python
def discretize_position(position):
    bins = np.zeros(len(position), dtype=np.int64)
    bins[position < 90] = 0
    bins[(position >= 90) & (position < 180)] = 1
    bins[(position >= 180) & (position < 270)] = 2
    bins[(position >= 270) & (position < 360)] = 3
    bins[position >= 360] = 4
    return bins
```

iii. Matches the instructions (5 bins, 90 cm each).

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as neural data within each trial.

ii. `trial_pos = position[s:e]`

iii. Same slice as neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = bts['lick/data'][:]
```

iii. Direct read from NWB lick data.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two-step processing: (1) Lick sensor error correction -- if more than 35% of frames in a trial have cumulative lick > 2, the entire trial's lick data is zeroed out. (2) Binarization -- any positive lick value mapped to 1.

ii.
```python
lick_corrected = lick.copy()
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1
    trial_lick = lick_corrected[s:e]
    n_frames = len(trial_lick)
    if n_frames > 0 and np.sum(trial_lick > 2) / n_frames > LICK_ERROR_THR:
        lick_corrected[s:e] = 0
lick_binary = (lick_corrected > 0).astype(np.int64)
```

iii. The lick sensor error correction comes from the reference code (`correct_lick_sensor_error` in behavior.py). The threshold of 35% matches the paper. The reference solution does NOT apply this correction.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as neural data.

ii. `trial_lick = lick_binary[s:e]`

iii. Same slice.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the `reward_zone` behavior time series and `position` behavior time series. When `reward_zone > 0`, the mean position is computed and the nearest zone center (A, B, or C) is assigned.

ii.
```python
def determine_reward_zone_per_trial(position, rz_signal, trial_starts, teleports):
    ...
    rz_active = trial_rz > 0
    if np.any(rz_active):
        rz_positions = trial_pos[rz_active]
        rz_center = np.mean(rz_positions)
    ...
    dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
    zone = min(dists, key=dists.get)
```

iii. Uses nearest-center classification. The reference uses a Viterbi algorithm for more stable across-trial assignments.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Per trial: find frames where `reward_zone > 0`, compute mean position, classify to nearest zone center (A=105, B=225, C=345). If no reward zone signal, use the previous trial's zone (or default to A for the first trial). Encode as 0=A, 1=B, 2=C.

ii.
```python
if np.any(rz_active):
    rz_positions = trial_pos[rz_active]
    rz_center = np.mean(rz_positions)
else:
    if len(zone_labels) > 0:
        zone_labels.append(zone_labels[-1])
        ...
        continue
    else:
        zone_labels.append('A')
        ...
        continue
dists = {k: abs(rz_center - c) for k, c in ZONE_CENTERS.items()}
zone = min(dists, key=dists.get)
```

iii. Simple nearest-center approach. Less robust than the reference's Viterbi but functionally similar for clean data.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and `autoreward` behavior time series.

ii.
```python
reward_timestamps = bts['Reward/timestamps'][:]
...
rewarded = determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_start_inds, teleport_inds)
for i in range(n_trials):
    s = trial_start_inds[i] - 1
    e = teleport_inds[i] - 1
    if np.any(autoreward[s:e] > 0):
        rewarded[i] = 1
```

iii. Uses both Reward timestamps and autoreward signal.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each trial, check if any Reward timestamp falls within the trial boundaries (by comparing timestamps). Additionally, if any autoreward > 0 within the trial, mark as rewarded. Binary: 0=no, 1=yes.

ii.
```python
def determine_reward_per_trial(reward_timestamps, behav_timestamps, trial_starts, teleports):
    for i in range(n_trials):
        s = trial_starts[i] - 1
        e = teleports[i] - 1
        t_start = behav_timestamps[s]
        t_end = behav_timestamps[min(e, len(behav_timestamps) - 1)]
        in_trial = (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
        if np.any(in_trial):
            rewarded[i] = 1
    return rewarded
```

iii. The reference does not include autoreward in the reward outcome determination.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Empty trials**: Trials where `e <= s` are skipped.
- **Missing reward zone data**: Trials where `reward_zone` is never active use the previous trial's zone (or default to A).
- **NaN in neural data**: After deconvolution, NaN values are replaced with 0.
- **Lick sensor errors**: Trials with >35% frames having lick > 2 have lick data zeroed.
- **Trial count mismatch**: If trial_start and teleport counts differ, truncate to the shorter.
- **Sessions with < 2 trials**: Skipped entirely.

ii.
```python
trial_neural[np.isnan(trial_neural)] = 0
...
if e <= s:
    continue
...
if len(neural_trials) < 2:
    continue
```

iii. The NaN-to-zero replacement for neural data is a practical choice for the decoder. The reference asserts that no NaN exists within trials.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Loading NWB files via h5py and reading large arrays
2. Computing dF/F and OASIS deconvolution for all ROIs (including non-cells)
3. The full conversion loop over 152 sessions (~16 minutes total)

ii. N/A

iii. The AI reports the full conversion took 984 seconds (~16 min). Per-session timing is printed.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over trials to build input/output arrays. The reward zone determination loop, lick error correction loop, and environment per-trial loop each iterate over all trials. These could potentially be vectorized with masked array operations.

ii. N/A

iii. The variable trial lengths make vectorization awkward but not impossible for fixed-length operations like reward zone determination.

## 13-c. What processing does the code repeat multiple times?

i. The dF/F computation is performed for ALL ROIs (including non-cells), then the iscell filter is applied. This means ~2/3 of the dF/F computation is wasted on non-cell ROIs. The reference applies iscell before dF/F to avoid this.

ii.
```python
events, dff = compute_dff_and_deconvolve(F_concat, Fneu_concat, ...)  # all ROIs
cell_indices = np.where(cell_mask)[0]
events_cells = events[cell_indices, :]  # filter after
```

iii. This is wasteful but does not affect correctness.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes dF/F for all ROIs (including non-cells that are filtered out), and also computes and returns the `dff` array from `compute_dff_and_deconvolve`, which is only used for plotting but not for the final converted data. The lick sensor error correction is additional processing not done in the reference.
