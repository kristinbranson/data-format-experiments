# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI lists all NWB files under `/app/data` by scanning `sub-*` directories, sorting by subject number then session number. Each NWB file is opened with `h5py.File` (not `pynwb`). All behavior streams and fluorescence/neuropil data are read from the HDF5 groups directly. A `list_sessions()` function enumerates all `.nwb` files deterministically.

ii.
```python
def list_sessions():
    files = []
    for sub in sorted(os.listdir(DATA_ROOT)):
        d = os.path.join(DATA_ROOT, sub)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.endswith('.nwb'):
                files.append(os.path.join(d, fn))
    def key(p):
        base = os.path.basename(p)
        sub = base.split('_')[0].replace('sub-m', '')
        ses = base.split('_')[1].replace('ses-', '')
        return (int(sub), int(ses))
    return sorted(files, key=key)
```
Loading data from each file:
```python
with h5py.File(path, 'r') as f:
    subject = f['general/subject/subject_id'][()].decode()
    session_id = f['general/session_id'][()].decode()
    ...
    beh = load_behavior(f)
    F, Fneu, plane_of_cell = load_fluorescence(f)
```

iii. CONVERSION_NOTES Step 2 documents 152 NWB files across 11 subjects. The code scans all subdirectories and files, matching the paper's 11 switch-task mice.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `general/subject/subject_id` field in each NWB file (e.g., `m3`, `m11`). All unique subjects are collected and sorted by numeric ID.

ii.
```python
subjects = sorted({r['subject'] for r in results}, key=lambda s: int(s[1:]))
sub_index = {s: i for i, s in enumerate(subjects)}
```

iii. The 11 subject directories match the 11 switch-task mice described in the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session number is the experiment day, extracted from `general/session_id`. Sessions are processed independently.

ii.
```python
exp_day = int(session_id)
```

iii. NWB filenames follow the pattern `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, with one file per session.

## 1-d. How are the data split into trials?

i. Trial starts are identified from the `trial_start` behavior variable (nonzero frames). Trial ends are identified from the `teleport` behavior variable (nonzero frames). The trial spans `[trial_start, teleport)`. Robustness handling drops unpaired leading teleports or trailing trial starts.

ii.
```python
trial_starts = np.where(beh['trial_start'] > 0)[0]
teleports = np.where(beh['teleport'] > 0)[0]
# robustness: drop an unpaired leading teleport / trailing trial start
if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
    teleports = teleports[1:]
if len(trial_starts) > len(teleports):
    trial_starts = trial_starts[:len(teleports)]
```

iii. CONVERSION_NOTES Step 4 discusses the trial boundary definition. The teleport index itself is excluded because the position value there is an interpolation artifact.

## 1-e. How are trials filtered based on quality controls?

i. Trials with lick-sensor errors are removed. A trial is flagged if >30% of its frames have a cumulative lick count >2. This matches the paper's Methods exactly (81 trials removed across 11 mice).

ii.
```python
LICK_ERROR_FRAC_THRESH = 0.3
LICK_ERROR_COUNT_THRESH = 2
...
seg = beh['lick'][st:sp_]
lick_error[i] = (np.sum(seg > LICK_ERROR_COUNT_THRESH) / len(seg)) > LICK_ERROR_FRAC_THRESH
...
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    if lick_error[i]:
        continue
```

iii. CONVERSION_NOTES Step 4 notes that using 0.30 threshold reproduces the paper's exact count of 81 removed trials, which also validates the trial boundary definition.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the raw suite2p fluorescence (`Fluorescence/plane*/data`) and neuropil (`Neuropil/plane*/data`) traces stored in the NWB. The NWB `Deconvolved` field is explicitly NOT used.

ii.
```python
Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)   # F from Fluorescence
Fneus.append(np.asarray(Np[:, :], dtype=np.float32)[:, sel].T) # Fneu from Neuropil
```

iii. CONVERSION_NOTES Step 1 and Step 5 explain that the paper's neural signal comes from `pp.dff(F, Fneu, ..., deconvolve=True)`, not from suite2p's own deconvolution stored in the NWB.

## 2-b. How is the `neural` data processed?

i. The AI implements the paper's full dF/F pipeline: (1) neuropil subtraction with coefficient 0.7, (2) per-trial neuropil mean added back, (3) maximin baseline (Gaussian smooth sigma=15 frames, then 300-frame minimum filter, then 300-frame maximum filter), (4) dF/F = (F - baseline) / |baseline|, (5) Gaussian smoothing with sigma=2 frames, (6) OASIS deconvolution with tau=0.7. The `keep_teleports` flag per animal/day determines whether baseline windows span the inter-trial interval.

ii.
```python
def dff_and_events(F, Fneu, trial_starts, teleports, keep_teleports, fs, ...):
    ...
    f_ -= neu_coef * f_neu_
    for start, stop in zip(start_inds, stop_inds):
        seg += neu_coef * np.nanmean(f_neu_[:, start:stop], axis=1, keepdims=True)
        base = nansmooth(seg, BASELINE_SMOOTH_SIGMA, axis=-1)
        base = sp.ndimage.minimum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        base = sp.ndimage.maximum_filter1d(base, BASELINE_FILTER_WIN, axis=-1)
        flow[:, start:stop] = base
    dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
    for start, stop in zip(start_inds, stop_inds):
        smoothed = nansmooth(dff[:, start:stop], DFF_SMOOTH_SIGMA, axis=-1)
        events[:, start:stop] = dcnv.oasis(np.ascontiguousarray(smoothed), 2000, tau, fs)
```

iii. CONVERSION_NOTES Steps 1, 3, and 5 describe this as faithfully porting `reward_relative.preprocessing.dff`. The `keep_teleports` table is copied from `teleport_metadata.py` but with mouse IDs translated from GCAMP to m-numbers.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Only `iscell==1` ROIs are retained (suite2p manual curation). (2) Putative interneurons are removed: cells with Pearson r(dF/F, speed) > 0.5. Additionally, cells with non-finite values in events are dropped.

ii.
```python
sel = iscell[offset:offset + n_roi]
Fs.append(np.asarray(Fp[:, :], dtype=np.float32)[:, sel].T)
...
speed_corr = (dv @ sv) / denom
is_interneuron = speed_corr > INTERNEURON_R_THRESH
finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
keep_cells = (~is_interneuron) & finite_cells
events = events[keep_cells]
```

iii. CONVERSION_NOTES Steps 1 and 5 document both filters as matching the paper's Methods. The interneuron correlation is implemented as a vectorized matrix operation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to trial start. Neural data is sliced as `events[:, st:sp_]` where `st` is the `trial_start` index and `sp_` is the `teleport` index. No temporal shifting is needed since the alignment event IS the trial start.

ii.
```python
act = events[:, st:sp_]
```

iii. The instructions specify "Temporally align based on start of the trial." Since data is sliced starting at the trial_start frame, alignment is inherent.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging frame rate is used (~15.5 Hz, ~64.5 ms per frame). No temporal rebinning is applied. The time bin size is computed as the median of the inter-frame intervals across sessions.

ii.
```python
dts = np.array([r['dt'] for r in results])
data['metadata'] = {
    'time_bin_size': float(np.median(dts) * 1000.0),
    ...
}
```

iii. The paper states "All behavioral and neural time series were sampled at ~15.5 Hz" and no rebinning is described.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Derived from the `position/timestamps` behavior time series (stored in `beh['time']`).

ii.
```python
beh = {
    'time': b['position/timestamps'][:],
    ...
}
...
tt = beh['time'][st:sp_] - beh['time'][st]
inp[0] = tt
```

iii. All behavior streams share the same timestamps (the imaging frame grid).

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial start timestamp is subtracted from each timepoint's timestamp within the trial.

ii.
```python
tt = beh['time'][st:sp_] - beh['time'][st]
inp[0] = tt
```

iii. Straightforward time-from-start computation.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Neural and behavioral data share the same time indexing (both are on the imaging frame grid). The same `[st:sp_)` slice is used for both, so alignment is inherent.

ii.
```python
act = events[:, st:sp_]
...
tt = beh['time'][st:sp_] - beh['time'][st]
```

iii. CONVERSION_NOTES Step 2 notes all behavioral streams are on the imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Derived from the `environment` behavior time series.

ii.
```python
'environment': b['environment/data'][:],
...
env_vals = np.unique(beh['environment'][st:sp_])
env_vals = env_vals[env_vals >= 0]
environment[i] = int(env_vals[0])
```

iii. The environment variable is 0 (ENV1) or 1 (ENV2), matching the paper's two-environment design.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The unique non-negative environment value within the trial is taken (asserted to be exactly one value). The value is constant across all timepoints in the trial.

ii.
```python
env_vals = np.unique(beh['environment'][st:sp_])
env_vals = env_vals[env_vals >= 0]
assert len(env_vals) == 1
environment[i] = int(env_vals[0])
...
inp[1] = environment[i]
```

iii. Pre-trial frames have environment=-1, which are excluded by the non-negative filter.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the within-session trial index, derived from the loop counter over trials (0-indexed).

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    ...
    inp[2] = i
```

iii. The trial number is the sequential index within the session, not the stored `trial number` variable.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index `i` is used directly. However, because lick-error trials are skipped, the trial number still reflects the original trial index (before filtering), not the filtered trial index.

ii.
```python
for i, (st, sp_) in enumerate(zip(trial_starts, teleports)):
    if lick_error[i]:
        continue
    ...
    inp[2] = i
```

iii. This means trial numbers may have gaps where lick-error trials were removed.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior time series. Reward event timestamps are mapped to frame indices via `np.searchsorted`, and a trial is considered rewarded if any reward event falls within it AND the reward zone was active.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
reward_frames = np.clip(reward_frames, 0, n_frames - 1)
...
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. This matches the reference code's `behav.get_trial_types` which checks both reward delivery and reward zone activity.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial, the previous trial's reward outcome is used. For the first trial of each session, the value is set to 1 (rewarded) rather than 0. The value is constant across timepoints.

ii.
```python
prev_outcome = np.empty(n_trials, dtype=np.int64)
prev_outcome[1:] = is_reward[:-1]
prev_outcome[0] = 1  # see CONVERSION_NOTES Step 5, decision 8
...
inp[3] = prev_outcome[i]
```

iii. CONVERSION_NOTES Step 5 justifies setting the first trial to 1: rewarded is the modal outcome (84.7%), and mice ran warm-up trials before imaging.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. Derived from the `position` behavior time series and the reward zone location per trial. The reward zone location comes from the NWB `identifier` field (which encodes the VR scene name), parsed by `zones_from_scene()` using the paper's logic from `behavior.get_reward_zones`.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
zone_starts = np.array([REWARD_ZONE_CM[z][0] for z in zone_labels])
zone_stops = np.array([REWARD_ZONE_CM[z][1] for z in zone_labels])
```

iii. CONVERSION_NOTES Steps 1 and 4 document this as a port of `behavior.get_reward_zones`, validated against empirical reward-zone-active positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is computed as: 0 when inside the zone, negative when before (position < zone start), positive when past (position > zone stop). Then discretized into 7 bins.

ii.
```python
d = np.zeros(T)
before = pos < zone_starts[i]
after = pos > zone_stops[i]
d[before] = pos[before] - zone_starts[i]
d[after] = pos[after] - zone_stops[i]
```

iii. This matches the paper's concept of reward-relative distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI uses explicit boolean conditions to discretize into 7 categories: `<-50` -> 0, `[-50,-10)` -> 1, `[-10,0)` -> 2, `==0` -> 3, `(0,10]` -> 4, `(10,50]` -> 5, `>50` -> 6.

ii.
```python
def digitize_distance_to_reward(d):
    out = np.empty(d.shape, dtype=np.int64)
    out[d < -50] = 0
    out[(d >= -50) & (d < -10)] = 1
    out[(d >= -10) & (d < 0)] = 2
    out[d == 0] = 3
    out[(d > 0) & (d <= 10)] = 4
    out[(d > 10) & (d <= 50)] = 5
    out[d > 50] = 6
    return out
```

iii. The bin boundaries match the instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same time indices `[st:sp_)` as the neural data within each trial -- no additional alignment needed.

ii.
```python
pos = beh['position'][st:sp_]
act = events[:, st:sp_]
```

iii. All data shares the imaging frame grid.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. Derived from the `position` behavior time series.

ii.
```python
pos = beh['position'][st:sp_]
```

iii. The `position` variable records the animal's position on the VR track in cm.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. No processing beyond extracting the per-trial slice and discretizing.

ii.
```python
out[1] = digitize_position(pos)
```

iii. Raw position values are used directly.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Discretized into 5 bins of 90 cm each using floor division: `np.clip(pos // 90.0, 0, 4)`.

ii.
```python
def digitize_position(pos):
    return np.clip((pos // 90.0), 0, 4).astype(np.int64)
```

iii. The 450 cm track divided into 5 bins gives 90 cm per bin. Values outside [0, 450) are clipped.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same time indices as the neural data within each trial -- no additional alignment needed.

ii. Same `[st:sp_)` slice as neural data.

iii. All data shares the imaging frame grid.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. Derived from the `lick` behavior time series.

ii.
```python
lick = beh['lick'][st:sp_]
```

iii. The `lick` variable records lick events per frame.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized: any positive lick value is mapped to 1, otherwise 0.

ii.
```python
out[3] = (lick > 0).astype(np.int64)
```

iii. The instructions specify binary output (no/yes). The raw lick values can be >1, so thresholding at >0 converts to binary.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same time indices as the neural data within each trial -- no additional alignment needed.

ii. Same `[st:sp_)` slice as neural data.

iii. All data shares the imaging frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Derived from the VR scene name extracted from the NWB `identifier` field. The `zones_from_scene()` function parses the scene string (e.g., `Env1_LocationB_to_A`) to determine which reward zone(s) are active, using the 30-trial switch rule for switch sessions.

ii.
```python
scene = identifier.split('/')[-1]
zone_labels = zones_from_scene(scene, n_trials)
...
out[4] = REWARD_ZONE_LABELS.index(zone_labels[i])
```

iii. This is a port of `behavior.get_reward_zones`. CONVERSION_NOTES Step 4 reports cross-validation: for all 12,216 trials with an active zone, the empirical first in-zone position matches the scene-derived zone start.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The scene name is parsed to determine the initial zone and (for switch sessions) the post-switch zone. The first 30 trials use the initial zone, remaining trials use the second zone. The zone label (A/B/C) is mapped to an integer (0/1/2).

ii.
```python
def zones_from_scene(scene, n_trials, change_trial=CHANGE_TRIAL):
    for first in REWARD_ZONE_LABELS:
        if f'{first}_to' in scene:
            second = scene[-1]
            labels = np.array([first] * min(change_trial, n_trials)
                              + [second] * max(0, n_trials - change_trial))
            return labels
    for zone in REWARD_ZONE_LABELS:
        if scene.endswith('Location' + zone):
            return np.array([zone] * n_trials)
```

iii. The `change_trial=30` matches the paper's "Each switch occurred after 30 trials."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. Derived from the `Reward` timestamps and `reward_zone` behavior time series.

ii.
```python
reward_frames = np.searchsorted(beh['time'], beh['reward_times'])
...
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
is_reward[i] = int(bool(any_reward and any_rzone))
...
out[5] = is_reward[i]
```

iii. The reward outcome requires both a reward delivery event AND an active reward zone, matching `behav.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary: 1 if any reward event occurred in the trial AND the reward zone was active, 0 otherwise. The value is constant across all timepoints in the trial.

ii.
```python
any_reward = np.any((reward_frames >= st) & (reward_frames < sp_))
any_rzone = np.any(beh['reward_zone'][st:sp_] > 0)
is_reward[i] = int(bool(any_reward and any_rzone))
```

iii. This dual condition matches `behav.get_trial_types` more closely than just checking for reward events.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Unpaired trial boundaries**: Leading teleports before first trial_start, or trailing trial_starts without teleports, are dropped.
- **Lick sensor errors**: Trials with >30% bad lick frames are removed entirely (81 trials).
- **Non-finite cells**: Cells with NaN/Inf in the deconvolved events are dropped.
- **NaN in events within trials**: Guarded with `np.nan_to_num(act, nan=0.0)` (should not occur).
- **Neural/behavior frame mismatch**: Asserted to match via F.shape[1] == n_frames.
- **Environment edge case**: Pre-trial frames with environment=-1 are excluded by filtering.

ii.
```python
# Unpaired boundaries
if len(teleports) and len(trial_starts) and teleports[0] < trial_starts[0]:
    teleports = teleports[1:]
...
# Non-finite cells
finite_cells = np.all(np.isfinite(events[:, valid]), axis=1) & np.isfinite(speed_corr)
keep_cells = (~is_interneuron) & finite_cells
...
# NaN guard
if not np.all(np.isfinite(act)):
    act = np.nan_to_num(act, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The lick-error handling matches the paper's Methods exactly (81 trials).

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **dF/F and OASIS deconvolution** (`dff_and_events`) -- the computational core of the pipeline.
2. **Reading fluorescence data** from NWB files (I/O bound).
3. **Saving the pickle file** (large dataset).

ii.
```python
timing['dff_events'] = time.time() - t3
```

iii. The code records timing per session and reports ETAs during processing.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The interneuron detection is already vectorized as a matrix multiplication (`dv @ sv`). The per-trial loop is inherent due to variable trial lengths. The `nansmooth` function applies `gaussian_filter1d` across all cells simultaneously.

ii.
```python
speed_corr = (dv @ sv) / denom  # vectorized correlation
```

iii. The AI's interneuron detection is more efficient than the reference's per-cell loop with `np.corrcoef`.

## 13-c. What processing does the code repeat multiple times?

i. Each NWB file is read only once. The behavior loading and fluorescence loading happen in a single pass per session. There is no separate "survey" step -- all processing happens in `process_session()`.

ii. N/A

iii. Unlike the reference which has a separate survey step that reads all NWB files before conversion, the AI processes each session in one pass.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores several metadata fields per session that are not used by the decoder (e.g., `scene`, `identifier`, `n_iscell`, `is_reward` array, `zone_labels`, timing info). These add negligible overhead.

ii. N/A

iii. These are metadata for documentation and debugging, not performance-impacting.
