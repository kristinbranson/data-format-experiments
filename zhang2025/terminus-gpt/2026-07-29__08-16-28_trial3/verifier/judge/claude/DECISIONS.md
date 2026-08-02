# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by globbing the local ONE cache directory structure (`data/one_cache/*/Subjects/*/*/*`) and checking for the existence of `alf/` subdirectories. For each valid session directory, it loads the trials table from `_ibl_trials.table.pqt`, spikes from `spikes.times.npy` and `spikes.clusters.npy` under `probe*/pykilosort` directories, wheel data from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, and motion energy from `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy`. Sessions missing any of these components are skipped.

ii.
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)

def choose_sessions(session_dirs, mode):
    valid = []
    for s in session_dirs:
        alf = s / 'alf'
        if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
            valid.append(s)
    if mode == 'sample':
        return valid[:2]
    return valid
```

iii. The AI chose to load data directly from local ALF files rather than using the ONE API, noting this was a performance optimization that reduced sample conversion time from ~27s to ~3s. The reference code uses the ONE API with `SessionLoader` and `SpikeSortingLoader`, but the underlying data is the same. The AI does not use a session list (like `bwm_release.csv` in the reference) to select which sessions to process; instead it processes all locally available sessions that have the required files.

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the filesystem path. The session directory structure is `lab/Subjects/subject_name/date/number`, and the subject name is taken as the 3rd-from-last path component.

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```

iii. This approach is consistent with the IBL ONE cache directory convention. The reference code obtains subject info from the session metadata tables, but the result is equivalent.

## 1-c. How are the data split into sessions?

i. Each directory under `data/one_cache/*/Subjects/subject/date/number` with an `alf/` subdirectory is treated as a separate session. Sessions are filtered to require a trials table, and during processing, sessions are further skipped if they lack required columns (stimOn_times, choice, probabilityLeft), have too few valid trials (<2), lack spike data, lack wheel data, or lack whisker motion energy data.

ii.
```python
session_dirs = choose_sessions(list_session_dirs(data_root), mode)
# ... in main loop:
if 'stimOn_times' not in trials_df.columns or 'choice' not in trials_df.columns or 'probabilityLeft' not in trials_df.columns:
    print('skip missing required trial columns', session_dir)
    continue
```

iii. The AI processes 461 candidate sessions and retains 159. The reference code uses a curated session list (`bwm_release.csv`) and processes sessions by EID. Many sessions are skipped due to missing whisker motion energy data.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` as a pandas DataFrame. Each row is one trial. The `stimOn_times` column is used to define each trial's temporal window.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
```

iii. This is consistent with the reference code, which also loads the trials table (via `SessionLoader.load_trials()`).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: trials are excluded only if `stimOn_times` is NaN, `choice` maps to an invalid value (i.e., choice==0 meaning no response), or `probabilityLeft` maps to an invalid value.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The reference code (`load_trials_and_mask`) applies much more extensive filtering: reaction time must be between 0.08s and 2.0s (`firstMovement_times - stimOn_times`), NaN exclusion on 6 columns (`stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, `feedbackType`), max trial length of 10s, and exclusion of no-choice trials (choice==0). The AI's filtering is substantially less strict.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps in seconds) and `spikes.clusters.npy` (cluster assignment for each spike), loaded from `probe*/pykilosort` directories under each session's ALF folder.

ii.
```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
st = np.load(st_p)
sc = np.load(sc_p).astype(int)
```

iii. This matches the reference code, which loads the same variables via `SpikeSortingLoader.load_spike_sorting()`.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping time bins from -0.5s to +1.5s relative to stimulus onset (100 bins total). Multiple probes within a session are merged by concatenating spikes and re-indexing clusters. Spike counts are accumulated using `np.add.at` into a (n_neurons, n_bins) matrix per trial.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
N_BINS = len(TIME_BINS)

def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:
        edges = t0 + TIME_BINS
        out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
        lo = np.searchsorted(spike_times, edges[0], side='left')
        hi = np.searchsorted(spike_times, t0 + T_END, side='left')
        ts = spike_times[lo:hi] - t0
        cl = spike_clusters[lo:hi]
        bins = np.floor((ts - T_START) / BINSIZE).astype(int)
        m = (bins >= 0) & (bins < N_BINS) & (cl >= 0) & (cl < n_neurons)
        np.add.at(out, (cl[m], bins[m]), 1)
        mats.append(out)
    return mats
```

iii. The binning approach (20ms bins, -0.5 to +1.5s, stimulus onset alignment) matches the reference code parameters (`binsize=0.02`, `time_window=(-0.5, 1.5)`, `align_time='stimOn_times'`). The implementation differs (reference uses `bincount2D` from brainbox) but is functionally equivalent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons to only include "good" clusters, defined as those with `label >= 1` in the `clusters.metrics.pqt` file. Clusters with label < 1 (noise/multiunit) are excluded.

ii.
```python
metrics = pd.read_parquet(metrics_p)
nclu = len(metrics)
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
# ...
remap[good] = np.arange(good.sum()) + offset
```

iii. The reference code's `prepare_data` function calls `load_spiking_data` WITHOUT quality filtering (`qc=None`), which returns ALL clusters regardless of label. Quality labels are stored in metadata (`good_clusters`) but are not used to filter at the caching stage. The AI's decision to filter to `label >= 1` is inconsistent with the reference code, which uses all clusters.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. The time window spans from `stimOn_times - 0.5s` to `stimOn_times + 1.5s`.

ii.
```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
# In bin_spikes:
for t0 in stim_on:
    edges = t0 + TIME_BINS  # TIME_BINS starts at -0.5
```

iii. Matches the reference code: `align_time='stimOn_times'`, `time_window=(-0.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s), producing 100 bins per trial. No additional rebinning is applied.

ii.
```python
BINSIZE = 0.02
TIME_BINS = np.arange(T_START, T_END, BINSIZE)  # 100 bins
```

iii. Matches the reference code parameter `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the time bin grid itself, not from any raw data variable. The bin centers are computed from the fixed constants `T_START`, `T_END`, and `BINSIZE`.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2  # [-0.49, -0.47, ..., 1.49]
# In main loop:
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    ...
])
```

iii. The time vector is the same for every trial and session, which is appropriate since all trials have the same temporal grid relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Bin centers are computed as `TIME_BINS + BINSIZE/2`, giving values from -0.49 to 1.49 in 0.02s steps.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. The reference code computes interpolation grid as `np.linspace(interval_beg + binsize, interval_end, n_bins)` which gives right edges of bins: [-0.48, -0.46, ..., 1.50]. The AI uses bin centers (left edge + half bin width), resulting in a 10ms offset from the reference grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `TIME_CENTERS` vector is used for both the time input and the interpolation points for behavioral signals. The neural data is binned with edges at `TIME_BINS`, so the time input represents the centers of the neural bins.

ii.
```python
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. Both the time input and neural data use the same 100-timepoint grid, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. The reference code does not compute "trial number in block" explicitly; it uses `probabilityLeft` as a block identity variable. The instructions specify this as a decoder input, so the AI had to implement it from scratch.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through trials sequentially. When `probabilityLeft` changes value, a counter resets to 1. Otherwise, the counter increments by 1 for each trial. The result is a per-trial integer indicating position within the current block.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    cur = prob_left[0]
    c = 0
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
    return out
```

iii. This is a reasonable implementation matching the instruction's description of "Trial number in block" as a continuous, per-trial input. The value is broadcast across all time bins for each trial.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in `_ibl_trials.table.pqt`.

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. Consistent with the reference code, which also uses `trials_df['choice']`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL convention uses 1 for left, -1 for right, and 0 for no response. The AI maps: 1 (left) -> 0, -1 (right) -> 1, 0 (no response) -> -1 (excluded by `choice >= 0` validity check).

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. The mapping matches the instructions: "left = 0, right = 1". The reference code uses raw IBL choice values; this remapping is specific to the decoder task format.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in `_ibl_trials.table.pqt`.

ii.
```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. Consistent with the reference code, which uses `trials_df['probabilityLeft']`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps the three possible probability values to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Values not matching any of these (checked with `np.isclose`) are set to -1 and excluded.

ii.
```python
def map_prior(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[np.isclose(vals, 0.2)] = 0
    out[np.isclose(vals, 0.5)] = 1
    out[np.isclose(vals, 0.8)] = 2
    return out
```

iii. The mapping matches the instructions: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2". Use of `np.isclose` is appropriate for floating point comparison.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.timestamps.npy` (wheel timestamps) and `_ibl_wheel.position.npy` (wheel position).

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)
```

iii. The reference code also loads wheel position and timestamps, but processes them through `SessionLoader.load_wheel()` which applies additional processing.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes velocity using `np.gradient(position, timestamps)` on the raw wheel position data, after removing NaN values and deduplicating timestamps. This gives an unfiltered numerical derivative. No absolute value is taken (so the result is velocity, not speed).

ii.
```python
def interp_wheel_speed(timestamps, position, stim_on):
    timestamps = np.asarray(timestamps, dtype=float)
    position = np.asarray(position, dtype=float)
    # ... NaN removal, deduplication ...
    vel = np.gradient(position, timestamps)
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
        trials.append(y.astype(np.float32))
    return trials
```

iii. The reference code uses `SessionLoader.load_wheel()` which: (1) interpolates position to 1000 Hz uniform sampling, (2) applies an 8th-order Butterworth low-pass filter at 20 Hz, (3) computes velocity as `diff(filtered_position) * fs`, and (4) takes absolute value for speed. The AI's approach differs in three critical ways: no interpolation to uniform sampling, no low-pass filtering, and no absolute value (computing velocity rather than speed).

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI discretizes into 3 bins using tertile-based thresholds. All finite values across all trials in a session are pooled, and the 1/3 and 2/3 quantiles define the bin boundaries. Values below the 1/3 quantile are bin 0, between 1/3 and 2/3 are bin 1, above 2/3 are bin 2. NaN values are assigned bin 0.

ii.
```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    out = []
    for x in list_of_arrays:
        y = np.zeros_like(x, dtype=np.int64)
        y[x > q1] = 1
        y[x > q2] = 2
        y[~np.isfinite(x)] = 0
        out.append(y)
    return out, (float(q1), float(q2))
```

iii. The instructions specify "Wheel speed discretized into 3 bins, time-varying". The reference code does not discretize wheel speed (it decodes the continuous value), so this is a task-specific requirement. The tertile approach ensures roughly equal bin populations. However, NaN values being assigned to bin 0 (low) is a concerning default that could bias the low bin.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel velocity is interpolated to the same time grid as neural data (`TIME_CENTERS = TIME_BINS + BINSIZE/2`), ensuring temporal alignment between neural and behavioral data.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The temporal alignment is consistent with the neural data grid. The reference code uses `get_behavior_per_interval` with the same `time_window` and `binsize` parameters, though with different interpolation grid (`np.linspace(interval_beg + binsize, interval_end, n_bins)` = right bin edges vs AI's bin centers).

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` files, along with `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy` for timestamps.

ii.
```python
def load_motion_energy(session_alf: Path):
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
    streams = []
    for tp, mp in [(left_t, left_me), (right_t, right_me)]:
        if tp is not None and mp is not None and tp.exists() and mp.exists():
            streams.append((np.load(tp), np.load(mp).astype(np.float32)))
    if not streams:
        return None
    return streams
```

iii. The reference code loads `whiskerMotionEnergy` specifically from `SessionLoader.load_motion_energy()`, which extracts the whisker-pad ROI motion energy column. The AI loads `ROIMotionEnergy.npy` which may contain motion energy for the entire ROI rather than just the whisker pad region.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads motion energy from both left and right cameras (when available), interpolates each to the trial time grid, then averages the valid values across cameras.

ii.
```python
def interp_motion_energy(streams, stim_on):
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        ys = []
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
        arr = np.stack(ys, axis=0)
        valid = np.isfinite(arr)
        denom = valid.sum(axis=0)
        summed = np.where(valid, arr, 0.0).sum(axis=0)
        y = np.divide(summed, denom, ...)
        trials.append(y.astype(np.float32))
    return trials
```

iii. The reference code loads only one camera (left first, falling back to right), does NOT average across cameras. The AI averages left and right camera motion energy, which is a different approach.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same tertile-based discretization as wheel speed. All finite values across all trials in a session are pooled, quantiles at 1/3 and 2/3 define boundaries, producing 3 bins. NaN values are assigned bin 0.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. Same approach as wheel speed discretization. The instructions say "Whisker motion energy discretized into 3 bins". Tertile-based binning is reasonable.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is interpolated to `TIME_CENTERS` (same grid as neural data), providing temporal alignment.

ii.
```python
x = t0 + TIME_CENTERS
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. Same alignment approach as wheel speed. The 10ms offset from the reference grid (bin centers vs right edges) applies here too.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- Sessions missing required columns are skipped entirely
- Sessions with fewer than 2 valid trials are skipped
- Sessions missing spike data, wheel data, or whisker motion energy are skipped
- Duplicate wheel timestamps are deduplicated before computing gradients
- NaN values in wheel/whisker interpolation are handled with `left=np.nan, right=np.nan`
- NaN values in discretization are assigned to bin 0
- Any unhandled exception in session processing is caught and the session is skipped

ii.
```python
# Duplicate wheel timestamp handling:
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
timestamps = uniq_t
position = position[uniq_idx]

# NaN handling in discretization:
y[~np.isfinite(x)] = 0

# Generic exception catching:
except Exception as e:
    print('skip session due to error', session_dir, repr(e))
```

iii. The AI documented fixing duplicate timestamp warnings and NaN handling issues during Step 10 (Critical Review). However, assigning NaN values to bin 0 in discretization could introduce systematic biases. The broad exception catching could mask real errors.

## 12-a. What are the most time-consuming steps of the code?

i. Based on the conversion output, each session takes ~0.7-1.2 seconds. The most time-consuming steps are likely spike binning (iterating over all trials with searchsorted and np.add.at) and wheel/whisker interpolation per trial.

ii.
```python
# Spike binning loops over all trials:
def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:
        # ... searchsorted, np.add.at per trial
```

iii. The AI noted replacing ONE object loading with direct file reads reduced time from ~27s to ~3s for sample conversion. The final full conversion of 159 sessions completed in reasonable time.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function loops over each trial individually with Python for-loops. This could potentially be vectorized by computing bin assignments for all trials simultaneously and using advanced indexing.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:  # Loop over each trial
        # ... per-trial processing
        np.add.at(out, (cl[m], bins[m]), 1)
        mats.append(out)
```

iii. Similarly, `interp_wheel_speed` and `interp_motion_energy` loop over trials. The reference code uses multiprocessing for spike binning and behavior interpolation, which the AI does not.

## 12-c. What processing does the code repeat multiple times?

i. For each session, the AI loads the trials table, then processes each trial individually for spikes, wheel, and whisker data. The trial validity check (`valid` mask) is computed once and reused. The `discretize_tertiles` function is called separately for wheel and whisker data, but each call internally concatenates and computes quantiles from scratch.

ii.
```python
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. The code does not appear to have significant redundant processing. Each major step is performed once per session.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes the full velocity time series for the wheel (including negative values) but then discretizes into bins, losing the sign information. Since the reference code uses wheel speed (absolute velocity), computing signed velocity is unnecessary work.

ii.
```python
vel = np.gradient(position, timestamps)  # signed velocity
# Later discretized into bins, losing sign info
```

iii. Additionally, the AI processes and stores brain region information as numeric IDs (strings) rather than anatomical acronyms, which provides less interpretable region labels but doesn't affect downstream decoding.
