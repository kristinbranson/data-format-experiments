# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from a local IBL ONE cache at `data/one_cache`. It iterates over all session directories found by globbing `*/Subjects/*/*/*` and filters to directories with an `alf` subfolder. For each session, it loads the trials table (`_ibl_trials.table.pqt`), spike data from `probe*/pykilosort` subdirectories, wheel data (`_ibl_wheel.timestamps.npy`, `_ibl_wheel.position.npy`), and whisker motion energy from camera files. Sessions missing any required data (trials table, spikes, wheel, or whisker ME) are skipped.

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

iii. The AI noted in CONVERSION_NOTES.md that the reference code uses `ONE` API calls to load data by session EID, while the AI adapted this to direct local file reads from the ONE cache for performance reasons. The AI documented this as an intentional optimization in Step 6 notes.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from the session directory path. The subject name is taken as the third-to-last component of the session path (e.g., `lab/Subjects/SUBJECT_NAME/date/number`).

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]

# In main():
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
```

iii. The AI documented that the ONE cache structure organizes data by `lab/Subjects/subject/date/number`, making path-based extraction straightforward. This matches the IBL data architecture.

## 1-c. How are the data split into sessions?

i. Each unique directory at the `lab/Subjects/subject/date/number` level is treated as a session. Sessions are processed sequentially, and each valid session becomes one entry in the output lists (`neural`, `input`, `output`).

ii.
```python
for si, session_dir in enumerate(session_dirs):
    # ... process session ...
    out['neural'].append(neural_trials)
    out['input'].append(sess_inputs)
    out['output'].append(sess_outputs)
```

iii. The AI noted this matches the reference code's session-level processing approach, where each EID corresponds to one session.

## 1-d. How are the data split into trials?

i. Trials are defined by the rows of the `_ibl_trials.table.pqt` parquet file for each session. Each trial has an associated `stimOn_times` used for temporal alignment. Neural data, wheel speed, and whisker motion energy are each aligned to stimulus onset for each valid trial, producing per-trial arrays.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
# ...
neural_trials = bin_spikes(spikes['times'], spikes['clusters'], len(clusters['acronym']), stim_on[valid])
wheel_trials = interp_wheel_speed(wt, wp, stim_on[valid])
whisk_trials = interp_motion_energy(me_streams, stim_on[valid])
```

iii. The AI documented that the reference code also uses the trials table and stimulus onset times for trial-by-trial alignment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) `stimOn_times` must be finite (not NaN), (2) `choice` must be valid (mapped to 0 or 1, not -1 meaning unmapped), and (3) `probabilityLeft` must be one of {0.2, 0.5, 0.8}. Sessions with fewer than 2 valid trials are skipped. There is NO `max_trial_len` filter applied.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The AI noted in Step 4 that the reference code uses `load_trials_and_mask(..., max_trial_len=10.0)` to filter trials, but the AI did not implement this max trial length filter. The AI's justification focused on the validity masks for stimOn_times, choice, and prior, but did not explicitly address the omission of `max_trial_len`.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) loaded from `probe*/pykilosort` subdirectories within each session's `alf` folder. Quality metrics are loaded from `clusters.metrics.pqt`.

ii.
```python
def load_spikes_for_session_dir(session_dir):
    for probe_dir in sorted((session_dir / 'alf').glob('probe*/pykilosort')):
        # ...
        st = np.load(st_p)  # spikes.times.npy
        sc = np.load(sc_p).astype(int)  # spikes.clusters.npy
        metrics = pd.read_parquet(metrics_p)  # clusters.metrics.pqt
```

iii. The AI documented that the reference code uses `load_spiking_data` to load spikes and clusters per probe, then merges probes within a session.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20 ms time bins spanning [-0.5, 1.5) seconds relative to stimulus onset, producing spike count matrices of shape (n_neurons, 100) per trial. Multiple probes within a session are merged by concatenating neuron indices. Spike counts are raw counts (not converted to firing rates).

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

iii. The AI documented this matches the reference code parameters (`binsize=0.02`, `time_window=(-0.5, 1.5)`, `align_time='stimOn_times'`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are quality-filtered using the `label` column from `clusters.metrics.pqt`. Only clusters with `label >= 1` are retained. This is applied per-probe before merging across probes.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
# ...
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
clusters_list.append({'acronym': acr[good]})
```

iii. The AI documented in Step 4 and CONVERSION_NOTES.md that the reference code defines `good_clusters = (clusters['label'] >= 1)`, and the AI followed this same criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes are binned into 100 time bins of 20 ms each, starting at -0.5 s and ending at +1.5 s relative to stimulus onset.

ii.
```python
def bin_spikes(spike_times, spike_clusters, n_neurons, stim_on):
    mats = []
    for t0 in stim_on:
        edges = t0 + TIME_BINS  # TIME_BINS = np.arange(-0.5, 1.5, 0.02)
        # ... bin spikes relative to t0 ...
```

iii. The AI documented that alignment to stimulus onset matches both the task specification and the reference code's `align_time='stimOn_times'` parameter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20 ms (0.02 s). There are 100 time bins per trial spanning [-0.5, 1.5) seconds. No rebinning is applied since the spikes are binned directly from spike times into 20 ms bins.

ii.
```python
BINSIZE = 0.02
# ...
'time_bin_size': BINSIZE * 1000.0,  # 20.0 ms
```

iii. The AI documented this matches the reference code's `binsize=0.02` parameter and the methods text's mention of "nonoverlapping 20-ms bins."

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Time since stimulus onset is derived from the fixed time bin centers, computed as `TIME_BINS + BINSIZE / 2`. It is not derived from any raw data variable per se, but from the predefined time axis relative to stimulus onset.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
# In trial construction:
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. The AI used the bin centers as the time-since-stimulus-onset input, which provides a continuous time-varying signal from -0.49 to 1.49 seconds.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The processing is straightforward: the time axis is defined as bin centers (bin edges + half bin width). The same time vector is used for every trial since all trials have the same fixed time window.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)  # [-0.5, -0.48, ..., 1.48]
TIME_CENTERS = TIME_BINS + BINSIZE / 2  # [-0.49, -0.47, ..., 1.49]
```

iii. No special processing is needed; this is a deterministic time axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time axis is inherently aligned with the neural data because both use the same 100 time bins defined by `TIME_BINS`. Each bin center corresponds to the center of the same time bin used for spike counting.

ii.
```python
# Both neural and input use the same TIME_BINS/TIME_CENTERS definitions
neural_trials = bin_spikes(..., stim_on[valid])  # uses TIME_BINS
inp = np.vstack([TIME_CENTERS.astype(np.float32), ...])  # uses TIME_CENTERS
```

iii. Alignment is guaranteed by construction since both share the same temporal grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Trial number in block is derived from the `probabilityLeft` column in `_ibl_trials.table.pqt`.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. The AI uses changes in `probabilityLeft` to detect block boundaries, then counts trials within each block.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI iterates through the `probabilityLeft` array. When the value changes from the previous trial, it indicates a new block, and the counter resets to 1. Otherwise, the counter increments. The result is a 1-indexed count of the trial's position within its block.

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

iii. The AI noted this is a reasonable derivation from the block structure in the task. The value is broadcast across all time bins for each trial as a per-trial constant.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in `_ibl_trials.table.pqt`.

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. The AI documented that the IBL convention uses 1 for left and -1 for right in the raw data.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The IBL choice encoding (1=left, -1=right) is remapped to the target encoding (left=0, right=1). Trials with other values (e.g., 0 for no-go) are marked as invalid (-1) and filtered out via the `valid` mask.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. The AI documented the IBL sign convention and the mapping to the decoder's binary encoding.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the `probabilityLeft` column in `_ibl_trials.table.pqt`.

ii.
```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. Straightforward extraction from the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The continuous probability values are mapped to categorical indices: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. `np.isclose` is used for comparison to handle floating-point imprecision. Trials with other values are marked as invalid (-1) and filtered out.

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

iii. The AI noted this matches the decoder task specification exactly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` (timestamps) and `_ibl_wheel.position.npy` (position).

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    return np.load(tp), np.load(pp)
```

iii. The AI documented that the reference code also uses wheel position and timestamps.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Wheel velocity is computed from position using `np.gradient`. The velocity is then interpolated to the trial's time bin centers using `np.interp`. Duplicate timestamps are removed before gradient computation to avoid division-by-zero errors.

ii.
```python
def interp_wheel_speed(timestamps, position, stim_on):
    # ... dedup timestamps, handle NaNs ...
    vel = np.gradient(position, timestamps)
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
        trials.append(y.astype(np.float32))
    return trials
```

iii. The AI noted that the reference code uses wheel speed/velocity and bins it to 20 ms windows. The AI uses `np.gradient` for differentiation and `np.interp` for temporal alignment.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Wheel speed is discretized into 3 bins (low=0, mid=1, high=2) using tertile thresholds computed across all valid values within a session. The thresholds are the 1/3 and 2/3 quantiles of the pooled finite wheel speed values for that session.

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

iii. The AI documented discretizing into 3 bins per the task specification. Thresholds are computed per-session from all trials in that session.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same time bin centers (`TIME_CENTERS`) used for neural data. This ensures temporal alignment between the wheel speed output and the neural activity.

ii.
```python
x = t0 + TIME_CENTERS  # Same time centers as neural data
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. Alignment is guaranteed by interpolating to the shared time grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy`, along with their timestamps `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy`.

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
    return streams
```

iii. The AI loads both left and right camera motion energy streams when available.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each available camera stream (left and/or right) is interpolated to the trial's time bin centers. If both left and right are available, they are averaged (mean of finite values). If only one is available, that stream is used directly.

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

iii. The AI documented that whisker motion energy is computed from video motion energy per the methods text. However, the reference code uses left whisker motion energy preferentially, falling back to right only if left is unavailable (not averaging both).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Whisker motion energy is discretized into 3 bins using the same `discretize_tertiles` function as wheel speed. Tertile thresholds are computed per-session from all finite values.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. Same approach as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated to the same `TIME_CENTERS` grid as neural data and wheel speed, ensuring temporal alignment.

ii.
```python
x = t0 + TIME_CENTERS
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. Same alignment approach as wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data quality issues are handled:
- Sessions missing required trial columns (`stimOn_times`, `choice`, `probabilityLeft`) are skipped.
- Sessions without spike data, wheel data, or whisker motion energy are skipped.
- Trials with NaN `stimOn_times` or invalid choice/prior values are excluded.
- Duplicate wheel timestamps are removed before gradient computation.
- NaN values from extrapolation (wheel/whisker outside data range) are handled; NaN whisker ME values default to bin 0 in discretization.
- General exceptions during session processing are caught and the session is skipped.

ii.
```python
# NaN handling in whisker discretization
y[~np.isfinite(x)] = 0  # NaN -> bin 0

# Exception handling
except Exception as e:
    print('skip session due to error', session_dir, repr(e))

# Duplicate timestamp handling
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
```

iii. The AI documented these edge cases in Step 10 of CONVERSION_NOTES.md, noting that duplicate wheel timestamps and all-NaN whisker averaging were discovered and fixed during the critical review.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are spike binning (iterating over each trial and using `np.add.at` for spike counting) and loading data files from disk (especially spike times and clusters for large sessions). The full conversion took approximately 621 seconds (~10.3 minutes) for 159 sessions.

ii.
```python
# bin_spikes iterates per-trial:
for t0 in stim_on:
    edges = t0 + TIME_BINS
    out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    # ... searchsorted, floor, np.add.at ...
```

iii. The AI documented timing information and noted the optimization from ONE API to direct file reads reduced processing time by ~10x.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_spikes` function loops over trials, performing per-trial spike binning with `np.add.at`. This could potentially be vectorized by binning all trials at once and then splitting. The wheel speed interpolation also loops over trials. The `trial_number_in_block` function uses a Python for-loop.

ii.
```python
# bin_spikes trial loop
for t0 in stim_on:
    # ... per-trial binning ...

# interp_wheel_speed trial loop
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, ...)

# trial_number_in_block Python loop
for i, p in enumerate(prob_left):
    if i == 0 or p != cur:
        cur = p; c = 1
    else:
        c += 1
```

iii. The AI noted code efficiency improvements in Step 6 (switching from ONE API to direct reads) but did not explicitly discuss vectorization opportunities for these loops.

## 10-c. What processing does the code repeat multiple times?

i. The `discretize_tertiles` function is called separately for wheel speed and whisker ME, each time pooling all trial values and computing quantiles. The `TIME_CENTERS` array is recomputed/used in multiple functions but defined globally. File path resolution (`find_latest_file`) is called multiple times per session for different file types.

ii.
```python
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. These are separate variables so the repetition is necessary. No significant repeated computation was identified.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes wheel speed thresholds (`wheel_thr`) and whisker motion energy thresholds (`whisk_thr`) and prints them but does not store them in the output. The code also loads and processes both left and right whisker motion energy when only one would suffice (per the reference code which uses left preferentially). Brain region labels are stored as numeric Allen Atlas IDs rather than acronyms, which is less interpretable but functionally equivalent. The `input_names` includes `trial_number_in_block` as a name but `input` is defined with 2 rows, both time-varying, even though trial number in block is constant across time bins.

ii.
```python
# Thresholds printed but not stored
print(f'... wheel_thr={wheel_thr} whisk_thr={whisk_thr} ...')

# trial_number_in_block broadcast to all time bins (unnecessary if decoder handles per-trial)
np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
```

iii. The AI did not explicitly discuss unnecessary processing in CONVERSION_NOTES.md. Broadcasting per-trial values to all time bins is necessary for the decoder format which expects (n_input, n_timepoints) shaped inputs.
