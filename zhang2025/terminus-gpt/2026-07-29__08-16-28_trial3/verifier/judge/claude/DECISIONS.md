# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly globbing the local ONE cache directory structure (`data/one_cache/*/Subjects/*/*/*`) to find session directories. It looks for directories where the last path component is numeric and an `alf/` subdirectory exists. Trials are loaded from parquet files found via glob patterns, spikes from `.npy` files in probe subdirectories, wheel data from `_ibl_wheel.*.npy` files, and motion energy from `*Camera.ROIMotionEnergy.npy` files. No ONE API is used for data resolution.

ii.
```python
data_root = Path('data/one_cache')

def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)

def load_trials_table(session_alf: Path):
    pqt = find_latest_file(session_alf, '#*/_ibl_trials.table.pqt')
    if pqt is None:
        pqt = session_alf / '_ibl_trials.table.pqt'
    ...
    return pd.read_parquet(pqt), pqt
```

iii. The AI chose direct file system access over the ONE API after finding that ONE object loading was slow (~27s vs ~3s for 2 sessions). The CONVERSION_NOTES.md states: "Replaced ONE object loading with direct local `.npy`/`.pqt` reads from session ALF directories, reducing sample conversion time from ~27 s to ~3 s for 2 sessions." However, this approach resulted in many sessions being missed (159 vs the expected ~444) because the file glob patterns did not resolve all camera/motion energy files that the ONE API would have found.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the directory path structure. The subject name is the third-to-last component of the session directory path (e.g., `data/one_cache/lab/Subjects/SUBJECT/date/number`).

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```

iii. The directory structure of the ONE cache encodes the subject name in the path, so this is a straightforward extraction.

## 1-c. How are the data split into sessions?

i. Sessions correspond to directories found by globbing the ONE cache. Each directory at the `*/Subjects/*/*/*` level (where the last component is numeric and has an `alf/` subdirectory) is treated as one session.

ii.
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)
```

iii. The ONE cache directory structure inherently organizes data by session, so the split is given by the file system.

## 1-d. How are the data split into trials?

i. Trials correspond to rows of the `_ibl_trials.table.pqt` parquet file loaded for each session. Each row is one trial.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
```

iii. The trials table is already structured with one row per trial, so no splitting is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on: (1) finite `stimOn_times`, (2) valid choice (mapped to 0 or 1, excluding no-response trials where choice=0), and (3) valid `probabilityLeft` (mapped to 0, 1, or 2). The AI does NOT filter by reaction time bounds (80ms-2s) and does NOT check for wheel/camera data coverage within the trial window.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The CONVERSION_NOTES.md under Step 4 mentions "Apply explicit valid-trial mask before packaging trials" and under Step 5 mentions "Filter to valid trials explicitly: Use trial masks / valid aligned intervals." However, the actual code does not implement the reaction time filtering (MIN_RT=0.08, MAX_RT=2.0) described in the data paper, nor does it check that wheel and camera data cover the trial window.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy`, loaded from probe subdirectories within each session. Cluster quality labels come from `clusters.metrics.pqt`.

ii.
```python
st = np.load(st_p)  # spikes.times.npy
sc = np.load(sc_p).astype(int)  # spikes.clusters.npy
metrics = pd.read_parquet(metrics_p)  # clusters.metrics.pqt
```

iii. These are the standard spike-sorted outputs from the IBL pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over the trial window (-0.5 to 1.5s relative to stimulus onset), producing spike counts per neuron per bin. The counts are NOT divided by the bin width, so the neural data is stored as raw spike counts rather than firing rates in Hz. When a session has multiple probes, neurons are merged with continuous indexing.

ii.
```python
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

iii. The AI's CONVERSION_NOTES.md documents using 20ms bins aligned to stimulus onset, consistent with the reference code. However, the code stores spike counts rather than firing rates (Hz). The reference code divides by BIN to convert to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in `clusters.metrics.pqt` are kept. The AI does NOT filter out clusters mapped to 'void' in the Beryl atlas (channels outside the brain), because it does not use the Beryl atlas mapping at all. Instead, brain region labels are stored as numeric Allen CCF region IDs converted to strings.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
...
good = labels >= 1
```

For brain regions:
```python
acr = np.array(['void'] * nclu, dtype=object)
if cl_chan_p.exists() and ch_brain_p.exists():
    cluster_channels = np.load(cl_chan_p).astype(int)
    channel_region_ids = np.load(ch_brain_p).astype(int)
    ...
    acr = np.array([str(x) for x in region_ids], dtype=object)
```

iii. The CONVERSION_NOTES.md mentions "strict cluster label filtering (`label >= 1`)" which matches the data paper's quality control. However, the 'void' filtering is not applied. The CONVERSION_NOTES.md acknowledges "brain-region labels are numeric region IDs as strings rather than acronyms because atlas mapping package was unavailable in the environment."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, spikes in the window from `stimOn_times + T_START` to `stimOn_times + T_END` are selected and binned relative to stimulus onset.

ii.
```python
for t0 in stim_on:
    edges = t0 + TIME_BINS
    ...
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
    ts = spike_times[lo:hi] - t0
```

iii. This matches the instructions which specify "Temporally align based on stimulus onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20ms (0.02s). The 2s window (-0.5 to 1.5s) is divided into 100 bins. No rebinning is applied.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
N_BINS = len(TIME_BINS)  # 100
```

iii. This matches the reference code's `binsize=0.02` and the methods paper's description of 20ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table (the alignment event) and the fixed time bin grid. The input is the center of each 20ms bin in the trial window.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    ...
])
```

iii. The time input is defined by the binning grid, not from any raw data variable beyond the alignment event.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The bin centers are computed as `T_START + i*BINSIZE + BINSIZE/2` for each bin index i. This produces values from -0.49 to 1.49 in 0.02 steps.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. No processing beyond computing bin centers. The same time vector is used for every trial.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same bin centers as the neural data bins, so they are inherently aligned.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

The neural data uses the same `TIME_BINS` for binning:
```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

iii. Both use the same temporal grid, so alignment is automatic.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. Changes in `probabilityLeft` mark block boundaries.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    ...
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

iii. The trials table has no explicit block identifier, so blocks are inferred from consecutive trials with the same `probabilityLeft` value.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number in block is computed by iterating through trials and incrementing a counter for each trial with the same `probabilityLeft` as the previous trial. When `probabilityLeft` changes, the counter resets to 1. The count starts at 1 (not 0). The count is computed before trial filtering, so dropped trials still advance the count.

ii.
```python
def trial_number_in_block(prob_left):
    ...
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
    return out
```

Called before filtering:
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
```

iii. The count starts at 1 rather than 0, which differs from the reference's `cumcount()` which starts at 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table. IBL convention: +1 = left, -1 = right, 0 = no response.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. Maps left (+1) to 0 and right (-1) to 1, matching the instructions "left = 0, right = 1". No-response trials (choice=0) are excluded via the valid mask.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple recoding from +1/-1 to 0/1. No-response trials are filtered out.

ii. Same as 5-a above.

iii. Straightforward mapping.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, which takes values 0.2, 0.5, and 0.8.

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

iii. Maps to 0, 1, 2 per the instructions "0.2 -> 0, 0.5 -> 1, 0.8 -> 2."

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple recoding from 0.2/0.5/0.8 to 0/1/2. Uses `np.isclose` for floating point comparison.

ii. Same as 6-a above.

iii. Straightforward mapping using `np.isclose` for robust float comparison.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded directly from the session's alf directory.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    ...
    return np.load(tp), np.load(pp)
```

iii. These are the standard IBL wheel data files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel velocity by applying `np.gradient` to the raw wheel position and timestamps, without prior interpolation to a regular grid or low-pass filtering. The velocity (not speed - no absolute value is taken) is then interpolated to trial-aligned bin centers using `np.interp`. Finally, the values are discretized into 3 bins using session-level tertiles (quantiles at 1/3 and 2/3).

ii.
```python
def interp_wheel_speed(timestamps, position, stim_on):
    ...
    vel = np.gradient(position, timestamps)
    trials = []
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
        trials.append(y.astype(np.float32))
    return trials
```

Discretization:
```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays ...])
    q1, q2 = np.quantile(allv, [1/3, 2/3])
    ...
```

iii. The reference uses `SessionLoader.load_wheel()` which internally interpolates to 1000Hz, applies a Butterworth low-pass filter at 20Hz, then computes velocity. The AI bypasses this entirely with `np.gradient` on raw data. This causes RuntimeWarnings from duplicate timestamps and produces a noisier, unfiltered velocity signal. Additionally, the AI computes velocity (signed) not speed (unsigned) since it does not take the absolute value.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The continuous wheel velocity values across all trials in a session are split into 3 bins at the 1/3 and 2/3 quantiles, producing roughly equal-sized categories labeled 0, 1, 2.

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

iii. Uses session-level quantiles to define thresholds, which produces roughly equal class sizes. This matches the reference's approach in principle. However, because velocity (not speed) is used, the thresholds can be negative (as seen in the output logs, e.g., `wheel_thr=(-0.036, 0.044)`), which is semantically different from thresholding speed (absolute velocity).

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel velocity is interpolated to the same bin centers (`TIME_CENTERS`) as the neural data, aligned to stimulus onset.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. Using the same time grid ensures alignment with neural data.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` with their corresponding timestamps `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy`. Both cameras are used when available, and their values are averaged.

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
    ...
    return streams
```

iii. Unlike the reference which uses only one camera (left preferred), the AI loads both cameras when available and averages them in `interp_motion_energy`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each available camera's motion energy is interpolated to trial-aligned bin centers. When both cameras are available, their values are averaged (finite values only). The result is then discretized into 3 bins using session-level tertiles.

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

iii. The averaging of both cameras differs from the reference, which uses only one camera (left preferred, right as fallback).

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: session-level tertiles at 1/3 and 2/3 quantiles, producing 3 roughly equal-sized bins.

ii. Uses the same `discretize_tertiles` function as wheel speed (see 7-c).

iii. Matches the reference's percentile-based approach.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The motion energy trace is interpolated to the same bin centers (`TIME_CENTERS`) as the neural data, aligned to stimulus onset.

ii.
```python
x = t0 + TIME_CENTERS
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. Same time grid as neural data ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing data is handled by: (1) skipping sessions without required trial columns, spikes, wheel, or whisker motion energy data; (2) skipping sessions with fewer than 2 valid trials; (3) a broad try/except around each session that catches and prints any error; (4) handling duplicate wheel timestamps by deduplication; (5) handling NaN values in whisker interpolation. However, individual trials are not checked for coverage by wheel/camera data - if data is unavailable for specific trials, `np.interp` returns NaN (left/right boundaries) but these NaN values are then set to bin 0 in the discretization step.

ii.
```python
except Exception as e:
    print('skip session due to error', session_dir, repr(e))
```

```python
y[~np.isfinite(x)] = 0  # in discretize_tertiles
```

iii. The broad try/except approach may mask bugs. Setting NaN discretized values to 0 ("low") rather than filtering those trials is a questionable choice.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike sorting data from disk (reading large `.npy` files for spike times and clusters) is the most time-consuming step, followed by spike binning.

ii.
```python
st = np.load(st_p)  # spikes.times.npy - can be hundreds of MB
sc = np.load(sc_p).astype(int)  # spikes.clusters.npy
```

iii. The full conversion took 621 seconds for 159 sessions (~3.9s per session on average).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial using `np.add.at`, which could be vectorized by offsetting spike indices across trials. The wheel interpolation loop iterates per trial for `np.interp`. The `trial_number_in_block` function iterates element-by-element in Python.

ii.
```python
for t0 in stim_on:  # per-trial spike binning
    ...
    np.add.at(out, (cl[m], bins[m]), 1)

for t0 in stim_on:  # per-trial wheel interpolation
    ...
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)

for i, p in enumerate(prob_left):  # element-wise block counting
    ...
```

iii. These loops are not major bottlenecks given the overall runtime, but the trial_number_in_block loop could have been vectorized using pandas or numpy operations like the reference's `cumcount()`.

## 10-c. What processing does the code repeat multiple times?

i. Wheel timestamp deduplication is repeated for each session but could be considered part of session-specific processing. No major repeated processing was identified.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. When both left and right camera motion energy are available, both are loaded and averaged, whereas only one camera is needed. The broad exception handling also means some computation may be wasted on sessions that ultimately fail.

ii.
```python
for tp, mp in [(left_t, left_me), (right_t, right_me)]:
    if tp is not None and mp is not None ...:
        streams.append(...)
```

iii. Loading and processing both cameras when only one is needed adds unnecessary I/O and computation.
