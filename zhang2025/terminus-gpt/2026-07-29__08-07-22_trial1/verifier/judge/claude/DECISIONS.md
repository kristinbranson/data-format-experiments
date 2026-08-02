# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `data/one_cache/Brainwidemap/sessions.pqt`, iterates through each row, and for each session constructs a path to the local ONE cache directory. It loads trial data from parquet files, spike data from `.npy` files, wheel position/timestamps from `.npy` files, and whisker motion energy from `.npy` files. Sessions missing required data (trials, spikes) are skipped.

ii.
```python
def build_dataset(sample=False):
    sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
    ...
    for i, (_, row) in enumerate(sessions.iterrows()):
        ...
        loaded = load_session(row)
        if loaded is None:
            log('  skipped: missing required data')
            continue
```

iii. The AI identified that the data was stored in IBL ONE/ALF cache format and used the local `sessions.pqt` file to enumerate sessions. The reference code uses `bwm_release.csv` and the ONE API, but the AI adapted to load directly from local files since the ONE API was not available in the environment.

## 1-b. How are the data split into subjects (mice)?

i. The AI extracts the subject name from the session metadata row (`row['subject']`). Each unique subject is tracked in a `subject_map` dictionary, and each session is assigned a `subject_idx` pointing into the unique subjects list.

ii.
```python
subject = str(row['subject'])
...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The AI followed the data structure where each session row has a `subject` field. The resulting dataset has 139 subjects, matching the reference paper.

## 1-c. How are the data split into sessions?

i. Each row in `sessions.pqt` corresponds to a session, identified by `lab/subject/date/number`. The AI iterates through all 480 rows in the file, successfully processing 459 sessions (21 skipped due to missing data).

ii.
```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"
```

iii. The AI constructs the directory path from session metadata fields and checks for existence. The final count of 459 sessions matches the reference paper's count.

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` in each session's ALF directory. Each trial corresponds to a row in this table. After loading, a trial mask is applied and trials are further filtered during output construction.

ii.
```python
def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)
```

iii. The AI loads the standard IBL trial table, which is consistent with the reference code's approach using `SessionLoader.load_trials()`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a minimal trial mask requiring finite values in `stimOn_times`, `choice`, and `probabilityLeft`. It also excludes trials where `choice` or `prior` map to NaN (which handles no-choice trials for `choice==0` only indirectly via `map_choice`). The AI does NOT filter by reaction time (0.08-2.0 seconds) and does NOT require finite values in `feedback_times`, `firstMovement_times`, or `feedbackType`.

ii.
```python
def trial_mask(trials):
    need = ['stimOn_times', 'choice', 'probabilityLeft']
    mask = np.ones(len(trials), dtype=bool)
    for c in need:
        if c in trials.columns:
            mask &= np.isfinite(trials[c].to_numpy())
    return mask
```

iii. The AI's CONVERSION_NOTES.md states it applies "reference trial mask and additionally drop trials missing requested outputs." However, the reference code (`load_trials_and_mask`) applies more stringent filtering: `min_rt=0.08`, `max_rt=2.0`, NaN exclusion on `stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType`, and excludes `choice==0` (no-response trials). The data paper also explicitly states: "trials were excluded if the time between stimulus onset and the first movement of the wheel were outside the range of 0.08-2.00 s."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files found in each probe's pykilosort directory. Cluster quality is assessed from `clusters.metrics.pqt`.

ii.
```python
def load_spikes_and_regions(session_dir):
    probe_dirs = find_probe_dirs(session_dir)
    ...
    for pd_ in probe_dirs:
        base = latest_revision_dir(pd_)
        st = base / 'spikes.times.npy'
        sc = base / 'spikes.clusters.npy'
        cm = base / 'clusters.metrics.pqt'
```

iii. The AI identified the correct raw data variables (spike times and cluster assignments) from the IBL ALF data structure.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 20ms non-overlapping time bins relative to stimulus onset. For each trial, a (n_neurons x n_timebins) matrix of spike counts is created using `np.digitize` and `np.add.at`. Probes within a session are merged by remapping cluster IDs.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    ...
    tbin = np.digitize(st, edges) - 1
    ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The AI binned raw spike counts at 20ms resolution, consistent with the methods paper's description for wheel/whisker decoding. However, the methods paper states 50ms bins for choice/prior decoding and 20ms for wheel/whisker. The AI used 20ms uniformly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons by checking `clusters.metrics.pqt` for the `label` column equal to 1 (good units), or falls back to `ks2_label == 'good'`. Only spikes from good clusters are retained. This produces 75,708 total well-isolated neurons across all sessions.

ii.
```python
if cm.exists():
    m = pd.read_parquet(cm)
    cols = {c.lower(): c for c in m.columns}
    if 'label' in cols:
        good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
    elif 'ks2_label' in cols:
        good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
if good_ids is None:
    good_ids = np.unique(spikes_c)
```

iii. The AI's quality filtering yields 75,708 neurons, matching the data paper's count. However, the reference code's `prepare_data` function calls `load_spiking_data` without the `qc` parameter (default `None`), meaning it loads ALL units (621,733), not just good ones. The data paper's analyses use well-isolated neurons, creating ambiguity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data is aligned to stimulus onset (`stimOn_times`). The time window spans from -0.2s to 1.0s relative to stimulus onset, producing 60 time bins at 20ms resolution.

ii.
```python
T_START = -0.2
T_END = 1.0
ALIGN_EVENT = 'stimulus onset'

def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
```

iii. The task instructions specify "Temporally align based on stimulus onset", which the AI follows. However, the reference code uses a time window of (-0.5, 1.5) = 2.0s with T=100 bins (at 20ms). The methods paper states: "Recordings are split into 2-s trials, each divided into 20-ms bins, producing T = 100 time steps." The AI's window of (-0.2, 1.0) = 1.2s with T=60 bins does not match.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a uniform 20ms bin size throughout. No rebinning is applied. The bin edges are constructed with `np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)`.

ii.
```python
BIN_SIZE_S = 0.02
```

iii. 20ms matches the methods paper's bin size for wheel/whisker decoding. The methods paper uses 50ms for choice/prior decoding, but the task instructions require a single uniform bin size for the combined decoder. Using 20ms is reasonable.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the time bin centers, computed from `T_START`, `T_END`, and `BIN_SIZE_S`. It is implicitly relative to `stimOn_times` since that's the alignment event.

ii.
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. The AI correctly computes time-since-stimulus-onset as the bin center times relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Time bin edges are created with `np.arange` from -0.2 to 1.0 at 20ms steps. Centers are the midpoints of adjacent edges. These centers (in seconds relative to stimulus onset) form the first row of the input array for each trial.

ii.
```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. This is a straightforward computation. The values range from -0.19 to 0.99 seconds.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Both the neural data and the time-since-stimulus-onset input use the same time bin edges and centers derived from the alignment event. They are inherently aligned by construction.

ii. Both use `build_time_edges()` and `build_time_centers()` which derive from the same `T_START`, `T_END`, and `BIN_SIZE_S` constants.

iii. The alignment is correct by construction since the same time grid is used for both.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trial table. Consecutive trials with the same `probabilityLeft` value are counted as belonging to the same block.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    ...
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
    return out
```

iii. The AI uses changes in `probabilityLeft` to detect block boundaries, which is a reasonable approach since block identity is defined by the prior probability.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A run-length counter is applied: it starts at 1 and increments when consecutive trials have the same `probabilityLeft`, resetting to 1 when it changes. This produces a 1-indexed trial position within each block. The value is then broadcast across all time bins for the trial.

ii.
```python
out[0] = 1
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
    out[i] = run
```

iii. This is a reasonable derivation. The values range from 1 to 99, reflecting block lengths of up to 99 trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column of the trial table (`_ibl_trials.table.pqt`).

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. This matches the IBL trial table structure.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL convention uses `choice=1` for left (CCW) and `choice=-1` for right (CW). The AI maps `1 -> 0` (left) and `-1 -> 1` (right), consistent with the task specification "left = 0, right = 1". The per-trial choice value is broadcast across all time bins.

ii.
```python
def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

iii. The mapping is correct per IBL conventions and the task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column of the trial table.

ii.
```python
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. This matches the IBL trial data structure.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps `probabilityLeft` values: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, using `np.isclose` for comparison. Values that don't match are mapped to NaN and those trials are excluded. The per-trial value is broadcast across all time bins.

ii.
```python
def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
    return np.nan
```

iii. The mapping matches the task specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` in the session's ALF directory.

ii.
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    ...
    return np.load(ts), np.load(pos)
```

iii. The AI loads the raw wheel data files directly from the local cache.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel speed by: (1) taking finite differences of timestamps and positions, (2) computing absolute velocity as `|dp/dt|`, (3) binning into 20ms trial-aligned windows using `bin_signal` (averaging values that fall in each bin).

ii.
```python
def wheel_speed(ts, pos):
    ...
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    v = np.zeros_like(pos, dtype=np.float32)
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv
```

iii. The reference code uses `SessionLoader.load_wheel()` which: (1) interpolates wheel position to 1000 Hz uniform sampling, (2) applies an 8th-order Butterworth low-pass filter (20 Hz corner frequency) to compute velocity, (3) takes absolute value for speed. The AI's approach is significantly different: raw finite differences without interpolation or filtering, producing noisier velocity estimates.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes session-level quantiles at 1/3 and 2/3 of all valid wheel speed values, then uses `np.digitize` to assign each time bin to one of 3 categories (0=low, 1=mid, 2=high). NaN values are filled with the lower quantile threshold before discretization.

ii.
```python
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
...
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The task instructions specify "Wheel speed discretized into 3 bins" but don't specify the thresholding method. Using session-level tertile quantiles is a reasonable approach for equal-frequency binning.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to stimulus onset using the same time bin centers as the neural data. The `bin_signal` function bins continuous wheel speed values into the trial-aligned 20ms bins by averaging values that fall within each bin.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The alignment uses the same time grid as neural data, which is correct. The reference code also aligns behavioral signals to the same time grid as spikes, though using interpolation rather than binning.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` files found in the session's ALF directory.

ii.
```python
def find_motion_energy(session_dir):
    alf = session_dir / 'alf'
    left = sorted(alf.glob('#*/leftCamera.ROIMotionEnergy.npy'))
    right = sorted(alf.glob('#*/rightCamera.ROIMotionEnergy.npy'))
    return (left[-1] if left else None), (right[-1] if right else None)
```

iii. The AI identifies the correct raw data files for motion energy.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads BOTH left and right camera motion energy arrays, truncates to the shorter length, and AVERAGES them. Timestamps are guessed from camera features files or generated via `np.linspace` over the session interval. The averaged signal is then binned into trial-aligned 20ms bins using `bin_signal`.

ii.
```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
    if not vals:
        return None
    n = min(len(v) for v in vals)
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr
```

iii. The reference code loads whisker motion energy from a single camera (left first, falling back to right), not averaging both. The AI's approach of averaging left and right cameras is different from the reference. Additionally, the AI guesses timestamps rather than loading proper camera timestamps, which could cause temporal misalignment.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: session-level quantiles at 1/3 and 2/3, then `np.digitize` into 3 bins.

ii.
```python
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
...
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. Same rationale as wheel speed discretization. Session-level tertile quantiles produce approximately equal-frequency bins.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is aligned to stimulus onset using the same time bin centers as neural data. The `bin_signal` function bins values into trial-aligned 20ms bins. However, timestamps are estimated rather than loaded from the data.

ii.
```python
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
...
ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The guessed timestamps are a significant concern. The `guess_motion_timestamps` function tries camera features files first, then falls back to `np.linspace` over the session interval. Inaccurate timestamps would cause misalignment between whisker motion energy and neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in several ways: (1) sessions with missing trial tables or spike data are skipped entirely; (2) trials with NaN in choice or prior are excluded; (3) NaN values in wheel speed and whisker motion energy are replaced with the lower quantile threshold before discretization; (4) if no wheel or motion energy data exists for a session, NaN-filled arrays are used and then discretized with default thresholds; (5) if no clusters.metrics.pqt exists, all clusters are used.

ii.
```python
if loaded is None:
    log('  skipped: missing required data')
    continue
...
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False)
```

iii. The approach is functional but has issues: filling NaN behavioral data with the lower quantile threshold biases the discretized output distribution. The reference code's `get_behavior_per_interval` marks intervals with missing/NaN data as bad intervals and excludes them.

## 12-a. What are the most time-consuming steps of the code?

i. Based on the conversion log, each session takes 15-90 seconds. The dominant cost is spike binning (`bin_spikes_for_trial`) which is called for every trial in every session, involving `np.searchsorted` and `np.add.at` operations on potentially millions of spikes. The full conversion of 459 sessions took approximately 5.5 hours.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    ...
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, edges[-1], side='right')
    ...
    np.add.at(mat, (sc, tbin), 1)
```

iii. The CONVERSION_NOTES.md estimates "Full run likely several hours without optimization." The reference code uses multiprocessing for spike binning which the AI does not.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop over trials in `load_session` (line 289) calls `bin_spikes_for_trial` for each trial sequentially. The `bin_signal` function (line 228) also loops over time bins. The `map_choice` and `map_prior` functions are called per-element rather than vectorized.

ii.
```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The reference code uses `bincount2D` and multiprocessing to parallelize spike binning. The AI's sequential approach is significantly slower.

## 12-c. What processing does the code repeat multiple times?

i. `build_time_edges()` and `build_time_centers()` are called multiple times (once per trial in `bin_spikes_for_trial`, and once in `load_session`). `np.searchsorted` is called per-trial rather than precomputing all trial boundaries at once.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()  # rebuilds edges every trial
```

iii. These are minor inefficiencies but contribute to overall runtime.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores all-zero neural trials (85 out of 294,851) which are likely uninformative. Brain region information is stored as 'unknown' for all neurons, making the `brain_regions` and `brain_region_idx` fields useless. The code also computes wheel speed and whisker motion energy even for sessions where those data may not be available, filling with NaN arrays that are then discretized into artificial category values.

ii.
```python
brain_regions = ['unknown']
...
region_idx = np.zeros(n_neurons, dtype=np.int64)
```

iii. The brain regions could have been populated from the cluster metadata (acronym field), which the reference code does track.
