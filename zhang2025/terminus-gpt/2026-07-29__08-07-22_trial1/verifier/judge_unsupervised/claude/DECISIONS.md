# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session metadata from `data/one_cache/Brainwidemap/sessions.pqt`, a parquet file listing 480 sessions with columns `lab`, `subject`, `date`, `number`, `task_protocol`, `projects`. It iterates through each row, constructs a directory path to the session's ALF data, and loads trial tables, spike data, wheel data, and motion energy from the ONE cache directory structure.

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
        ...
```

```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"
```

iii. The AI noted the IBL ONE cache structure during Step 2 exploration and chose to iterate through the Brainwidemap sessions.pqt to enumerate all sessions. This is consistent with how IBL data is organized.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `subject` field in each session's row from sessions.pqt. A dictionary maps subject names to indices, building the `subjects` list and `subject_idx` array.

ii.
```python
subject = str(row['subject'])
...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The AI correctly uses the subject field from the session metadata. The final dataset reports 139 subjects matching the paper's count.

## 1-c. How are the data split into sessions?

i. Each row in sessions.pqt represents one session. Sessions are identified by the combination of lab/subject/date/number. Each successfully loaded session becomes one element in the `neural`, `input`, and `output` lists.

ii.
```python
for i, (_, row) in enumerate(sessions.iterrows()):
    ...
    loaded = load_session(row)
    if loaded is None:
        continue
    neural, inp, out, subj, region_names, region_idx = loaded
    all_neural.append(neural)
    all_input.append(inp)
    all_output.append(out)
```

iii. 480 sessions were attempted; 459 were successfully loaded (21 skipped due to missing data files).

## 1-d. How are the data split into trials?

i. Trials are loaded from `_ibl_trials.table.pqt` in each session's ALF directory. The function `find_trial_table` searches both revision-tagged directories (`#*/`) and the base `alf/` directory.

ii.
```python
def find_trial_table(session_dir):
    alf = session_dir / 'alf'
    cands = sorted(alf.glob('#*/_ibl_trials.table.pqt'))
    if cands:
        return cands[-1]
    p = alf / '_ibl_trials.table.pqt'
    return p if p.exists() else None

def load_trials(session_dir):
    p = find_trial_table(session_dir)
    if p is None:
        return None
    return pd.read_parquet(p)
```

iii. The AI follows IBL convention of loading the trial table from the ALF directory, preferring the latest revision when multiple exist.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a minimal trial mask: it requires `stimOn_times`, `choice`, and `probabilityLeft` to be finite (non-NaN). Trials where `choice` or `prior` mapping produces NaN are also excluded (which implicitly removes no-choice trials where choice==0). Sessions with fewer than 2 valid trials are skipped entirely. However, the AI does NOT apply reaction time filtering (min_rt=0.08, max_rt=2.0) or check for NaN in `feedback_times`, `firstMovement_times`, or `feedbackType`, which the reference code's `load_trials_and_mask` does.

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

And in `load_session`:
```python
mask = trial_mask(trials)
trials = trials.loc[mask].reset_index(drop=True)
if len(trials) < 2:
    return None
```

iii. The AI's CONVERSION_NOTES state the intent to "apply the same trial mask logic as the reference code" but the implementation omits the reaction time filter and additional NaN exclusions present in the reference `load_trials_and_mask` function.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` files found in each probe's pykilosort directory within the session's ALF folder.

ii.
```python
def load_spikes_and_regions(session_dir):
    probe_dirs = find_probe_dirs(session_dir)
    ...
    for pd_ in probe_dirs:
        base = latest_revision_dir(pd_)
        st = base / 'spikes.times.npy'
        sc = base / 'spikes.clusters.npy'
        ...
        spikes_t = np.load(st)
        spikes_c = np.load(sc)
```

iii. The AI correctly identifies spike times and cluster assignments as the source data for neural activity, consistent with IBL data conventions.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into non-overlapping 20ms bins aligned to stimulus onset for each trial. The time window spans -0.2s to 1.0s relative to stimulus onset, producing 60 time bins per trial. Spike counts are accumulated using `np.add.at` after digitizing spike times into bin edges.

ii.
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0

def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, edges[-1], side='right')
    st = spike_times[lo:hi]
    sc = spike_clusters[lo:hi]
    ...
    tbin = np.digitize(st, edges) - 1
    ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The AI chose 20ms bins consistent with the reference papers. However, the time window (-0.2s to 1.0s, 60 bins) differs from the reference code's (-0.5s to 1.5s, 100 bins = 2s total).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `clusters.metrics.pqt` from each probe's pykilosort directory. Clusters with `label == 1` are kept as "good" units. If no metrics file is found, all clusters are kept.

ii.
```python
cm = base / 'clusters.metrics.pqt'
...
if cm.exists():
    m = pd.read_parquet(cm)
    cols = {c.lower(): c for c in m.columns}
    if 'label' in cols:
        good_ids = np.flatnonzero(m[cols['label']].to_numpy() == 1)
    elif 'ks2_label' in cols:
        good_ids = np.flatnonzero(m[cols['ks2_label']].astype(str).str.lower().eq('good').to_numpy())
if good_ids is None:
    good_ids = np.unique(spikes_c)
keep = np.isin(spikes_c, good_ids)
```

iii. The reference code uses `clusters_labeled['label'] >= qc` with `qc=1`. Since IBL labels are 0 (bad) or 1 (good), `== 1` and `>= 1` are functionally equivalent. The agent's approach matches.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset (`stimOn_times`). For each trial, time bin edges are computed as `stim_time + np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)`.

ii.
```python
stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. Alignment to stimulus onset is correct per the instructions ("Temporally align based on stimulus onset").

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s). No rebinning is applied; spikes are directly binned from raw spike times into 20ms bins. The total number of time bins per trial is 60 (from -0.2s to 1.0s).

ii.
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0
```

iii. The 20ms bin size matches both the reference paper ("firing rates in 20-ms bins") and the reference code (`binsize: 0.02`). However, the time window length differs: 1.2s (60 bins) vs. reference's 2.0s (100 bins, from -0.5s to 1.5s). The reference code's `0_data_caching.py` specifies `'time_window': (-0.5, 1.5)` and `'interval_len': 2`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time-since-stimulus-onset input is derived from the time bin centers, which are computed from the constants T_START and T_END. It is not loaded from raw data but constructed analytically.

ii.
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2

# In load_session:
centers = build_time_centers().astype(np.float32)
...
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. The time axis is constructed from the bin centers, representing continuous time since stimulus onset. This is correct conceptually.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Time edges are generated via `np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)`, then centers are computed as the midpoints of adjacent edges. The resulting array represents seconds relative to stimulus onset.

ii.
```python
def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)

def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. The processing is straightforward: bin centers from -0.19s to 0.99s in 20ms steps.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same time bin centers are used for both neural binning and the time input, so they are inherently aligned. The first row of each trial's input matrix is the time centers array.

ii.
```python
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. Using identical time bins ensures alignment between neural data and the time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. Consecutive trials with the same `probabilityLeft` value are assumed to belong to the same block.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

iii. The agent uses probabilityLeft as a proxy for block identity, which is the standard IBL approach since block boundaries coincide with changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A run-length counter tracks consecutive trials with the same `probabilityLeft` value. Each time the value changes, the counter resets to 1. The result is broadcast across time bins as a per-trial constant.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    if len(prob_left) == 0:
        return out
    run = 1
    out[0] = 1
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
    return out
```

iii. This is a reasonable approach. The instructions specify "Trial number in block" as a continuous, per-trial input. The counter starts at 1 and increments for each consecutive trial in the same block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column in the trials table (`_ibl_trials.table.pqt`).

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. Standard IBL trial variable.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL convention is choice=1 for left (CCW) and choice=-1 for right (CW). The agent maps: 1 -> 0 (left), -1 -> 1 (right). Choice=0 (no-choice) maps to NaN, and those trials are later excluded.

ii.
```python
def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

iii. The mapping matches the instructions: "left = 0, right = 1". Per-trial value broadcast across all time bins.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column in the trials table.

ii.
```python
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. Standard IBL trial variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Maps: 0.2 -> 0, 0.5 -> 1, 0.8 -> 2. Uses `np.isclose` for comparison. Values not matching any of these map to NaN.

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

iii. Matches the instructions: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` in the session's ALF directory.

ii.
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    if not (pos.exists() and ts.exists()):
        return None, None
    return np.load(ts), np.load(pos)
```

iii. The AI loads raw wheel position and timestamps, then computes speed from these.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Speed is computed as the absolute value of the derivative of position (|dp/dt|). The agent manually differentiates: computes `np.diff(pos) / np.diff(ts)` and takes the absolute value. Midpoint timestamps are used.

ii.
```python
def wheel_speed(ts, pos):
    if ts is None or pos is None or len(ts) < 2:
        return None, None
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    v = np.zeros_like(pos, dtype=np.float32)
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv
```

iii. The reference code uses `SessionLoader.load_wheel()` which computes velocity through the brainbox library's interpolation-based approach. The agent's manual differentiation is a simplified approximation. The reference code accesses `sess_loader.wheel['velocity']` and takes `np.abs()` for speed.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Per-session quantile-based discretization: all finite wheel speed values within a session are pooled, 1/3 and 2/3 quantiles are computed, and `np.digitize` maps values into 3 bins (0=low, 1=mid, 2=high). NaN values are replaced with the lower quantile boundary before digitization.

ii.
```python
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
...
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The instructions say "Wheel speed discretized into 3 bins, time-varying". The agent chose session-level quantile boundaries. The discretization is per-session rather than global.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is binned into the same stimulus-onset-aligned time bins as neural data using `bin_signal`, which averages wheel speed values falling within each 20ms bin.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers) if wheel_v is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. Uses the same time bin structure as neural data, ensuring temporal alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` in the session's ALF directory. The AI loads both cameras when available and averages them.

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

iii. The reference code uses only the left camera whisker motion energy (`sess_loader.load_motion_energy(views=['left'])`), accessing `sess_loader.motion_energy['leftCamera']['whiskerMotionEnergy']`. The AI's decision to average left and right cameras diverges from the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Left and right camera motion energy values are loaded, truncated to the shorter length, and averaged. The result is then binned into trial-aligned 20ms bins using `bin_signal`.

ii.
```python
n = min(len(v) for v in vals)
arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
```

iii. The averaging of both cameras is not consistent with the reference code, which uses left camera only.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: per-session quantile-based discretization into 3 bins using 1/3 and 2/3 quantile boundaries.

ii.
```python
mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
...
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. Follows same discretization strategy as wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy timestamps are estimated via `guess_motion_timestamps`, which attempts to find camera feature timestamps, and falls back to `np.linspace` spread across the session interval. These are then used to bin values into stimulus-onset-aligned 20ms bins.

ii.
```python
def guess_motion_timestamps(session_dir, n):
    alf = session_dir / 'alf'
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        try:
            df = pd.read_parquet(feats[0])
            for c in df.columns:
                if 'times' in c.lower() or 'timestamp' in c.lower():
                    x = df[c].to_numpy()
                    if len(x) >= n:
                        return x[:n]
        except Exception:
            pass
    trials = load_trials(session_dir)
    if trials is not None and 'stimOn_times' in trials.columns:
        t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
        t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
        return np.linspace(t0, t1, n, dtype=np.float32)
    return np.arange(n, dtype=np.float32) * BIN_SIZE_S
```

iii. The reference code obtains motion energy timestamps from `sess_loader.motion_energy['leftCamera']['times']`, which internally maps to `_ibl_leftCamera.times.npy`. The agent's `guess_motion_timestamps` is a heuristic fallback that does not use the actual camera timestamp files, potentially causing temporal misalignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches:
- Sessions with missing trial tables, spike data, or fewer than 2 valid trials are skipped entirely (21 out of 480).
- NaN values in wheel speed and whisker motion energy are replaced with the lower quantile boundary before discretization.
- Trials missing valid choice or prior values (NaN after mapping) are excluded.
- All-zero neural activity trials (86 out of 294,851) are tolerated.
- If `clusters.metrics.pqt` is missing, all spike clusters are kept (no quality filtering).

ii.
```python
# Missing sessions
if loaded is None:
    log('  skipped: missing required data')
    continue

# NaN in behavioral signals
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] ...), wq, right=False)

# No metrics file fallback
if good_ids is None:
    good_ids = np.unique(spikes_c)
```

iii. The AI's CONVERSION_NOTES document these edge cases, noting the 86 all-zero trials as a rare occurrence (~0.029%).

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the per-trial spike binning in `bin_spikes_for_trial`, called in a loop over all trials within each session. The full conversion took approximately 6.5 hours for 459 sessions. The `bin_signal` function for wheel and motion energy is also called per-trial and contains an inner loop over time bins.

ii.
```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The AI noted in Step 7 that sample conversion took ~35-75s per session depending on spike count.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The `bin_signal` function contains a Python for-loop over time bins that could be vectorized. The spike cluster remapping also uses a Python list comprehension for remapping.

ii.
```python
# bin_signal inner loop
for i in range(len(centers)):
    m = idx == i
    if np.any(m):
        out[i] = np.nanmean(values[m])

# Cluster remapping
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
```

iii. Both loops process potentially large arrays element-by-element. The bin_signal loop could use `np.bincount` or similar. The cluster remapping could use vectorized index mapping.

## 10-c. What processing does the code repeat multiple times?

i. `build_time_edges()` and `build_time_centers()` are called repeatedly (once per trial for neural binning, once for centers computation in load_session). `load_trials` is called again inside `guess_motion_timestamps` when the fallback is used, duplicating the trial table loading. The `discretize_three_bins` function is called but its results are not used (the per-session quantile thresholds are computed and applied separately).

ii.
```python
# discretize_three_bins called but results unused
wheel_bins_all = discretize_three_bins(all_wheel_cont) ...
me_bins_all = discretize_three_bins(all_me_cont) ...
# Then separate quantile computation used instead:
wq = np.quantile(all_wheel_cont, [1/3, 2/3]) ...
```

iii. The double trial loading and unused discretization call represent wasted computation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several unnecessary computations:
- Brain regions are all set to `'unknown'` making the `brain_regions` and `brain_region_idx` fields uninformative but still stored.
- The `discretize_three_bins` function is called to produce `wheel_bins_all` and `me_bins_all` but these results are never used; separate quantile computations are done instead.
- The variable `v` in `wheel_speed` is initialized as `np.zeros_like(pos)` but never used; only `mids` and `vv` are returned.

ii.
```python
brain_regions = ['unknown']
...
region_names.extend(['unknown'] * len(uniq))

# Unused variable
v = np.zeros_like(pos, dtype=np.float32)
```

iii. The brain region assignment is particularly notable because the reference code carefully maps clusters to Beryl atlas regions using `BrainRegions().acronym2acronym(neural_dict['cluster_regions'], mapping='Beryl')`.
