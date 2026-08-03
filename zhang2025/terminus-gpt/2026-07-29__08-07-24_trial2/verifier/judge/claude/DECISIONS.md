# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing the filesystem for `_ibl_trials.table.pqt` files under `data/one_cache`, then walking up the directory tree to find `alf` directories and their parent session directories. It does NOT use the ONE API or the Brainwidemap release tag. Each data type (trials, wheel, spikes, motion energy) is loaded by directly opening `.npy` and `.pqt` files from the discovered session paths.

ii.
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    sessions = []
    for p in trial_tables:
        sess = p.parent
        while sess.name != 'alf' and sess != sess.parent:
            sess = sess.parent
        if sess.name == 'alf':
            sessions.append(sess.parent)
    out = sorted(set(sessions))
    return out
```

iii. The AI's CONVERSION_NOTES.md notes that data are organized as an IBL ONE cache, and it chose to glob for trial tables rather than using the ONE API. The trajectory shows no use of `one.search()` or `one.load_cache(tag=...)`.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject name from the session directory path, specifically `session_path.parts[-3]` (the subject directory in the IBL path structure `lab/Subjects/subject/date/number`).

ii.
```python
subject = session_path.parts[-3]
```

iii. The AI relies on the filesystem path convention to identify subjects. This is functionally equivalent to using the ONE API subject field, since the paths follow IBL conventions.

## 1-c. How are the data split into sessions?

i. The AI discovers sessions by finding all directories containing `_ibl_trials.table.pqt` files. Each such directory is treated as a session. There is no filtering by the Brainwidemap release tag.

ii.
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    ...
```

iii. The AI notes in CONVERSION_NOTES.md that sessions are identified from the file structure. No release-tag filtering is applied.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI reads this table and uses each row as a trial.

ii.
```python
def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    ...
    return pd.read_parquet(trial_file)
```

iii. No special decision needed; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) `stimOn_times` must not be NaN, (2) `choice` must be in `{-1, 1}` (drops no-response trials), and (3) `probabilityLeft` must not be NaN. After spike binning, it additionally drops trials where the binned neural activity is entirely zero. Critically, the AI does NOT filter on reaction time bounds (0.08-2.0s) and does NOT check wheel/camera temporal coverage of the trial window.

ii.
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. The CONVERSION_NOTES.md mentions filtering choice==0 trials and all-zero neural trials but does not discuss reaction time filtering or wheel/camera coverage checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Same as reference: `spikes.times.npy` and `spikes.clusters.npy` per probe, loaded from the filesystem.

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. Straightforward loading of spike sorting arrays.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins, but over a **-0.2 to 1.0s window** (60 bins) rather than the reference's -0.5 to 1.5s window (100 bins). The spike counts are kept as raw counts and are NOT divided by the bin width to convert to firing rates (Hz). The reference converts to Hz by dividing by BIN (0.02).

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    ...
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
    ...
    return trial_mats, edges
```

iii. CONVERSION_NOTES.md says "Bin spike counts in fixed stimulus-aligned time bins" but does not discuss the choice of window or whether to convert to firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters clusters using `label >= 1` from `clusters.metrics.pqt`, similar to the reference. However, it also applies an additional `noise_cutoff < 20` filter when the column is available. The reference only uses `label >= 1`.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
```

iii. CONVERSION_NOTES.md mentions the paper's criteria of "amplitude > 50 uV, noise cut-off < 20 uV, and refractory period violation criterion" and states the code implements paper-compatible filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to stimulus onset (`stimOn_times`) by subtracting the onset time. This matches the reference approach.

ii.
```python
rel = spike_times - s  # s is stimOn_times for that trial
```

iii. Consistent with the instruction to align to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, matching the reference. However, the time window is -0.2 to 1.0s (60 bins) instead of the reference's -0.5 to 1.5s (100 bins). No rebinning or smoothing is applied.

ii.
```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

iii. CONVERSION_NOTES.md does not explicitly justify the -0.2 to 1.0s window choice. The reference code and papers use -0.5 to 1.5s.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table, same as reference. The bin centers of the trial time grid serve as the time-since-stimulus-onset values.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. Same approach as reference.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The bin centers are computed from the edges of the spike binning grid. Due to the different time window (-0.2 to 1.0s vs reference -0.5 to 1.5s), the time values span a different range.

ii.
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. No special processing beyond computing bin centers.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is the bin centers of the same grid used to bin spikes, so they are inherently aligned.

ii.
```python
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. Same approach as reference.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table, same as reference. Block boundaries are detected where `probabilityLeft` changes value.

ii.
```python
def compute_trial_number_in_block(prob_left):
    prob_left = np.asarray(prob_left)
    out = np.zeros(len(prob_left), dtype=np.float32)
    c = 0
    prev = None
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
        else:
            c += 1
        out[i] = c
    return out
```

iii. Same source variable as reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trial position within a block starting from **1** (the first trial in a block is 1). The reference starts from **0** (uses pandas `cumcount()` which is 0-indexed). Also, the AI computes trial_in_block AFTER filtering trials (on the filtered trials table), whereas the reference computes it BEFORE filtering to preserve the animal's true block position.

ii.
```python
# AI code - counts from 1, computed after filtering
def compute_trial_number_in_block(prob_left):
    ...
    if i == 0 or v != prev:
        c = 1  # starts at 1
        prev = v
    else:
        c += 1
    out[i] = c
    return out

# Called after filtering:
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

vs reference:
```python
# Reference - counts from 0, computed before filtering
block = (trials.probabilityLeft != trials.probabilityLeft.shift()).cumsum()
variables = trial_variables(trials)  # before filtering
...
variables = variables[keep]  # then filtered
```

iii. CONVERSION_NOTES.md does not discuss the 0-vs-1 indexing or the order of filtering vs computation.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, same as reference.

ii.
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. Same source as reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Maps left choice (1) to 0 and right choice (-1) to 1, matching the reference mapping. No-response trials (choice=0) are filtered out.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out
```

iii. Same mapping as reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table, same as reference.

ii.
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Maps 0.2 to 0, 0.5 to 1, 0.8 to 2, matching the reference and instructions.

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. Same mapping as reference.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, loaded directly from files. The reference uses `SessionLoader.load_wheel()` which provides interpolated position and filtered velocity.

ii.
```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    ...
    return np.load(posf), np.load(tsf)
```

iii. Same raw data source, but different loading mechanism.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel speed by raw finite differencing (`dp/dt`) of the wheel position after sorting timestamps and removing duplicates. It does NOT apply the IBL's standard 1000 Hz interpolation and 20 Hz Butterworth low-pass filter that `SessionLoader.load_wheel()` applies. The reference uses the filtered velocity from SessionLoader and takes the absolute value.

ii.
```python
order = np.argsort(wheel_ts)
wheel_ts = np.asarray(wheel_ts)[order]
wheel_pos = np.asarray(wheel_pos)[order]
uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
wheel_ts = wheel_ts[uniq_mask]
wheel_pos = wheel_pos[uniq_mask]
if len(wheel_ts) >= 2:
    dt = np.diff(wheel_ts)
    dp = np.diff(wheel_pos)
    speed_mid = np.abs(dp / dt).astype(np.float32)
    ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
    wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, ...)
```

iii. CONVERSION_NOTES.md mentions "robust wheel-speed computation" with sorted timestamps and duplicate removal but does not discuss the missing Butterworth filter or 1000 Hz interpolation step.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes tertile thresholds by concatenating all trial traces across the session and computing the 1/3 and 2/3 quantiles. The reference computes percentiles (33.3%, 66.7%) of the session's trace (all trials stacked). Both result in 3 equal-sized bins.

ii.
```python
def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    finite = np.isfinite(x)
    ...
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)

def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
```

iii. The approach is conceptually similar to the reference. The AI uses strict `>` for thresholding while the reference uses `np.digitize` which uses `<` for the default right=False. The result is that values exactly at the threshold are assigned differently, but this is a minor difference.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same bin centers as the neural data using `np.interp`, same approach as reference.

ii.
```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    ...
    y = np.interp(t, timestamps, values)
```

iii. Same alignment approach as reference.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` with corresponding camera times. Both sides are loaded if available.

ii.
```python
def load_motion_energy(session_path: Path):
    alf = session_path / 'alf'
    left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
    right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
    left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
    right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
    streams = []
    if left_me and left_t:
        streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
    if right_me and right_t:
        streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
    return streams
```

iii. Same raw data source.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads all available camera sides and, if both left and right are available, **averages** them. The reference prefers the left camera and only falls back to the right if left is unavailable (never averages). The motion energy is then interpolated to bin centers and discretized into 3 bins.

ii.
```python
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. CONVERSION_NOTES.md mentions "choose available side or combining sides sensibly" as a planned decision.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same tertile approach as wheel speed: concatenate all trial traces, compute 1/3 and 2/3 quantiles, discretize with `>` thresholds.

ii.
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
discretize_with_thresholds(me_trials[i], me_q1, me_q2)
```

iii. Same as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated to the same bin centers as neural data using `np.interp`, same approach as reference.

ii.
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. Same alignment approach as reference.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data in several ways: (1) sessions without trial tables are skipped, (2) trials with NaN `stimOn_times` are dropped, (3) probes without `clusters.metrics.pqt` are skipped, (4) sessions with fewer than 2 valid trials are skipped, (5) trials with all-zero neural activity are dropped. However, it does NOT check for wheel or camera temporal coverage of the trial window (the reference does this).

ii.
```python
if trials is None or len(trials) < 2:
    return None
...
if spike_data[0] is None:
    return None
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. CONVERSION_NOTES.md describes handling of all-zero neural trials and wheel timestamp issues but does not discuss wheel/camera temporal coverage checks.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies session processing (loading spike data from disk) as the most expensive step. The code runs sequentially (no parallel processing), processing one session at a time. The reference uses `ProcessPoolExecutor` for parallel processing.

ii.
```python
for i, sess in enumerate(sessions, 1):
    st = time.time()
    try:
        p = process_session(sess, show_processing=args.show_processing)
    except Exception as e:
        ...
```

iii. CONVERSION_NOTES.md does not note the lack of parallel processing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI has per-trial loops for spike binning (`bin_spikes_for_trials`), wheel speed interpolation (`interp_to_trial_bins`), and motion energy interpolation. Additionally, `compute_trial_number_in_block` uses an explicit Python loop over all trials. The `map_prior` function also uses a Python list comprehension instead of vectorized operations. The cluster remapping `[remap[c] for c in sc]` is also a slow Python loop.

ii.
```python
# Per-trial spike binning loop
for s in stim_on:
    rel = spike_times - s
    ...

# Trial-number loop
for i, v in enumerate(prob_left):
    if i == 0 or v != prev:
        c = 1
    ...

# Cluster remapping loop
sc = np.array([remap[c] for c in sc], dtype=np.int64)
```

iii. CONVERSION_NOTES.md does not discuss vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. The AI re-sorts and deduplicates wheel timestamps for every session, which could be done once. The AI also loads and parses the trial table in `load_trials` and then re-accesses the same columns multiple times in `process_session`. No major repeated computation is evident.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads motion energy for both left and right cameras even when only one is needed. When both are available, it averages them, which is unnecessary work (the reference just picks one). The `map_choice` function creates a full array initialized to -1 and then overwrites, which is minor but unnecessary.

ii.
```python
# Loads both sides
if left_me and left_t:
    streams.append(...)
if right_me and right_t:
    streams.append(...)
```

iii. CONVERSION_NOTES.md does not discuss this.
