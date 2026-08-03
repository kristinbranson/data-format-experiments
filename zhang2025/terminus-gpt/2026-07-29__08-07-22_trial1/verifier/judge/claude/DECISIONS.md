# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads a sessions metadata table from `data/one_cache/Brainwidemap/sessions.pqt` and iterates over each row. For each session, it constructs a directory path from the lab, subject, date, and number fields, then loads trial tables, spikes, wheel data, and motion energy from files within that directory. It does NOT use the ONE API; instead it directly opens `.npy` and `.pqt` files from the ALF directory structure.

ii.
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
# ...
for i, (_, row) in enumerate(sessions.iterrows()):
    # ...
    loaded = load_session(row)
```

```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"
```

iii. The AI chose to bypass the ONE API and load files directly from the cache directory structure, constructing paths manually from session metadata fields.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column of the sessions table. As sessions are processed, subjects are accumulated into a list and a mapping dictionary. The `subject_idx` array maps each session to its subject index.

ii.
```python
subject = str(row['subject'])
# ...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The subject is taken directly from the sessions metadata table, which already contains the subject name per session.

## 1-c. How are the data split into sessions?

i. Each row in the sessions parquet table corresponds to one session. The code iterates over all rows (up to 480), processing each as a separate session.

ii.
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
```

iii. Sessions are already defined as rows in the metadata table.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. After loading, a mask is applied and remaining trials are iterated over to create per-trial neural, input, and output arrays.

ii.
```python
trials = load_trials(session_dir)
mask = trial_mask(trials)
trials = trials.loc[mask].reset_index(drop=True)
```

iii. The trials table already defines individual trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a minimal filter: it only checks that `stimOn_times`, `choice`, and `probabilityLeft` are finite (not NaN). It does NOT apply reaction time bounds (80ms to 2s), does NOT exclude no-choice trials (choice=0), and does NOT check that wheel and camera data span the trial window.

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

iii. The CONVERSION_NOTES.md does not discuss why the reaction time filter or coverage checks were omitted. The reference code applies `(reaction >= 0.08) & (reaction <= 2.0)`, excludes no-choice trials, checks that choice and probabilityLeft are valid values, and verifies wheel/camera temporal coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI loads `spikes.times.npy` and `spikes.clusters.npy` from each probe directory, along with `clusters.metrics.pqt` for quality filtering.

ii.
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
cm = base / 'clusters.metrics.pqt'
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. Same raw variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into time bins using `np.digitize` and `np.add.at`. The AI uses a window of [-0.2, 1.0] seconds (60 bins of 20ms). The result is stored as raw spike counts (NOT divided by bin width to convert to firing rate in Hz).

ii.
```python
T_START = -0.2
T_END = 1.0
BIN_SIZE_S = 0.02

def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    # ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The AI chose a [-0.2, 1.0] window based on its interpretation of the task, but the reference code uses [-0.5, 1.5]. The AI also stores spike counts rather than firing rates (Hz).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters clusters using `clusters.metrics.pqt`, keeping only those with `label == 1`. If no metrics file exists, all clusters are kept.

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

iii. The `label == 1` criterion matches the reference's `label >= 1` (equivalent since max is 1). The fallback to all clusters when no metrics file exists is a reasonable edge case handler. The total neuron count (75,708) matches the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are binned relative to the stimulus onset time (`stimOn_times`), by adding it to the bin edges.

ii.
```python
edges = stim_time + build_time_edges()
```

iii. This correctly aligns to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms, but the window is [-0.2, 1.0] = 1.2s, producing 60 bins. The reference uses [-0.5, 1.5] = 2.0s, producing 100 bins. No rebinning or interpolation is applied.

ii.
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0
```

iii. The 20ms bin size matches the reference, but the window differs significantly.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin centers of the time grid, computed from `T_START`, `T_END`, and `BIN_SIZE_S`.

ii.
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. Same approach as the reference, but with different window bounds.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing bin centers. The time values are the same for every trial.

ii.
```python
centers = build_time_centers().astype(np.float32)
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. Straightforward definition, same approach as reference.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same bin edges are used for both neural spike binning and the time input, so they are inherently aligned.

ii.
```python
edges = stim_time + build_time_edges()  # for neural
centers = build_time_centers()           # for input
```

iii. Correctly aligned by construction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. Block boundaries are detected where `probabilityLeft` changes between consecutive trials.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
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

iii. The same source variable as the reference.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts the position within each block starting from 1 (1-indexed). The reference uses `cumcount()` which starts from 0. Additionally, the AI computes this AFTER applying the trial mask, whereas the reference computes it BEFORE filtering, so the trial counts reflect the animal's true position in the block.

ii.
```python
prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
trial_in_block = trial_number_in_block(prob_left)
```

Note that `trials` at this point has already been filtered by `trial_mask`, so dropped trials shift the count.

iii. The 1-indexing vs 0-indexing is a minor difference. Computing after filtering is more concerning as it changes the trial count values.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table.

ii.
```python
def map_choice(v):
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

iii. Same source variable as reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Maps +1 (left) to 0 and -1 (right) to 1. NaN values from choice=0 (no response) are propagated and later filtered.

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. Same mapping as the reference (`{1.0: 0, -1.0: 1}`). However, no-response trials (choice=0) are not explicitly excluded in the trial mask - they become NaN and are filtered later in the per-trial loop.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From the `probabilityLeft` column of the trials table.

ii.
```python
def map_prior(v):
    if np.isclose(v, 0.2): return 0
    if np.isclose(v, 0.5): return 1
    if np.isclose(v, 0.8): return 2
    return np.nan
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Maps 0.2 to 0, 0.5 to 1, 0.8 to 2 using `np.isclose` for floating point comparison.

ii.
```python
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. Same mapping as reference. Using `np.isclose` is slightly more robust than exact equality.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy` loaded directly from the ALF directory.

ii.
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    return np.load(ts), np.load(pos)
```

iii. Same raw source files as what `SessionLoader.load_wheel()` reads internally.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes velocity as `np.abs(dp/dt)` using raw `np.diff` on position and timestamps. This does NOT apply the 1000 Hz interpolation or 20 Hz Butterworth low-pass filter that `SessionLoader` applies internally. The velocity midpoints and values are computed, then binned into trial windows using `bin_signal` (which averages samples falling into each bin), and finally discretized into 3 categories using session-wide tercile thresholds.

ii.
```python
def wheel_speed(ts, pos):
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    v = np.zeros_like(pos, dtype=np.float32)
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv
```

```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. The reference uses `SessionLoader.load_wheel()` which interpolates to 1000 Hz and applies a Butterworth filter before computing velocity. The AI's raw diff approach produces noisier velocity estimates. Also, the AI uses bin averaging (`bin_signal`) instead of linear interpolation (`np.interp`).

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-wide tercile thresholds are computed using `np.quantile([1/3, 2/3])` on all valid wheel speed values across all trials. Then `np.digitize` is used to assign categories 0, 1, 2.

ii.
```python
def discretize_three_bins(x):
    q1, q2 = np.quantile(x[valid], [1/3, 2/3])
    out[valid] = np.digitize(x[valid], [q1, q2], right=False).astype(np.int64)
    return out
```

```python
wq = np.quantile(all_wheel_cont, [1/3, 2/3])
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] ...), wq, right=False)
```

iii. Similar to the reference's `np.percentile(trace, [33.33, 66.67])` approach, but the AI first collects all valid values, computes thresholds, then applies them per-trial with NaN-filling. The reference applies percentiles to the full 2D trial-by-time array at once.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is binned at the same time centers as the neural data (relative to stimulus onset) using `bin_signal`, which averages wheel speed samples falling into each bin.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. The reference uses `np.interp` (linear interpolation) at bin centers, while the AI uses bin averaging. Both use the same stimulus-onset-aligned time grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI loads `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` from the ALF directory. If both exist, it averages them.

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

iii. The reference prefers the left camera and only uses the right as fallback. The AI averages both when available, which is a different approach.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded and (if both cameras exist) averaged. Timestamps are NOT loaded from the camera times files; instead they are guessed from camera features files or synthesized from session timing. The trace is then binned at trial time centers using `bin_signal` (averaging) and discretized into 3 categories.

ii.
```python
def guess_motion_timestamps(session_dir, n):
    alf = session_dir / 'alf'
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        # try to extract timestamps from features
        ...
    # fallback: synthesize from session interval
    trials = load_trials(session_dir)
    t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
    t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
    return np.linspace(t0, t1, n, dtype=np.float32)
```

iii. The reference uses `SessionLoader.load_motion_energy()` which properly loads camera timestamps from `_ibl_<side>Camera.times.npy`. The AI's timestamp guessing is a significant issue that could lead to temporal misalignment of whisker data with neural data.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: session-wide tercile thresholds computed using `np.quantile([1/3, 2/3])`, then `np.digitize` assigns categories 0, 1, 2.

ii.
```python
mq = np.quantile(all_me_cont, [1/3, 2/3])
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] ...), mq, right=False)
```

iii. Similar to reference approach but with NaN handling differences.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The motion energy is binned at stimulus-onset-aligned time centers using `bin_signal`. However, since the timestamps are guessed rather than properly loaded, the alignment may be incorrect.

ii.
```python
ms = bin_signal(me_ts, me, st + centers)
```

iii. The reference uses proper camera timestamps loaded via `SessionLoader`, ensuring accurate alignment. The AI's guessed timestamps could introduce temporal misalignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions missing required data files (trials, spikes) are skipped. Sessions with fewer than 2 valid trials are skipped. NaN values in choice or prior cause individual trials to be skipped in the per-trial output construction loop. Missing wheel or motion energy data produces NaN-filled arrays that are later filled with threshold values during discretization.

ii.
```python
if loaded is None:
    log('  skipped: missing required data')
    continue
```

```python
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] ...), wq, right=False)
```

iii. The NaN-to-number filling for wheel/whisker data means that missing behavioral data is assigned to the lowest bin rather than being excluded, which differs from the reference's approach of requiring data coverage.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (reading large `.npy` files) and the per-trial spike binning loop are the most time-consuming. The full conversion processes 480 sessions sequentially, taking several hours.

ii.
```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The conversion runs sequentially (single-threaded), unlike the reference which uses `ProcessPoolExecutor` with 10 workers.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike cluster remapping loop uses a Python dict comprehension and list iteration instead of vectorized numpy operations:

ii.
```python
remap = {cid: i + offset for i, cid in enumerate(uniq)}
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
```

Also the `bin_signal` function loops over each bin:
```python
for i in range(len(centers)):
    m = idx == i
    if np.any(m):
        out[i] = np.nanmean(values[m])
```

iii. The reference avoids these loops by using `np.cumsum` for renumbering and `np.interp` for signal resampling.

## 10-c. What processing does the code repeat multiple times?

i. The `build_time_edges()` function is called multiple times per trial (once in `bin_spikes_for_trial` and once in `bin_signal`), recomputing the same array each time.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()  # called per trial
```

iii. A minor inefficiency; the edges could be computed once and reused.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes both left and right camera motion energy when both are available, averaging them, even though only one camera's data is needed. The `guess_motion_timestamps` function may load and parse camera features files unnecessarily.

ii.
```python
for p in [left_p, right_p]:
    if p is not None and p.exists():
        vals.append(np.load(p).astype(np.float32))
arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
```

iii. The reference only loads a single camera (left preferred, right fallback), avoiding unnecessary I/O and averaging.
