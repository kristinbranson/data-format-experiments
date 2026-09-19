# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the sessions parquet table from `data/one_cache/Brainwidemap/sessions.pqt` directly (not using the ONE API). It iterates over all rows in the sessions table, constructs the session directory path from the row's `lab`, `subject`, `date`, and `number` fields, and loads files by navigating the filesystem directly. It reads trial tables, spike data, wheel data, and motion energy by opening `.npy` and `.pqt` files from the ALF directory structure.

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

iii. The AI chose to load data directly from the filesystem rather than using the ONE API. The CONVERSION_NOTES.md notes the data structure is an IBL ONE cache but does not explicitly justify bypassing the API. This approach works but loses the dataset validation and revision resolution that the ONE API provides.

## 1-b. How are the data split into subjects?

i. The subject name comes from the `subject` column in the sessions parquet table. Subjects are accumulated in a dictionary mapping subject names to indices as sessions are processed.

ii.
```python
subj = str(row['subject'])
# ...
if subj not in subject_map:
    subject_map[subj] = len(subjects)
    subjects.append(subj)
subject_idx.append(subject_map[subj])
```

iii. The subject splitting is straightforward and derived from the sessions table metadata. No explicit justification given.

## 1-c. How are the data split into sessions?

i. Each row in the sessions parquet table corresponds to one session. The AI iterates over these rows sequentially.

ii.
```python
sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
for i, (_, row) in enumerate(sessions.iterrows()):
    loaded = load_session(row)
```

iii. Sessions are already one row per entry in the table. No decision needed.

## 1-d. How are the data split into trials?

i. Each row in the trials parquet table corresponds to one trial. The AI loads the trial table and iterates over trials to construct per-trial arrays.

ii.
```python
trials = pd.read_parquet(p)
# ...
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(...))
```

iii. No decision needed; the trial table already defines trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: it checks that `stimOn_times`, `choice`, and `probabilityLeft` are finite. It does NOT apply reaction time bounds (80ms to 2s), does NOT exclude no-choice trials (choice==0), and does NOT check for wheel/camera coverage of the trial window.

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

A second filter later drops trials where choice or prior are NaN:
```python
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
```

iii. The CONVERSION_NOTES.md mentions "Apply reference trial mask and additionally drop trials missing requested outputs" in Step 4, but the actual implementation does not apply the reference trial mask (reaction time bounds, no-choice exclusion). The code only checks for finite values.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` loaded directly from the probe directories in the ALF structure.

ii.
```python
st = base / 'spikes.times.npy'
sc = base / 'spikes.clusters.npy'
spikes_t = np.load(st)
spikes_c = np.load(sc)
```

iii. Same raw variables as the reference. Direct file loading rather than API-mediated loading.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into time bins aligned to stimulus onset using `np.digitize`. The bin edges are computed from `T_START + build_time_edges()`. Spike counts are accumulated per neuron per bin using `np.add.at`. The result is raw spike counts (NOT converted to firing rates by dividing by bin width). Multiple probes are merged by offsetting cluster indices.

ii.
```python
def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    # ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The AI does not convert spike counts to firing rates (Hz). The reference code divides by bin size (`/ BIN`). The AI's CONVERSION_NOTES mention "bin spike counts" but the code leaves them as counts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters clusters by reading `clusters.metrics.pqt` and keeping only clusters where `label == 1`. If the metrics file doesn't exist, ALL clusters are kept. The AI does NOT filter by brain region (void/root) since it doesn't use the brain atlas.

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

iii. The AI applies the label==1 filter from the metrics file, which is the same QC threshold as the reference. However, it falls back to keeping all clusters if no metrics file exists, and it does not exclude `void` brain regions (units placed outside the brain by histology).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to stimulus onset (`stimOn_times`). Bin edges are computed as `stim_time + build_time_edges()`.

ii.
```python
edges = stim_time + build_time_edges()
```

iii. Correctly aligns to stimulus onset as required by the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 20ms (`BIN_SIZE_S = 0.02`). The time window is [-0.2, 1.0] seconds, yielding 60 time bins. No rebinning is applied.

ii.
```python
BIN_SIZE_S = 0.02
T_START = -0.2
T_END = 1.0

def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

iii. The 20ms bin size matches the reference papers. However, the time window [-0.2, 1.0] differs from the reference code's [-0.5, 1.5]. The reference code uses a 2-second window producing 100 time bins; the AI uses a 1.2-second window producing 60 time bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table, used as the alignment event. The time input is the array of bin centers.

ii.
```python
centers = build_time_centers().astype(np.float32)
# ...
input_trials.append(np.vstack([centers, np.full_like(centers, trial_in_block[i])]).astype(np.float32))
```

iii. Correctly derives the time input from the bin centers relative to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing beyond computing the bin centers from the time edges. The centers are computed as `(edges[:-1] + edges[1:]) / 2`.

ii.
```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2
```

iii. Straightforward computation, same as reference.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input uses the same bin centers as the neural binning, so they are aligned by construction.

ii.
```python
centers = build_time_centers().astype(np.float32)
# neural uses: edges = stim_time + build_time_edges()
# input uses: centers (same grid)
```

iii. Alignment is correct by construction.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table, where changes in `probabilityLeft` indicate block transitions.

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

iii. Same approach as reference: detecting block boundaries from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counts trials within each block starting from 1 (the first trial in a block gets value 1). The reference counts from 0 (using `cumcount()` which is 0-indexed). The count is computed on all trials before any quality filtering is applied.

ii.
```python
run = 1
out[0] = 1
for i in range(1, len(prob_left)):
    if prob_left[i] == prob_left[i - 1]:
        run += 1
    else:
        run = 1
    out[i] = run
```

iii. The AI starts counting from 1 while the reference starts from 0 (`cumcount()` returns 0-based indices). Both compute the count before trial filtering so that the block position reflects the animal's actual position.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column of the trials table, where IBL convention is +1 for left and -1 for right.

ii.
```python
def map_choice(v):
    return 0 if v == 1 else 1 if v == -1 else np.nan
```

iii. Correctly maps +1 (left) to 0 and -1 (right) to 1, matching the reference.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Simple remapping of +1 to 0 (left) and -1 to 1 (right). Values that are neither +1 nor -1 are mapped to NaN and the trial is later excluded.

ii.
```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. Same mapping as reference.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table.

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

iii. Same source variable as reference.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Maps 0.2 to 0, 0.5 to 1, 0.8 to 2, matching the instructions. Uses `np.isclose` for floating-point comparison.

ii.
```python
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. Same mapping as reference and as specified in the instructions.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`, loaded directly from the filesystem.

ii.
```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    return np.load(ts), np.load(pos)
```

iii. Uses the raw wheel files rather than the `SessionLoader` which applies interpolation and filtering.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel velocity by simple finite differences of position divided by time intervals (`dp/dt`), takes absolute value for speed. No interpolation to a regular grid and no Butterworth low-pass filter are applied. This differs from the reference which uses `SessionLoader.load_wheel()` which interpolates to 1000 Hz and applies a 20 Hz Butterworth filter.

The speed is then binned into the trial time bins using `bin_signal()` (averaging within bins), and discretized into 3 categories using session-wide 1/3 and 2/3 quantiles.

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

iii. The CONVERSION_NOTES mention following "the same general IBL processing style" but the code does not use the IBL `SessionLoader` for wheel processing, which applies interpolation and Butterworth filtering. The raw differentiation approach will produce a noisier, differently-valued speed signal.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Discretized into 3 bins using session-wide 1/3 and 2/3 quantiles of the finite (non-NaN) values.

ii.
```python
def discretize_three_bins(x):
    q1, q2 = np.quantile(x[valid], [1/3, 2/3])
    out[valid] = np.digitize(x[valid], [q1, q2], right=False).astype(np.int64)
    return out
```

iii. Same approach as reference (session-wide percentile-based thresholding), though the underlying speed values differ due to different processing.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is binned into the same time bins as neural data using `bin_signal()`, which assigns speed samples to bins and averages them within each bin.

ii.
```python
ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

```python
def bin_signal(ts, values, centers):
    edges = np.concatenate([[centers[0] - BIN_SIZE_S / 2], centers + BIN_SIZE_S / 2])
    idx = np.digitize(ts, edges) - 1
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])
    return out
```

iii. Uses bin-averaging rather than interpolation to the bin centers (as the reference does with `np.interp`). This is a valid approach but differs from the reference.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy`. The AI loads BOTH cameras when available and averages them, rather than preferring the left camera as the reference does.

ii.
```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    for p in [left_p, right_p]:
        if p is not None and p.exists():
            vals.append(np.load(p).astype(np.float32))
    n = min(len(v) for v in vals)
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr
```

iii. The AI averages both cameras instead of preferring the left camera. This produces a different signal from the reference.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The raw motion energy values are loaded and (if both cameras available) averaged. They are then binned into trial time bins using `bin_signal()` and discretized into 3 categories using session-wide quantiles.

The timestamps for the motion energy are NOT loaded from the actual camera timestamps file. Instead, the AI uses `guess_motion_timestamps()` which attempts to find timestamps from camera features files, or falls back to generating evenly-spaced timestamps spanning the session.

ii.
```python
me = load_motion_energy(session_dir)
me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None

def guess_motion_timestamps(session_dir, n):
    feats = sorted(alf.glob('_ibl_*Camera.features.pqt'))
    if feats:
        # try to extract timestamps from features file
        ...
    trials = load_trials(session_dir)
    if trials is not None:
        t0 = np.nanmin(trials['stimOn_times'].to_numpy()) + T_START
        t1 = np.nanmax(trials['stimOn_times'].to_numpy()) + T_END
        return np.linspace(t0, t1, n, dtype=np.float32)
    return np.arange(n, dtype=np.float32) * BIN_SIZE_S
```

iii. The reference loads actual camera timestamps (`_ibl_leftCamera.times.npy`). The AI's `guess_motion_timestamps` is a major concern as it fabricates approximate timestamps rather than using the actual camera timestamps that exist in the data.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: session-wide 1/3 and 2/3 quantiles.

ii.
```python
# Same discretize_three_bins function as wheel
mq = np.quantile(all_me_cont, [1/3, 2/3])
mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False)
```

iii. Same percentile-based approach as reference.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Binned into the same time bins as neural data using `bin_signal()`. However, the timestamps used for alignment are fabricated by `guess_motion_timestamps` rather than actual camera timestamps, making the alignment unreliable.

ii.
```python
ms = bin_signal(me_ts, me, st + centers)
```

iii. The use of fabricated timestamps means the temporal alignment of whisker motion energy with neural data is approximate at best and potentially wrong.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with missing required data (no trial table, no spikes) are skipped. Sessions with fewer than 2 valid trials are skipped. NaN values in wheel/whisker data are handled by `nan_to_num` during discretization. Trials with NaN choice or prior are excluded in a second pass.

ii.
```python
if loaded is None:
    log('  skipped: missing required data')
    continue
```

```python
if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
    continue
```

```python
wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0]), wq, right=False)
```

iii. Basic missing data handling. The `nan_to_num` approach for wheel/whisker data replaces NaN with the lower quantile threshold, effectively assigning missing data to the "low" category, which introduces bias.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and processing spike data for each session, including reading large `.npy` files and binning spikes into trial matrices. The per-trial loop for spike binning (`bin_spikes_for_trial`) is also expensive. The conversion log shows ~30-80 seconds per session.

ii.
```python
spikes_t = np.load(st)
spikes_c = np.load(sc)
# ...
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, st))
```

iii. The AI's CONVERSION_NOTES note "~35-75 s/session" and "Full run likely several hours without optimization."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop could be vectorized by offsetting spike indices across trials. The `bin_signal` function loops over all time bins individually, which could be vectorized. The `map_choice` and `map_prior` functions use Python loops over individual values instead of vectorized operations.

ii.
```python
for i, st in enumerate(stim_times):
    neural_trials.append(bin_spikes_for_trial(...))
```

```python
choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

```python
def bin_signal(ts, values, centers):
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])
```

iii. The AI identified in CONVERSION_NOTES that "Full run likely several hours without optimization" but did not implement vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The `build_time_edges()` and `build_time_centers()` functions are called repeatedly (once per trial for binning, once per session for inputs) instead of being computed once. The trial filtering is effectively done twice: once with `trial_mask()` and a second time when checking `np.isfinite(choice[i])` during output construction.

ii.
```python
# Called in bin_spikes_for_trial for every trial:
edges = stim_time + build_time_edges()

# Called per session:
centers = build_time_centers().astype(np.float32)
```

iii. Not addressed in CONVERSION_NOTES.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code remaps cluster IDs with a Python loop (`[remap[c] for c in spikes_c]`) creating new mappings for each probe, even though the cluster-to-region mapping is discarded (all regions are labeled 'unknown'). The code also loads both left and right camera motion energy and averages them, even though only one camera's data is needed per the reference methodology.

ii.
```python
remap = {cid: i + offset for i, cid in enumerate(uniq)}
spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
# ...
region_names.extend(['unknown'] * len(uniq))
```

iii. Brain region information is discarded (all set to 'unknown'), making the atlas-related processing unnecessary.
