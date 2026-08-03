# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by directly scanning the filesystem under `data/one_cache` for session directories matching the pattern `*/Subjects/*/*/*` that contain an `alf` subdirectory. It reads `.npy` and `.pqt` files directly from these directories rather than using the ONE API's search and indexing system. No release tag is loaded; instead, any directory with a trials table is considered a candidate session.

ii.
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)
```

iii. The AI's CONVERSION_NOTES.md notes that it replaced ONE object loading with direct local `.npy`/`.pqt` reads for speed: "Replaced ONE object loading with direct local `.npy`/`.pqt` reads from session ALF directories, reducing sample conversion time from ~27 s to ~3 s for 2 sessions."

## 1-b. How are the data split into subjects?

i. The subject name is extracted from the filesystem path (the third-to-last component of the session directory path under `Subjects/`). Subjects are tracked via a dictionary mapping subject names to indices, built incrementally as sessions are processed.

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```
```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
```

iii. No explicit justification in CONVERSION_NOTES.md; the path structure is the standard IBL ONE cache layout.

## 1-c. How are the data split into sessions?

i. Each session directory found by the filesystem glob represents one session. Sessions are processed sequentially in sorted filesystem order.

ii.
```python
session_dirs = choose_sessions(list_session_dirs(data_root), mode)
```

iii. No explicit justification; sessions are the natural organizational unit in the IBL data.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. The AI reads this table and uses each row as a trial.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
```

iii. No explicit justification; the trials table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on three conditions: (1) `stimOn_times` must be finite, (2) choice must be valid (not 0/no-response), and (3) `probabilityLeft` must map to a valid prior value. Notably, the AI does NOT filter by reaction time bounds (no 80ms-2s filter), does NOT check for `firstMovement_times`, and does NOT check that wheel and camera data cover the trial window.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
```

iii. CONVERSION_NOTES.md does not explicitly discuss the omission of reaction time filtering or coverage checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Cluster quality is read from `clusters.metrics.pqt`. Brain region information comes from `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
st = np.load(st_p)  # spikes.times.npy
sc = np.load(sc_p).astype(int)  # spikes.clusters.npy
metrics = pd.read_parquet(metrics_p)  # clusters.metrics.pqt
```

iii. CONVERSION_NOTES.md Step 1 identifies `load_spiking_data` and `bin_spiking_data` as reference functions.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms bins over a -0.5s to +1.5s window (100 bins) aligned to stimulus onset. The AI uses `np.add.at` to count spikes per bin per neuron. However, the counts are NOT divided by the bin width, so the resulting data is in spike counts, not firing rates (Hz). When a session has multiple probes, their neurons are pooled with an offset.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
N_BINS = len(TIME_BINS)  # 100
```
```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
m = (bins >= 0) & (bins < N_BINS) & (cl >= 0) & (cl < n_neurons)
np.add.at(out, (cl[m], bins[m]), 1)
```

iii. CONVERSION_NOTES.md Step 5 states "Bin into 20 ms bins from -0.5 s to +1.5 s relative to stimulus onset" matching the reference parameters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` in the metrics table are kept, matching the reference's quality control criterion.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
```

iii. CONVERSION_NOTES.md Step 4 notes: "Neuron quality: `good_clusters` defined as `(clusters['label'] >= 1)`".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are extracted around stimulus onset using `np.searchsorted`, then binned relative to stimulus onset. The spike times are subtracted by `t0` (stimulus onset) before binning.

ii.
```python
for t0 in stim_on:
    edges = t0 + TIME_BINS
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
    ts = spike_times[lo:hi] - t0
    bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

iii. CONVERSION_NOTES.md Step 5 states alignment is to `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20ms bins, 100 bins per trial spanning -0.5s to +1.5s. No rebinning is applied; spikes are binned directly at this resolution.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
N_BINS = len(TIME_BINS)
```

iii. CONVERSION_NOTES.md confirms "20 ms bins for wheel decoding in reference analyses".

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin edges defined by the constants `T_START`, `T_END`, and `BINSIZE`. The bin centers are computed as `TIME_BINS + BINSIZE / 2`.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. No special justification needed; this is a constructed time axis.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The time input is the bin centers of the neural binning grid, repeated identically for every trial. No trial-specific processing is needed.

ii.
```python
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. N/A

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input IS the neural binning grid (bin centers), so alignment is inherent.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from `probabilityLeft` in the trials table. A change in `probabilityLeft` value marks a new block boundary.

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

iii. CONVERSION_NOTES.md Step 5 states: "Compute within-block trial index from consecutive equal-probability blocks".

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI detects block boundaries by comparing consecutive `probabilityLeft` values. The trial number within a block starts at 1 (not 0) and increments for each trial. This count is computed on ALL trials before any filtering, so the count reflects the animal's true position in the block. The value is broadcast across all time bins for each trial.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```
(Called before the `valid` filter is applied, using all trials.)

iii. No explicit justification for starting at 1 vs 0.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From the `choice` column in the trials table. IBL convention: +1 = left, -1 = right, 0 = no response.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. CONVERSION_NOTES.md Step 5 confirms: "Map left=0, right=1 using IBL coding".

## 5-b. What processing is involved in computing `output` *Choice*?

i. The choice is recoded from IBL convention (+1/-1) to decoder convention (left=0, right=1). No-response trials (choice=0) are filtered out during trial selection. The per-trial value is broadcast across all time bins.

ii.
```python
out_trial = np.vstack([
    np.full(N_BINS, choice[tr], dtype=np.int64),
    ...
])
```

iii. Matches the instruction specification: "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `probabilityLeft` in the trials table, which takes values 0.2, 0.5, and 0.8.

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

iii. Matches instruction specification: "0.2 -> 0, 0.5 -> 1, 0.8 -> 2".

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Simple recoding using `np.isclose` for floating point comparison. The per-trial value is broadcast across all time bins.

ii. Same as 6-a code snippet.

iii. N/A

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`, loaded directly from the filesystem.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    ...
    return np.load(tp), np.load(pp)
```

iii. CONVERSION_NOTES.md Step 5 identifies wheel position and timestamps as source variables.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI deduplicates timestamps, then computes velocity using `np.gradient(position, timestamps)` on the raw (irregularly sampled) data. This is critically different from the reference, which uses `SessionLoader.load_wheel()` that (1) interpolates position to an even 1000 Hz grid and (2) applies a 20 Hz Butterworth low-pass filter before differentiating. Furthermore, the AI does NOT take the absolute value of velocity, so the result is signed velocity, not speed.

ii.
```python
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
timestamps = uniq_t
position = position[uniq_idx]
vel = np.gradient(position, timestamps)
```
```python
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. CONVERSION_NOTES.md Step 6 mentions "wheel-speed interpolation" but does not discuss the lack of Butterworth filtering or absolute value.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI uses session-level tertile thresholds (1/3 and 2/3 quantiles) to discretize the wheel velocity into 3 bins (0, 1, 2). Quantiles are computed over all finite values across all trials in the session.

ii.
```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    ...
    y[x > q1] = 1
    y[x > q2] = 2
```

iii. CONVERSION_NOTES.md Step 5 states: "discretize into 3 bins per task spec".

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel velocity is interpolated onto the same bin centers (`TIME_CENTERS`) as the neural data, relative to stimulus onset for each trial.

ii.
```python
x = t0 + TIME_CENTERS
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. N/A

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` with their corresponding `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy`.

ii.
```python
def load_motion_energy(session_alf: Path):
    left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
    right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
    left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
    right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```

iii. CONVERSION_NOTES.md Step 5 identifies "leftCamera.ROIMotionEnergy.npy and rightCamera.ROIMotionEnergy.npy + camera times" as source variables.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads ALL available camera streams (both left and right if present) and averages them together. This differs from the reference, which picks only one camera (left preferred, right as fallback). The traces are interpolated onto the trial bin centers, then discretized into 3 bins using session-level tertiles.

ii.
```python
def interp_motion_energy(streams, stim_on):
    for t0 in stim_on:
        x = t0 + TIME_CENTERS
        ys = []
        for ts, me in streams:
            ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
        arr = np.stack(ys, axis=0)
        ...
        y = np.divide(summed, denom, ...)
```

iii. CONVERSION_NOTES.md Step 5 states: "Use available whisker ROI motion energy stream(s), align/interpolate to 20 ms bins, combine left/right sensibly if both available".

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same method as wheel speed: session-level tertile thresholds (1/3 and 2/3 quantiles) to discretize into 3 bins.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. Same justification as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The motion energy is interpolated onto the same bin centers (`TIME_CENTERS`) as the neural data, relative to stimulus onset for each trial.

ii.
```python
x = t0 + TIME_CENTERS
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. N/A

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions missing required columns (`stimOn_times`, `choice`, `probabilityLeft`), spikes, wheel, or whisker motion energy are skipped entirely. Sessions with fewer than 2 valid trials are skipped. Wheel timestamps are deduplicated. NaN values in interpolated wheel/whisker data are set to category 0 during discretization. If a cluster's label column is missing from the metrics table, all clusters are assumed good (label defaults to 1).

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
```
```python
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
```
```python
y[~np.isfinite(x)] = 0
```

iii. CONVERSION_NOTES.md Step 10 mentions duplicate timestamp handling and NaN handling for whisker data.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spikes data from disk (reading large `.npy` files for spike times and clusters per probe). Processing is sequential (no parallel processing), so the total time scales linearly with the number of sessions.

ii.
```python
st = np.load(st_p)  # spikes.times.npy - can be hundreds of MB
sc = np.load(sc_p).astype(int)  # spikes.clusters.npy
```

iii. CONVERSION_NOTES.md Step 7 notes the conversion takes ~0.7-1.2 s/session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over trials one at a time. The wheel and whisker interpolation loops also iterate per trial. These could potentially be vectorized by offsetting spike indices or building a single interpolation query vector.

ii.
```python
for t0 in stim_on:
    ...
    np.add.at(out, (cl[m], bins[m]), 1)
```
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. No explicit discussion in CONVERSION_NOTES.md about vectorization opportunities.

## 10-c. What processing does the code repeat multiple times?

i. No obviously repeated processing was identified. Each session is processed once sequentially.

ii. N/A

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores neural data as raw spike counts rather than firing rates. Depending on the downstream decoder, this may or may not matter (a linear scaling factor). Additionally, the AI loads and processes brain region IDs as numeric strings rather than atlas acronyms, which is less informative for downstream interpretation.

ii.
```python
np.add.at(out, (cl[m], bins[m]), 1)  # counts, not rates
```

iii. No explicit discussion of unnecessary processing.
