# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds sessions by globbing the filesystem for `_ibl_trials.table.pqt` files under `data/one_cache`, then walking up the directory tree to find session directories. It does not use the ONE API or the Brainwidemap release index. It reads trial tables directly with `pd.read_parquet()`, spike data with `np.load()`, wheel data with `np.load()`, and motion energy with `np.load()`. There is no filtering by a `DATALIMIT_SUBSET.csv` file and no use of release tags.

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

iii. The CONVERSION_NOTES.md does not provide a detailed justification. The AI simply globbed for trial tables and inferred session paths from the directory structure, bypassing the ONE API entirely.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the directory path structure. The AI takes `session_path.parts[-3]` as the subject name, relying on the IBL ONE cache directory convention of `<lab>/Subjects/<subject>/<date>/<number>`.

ii.
```python
subject = session_path.parts[-3]
```

iii. No explicit justification given. This is a reasonable inference from the filesystem layout.

## 1-c. How are the data split into sessions?

i. Each directory found by `find_sessions` is treated as one session. The session is determined by the filesystem path containing an `alf` directory with a `_ibl_trials.table.pqt` file.

ii.
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    ...
```

iii. No explicit justification. The ONE cache is naturally organized by session directory.

## 1-d. How are the data split into trials?

i. The trials table (`_ibl_trials.table.pqt`) has one row per trial. Each row is treated as a separate trial.

ii.
```python
trials = load_trials(session_path)
# ...
trials = pd.read_parquet(trial_file)
```

iii. No explicit justification needed; the trials table naturally defines trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials that have valid `stimOn_times`, a valid choice (in {-1, 1}), and non-null `probabilityLeft`. Additionally, trials where the binned neural activity is entirely zero are dropped. There is **no filtering based on reaction time bounds** (80 ms to 2 s), and **no check that wheel and camera timestamps cover the trial window**.

ii.
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
# ...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. The CONVERSION_NOTES.md mentions filtering `choice in {-1, 1}` and removing all-zero neural trials. It does not mention reaction time filtering or coverage checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `spikes.times.npy` and `spikes.clusters.npy` per probe, along with `clusters.metrics.pqt` for quality filtering and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` for region assignment.

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. The CONVERSION_NOTES.md documents that spikes.times and spikes.clusters are the source neural variables.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into time bins around stimulus onset using `np.add.at`. The result is **spike counts** (not firing rates -- no division by bin width). Multiple probes within a session are merged by renumbering clusters with an offset.

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    trial_mats = []
    for s in stim_on:
        rel = spike_times - s
        mask = (rel >= t0) & (rel < t1)
        rel = rel[mask]
        clu = spike_clusters[mask]
        mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
        if len(rel):
            tb = np.floor((rel - t0) / bin_size).astype(int)
            good = (tb >= 0) & (tb < mat.shape[1]) & (clu >= 0) & (clu < n_neurons)
            np.add.at(mat, (clu[good], tb[good]), 1)
        trial_mats.append(mat)
    return trial_mats, edges
```

iii. The CONVERSION_NOTES.md mentions "binned spike counts" as the neural representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are filtered using `clusters.metrics.pqt`. If the `noise_cutoff` column exists, clusters with `noise_cutoff >= 20` are removed. If the `label` column exists, clusters with `label < 1` are removed. The AI does **not** filter by brain region (no `void` exclusion). Brain regions are labeled using raw CCF IDs (e.g. `ccf_1020`) rather than Beryl atlas acronyms.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
```

iii. The CONVERSION_NOTES.md mentions filtering based on available cluster metrics and mentions the paper's criteria (amplitude > 50 uV, noise cut-off < 20, refractory period violation). However, the code only implements `noise_cutoff < 20` and `label >= 1`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to stimulus onset by subtracting `stimOn_times` from spike times, then binning into time bins relative to that onset.

ii.
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. The CONVERSION_NOTES.md confirms alignment to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a bin size of 20 ms with a time window of -0.2 to 1.0 s, producing 60 time bins per trial. No rebinning or interpolation is applied. The reference uses -0.5 to 1.5 s producing 100 time bins.

ii.
```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

iii. The CONVERSION_NOTES.md does not justify the choice of -0.2 to 1.0 s window. The metadata in the output confirms `off_start: -0.2` and `off_end: 1.0`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from `stimOn_times` in the trials table. The time grid is the bin centers of the edges array computed from -0.2 to 1.0 s in 20 ms steps.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. No specific justification beyond the alignment event being stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The bin centers are computed as `(edges[:-1] + edges[1:]) / 2`. These are the same for every trial -- they represent the time relative to stimulus onset at the center of each bin.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. No explicit justification; standard bin center computation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The input time values are the centers of the same bins used for neural data, so they are inherently aligned.

ii.
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
# ...
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. Same time grid is used for both neural binning and input time values.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. Derived from the `probabilityLeft` column of the trials table. A change in `probabilityLeft` marks the start of a new block.

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

iii. The CONVERSION_NOTES.md confirms computing trial number from consecutive runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop iterates over trials; when `probabilityLeft` changes, the counter resets to 1. The count starts at 1 (not 0). The counting is done on **filtered** trials (after removing invalid trials), so the block count may not reflect the animal's true position in the block if earlier trials were removed.

ii.
```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```
This is called after filtering:
```python
trials = trials.loc[valid].reset_index(drop=True)
# ...
trials = trials.loc[keep_trial].reset_index(drop=True)
# block_trial computed here, on filtered trials
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. No explicit justification for starting at 1 vs 0 or for computing on filtered trials.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Derived from the `choice` column of the trials table, where IBL convention uses +1 for left and -1 for right.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out
```

iii. The CONVERSION_NOTES.md confirms the IBL sign convention and the mapping to 0/1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Left (choice=1) is mapped to 0, right (choice=-1) is mapped to 1. No-response trials (choice=0) are excluded during trial filtering. The choice is then broadcast across all time bins as a per-trial constant.

ii.
```python
out = np.vstack([
    np.full_like(centers, choice[i], dtype=np.int64),
    ...
])
```

iii. Matches the instruction specification (left=0, right=1).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Derived from the `probabilityLeft` column of the trials table.

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. Matches the instruction specification.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The three values 0.2, 0.5, 0.8 are mapped to 0, 1, 2 respectively. Values not in the map get -1, though such trials should be filtered out. The prior is broadcast across all time bins.

ii.
```python
m = {0.2: 0, 0.5: 1, 0.8: 2}
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. Matches the instruction specification (0.2 -> 0, 0.5 -> 1, 0.8 -> 2).

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    if not (posf.exists() and tsf.exists()):
        return None, None
    return np.load(posf), np.load(tsf)
```

iii. The CONVERSION_NOTES.md confirms wheel position and timestamps as the source.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI computes wheel speed by: (1) sorting timestamps and removing duplicates, (2) computing raw finite differences `|dp/dt|` at midpoints, (3) interpolating these midpoint speeds onto the trial bin centers. This is a **raw finite difference** without any interpolation to a uniform grid or low-pass filtering (no Butterworth filter). The reference uses `SessionLoader.load_wheel()` which interpolates to 1000 Hz, applies a 20 Hz Butterworth low-pass filter, then differentiates.

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
    wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The CONVERSION_NOTES.md acknowledges the wheel processing approach but does not mention the lack of interpolation to a uniform grid or Butterworth filtering.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 33rd and 67th percentiles of the concatenated wheel speed values across all trials in a session are computed, then each value is classified as 0 (low), 1 (medium), or 2 (high) based on these thresholds.

ii.
```python
def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return 0.0, 0.0
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)

def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
```

iii. No explicit justification beyond matching the "3 bins" requirement.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same bin centers as the neural data using `np.interp`.

ii.
```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
        out.append(y.astype(np.float32))
    return out
```

iii. Uses same time grid as neural data for alignment.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` with corresponding `_ibl_leftCamera.times.npy` / `_ibl_rightCamera.times.npy`.

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

iii. The CONVERSION_NOTES.md confirms using ROI motion energy from available camera views.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The motion energy trace is interpolated onto trial bin centers. When **both** left and right cameras are available, the AI **averages** them. The reference prefers left camera and only falls back to right. No additional filtering or normalization is applied.

ii.
```python
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The CONVERSION_NOTES.md mentions choosing "available side or combining sides sensibly" but the averaging approach differs from the reference's left-preferred strategy.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same approach as wheel speed: 33rd and 67th percentiles of the session's concatenated motion energy values, then discretized into 0, 1, 2.

ii.
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
# ...
discretize_with_thresholds(me_trials[i], me_q1, me_q2),
```

iii. Same justification as wheel speed discretization.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Interpolated onto the same bin centers as neural data, same approach as wheel speed.

ii.
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. Same time grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with missing trial tables, no curated spikes, or fewer than 2 valid trials are skipped. Trials with NaN `stimOn_times`, invalid choice, or null `probabilityLeft` are dropped. Trials with all-zero neural activity are dropped. Sessions where wheel data is missing get zero-filled wheel speed arrays. Sessions where motion energy is missing get zero-filled whisker arrays. Failed sessions are caught with a try/except and skipped.

ii.
```python
if trials is None or len(trials) < 2:
    return None
# ...
if spike_data[0] is None:
    return None
# ...
try:
    p = process_session(sess, show_processing=args.show_processing)
except Exception as e:
    print(f'[WARN] failed session {sess}: {e}')
    p = None
```

iii. The CONVERSION_NOTES.md documents handling of zero-neural trials and wheel timestamp issues.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike data from disk (`np.load` of spike times and clusters arrays), followed by the per-trial spike binning loop. The full conversion of 461 sessions took approximately 14,653 seconds (~4 hours), processing sequentially (no parallelism).

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. The conversion log shows individual session times ranging from <1s to >300s.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates per trial, masking all spikes against each trial's time window. This could be vectorized by pre-sorting spikes and using searchsorted to find trial boundaries. The `compute_trial_number_in_block` function uses a Python loop that could be vectorized with `pandas.groupby.cumcount` or numpy operations. The cluster remapping uses a Python list comprehension (`[remap[c] for c in sc]`) that could be vectorized.

ii.
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
    # ...
```

```python
sc = np.array([remap[c] for c in sc], dtype=np.int64)
```

iii. No discussion of vectorization in CONVERSION_NOTES.md.

## 10-c. What processing does the code repeat multiple times?

i. The spike binning function computes `spike_times - s` for every trial from the full spike array, effectively scanning all spikes for each trial rather than pre-sorting and slicing. The `rglob` operations are called multiple times per probe to find different files.

ii.
```python
for s in stim_on:
    rel = spike_times - s  # subtracts all spikes from each trial onset
    mask = (rel >= t0) & (rel < t1)
```

iii. No discussion in CONVERSION_NOTES.md.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. When both left and right camera motion energies are available, the AI loads and processes both cameras and averages them. The reference only uses one camera (left preferred). Also, the `rglob` pattern searches recurse through the entire directory tree when only a single level would be needed.

ii.
```python
if left_me and left_t:
    streams.append(...)
if right_me and right_t:
    streams.append(...)
# ...
me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. No discussion in CONVERSION_NOTES.md.
