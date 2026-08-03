# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers sessions by recursively searching for `_ibl_trials.table.pqt` files under `data/one_cache/`, then walking up the directory tree to find the parent `alf` directory. Each session directory is processed individually in a serial loop. Per session, it loads: trial metadata from Parquet, spike data from numpy files per probe, wheel position/timestamps, and camera motion energy files. All 461 sessions are processed sequentially.

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

iii. The AI identified the IBL ONE cache directory structure during data exploration (Step 2) and confirmed that trial table parquet files serve as session markers. The approach iterates all sessions found in the cache.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are extracted from the session directory path. The subject name is the third-from-last component of the session path (e.g., `lab/Subjects/<subject>/date/number`). A sorted list of unique subjects is built, and each session is assigned a subject index.

ii.
```python
subject = session_path.parts[-3]
# In build_dataset:
subjects = sorted({p['subject'] for p in processed})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI used the ONE cache path convention where subjects are organized as `lab/Subjects/<subject_name>/date/session_number`. This is consistent with the IBL data architecture.

## 1-c. How are the data split into sessions?

i. Each directory containing an `_ibl_trials.table.pqt` file constitutes a session. Sessions are discovered via globbing and deduplicated. 461 sessions were found and kept (460 after filtering in some runs). Each session is processed independently and contributes one entry to the output lists.

ii.
```python
for i, sess in enumerate(sessions, 1):
    st = time.time()
    try:
        p = process_session(sess, show_processing=args.show_processing)
    except Exception as e:
        print(f'[WARN] failed session {sess}: {e}')
        p = None
    if p is not None:
        processed.append(p)
```

iii. The AI followed the IBL data organization where each session is a separate directory with ALF-format files. Sessions that fail processing are skipped with a warning.

## 1-d. How are the data split into trials?

i. Trials are rows in the `_ibl_trials.table.pqt` Parquet file. Each row represents one trial with columns for timing events, choices, contrasts, and probabilities. After filtering for valid trials, the remaining rows define the trials used for conversion.

ii.
```python
def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)
```

iii. The AI used the standard IBL trial table format. Trials are naturally defined by rows in the Parquet file.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on: (1) non-NaN `stimOn_times`, (2) valid choice values (must be -1 or 1, excluding no-go trials coded as 0), (3) non-NaN `probabilityLeft`, and (4) non-zero neural activity (trials where all spike bins are zero are removed). Sessions with fewer than 2 valid trials after filtering are skipped.

ii.
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
if len(trials) < 2:
    return None
# ...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. The AI noted that choice==0 represents no-go/miss trials that are invalid for binary choice decoding. The zero-neural trial filter was added after full conversion revealed warnings about all-zero neural activity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments) files from each probe directory, along with `clusters.metrics.pqt` (quality metrics for filtering), `clusters.channels.npy` (channel assignments), and `channels.brainLocationIds_ccf_2017.npy` (brain region IDs).

ii.
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. The AI identified these as the standard IBL spike-sorting output files from the ALF directory structure.

## 2-b. How is the `neural` data processed?

i. Spikes are binned into 20ms non-overlapping time bins from -0.2s to +1.0s relative to stimulus onset, producing spike count matrices of shape (n_neurons, 60) per trial. Neurons from multiple probes within a session are concatenated with an offset to create a unified neuron index space. The result is stored as float32.

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

iii. The AI chose 20ms bins consistent with the reference paper's specification of "nonoverlapping 20-ms bins" for wheel decoding. The -0.2s to 1.0s window was chosen as a reasonable pre/post-stimulus window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered using two criteria from `clusters.metrics.pqt`: (1) `noise_cutoff < 20` and (2) `label >= 1`. Only neurons passing both criteria are retained.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
kept_ids = np.where(keep)[0]
```

iii. The AI noted the paper requires amplitude > 50 uV, noise cutoff < 20, and refractory period violation criteria (per RIGOR/ref. 28). It used `noise_cutoff < 20` directly and `label >= 1` as a proxy for the combined quality assessment. The code comment notes: "provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later." The paper's three criteria (amplitude > 50 uV, noise cutoff < 20, refractory period violation) were not all explicitly implemented.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to stimulus onset. For each trial, spike times are expressed relative to `stimOn_times` from the trial table, then binned into the fixed time window [-0.2s, 1.0s].

ii.
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. The task instructions explicitly require alignment to stimulus onset. The AI used `stimOn_times` from the trial table for this alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 20ms (0.02s). Spike counts are directly binned at this resolution from raw spike times -- no intermediate binning or rebinning is applied. The window spans -0.2s to 1.0s, producing 60 time bins per trial. Metadata records `time_bin_size: 20.0` (in ms).

ii.
```python
def bin_spikes_for_trials(..., bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

iii. The 20ms bin size matches the reference paper's specification for wheel decoding ("averaged wheel values in nonoverlapping 20-ms bins").

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the bin edges computed for the spike binning time window, not from any specific raw data variable. The bin centers represent the time since stimulus onset.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
# In input construction:
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. Since trials are aligned to stimulus onset, the time axis is implicitly defined by the bin edges. The AI uses bin centers as the continuous time-since-stimulus-onset values.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. Bin edges are computed with `np.arange(-0.2, 1.0 + 1e-9, 0.02)`, producing 60 edges. Bin centers are the midpoints: `(edges[:-1] + edges[1:]) / 2`. This gives values from approximately -0.19s to +0.99s. The same time vector is used for every trial.

ii.
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. This is a straightforward computation. Since all trials share the same time grid relative to stimulus onset, the time-since-stimulus-onset input is identical across trials.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time vector uses the same bin edges as the neural spike binning, so it is inherently aligned. Both neural data and the time input share the same 60-bin time grid from -0.2s to 1.0s.

ii. Both use `edges` from `bin_spikes_for_trials`. The `centers` variable derived from these edges is used directly as `input[0]`.

iii. Alignment is guaranteed by construction since the same edges define both the neural bins and the time input.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column in the trial table. Changes in `probabilityLeft` indicate block boundaries.

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

iii. The AI infers block boundaries from consecutive runs of constant `probabilityLeft` values. This is a reasonable proxy since blocks in the IBL task are defined by constant prior probability.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The function iterates through `probabilityLeft` values sequentially. When the value changes from the previous trial, the counter resets to 1. Otherwise, it increments. This produces a per-trial integer (stored as float32) indicating the trial's position within its block. It is broadcast as a constant across all time bins within a trial.

ii.
```python
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. The agent did not extensively discuss this decision in the trajectory. It's a natural computation based on detecting block transitions through changes in prior probability.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the `choice` column in the trial table (`_ibl_trials.table.pqt`).

ii.
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. The AI identified `choice` as a standard IBL trial variable with values {-1, 0, 1}.

## 5-b. What processing is involved in computing `output` *Choice*?

i. IBL convention encodes choice as: left=1, right=-1, no-go=0. The AI maps: left(1)->0, right(-1)->1. No-go trials (choice==0) are excluded during trial filtering. The choice value is broadcast as a constant across all time bins.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out

# Per trial:
np.full_like(centers, choice[i], dtype=np.int64)
```

iii. The AI noted in its trajectory that raw choice values include 0 for no-go/miss trials. These are filtered out before mapping. The mapping left=0, right=1 matches the task specification.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trial table.

ii.
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. The AI identified `probabilityLeft` as the direct source for block prior probability.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Direct categorical mapping: 0.2->0, 0.5->1, 0.8->2. Values not matching these three are mapped to -1 (though this should not occur after trial filtering). The value is broadcast as a constant across all time bins.

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. This mapping follows the task specification exactly: 0.2->0, 0.5->1, 0.8->2.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` (wheel position) and `_ibl_wheel.timestamps.npy` (timestamps).

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

iii. The AI used the raw wheel position and timestamps from the standard IBL ALF files.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Processing steps: (1) Sort timestamps and remove duplicates, (2) compute finite-difference speed as `abs(dp/dt)` from consecutive position samples, (3) compute midpoint timestamps for the speed values, (4) interpolate speed to trial time bins using `np.interp`.

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

iii. The AI initially used `np.gradient` but switched to finite differences after encountering RuntimeWarnings from duplicate/non-monotonic timestamps. The sorting and deduplication steps were added to handle data quality issues.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Session-level tertile thresholds are computed from all trial wheel speed values (concatenated across trials within a session). The 1/3 and 2/3 quantiles define the boundaries. Values are discretized: <=q1 -> 0 (low), q1<x<=q2 -> 1 (medium), >q2 -> 2 (high).

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

iii. The AI initially used per-trial tertiles but switched to session-level thresholds for consistency and efficiency. The task specifies "discretized into 3 bins" without specifying the exact method, so session-level tertiles are a reasonable choice.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated to the same trial time bins as the neural data using `np.interp`. For each trial, the interpolation targets are `stim_onset + bin_centers`, ensuring alignment with the stimulus-onset-aligned neural time grid.

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

iii. Linear interpolation to bin centers ensures temporal alignment with the neural spike count bins.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy is derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` with corresponding timestamp files `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy`.

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

iii. The AI used the ROI motion energy files from the IBL camera data, consistent with the paper's description of whisker motion energy from left and right camera whisker pad ROIs.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each camera stream is independently interpolated to trial time bins. If both left and right camera streams are available, they are averaged element-wise per trial. If only one stream is available, it is used alone. If no streams are available, zeros are used.

ii.
```python
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The AI chose to average left and right camera streams when both are available, which is a reasonable approach though the reference paper may handle camera selection differently.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same session-level tertile approach as wheel speed. The 1/3 and 2/3 quantiles of all whisker motion energy values across trials within a session define the bin boundaries.

ii.
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
# Then:
discretize_with_thresholds(me_trials[i], me_q1, me_q2)
```

iii. Consistent with the wheel speed discretization approach.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Same interpolation method as wheel speed: `np.interp` to `stim_onset + bin_centers`, producing values aligned to the neural time grid.

ii.
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. Uses the same `interp_to_trial_bins` function ensuring consistent temporal alignment across all data streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing-data cases are handled: (1) Missing wheel data: zero arrays substituted. (2) Missing motion energy data: zero arrays substituted. (3) Missing spike data: session skipped entirely. (4) Missing trial table: session skipped. (5) NaN stimulus onset times or probability left: trials filtered out. (6) No-go trials (choice==0): filtered out. (7) All-zero neural trials: filtered out. (8) Sessions with <2 valid trials: skipped. (9) Duplicate/non-monotonic wheel timestamps: sorted and deduplicated. (10) Missing probe quality metrics: probe skipped (continues to next probe).

ii.
```python
# Missing wheel:
wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
# Missing ME:
me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
# Session-level skip:
if spike_data[0] is None:
    return None
# Exception handling:
except Exception as e:
    print(f'[WARN] failed session {sess}: {e}')
    p = None
```

iii. The AI adopted a defensive approach: skip what can't be loaded, substitute zeros for missing behavioral signals, and filter invalid trials. A broad exception handler catches any unexpected errors at the session level.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading spike data per session (reading large numpy files), (2) spike binning with the per-trial loop in `bin_spikes_for_trials` which iterates over all trials and uses `np.add.at`, (3) the Python-level cluster remapping loop. Full conversion took approximately 167 minutes for 461 sessions, with individual sessions taking from ~2s to ~135s depending on data size.

ii. Per conversion output: `processed 24/461 sessions; kept 24; dt=134.48s` shows some sessions taking over 2 minutes.

iii. The AI estimated ~10.7s/session on average but some sessions with large spike datasets took much longer. The 15-minute guideline from the instructions was significantly exceeded.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Key unvectorized loops: (1) The spike cluster remapping `sc = np.array([remap[c] for c in sc], dtype=np.int64)` iterates over potentially millions of spikes in Python. Could use `np.searchsorted` or a lookup array. (2) The `compute_trial_number_in_block` function uses a Python for-loop. (3) The `map_prior` function uses a list comprehension with `dict.get`. (4) The per-trial loop in `process_session` for constructing inputs/outputs.

ii.
```python
# Slow cluster remap:
sc = np.array([remap[c] for c in sc], dtype=np.int64)

# Slow trial number computation:
for i, v in enumerate(prob_left):
    if i == 0 or v != prev:
        c = 1
        prev = v
    else:
        c += 1
    out[i] = c

# Slow prior mapping:
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. The AI did not address these vectorization opportunities despite the task instructions emphasizing efficient code and vectorized loops.

## 10-c. What processing does the code repeat multiple times?

i. (1) The bin edges and centers are recomputed for each session (could be computed once). (2) Tertile thresholds are computed per session, repeating the concatenation and quantile computation. (3) The `interp_to_trial_bins` function recomputes bin centers from edges each time it's called. (4) The `rglob` searches for files are repeated for each probe within each session.

ii.
```python
# In interp_to_trial_bins, recomputes centers each call:
centers = (edges[:-1] + edges[1:]) / 2
```

iii. These redundancies are minor in terms of correctness but contribute to inefficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Brain region labels are computed as `ccf_<integer_id>` rather than human-readable acronyms, making them less useful but not discarded. (2) The `reducer='linear'` parameter in `interp_to_trial_bins` is never used (always defaults). (3) If wheel or motion energy data is missing, zero arrays are created and discretized, producing outputs that carry no real information but are still included in the dataset. (4) The `session_id` is computed and stored per session but may not be used downstream.

ii.
```python
# Unused parameter:
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
```

iii. The zero-array substitution for missing behavioral data is the most concerning unnecessary processing, as it introduces meaningless discretized outputs that could confuse the decoder.
