# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script uses `data/one_cache/Brainwidemap/sessions.pqt` as the master session list. It iterates over every row, converts each row into a local IBL/ALF session path, and loads per-session trial, spike, wheel, and motion-energy files directly from the cache.

ii. ```python
def session_dir_from_row(row):
    return Path('data') / 'one_cache' / str(row['lab']) / 'Subjects' / str(row['subject']) / str(row['date']) / f"{int(row['number']):03d}"

def build_dataset(sample=False):
    sessions = pd.read_parquet('data/one_cache/Brainwidemap/sessions.pqt')
    ...
    for i, (_, row) in enumerate(sessions.iterrows()):
        loaded = load_session(row)
```

iii. In `CONVERSION_NOTES.md`, the agent says the data are stored as an IBL ONE cache and that conversion should follow the ONE/ALF session structure. The trajectory also shows it deliberately used `sessions.pqt` as the session inventory and skipped sessions missing required files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column in `sessions.pqt`. A unique-subject list is built on the fly, and each retained session gets an integer `subject_idx`.

ii. ```python
        if subj not in subject_map:
            subject_map[subj] = len(subjects)
            subjects.append(subj)
        subject_idx.append(subject_map[subj])
```

iii. The notes say the dataset is organized by lab and subject directories, so the agent treated the session metadata `subject` field as the canonical mouse identifier.

## 1-c. How are the data split into sessions?

i. Each row of `Brainwidemap/sessions.pqt` is treated as one candidate session. A session is retained only if `load_session(row)` succeeds.

ii. ```python
    for i, (_, row) in enumerate(sessions.iterrows()):
        ...
        loaded = load_session(row)
        if loaded is None:
            log('  skipped: missing required data')
            continue
```

iii. The notes say the top-level Brainwidemap metadata tables define the session inventory. The trajectory shows the agent explicitly chose to scan those rows until it found valid sessions.

## 1-d. How are the data split into trials?

i. The script loads one trial table per session, applies a trial mask, resets the index, and then treats each remaining row as one trial. Trial tensors are built by iterating through `stimOn_times`.

ii. ```python
    trials = load_trials(session_dir)
    ...
    mask = trial_mask(trials)
    trials = trials.loc[mask].reset_index(drop=True)
    ...
    stim_times = trials['stimOn_times'].to_numpy(dtype=np.float64)
    for i, st in enumerate(stim_times):
        neural_trials.append(bin_spikes_for_trial(..., st))
```

iii. The notes say the reference code uses the trial table plus a valid-trial mask, and the agent followed that general pattern.

## 1-e. How are trials filtered based on quality controls?

i. The implemented filter is a simplified mask: it requires finite `stimOn_times`, `choice`, and `probabilityLeft`. After mapping labels, trials with non-finite mapped `choice` or `prior` are also dropped. The code does not implement the reference reaction-time or feedback-event exclusions.

ii. ```python
def trial_mask(trials):
    need = ['stimOn_times', 'choice', 'probabilityLeft']
    mask = np.ones(len(trials), dtype=bool)
    for c in need:
        if c in trials.columns:
            mask &= np.isfinite(trials[c].to_numpy())
    return mask

...
    for i in range(len(trials)):
        if not np.isfinite(choice[i]) or not np.isfinite(prior[i]):
            continue
        ...
        valid_trial_keep.append(i)
```

iii. In the notes, the agent said it intended to use the reference `load_trials_and_mask` logic and exclude trials missing required outputs. The trajectory later shows it focused on fixing NaN choice/prior labels, but it never added the full reference QC mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from spike times and spike-cluster assignments loaded from each probe’s `pykilosort` directory. Cluster metrics are used for unit selection, but the trial-wise neural tensor is built only from spikes.

ii. ```python
        st = base / 'spikes.times.npy'
        sc = base / 'spikes.clusters.npy'
        cm = base / 'clusters.metrics.pqt'
        ...
        spikes_t = np.load(st)
        spikes_c = np.load(sc)
```

iii. The notes explicitly identify `spikes.times`, `spikes.clusters`, and curated cluster metadata as the source for the neural signal, matching the IBL ephys convention.

## 2-b. How is the `neural` data processed?

i. The script merges probes within a session by remapping cluster IDs into one contiguous index space, then bins raw spike counts into 20 ms bins for each trial in a fixed stimulus-locked window. No smoothing, firing-rate normalization, or z-scoring is applied.

ii. ```python
        uniq = np.array(sorted(np.unique(spikes_c)))
        remap = {cid: i + offset for i, cid in enumerate(uniq)}
        spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)
        offset += len(uniq)

def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
    mat = np.zeros((n_neurons, len(edges) - 1), dtype=np.float32)
    ...
    np.add.at(mat, (sc, tbin), 1)
    return mat
```

iii. The notes say the reference pipeline bins spike counts trial-wise and merges probes within session, and the trajectory shows the agent intentionally replaced the initial placeholder with this spike-binning implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered using `clusters.metrics.pqt`: if a `label` column exists, only rows with `label == 1` are kept; otherwise, if `ks2_label` exists, only rows marked `"good"` are kept. If no metrics file is available, all clusters are kept.

ii. ```python
        good_ids = None
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

iii. The notes repeatedly say “good units only” and “use curated good units / passing units as in the IBL reference code and QC metadata.” The trajectory also mentions that good-unit filtering was inferred from cluster-metric columns.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. For each trial, spikes are counted from 0.2 s before stimulus onset to 1.0 s after stimulus onset.

ii. ```python
ALIGN_EVENT = 'stimulus onset'
T_START = -0.2
T_END = 1.0

def bin_spikes_for_trial(spike_times, spike_clusters, n_neurons, stim_time):
    edges = stim_time + build_time_edges()
```

iii. The notes say “Stimulus onset alignment: Required by the decoder task,” and the trajectory explicitly describes this as an intentional adaptation from the reference code because the user requested stimulus-onset alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins throughout. Neural data are binned directly at 20 ms; there is no second-stage rebinning after trial tensors are created.

ii. ```python
BIN_SIZE_S = 0.02

def build_time_edges():
    return np.arange(T_START, T_END + BIN_SIZE_S, BIN_SIZE_S)
```

iii. The notes justify a “20 ms common bin size” because the papers use 20 ms bins for population and wheel analyses, and because a common bin size simplifies alignment across neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a raw file. It is constructed from the fixed bin centers of the stimulus-aligned trial window, which are implicitly relative to `stimOn_times`.

ii. ```python
def build_time_centers():
    e = build_time_edges()
    return (e[:-1] + e[1:]) / 2

    centers = build_time_centers().astype(np.float32)
```

iii. The notes say the stimulus-onset-aligned time axis should be a decoder input, derived from the trial interval construction rather than from a separate stored signal.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes evenly spaced bin centers for the fixed trial window and writes those centers directly as the first input row for every trial.

ii. ```python
    input_trials.append(np.vstack([
        centers,
        np.full_like(centers, trial_in_block[i])
    ]).astype(np.float32))
```

iii. The notes describe this as a “continuous time-since-stimulus-onset” input, replicated across trials with the same time axis.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The same `centers` array used for behavioral binning is paired with the spike-count matrix for each trial, so input and neural data share the same number of bins and the same stimulus-relative window.

ii. ```python
    centers = build_time_centers().astype(np.float32)
    ...
    neural_trials.append(bin_spikes_for_trial(..., st))
    input_trials.append(np.vstack([centers, ...]).astype(np.float32))
```

iii. The notes say the common 20 ms binning and stimulus-onset alignment were chosen specifically to keep all streams synchronized.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. ```python
    prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
    trial_in_block = trial_number_in_block(prob_left)
```

iii. The notes explicitly map `trials.probabilityLeft` to “trial number within block” via a custom run-length transform.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code computes a run length: it starts at 1 and increments while consecutive `probabilityLeft` values stay the same, then resets to 1 when the block prior changes.

ii. ```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    ...
    for i in range(1, len(prob_left)):
        if prob_left[i] == prob_left[i - 1]:
            run += 1
        else:
            run = 1
        out[i] = run
```

iii. The notes justify this as a necessary derived quantity because “trial number in block” is not a native ALF variable.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is taken from the trial table’s `choice` column.

ii. ```python
    choice = np.array([map_choice(v) for v in trials['choice'].to_numpy()], dtype=np.float32)
```

iii. The notes identify `trials.choice` as the source variable for the decoded choice output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps IBL trial codes to the requested labels: `1 -> 0` for left and `-1 -> 1` for right. The resulting categorical value is broadcast across all time bins in the trial.

ii. ```python
def map_choice(v):
    # IBL usually: 1=CCW(left), -1=CW(right)
    return 0 if v == 1 else 1 if v == -1 else np.nan

...
            np.full(centers.shape, int(choice[i]), dtype=np.int64),
```

iii. The notes say “Map left=0, right=1” and the trajectory later shows a raw-vs-converted sanity check for the first trial’s choice label.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The output prior is taken from the trial table’s `probabilityLeft` column.

ii. ```python
    prob_left = trials['probabilityLeft'].to_numpy(dtype=np.float32)
    prior = np.array([map_prior(v) for v in prob_left], dtype=np.float32)
```

iii. The notes explicitly map `trials.probabilityLeft` to the requested prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code discretizes the block prior exactly as requested: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. The result is broadcast across the trial’s time bins.

ii. ```python
def map_prior(v):
    if np.isclose(v, 0.2):
        return 0
    if np.isclose(v, 0.5):
        return 1
    if np.isclose(v, 0.8):
        return 2
    return np.nan
```

iii. The notes justify this as direct compliance with the decoder-task label mapping.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. ```python
def load_wheel(session_dir):
    alf = session_dir / 'alf'
    pos = alf / '_ibl_wheel.position.npy'
    ts = alf / '_ibl_wheel.timestamps.npy'
    ...
    return np.load(ts), np.load(pos)
```

iii. The notes say wheel variables should come from cached wheel data when available. The trajectory later confirms the agent used raw wheel files rather than the reference `SessionLoader`.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The script computes wheel speed as the absolute finite difference of wheel position over time, using midpoints of successive wheel timestamps. It then averages those values into stimulus-aligned bins.

ii. ```python
def wheel_speed(ts, pos):
    dt = np.diff(ts)
    dp = np.diff(pos)
    good = dt > 0
    mids = ts[:-1][good] + dt[good] / 2
    vv = np.abs(dp[good] / dt[good]).astype(np.float32)
    return mids, vv

...
        ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. The notes justify using wheel speed because it is one of the requested outputs. The trajectory describes this as a “first-pass loader” and later focuses on making its outputs categorical and verification-safe.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The code pools all finite wheel-speed bin values within a session, computes the 1/3 and 2/3 quantiles, and uses those session-specific thresholds to digitize each trial into categories `0, 1, 2`. Missing bins are filled before digitization using the lower threshold.

ii. ```python
    all_wheel_cont = np.concatenate(all_wheel_cont) if any(len(x) for x in all_wheel_cont) else np.array([], dtype=np.float32)
    wq = np.quantile(all_wheel_cont, [1/3, 2/3]) if len(all_wheel_cont) else [0, 0]
    ...
    wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The notes say wheel speed must be discretized into 3 bins because the task requires categorical outputs, and mention choosing thresholds after inspecting distributions. The trajectory later notes this was done “for reproducibility.”

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by computing stimulus-relative bin centers and averaging wheel-speed samples that fall inside each 20 ms bin at `stimOn_time + center`.

ii. ```python
def bin_signal(ts, values, centers):
    edges = np.concatenate([[centers[0] - BIN_SIZE_S / 2], centers + BIN_SIZE_S / 2])
    ...
    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])

...
        ws = bin_signal(wheel_v_ts, wheel_v, st + centers)
```

iii. The notes say all streams should share the same stimulus-onset alignment and 20 ms binning. The trajectory later frames this as a task-driven adaptation from the first-movement alignment used in the methods paper.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The script loads `leftCamera.ROIMotionEnergy.npy` and `rightCamera.ROIMotionEnergy.npy` from the session’s revision folders, if present.

ii. ```python
def find_motion_energy(session_dir):
    alf = session_dir / 'alf'
    left = sorted(alf.glob('#*/leftCamera.ROIMotionEnergy.npy'))
    right = sorted(alf.glob('#*/rightCamera.ROIMotionEnergy.npy'))
    return (left[-1] if left else None), (right[-1] if right else None)
```

iii. The notes say whisker motion energy should come from cached video-derived variables, and the trajectory later acknowledges that its timestamp handling for this signal was approximate.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. If both left and right motion-energy arrays exist, the code truncates them to the same length and averages them samplewise. It then assigns timestamps using a heuristic helper and bins the result into stimulus-aligned 20 ms windows.

ii. ```python
def load_motion_energy(session_dir):
    left_p, right_p = find_motion_energy(session_dir)
    vals = []
    ...
    n = min(len(v) for v in vals)
    arr = np.mean(np.vstack([v[:n] for v in vals]), axis=0)
    return arr

def guess_motion_timestamps(session_dir, n):
    ...
```

iii. The notes initially planned to use cached video-derived whisker variables. The trajectory later states the script still had “approximate whisker timestamps and simplified video handling.”

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, the code pools all finite whisker-motion values within a session, computes session-specific tertile thresholds, digitizes each trial into `0, 1, 2`, and fills missing bins before digitization using the lower threshold.

ii. ```python
    all_me_cont = np.concatenate(all_me_cont) if any(len(x) for x in all_me_cont) else np.array([], dtype=np.float32)
    mq = np.quantile(all_me_cont, [1/3, 2/3]) if len(all_me_cont) else [0, 0]
    ...
    mb = np.digitize(np.nan_to_num(tmp_me[i], nan=mq[0] if np.ndim(mq) else 0.0), mq, right=False).astype(np.int64)
```

iii. The notes say whisker motion energy must be discretized into 3 bins because the task requires categorical outputs.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The code aligns whisker motion energy by pairing the guessed session timestamps with stimulus-relative bin centers and averaging samples within each 20 ms neural bin.

ii. ```python
    me = load_motion_energy(session_dir)
    me_ts = guess_motion_timestamps(session_dir, len(me)) if me is not None else None
    ...
    ms = bin_signal(me_ts, me, st + centers) if me is not None else np.full(len(centers), np.nan, dtype=np.float32)
```

iii. The notes say all streams should be stimulus-onset aligned. The trajectory later explicitly calls out “approximate whisker timestamps” as a likely cause of weaker whisker decoding.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code mostly handles missing data by skipping entire sessions that lack required trial or spike files, dropping trials with invalid mapped `choice` or `prior`, filling missing wheel/whisker bins before discretization, and falling back to permissive defaults when metadata are missing. Brain regions are set to `unknown` for all neurons.

ii. ```python
    if not session_dir.exists():
        return None
    ...
    if trials is None or 'stimOn_times' not in trials.columns:
        return None
    ...
    if spike_times is None:
        return None
    ...
    region_names.extend(['unknown'] * len(uniq))
    ...
    wb = np.digitize(np.nan_to_num(tmp_wheel[i], nan=wq[0] if np.ndim(wq) else 0.0), wq, right=False).astype(np.int64)
```

iii. The notes say missing data should be handled “appropriately” and documented. The trajectory shows specific fixes for NaN wheel/whisker outputs, sample-mode session selection, and acceptance of rare all-zero neural trials.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are the per-session trial loop, trial-by-trial spike binning, and per-trial binning of wheel and whisker signals. The code also scans every session row in `sessions.pqt`.

ii. ```python
    for i, st in enumerate(stim_times):
        neural_trials.append(bin_spikes_for_trial(..., st))
        ...
        ws = bin_signal(...)
        ms = bin_signal(...)

    for i, (_, row) in enumerate(sessions.iterrows()):
        loaded = load_session(row)
```

iii. The trajectory explicitly identifies full-conversion runtime as a problem and notes that valid sessions took roughly tens of seconds because spike binning was done trial-by-trial in Python.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are straightforward vectorization candidates: the per-trial spike binning loop, the `bin_signal` per-bin loop, the list-comprehension remap of cluster IDs, and the run-length loop for trial-in-block.

ii. ```python
    spikes_c = np.array([remap[c] for c in spikes_c], dtype=np.int64)

    for i in range(len(centers)):
        m = idx == i
        if np.any(m):
            out[i] = np.nanmean(values[m])

    for i, st in enumerate(stim_times):
        ...
```

iii. The notes briefly mention vectorization and speedups, and the trajectory explicitly says the code should have been optimized because the first full run projected to take several hours.

## 12-c. What processing does the code repeat multiple times?

i. The code repeats path globbing and file discovery for each session, repeatedly loads the trial table in multiple helpers, and separately accumulates, concatenates, and then re-digitizes wheel and whisker data every session.

ii. ```python
def find_trial_table(session_dir):
    alf = session_dir / 'alf'
    cands = sorted(alf.glob('#*/_ibl_trials.table.pqt'))

def guess_motion_timestamps(session_dir, n):
    ...
    trials = load_trials(session_dir)

    all_wheel_cont.append(ws[np.isfinite(ws)])
    all_me_cont.append(ms[np.isfinite(ms)])
```

iii. The trajectory mentions repeated session scanning and repeated loading work as part of its runtime concerns, especially before the long full conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `wheel_bins_all` and `me_bins_all` but never uses them, loads `clusters.channels.npy` without using it, keeps `n_target` without using it, and builds `region_names` only to collapse all regions to `unknown`.

ii. ```python
    wheel_bins_all = discretize_three_bins(all_wheel_cont) if len(all_wheel_cont) else np.array([], dtype=np.float32)
    me_bins_all = discretize_three_bins(all_me_cont) if len(all_me_cont) else np.array([], dtype=np.float32)

        ch = base / 'clusters.channels.npy'
        ...
    n_target = 2 if sample else len(sessions)
```

iii. The trajectory focused on runtime but did not remove these extra computations, so they remained in the final script even though they do not affect the saved dataset.
