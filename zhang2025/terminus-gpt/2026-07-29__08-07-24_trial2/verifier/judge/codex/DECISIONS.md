# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively scans `data/one_cache` for every `_ibl_trials.table.pqt`, infers the parent session directory for each hit, and then processes each discovered session independently. Within each session it loads the latest trial parquet, per-probe spike files, wheel files, and camera motion-energy files directly from the ALF cache instead of using the reference `ONE`/`SessionLoader` access path or the reference release tables.

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

def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    if not trial_files:
        return None
    trial_file = trial_files[-1]
    return pd.read_parquet(trial_file)
```

iii. The notes say the agent wanted to use the local IBL ONE cache as the source of truth and treat the session as the top-level unit. The trajectory also says it chose direct ALF loading because the needed variables were already present locally.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session path. The code assumes the ALF cache layout `.../Subjects/<subject>/<date>/<number>/...` and uses `session_path.parts[-3]` as the subject ID.

ii.
```python
subject = session_path.parts[-3]
...
'subject': subject,
```

iii. The notes say subject IDs should come from the session path structure, and the trajectory confirms the cache layout was explored specifically to extract subject/date/session information from directory names.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique parent directories of `alf` folders that contain a trial table. Each retained session becomes one top-level element of `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
if sess.name == 'alf':
    sessions.append(sess.parent)
...
'neural': [p['neural'] for p in processed],
'input': [p['input'] for p in processed],
'output': [p['output'] for p in processed],
```

iii. The notes explicitly record the decision that "use session as top-level unit in converted dataset" because both the papers and the reference code are organized around session-level decoding.

## 1-d. How are the data split into trials?

i. Trials are split by rows of the session trial table after filtering. For each retained trial row, the code creates one neural matrix, one input matrix, and one output matrix.

ii.
```python
trials = load_trials(session_path)
...
trials = trials.loc[valid].reset_index(drop=True)
...
for i in range(len(trials)):
    inp = np.vstack([...])
    out = np.vstack([...])
    inputs.append(inp)
    outputs.append(out)
```

iii. The justification in the notes is that the raw trial table already contains the task events needed for stimulus-aligned trial segmentation, especially `stimOn_times`, `choice`, and `probabilityLeft`.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials with non-null `stimOn_times`, `probabilityLeft`, and binary `choice` in `{-1, 1}`. After spike binning it also drops trials whose binned neural matrix is entirely zero. It does not implement the reference trial mask requiring non-null `feedback_times`, `feedbackType`, and `firstMovement_times`, and it does not impose the 0.08-2.0 s reaction-time filter.

ii.
```python
required = ['stimOn_times', 'choice', 'probabilityLeft']
...
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. The notes justify the no-choice and all-zero-neural filters as fixes discovered during validation. The trajectory shows those filters were added to eliminate decoder verification warnings, not because the agent had reconstructed the full reference trial QC mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from per-probe `spikes.times.npy` and `spikes.clusters.npy`, with `clusters.metrics.pqt` used for curation and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` used to assign region labels.

ii.
```python
metrics = pd.read_parquet(cand[-1])
chan_file = sorted(probe_dir.rglob('clusters.channels.npy'))[-1]
clu_channels = np.load(chan_file)
reg_id_files = sorted(probe_dir.rglob('channels.brainLocationIds_ccf_2017.npy'))
reg_ids = np.load(reg_id_files[-1]) if reg_id_files else None
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. The notes say this is electrophysiology data and therefore the correct raw neural source is spike-sorted units, not calcium signals or any derived firing-rate file.

## 2-b. How is the `neural` data processed?

i. The code merges probes within a session, remaps cluster IDs to a session-wide index, and bins spikes into trial-aligned spike-count matrices using a fixed window from -0.2 s to 1.0 s around stimulus onset with 20 ms bins. The result for each trial is `n_neurons x 60` spike counts.

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
            ...
            np.add.at(mat, (clu[good], tb[good]), 1)
        trial_mats.append(mat)
```

iii. The notes say the agent intentionally used binned spike counts and merged probes within a session because that matches the electrophysiology papers at a high level. The trajectory also notes that it chose a single stimulus-aligned representation for all outputs to satisfy the task format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by `noise_cutoff < 20` and `label >= 1` if those columns exist. Trials with all-zero neural bins are removed afterward. The code does not apply the paper's amplitude threshold, refractory-period criterion, grey-matter restriction, or region/session count restriction.

ii.
```python
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
kept_ids = np.where(keep)[0]
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. The notes say the agent planned to implement paper-compatible "well-isolated neuron" filtering, but the trajectory shows it only confirmed the existence of `noise_cutoff` and `label` fields and stopped there.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every neural trial is aligned to `stimOn_times`. For each trial, spike times are converted to time relative to that trial's stimulus onset and binned in a fixed stimulus-centered window.

ii.
```python
neural, edges = bin_spikes_for_trials(
    spike_times,
    spike_clusters,
    trials['stimOn_times'].to_numpy(),
    n_neurons
)
```

iii. The notes explicitly call this a task-driven deviation from the methods paper: the task required stimulus-onset alignment for all converted trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins for all sessions and all variables. No additional rebinning is applied after trial construction.

ii.
```python
def bin_spikes_for_trials(..., bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
...
'time_bin_size': bin_size * 1000.0,
```

iii. The notes justify 20 ms bins as matching the paper's dynamic-behavior decoding and as a convenient single common grid. There is no note showing that the agent reconciled this with the reference 50 ms choice/prior bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived indirectly from `stimOn_times`: once each trial is aligned to stimulus onset, the input is just the common relative-time vector for the chosen bin grid.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. The notes say the task explicitly asked for a time-since-stimulus input, so the agent represented it as relative time on the aligned trial grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes the midpoint of each 20 ms neural bin in the fixed stimulus-aligned window and uses that vector unchanged for every trial in the session.

ii.
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
...
np.vstack([
    centers,
    ...
])
```

iii. The trajectory's sanity checks explicitly verified that this vector equals the expected bin centers, so this was a deliberate and validated representation.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is perfectly aligned because it is derived from the same `edges` array used to bin spikes. Each time sample is the center of a neural time bin.

ii.
```python
neural, edges = bin_spikes_for_trials(...)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. The notes say the input time vector was sanity-checked against the expected stimulus-aligned bin centers, confirming the intended alignment.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trial table.

ii.
```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. The notes explicitly mapped "trial number within block" to runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans the per-session `probabilityLeft` sequence, resets the counter to 1 whenever the value changes, and increments the counter within each constant-probability block. That scalar is then repeated across all time bins of the trial.

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

iii. The notes justify this as the natural way to recover block position from the block-probability sequence.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the trial-table `choice` column.

ii.
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. The notes and trajectory both say the trial table directly exposes `choice`, so no indirect reconstruction was needed.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps IBL `choice` values `1 -> 0` (left) and `-1 -> 1` (right), rejects `choice == 0` earlier, and repeats the categorical value across all time bins of the trial.

ii.
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[arr == 1] = 0
    out[arr == -1] = 1
    return out
...
np.full_like(centers, choice[i], dtype=np.int64)
```

iii. The trajectory shows the agent explicitly checked the raw value convention `{-1, 0, 1}` and justified the mapping using the IBL rightward `-1` convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trial-table `probabilityLeft` column.

ii.
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. The notes explicitly map `probabilityLeft` to the requested "prior probability of left" output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code discretizes `probabilityLeft` with the fixed mapping `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, then repeats that category across all time bins in the trial.

ii.
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
...
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. The mapping exactly follows the task specification, and the notes state that this was an intentional direct encoding of the requested categories.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

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

iii. The notes say the raw wheel position/timestamp arrays were available in the cache and were the natural source for a wheel-speed output.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The code sorts wheel timestamps, removes duplicate timestamps, computes an absolute finite-difference speed `abs(dp / dt)` at midpoint timestamps, and linearly interpolates that speed onto the neural trial-bin centers. If wheel data are missing, it fills the trial with zeros.

ii.
```python
wheel_pos, wheel_ts = load_wheel(session_path)
if wheel_pos is not None:
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

iii. The notes justify this as a robustness fix after runtime warnings from duplicate or non-monotonic wheel timestamps. The trajectory shows it was added to clean verification, not because the agent had matched the reference `SessionLoader.wheel['velocity']` preprocessing.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. The code computes two tertile thresholds from all wheel-speed samples in that session and maps each time bin to `0/1/2` for low/medium/high.

ii.
```python
wheel_q1, wheel_q2 = tertile_thresholds(wheel_trials)
...
discretize_with_thresholds(wheel_trials[i], wheel_q1, wheel_q2),
```

iii. No explicit written justification was recorded beyond the task requirement that wheel speed become a 3-class categorical output.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned by interpolation onto the same stimulus-aligned bin centers used for neural data.

ii.
```python
wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The notes say all modalities were forced onto one stimulus-aligned grid because that was the explicit task requirement.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` together with the matching `_ibl_leftCamera.times.npy` and `_ibl_rightCamera.times.npy` arrays.

ii.
```python
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
```

iii. The notes explicitly identify the ROI motion-energy arrays as the available whisker-motion source in the local IBL cache.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each available camera stream is linearly interpolated to the stimulus-aligned trial bins. If both left and right streams exist, the code averages them per bin; if neither exists, it fills the trial with zeros.

ii.
```python
if me_streams:
    aligned = []
    for vals, ts, side in me_streams:
        aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
    if len(aligned) == 1:
        me_trials = aligned[0]
    else:
        me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
else:
    me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

iii. The notes only said the agent needed a "consistent rule" if both cameras were available. The trajectory later mentions averaging as that chosen rule; no reference-based justification was recorded.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The code computes session-level tertile thresholds across all whisker-motion values and maps each time bin to `0/1/2`.

ii.
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
...
discretize_with_thresholds(me_trials[i], me_q1, me_q2),
```

iii. No explicit justification was recorded beyond the task requirement that the output be discretized into three categories.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is aligned by interpolation onto the same stimulus-aligned bin centers used for spike counts.

ii.
```python
aligned.append(
    interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges)
)
```

iii. The notes say the agent intentionally put all neural and behavioral streams on one common stimulus-aligned grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code drops sessions with missing trial tables, too few trials, missing required trial columns, or no curated spikes. It drops invalid choice trials and all-zero-neural trials. For wheel and whisker data, however, it usually does not drop the session; instead it fills missing traces with zeros. Duplicate wheel timestamps are repaired by sorting and deduplication, and when multiple trial-table versions exist the latest file is chosen.

ii.
```python
if trials is None or len(trials) < 2:
    return None
...
if any(c not in trials.columns for c in required):
    return None
...
if spike_data[0] is None:
    return None
...
else:
    wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
...
else:
    me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

iii. The trajectory shows the zero-neural and wheel-timestamp fixes were validation-driven. There is no evidence that zero-filling missing behavioral targets came from the reference code or papers.

## 12-a. What are the most time-consuming steps of the code?

i. The slowest steps are session-by-session recursive file loading and, especially, per-trial spike binning. `bin_spikes_for_trials` loops over trials and, for each trial, subtracts the full spike-time vector and bins spikes with `np.add.at`, which is expensive on the full dataset.

ii.
```python
for i, sess in enumerate(sessions, 1):
    ...
    p = process_session(sess, show_processing=args.show_processing)
...
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
```

iii. The notes acknowledge that a full run took long enough to require repeated progress polling and that performance work focused on getting the converter through the entire 461-session dataset.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are the trial loop in `bin_spikes_for_trials`, the trial loop in `interp_to_trial_bins`, the Python-level remap of every spike cluster ID, and the per-trial construction of repeated static input/output arrays.

ii.
```python
for s in stim_on:
    ...
for s in stim_on:
    t = s + centers
    y = np.interp(t, timestamps, values)
...
sc = np.array([remap[c] for c in sc], dtype=np.int64)
...
for i in range(len(trials)):
    inp = np.vstack([...])
    out = np.vstack([...])
```

iii. No explicit efficiency justification was recorded, but these loops are exactly where the implementation spends avoidable Python overhead.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly walks the filesystem with `rglob`, repeatedly interpolates behavioral streams trial-by-trial, and repeatedly allocates full-length constant time series for per-trial static variables such as choice, prior, and trial number in block.

ii.
```python
trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
...
for probe_dir in session_probe_dirs(session_path):
    pks = sorted(probe_dir.rglob('pykilosort'))
    ...
    spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
...
out.append(y.astype(np.float32))
...
np.full_like(centers, choice[i], dtype=np.int64)
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. The notes do not justify these repetitions; they are side effects of a straightforward direct-from-files implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code saves optional processing plots, computes session-level brain-region strings as raw `ccf_*` IDs rather than the atlas acronyms used by the reference code, and expands static per-trial variables into full time series even though those outputs are constant within trial. It also averages left/right whisker streams even though downstream decoding only needs one categorical whisker signal.

ii.
```python
if show_processing and len(neural) > 0:
    save_processing_plot(...)
...
reg = f'ccf_{int(reg_ids[ch])}' ...
...
np.full_like(centers, choice[i], dtype=np.int64)
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. The notes mention the processing plots as a validation artifact and explicitly note that brain regions were left as stable CCF-ID labels rather than anatomical acronyms.
