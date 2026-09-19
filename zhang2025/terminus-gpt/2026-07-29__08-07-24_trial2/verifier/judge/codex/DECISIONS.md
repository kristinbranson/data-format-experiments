# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use `ONE`, `SessionLoader`, or `SpikeSortingLoader`. It recursively scans `data/one_cache` for every `_ibl_trials.table.pqt`, treats each parent session directory as one session, and then loads trials, wheel, motion-energy, and spike files directly from disk with `pandas.read_parquet` and `numpy.load`. It processes sessions serially in a single loop. It does not pre-filter sessions to those with all required modalities; missing wheel or motion-energy streams are handled later inside `process_session`.

ii. 
```python
def find_sessions(base=Path('data/one_cache')):
    trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
    ...
    return sorted(set(sessions))
```

```python
def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    ...
    return pd.read_parquet(trial_file)
```

```python
for i, sess in enumerate(sessions, 1):
    p = process_session(sess, show_processing=args.show_processing)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this by documenting that the dataset is an IBL ONE cache laid out under subject/date/session directories, and in Step 5 it explicitly chose `subject/date/session path` as the source for `subjects` and session ordering. The trajectory shows it intentionally built a direct file-based loader first rather than relying on IBL loader classes.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the directory structure. For each processed session, the AI takes `session_path.parts[-3]` as the subject name, then builds `subjects` as the sorted unique set of those strings and `subject_idx` as the per-session index into that set.

ii. 
```python
subject = session_path.parts[-3]
```

```python
subjects = sorted({p['subject'] for p in processed})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64),
```

iii. Step 5 of `CONVERSION_NOTES.md` says `subject/date/session path` should be used to extract `subjects` and `subject_idx`, so this was a deliberate path-based parsing choice.

## 1-c. How are the data split into sessions?

i. Sessions are defined by the filesystem directory that contains an `alf/` folder with a trial table. The AI discovers all such directories with `find_sessions`, deduplicates them, sorts them, and then treats each one as one output session.

ii. 
```python
for p in trial_tables:
    sess = p.parent
    while sess.name != 'alf' and sess != sess.parent:
        sess = sess.parent
    if sess.name == 'alf':
        sessions.append(sess.parent)
```

iii. The notes describe the raw data as session-organized ALF directories and say session should be the top-level unit in the converted dataset, so the AI treated directory boundaries as session boundaries.

## 1-d. How are the data split into trials?

i. Trials come from rows of the `_ibl_trials.table.pqt` table. After loading the full trial table, the AI filters rows, resets the index, and then creates one neural/input/output item per remaining row.

ii. 
```python
trials = load_trials(session_path)
...
trials = trials.loc[valid].reset_index(drop=True)
```

```python
for i in range(len(trials)):
    ...
    inputs.append(inp)
    outputs.append(out)
```

iii. The AI’s notes state that trial tables contain the needed trial variables directly, so it treated each table row as one trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials with non-missing `stimOn_times`, `choice` in `{-1, 1}`, and non-missing `probabilityLeft`. After spike binning it drops any trial whose neural matrix is all zeros. It does not apply the reference reaction-time filter, does not check wheel/camera coverage before keeping a trial, and does not remove trials based on missing wheel or whisker streams; those streams are instead zero-filled if absent.

ii. 
```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
```

```python
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
trials = trials.loc[keep_trial].reset_index(drop=True)
neural = [m for m, k in zip(neural, keep_trial) if k]
```

iii. The explicit justifications recorded later in `CONVERSION_NOTES.md` are that filtering no-choice trials fixed binary-choice coding and filtering all-zero neural trials removed verification warnings. No explicit justification was documented for omitting the reference reaction-time and stream-coverage masks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived primarily from `spikes.times.npy` and `spikes.clusters.npy` from each probe. The code also reads `clusters.metrics.pqt`, `clusters.channels.npy`, and optionally `channels.brainLocationIds_ccf_2017.npy` for curation and region metadata, but the time-varying neural matrices themselves are built from spike times plus spike cluster assignments.

ii. 
```python
metrics = pd.read_parquet(cand[-1])
...
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. Step 5 in the notes maps `spikes.times.npy + spikes.clusters.npy + curated cluster list` to the target `neural` field, and Step 3 notes that this is electrophysiology data that should be represented as binned spikes.

## 2-b. How is the `neural` data processed?

i. The AI merges kept clusters across probes by remapping cluster ids with an `offset`, then bins spikes into 20 ms bins from `-0.2` to `1.0` seconds around stimulus onset. The output is a spike-count matrix per trial of shape `(n_neurons, n_bins)`. It does not divide by bin width, so the stored neural values are counts, not firing rates.

ii. 
```python
remap = {old: i + offset for i, old in enumerate(kept_ids)}
...
sc = np.array([remap[c] for c in sc], dtype=np.int64)
```

```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
```

iii. The notes justify this as using “binned spike counts rather than rates/dF/F because this is electrophysiology data,” and the trajectory repeatedly frames the representation as stimulus-aligned spike binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies a provisional per-cluster mask using `noise_cutoff < 20` when that column exists and `label >= 1` when that column exists. It keeps all spikes whose cluster id is in the kept set. It does not implement the paper’s amplitude or refractory-violation criteria, and it does not exclude atlas label `void`.

ii. 
```python
# provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
```

iii. The code comment itself calls this “provisional curation,” and the trajectory explicitly says the script “currently uses a provisional neuron curation rule (noise_cutoff and label only).”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each trial, the AI subtracts that trial’s `stimOn_times` value from every spike time, then bins spikes in a fixed window relative to that event. This aligns neural activity to stimulus onset.

ii. 
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. Step 5 of the notes lists alignment to stimulus onset as the first key decision and states this was done because the task explicitly required it.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses fixed 20 ms bins and does not perform any later temporal rebinning. The neural data are binned directly onto that grid.

ii. 
```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

```python
'time_bin_size': bin_size * 1000.0,
```

iii. The notes repeatedly refer to 20 ms bins from the reference workflow, so the bin size was a deliberate carry-over.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the trial alignment event `stimOn_times`, but numerically the stored input is the vector of fixed bin centers relative to that event rather than a column read directly from disk.

ii. 
```python
required = ['stimOn_times', 'choice', 'probabilityLeft']
...
neural, edges = bin_spikes_for_trials(..., trials['stimOn_times'].to_numpy(), n_neurons)
```

```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. Step 5 of the notes says `_ibl_trials.stimOn_times / stimOn_times column` should be converted into a common trial time grid because trials are aligned to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI creates a common vector of bin centers from the neural bin edges and uses that vector as the first input row for every trial. In this code the centers span `-0.2` to `1.0` seconds in 20 ms steps.

ii. 
```python
edges = np.arange(t0, t1 + 1e-9, bin_size)
...
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

```python
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. The notes justify this as representing time-since-stimulus-onset on a common per-bin relative time grid.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the exact same bin edges and bin centers as the neural data. The neural spikes are binned with `edges`, and the time input is the center of those same bins.

ii. 
```python
neural, edges = bin_spikes_for_trials(...)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

iii. The notes explicitly say that because trials are aligned to stimulus onset, the input can be represented by the common per-bin relative time values.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` column in the trial table.

ii. 
```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
```

iii. Step 5 of the notes maps “trial index within block from probabilityLeft runs” to this decoder input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI walks through the filtered `probabilityLeft` sequence in order, resets the counter whenever the value changes, increments within a run, and stores the resulting count. The count starts at `1`, not `0`, and because it is computed after trial filtering, dropped trials do not contribute to the within-block count.

ii. 
```python
for i, v in enumerate(prob_left):
    if i == 0 or v != prev:
        c = 1
        prev = v
    else:
        c += 1
    out[i] = c
```

iii. The notes justify the variable choice by saying block-trial number should be computed from consecutive runs of constant `probabilityLeft`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived directly from the trial table’s `choice` column.

ii. 
```python
choice = map_choice(trials['choice'].to_numpy())
```

iii. The notes identify `choice column` as the source variable for this output.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Trials with `choice == 0` are filtered out earlier. Remaining choices are mapped so `1 -> 0` and `-1 -> 1`, then repeated across all time bins for that trial.

ii. 
```python
out = np.full(arr.shape, -1, dtype=np.int64)
out[arr == 1] = 0
out[arr == -1] = 1
```

```python
np.full_like(centers, choice[i], dtype=np.int64)
```

iii. The code comment says this assumes the IBL convention `left=1`, `right=-1`, and the notes say choice should be mapped to left `0`, right `1`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trial table’s `probabilityLeft` column.

ii. 
```python
prior = map_prior(trials['probabilityLeft'].to_numpy())
```

iii. The notes identify `probabilityLeft column` as the source for the prior-probability output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats the resulting categorical value across all time bins of the trial.

ii. 
```python
m = {0.2: 0, 0.5: 1, 0.8: 2}
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

```python
np.full_like(centers, prior[i], dtype=np.int64)
```

iii. This follows the mapping recorded in Step 5 of the notes.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
posf = alf / '_ibl_wheel.position.npy'
tsf = alf / '_ibl_wheel.timestamps.npy'
...
return np.load(posf), np.load(tsf)
```

iii. Step 5 of the notes explicitly maps those wheel files to the wheel-speed output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI sorts wheel timestamps, removes duplicate timestamps, computes absolute finite-difference speed `abs(diff(position) / diff(time))` at midpoint timestamps, and then linearly interpolates that speed trace onto the stimulus-aligned trial-bin centers. If wheel data are missing or too short, it substitutes all-zero traces.

ii. 
```python
order = np.argsort(wheel_ts)
wheel_ts = np.asarray(wheel_ts)[order]
wheel_pos = np.asarray(wheel_pos)[order]
uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
wheel_ts = wheel_ts[uniq_mask]
wheel_pos = wheel_pos[uniq_mask]
```

```python
dt = np.diff(wheel_ts)
dp = np.diff(wheel_pos)
speed_mid = np.abs(dp / dt).astype(np.float32)
ts_mid = ((wheel_ts[:-1] + wheel_ts[1:]) / 2).astype(np.float64)
wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
```

iii. The notes say wheel speed should be derived from raw wheel position timestamps and discretized for the decoder, and Step 10 later justifies the sort/deduplicate logic as a fix for duplicate or non-monotonic wheel timestamps.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes one pair of tertile thresholds from all wheel-speed values in the session, then discretizes each time bin as `0`, `1`, or `2` depending on whether it is below the first threshold, between thresholds, or above the second threshold.

ii. 
```python
wheel_q1, wheel_q2 = tertile_thresholds(wheel_trials)
...
discretize_with_thresholds(wheel_trials[i], wheel_q1, wheel_q2)
```

```python
q1, q2 = np.quantile(x[finite], [1/3, 2/3])
...
y[x > q1] = 1
y[x > q2] = 2
```

iii. The trajectory discusses moving away from per-trial tertiles and the final code uses session-level thresholds, which matches the notes’ plan to discretize wheel speed into three bins.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same stimulus-aligned bin centers derived from the neural `edges`, so wheel and neural arrays share the same time axis.

ii. 
```python
centers = (edges[:-1] + edges[1:]) / 2
...
t = s + centers
y = np.interp(t, timestamps, values)
```

iii. The notes say wheel should be differentiated/interpolated to stimulus-aligned bins, matching the neural trial grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` and `_ibl_leftCamera.times.npy` when present, and also from `rightCamera.ROIMotionEnergy.npy` and `_ibl_rightCamera.times.npy` when present. If both sides exist, both are loaded.

ii. 
```python
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
left_t = sorted(alf.rglob('_ibl_leftCamera.times.npy'))
right_t = sorted(alf.rglob('_ibl_rightCamera.times.npy'))
```

```python
if left_me and left_t:
    streams.append((np.load(left_me[-1]), np.load(left_t[-1]), 'left'))
if right_me and right_t:
    streams.append((np.load(right_me[-1]), np.load(right_t[-1]), 'right'))
```

iii. Step 5 of the notes says whisker motion energy should come from ROI motion-energy traces and that if both left and right are available the code should pick a consistent rule such as averaging or preferring one side.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each available camera motion-energy stream is linearly interpolated onto the trial-bin centers. If only one side is present, that aligned trace is used. If both are present, the AI averages the aligned left and right traces timepoint-by-timepoint. If no stream exists, it uses all zeros.

ii. 
```python
for vals, ts, side in me_streams:
    aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
if len(aligned) == 1:
    me_trials = aligned[0]
else:
    me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The notes justify this broadly as “use ROI motion energy as whisker motion energy,” with the side-selection rule left open; the implemented rule is averaging when both sides are present.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, it uses one pair of tertile thresholds computed from all whisker motion-energy values in the session, then discretizes each aligned time bin into categories `0`, `1`, or `2`.

ii. 
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
...
discretize_with_thresholds(me_trials[i], me_q1, me_q2)
```

iii. The notes planned a 3-bin discretization for whisker motion energy, and the final implementation uses session-level tertiles.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI interpolates motion-energy values onto the same bin centers used for neural binning, with times expressed relative to each trial’s stimulus onset.

ii. 
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. The notes say whisker motion energy should be interpolated to the same stimulus-aligned bins as the neural activity.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing essential structures cause the whole session to be skipped: missing trial table, missing required trial columns, missing surviving spikes, or fewer than two kept trials. Missing wheel or whisker streams do not drop the session or trial; the code substitutes zero-valued traces. Duplicate or non-monotonic wheel timestamps are handled by sorting and deduplicating. Trials with all-zero binned neural activity are dropped after binning.

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
```

```python
else:
    wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
...
else:
    me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

```python
uniq_mask = np.concatenate([[True], np.diff(wheel_ts) > 0])
...
keep_trial = np.array([np.any(m != 0) for m in neural], dtype=bool)
```

iii. The clearest justification is in Step 10 of `CONVERSION_NOTES.md`: the AI says zero-neural trials were filtered to remove verification warnings and wheel timestamps were sorted/deduplicated to eliminate runtime warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive parts of this code are repeated recursive filesystem searches, loading full spike arrays and cluster tables from disk for every probe, and `bin_spikes_for_trials`, which rescans the entire spike vector for every trial in the session. The conversion log confirms wide session-to-session runtime variability consistent with spike I/O and repeated per-trial work.

ii. 
```python
trial_tables = sorted(base.rglob('_ibl_trials.table.pqt'))
...
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. No explicit performance analysis was written into the notes, but the trajectory repeatedly remarks that full conversion is slow and that wheel-speed warnings and per-session processing time needed cleanup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: `compute_trial_number_in_block`, the Python dictionary/list-comprehension remap of cluster ids, the per-trial spike-binning loop in `bin_spikes_for_trials`, the per-trial interpolation loop in `interp_to_trial_bins`, and the per-trial averaging loop used when both camera streams are present.

ii. 
```python
for i, v in enumerate(prob_left):
    ...
```

```python
sc = np.array([remap[c] for c in sc], dtype=np.int64)
...
for s in stim_on:
    ...
for s in stim_on:
    ...
```

iii. The AI did not document an explicit justification for leaving these loops in place. The notes’ “Code inefficiencies identified” section was left blank.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly traverses session directories with `rglob` for each modality, repeatedly rescans all spike times once per trial when binning, and repeatedly interpolates behavioral traces trial-by-trial even though each trial uses the same bin centers. It also recomputes bin centers inside `interp_to_trial_bins` every time the helper is called.

ii. 
```python
trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
```

```python
for s in stim_on:
    rel = spike_times - s
```

```python
centers = (edges[:-1] + edges[1:]) / 2
for s in stim_on:
    t = s + centers
```

iii. No explicit justification was recorded for this repetition.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. There are a few pieces of work that do not affect the final saved dataset: `pks = sorted(probe_dir.rglob('pykilosort'))` is computed but never used; the `reducer` argument of `interp_to_trial_bins` is unused; each session’s `session_id` is built and returned by `process_session` but then omitted from `build_dataset`; and if both camera streams exist the code loads and averages both even though the reference logic expected selecting one side.

ii. 
```python
pks = sorted(probe_dir.rglob('pykilosort'))
if not pks:
    pks = [probe_dir]
```

```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
```

```python
return {
    'session_id': '/'.join(session_path.parts[-4:]),
    ...
}
...
data = {
    'neural': [p['neural'] for p in processed],
```

iii. No explicit justification for these extra steps was documented.
