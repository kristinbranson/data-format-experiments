# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the IBL `ONE` API or `SessionLoader`/`SpikeSortingLoader`. It scans the local `data/one_cache` filesystem for `_ibl_trials.table.pqt` files, infers session directories from those paths, then loads each modality directly from ALF files with `pd.read_parquet` and `np.load`.

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

```python
def load_trials(session_path: Path):
    trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
    ...
    return pd.read_parquet(trial_file)
```

```python
return np.load(posf), np.load(tsf)
...
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

iii. `CONVERSION_NOTES.md` says the data are organized as an IBL ONE cache and the trajectory shows the agent intentionally working from that cache layout. There is no evidence that it tried to reproduce the reference loaders; its practical justification was that the cached files already contain the needed ALF arrays.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the cache directory structure. For each processed session, the subject string is taken from `session_path.parts[-3]`, then `build_dataset()` makes `subjects` as the sorted unique subject IDs and `subject_idx` as the per-session index into that list.

ii. 
```python
subject = session_path.parts[-3]
...
return {
    'session_id': '/'.join(session_path.parts[-4:]),
    'subject': subject,
```

```python
subjects = sorted({p['subject'] for p in processed})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subject_idx': np.array([subject_to_idx[p['subject']] for p in processed], dtype=np.int64),
```

iii. The notes describe the cache as `data/one_cache/<lab>/Subjects/<subject>/<date>/<session_number>/alf`, so the agent treated the path structure itself as authoritative subject metadata.

## 1-c. How are the data split into sessions?

i. A session is defined as the directory above `alf`. The code finds every trial table, walks upward to `alf`, and uses `alf.parent` as the session root. Duplicate session paths are removed with `set()` and the remaining sessions are sorted.

ii. 
```python
for p in trial_tables:
    sess = p.parent
    while sess.name != 'alf' and sess != sess.parent:
        sess = sess.parent
    if sess.name == 'alf':
        sessions.append(sess.parent)
out = sorted(set(sessions))
```

iii. The notes explicitly say “use session as top-level unit in converted dataset,” based on the cache’s subject/date/session organization.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the session trial table. After loading `_ibl_trials.table.pqt`, the code filters rows and then treats each remaining row as one trial.

ii. 
```python
trials = load_trials(session_path)
...
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
trials = trials.loc[valid].reset_index(drop=True)
```

```python
for i in range(len(trials)):
    ...
    inputs.append(inp)
    outputs.append(out)
```

iii. The notes repeatedly treat the trials table as the session-level source of `stimOn_times`, `choice`, and `probabilityLeft`, so the agent accepted its row structure as the trial definition.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a narrower and different filter than the reference. It keeps only trials with non-null `stimOn_times`, binary `choice` values `{-1, 1}`, and non-null `probabilityLeft`. After neural binning, it further drops trials whose entire neural matrix is zero. It does not filter by reaction time or by wheel/camera coverage.

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

iii. `CONVERSION_NOTES.md` Step 10 says the agent “resolved earlier” invalid binary choice coding by filtering to `choice in {-1, 1}` and removed verification warnings by filtering all-zero-neural trials. There is no justification in the notes for omitting the reference reaction-time and coverage masks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural matrices are derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Additional raw files are used only to curate units and assign regions: `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

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

iii. The notes explicitly map `spikes.times.npy + spikes.clusters.npy + curated cluster list` to `neural`, with probe metadata retained for curation and brain-region indexing.

## 2-b. How is the `neural` data processed?

i. The AI merges probes within a session by remapping kept cluster IDs into one global neuron index. It then bins spikes into 20 ms counts in a stimulus-aligned window from `-0.2` s to `1.0` s and stores the resulting counts as `float32`. It does not convert counts to firing rates and does not sort merged spikes by time before per-trial slicing.

ii. 
```python
remap = {old: i + offset for i, old in enumerate(kept_ids)}
mask = np.isin(spike_clusters, kept_ids)
sc = spike_clusters[mask]
st = spike_times[mask]
sc = np.array([remap[c] for c in sc], dtype=np.int64)
all_times.append(st)
all_clusters.append(sc)
...
offset += len(kept_ids)
```

```python
def bin_spikes_for_trials(spike_times, spike_clusters, stim_on, n_neurons, t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
```

```python
neural[i] = neural[i].astype(np.float32)
```

iii. The notes justify spike-count binning on the grounds that this is electrophysiology data and the reference decodes from binned spikes. The same notes also say all modalities should be aligned to stimulus onset. There is no explicit justification in the notes for changing the trial window to `[-0.2, 1.0]` or for keeping counts instead of converting to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI performs “provisional curation” per probe: start with all clusters, then keep only those with `noise_cutoff < 20` if that column exists and `label >= 1` if that column exists. It does not implement the paper’s amplitude criterion or any refractory-period criterion.

ii. 
```python
# provisional curation using available fields; amplitude criterion may require conversion/field interpretation refinement later
keep = np.ones(len(metrics), dtype=bool)
if 'noise_cutoff' in metrics.columns:
    keep &= metrics['noise_cutoff'].to_numpy() < 20
if 'label' in metrics.columns:
    keep &= metrics['label'].to_numpy() >= 1
kept_ids = np.where(keep)[0]
```

iii. `CONVERSION_NOTES.md` says “Apply paper-compatible well-isolated neuron filtering using available cluster metrics before conversion.” The inline code comment is more candid: it calls the curation provisional and says the amplitude criterion “may require refinement later.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times` by subtracting the trial’s stimulus onset from all spike times and then keeping spikes in the per-trial relative-time window.

ii. 
```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
    rel = rel[mask]
```

iii. The notes make this decision explicit: “Align all modalities to stimulus onset because the task explicitly requires this.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins. No later temporal rebinning is applied after those bins are constructed.

ii. 
```python
def bin_spikes_for_trials(..., t0=-0.2, t1=1.0, bin_size=0.02):
    edges = np.arange(t0, t1 + 1e-9, bin_size)
```

```python
'time_bin_size': bin_size * 1000.0,
```

iii. The notes repeatedly refer to fixed stimulus-aligned bins and the metadata records a `20` ms bin size. No additional rebinning step appears anywhere in the code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The input is not read from a raw “time since onset” variable. It is generated from the common bin centers used for stimulus-aligned trial windows, with `stimOn_times` serving as the alignment anchor for those windows.

ii. 
```python
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

```python
wheel_trials = interp_to_trial_bins(speed_mid, ts_mid, trials['stimOn_times'].to_numpy(), edges)
...
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. The notes say this variable should be represented “as the common per-bin relative time values” once trials are aligned to stimulus onset.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code computes bin centers from the spike/bin edges and repeats that same `centers` vector in every trial as the first input channel. The actual range is `-0.19` s to `0.99` s because the underlying edges are `[-0.2, 1.0]` in 20 ms steps.

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

iii. The notes justify using a common trial time grid after stimulus alignment, but they do not explain why the window was shortened from the reference window.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the exact same bin edges and bin centers as the neural binning, so the first input channel is on the same per-trial time axis as the neural matrix.

ii. 
```python
neural, edges = bin_spikes_for_trials(...)
centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
```

```python
inp = np.vstack([
    centers,
    np.full_like(centers, block_trial[i], dtype=np.float32),
]).astype(np.float32)
```

iii. This is consistent with the notes’ plan to use a “common trial time grid” for all stimulus-aligned modalities.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from `probabilityLeft` in the trial table. Consecutive runs of the same `probabilityLeft` value are treated as blocks.

ii. 
```python
def compute_trial_number_in_block(prob_left):
    prob_left = np.asarray(prob_left)
    ...
    for i, v in enumerate(prob_left):
        if i == 0 or v != prev:
            c = 1
            prev = v
```

iii. The notes explicitly say to compute trial number within block “from consecutive runs of constant `probabilityLeft`.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI computes the count after trial filtering, not on the original unfiltered session table. It uses a one-based counter that resets to `1` whenever `probabilityLeft` changes, then broadcasts that scalar across all time bins of the trial.

ii. 
```python
if i == 0 or v != prev:
    c = 1
    prev = v
else:
    c += 1
out[i] = c
```

```python
block_trial = compute_trial_number_in_block(trials['probabilityLeft'].to_numpy())
...
np.full_like(centers, block_trial[i], dtype=np.float32),
```

iii. The notes justify the block segmentation itself, but they do not justify either the one-based indexing or recomputing it after trials have already been removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in the trial table.

ii. 
```python
def map_choice(choice_vals):
    arr = np.asarray(choice_vals)
    ...
    out[arr == 1] = 0
    out[arr == -1] = 1
```

```python
valid = trials['stimOn_times'].notna() & trials['choice'].isin([-1, 1]) & trials['probabilityLeft'].notna()
```

iii. The notes say this output should come from the raw choice column, using the IBL left/right coding once the sign convention was confirmed from the data.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code first filters out non-binary trials (`choice == 0` is removed), then recodes `1 -> 0` and `-1 -> 1`. The per-trial category is repeated across all time bins.

ii. 
```python
valid = ... & trials['choice'].isin([-1, 1]) & ...
```

```python
out = np.full(arr.shape, -1, dtype=np.int64)
out[arr == 1] = 0
out[arr == -1] = 1
```

```python
np.full_like(centers, choice[i], dtype=np.int64),
```

iii. `CONVERSION_NOTES.md` Step 10 says invalid binary coding from `choice == 0` trials was fixed by filtering to `{-1, 1}`.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trial table.

ii. 
```python
def map_prior(prob_left):
    m = {0.2: 0, 0.5: 1, 0.8: 2}
    return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

iii. The notes’ variable-mapping table explicitly maps `probabilityLeft` to the prior output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code recodes `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`, then repeats the per-trial category across all time bins.

ii. 
```python
m = {0.2: 0, 0.5: 1, 0.8: 2}
return np.array([m.get(float(x), -1) for x in prob_left], dtype=np.int64)
```

```python
np.full_like(centers, prior[i], dtype=np.int64),
```

iii. The notes say this recoding follows the decoder-task specification directly.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from raw wheel position and wheel timestamps: `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii. 
```python
def load_wheel(session_path: Path):
    alf = session_path / 'alf'
    posf = alf / '_ibl_wheel.position.npy'
    tsf = alf / '_ibl_wheel.timestamps.npy'
    ...
    return np.load(posf), np.load(tsf)
```

iii. The notes’ variable-mapping table explicitly maps those raw wheel arrays to `output[2]`.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI sorts wheel timestamps, removes duplicates, computes absolute finite-difference speed `abs(dp / dt)` at sample midpoints, then interpolates that speed trace onto each trial’s stimulus-aligned bin centers. It does not use the IBL wheel interpolation/filtering pipeline.

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

iii. The notes justify using raw wheel position/timestamps to derive a speed signal, and Step 10 says the deduplication/sorting logic was added to remove `RuntimeWarning`s from duplicate or non-monotonic timestamps.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The code computes session-level tercile thresholds over all binned wheel-speed values from all kept trials in the session, then applies those two thresholds to each bin to produce classes `0`, `1`, and `2`.

ii. 
```python
def tertile_thresholds(arrays):
    x = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None and len(a) > 0])
    ...
    q1, q2 = np.quantile(x[finite], [1/3, 2/3])
    return float(q1), float(q2)
```

```python
def discretize_with_thresholds(x, q1, q2):
    x = np.asarray(x)
    y = np.zeros_like(x, dtype=np.int64)
    y[x > q1] = 1
    y[x > q2] = 2
    return y
```

iii. The notes say wheel speed should be discretized into 3 bins and the trajectory later remarks that the outputs became nearly balanced because of per-session tertile discretization.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated onto the same per-trial bin centers used for neural binning, with each trial aligned to its own `stimOn_times`.

ii. 
```python
def interp_to_trial_bins(values, timestamps, stim_on, edges, reducer='linear'):
    centers = (edges[:-1] + edges[1:]) / 2
    out = []
    for s in stim_on:
        t = s + centers
        y = np.interp(t, timestamps, values)
```

iii. The notes say all modalities should share the common stimulus-aligned trial grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from `leftCamera.ROIMotionEnergy.npy` and/or `rightCamera.ROIMotionEnergy.npy` together with their camera time arrays. If both views are present, the AI uses both.

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

iii. The notes say whisker motion energy should come from ROI motion energy and explicitly leave the side-choice rule open: “if both left and right are available, choose a consistent rule (for example mean or preferred available side) and document it.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. Each available camera stream is linearly interpolated onto stimulus-aligned bin centers. If both left and right camera traces exist, the AI averages the two per-bin trial traces. If no camera trace exists, it fills the session with zeros.

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

iii. The notes justify using ROI motion energy directly and choosing a consistent rule for left/right availability. The code implements that rule as a mean across both views when both exist.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The code uses the same session-level tercile-threshold approach as wheel speed: flatten all binned whisker values from the kept trials of a session, compute the `1/3` and `2/3` quantiles, and assign bins to categories `0`, `1`, or `2`.

ii. 
```python
me_q1, me_q2 = tertile_thresholds(me_trials)
...
discretize_with_thresholds(me_trials[i], me_q1, me_q2),
```

iii. The notes specify 3-bin categorical whisker output, and the implementation mirrors the wheel-speed discretization exactly.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Whisker motion energy is interpolated to the same stimulus-aligned bin centers used for neural data.

ii. 
```python
aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
```

iii. This follows the notes’ general decision to put all modalities on the same stimulus-aligned grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed trial tables cause the whole session to be skipped. Missing required trial columns, missing neural data, fewer than two trials, or no nonzero-neural trials also cause the session to be skipped. Missing wheel or motion-energy streams do not cause trials to be dropped; instead, the code fills those outputs with zeros. Duplicate or non-monotonic wheel timestamps are handled by sorting and deduplicating before speed calculation.

ii. 
```python
if trials is None or len(trials) < 2:
    return None
required = ['stimOn_times', 'choice', 'probabilityLeft']
if any(c not in trials.columns for c in required):
    return None
...
if spike_data[0] is None:
    return None
...
if len(neural) < 2:
    return None
```

```python
if wheel_pos is not None:
    ...
else:
    wheel_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

```python
if me_streams:
    ...
else:
    me_trials = [np.zeros_like(centers) for _ in range(len(trials))]
```

iii. `CONVERSION_NOTES.md` Step 10 explicitly says duplicate wheel timestamps were handled by sorting/removing duplicates, invalid `choice == 0` trials were filtered, and all-zero-neural trials were dropped to satisfy verifier checks.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is likely loading large spike arrays from disk for every probe and then repeatedly rescanning all spike times for every trial. The code also performs many recursive `rglob()` searches within every session and every probe.

ii. 
```python
spike_times = np.load(sorted(probe_dir.rglob('spikes.times.npy'))[-1])
spike_clusters = np.load(sorted(probe_dir.rglob('spikes.clusters.npy'))[-1])
```

```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

```python
trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
...
cand = sorted(probe_dir.rglob('clusters.metrics.pqt'))
...
sorted(probe_dir.rglob('spikes.times.npy'))
```

iii. The notes do not contain a real performance analysis. This answer is inferred from the structure of the code and from the trajectory, which focused on correctness rather than runtime.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are obvious vectorization candidates: the manual loop for `trial_number_in_block`, the per-trial spike-binning loop, the per-trial interpolation loop used for wheel and whisker, and the per-trial left/right camera averaging loop.

ii. 
```python
for i, v in enumerate(prob_left):
    if i == 0 or v != prev:
        c = 1
```

```python
for s in stim_on:
    rel = spike_times - s
    ...
    np.add.at(mat, (clu[good], tb[good]), 1)
```

```python
for s in stim_on:
    t = s + centers
    y = np.interp(t, timestamps, values)
```

```python
me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The notes never justify these loops on readability or simplicity grounds. They are just the direct implementation the agent wrote.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats recursive file discovery many times inside session/probe processing, and it repeats full-session spike subtraction/filtering once per trial instead of reusing pre-windowed spike slices. It also recomputes interpolation separately for wheel and each camera stream with the same `stim_on` and `edges`.

ii. 
```python
trial_files = sorted((session_path / 'alf').rglob('_ibl_trials.table.pqt'))
...
left_me = sorted(alf.rglob('leftCamera.ROIMotionEnergy.npy'))
right_me = sorted(alf.rglob('rightCamera.ROIMotionEnergy.npy'))
...
cand = sorted(probe_dir.rglob('clusters.metrics.pqt'))
...
sorted(probe_dir.rglob('spikes.times.npy'))
sorted(probe_dir.rglob('spikes.clusters.npy'))
```

```python
for s in stim_on:
    rel = spike_times - s
    mask = (rel >= t0) & (rel < t1)
```

iii. There is no explicit note about these repeated computations; they are visible directly in the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script always imports `matplotlib` and contains optional processing-plot generation that is irrelevant to the saved decoder dataset. Inside `load_curated_spikes`, it also searches for `pykilosort` directories and stores them in `pks`, but never uses that variable afterwards. When both camera views are present, it loads and interpolates both full streams even though downstream analysis only consumes the final averaged trace.

ii. 
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

```python
def save_processing_plot(session_id, neural_trial, input_trial, output_trial, outpath):
    ...
```

```python
pks = sorted(probe_dir.rglob('pykilosort'))
if not pks:
    pks = [probe_dir]
```

```python
for vals, ts, side in me_streams:
    aligned.append(interp_to_trial_bins(vals.astype(np.float32), ts, trials['stimOn_times'].to_numpy(), edges))
...
me_trials = [np.mean(np.vstack([aligned[0][i], aligned[1][i]]), axis=0) for i in range(len(trials))]
```

iii. The notes say the processing plots were part of the workflow validation, but they are not used in the final pickle. There is no stated justification for the unused `pks` search.
