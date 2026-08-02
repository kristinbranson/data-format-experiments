# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans the local ONE cache filesystem directly rather than using the reference `ONE`/freeze-file workflow. It enumerates session directories under `data/one_cache/*/Subjects/*/*/*`, keeps only those with an `alf` folder and a trials table, and then loads each session's trials parquet, spike files, wheel files, and whisker motion-energy files from ALF paths. In practice this means it only processes locally present sessions and later drops sessions missing wheel or whisker data.

ii.
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)

def choose_sessions(session_dirs, mode):
    valid = []
    for s in session_dirs:
        alf = s / 'alf'
        if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
            valid.append(s)
```

```python
trials_df, trial_path = load_trials_table(session_alf)
spikes, clusters = load_spikes_for_session_dir(session_dir)
wt, wp = load_wheel(session_alf)
me_streams = load_motion_energy(session_alf)
```

iii. The justification in the agent's notes is speed and locality: it says it replaced `ONE` object loading with direct local `.npy`/`.pqt` ALF reads, reporting roughly a 10x speedup on the sample run. The Step 2 and Step 6 notes frame the local ONE cache as the data source and explicitly describe the change away from the reference `ONE` loaders.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the session path. The script takes `session_dir.parts[-3]` as the subject identifier, builds a unique `subjects` list, and stores one `subject_idx` per kept session.

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]

subj = subject
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. The notes justify this as a metadata-mapping shortcut: Step 5 says to extract subjects from session path metadata in the local ONE cache so session order matches `neural`/`input`/`output`.

## 1-c. How are the data split into sessions?

i. Each numeric directory at `lab/Subjects/subject/date/number` with an `alf` subdirectory is treated as one session. In full mode, every such session with a trials table becomes a candidate session before later filtering.

ii.
```python
for p in data_root.glob('*/Subjects/*/*/*'):
    if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
        sessions.append(p)

if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
    valid.append(s)
```

iii. The notes justify this with the Step 2 dataset exploration: the data are described as a local ONE cache organized by lab/subject/date/session folders, so the agent mirrored that organization in its loader.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of the session trials table. After computing a boolean `valid` mask, the script creates one neural/input/output entry per row index in `np.where(valid)[0]`.

ii.
```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
choice = map_choice(trials_df['choice'].to_numpy())
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)

for i, tr in enumerate(np.where(valid)[0]):
    ...
    sess_inputs.append(inp)
    sess_outputs.append(out_trial)
```

iii. The Step 5 notes say the target format should be trial-based and aligned to stimulus onset, so the agent used one row of the trials table per converted trial.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. A trial is kept if `stimOn_times` is finite, `choice` maps to left/right, and `probabilityLeft` maps to 0.2/0.5/0.8. Sessions with fewer than two such trials are dropped. The code does not apply the reference reaction-time filter, trial-duration filter, or checks for `feedback_times`, `firstMovement_times`, and `feedbackType`.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The notes and trajectory show the agent knew the reference code had a richer `load_trials_and_mask(...)` step, but the final script simplified this to the columns needed for the requested outputs plus a "must have at least 2 valid trials" rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from spike times and spike cluster assignments, with cluster QC and region metadata loaded from cluster/channel files. Specifically, the script uses `spikes.times.npy`, `spikes.clusters.npy`, `clusters.metrics.pqt`, `clusters.channels.npy`, and `channels.brainLocationIds_ccf_2017.npy`.

ii.
```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
metrics_p = src / 'clusters.metrics.pqt'
cl_chan_p = src / 'clusters.channels.npy'
ch_brain_p = src / 'channels.brainLocationIds_ccf_2017.npy'
```

iii. The notes justify this as a direct-file version of the reference spike loader: Step 5 maps neural data to `spikes.times.npy` plus `spikes.clusters.npy`, with merged probes and preserved region labels.

## 2-b. How is the `neural` data processed?

i. For each probe, the script selects the latest revision directory, loads spikes and clusters, filters clusters by `label >= 1`, remaps cluster IDs across probes, concatenates all probes from the session, sorts spikes by time, and bins spike counts into 20 ms bins from -0.5 s to +1.5 s around each kept trial's stimulus onset.

ii.
```python
good = labels >= 1
remap[good] = np.arange(good.sum()) + offset
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
clusters_list.append({'acronym': acr[good]})
...
order = np.argsort(spike_times)
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]
```

```python
for t0 in stim_on:
    edges = t0 + TIME_BINS
    out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
    ...
    np.add.at(out, (cl[m], bins[m]), 1)
```

iii. The Step 4 and Step 5 notes justify probe merging, 20 ms binning, and stimulus-onset alignment as the main reference behaviors to preserve from `prepare_data` and `bin_spiking_data`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only explicit neural QC filter in the script is `clusters.metrics.pqt['label'] >= 1`. Clusters failing that flag are removed before spike binning. The script does not implement additional paper-level region/session inclusion rules.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
...
keep2 = good[sc2]
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
clusters_list.append({'acronym': acr[good]})
```

iii. The notes explicitly describe this as "strict cluster QC filter using `label >= 1`" and connect it to the reference `good_clusters` metadata and "well-isolated neurons" language from the papers.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset. For each kept trial, the script bins spikes in a fixed window from -0.5 s to +1.5 s relative to `stimOn_times`.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)

neural_trials = bin_spikes(spikes['times'], spikes['clusters'], len(clusters['acronym']), stim_on[valid])
```

iii. The notes repeatedly justify this with the task requirement to "temporally align based on stimulus onset" and with the reference cache params `align_time='stimOn_times'`, `time_window=(-.5, 1.5)`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 20 ms bins, producing 100 bins per trial over a 2 s window. No further temporal rebinning is applied after spike counting.

ii.
```python
BINSIZE = 0.02
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
N_BINS = len(TIME_BINS)
```

iii. The Step 5 notes say to use 20 ms bins because the reference caching script uses `binsize=0.02`; the README repeats the same design choice.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is mostly derived from fixed constants defining the common time grid (`TIME_CENTERS`), with `stimOn_times` providing the alignment origin for the trial. No per-trial raw trace is loaded for this input.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
```

iii. The notes justify this as a task-format variable rather than a native recorded stream: Step 5 says to use the common trial-relative time axis as the first decoder input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script does not estimate this from data. It simply broadcasts the precomputed `TIME_CENTERS` vector into every trial as the first input row.

ii.
```python
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. The Step 5 mapping table says "Repeat common time vector across trials as a 1 x T input."

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It is aligned exactly to the same 100-bin grid used for neural spike counts. The same `TIME_CENTERS` that define behavior interpolation are written into the input array for each trial.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    ...
])
```

iii. The notes justify this with the general decision to put all converted streams on one stimulus-locked 20 ms grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table.

ii.
```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. The Step 5 notes say to compute trial number in block from the trial table's `probabilityLeft` sequence.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code walks through the `probabilityLeft` sequence, resets the counter to 1 whenever the value changes, and otherwise increments it. That means the unbiased 0.5 block is also counted as a block.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    ...
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
```

iii. The notes justify this as a simple within-block index derived from consecutive equal-probability runs in `probabilityLeft`.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived directly from the trials table `choice` column.

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. The Step 5 notes say to map the IBL `choice` coding into the requested left/right categories after checking the sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The script maps IBL's `choice == 1` to left (`0`) and `choice == -1` to right (`1`). Any other value stays `-1` and is filtered out by the `valid` mask. For kept trials, the chosen class is repeated across all 100 time bins.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

```python
np.full(N_BINS, choice[tr], dtype=np.int64)
```

iii. The notes explicitly mention verifying the sign convention and storing choice as a per-trial categorical output.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived directly from the trials table `probabilityLeft` column.

ii.
```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. The Step 5 mapping table says to source prior from `probabilityLeft`.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The script maps the three expected block values to category IDs: `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`. Any other value stays `-1` and is removed by the `valid` mask. For kept trials, the category is repeated across all 100 time bins.

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

iii. The notes justify this as the task-specified categorical remapping of block prior.

## 9-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    ...
    return np.load(tp), np.load(pp)
```

iii. The Step 5 notes justify this as the raw wheel position/timestamp source corresponding to the reference wheel loader.

## 9-b. What processing is involved in computing `output` *Wheel speed*?

i. The script removes non-finite and duplicate timestamps, differentiates raw wheel position with `np.gradient`, and linearly interpolates the resulting signed velocity onto the stimulus-locked 20 ms trial grid. It does not take an absolute value, so it is actually using wheel velocity rather than wheel speed.

ii.
```python
keep = np.isfinite(timestamps) & np.isfinite(position)
timestamps = timestamps[keep]
position = position[keep]
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
timestamps = uniq_t
position = position[uniq_idx]
vel = np.gradient(position, timestamps)
...
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The notes justify the differentiation/interpolation step as the way to obtain a time-varying wheel signal on the common neural grid. The later Step 10 notes justify the duplicate-timestamp handling as a warning cleanup, not a change in intended semantics.

## 9-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation, wheel values from all kept trials in the session are pooled, the 1/3 and 2/3 quantiles are computed, and each bin is labeled low/mid/high by those thresholds. Non-finite values are forced to category `0`.

ii.
```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    ...
    y[x > q1] = 1
    y[x > q2] = 2
    y[~np.isfinite(x)] = 0
```

iii. The Step 5 notes explicitly propose quantile-style thresholds "to avoid degenerate classes," and the README says wheel speed is discretized into 3 bins.

## 9-d. How is `output` *Wheel speed* aligned with the neural data?

i. It is interpolated onto the same `TIME_CENTERS` grid used for neural counts, with each trial aligned to that trial's `stimOn_times`.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
    trials.append(y.astype(np.float32))
```

iii. The notes justify this with the task-specific decision to align all outputs to stimulus onset, even though the original wheel-decoding analysis used first movement onset.

## 10-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from left/right camera timestamps and ROI motion-energy arrays: `_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`, `leftCamera.ROIMotionEnergy.npy`, and `rightCamera.ROIMotionEnergy.npy`.

ii.
```python
left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```

iii. The Step 3 and Step 5 notes justify this with the paper's description of whisker motion energy as ROI frame-difference energy from the side videos.

## 10-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads whichever side-camera streams are present, interpolates each one onto the stimulus-locked 20 ms grid, and averages the available streams bin-by-bin. If no camera stream is present, the whole session is skipped.

ii.
```python
for tp, mp in [(left_t, left_me), (right_t, right_me)]:
    if tp is not None and mp is not None and tp.exists() and mp.exists():
        streams.append((np.load(tp), np.load(mp).astype(np.float32)))
...
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
arr = np.stack(ys, axis=0)
valid = np.isfinite(arr)
denom = valid.sum(axis=0)
summed = np.where(valid, arr, 0.0).sum(axis=0)
y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
```

iii. The notes explicitly say to "combine left/right sensibly if both available" and to put whisker motion energy on the same 20 ms aligned grid.

## 10-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It is discretized with the same per-session tertile procedure used for wheel values: pooled finite values define `q1` and `q2`, higher bins become categories `1` and `2`, and non-finite values become `0`.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
...
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

iii. The notes justify this with the same quantile-style discretization decision used for wheel speed.

## 10-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is linearly interpolated onto the same `TIME_CENTERS` grid used for the neural data, with each trial aligned to that trial's stimulus onset.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    ys = []
    for ts, me in streams:
        ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. The notes justify this with the same "all streams stimulus-aligned" task adaptation described for wheel speed.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or malformed data are handled ad hoc. Missing trials tables raise an error and skip the session. Missing spikes, wheel data, or whisker motion-energy data cause the whole session to be skipped. Non-finite wheel or whisker bins are allowed through interpolation and later collapsed to category `0` during discretization. Duplicate wheel timestamps are deduplicated. Missing channel-to-region mappings fall back to `"void"`/`0`. The final dataset still contains some all-zero neural trials according to `verification_full_out.txt`.

ii.
```python
if spikes is None or len(clusters['acronym']) == 0:
    print('skip no spikes', session_dir)
    continue
...
if wt is None:
    print('skip no wheel', session_dir)
    continue
...
if me_streams is None:
    print('skip no whisker motion energy', session_dir)
    continue
```

```python
y[~np.isfinite(x)] = 0
...
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
```

iii. The justification comes from the trajectory's Step 10 review: the agent explicitly patched wheel duplicate timestamps and all-NaN whisker averaging to eliminate runtime warnings, while leaving the broader "skip session" and "NaN -> low bin" behavior intact.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are the outer loop over hundreds of sessions, loading per-probe spike files, per-trial spike histogramming in `bin_spikes`, and per-trial interpolation/discretization of wheel and whisker traces. The agent also spent significant time on full conversion reruns after patching warning-producing code.

ii.
```python
for si, session_dir in enumerate(session_dirs):
    ...
    spikes, clusters = load_spikes_for_session_dir(session_dir)
    ...
    neural_trials = bin_spikes(...)
    ...
    wheel_trials = interp_wheel_speed(...)
    whisk_trials = interp_motion_energy(...)
```

iii. The Step 6 and Step 7 notes explicitly discuss runtime, reporting that direct ALF reads reduced sample conversion from about 27 s to about 3 s for two sessions and estimating full conversion in minutes to tens of minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization candidates are the Python loop in `trial_number_in_block`, the per-trial loop in `bin_spikes`, the per-trial interpolation loops in `interp_wheel_speed` and `interp_motion_energy`, the `for i, tr in enumerate(np.where(valid)[0])` trial-packing loop, and the repeated region-index construction loop.

ii.
```python
for i, p in enumerate(prob_left):
    ...

for t0 in stim_on:
    ...

for i, tr in enumerate(np.where(valid)[0]):
    ...

for reg in clusters['acronym']:
    ...
```

iii. The notes do not explicitly list these vectorization opportunities, but the Step 6 speed discussion shows the agent was already concerned with conversion-time bottlenecks.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly casts or rebuilds the common time vector inside the per-trial packing loop, recomputes `sum(len(c['acronym']) for c in clusters_list)` for every probe while building cluster offsets, loops once to interpolate continuous wheel/whisk traces and then again to discretize them, and constructs per-trial constant arrays for choice, prior, and block number separately for every trial.

ii.
```python
offset = sum(len(c['acronym']) for c in clusters_list)
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
out_trial = np.vstack([
    np.full(N_BINS, choice[tr], dtype=np.int64),
    np.full(N_BINS, prior[tr], dtype=np.int64),
    wheel_bins[i].astype(np.int64),
    whisk_bins[i].astype(np.int64),
])
```

iii. The notes justify some of this indirectly: they mention speedups from avoiding expensive `ONE` loads, but they do not describe additional cleanup of repeated inner-loop work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several values are computed or imported but never used downstream: `defaultdict` and `ONE` are unused imports; `trial_path`, `lab`, `date`, and `number` are unpacked but unused; the continuous wheel and whisker traces are computed only to be immediately collapsed into 3 categories; and the logged tertile thresholds are not saved in the dataset metadata.

ii.
```python
from collections import defaultdict
from one.api import ONE
...
trials_df, trial_path = load_trials_table(session_alf)
...
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
...
print(f'kept session {kept}: ... wheel_thr={wheel_thr} whisk_thr={whisk_thr} ...')
```

iii. There is no explicit justification for these extra computations in the notes. They appear to be leftovers from development, logging, or format-adaptation rather than required reference processing.
