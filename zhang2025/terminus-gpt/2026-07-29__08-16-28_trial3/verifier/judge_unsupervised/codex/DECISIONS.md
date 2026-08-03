# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans the local ONE cache under `data/one_cache`, treats every `lab/Subjects/subject/date/number` directory with an `alf` folder as a candidate session, and keeps only sessions that have a trials parquet. In `--full` mode it processes all such sessions; in `--sample` mode it keeps the first two.

ii. ```python
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
    if mode == 'sample':
        return valid[:2]
    return valid

data_root = Path('data/one_cache')
session_dirs = choose_sessions(list_session_dirs(data_root), mode)
```

iii. The trajectory says the agent deliberately switched away from the reference ONE-based loader to direct local ALF reads for speed. `CONVERSION_NOTES.md` Step 6 says this reduced sample conversion time by about 10x.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the directory structure: `lab/Subjects/<subject>/<date>/<number>`. Unique subject strings are stored in `subjects`, and each kept session gets a `subject_idx`.

ii. ```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]

subj = subject
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the agent planned to use session-path metadata to populate `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each session directory `.../<subject>/<date>/<number>` is treated as one session. The script appends one entry per kept session to `neural`, `input`, `output`, and `brain_region_idx`.

ii. ```python
for si, session_dir in enumerate(session_dirs):
    ...
    out['neural'].append(neural_trials)
    out['input'].append(sess_inputs)
    out['output'].append(sess_outputs)
    out['brain_region_idx'].append(np.asarray(reg_idx, dtype=np.int64))
```

iii. The agent’s notes describe the workflow as session-level preprocessing on the local ONE cache, matching how the reference code caches one dataset per session.

## 1-d. How are the data split into trials?

i. Trials are rows of the trials table. After applying the `valid` mask, each surviving row becomes one neural trial matrix, one input array, and one output array.

ii. ```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
choice = map_choice(trials_df['choice'].to_numpy())
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)

neural_trials = bin_spikes(..., stim_on[valid])

for i, tr in enumerate(np.where(valid)[0]):
    ...
    sess_inputs.append(inp)
    sess_outputs.append(out_trial)
```

iii. In Step 5, the agent mapped trial-table variables directly onto per-trial decoder inputs and outputs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if `stimOn_times` is finite and both `choice` and `probabilityLeft` map to valid categories. Sessions are skipped entirely if they have fewer than 2 such trials, no spikes, no wheel data, or no whisker-motion-energy stream.

ii. ```python
if 'stimOn_times' not in trials_df.columns or 'choice' not in trials_df.columns or 'probabilityLeft' not in trials_df.columns:
    print('skip missing required trial columns', session_dir)
    continue
...
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
...
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

iii. The notes say the agent wanted “explicit valid-trial” handling, but the final code simplifies that to a small set of per-trial checks plus aggressive whole-session skipping when required streams are absent.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `spikes.times.npy` and `spikes.clusters.npy`, with cluster metadata from `clusters.metrics.pqt`. Region labels are inferred using `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`.

ii. ```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
metrics_p = src / 'clusters.metrics.pqt'
...
cl_chan_p = src / 'clusters.channels.npy'
ch_brain_p = src / 'channels.brainLocationIds_ccf_2017.npy'
```

iii. Step 5 explicitly mapped `spikes.times.npy` plus `spikes.clusters.npy` to the target `neural` field.

## 2-b. How is the `neural` data processed?

i. For each probe in a session, the latest revision folder is chosen, spikes and cluster metadata are loaded, clusters are filtered by `label >= 1`, cluster ids are remapped across probes, all probes are concatenated and time-sorted, then spike counts are binned per trial into 20 ms bins from -0.5 s to +1.5 s around stimulus onset.

ii. ```python
for probe_dir in sorted((session_dir / 'alf').glob('probe*/pykilosort')):
    revs = sorted([d for d in probe_dir.iterdir() if d.is_dir() and d.name.startswith('#')])
    src = revs[-1] if revs else probe_dir
    ...
    good = labels >= 1
    ...
    spikes_list.append((st2[keep2], remap[sc2[keep2]]))

spike_times = np.concatenate([x[0] for x in spikes_list])
spike_clusters = np.concatenate([x[1] for x in spikes_list])
order = np.argsort(spike_times)
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]

for t0 in stim_on:
    out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
    ...
    np.add.at(out, (cl[m], bins[m]), 1)
    mats.append(out)
```

iii. The agent’s notes justify 20 ms stimulus-aligned spike binning and probe merging by citing the reference `0_data_caching.py` parameters and the paper’s statement that neurons were combined across probes within a session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only clusters whose `label` is at least 1, drops spike entries that point outside the cluster table, and skips sessions where this leaves no neurons.

ii. ```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
...
keep_spk = (sc >= 0) & (sc < nclu)
sc2 = sc[keep_spk]
st2 = st[keep_spk]
keep2 = good[sc2]
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
...
if spikes is None or len(clusters['acronym']) == 0:
    print('skip no spikes', session_dir)
    continue
```

iii. In Step 4, the agent called out the reference code’s `good_clusters = (clusters['label'] >= 1)` convention and chose to follow it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to `stimOn_times`. Spike times are restricted to the window from `t0 - 0.5` to `t0 + 1.5`, then shifted by subtracting `t0`.

ii. ```python
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)

for t0 in stim_on:
    edges = t0 + TIME_BINS
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
    ts = spike_times[lo:hi] - t0
```

iii. The notes repeatedly cite the reference parameters `align_time='stimOn_times'` and the task requirement to align to stimulus onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 20 ms. The only temporal binning is the initial conversion to fixed 20 ms bins; there is no later rebinning.

ii. ```python
BINSIZE = 0.02
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
'time_bin_size': BINSIZE * 1000.0,
```

iii. `CONVERSION_NOTES.md` Step 4/5 says the agent chose 20 ms bins because the reference preprocessing uses `binsize=0.02`.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw time series. It is derived from the stimulus-onset alignment definition: `stimOn_times` supplies the event at time zero, and the input itself is the fixed relative time vector `TIME_CENTERS`.

ii. ```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. In Step 5, the agent described this field as a “common time vector across trials” tied to the reference alignment parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The script constructs a single 100-bin vector of bin centers spanning -0.49 s to 1.49 s relative to stimulus onset and reuses it for every kept trial.

ii. ```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. The notes justify this as the time-varying decoder input corresponding to the shared stimulus-aligned grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It uses exactly the same 100 time bins as the neural data and is attached only to the same set of `valid` trials used for the spike matrices.

ii. ```python
neural_trials = bin_spikes(..., stim_on[valid])
...
for i, tr in enumerate(np.where(valid)[0]):
    inp = np.vstack([
        TIME_CENTERS.astype(np.float32),
        np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
    ])
```

iii. The agent’s Step 5 mapping says this input should be repeated on the common stimulus-aligned grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial table’s `probabilityLeft` column.

ii. ```python
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

iii. Step 5 explicitly maps “trial table `probabilityLeft` and trial order within block” to this decoder input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script counts consecutive trials with the same `probabilityLeft` value. The counter resets to 1 when `probabilityLeft` changes and is then broadcast across all time bins of that trial.

ii. ```python
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
    return out
...
np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
```

iii. The justification in Step 5 is that block identity is carried by `probabilityLeft`, so the within-block trial number can be reconstructed from consecutive equal-probability runs.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice is derived from the trials table’s `choice` column.

ii. ```python
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. The Step 5 mapping notes say `choice` should come directly from the trial table after checking the IBL sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code remaps IBL choice codes to decoder classes: `1 -> 0` for left, `-1 -> 1` for right, and leaves everything else as invalid (`-1`). The chosen class is then repeated across all time bins of the trial.

ii. ```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
...
np.full(N_BINS, choice[tr], dtype=np.int64)
```

iii. The agent noted in Step 5 that it needed to verify the IBL sign convention and then map left/right to `0/1` as required by the task.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials table’s `probabilityLeft` column.

ii. ```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. Step 5 directly maps `probabilityLeft` to the prior-left output.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code converts `probabilityLeft` from its floating-point values to categorical labels: `0.2 -> 0`, `0.5 -> 1`, and `0.8 -> 2`. That category is then repeated across all time bins in the trial.

ii. ```python
def map_prior(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[np.isclose(vals, 0.2)] = 0
    out[np.isclose(vals, 0.5)] = 1
    out[np.isclose(vals, 0.8)] = 2
    return out
...
np.full(N_BINS, prior[tr], dtype=np.int64)
```

iii. The notes say this mapping was chosen to match the exact output specification in the task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii. ```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)
```

iii. Step 5 maps those two wheel files to the wheel-speed output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The wheel trace is finite-filtered, duplicate timestamps are removed, speed is estimated as `np.gradient(position, timestamps)`, and the resulting velocity is interpolated onto the stimulus-aligned bin centers for each trial.

ii. ```python
keep = np.isfinite(timestamps) & np.isfinite(position)
timestamps = timestamps[keep]
position = position[keep]
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
timestamps = uniq_t
position = position[uniq_idx]
vel = np.gradient(position, timestamps)
...
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
    trials.append(y.astype(np.float32))
```

iii. In the trajectory, the agent explicitly replaced a dependency on `brainbox.behavior.wheel.velocity` with this direct `np.gradient` implementation to keep the script self-contained.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All finite wheel-speed samples from the session are pooled, the 1/3 and 2/3 quantiles are computed, and each sample is labeled `low/mid/high` as `0/1/2`. Non-finite samples are assigned to class `0`.

ii. ```python
def discretize_tertiles(list_of_arrays):
    allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
    q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
    out = []
    for x in list_of_arrays:
        y = np.zeros_like(x, dtype=np.int64)
        y[x > q1] = 1
        y[x > q2] = 2
        y[~np.isfinite(x)] = 0
        out.append(y)
    return out, (float(q1), float(q2))
...
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
```

iii. Step 5 says the agent planned “quantile-style thresholds” for wheel speed to create the required 3-way categorical output.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. For every kept trial, wheel speed is evaluated on the same `TIME_CENTERS` grid used by the neural spike counts, after shifting those bin centers by that trial’s `stimOn_times`.

ii. ```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
    trials.append(y.astype(np.float32))
...
out_trial = np.vstack([
    np.full(N_BINS, choice[tr], dtype=np.int64),
    np.full(N_BINS, prior[tr], dtype=np.int64),
    wheel_bins[i].astype(np.int64),
    whisk_bins[i].astype(np.int64),
])
```

iii. The notes justify stimulus-onset alignment and the shared 20 ms grid as the central design choice for all time-varying streams.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from left/right camera timestamps (`_ibl_leftCamera.times.npy`, `_ibl_rightCamera.times.npy`) and left/right ROI motion-energy arrays (`leftCamera.ROIMotionEnergy.npy`, `rightCamera.ROIMotionEnergy.npy`).

ii. ```python
left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```

iii. Step 5 maps the left/right ROI motion-energy streams to the whisker-motion-energy decoder output.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The script loads whichever left/right streams exist, interpolates each onto the stimulus-aligned bin centers for each trial, and then averages the available left/right values at each time point.

ii. ```python
streams = []
for tp, mp in [(left_t, left_me), (right_t, right_me)]:
    if tp is not None and mp is not None and tp.exists() and mp.exists():
        streams.append((np.load(tp), np.load(mp).astype(np.float32)))
...
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    ys = []
    for ts, me in streams:
        ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
    arr = np.stack(ys, axis=0)
    ...
    y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
```

iii. `CONVERSION_NOTES.md` says the agent wanted to “combine left/right sensibly if both available,” based on the methods description of whisker-pad motion energy in the left and right videos.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session tertile procedure as wheel speed: pool all finite whisker-motion-energy samples, compute the 1/3 and 2/3 quantiles, label values as `0/1/2`, and send NaNs to `0`.

ii. ```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
...
y = np.zeros_like(x, dtype=np.int64)
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

iii. Step 5 says the agent planned the same quantile-style discretization for whisker motion energy as for wheel speed.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated onto the same trial-relative `TIME_CENTERS` grid used for spikes and wheel speed, with each trial anchored at `stimOn_times`.

ii. ```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    ...
    trials.append(y.astype(np.float32))
...
out_trial = np.vstack([
    np.full(N_BINS, choice[tr], dtype=np.int64),
    np.full(N_BINS, prior[tr], dtype=np.int64),
    wheel_bins[i].astype(np.int64),
    whisk_bins[i].astype(np.int64),
])
```

iii. The agent’s notes explicitly say all streams should share the stimulus-aligned 20 ms temporal grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing trial columns or missing required streams cause the whole session to be skipped. Non-finite `stimOn_times`, invalid choice labels, and unmapped `probabilityLeft` values drop individual trials. Duplicate wheel timestamps are deduplicated. Non-finite wheel/whisker samples propagate as NaNs until discretization, where they are forced into category `0`. Out-of-range spike cluster indices are discarded. Any other exception also causes the whole session to be skipped.

ii. ```python
if not pqt.exists():
    raise FileNotFoundError(...)
...
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
...
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
...
y[~np.isfinite(x)] = 0
...
keep_spk = (sc >= 0) & (sc < nclu)
...
except Exception as e:
    print('skip session due to error', session_dir, repr(e))
```

iii. The agent’s justifications emphasize robustness and speed. The trajectory also shows it preferred direct local file access and skipping problematic sessions over adding recovery/download logic.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the per-session file reads, per-trial spike binning, and per-trial interpolation of wheel speed and whisker motion energy. The notes also say the original version’s ONE object loading was a major bottleneck before the agent rewrote it.

ii. ```python
for si, session_dir in enumerate(session_dirs):
    ...
    neural_trials = bin_spikes(...)
    ...
    wheel_trials = interp_wheel_speed(...)
    ...
    whisk_trials = interp_motion_energy(...)
```

iii. `CONVERSION_NOTES.md` Step 6 explicitly says direct local ALF reads replaced much slower ONE object loading.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious Python-level loops are the per-trial loop in `bin_spikes`, the per-trial interpolation loops in `interp_wheel_speed` and `interp_motion_energy`, the loop in `trial_number_in_block`, and the loop that builds trial-level input/output arrays.

ii. ```python
for i, p in enumerate(prob_left):
    ...

for t0 in stim_on:
    ...

for i, tr in enumerate(np.where(valid)[0]):
    ...
```

iii. The agent did not spell these out in the notes, but its optimization work focused on speed and these are the main remaining Python loops.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly converts `TIME_CENTERS` to `float32` for every trial, repeatedly allocates `np.full` arrays for per-trial constant outputs, recomputes the cluster-offset sum inside each probe loop, and performs interpolation first and discretization in a second pass for both wheel and whisker streams.

ii. ```python
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

iii. This is not explicitly justified in the notes; it is repeated work visible directly in the implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script imports `ONE` and `defaultdict` without using them, reads `trial_path` and unpacks `lab/date/number` even though only `subject` is later used, computes `wheel_thr` and `whisk_thr` only for logging, and optionally creates debug plots that are not part of the saved dataset.

ii. ```python
from collections import defaultdict
from one.api import ONE
...
trials_df, trial_path = load_trials_table(session_alf)
...
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
...
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
...
if args.show_processing and kept < 2:
    make_plot(...)
```

iii. These look like leftovers from development/debugging rather than decisions the agent defended in the notes.
