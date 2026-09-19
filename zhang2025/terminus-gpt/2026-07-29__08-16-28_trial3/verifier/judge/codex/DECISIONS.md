# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE release index or the reference loaders. It scans `data/one_cache` for session directories, keeps directories that contain an `alf/` folder plus a trials table, then reads raw `.pqt` and `.npy` files directly from those ALF folders. Session selection is purely filesystem-driven; `--sample` just keeps the first two valid sessions.

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
    if mode == 'sample':
        return valid[:2]
    return valid
```

```python
trials_df, trial_path = load_trials_table(session_alf)
wt, wp = load_wheel(session_alf)
me_streams = load_motion_energy(session_alf)
spikes, clusters = load_spikes_for_session_dir(session_dir)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly says it replaced ONE-based loading with "direct local `.npy`/`.pqt` reads from session ALF directories" because the initial ONE/object-loading version was too slow.

## 1-b. How are the data split into subjects (mice)?

i. The AI infers subject identity from the session path layout, not from ONE metadata. It takes the directory component three levels above the session number (`lab/Subjects/<subject>/<date>/<number>`) and assigns each unique subject an index in first-seen order.

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```

```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]

if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. The notes say the local ONE cache hierarchy is `lab/Subjects/subject/date/...`, and the mapping table in Step 5 says subjects/`subject_idx` will be extracted from that local cache structure.

## 1-c. How are the data split into sessions?

i. Each numeric session directory under `data/one_cache/<lab>/Subjects/<subject>/<date>/<number>` is treated as one session if it has `alf/` and a trials table. The code does not use `eid`s or the release session table to define sessions.

ii.
```python
for p in data_root.glob('*/Subjects/*/*/*'):
    if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
        sessions.append(p)
```

```python
if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
    valid.append(s)
```

iii. `CONVERSION_NOTES.md` Step 2 says the data are organized as a local ONE cache with lab-specific directories; Step 6 says the final script uses "direct ALF file loading" from those session directories.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of `_ibl_trials.table.pqt`. After a boolean `valid` mask is built, the kept trial indices are iterated and one neural/input/output item is produced per surviving row.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
```

```python
for i, tr in enumerate(np.where(valid)[0]):
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

iii. The notes repeatedly describe the workflow as session-level loading followed by trial-by-trial alignment and packaging on a common time base.

## 1-e. How are trials filtered based on quality controls?

i. The final code applies only a minimal trial mask: `stimOn_times` must be finite, choice must map to left/right, and `probabilityLeft` must map to one of `0.2/0.5/0.8`. It does not apply the reference reaction-time filter and does not drop trials whose wheel/camera windows are not fully covered; those trials can survive with NaNs in interpolated behavioral traces.

ii.
```python
choice = map_choice(trials_df['choice'].to_numpy())
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The notes originally planned to "filter to valid trials explicitly" and to use trial masks / valid aligned intervals, but the implemented code only kept the simpler mask above.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The per-trial neural matrices are derived from `spikes.times.npy` and `spikes.clusters.npy` loaded from each probe's `pykilosort` directory. `clusters.metrics.pqt` is used for QC labels, and `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` are used only to assign per-unit region metadata.

ii.
```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
metrics_p = src / 'clusters.metrics.pqt'
...
st = np.load(st_p)
sc = np.load(sc_p).astype(int)
metrics = pd.read_parquet(metrics_p)
```

```python
cl_chan_p = src / 'clusters.channels.npy'
ch_brain_p = src / 'channels.brainLocationIds_ccf_2017.npy'
```

iii. The Step 5 variable mapping in `CONVERSION_NOTES.md` says the neural output comes from "`spikes.times.npy` + `spikes.clusters.npy` merged across probes within session" with region metadata carried separately.

## 2-b. How is the `neural` data processed?

i. The AI merges good clusters across probes within a session, renumbers them across probes, sorts all spikes by time, and bins spikes into 20 ms trial-relative bins from `-0.5` s to `+1.5` s around stimulus onset. The saved matrices are spike counts per bin; the code does not divide by bin width to convert them to Hz.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
```

```python
offset = sum(len(c['acronym']) for c in clusters_list)
remap[good] = np.arange(good.sum()) + offset
...
spike_times = np.concatenate([x[0] for x in spikes_list])
spike_clusters = np.concatenate([x[1] for x in spikes_list])
order = np.argsort(spike_times)
```

```python
out = np.zeros((n_neurons, N_BINS), dtype=np.float32)
...
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
...
np.add.at(out, (cl[m], bins[m]), 1)
```

iii. The notes say the conversion should use 20 ms stimulus-aligned binning and probe merging to match the reference. They also say the direct-loader rewrite preserved "session-level probe merging" while speeding up I/O.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neural QC actually applied is `label >= 1` from `clusters.metrics.pqt`. The code does not implement the reference `void`-region exclusion; it simply stringifies channel region IDs and keeps all label-passing clusters.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
...
good = labels >= 1
...
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
clusters_list.append({'acronym': acr[good]})
```

iii. `CONVERSION_NOTES.md` repeatedly justifies "strict cluster QC filter using `label >= 1`" as matching the reference code. The same notes also acknowledge that brain-region labels are only numeric region IDs as strings because atlas mapping was unavailable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to stimulus onset by taking each trial's `stimOn_times` value as `t0`, slicing spikes in `[t0 - 0.5, t0 + 1.5)`, subtracting `t0`, and histogramming the resulting relative times.

ii.
```python
for t0 in stim_on:
    edges = t0 + TIME_BINS
    ...
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
    ts = spike_times[lo:hi] - t0
```

iii. The notes say one of the main decisions was "Use 20 ms bins and stimulus-onset alignment," explicitly to match the task specification and reference caching parameters.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins over a 2 s window, giving 100 bins per trial. No later temporal rebinning is applied.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
N_BINS = len(TIME_BINS)
```

iii. Step 4 and Step 5 of the notes explicitly justify `binsize=0.02` and `time_window=(-0.5, 1.5)` from the reference workflow.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read as a raw stream. The code constructs a fixed bin-center vector from the chosen analysis window and bin size; it is only tied to the raw data through the decision to align each trial to `stimOn_times`.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

```python
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. The notes say this input is a "trial-relative time axis" repeated across trials and chosen to match the reference decoding parameters.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The input is computed as the centers of the 20 ms bins spanning `[-0.5, 1.5)`, then copied into every kept trial as the first input row.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
TIME_CENTERS.astype(np.float32)
```

iii. The Step 5 mapping table says this variable is the "common time vector across trials" rather than a separately loaded behavioral measurement.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It is the same time grid used for neural binning and behavioral interpolation. Neural spikes are binned with `TIME_BINS`, and the first input row is `TIME_CENTERS`, so the two are bin-matched by construction.

ii.
```python
edges = t0 + TIME_BINS
...
x = t0 + TIME_CENTERS
...
inp = np.vstack([TIME_CENTERS.astype(np.float32), ...])
```

iii. The notes repeatedly state that all streams should be aligned to one common stimulus-locked temporal grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the `probabilityLeft` column of the trials table. Block boundaries are inferred whenever `probabilityLeft` changes value.

ii.
```python
def trial_number_in_block(prob_left):
    out = np.zeros(len(prob_left), dtype=np.float32)
    ...
    cur = prob_left[0]
    ...
    if i == 0 or p != cur:
        cur = p
        c = 1
```

iii. The Step 5 notes explicitly map "`probabilityLeft` and trial order within block" to this decoder input.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code scans trials in order, resets the count whenever `probabilityLeft` changes, and otherwise increments within the block. The resulting value is 1-based, then broadcast across all 100 time bins of each kept trial.

ii.
```python
if i == 0 or p != cur:
    cur = p
    c = 1
else:
    c += 1
out[i] = c
```

```python
np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
```

iii. The notes say this should be computed from consecutive equal-`probabilityLeft` blocks; the final code implements that idea, but with 1-based indexing.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the `choice` column in `_ibl_trials.table.pqt`.

ii.
```python
def map_choice(vals):
    vals = np.asarray(vals)
    out = np.full(vals.shape, -1, dtype=np.int64)
    out[vals == 1] = 0   # left
    out[vals == -1] = 1  # right
    return out
```

iii. The notes explicitly say this output comes from trial-table `choice` and should be mapped "left=0, right=1" after verifying IBL's sign convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code recodes IBL's trial labels as `+1 -> 0` and `-1 -> 1`; any other value stays `-1` and is excluded by the `valid` mask. The chosen class is then repeated across all time bins of the trial.

ii.
```python
out = np.full(vals.shape, -1, dtype=np.int64)
out[vals == 1] = 0
out[vals == -1] = 1
```

```python
np.full(N_BINS, choice[tr], dtype=np.int64)
```

iii. The Step 5 mapping table says choice is a per-trial categorical output, so the code broadcasts the per-trial label over time.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the `probabilityLeft` column in the trials table.

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

iii. The notes explicitly map `probabilityLeft` to this output with the required `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2` recoding.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code only remaps the three numeric prior values to categorical indices, rejects anything else with `-1`, and broadcasts the kept category across all bins of the trial.

ii.
```python
out[np.isclose(vals, 0.2)] = 0
out[np.isclose(vals, 0.5)] = 1
out[np.isclose(vals, 0.8)] = 2
```

```python
np.full(N_BINS, prior[tr], dtype=np.int64)
```

iii. The notes describe this variable as a per-trial categorical output with no additional processing beyond recoding.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived directly from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    if not tp.exists() or not pp.exists():
        return None, None
    return np.load(tp), np.load(pp)
```

iii. The Step 5 variable mapping in the notes explicitly names those two wheel files as the source for the wheel output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The code removes non-finite samples, deduplicates repeated timestamps, computes a temporal gradient of wheel position, and linearly interpolates that gradient onto each trial's stimulus-aligned 20 ms bin centers. Despite the output name, the code does not take the absolute value, so it is effectively working with signed wheel velocity.

ii.
```python
keep = np.isfinite(timestamps) & np.isfinite(position)
timestamps = timestamps[keep]
position = position[keep]
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
timestamps = uniq_t
position = position[uniq_idx]
vel = np.gradient(position, timestamps)
```

```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
    trials.append(y.astype(np.float32))
```

iii. The notes planned to "differentiate/interpolate" the wheel trace from raw ALF files and later document a warning fix: duplicate wheel timestamps were removed before `np.gradient`.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. All finite wheel values from the kept trials of one session are pooled, the 1/3 and 2/3 quantiles are computed, and each time bin is labeled `0/1/2` based on those thresholds. Any NaN bin is forced to class `0`.

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

```python
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
```

iii. The notes say wheel and whisker outputs would be discretized into 3 bins using quantile-style thresholds chosen after checking the distributions.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel trace is sampled at `t0 + TIME_CENTERS` for each trial, so it is on the same 100-bin stimulus-locked grid as the neural matrices.

ii.
```python
x = t0 + TIME_CENTERS
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The notes justify putting all streams on one common stimulus-aligned time base.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from camera frame times plus `ROIMotionEnergy` arrays. Unlike the reference solution, the AI will use whichever of left and right exist, and if both exist it keeps both streams.

ii.
```python
left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
...
for tp, mp in [(left_t, left_me), (right_t, right_me)]:
    if tp is not None and mp is not None and tp.exists() and mp.exists():
        streams.append((np.load(tp), np.load(mp).astype(np.float32)))
```

iii. The Step 5 notes say the conversion would use "available whisker ROI motion energy stream(s)" and, if both were present, "combine left/right sensibly."

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. For each kept trial, the code interpolates every available whisker motion-energy stream to the stimulus-aligned bin centers and then averages across streams using only finite values. No extra filtering or normalization is applied.

ii.
```python
for t0 in stim_on:
    x = t0 + TIME_CENTERS
    ys = []
    for ts, me in streams:
        ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
    arr = np.stack(ys, axis=0)
    valid = np.isfinite(arr)
    denom = valid.sum(axis=0)
    summed = np.where(valid, arr, 0.0).sum(axis=0)
    y = np.divide(summed, denom, out=np.full(arr.shape[1], np.nan, dtype=np.float32), where=denom > 0)
```

iii. The notes say the AI intended to align/interpolate whisker motion energy to 20 ms bins and to combine left/right streams when appropriate; Step 10 later mentions adding explicit all-NaN handling for this averaging.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-wise tertile procedure as wheel: pool finite values from kept trials, split at the 1/3 and 2/3 quantiles, and map NaN bins to class `0`.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

```python
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

iii. The notes say whisker motion energy would be discretized into 3 bins with the same quantile-style rule as wheel.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like the wheel trace, whisker motion energy is interpolated onto `t0 + TIME_CENTERS` for each trial, so its time axis is bin-aligned with the neural data.

ii.
```python
x = t0 + TIME_CENTERS
...
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. The notes justify using one common stimulus-aligned grid for neural and behavioral outputs.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing files generally cause a whole session to be skipped (`no trials`, `no spikes`, `no wheel`, `no whisker motion energy`, or too few valid trials). Within streams, the code drops non-finite wheel samples, deduplicates repeated wheel timestamps, averages only finite whisker values, and converts any remaining NaN behavior bins into class `0` rather than dropping the corresponding trials.

ii.
```python
if wt is None:
    print('skip no wheel', session_dir)
    continue
...
if me_streams is None:
    print('skip no whisker motion energy', session_dir)
    continue
```

```python
keep = np.isfinite(timestamps) & np.isfinite(position)
...
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
```

```python
y[~np.isfinite(x)] = 0
```

iii. Step 10 of the notes explicitly records two patched edge cases: duplicate wheel timestamps and all-NaN whisker interpolation bins. The notes also acknowledge many sessions were excluded for missing wheel/whisker data.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is session-by-session disk I/O for spike arrays and metrics, followed by per-trial spike binning and behavioral interpolation. The notes also say the original ONE/object-loading version was much slower because it pulled unnecessary arrays.

ii.
```python
for probe_dir in sorted((session_dir / 'alf').glob('probe*/pykilosort')):
    ...
    st = np.load(st_p)
    sc = np.load(sc_p).astype(int)
    metrics = pd.read_parquet(metrics_p)
```

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

iii. `CONVERSION_NOTES.md` Step 6 says the initial ONE/object loader was a major slowdown and that switching to direct local file reads reduced sample conversion time by about 10x.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial loops in `bin_spikes`, `interp_wheel_speed`, `interp_motion_energy`, and the final loop that assembles per-trial input/output arrays.

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

```python
for i, tr in enumerate(np.where(valid)[0]):
    inp = np.vstack([...])
    out_trial = np.vstack([...])
```

iii. The AI did not document a vectorization pass. Its performance notes focus on faster file loading, not on rewriting the remaining per-trial loops.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly rebuilds the same trial-level structures per trial and per session: `TIME_CENTERS.astype(np.float32)`, broadcasted `np.full` arrays for per-trial labels, repeated `np.interp` calls per trial, and repeated spike-window searches per trial. It also re-globs revisioned files independently for trials and camera streams.

ii.
```python
for i, tr in enumerate(np.where(valid)[0]):
    inp = np.vstack([
        TIME_CENTERS.astype(np.float32),
        np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
    ])
    out_trial = np.vstack([
        np.full(N_BINS, choice[tr], dtype=np.int64),
        np.full(N_BINS, prior[tr], dtype=np.int64),
        ...
    ])
```

```python
for t0 in stim_on:
    lo = np.searchsorted(spike_times, edges[0], side='left')
    hi = np.searchsorted(spike_times, t0 + T_END, side='left')
```

iii. The notes do not call these out explicitly; they mainly discuss repeated heavy object loading in the earlier discarded implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several values that never enter the saved dataset: `trial_path`, parsed `lab/date/number`, logged quantile thresholds, and the optional plotting path/state. It also computes session IDs for plots/logging only. These are harmless, but they are not used by downstream decoding.

ii.
```python
trials_df, trial_path = load_trials_table(session_alf)
...
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
...
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

```python
if args.show_processing and kept < 2:
    make_plot(get_session_id(session_dir), neural_trials, wheel_trials, whisk_trials, f'processing_{kept}.png')
```

iii. The notes mention the optional processing plots and logging outputs, and they discuss speed optimizations around discarded earlier loading approaches, but not these smaller unused variables.
