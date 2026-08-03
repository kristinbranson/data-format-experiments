# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the reference ONE/`SessionLoader`/`SpikeSortingLoader` workflow. It scanned the local `data/one_cache` filesystem for session directories, then read the latest revision of each needed file directly from disk with `pandas.read_parquet` and `numpy.load`. For each kept session it loaded the trials table, per-probe spike files, wheel files, and whichever camera motion-energy files existed.

ii. 
```python
def list_session_dirs(data_root: Path):
    sessions = []
    for p in data_root.glob('*/Subjects/*/*/*'):
        if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
            sessions.append(p)
    return sorted(sessions)
```

```python
def load_trials_table(session_alf: Path):
    pqt = find_latest_file(session_alf, '#*/_ibl_trials.table.pqt')
    ...
    return pd.read_parquet(pqt), pqt
```

```python
spikes, clusters = load_spikes_for_session_dir(session_dir)
wt, wp = load_wheel(session_alf)
me_streams = load_motion_energy(session_alf)
```

iii. The justification documented in `CONVERSION_NOTES.md` is speed and avoiding unnecessary I/O: the agent says it replaced ONE-based loading with direct local ALF reads and got about a 10x sample speedup.

## 1-b. How are the data split into subjects?

i. Subject IDs are taken from the session path, not from ONE metadata. The script then assigns each distinct subject a running integer index as sessions are appended.

ii.
```python
def get_subject_from_session(session_dir: Path):
    return session_dir.parts[-3]
```

```python
rel = session_dir.relative_to(data_root)
parts = rel.parts
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
...
subj = subject
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
out['subject_idx'].append(subject_to_idx[subj])
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly planned to use “subject/session path metadata” for `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. A session is any directory matching `*/Subjects/*/*/*` whose final component is numeric and which contains an `alf/` directory. Candidate sessions only need a trials table initially; later the loop skips sessions missing spikes, wheel, or whisker data.

ii.
```python
for p in data_root.glob('*/Subjects/*/*/*'):
    if p.is_dir() and p.name.isdigit() and (p / 'alf').exists():
        sessions.append(p)
```

```python
def choose_sessions(session_dirs, mode):
    valid = []
    for s in session_dirs:
        alf = s / 'alf'
        if find_latest_file(alf, '#*/_ibl_trials.table.pqt') or (alf / '_ibl_trials.table.pqt').exists():
            valid.append(s)
```

iii. The notes describe the dataset as a local ONE cache on disk and frame the conversion as a “local-cache conversion script,” which explains the directory-based session definition.

## 1-d. How are the data split into trials?

i. Trials are taken directly from rows of the trials table. The code creates per-trial arrays by indexing the trial-wise columns and iterating over the surviving trial indices.

ii.
```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
choice = map_choice(trials_df['choice'].to_numpy())
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
trial_in_block = trial_number_in_block(trials_df['probabilityLeft'].to_numpy())
```

```python
for i, tr in enumerate(np.where(valid)[0]):
    ...
    sess_inputs.append(inp)
    sess_outputs.append(out_trial)
```

iii. This follows the agent’s notes that session-level processing should operate from the IBL trials table.

## 1-e. How are trials filtered based on quality controls?

i. The final code uses a very light trial mask: keep trials with finite `stimOn_times`, mappable `choice`, and mappable `probabilityLeft`. It does not use reaction-time filtering, `firstMovement_times`, or wheel/camera coverage checks. Sessions with fewer than two surviving trials are skipped.

ii.
```python
valid = np.isfinite(stim_on) & (choice >= 0) & (prior >= 0)
if valid.sum() < 2:
    print('skip too few valid trials', session_dir)
    continue
```

iii. The notes had planned “explicit valid-trial mask” logic, but the trajectory shows the agent ultimately focused on handling NaNs inside interpolation and discretization rather than dropping edge trials with incomplete behavioral coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are built from `spikes.times.npy` and `spikes.clusters.npy` for each probe. Cluster metadata are additionally read from `clusters.metrics.pqt` for QC and from `clusters.channels.npy` plus `channels.brainLocationIds_ccf_2017.npy` for region labels.

ii.
```python
st_p = src / 'spikes.times.npy'
sc_p = src / 'spikes.clusters.npy'
metrics_p = src / 'clusters.metrics.pqt'
...
cl_chan_p = src / 'clusters.channels.npy'
ch_brain_p = src / 'channels.brainLocationIds_ccf_2017.npy'
```

iii. `CONVERSION_NOTES.md` Step 5 maps `spikes.times.npy` and `spikes.clusters.npy` to the target `neural` field and treats cluster metadata as supporting QC/region information.

## 2-b. How is the `neural` data processed?

i. Good clusters from all probes in a session are merged into one population with offset cluster IDs, spikes are sorted by time, and per-trial spike counts are accumulated into 20 ms bins from -0.5 s to 1.5 s around stimulus onset. The stored matrices are counts cast to `float32`; they are not converted to firing rates in Hz.

ii.
```python
offset = sum(len(c['acronym']) for c in clusters_list)
remap[good] = np.arange(good.sum()) + offset
...
order = np.argsort(spike_times)
spike_times = spike_times[order]
spike_clusters = spike_clusters[order]
```

```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
m = (bins >= 0) & (bins < N_BINS) & (cl >= 0) & (cl < n_neurons)
np.add.at(out, (cl[m], bins[m]), 1)
```

iii. The notes justify 20 ms binning, stimulus-onset alignment, and probe merging as matching the reference workflow; they do not justify leaving the result as counts instead of Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only clusters with `label >= 1` are kept. Spikes assigned to invalid cluster indices are also dropped.

ii.
```python
labels = metrics['label'].to_numpy() if 'label' in metrics.columns else np.ones(nclu)
good = labels >= 1
...
keep_spk = (sc >= 0) & (sc < nclu)
sc2 = sc[keep_spk]
st2 = st[keep_spk]
keep2 = good[sc2]
spikes_list.append((st2[keep2], remap[sc2[keep2]]))
```

iii. `CONVERSION_NOTES.md` repeatedly cites the reference practice of strict QC using `label >= 1`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to its `stimOn_times` entry. For a given trial, the code slices spikes in the absolute window `[stimOn-0.5, stimOn+1.5)` and subtracts the trial’s stimulus onset before binning.

ii.
```python
lo = np.searchsorted(spike_times, edges[0], side='left')
hi = np.searchsorted(spike_times, t0 + T_END, side='left')
ts = spike_times[lo:hi] - t0
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

iii. The notes explicitly say the conversion should use 20 ms bins aligned to `stimOn_times`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins, giving 100 bins over the 2 s window. No extra rebinning is applied after this.

ii.
```python
BINSIZE = 0.02
T_START = -0.5
T_END = 1.5
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
N_BINS = len(TIME_BINS)
```

iii. The notes and README both justify 20 ms binning as the intended reference configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is not read from a dedicated raw variable. The time axis is defined from the fixed binning constants and used relative to each trial’s `stimOn_times`.

ii.
```python
TIME_BINS = np.arange(T_START, T_END, BINSIZE)
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

```python
stim_on = np.asarray(trials_df['stimOn_times'], dtype=float)
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. `CONVERSION_NOTES.md` Step 5 planned to “repeat common time vector across trials” as the first decoder input.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code constructs a common vector of 100 bin centers from -0.49 s to 1.49 s and repeats that same vector for every trial as the first input row.

ii.
```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
...
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

iii. The notes describe this as a simple representation choice for the decoder input rather than a measurement extracted from the raw files.

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. It uses the same 20 ms grid that neural spikes are binned onto, so the first input row and the neural matrices are bin-aligned by construction.

ii.
```python
bins = np.floor((ts - T_START) / BINSIZE).astype(int)
```

```python
TIME_CENTERS = TIME_BINS + BINSIZE / 2
```

iii. The notes justify stimulus-onset alignment and a common time base for all streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the trial-table `probabilityLeft` sequence. A change in `probabilityLeft` starts a new block.

ii.
```python
def trial_number_in_block(prob_left):
    ...
    cur = prob_left[0]
    c = 0
    for i, p in enumerate(prob_left):
        if i == 0 or p != cur:
            cur = p
            c = 1
        else:
            c += 1
        out[i] = c
```

iii. `CONVERSION_NOTES.md` Step 5 says trial number in block should come from “`probabilityLeft` and trial order within block.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The script counts consecutive trials within each run of constant `probabilityLeft`, starting at 1 for the first trial of each block. The count is computed before trial filtering and then broadcast across all 100 time bins of each kept trial.

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

iii. The notes only justify the source variable and the idea of using within-block trial order. Sample statistics in the notes show a range starting at 1, consistent with the one-based implementation.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. `Choice` is derived from the trials-table `choice` column.

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
choice = map_choice(trials_df['choice'].to_numpy())
```

iii. The notes explicitly planned to verify the IBL sign convention and map left/right onto 0/1.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The only processing is recoding `choice == 1` to 0 (left) and `choice == -1` to 1 (right); anything else remains `-1` and is filtered out. The categorical value is then repeated across time bins for each kept trial.

ii.
```python
out = np.full(vals.shape, -1, dtype=np.int64)
out[vals == 1] = 0
out[vals == -1] = 1
```

```python
np.full(N_BINS, choice[tr], dtype=np.int64)
```

iii. This matches the mapping described in the notes.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from the trials-table `probabilityLeft` column.

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

```python
prior = map_prior(trials_df['probabilityLeft'].to_numpy())
```

iii. `CONVERSION_NOTES.md` Step 5 planned exactly this 0.2/0.5/0.8 to 0/1/2 mapping.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The code maps the three allowed prior values onto category IDs 0, 1, and 2, filters out anything else, and repeats the resulting category across the full trial window.

ii.
```python
out[np.isclose(vals, 0.2)] = 0
out[np.isclose(vals, 0.5)] = 1
out[np.isclose(vals, 0.8)] = 2
```

```python
np.full(N_BINS, prior[tr], dtype=np.int64)
```

iii. The justification in the notes is simply that this is the task-specified categorical mapping.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
def load_wheel(session_alf: Path):
    tp = session_alf / '_ibl_wheel.timestamps.npy'
    pp = session_alf / '_ibl_wheel.position.npy'
    ...
    return np.load(tp), np.load(pp)
```

iii. The notes map wheel position and timestamps to the wheel-speed output.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The script drops non-finite samples, deduplicates timestamps, computes a signed velocity with `np.gradient(position, timestamps)`, and linearly interpolates that velocity onto each trial’s 100 stimulus-aligned bin centers. It does not take the absolute value and does not reproduce `SessionLoader`’s 1 kHz interpolation plus filtered velocity calculation.

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

iii. The notes justify differentiating and interpolating the raw wheel trace, and the trajectory later justifies the timestamp deduplication as a fix for numerical warnings.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation, all finite wheel values from the session’s kept trials are concatenated. The 1/3 and 2/3 quantiles are used as thresholds: `<= q1` maps to 0, `(q1, q2]` to 1, and `> q2` to 2. Non-finite values are forced to 0.

ii.
```python
allv = np.concatenate([x[np.isfinite(x)] for x in list_of_arrays if np.isfinite(x).any()])
q1, q2 = np.quantile(allv, [1/3, 2/3]) if len(allv) else (0.0, 1.0)
...
y = np.zeros_like(x, dtype=np.int64)
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

```python
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent chose “global/session-aware quantile-style thresholds to avoid degenerate classes,” and the trajectory logs the per-session thresholds.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are sampled at `t0 + TIME_CENTERS` for each trial, so they live on the same 100-bin stimulus-aligned grid as the neural matrices.

ii.
```python
x = t0 + TIME_CENTERS
y = np.interp(x, timestamps, vel, left=np.nan, right=np.nan)
```

iii. The notes justify using a common 20 ms, stimulus-aligned time base for all streams.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The script loads whichever of the left and right camera streams exist: `_ibl_leftCamera.times.npy` with `leftCamera.ROIMotionEnergy.npy`, and `_ibl_rightCamera.times.npy` with `rightCamera.ROIMotionEnergy.npy`.

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

iii. `CONVERSION_NOTES.md` Step 5 says to use the available whisker ROI motion-energy stream(s) and “combine left/right sensibly if both available.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. For each trial, the code interpolates every available camera stream to the common stimulus-aligned bin centers and averages the finite values across streams at each bin. No extra filtering or normalization is applied before discretization.

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

iii. The trajectory later explains the explicit finite-value averaging as a fix for all-NaN warnings; the notes justify combining left/right streams if both are present.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-wide tertile discretization as wheel speed: thresholds are the 1/3 and 2/3 quantiles of all finite interpolated whisker values from kept trials, with non-finite bins forced to category 0.

ii.
```python
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

```python
y = np.zeros_like(x, dtype=np.int64)
y[x > q1] = 1
y[x > q2] = 2
y[~np.isfinite(x)] = 0
```

iii. The notes say both continuous outputs would be discretized with quantile-style thresholds, and the trajectory logs the resulting per-session whisker thresholds.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The camera trace is interpolated at the same `t0 + TIME_CENTERS` query points used for other time-varying streams, so it shares the neural time axis bin-for-bin.

ii.
```python
x = t0 + TIME_CENTERS
ys.append(np.interp(x, ts[:len(me)], me, left=np.nan, right=np.nan))
```

iii. The notes justify a common 20 ms, stimulus-aligned grid across neural and behavioral streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly skips whole sessions or fills through edge cases rather than dropping individual bad trials. It skips sessions missing trial columns, spikes, wheel, whisker, or at least two valid trials; deduplicates wheel timestamps; averages whisker streams only over finite values; assigns non-finite discretized bins to 0; and catches any session-level exception and skips that session.

ii.
```python
if 'stimOn_times' not in trials_df.columns or 'choice' not in trials_df.columns or 'probabilityLeft' not in trials_df.columns:
    print('skip missing required trial columns', session_dir)
    continue
...
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

```python
uniq_t, uniq_idx = np.unique(timestamps, return_index=True)
...
y[~np.isfinite(x)] = 0
...
except Exception as e:
    print('skip session due to error', session_dir, repr(e))
```

iii. The trajectory explicitly documents the wheel timestamp deduplication and whisker all-NaN handling as fixes added after seeing warnings in full conversion.

## 10-a. What are the most time-consuming steps of the code?

i. The agent’s notes identify data loading as the main bottleneck, especially the earlier ONE-based version that loaded unnecessary spike/template/waveform arrays. In the final code, the expensive work is still per-session file I/O for spike files and the trial-wise binning/interpolation that follows.

ii.
```python
st = np.load(st_p)
sc = np.load(sc_p).astype(int)
metrics = pd.read_parquet(metrics_p)
```

```python
for t0 in stim_on:
    ...
    np.add.at(out, (cl[m], bins[m]), 1)
```

iii. `CONVERSION_NOTES.md` Step 6 says switching to direct local ALF reads reduced sample runtime from about 27 s to about 3 s for two sessions, so speed was a documented concern.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain scalar or per-trial: the block-count loop, the per-trial spike-binning loop, the per-trial wheel interpolation loop, the per-trial whisker interpolation loop, the per-trial input/output assembly loop, and the brain-region indexing loop.

ii.
```python
for i, p in enumerate(prob_left):
    ...
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

```python
for i, tr in enumerate(np.where(valid)[0]):
    ...
```

iii. The notes discuss speedups from reducing file I/O, but they do not document any stronger justification for leaving these loops unvectorized.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly casts or reconstructs per-trial constants during assembly, performs separate trial-wise interpolation passes for wheel and whisker, repeatedly scans for latest revision files per modality, and loops over `np.where(valid)[0]` even after building aligned trial arrays.

ii.
```python
inp = np.vstack([
    TIME_CENTERS.astype(np.float32),
    np.full(N_BINS, trial_in_block[tr], dtype=np.float32)
])
```

```python
left_t = find_latest_file(session_alf, '#*/_ibl_leftCamera.times.npy')
right_t = find_latest_file(session_alf, '#*/_ibl_rightCamera.times.npy')
left_me = find_latest_file(session_alf, '#*/leftCamera.ROIMotionEnergy.npy')
right_me = find_latest_file(session_alf, '#*/rightCamera.ROIMotionEnergy.npy')
```

iii. No explicit justification was documented beyond a general focus on getting the conversion fast enough and structurally valid.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some extra work that is not kept in the final dataset: it imports unused modules, parses unused path fields, returns an unused `trial_path`, optionally makes plots only for inspection, and for whisker/wheel it computes continuous interpolated traces even though only the discretized categories are ultimately saved.

ii.
```python
from collections import defaultdict
...
from one.api import ONE
```

```python
trials_df, trial_path = load_trials_table(session_alf)
...
lab, _, subject, date, number = parts[0], parts[1], parts[2], parts[3], parts[4]
```

```python
wheel_trials = interp_wheel_speed(wt, wp, stim_on[valid])
whisk_trials = interp_motion_energy(me_streams, stim_on[valid])
wheel_bins, wheel_thr = discretize_tertiles(wheel_trials)
whisk_bins, whisk_thr = discretize_tertiles(whisk_trials)
```

iii. The notes justify the optional plots for process inspection, but otherwise do not give an explicit reason for the extra discarded work.
