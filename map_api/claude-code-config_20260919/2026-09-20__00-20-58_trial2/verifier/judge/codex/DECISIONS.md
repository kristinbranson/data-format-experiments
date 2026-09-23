# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates all NWB files under `/app/data/sub-*`, treats each file as one session, and loads each session with `pynwb.NWBHDF5IO`. Within each file it reads `nwb.trials`, `nwb.units`, `BehavioralEvents`, `BehavioralTimeSeries`, and `nwb.subject`.

ii. 
```python
def list_session_files(data_dir=DATA_DIR):
    files = []
    for sub in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, sub)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.endswith('.nwb'):
                files.append(os.path.join(d, f))
    return files
```

```python
from pynwb import NWBHDF5IO
...
io = NWBHDF5IO(path, 'r', load_namespaces=True)
nwb = io.read()
...
trials = nwb.trials
units = nwb.units
```

iii. The justification in `CONVERSION_NOTES.md` is that NWB is the published format and `pynwb` is required by the instructions. The notes also say there are 174 NWB files, one per behavioural session.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `nwb.subject.subject_id` for each session. After all sessions are processed, unique subject IDs are sorted into `subjects`, and each session gets a `subject_idx`.

ii. 
```python
subject = str(nwb.subject.subject_id)
```

```python
subjects = sorted({r['subject'] for r in results})
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The notes say `nwb.subject.subject_id` is the canonical subject identifier in the NWB files and matches the subject-level organization of the dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; no additional grouping is done. Session identity is taken from `nwb.identifier`.

ii. 
```python
files = list_session_files()
```

```python
session_id = nwb.identifier
```

iii. The notes explicitly state that the dataset contains one NWB file per behavioural session, so file boundaries are session boundaries.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI reads `nwb.trials`, keeps row order, and checks that the number of go cues equals the number of raw trials.

ii. 
```python
trials = nwb.trials
n_trials_raw = len(trials)
...
go_times = _events(nwb, 'go_start_times')
if len(go_times) != n_trials_raw:
    io.close()
    raise RuntimeError(f'{session_id}: {len(go_times)} go cues for {n_trials_raw} trials')
```

iii. The justification in the notes is that `go_start_times` provides one go cue per trial, so the trials table is the authoritative trial structure.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only observed trials from `units.obs_intervals`, then removes `auto_water` and `free_water` trials, then removes trials that are not marked good for every retained unit via `units.is_good_trials`, and finally drops trials whose binned neural activity is entirely zero. Sessions with fewer than 2 remaining trials are dropped.

ii. 
```python
obs = np.asarray(units['obs_intervals'][int(good[0])], dtype=np.float64)
obs_trial = np.searchsorted(trial_start, obs[:, 0] - 1e-6)
...
keep = (~auto_water[obs_trial]) & (~free_water[obs_trial])
is_good_trials = np.asarray(units['is_good_trials'][:])[good]
keep &= is_good_trials.all(axis=0)
keep_idx = obs_trial[keep]
...
if n_trials < 2:
    io.close()
    return None
```

```python
nonempty = fr.sum(axis=(0, 2)) > 0
if not nonempty.all():
    ...
    fr = fr[:, nonempty, :]
    keep_idx = keep_idx[nonempty]
```

iii. The notes justify this as combining reference-code curation (`auto_water`, `free_water`) with recording-validity checks (`obs_intervals`, `is_good_trials`, spike-less trials). The AI explicitly says it keeps early-lick, `ignore`, and photostim trials because the decoder task requires them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, with `BehavioralEvents/go_start_times` used to place the trial-aligned bin edges.

ii. 
```python
sv = units['spike_times']
ends = np.asarray(sv.data[:])
starts = np.concatenate([[0], ends[:-1]])
flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)
...
go = go_times[keep_idx]
```

iii. The notes say spike times are the only raw neural representation in the NWB files, so firing rates are computed directly from them.

## 2-b. How is the `neural` data processed?

i. For each retained unit, the AI bins spikes into 80 non-overlapping 50 ms bins aligned to the go cue and converts counts to firing rates in Hz by dividing by `BIN_SIZE`.

ii. 
```python
def bin_spikes_rate(spike_times, go_times):
    edges = go_times[:, None] + BIN_EDGES[None, :]
    idx = np.searchsorted(spike_times, edges.ravel(), side='left')
    idx = idx.reshape(edges.shape)
    counts = np.diff(idx, axis=1)
    return (counts / BIN_SIZE).astype(np.float32)
```

```python
fr = np.empty((n_neurons, n_trials, NBINS), dtype=np.float32)
for i, u in enumerate(good):
    st = flat_spikes[starts[u]:ends[u]]
    fr[i] = bin_spikes_rate(st, go)
```

iii. The notes justify this as matching the reference’s `sliding_histogram(..., rate=True)` except for the task-mandated 50 ms non-overlapping bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by keeping only `units.classification == 'good'`. Sessions with zero such units are dropped.

ii. 
```python
classification = np.asarray(units['classification'][:])
good = np.flatnonzero(classification == 'good')
if len(good) == 0:
    io.close()
    return None
```

iii. The notes say this column is the QC-classifier output from the spike-sorting white paper and is the intended good-unit curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the auditory go cue. The AI uses absolute go-cue timestamps and adds fixed offsets `[-2.5, 1.5]` to define each trial window.

ii. 
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

```python
edges = go_times[:, None] + BIN_EDGES[None, :]
```

iii. The notes state that all relevant NWB timestamps share a session-absolute clock, so subtracting or offsetting by the go cue is sufficient alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins, producing 80 bins from -2.5 s to +1.5 s. The AI rebins raw spike times directly onto that grid; there is no additional smoothing or resampling.

ii. 
```python
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes justify this as a deliberate departure from the paper’s 40 ms / 3.4 ms sliding bins because the decoder task explicitly requires 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` together with `go_start_times`. For each trial, the AI chooses the last sample-epoch start before the go cue.

ii. 
```python
sample_start = _events(nwb, 'sample_start_times')
tone_pos = np.searchsorted(sample_start, go, side='right') - 1
...
tone_time = sample_start[tone_pos]
```

iii. The notes justify using the last sample start because early licks can replay the sample epoch, so the last one before the go cue is the effective tone onset for that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes a continuous ramp: at each 50 ms bin center, value = `(time from go-bin center) + (go time - tone time)`.

ii. 
```python
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. The notes say this was chosen because the task explicitly asks for a continuous, time-varying time-from-tone signal.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is defined on the same 80 go-cue-aligned bin centers used for neural firing rates, one value per neural time bin.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS[None, :] + (go - tone_time)[:, None]
```

iii. The notes explicitly say all streams are aligned on the go cue and share the same bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from the trial-table string columns.

ii. 
```python
ps_start = _events(nwb, 'photostim_start_times')
ps_stop = _events(nwb, 'photostim_stop_times')
```

iii. The notes justify this as using the actual event timestamps on the shared session clock, then re-expressing them relative to each trial’s go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI makes a binary time series over the 80 bins. A bin is set to 1 if the photostim interval overlaps that bin by more than `OVERLAP_TOL = 1e-3` seconds; otherwise 0.

ii. 
```python
OVERLAP_TOL = 1e-3
...
def interval_overlap_bins(starts, stops, go_time):
    on = np.zeros(NBINS, dtype=np.float32)
    ...
    for a, b in zip(starts - go_time, stops - go_time):
        ...
        ov = np.minimum(b, BIN_EDGES[1:]) - np.maximum(a, BIN_EDGES[:-1])
        on[ov > OVERLAP_TOL] = 1.0
    return on
```

```python
for j, g in enumerate(go):
    m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
    if m.any():
        photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```

iii. The notes say a naive “touches the bin” rule produced spurious activation at bin boundaries, so the overlap tolerance was added after review.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim is aligned by comparing absolute photostim event times to each trial’s absolute go cue, then rasterizing onto the same go-cue-relative 50 ms bins as the neural data.

ii. 
```python
for j, g in enumerate(go):
    m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
    if m.any():
        photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```

iii. The notes justify this by the shared NWB session clock across neural, behavioural, and photostim event streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from a single raw variable. The AI derives it from the trial-table columns `trial_instruction` and `outcome`.

ii. 
```python
instruction = np.asarray(trials['trial_instruction'][:])
outcome = np.asarray(trials['outcome'][:])
```

```python
instr_k = instruction[keep_idx]
out_k = outcome[keep_idx]
```

iii. The notes say hit means the instructed side, miss means the opposite side, and ignore means no lick, so these two columns fully determine the choice label.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0 = left`, `1 = right`, `2 = no lick`, then broadcasts that per-trial label across all 80 time bins.

ii. 
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
...
choice = np.full(n_trials, CHOICE_NOLICK, dtype=np.int64)
hit = out_k == 'hit'
miss = out_k == 'miss'
choice[hit & (instr_k == 'left')] = CHOICE_LEFT
choice[hit & (instr_k == 'right')] = CHOICE_RIGHT
choice[miss & (instr_k == 'left')] = CHOICE_RIGHT
choice[miss & (instr_k == 'right')] = CHOICE_LEFT
```

```python
outputs = [np.stack([np.full(NBINS, choice[j]),
                     np.full(NBINS, outcome_code[j]),
                     np.full(NBINS, early_code[j]),
                     tongue_class[j]]).astype(np.int64)
           for j in range(n_trials)]
```

iii. The notes justify broadcasting because the target format uses a single `(n_output, n_timepoints)` array per trial and the first three outputs are per-trial variables.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii. 
```python
outcome = np.asarray(trials['outcome'][:])
...
out_k = outcome[keep_idx]
```

iii. The notes say the raw NWB trial table already contains the required `ignore`, `miss`, and `hit` categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, `hit -> 2` and repeats that code across all 80 bins.

ii. 
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_code = np.array([OUTCOME_CODE[o] for o in out_k], dtype=np.int64)
```

```python
outputs = [np.stack([np.full(NBINS, choice[j]),
                     np.full(NBINS, outcome_code[j]),
                     np.full(NBINS, early_code[j]),
                     tongue_class[j]]).astype(np.int64)
           for j in range(n_trials)]
```

iii. The notes say this follows the task specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii. 
```python
early_lick = np.asarray(trials['early_lick'][:])
...
early_k = early_lick[keep_idx]
```

iii. The notes describe this as an explicit trial-table flag for whether the animal licked during the sample or delay epoch.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `early` to 1 and `no early` to 0, then repeats that per-trial value across the 80 bins.

ii. 
```python
early_code = (early_k == 'early').astype(np.int64)
```

```python
outputs = [np.stack([np.full(NBINS, choice[j]),
                     np.full(NBINS, outcome_code[j]),
                     np.full(NBINS, early_code[j]),
                     tongue_class[j]]).astype(np.int64)
           for j in range(n_trials)]
```

iii. The notes say this is another per-trial variable that is broadcast to fit the common output array shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as `y` and column 2 as the tracking likelihood, together with the frame timestamps.

ii. 
```python
ts_obj = bts.time_series['Camera0_side_TongueTracking']
frame_t = np.asarray(ts_obj.timestamps[:], dtype=np.float64)
data = np.asarray(ts_obj.data[:, 1:3], dtype=np.float64)
y = data[:, 0]
visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
```

iii. The notes justify this as the available tongue tracking stream in the NWB files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first marks frames with likelihood `<= 0.9` as not visible, assigns visible frames to trials, keeps only visible frames within the `[-2.5, 1.5]` window around each go cue, averages `y` within each `(trial, bin)` using `np.bincount`, and then discretizes those per-bin means per session.

ii. 
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
...
visible = data[:, 1] > TONGUE_LIKELIHOOD_THRESH
```

```python
frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1
...
rel = frame_t[vis] - go[pos]
inwin = (rel >= OFF_START) & (rel < OFF_END)
...
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
...
sums = np.bincount(flat, weights=y[vis], minlength=n_cells)
cnts = np.bincount(flat, minlength=n_cells)
...
out = mean.reshape(n_trials, NBINS)
```

iii. The notes justify the higher visibility threshold by saying the likelihood distribution is strongly bimodal, and justify per-bin averaging because the decoder output is bin-wise and the instructions require per-session discretization.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After computing per-bin visible tongue-y means, the AI takes the 40th and 60th percentiles over all visible retained `(trial, bin)` values in the session. It assigns 0 below the 40th percentile, 1 between the percentiles, 2 above the 60th percentile, and 3 when no visible frame exists.

ii. 
```python
def discretise_tongue(tongue_y):
    cls = np.full(tongue_y.shape, 3, dtype=np.int64)
    vis = ~np.isnan(tongue_y)
    if vis.sum() == 0:
        return cls, (np.nan, np.nan)
    vals = tongue_y[vis]
    p_lo = np.percentile(vals, TONGUE_LOW_PCT)
    p_hi = np.percentile(vals, TONGUE_HIGH_PCT)
    v = tongue_y[vis]
    c = np.ones(v.shape, dtype=np.int64)
    c[v < p_lo] = 0
    c[v > p_hi] = 2
    cls[vis] = c
    return cls, (float(p_lo), float(p_hi))
```

iii. The notes justify this as matching the task’s required 40/60 session-percentile discretization while keeping “not visible” as a separate class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue frames to the neural bins by assigning each frame to its trial, converting frame timestamps to time relative to the trial’s go cue, and then binning onto the same 50 ms `[-2.5, 1.5]` grid as the neural data.

ii. 
```python
frame_trial = np.searchsorted(trial_start, frame_t, side='right') - 1
...
rel = frame_t[vis] - go[pos]
inwin = (rel >= OFF_START) & (rel < OFF_END)
...
bin_idx = np.floor((rel - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. The notes justify the explicit frame-to-trial assignment by saying it prevents a window that extends outside its own trial from accidentally absorbing frames from a neighbouring trial.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly: sessions with no good units are dropped; trials outside `obs_intervals` are excluded; trials with all-zero neural data are dropped; missing tongue series yields all-NaN tongue bins; bins with no visible tongue frame become class 3; missing CCF x falls back to the probe target hemisphere; mismatched photostim start/stop counts produce a warning and a zero photostim input.

ii. 
```python
good = np.flatnonzero(classification == 'good')
if len(good) == 0:
    io.close()
    return None
```

```python
bts = nwb.acquisition.get('BehavioralTimeSeries', None)
if bts is None or 'Camera0_side_TongueTracking' not in bts.time_series:
    return out, 0.0
```

```python
missing = np.isnan(elec_x)
if missing.any():
    hemisphere[missing] = np.array([t.split(' ')[0] for t in probe_target])[missing]
```

```python
elif len(ps_start) != len(ps_stop):
    print(f'  WARNING {session_id}: photostim start/stop counts differ '
          f'({len(ps_start)}/{len(ps_stop)}); photostim input left at 0', flush=True)
```

iii. The notes repeatedly justify this as preferring to drop clearly invalid recordings, while representing legitimate absence of a measurement, such as tongue invisibility, as an explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is loading large NWB arrays, especially flattened spike times and tongue tracking data, then looping over units to bin spikes. Full-dataset pickling is also a substantial cost.

ii. 
```python
sv = units['spike_times']
ends = np.asarray(sv.data[:])
flat_spikes = np.asarray(sv.target.data[:], dtype=np.float64)
...
for i, u in enumerate(good):
    st = flat_spikes[starts[u]:ends[u]]
    fr[i] = bin_spikes_rate(st, go)
```

```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The notes explicitly say I/O dominates runtime, with spike-time reads, tongue arrays, per-unit `searchsorted`, and final pickle writing as the main costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious Python loops are the per-unit loop used to bin spikes and the per-trial loop used to rasterize photostimulation intervals. The code already vectorizes across trial bins and uses `np.bincount` for tongue aggregation.

ii. 
```python
for i, u in enumerate(good):
    st = flat_spikes[starts[u]:ends[u]]
    fr[i] = bin_spikes_rate(st, go)
```

```python
for j, g in enumerate(go):
    m = (ps_stop > g + OFF_START) & (ps_start < g + OFF_END)
    if m.any():
        photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```

iii. The notes justify leaving the per-unit spike loop because spike trains are ragged per unit. They claim the major loops were already vectorized where it mattered.

## 10-c. What processing does the code repeat multiple times?

i. It repeats a few small operations rather than whole conversion passes: interval-overlap tests are repeated trial by trial for photostim; constant output labels are repeatedly expanded to length-80 arrays trial by trial; and subject indexing uses repeated `subjects.index(...)` lookups at assembly time.

ii. 
```python
for j, g in enumerate(go):
    ...
    photostim_on[j] = interval_overlap_bins(ps_start[m], ps_stop[m], g)
```

```python
outputs = [np.stack([np.full(NBINS, choice[j]),
                     np.full(NBINS, outcome_code[j]),
                     np.full(NBINS, early_code[j]),
                     tongue_class[j]]).astype(np.int64)
           for j in range(n_trials)]
```

```python
subject_idx = np.array([subjects.index(r['subject']) for r in results], dtype=np.int64)
```

iii. The notes argue that the conversion is effectively a single pass over the data and does not recompute major derived quantities multiple times.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Beyond the decoder-facing arrays, the code computes and stores extra metadata and diagnostics that are not used by the downstream decoder, such as hemisphere labels, tongue percentiles, visible-frame fractions, timing breakdowns, and optional debug payloads for plotting. It also reads some values mainly for diagnostics and plots.

ii. 
```python
result = dict(
    ...
    hemisphere=np.where(hemisphere == 'left', 0, 1).astype(np.int8),
    ...
    tongue_percentiles=pct,
    tongue_visible_frac=float(tongue_visible_frac),
    timing=timing,
    total_time=time.time() - t0,
)
if collect_debug:
    result['debug'] = dict(
        go=go, tone_time=tone_time, trial_start=trial_start[keep_idx],
        trial_stop=trial_stop[keep_idx], time_from_tone=time_from_tone,
        ...
    )
```

iii. The notes justify these extras as sanity-checking and documentation support rather than core decoder inputs or outputs.
