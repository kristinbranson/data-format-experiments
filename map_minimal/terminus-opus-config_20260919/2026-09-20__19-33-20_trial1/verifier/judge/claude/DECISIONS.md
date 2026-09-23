# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is distributed as one NWB file per session under `data/sub-<subject_id>/`. All session files are discovered with a glob, and each file is opened with `h5py` (not `pynwb`) and processed. The AI first reads a lightweight `session_info()` to decide whether the session passes behavioral criteria, then re-opens the file for full conversion. Subjects, trials, and units are read from the HDF5 groups (`intervals/trials`, `units`, `acquisition/BehavioralEvents`).

ii. Finding all data:
```python
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

Reading session info:
```python
def session_info(path):
    with h5py.File(path, 'r') as f:
        t = f['intervals/trials']
        info = dict(
            path=path,
            subject=f['general/subject/subject_id'][()].decode(),
            session_id=f['identifier'][()].decode(),
            ...
        )
```

Full processing:
```python
def process_session(path):
    info = session_info(path)
    used, perf, ncl, ncr = session_is_used(info)
    if not used:
        return None
    with h5py.File(path, 'r') as f:
        ...
```

iii. The agent noted "Data is organized by subject in NWB-like directories" and "174 NWB sessions, 28 mice." The agent chose `h5py` over `pynwb` for direct low-level access to the HDF5 structure. Sessions are processed in parallel using `multiprocessing.Pool`.

## 1-b. How are the data split into subjects?

i. Each NWB file records its animal in `general/subject/subject_id`. That value is read for every session and carried through to assembly, where `subjects` is the list of unique ids (in order of first appearance) and `subject_idx` gives each session's index into that list.

ii. Per session:
```python
subject=f['general/subject/subject_id'][()].decode(),
```

At assembly:
```python
if s['subject'] not in subjects:
    subjects.append(s['subject'])
subject_idx.append(subjects.index(s['subject']))
```

iii. The agent identified that subjects are identified by numeric IDs (e.g., `'440956'`). After session-level behavioral filtering, 25 subjects remain (from 28 total).

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Each session is identified by `nwb.identifier` (read as `f['identifier'][()].decode()`). After applying behavioral performance criteria, 105 sessions are retained (106 pass behavioral criteria, but one has no good units).

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
session_id=f['identifier'][()].decode(),
```

iii. The agent noted "174 sessions total; 106 pass the data paper's criteria (>65% performance on control non-early-lick trials, >=50 correct left and right); 105 are converted (one has no QC-passing unit)."

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. Go-cue times are matched to trials via `searchsorted` on trial start times.

ii.
```python
t = f['intervals/trials']
start=np.asarray(t['start_time'][:]),
stop=np.asarray(t['stop_time'][:]),
```

```python
def go_cue_times(f, start, stop):
    go = _event_times(f, 'go_start_times')
    idx = np.searchsorted(start, go, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, g in zip(idx, go):
        if 0 <= i < len(out) and np.isnan(out[i]):
            out[i] = g
    return out
```

iii. The agent matched go cue events to trials using their temporal position relative to trial start times.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on multiple criteria: (1) must have a valid go cue and tone onset (not NaN), (2) must not be free_water or auto_water trials, (3) must be within the units' observation intervals, and (4) trials with all-zero neural data across all units are removed. A session must have at least 2 trials after filtering. Additionally, sessions are pre-filtered based on behavioral performance (>65% correct on control non-early-lick trials, >=50 correct left and right).

ii.
```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
trials = np.where(keep)[0]
if len(trials) < 2:
    return None
```

```python
# a handful of trials at the very end of a recording contain no spikes
nonempty = rates.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    trials = trials[nonempty]
    rates = rates[:, nonempty, :]
```

Session-level behavioral filter:
```python
def session_is_used(info):
    ctrl = (info['photostim_onset'] == 'N/A') & (info['early'] == 'no early')
    perf = float(np.mean(info['outcome'][ctrl] == 'hit'))
    ncl = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'left')))
    ncr = int(np.sum(ctrl & (info['outcome'] == 'hit') & (info['instruction'] == 'right')))
    used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
            and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
    return used, perf, ncl, ncr
```

iii. The agent stated: "Session criteria from data paper (>65% performance, >=50 correct L and R) gives exactly 106 sessions." For trial filtering: "all task trials, including photostimulation, early-lick, error and no-response trials (they are decoder inputs/outputs here), excluding free-water/auto-water trials and trials outside the ephys observation intervals."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times`, the sorted spike times of each unit. Only units with `classification == 'good'` contribute. Go-cue times are used to place the bin edges.

ii.
```python
u = f['units']
good = np.where(u['classification'][:] == b'good')[0]
sti = np.asarray(u['spike_times_index'][:])
...
for k, iu in enumerate(good):
    a = 0 if iu == 0 else sti[iu - 1]
    spikes = np.asarray(u['spike_times'][a:sti[iu]])
```

iii. The agent confirmed spike_times is the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to firing rates in Hz. For each good unit, bin edges for all trials are constructed relative to the go cue, `searchsorted` gives spike counts per bin, and counts are divided by bin width to give Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
...
flat_edges = edges.ravel()
for k, iu in enumerate(good):
    a = 0 if iu == 0 else sti[iu - 1]
    spikes = np.asarray(u['spike_times'][a:sti[iu]])
    if spikes.size and np.any(np.diff(spikes) < 0):
        spikes = np.sort(spikes)
    pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
    rates[k] = np.diff(pos, axis=1).astype(np.float32) / BIN_SIZE
```

iii. The agent chose the same searchsorted-based binning approach as the reference, converting spike counts to Hz by dividing by the 50ms bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == b'good'` are kept. A session with no good units is dropped entirely. No additional metric thresholds are applied.

ii.
```python
good = np.where(u['classification'][:] == b'good')[0]
```

And in `session_info`:
```python
ngood=int((f['units']['classification'][:] == b'good').sum()),
```

iii. The agent confirmed: "units labelled 'good' by the region-specific quality-control classifiers of the accompanying spike-sorting white paper (units/classification in the NWB files), as used in both reference papers." Total good units: 41,197 across 105 sessions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are constructed relative to the go cue for each trial. Since spike times and go-cue times are on the same session-absolute clock, no separate alignment step is needed.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
```

iii. The agent noted that all NWB timestamps share a global session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. The bin grid is defined once and reused for every trial.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The window and 50 ms bin width are specified in the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, the tone onsets of the session, together with the go cue of each trial. The tone taken for a trial is the last one before its go cue.

ii.
```python
def tone_onset_times(f, start, go):
    samp = _event_times(f, 'sample_start_times')
    idx = np.searchsorted(start, samp, 'right') - 1
    out = np.full(len(start), np.nan)
    for i, s in zip(idx, samp):
        if i < 0 or i >= len(out) or np.isnan(go[i]) or s > go[i]:
            continue
        if np.isnan(out[i]) or s > out[i]:
            out[i] = s
    return out
```

iii. The agent noted: "Licking during the sample/delay epoch triggers a replay of the epoch, so a trial can contain several sample-epoch onsets; we use the last one before the go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the difference between the bin center times and the tone onset time for that trial.

ii.
```python
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

Where `centers` are the absolute times of bin centers (not relative):
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
```

iii. Since bin centers are absolute times and tone onsets are absolute times, the difference gives seconds from tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers used for the time-from-tone computation are derived from the same go-cue-aligned edges used for the neural data, so they are inherently aligned.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. Same grid for both neural and input data ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_start_times` and `photostim_stop_times` in `BehavioralEvents`, which provide session-absolute timestamps of photostimulation onset and offset.

ii.
```python
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
```

iii. The agent chose to use the event-based timestamps rather than the trials table fields (`photostim_onset`/`photostim_duration`), reasoning that event times provide precise session-absolute timestamps.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its interval overlaps with any photostim start/stop pair, and 0 otherwise. The overlap check uses bin edges (not centers).

ii.
```python
stim_on = np.zeros((len(trials), N_BINS), dtype=np.float32)
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
for s_, e_ in zip(ps, pe):
    ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
    stim_on[ov] = 1.0
```

iii. The overlap-based approach marks a bin as 1 if any part of it overlaps with the photostim interval.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim check uses the same go-cue-aligned bin edges as the neural data, ensuring alignment.

ii.
```python
ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
```

iii. Same bin grid for all data streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` ('left'/'right') and `outcome` ('hit'/'miss'/'ignore') from the trials table. There is no direct choice column.

ii.
```python
choice = np.full(len(trials), 2, dtype=np.int64)  # 2 = no lick
hit = outcome == 'hit'
miss = outcome == 'miss'
choice[hit & (instruction == 'left')] = 0
choice[hit & (instruction == 'right')] = 1
choice[miss & (instruction == 'left')] = 1
choice[miss & (instruction == 'right')] = 0
```

iii. The agent noted: "A 'hit' means it licked the instructed port, a 'miss' the opposite one, and an 'ignore' trial means the animal did not lick."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick, and repeated across all 80 time bins.

ii.
```python
outputs = np.stack([np.repeat(choice[:, None], N_BINS, axis=1),
                    ...], axis=1)
```

iii. Per-trial values are repeated across bins so all outputs share one `(n_output, n_timepoints)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome = info['outcome'][trials]
...
outcome_code = np.zeros(len(trials), dtype=np.int64)  # 0 = ignore
outcome_code[miss] = 1
outcome_code[hit] = 2
```

iii. The trials table stores the outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to 0=ignore, 1=miss, 2=hit, and repeated across all 80 bins.

ii.
```python
outcome_code = np.zeros(len(trials), dtype=np.int64)
outcome_code[miss] = 1
outcome_code[hit] = 2
```

iii. Same encoding as the instructions specify.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early = info['early'][trials]
early_code = (early == 'early').astype(np.int64)
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes, and repeated across all 80 bins.

ii.
```python
early_code = (early == 'early').astype(np.int64)
...
np.repeat(early_code[:, None], N_BINS, axis=1)
```

iii. Binary encoding matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is `(n_frames, 3)` = x, y, likelihood, with matching timestamps. Column 1 (y) is the value; column 2 (likelihood) decides visibility.

ii.
```python
grp = f['acquisition/BehavioralTimeSeries']
key = 'Camera0_side_TongueTracking'
data = np.asarray(grp[key]['data'][:])
ts = np.asarray(grp[key]['timestamps'][:])
y = data[:, 1]
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. The agent confirmed this is the only tongue measurement in the files.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.9 are marked as not visible. The 40th and 60th percentiles of visible frame y-positions are computed **over all visible frames** in the session (not over bin means). For each trial's time bins, visible frames are averaged within each bin, and the mean is classified against the session percentiles: 0 (below 40th), 1 (40th-60th), 2 (above 60th), 3 (not visible).

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
lo, hi = np.percentile(y[vis], [TONGUE_LOW_PCTL, TONGUE_HIGH_PCTL]) if vis.sum() else (0., 0.)
```

Per-trial binning uses cumulative sums for efficiency:
```python
yv = np.where(vis, y, 0.0)
cum_y = np.concatenate([[0.0], np.cumsum(yv)])
cum_n = np.concatenate([[0], np.cumsum(vis.astype(np.int64))])
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
s = cum_y[pos[:, 1:]] - cum_y[pos[:, :-1]]
mean_y = np.where(n > 0, s / np.maximum(n, 1), np.nan)
```

iii. The agent noted the likelihood distribution is "strongly bimodal" so the exact threshold is "immaterial."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Bins with visible frames are classified: 0 if mean < 40th percentile, 1 if between 40th and 60th (inclusive on both ends), 2 if above 60th. Bins with no visible frames get class 3.

ii.
```python
code = np.full((ntrials, nbins), 3, dtype=np.int64)
seen = n > 0
code[seen & (mean_y < lo)] = 0
code[seen & (mean_y >= lo) & (mean_y <= hi)] = 1
code[seen & (mean_y > hi)] = 2
```

iii. Follows the instruction's percentile-based discretization.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The camera timestamps share the same session-absolute clock. The same go-cue-aligned bin edges are used to find which camera frames fall in each bin via `searchsorted`.

ii.
```python
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
```

iii. Same bin grid for all data streams ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good units**: dropped (`ngood > 0` check in `session_is_used` and `if len(good) == 0` implicitly).
- **Trials outside observation intervals**: excluded via `observed_trials()`.
- **Trials with no go cue or tone**: excluded (NaN check).
- **Free/auto water trials**: excluded.
- **Trials with all-zero neural data**: removed after rate computation (`nonempty` check).
- **Tongue not visible**: assigned class 3 ("not visible").
- **Unsorted spike times**: sorted if needed (`if spikes.size and np.any(np.diff(spikes) < 0): spikes = np.sort(spikes)`).

ii.
```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
```

```python
nonempty = rates.sum(axis=(0, 2)) > 0
if not np.all(nonempty):
    trials = trials[nonempty]
```

```python
if spikes.size and np.any(np.diff(spikes) < 0):
    spikes = np.sort(spikes)
```

iii. The agent found that "in some NWB files the trials table spans the whole behavioural session while the ephys units are only observed during a subset of trials" and added the obs_intervals filter. The spike-sorting check was a defensive measure.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and extracting spike times data. The code uses multiprocessing (`Pool(nproc)`) to parallelize session processing, which significantly speeds up conversion. The full conversion of 105 sessions completes quickly with parallelization.

ii.
```python
with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```

iii. The agent noted surprisingly fast conversion times, likely due to file caching effects.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) the per-unit loop for spike binning (searchsorted per unit), and (2) the go_cue_times and tone_onset_times functions use Python loops to match events to trials.

ii.
```python
for k, iu in enumerate(good):
    a = 0 if iu == 0 else sti[iu - 1]
    spikes = np.asarray(u['spike_times'][a:sti[iu]])
    ...
```

```python
for i, g in zip(idx, go):
    if 0 <= i < len(out) and np.isnan(out[i]):
        out[i] = g
```

iii. The per-unit loop is inherent to the ragged spike time storage. The event-matching loops could potentially be vectorized but are not a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. Each NWB file is opened twice: once in `session_info()` to check behavioral criteria, and once in `process_session()` for full conversion. This reads the trials table and classification data twice for sessions that pass the filter.

ii.
```python
def process_session(path):
    info = session_info(path)  # first open
    used, perf, ncl, ncr = session_is_used(info)
    if not used:
        return None
    with h5py.File(path, 'r') as f:  # second open
        ...
```

iii. The two-pass approach was chosen to quickly skip sessions that don't pass behavioral criteria before doing the expensive full conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The elaborate coarse region mapping (14+ categories with keyword matching and ALM-specific logic based on probe targets) goes beyond what's strictly needed. The `frac_observed` computation and detailed session info metadata (performance, n_correct_left, n_correct_right) are computed but may not be used by the decoder.

ii.
```python
def coarse_region(anno, probe_targets_alm):
    """Map a CCF annotation string to one of the coarse brain areas."""
    al = anno.strip().lower()
    if probe_targets_alm and al.startswith(ALM_ANNOTATION_PREFIXES):
        return 'ALM'
    for name, keys in REGION_KEYS:
        for k in keys:
            if k in al:
                return name
    ...
```

```python
frac_observed=float(np.mean(np.clip(
    (np.minimum(stop[trials], g + OFF_END) - np.maximum(start[trials], g + OFF_START))
    / (OFF_END - OFF_START), 0, 1))),
```

iii. The region mapping is more detailed than needed but provides useful metadata. The frac_observed metric is informational.
