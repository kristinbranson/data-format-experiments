# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files using `h5py` directly (not `pynwb`). All sessions are found by globbing `sub-*/*.nwb` under the data directory. Each file is opened with `h5py.File()` and trials, units, events, and video data are read from the HDF5 groups directly. Processing is parallelized with `multiprocessing.Pool`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with Pool(args.nproc) as pool:
    infos = []
    for info in pool.imap_unordered(process_session, files):
        infos.append(info)
```

```python
with h5py.File(path, 'r') as f:
    trials = f['intervals/trials']
    start = trials['start_time'][:]
    ...
    events = f['acquisition/BehavioralEvents']
    go = events['go_start_times/timestamps'][:]
```

iii. The agent used h5py for direct, low-level access to the NWB HDF5 structure. Both h5py and pynwb were available. No explicit justification for h5py over pynwb was provided, but h5py offers faster direct access without the pynwb object model overhead, which matters for 174 large files processed in parallel.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the filename rather than from NWB metadata fields. The function `session_name()` parses the filename to extract the subject ID (the numeric string from `sub-<id>`). Subjects are collected as a list during assembly, maintaining insertion order.

ii.
```python
def session_name(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return sub, ses
```

```python
s = sess['subject']
if s not in subjects:
    subjects.append(s)
subject_idx.append(subjects.index(s))
```

iii. The filename contains the subject identifier in the `sub-<id>` portion. The agent extracted this from filenames during initial inspection.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session, so no splitting is needed. Session order follows the sorted file list. Sessions are identified by the parsed subject and session strings from the filename.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
subject, ses = session_name(path)
info = {'file': os.path.basename(path), 'subject': subject, 'session': ses}
```

iii. The dandiset stores one session per NWB file, so no additional grouping is required.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. The go cue count is verified against the trial count.

ii.
```python
trials = f['intervals/trials']
start = trials['start_time'][:]
stop = trials['stop_time'][:]
...
go = events['go_start_times/timestamps'][:]
...
if len(go) != ntrials_all:
    info['skip'] = 'go cue count does not match trial count'
    return info
```

iii. Trials are defined by the trials table in the NWB file, with go cue count as a consistency check.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial filters:
1. **obs_intervals**: Only trials covered by ephys recording for ALL good units (intersection across units).
2. **auto_water and free_water**: Both excluded (water not contingent on choice).
3. **Video coverage**: Trials with fewer than 50% of expected video frames in the [-2.5, 1.5] s window are dropped.
4. **Session-level behavioral criteria**: Performance >65%, at least 50 correct lick-left and 50 correct lick-right control trials.
5. **Minimum 50 usable trials per session**.
6. Photostim, early-lick, miss, and ignore trials are KEPT.

ii.
```python
# obs_intervals intersection across all good units
recorded = np.ones(ntrials_all, dtype=bool)
for uid in good0:
    o0 = 0 if uid == 0 else oii[uid - 1]
    ints = obs[o0:oii[uid]]
    idx = np.searchsorted(start, ints[:, 0] + 1e-6) - 1
    ...
    recorded &= m

# session selection criteria
control = recorded & (~photostim_trial) & (early == 'no early') & (~auto_water) & (~free_water)
responded = control & (outcome != 'ignore')
performance = float(np.mean(outcome[responded] == 'hit')) if responded.sum() else 0.0
...
if performance <= MIN_PERFORMANCE:
    info['skip'] = 'performance <= 65%%: %.3f' % performance
    return info
if min(n_correct_left, n_correct_right) < MIN_CORRECT_PER_DIRECTION:
    info['skip'] = 'fewer than 50 correct trials in one direction'
    return info

# trial-level filters
keep = recorded & (~auto_water) & (~free_water)
...
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
...
if len(trial_idx) < MIN_TRIALS_PER_SESSION:
    info['skip'] = 'only %d usable trials (mostly missing video)' % len(trial_idx)
    return info
```

iii. The agent justified session-level criteria by quoting the data paper: "We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65 %), and at least 50 correct lick left and lick right trials each." Auto-water and free-water exclusion matched the reference code behavior. Video coverage was added because "tongue output would be undefined" for trials lacking video. The minimum 50 trials per session was "consistent with the data paper's session-level trial-count requirement."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (sorted spike times of each unit). Go cue times from `BehavioralEvents/go_start_times` define the alignment. Only units passing QC filters contribute.

ii.
```python
spike_times = units['spike_times']
sidx = units['spike_times_index'][:]
...
for n, uid in enumerate(unit_ids):
    s0 = 0 if uid == 0 else sidx[uid - 1]
    sp = spike_times[s0:sidx[uid]]
```

iii. `spike_times` is the only neural representation in the NWB file.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue (80 bins). Spike counts are computed using `np.bincount` per unit per trial, then divided by the bin width to get firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
neural = np.zeros((ntrials, nneurons, NBINS), dtype=np.float32)
...
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        if hi[t] <= lo[t]:
            continue
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
        np.clip(b, 0, NBINS - 1, out=b)
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
neural /= BIN_SIZE  # spikes/s
```

iii. The decoder task specifies 50 ms bins and the window parameters. The agent follows these directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied to units:
1. `units/classification == 'good'` (QC classifier verdict).
2. `units/anno_name != ''` (non-empty CCF annotation, meaning the unit is histologically localized).

Additionally, units without a valid coarse region mapping or finite CCF x-coordinate are excluded. A session with no surviving units is dropped.

ii.
```python
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
good = (classification == 'good') & (anno != '')
...
for i in unit_ids:
    reg = coarse_region(anno[i], utarget[i])
    x = ux[i]
    if reg is None or not np.isfinite(x):
        keep_unit.append(False)
```

iii. The agent verified: "Ontology mapping reproduces the paper's per-region unit counts exactly (thalamus 12808, striatum 7664, midbrain 7495, medulla 2928), confirming 'classification==good' + anno_name is the right unit selection." Total: 69,453 good units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and event times are on the same session-absolute clock. The bin edges for each trial are computed as `go_cue_time + OFF_START` to `go_cue_time + OFF_END`. Spikes are searched within each trial's window per unit.

ii.
```python
edges_lo = win_lo[trial_idx]  # win_lo = go + OFF_START
for n, uid in enumerate(unit_ids):
    ...
    lo = np.searchsorted(sp, edges_lo)
    hi = np.searchsorted(sp, win_hi[trial_idx])
    for t in range(ntrials):
        ...
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
```

iii. All NWB timestamps share one global clock, so alignment only requires computing absolute bin edges from the go cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue, giving 80 bins per trial. No rebinning is applied; spike times are binned directly from raw timestamps.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The 50 ms bin width and the -2.5 to +1.5 s window are specified in the decoder task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset timestamps) and the go cue times. The tone for each trial is the last sample-epoch start before that trial's go cue.

ii.
```python
sample_on = events['sample_start_times/timestamps'][:]
...
tone_idx = np.searchsorted(sample_on, go) - 1
tone_rel_go = sample_on[np.maximum(tone_idx, 0)] - go
```

iii. "Tone (sample epoch) onset: last sample-epoch start before the go cue; with early licks the sample epoch is replayed, so the last one is the relevant one."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, `tone_rel_go` is the time of the tone relative to the go cue (negative). The input for each bin is `bin_center - tone_rel_go`, giving the time elapsed since the tone onset.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
...
for k, t in enumerate(trial_idx):
    inputs[k, 0] = bin_centers - tone_rel_go[t]
```

iii. Since `tone_rel_go` is negative (tone is before go cue), `bin_centers - tone_rel_go` gives a positive, increasing value representing seconds since tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The bin centers are defined relative to the go cue (`OFF_START + (i + 0.5) * BIN_SIZE`), the same grid used for neural binning. The tone onset offset is subtracted to convert from go-cue-relative to tone-onset-relative time.

ii.
```python
bin_centers = OFF_START + (np.arange(NBINS) + 0.5) * BIN_SIZE
```

iii. Both neural and input data use the same go-cue-aligned bin grid, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_start_times/timestamps` and `photostim_stop_times/timestamps` in the BehavioralEvents acquisition group, rather than from the trials table columns.

ii.
```python
stim_on = events['photostim_start_times/timestamps'][:]
stim_off = events['photostim_stop_times/timestamps'][:]
```

iii. The agent used the event timestamps directly, which provide absolute session-time onset/offset of photostimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 in bins overlapping a photostim interval, 0 elsewhere. The stim events are matched to trials via `np.searchsorted` on trial start times. For each stim event, the bins it covers are computed from the onset/offset relative to the trial's window start.

ii.
```python
stim_trial = np.searchsorted(start, stim_on) - 1
for on, off, tr in zip(stim_on, stim_off, stim_trial):
    if tr < 0 or tr >= ntrials_all or not keep[tr]:
        continue
    k = int(np.searchsorted(trial_idx, tr))
    b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
    b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
    b0 = max(b0, 0)
    b1 = min(b1, NBINS)
    if b1 > b0:
        inputs[k, 1, b0:b1] = 1.0
```

iii. The agent uses floor/ceil to find bins that overlap the stim interval, rather than checking bin centers. This means any bin partially overlapping the photostim interval is marked as 1.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset times are in session-absolute time. The bin assignment is computed relative to `win_lo[tr]` (= go cue + OFF_START), the same reference used for neural binning.

ii.
```python
b0 = int(np.floor((on - win_lo[tr]) / BIN_SIZE))
b1 = int(np.ceil((off - win_lo[tr]) / BIN_SIZE))
```

iii. Both streams share the same absolute time axis and the same go-cue-aligned bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore) in the trials table. Choice is not stored directly.

ii.
```python
choice_map = {'left': 0, 'right': 1}
for k, t in enumerate(trial_idx):
    if outcome[t] == 'hit':
        ch = choice_map[instruction[t]]
    elif outcome[t] == 'miss':
        ch = choice_map['right' if instruction[t] == 'left' else 'left']
    else:
        ch = 2
    outputs[k, 0, :] = ch
```

iii. The agent verified 100% agreement between this derivation and actual lick-event data.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Hit trials get the instructed side (left=0, right=1), miss trials get the opposite side, ignore trials get "no lick" (2). The value is per-trial, broadcast across all 80 bins.

ii. (Same as 5-a code snippet)

iii. The encoding follows the instructions' categories: left, right, no lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', or 'hit'.

ii.
```python
outcome = to_str(trials['outcome'][:])
...
outputs[k, 1, :] = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome[t]]
```

iii. The outcome is stored explicitly in the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. Per-trial value, broadcast across all 80 bins.

ii. (Same as 6-a code snippet)

iii. Matches the instruction categories.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds 'no early' or 'early'.

ii.
```python
early = to_str(trials['early_lick'][:])
...
outputs[k, 2, :] = 1 if early[t] == 'early' else 0
```

iii. The early lick flag is stored explicitly in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to: no=0, yes=1. Per-trial value, broadcast across all 80 bins.

ii. (Same as 7-a code snippet)

iii. Matches the instruction categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data = (tongue_x, tongue_y, tongue_likelihood) with matching timestamps. Column 1 (tongue_y) is the position value; column 2 (likelihood) determines visibility.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
tongue_y = vdata[:, 1]
visible = vdata[:, 2] > LIKELIHOOD_THRESH  # LIKELIHOOD_THRESH = 0.9
```

iii. This is the only tongue measurement in the file.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Steps:
1. Frames with likelihood <= 0.9 are excluded (tongue not visible).
2. Per-session percentiles (40th, 60th) are computed over ALL visible frames across the entire session (not binned).
3. Per trial per bin, the **median** of visible tongue y-values is computed.
4. The median is discretized: <40th pctile = 0, 40th-60th = 1, >60th = 2.
5. Bins with no visible frames = 3 (not visible).

ii.
```python
visible = vdata[:, 2] > LIKELIHOOD_THRESH
if visible.sum() > 0:
    p40, p60 = np.percentile(tongue_y[visible], [40, 60])

...
for bi in range(NBINS):
    m = (b == bi) & vv
    if not np.any(m):
        continue
    y = float(np.median(yy[m]))
    cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. The agent used a 0.9 likelihood threshold noting that tongue visibility is "strongly bimodal" with essentially all confident detections above 0.9. Percentiles are computed over raw visible frames (not bin means). Within each bin, the agent uses median rather than mean.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles of tongue y-position over all visible frames. For each bin:
- Class 0: y < 40th percentile
- Class 1: 40th percentile <= y <= 60th percentile
- Class 2: y > 60th percentile
- Class 3: no visible frames in the bin

ii.
```python
p40, p60 = np.percentile(tongue_y[visible], [40, 60])
...
cls[bi] = 0 if y < p40 else (1 if y <= p60 else 2)
```

iii. The boundary conditions use strict `<` for the lower threshold and `<=` for the upper, matching the instructions' "< 40th", "40th to 60th", "> 60th".

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock. For each trial, frames within [go + OFF_START, go + OFF_END] are selected. Each frame is assigned to a bin by its offset from the window start divided by the bin size. This uses the same go-cue-aligned grid as neural data.

ii.
```python
b = ((tt - win_lo[t]) / BIN_SIZE).astype(np.int64)
np.clip(b, 0, NBINS - 1, out=b)
```

iii. All streams share the same absolute clock and go-cue-aligned bin grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Session without CCF annotations**: `anno_name` is empty for all units; session is dropped because no units pass `anno_name != ''`.
- **Partial ephys coverage**: Intersection of obs_intervals across all good units determines which trials have data; trials outside are excluded.
- **Auto/free water trials**: Excluded as non-standard trials.
- **Low video coverage**: Trials with <50% expected video frames are dropped.
- **No visible tongue in a bin**: Assigned class 3 ("not visible").
- **No visible tongue in session**: Session is dropped entirely.

ii.
```python
if not np.isfinite(p40):
    info['skip'] = 'no visible tongue frames in session'
    return info
...
keep &= nframes >= VIDEO_COVERAGE_MIN * expected_frames
```

iii. The agent takes a conservative approach, dropping data that would be unreliable rather than imputing or keeping zeros.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Reading NWB files with h5py (large spike_times and video data arrays).
2. The nested per-unit, per-trial spike binning loop.
3. The per-trial, per-bin tongue y-position computation loop.
4. Pickling the final large dataset.

ii.
```python
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        ...
        b = ((sp[lo[t]:hi[t]] - edges_lo[t]) / BIN_SIZE).astype(np.int64)
        ...
        neural[t, n] = np.bincount(b, minlength=NBINS)[:NBINS]
```

iii. The multiprocessing parallelization helps with I/O-bound file reading, but the inner loops are still sequential.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested loops could be improved:
1. **Neural binning**: The per-unit, per-trial double loop could be partially vectorized by flattening all trial edges per unit into a single searchsorted call (as the reference does).
2. **Tongue y-position**: The per-trial, per-bin loop iterates over every bin individually, which could be vectorized using bincount/binned statistics.

ii.
```python
# Neural: nested loop over units and trials
for n, uid in enumerate(unit_ids):
    ...
    for t in range(ntrials):
        ...

# Tongue: nested loop over trials and bins
for k, t in enumerate(trial_idx):
    ...
    for bi in range(NBINS):
        m = (b == bi) & vv
```

iii. The reference code vectorizes the per-trial dimension for neural binning by flattening all bin edges and using a single searchsorted per unit.

## 10-c. What processing does the code repeat multiple times?

i. The code reads units data twice from the HDF5 file in two separate blocks:
1. First to determine `good0` (for obs_intervals filtering).
2. Again for the final unit curation (`classification`, `anno`).

ii.
```python
# First read
units0 = f['units']
cls0 = to_str(units0['classification'][:])
anno0 = to_str(units0['anno_name'][:])
good0 = np.where((cls0 == 'good') & (anno0 != ''))[0]
...

# Second read
units = f['units']
classification = to_str(units['classification'][:])
anno = to_str(units['anno_name'][:])
good = (classification == 'good') & (anno != '')
```

iii. The double read of unit classification and annotation data is redundant and could be consolidated.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of processing are done that may not be needed downstream:
1. **Coarse region mapping with Allen ontology**: The code downloads and parses the entire Allen CCF structure graph to map fine-grained annotations to coarse regions. The reference code just uses simple string splitting on `anno_name`.
2. **Hemisphere determination**: The code determines left/right hemisphere from CCF x-coordinates, adding hemisphere to region names (e.g., "left Thalamus"). The reference does not include hemisphere.
3. **Session performance computation**: Performance metrics are computed and checked against thresholds, which the reference does not do.

ii.
```python
def coarse_region(anno, probe_target):
    path = ONTOLOGY.get(anno)
    ...
    for key, name in [('Medulla', 'Medulla'), ('Pons', 'Pons'), ...]:
        if key in inpath:
            return name
    ...
    side = 'left' if x >= ML_MIDLINE else 'right'
    regions.append('%s %s' % (side, reg))
```

iii. The ontology mapping and hemisphere assignment add complexity but produce different brain region labels than needed. The downstream decoder uses brain_region_idx for region-specific analysis, so the exact region naming scheme matters for interpretability but the extra processing is not strictly necessary.
