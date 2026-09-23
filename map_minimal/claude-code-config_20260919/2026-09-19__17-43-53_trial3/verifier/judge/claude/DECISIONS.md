# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The dataset is stored as one NWB file per session under `data/sub-<subject_id>/`. All files are found with a glob pattern and each is opened with `h5py` (not `pynwb`) and processed by `convert_session()`. Trials, units, behavioral events, and tongue tracking are read from HDF5 groups within each file. A multiprocessing pool (16 workers) parallelizes session conversion, with per-session results cached to disk as pickle files.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```

```python
with h5py.File(path, 'r') as f:
    tbl = session_trial_table(f)
    ...
    units = f['units']
```

iii. The agent recognized NWB as the published format. It chose `h5py` over `pynwb` for direct HDF5 access, which avoids pynwb's overhead and namespace loading. It also used multiprocessing and caching to speed up the conversion, noting "174 NWB sessions found" in output logs.

## 1-b. How are the data split into subjects?

i. Subject identity is extracted from the NWB filename by parsing out the `sub-<id>` prefix and stripping `sub-`. Subjects are collected into an `OrderedDict` during assembly, maintaining insertion order. Subject IDs are the numeric DANDI identifiers (e.g. `440956`), not the mouse names from the papers.

ii.
```python
subject = os.path.basename(path).split('_')[0].replace('sub-', '')
```

```python
subjects = OrderedDict()
...
if res['subject'] not in subjects:
    subjects[res['subject']] = len(subjects)
data['subject_idx'].append(subjects[res['subject']])
```

iii. The agent derived subject identity from the filename rather than from `nwb.subject.subject_id`, but these correspond to the same numeric IDs. The final summary reports 28 mice.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by the filename stem (e.g. `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order follows the sorted file list.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
name = os.path.basename(path).replace('.nwb', '')
```

iii. The agent correctly recognized one-file-per-session structure. After quality filtering, 145 sessions are retained.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`), one row per behavioral trial. The go cue count is verified to match the trial count.

ii.
```python
tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(tbl['go_time']) == len(tbl['start_time'])
```

iii. The agent correctly used the trials table as the trial definition and validated that go cue events match trial count.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in several ways:
1. **Free water trials** excluded (`free_water == 0`).
2. **Short trials**: Trials where fewer than 20 bins of the [-2.5, 1.5]s window fall within the trial's observed interval (`start_time` to `stop_time`) are excluded.
3. **No-ephys trials**: Trials where the total firing rate across all neurons is zero are dropped (detects recording gaps in 8 sessions).
4. Photostim, early-lick, and ignore trials are kept because they are decoder inputs/outputs.

ii.
```python
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
short = np.array([len(b) < MIN_BINS_PER_TRIAL for b in bins])
keep_trial = (tbl['free_water'] == 0) & ~short
```

```python
has_spikes = rates.sum(axis=(0, 2)) > 0
trials = trials[has_spikes]
```

iii. The agent noted that free-water trials are excluded per the method paper, while photostim/early-lick/ignore trials are kept as "DEVIATION" since they are decoder inputs/outputs. The short-trial filter and no-ephys filter are additional quality controls beyond what the reference uses.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (via `units/spike_times_index` for ragged indexing) — the sorted spike times of each unit in session-absolute seconds. Only units with `classification == 'good'` contribute. Go cue times are used for alignment.

ii.
```python
spike_index = units['spike_times_index'][:]
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
    counts = np.diff(
        np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
    rates[i] = counts / BIN_WIDTH
```

iii. The agent used spike times as the only neural representation, identical to the reference approach.

## 2-b. How is the `neural` data processed?

i. Spike times are converted to firing rates (Hz) in 50 ms non-overlapping bins. For each good unit, `np.searchsorted` finds the spike count at each bin edge, and differencing gives the count per bin. Counts are divided by the 0.05 s bin width to get Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
counts = np.diff(
    np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
rates[i] = counts / BIN_WIDTH
```

iii. The agent used the same searchsorted-based binning approach as the reference, producing firing rates in Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered in two stages:
1. `classification == 'good'` AND `anno_name != ''` (units must have both the QC classifier label and a CCF annotation).
2. Units with zero spikes across all analysis windows are dropped (the "alive" filter).

ii.
```python
classification = _str_col(units['classification'])
anno = _str_col(units['anno_name'])
good = np.flatnonzero((classification == 'good') & (anno != ''))
```

```python
alive = rates.any(axis=(1, 2))
good = good[alive]
rates = rates[alive]
```

iii. The agent applied both the classification filter and an additional annotation filter (requiring non-empty `anno_name`), plus removed silent units. The reference only uses `classification == 'good'`. The agent justified the annotation filter by noting "Those units are exactly the ones carrying a CCF annotation."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are defined relative to the go cue and added to each trial's go cue time to produce absolute time edges. Spikes are binned against these edges using `searchsorted`. All timestamps are on the same session-absolute clock.

ii.
```python
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The agent correctly aligned to the go cue onset as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin width is 50 ms, producing 80 non-overlapping bins spanning [-2.5, 1.5] s relative to the go cue. However, the AI applies a per-trial mask that drops bins outside the trial's observed interval (from `start_time` to `stop_time`), so the number of bins varies per trial. No additional rebinning is applied.

ii.
```python
BIN_WIDTH = 0.05         # s
BIN_EDGES = np.round(T_START + BIN_WIDTH * np.arange(
    int(round((T_END - T_START) / BIN_WIDTH)) + 1), 10)
```

```python
def mask_bins(tbl, trial):
    lo = tbl['start_time'][trial] - tbl['go_time'][trial]
    hi = tbl['stop_time'][trial] - tbl['go_time'][trial]
    keep = np.flatnonzero((BIN_EDGES[:-1] >= lo) & (BIN_EDGES[1:] <= hi))
    return keep
```

iii. The agent stated: "Zero-filling would have manufactured 'all neurons silent' samples perfectly correlated with the miss outcome — a free answer for the decoder — and dropping the trials instead would have deleted the miss class." This motivated the variable-length trial approach.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents` (the tone onsets), together with the go cue times. The tone for each trial is the last sample onset before the go cue, accounting for early-lick replays.

ii.
```python
sample = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
go = tbl['go_time']
...
trial_of = np.searchsorted(start, sample, side='right') - 1
valid = (trial_of >= 0) & (trial_of < len(go))
for t, s in zip(trial_of[valid], sample[valid]):
    if s <= go[t]:
        tone[t] = s
```

iii. The agent correctly identified that early licks replay the sample epoch, so the last tone before the go cue is used. It also added a fallback for trials without a recorded sample onset (filling with the session median offset), though none were found in this dataset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time relative to the go cue is computed, then subtracted from the bin centers to get time from tone onset. The sign convention is: the value at each bin center is `BIN_CENTERS[b] - tone_rel[t]`, where `tone_rel` is the tone onset time relative to go cue (negative, since tone comes before go cue).

ii.
```python
rel = tone - go  # negative, tone is before go cue
...
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. The agent computed tone-relative time correctly, though the sign convention differs slightly in code structure from the reference (which uses `CENTERS + (go - tone)`). Both yield the same result: time from tone onset in seconds, increasing as time progresses.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin grid (BIN_CENTERS relative to go cue) is used for both neural data and this input, ensuring alignment. However, because the AI masks bins per trial, only the bins within the trial's observed interval are kept.

ii.
```python
inp[0] = BIN_CENTERS[b] - tone_rel[t]  # b is the masked bin indices
```

iii. Alignment is inherent since both neural and input data use the same go-cue-relative bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and the go cue used to compute the stimulation window relative to the go cue.

ii.
```python
stim = tbl['photostim_onset'] != 'N/A'
idx = np.flatnonzero(stim)
for i in idx:
    onset = float(tbl['photostim_onset'][i])
    dur = float(tbl['photostim_duration'][i])
    on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
    off[i] = on[i] + dur
```

iii. The agent correctly identified that photostim_onset is stored relative to trial start and needs conversion to go-cue-relative time. It verified these against the photostim event timestamps.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls between the onset and offset of stimulation, 0 otherwise. Non-stimulated trials have NaN onset/offset, which naturally compare as False.

ii.
```python
if np.isfinite(stim_on[t]):
    inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
              (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
else:
    inp[1] = 0.0
```

iii. The agent used an explicit if/else for stimulated vs non-stimulated trials, while the reference uses NaN comparison. Both achieve the same result.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are expressed relative to the go cue, matching the bin grid used for neural data. Only bins within the masked interval are kept.

ii.
```python
inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
          (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
```

iii. Same go-cue-relative coordinate system as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the actual lick port events: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first lick in the 1.5 s answer period after the go cue determines the choice. This differs from the reference, which derives choice from `trial_instruction` x `outcome`.

ii.
```python
def lick_choice(left, right, tbl):
    go = tbl['go_time']
    choice = np.full(len(go), 2, dtype=np.int8)
    for i, g in enumerate(go):
        li = np.searchsorted(left, g)
        ri = np.searchsorted(right, g)
        tl = left[li] if li < len(left) else np.inf
        tr = right[ri] if ri < len(right) else np.inf
        end = g + T_END
        if tl >= end and tr >= end:
            continue
        choice[i] = 0 if tl <= tr else 1
    return choice
```

iii. The agent stated: "This agrees with the label implied by outcome x instruction (hit -> instructed port, miss -> opposite port, ignore -> no lick) on >99.3% of trials; the direct read-out from the lick ports is used because 'lick direction choice' is a behavioral measurement rather than a re-encoding of outcome."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Coded as 0 (left), 1 (right), 2 (no lick). Per-trial value is repeated across all kept bins for the trial.

ii.
```python
outp[0] = choice[t]
```

iii. Same coding as the reference. The value is repeated per time bin.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which holds `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_code = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
```

iii. Direct use of the trials table outcome field, same as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0 (ignore), 1 (miss), 2 (hit) via a dictionary. Per-trial value repeated across all kept bins.

ii.
```python
outp[1] = outcome[t]
```

iii. Same coding and approach as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which holds `'no early'` and `'early'`.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
```

iii. Direct use of the trials table field, same as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes) via boolean comparison. Per-trial value repeated across bins.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
...
outp[2] = early[t]
```

iii. Same coding as the reference (`no early` -> 0, `early` -> 1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, whose data is `(n_frames, 3)` = x, y, likelihood, with matching timestamps. Column 1 (y) is used; column 2 (likelihood) determines visibility.

ii.
```python
key = 'acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking'
ts = f[key + '/timestamps'][:]
data = f[key + '/data'][:]
y = data[:, 1].astype(np.float64)
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. Same source data as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing steps:
1. Frames with DLC likelihood <= 0.5 are marked as not visible.
2. **Outlier rejection**: A 5-sigma velocity threshold on frame-to-frame movement identifies outliers (frames where position jumps on the way in and out), which are then interpolated from neighbors. This follows the method paper.
3. Per-bin mean of visible frames' y-positions is computed.
4. Session-wide percentiles (40th, 60th) are computed over the visible bin-mean values.
5. Each bin is classified: 0 (< 40th), 1 (40th-60th), 2 (> 60th), 3 (not visible).

ii.
```python
# Outlier rejection
vel = np.full(len(dt), np.nan)
vel[adjacent] = np.diff(yv)[adjacent] / dt[adjacent]
sigma = np.nanstd(vel)
jump = np.abs(vel) > MARKER_OUTLIER_SIGMA * sigma
outlier = np.zeros(len(yv), dtype=bool)
outlier[1:-1] = jump[:-1] & jump[1:]
```

```python
# Percentile computation and classification
vis_vals = bin_y[observed & np.isfinite(bin_y)]
p40, p60 = np.percentile(vis_vals, TONGUE_PCTILES)
tongue_class[vis_bin & (bin_y < p40)] = 0
tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
tongue_class[vis_bin & (bin_y > p60)] = 2
```

iii. The agent added 5-sigma outlier rejection following the method paper, which the reference does not include. The agent also computes percentiles over observed bins only (using the `observed` mask from `mask_bins`).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The session's 40th and 60th percentiles of visible per-bin tongue y-position are used as thresholds. Values below the 40th percentile get class 0, between 40th and 60th get class 1, above 60th get class 2, and bins with no visible frames get class 3.

ii.
```python
tongue_class[vis_bin & (bin_y < p40)] = 0
tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
tongue_class[vis_bin & (bin_y > p60)] = 2
```

iii. The boundary conditions differ slightly from the reference: the AI uses `<` for the lower threshold and `<=` for the 40th-60th range, while the reference uses `np.digitize(m[ok], edges)` which gives `[0, p40)` -> 0, `[p40, p60)` -> 1, `[p60, inf)` -> 2. The AI's middle class is `[p40, p60]` (inclusive on both ends), while the reference's is `[p40, p60)`.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock. For each trial, frames within each bin's time range are found via `searchsorted` on the bin edges, and visible frames' y-values are averaged to get the bin value. Only bins within the trial's masked interval are kept.

ii.
```python
for k, t in enumerate(trials):
    lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
    hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
    has_video[k] = hi > lo
    for b in range(NBINS):
        if hi[b] <= lo[b]:
            continue
        sl = slice(lo[b], hi[b])
        vis = tongue_vis[sl]
        if vis.any():
            bin_y[k, b] = tongue_y[sl][vis].mean()
```

iii. The same go-cue-relative bin grid is used for tongue data as for neural data. The agent bins camera frames into the same 50 ms bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions without good units or annotations**: Dropped (classification not 'good' or empty anno_name).
- **Sessions with poor video coverage**: Dropped if camera covers < 95% of analysis window (7 sessions).
- **Sessions with bad tongue tracking**: Dropped if tongue-lick agreement precision < 0.2 or recall < 0.5 (1 session).
- **Trials without spike data**: Dropped via the no-ephys filter (rates sum to zero).
- **Short trials**: Dropped if fewer than 20 bins in the observed interval.
- **Free water trials**: Dropped.
- **Tongue tracking outliers**: 5-sigma velocity outliers are interpolated from neighbors.
- **Bins with no visible tongue**: Assigned class 3 (not visible).
- **Missing tone onset**: Falls back to session median offset (no cases found).

ii.
```python
if coverage < MIN_VIDEO_COVERAGE:
    out['rejected'] = 'video does not cover the analysis window'
    return out
if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL:
    out['rejected'] = 'tongue tracking failed'
    return out
```

iii. The agent applied more extensive quality controls than the reference, including video coverage checks, tongue-lick agreement validation, and outlier rejection.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files and extracting spike times dominates. The agent uses multiprocessing (16 workers) to parallelize session conversion, with per-session caching to pickle files. The final assembly and pickling of the result is another costly step.

ii.
```python
with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
```

iii. The parallelization and caching strategy is more sophisticated than the reference's sequential approach.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain:
1. The per-unit spike binning loop (one `searchsorted` per unit) — same as the reference, inherent to ragged storage.
2. The per-trial tongue binning loop (iterates over bins within each trial).
3. The `lick_choice` function iterates over trials to find the first lick.
4. The `photostim_intervals` function iterates over stimulated trials.
5. The `tone_onset_times` function uses a Python loop to assign sample onsets to trials.

ii.
```python
for i, u in enumerate(good):
    lo = 0 if u == 0 else spike_index[u - 1]
    st = units['spike_times'][lo:spike_index[u]]
    ...
```

```python
for i, g in enumerate(go):
    li = np.searchsorted(left, g)
    ...
```

iii. The lick_choice, photostim_intervals, and tone_onset_times functions all use Python loops over trials that could potentially be vectorized with numpy operations.

## 10-c. What processing does the code repeat multiple times?

i. The code processes each session once (with caching), so no recomputation occurs. However, the `mask_bins` function is called once per trial before any filtering, meaning it's computed for trials that may later be dropped (free water, no ephys, etc.).

ii.
```python
bins = [mask_bins(tbl, t) for t in range(len(tbl['start_time']))]
```

iii. Computing mask_bins for all trials before filtering means some work is wasted on trials that will be excluded.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
1. **Allen ontology download and region mapping**: Downloads and parses the full Allen brain atlas structure graph to map CCF annotations to coarse regions, rather than using simple string parsing as the reference does.
2. **Session performance computation**: Computes behavioral performance metrics for session selection, which is an extra filter not in the reference.
3. **Video QC metrics**: Computes video coverage and tongue-lick agreement for session quality control, adding processing not in the reference.
4. **Outlier rejection**: The 5-sigma velocity outlier detection and imputation on tongue tracking is extra processing.
5. **Detailed trial_metadata**: Stores extensive trial-level metadata that is not used by the decoder.
6. **mask_bins for discarded trials**: As noted in 10-c.

ii.
```python
region_map = load_region_map()  # downloads Allen ontology
perf, n_left, n_right = session_performance(tbl)  # behavioral QC
precision, recall = tongue_lick_agreement(...)  # video QC
```

iii. Much of this extra processing supports more rigorous quality control. The Allen ontology approach is more principled than string parsing but adds complexity and an external dependency.
