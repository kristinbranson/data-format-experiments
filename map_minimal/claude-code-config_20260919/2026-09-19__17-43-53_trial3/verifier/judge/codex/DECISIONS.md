# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every NWB file under `/app/data/sub-*/*.nwb`, sorts the paths, loads each file with `h5py`, and converts sessions in a multiprocessing pool. Per-session results are cached as pickles before accepted sessions are assembled.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with Pool(n_workers) as pool:
    caches = pool.map(_worker, [(p, region_map) for p in files], chunksize=1)
...
with h5py.File(path, 'r') as f:
    tbl = session_trial_table(f)
```

iii. The trajectory says the directory contains 174 NWB sessions from 28 mice. The agent treated one NWB file as one session, used HDF5 directly for speed, parallelized conversion, and cached sessions to make expensive reruns resumable.

## 1-b. How are the data split into subjects?

i. The subject identifier is parsed from the NWB filename, then an insertion-ordered mapping assigns each retained session a subject index.

ii.
```python
subject = os.path.basename(path).split('_')[0].replace('sub-', '')
...
if res['subject'] not in subjects:
    subjects[res['subject']] = len(subjects)
data['subject_idx'].append(subjects[res['subject']])
```

iii. Filenames and directories use `sub-<mouse>`, so the agent regarded that token as the canonical grouping key. The final trajectory reports all 28 mice remain.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its basename becomes `session_name`; rejected sessions are omitted during final assembly.

ii.
```python
name = os.path.basename(path).replace('.nwb', '')
...
if 'rejected' in res:
    rejected.append(...)
    continue
data['neural'].append(res['neural'])
```

iii. The agent inferred session boundaries from the published one-file-per-session layout. It additionally justified rejecting sessions that fail paper-level behavioral criteria or cannot support the requested tongue output.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`; go-cue timestamps are read separately and asserted to have the same length. Retained row indices are used consistently for all streams.

ii.
```python
trials = f['intervals/trials']
tbl = {'start_time': trials['start_time'][:], ...}
tbl['go_time'] = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(tbl['go_time']) == len(tbl['start_time'])
...
trials = np.flatnonzero(keep_trial)
```

iii. The NWB trial table explicitly defines behavioral trials. The length assertion was used to verify an unambiguous one-to-one trial/go-cue mapping.

## 1-e. How are trials filtered based on quality controls?

i. Before trial filtering, sessions must have performance above 65%, at least 50 correct trials per direction, usable video coverage, and tongue/lick agreement. Within accepted sessions, free-water trials, trials with fewer than 20 fully observed 50-ms bins, and trials with no spikes across all retained units are removed. Photostimulation, early-lick, ignore, and auto-water trials are retained.

ii.
```python
if perf <= MIN_PERFORMANCE or min(n_left, n_right) < MIN_CORRECT_PER_DIRECTION:
    return rejected
...
keep_trial = (tbl['free_water'] == 0) & ~short
...
has_spikes = rates.sum(axis=(0, 2)) > 0
trials = trials[has_spikes]
...
if coverage < MIN_VIDEO_COVERAGE: return out
if precision < MIN_TONGUE_LICK_PRECISION or recall < MIN_TONGUE_LICK_RECALL: return out
```

iii. The trajectory cites the data paper's session criteria and method paper's free-water exclusion. It deliberately retained early/ignore/photostim trials because they define requested decoder variables. Empty-spike trials were interpreted as recording gaps; video QC was added because tongue labels could not otherwise be constructed reliably.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `units/spike_times_index`, using `units/classification` and `units/anno_name` for unit selection and go cues for trial-relative bin placement.

ii.
```python
units = f['units']
classification = _str_col(units['classification'])
good = np.flatnonzero((classification == 'good') & (anno != ''))
spike_index = units['spike_times_index'][:]
st = units['spike_times'][lo:spike_index[u]]
```

iii. The agent identified spike times as the source neural measurement and the paper's classifier label as the intended spike-sorting QC verdict.

## 2-b. How is the `neural` data processed?

i. For each retained unit, absolute go-aligned bin edges are searched in its sorted spike train. Adjacent cumulative indices are differenced into spike counts and divided by 0.05 s to obtain Hz. No smoothing or normalization is applied. The result is later sliced to the bins fully inside each trial interval.

ii.
```python
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
counts = np.diff(
    np.searchsorted(st, edges).reshape(len(trials), NBINS + 1), axis=1)
rates[i] = counts / BIN_WIDTH
...
neural.append(np.ascontiguousarray(rates[:, k, b]))
```

iii. The agent followed the requested firing-rate computation. It truncated unobserved bins because the NWBs export spikes only within trial intervals and argued that filling missing bins with zeros would leak outcome through artificial silence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'`, a nonempty anatomical annotation, and at least one spike in all retained analysis windows. Sessions with no surviving units are rejected.

ii.
```python
good = np.flatnonzero((classification == 'good') & (anno != ''))
...
alive = rates.any(axis=(1, 2))
good = good[alive]
rates = rates[alive]
```

iii. The classifier reproduces the paper's reported good-unit fraction. Anatomical labels are required for region indices. The agent dropped silent units because they contain no decoder information and create null PCA directions, while rejecting the method paper's 2-Hz cutoff as analysis-specific rather than QC.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Session-absolute go-cue times are added to relative edges from -2.5 to +1.5 s. Only complete bins within the trial's `start_time`/`stop_time` are retained, so trials share an alignment grid but can have different lengths.

ii.
```python
edges = (go[trials][:, None] + BIN_EDGES[None, :]).ravel()
...
lo = tbl['start_time'][trial] - tbl['go_time'][trial]
hi = tbl['stop_time'][trial] - tbl['go_time'][trial]
keep = np.flatnonzero((BIN_EDGES[:-1] >= lo) & (BIN_EDGES[1:] <= hi))
```

iii. The agent reasoned that all NWB clocks are shared, so adding offsets performs alignment directly. Variable extents were a deliberate safeguard against treating unrecorded periods as neural silence.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms, with 80 possible nonoverlapping bins over four seconds. Raw spikes are binned directly at that resolution; no subsequent temporal rebinning is performed.

ii.
```python
BIN_WIDTH = 0.05
BIN_EDGES = np.round(T_START + BIN_WIDTH * np.arange(...), 10)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
```

iii. The width and window come directly from the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial `start_time`, and each trial's go-cue time. The last sample onset assigned to a trial and no later than its go cue is used.

ii.
```python
sample = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
trial_of = np.searchsorted(start, sample, side='right') - 1
for t, s in zip(trial_of[valid], sample[valid]):
    if s <= go[t]: tone[t] = s
rel = tone - go
```

iii. Early licking can replay the sample epoch, so the trajectory says the final tone before go is the behaviorally relevant onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is expressed relative to go, missing offsets are replaced by the session median, and each retained bin center subtracts that relative tone time.

ii.
```python
rel = tone - go
if np.any(np.isnan(rel)):
    rel[np.isnan(rel)] = np.nanmedian(rel)
...
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. This converts go-relative centers into elapsed seconds from tone. The median fallback was defensive; the agent reports finding no missing sample onset in this dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same retained go-relative bin centers and sliced by the same bin index array as neural data.

ii.
```python
b = bins[t]
neural.append(rates[:, k, b])
inp[0] = BIN_CENTERS[b] - tone_rel[t]
```

iii. Shared bin centers and retained-bin indices guarantee one input value per neural timepoint.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from trials-table `photostim_onset`, `photostim_duration`, and `start_time`, plus go-cue time.

ii.
```python
onset = float(tbl['photostim_onset'][i])
dur = float(tbl['photostim_duration'][i])
on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
off[i] = on[i] + dur
```

iii. The agent verified in the trajectory that onset is relative to trial start and matches recorded photostim events; `N/A` denotes unstimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset and offset are converted to go-relative seconds. A retained bin is 1 when its center is in the half-open stimulation interval, otherwise 0; unstimulated trials are all zero.

ii.
```python
inp[1] = ((BIN_CENTERS[b] >= stim_on[t]) &
          (BIN_CENTERS[b] < stim_off[t])).astype(np.float32)
```

iii. A binary time series represents whether light is on at every neural sample, as requested.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation times are converted to the same go-relative clock and tested at the same retained bin centers used for neural data.

ii.
```python
on[i] = tbl['start_time'][i] + onset - tbl['go_time'][i]
...
BIN_CENTERS[b] >= stim_on[t]
```

iii. The shared clock and bin index ensure exact temporal correspondence.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from absolute `left_lick_times`, `right_lick_times`, and each trial's go cue, rather than from instruction and outcome.

ii.
```python
return (np.sort(be['left_lick_times/timestamps'][:]),
        np.sort(be['right_lick_times/timestamps'][:]))
...
li = np.searchsorted(left, g)
ri = np.searchsorted(right, g)
```

iii. The agent argued direct port events are the actual behavioral measurement. It checked that they agree with instruction × outcome on more than 99.3% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first left or right lick from go cue through (but excluding) go + 1.5 s determines class 0 or 1; no lick gives class 2. This per-trial class is repeated across retained time bins.

ii.
```python
choice = np.full(len(go), 2, dtype=np.int8)
...
if tl >= end and tr >= end: continue
choice[i] = 0 if tl <= tr else 1
...
outp[0] = choice[t]
```

iii. This implements left/right/no-lick directly and uses the specified response window. Repetition permits all outputs to share one time-indexed array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` strings.

ii.
```python
'outcome': _str_col(trials['outcome'])
```

iii. The source already contains exactly the requested three categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to `ignore=0`, `miss=1`, and `hit=2`; the per-trial value is repeated across retained bins.

ii.
```python
outcome_code = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_code[o] for o in tbl['outcome']], dtype=np.int8)
...
outp[1] = outcome[t]
```

iii. The ordering matches the requested output values and preserves the categorical per-trial label.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
'early_lick': _str_col(trials['early_lick'])
```

iii. The trial table explicitly records whether an early lick occurred.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1 and all `'no early'` values to 0; the value is repeated across retained bins.

ii.
```python
early = (tbl['early_lick'] == 'early').astype(np.int8)
...
outp[2] = early[t]
```

iii. This gives the requested no/yes coding while keeping early-lick trials rather than filtering them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps and data columns 1 (y) and 2 (DeepLabCut likelihood), plus lick event times for session QC.

ii.
```python
ts = f[key + '/timestamps'][:]
data = f[key + '/data'][:]
y = data[:, 1].astype(np.float64)
visible = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. This is the dataset's side-camera tongue trace. Likelihood distinguishes genuine protrusions from meaningless coordinates while the tongue is retracted.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames above likelihood 0.5 are visible. Isolated visible-frame velocity outliers are detected at five standard deviations and interpolated. Visible y values are averaged per 50-ms trial bin; no visible frame produces NaN/class 3. Video coverage and lick agreement can reject a session.

ii.
```python
visible = data[:, 2] > 0.5
...
outlier[1:-1] = jump[:-1] & jump[1:]
yv = np.interp(np.arange(len(yv)), keep, yv[keep])
...
if vis.any():
    bin_y[k, b] = tongue_y[sl][vis].mean()
```

iii. The likelihood threshold exploits its bimodal distribution. Five-sigma velocity cleanup follows the method paper. Averaging into the same 50-ms units makes discretization and neural alignment comparable.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over finite, observed 50-ms bin means. Values below p40 are 0, p40 through p60 inclusive are 1, above p60 are 2, and bins without visible frames are 3. If fewer than 10 valid bins exist, both edges become infinity.

ii.
```python
p40, p60 = np.percentile(vis_vals, (40.0, 60.0))
tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
tongue_class[vis_bin & (bin_y < p40)] = 0
tongue_class[vis_bin & (bin_y >= p40) & (bin_y <= p60)] = 1
tongue_class[vis_bin & (bin_y > p60)] = 2
```

iii. The percentile levels and per-session scope come from the instructions. Percentiles use bin means because those are the values being classified.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps and go cues share the session clock. `searchsorted` finds frames in every absolute go-relative bin, and the final tongue classes are sliced by the same retained-bin indices as neural data.

ii.
```python
lo = np.searchsorted(ts, go[t] + BIN_EDGES[:-1])
hi = np.searchsorted(ts, go[t] + BIN_EDGES[1:])
...
outp[3] = tongue_class[k, b]
```

iii. No clock correction is needed; identical edge intervals and the shared `b` index align both streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/unusable sessions are rejected for no good units, absent/low-coverage video, or poor tongue/lick agreement. Free-water, too-short, and all-zero-spike trials are dropped. Silent or unannotated units are dropped. Missing sample onsets get the session-median offset. Invisible tongue bins become explicit class 3 rather than imputed; detected y outliers are interpolated.

ii.
```python
if ts is None: return rejected
if coverage < MIN_VIDEO_COVERAGE: return rejected
...
rel[np.isnan(rel)] = np.nanmedian(rel)
...
tongue_class = np.full((len(trials), NBINS), 3, dtype=np.int8)
```

iii. The trajectory distinguishes recording absence from meaningful missingness: unusable recordings are excluded, retracted tongue is categorical, and rare isolated tracking errors are repaired according to the method paper.

## 10-a. What are the most time-consuming steps of the code?

i. NWB/HDF5 reading, per-unit spike binning, per-frame tongue binning, writing large per-session caches, and finally serializing the approximately 9.5-GB dataset dominate.

ii.
```python
for i, u in enumerate(good):
    st = units['spike_times'][lo:spike_index[u]]
    counts = np.diff(np.searchsorted(st, edges)...)
...
for k, t in enumerate(trials):
    for b in range(NBINS): ...
```

iii. The trajectory repeatedly monitored multi-minute conversion and validation runs and introduced a 24-worker pool plus caches. Its final conversion contains 75,670 trials and 57,774 neurons, making neural payload construction and disk I/O the largest work.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-stimulated-trial interval loop, per-trial lick-choice loop, unit loop, trial mask loops, nested trial × tongue-bin loop, and final per-trial assembly could potentially be vectorized. The unit loop already vectorizes all trials for each ragged unit.

ii.
```python
for i in idx: ...
for i, g in enumerate(go): ...
for i, u in enumerate(good): ...
for k, t in enumerate(trials):
    ...
    for b in range(NBINS): ...
```

iii. Ragged spike trains make the unit loop difficult to eliminate, but the nested tongue loop could use global frame/bin indices and grouped reductions. The agent favored clear loops and session-level multiprocessing.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly builds/searches the same go-relative edge grid per unit, recomputes per-trial bin masks in several loops, traverses trial bins for video and assembly, and writes then rereads every session cache during a single run.

ii.
```python
bins = [mask_bins(tbl, t) for t in range(...)]
...
for k, t in enumerate(trials): observed[k, bins[t]] = True
...
for k, t in enumerate(trials): b = bins[t]
```

iii. The cache repetition is an intentional resumability tradeoff. Edge searches are necessarily repeated per ragged spike train; retained-bin arrays are reused, though iterated more than once.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes performance/QC diagnostics, lick-agreement statistics, video masks, rejection metadata, trial metadata, ontology groupings, and per-session caches that the decoder itself does not consume. It also bins full 80-bin neural windows before slicing away unobserved bins and computes rates for trials later found to have no spikes.

ii.
```python
precision, recall = tongue_lick_agreement(...)
has_video = np.zeros((len(trials), NBINS), dtype=bool)
rates = np.zeros((len(good), len(trials), NBINS), dtype=np.float32)
...
neural.append(rates[:, k, b])
```

iii. Most extra work supports QC, provenance, or robust reruns rather than decoder tensors. The trajectory explicitly used these diagnostics to justify session exclusions and verify the final artifact, so they are operationally useful even if downstream training discards them.
