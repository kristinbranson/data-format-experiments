# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every NWB file under `/app/data/sub-*/*.nwb`, sorts the paths, and processes files in a multiprocessing pool using `h5py`. Each file is first opened for session metadata and then reopened for full conversion.

ii.
```python
paths = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```

iii. The trajectory says the agent established that there are 174 NWB sessions and 28 mice, and chose parallel per-session conversion because the files are independent and the final dataset is large.

## 1-b. How are the data split into subjects?

i. The subject is read from each NWB file's `general/subject/subject_id`. During assembly, subjects are accumulated in first-seen order and each retained session receives the corresponding integer index.

ii.
```python
subject=f['general/subject/subject_id'][()].decode(),
...
if s['subject'] not in subjects:
    subjects.append(s['subject'])
subject_idx.append(subjects.index(s['subject']))
```

iii. The agent treated the NWB subject field as the canonical mouse identifier and reported 25 mice after its session filtering.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. However, only sessions passing behavioral criteria (>65% control/non-early hit rate, at least 50 correct left and 50 correct right trials) and containing a QC-good unit are retained.

ii.
```python
used = (perf > MIN_PERFORMANCE and ncl >= MIN_CORRECT_PER_DIRECTION
        and ncr >= MIN_CORRECT_PER_DIRECTION and info['ngood'] > 0)
```

iii. The agent justified this with the data paper's inclusion criteria and noted that they select 106 sessions, matching the method paper; one of those has no good unit, leaving 105.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. Go and sample events are assigned to trials by locating the preceding trial start; the first go and latest sample onset before that go are retained.

ii.
```python
idx = np.searchsorted(start, go, 'right') - 1
...
idx = np.searchsorted(start, samp, 'right') - 1
```

iii. The trajectory notes that early licking can replay sample/delay epochs, so a trial may contain multiple sample events; the last pre-go sample is the relevant tone.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps trials with go and tone events that are neither free-water nor auto-water and occur in unit observation intervals. It additionally removes trials whose retained neural window contains zero spikes, and drops a session if fewer than two trials remain.

ii.
```python
keep = (~np.isnan(go)) & (~np.isnan(tone)) & (~info['free_water']) & (~info['auto_water'])
keep &= observed_trials(f, start)
...
nonempty = rates.sum(axis=(0, 2)) > 0
```

iii. It found that nine recordings cover only a prefix of behavioral trials and that one retained trial was wholly silent because recording ended. Early-lick, error, ignore, and photostimulation trials were deliberately retained because they are requested decoder variables.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times`, using `units/spike_times_index` to slice each unit, `units/classification` to select units, and go-cue events to define trial windows.

ii.
```python
good = np.where(u['classification'][:] == b'good')[0]
sti = np.asarray(u['spike_times_index'][:])
spikes = np.asarray(u['spike_times'][a:sti[iu]])
```

iii. The agent identified spike times as the available neural representation and the classifier's `good` label as the paper's QC verdict.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes are sorted if necessary, counted in every 50-ms bin with `searchsorted` and differencing, and divided by 0.05 s to yield firing rates in spikes/s. There is no smoothing or normalization.

ii.
```python
pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
rates[k] = np.diff(pos, axis=1).astype(np.float32) / BIN_SIZE
```

iii. The agent said this follows the reference firing-rate calculation while replacing its native sliding bins with the task-required non-overlapping 50-ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `good` are included; sessions without such units are rejected. No separate firing-rate cutoff is applied.

ii.
```python
ngood=int((f['units']['classification'][:] == b'good').sum())
good = np.where(u['classification'][:] == b'good')[0]
```

iii. The trajectory links this field to the region-specific classifier in the spike-sorting white paper and explicitly rejects an unrelated 2-Hz filter used for another analysis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are formed from each trial's go cue plus offsets from -2.5 to +1.5 seconds; session-absolute spike times are counted against those edges.

ii.
```python
edges = g[:, None] + OFF_START + np.arange(N_BINS + 1)[None, :] * BIN_SIZE
```

iii. The agent determined that spikes and behavioral events share the same clock, so no clock correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The result has 80 non-overlapping 50-ms bins spanning four seconds. Raw spike timestamps are histogrammed directly into those bins; no further rebinning occurs.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The task explicitly requires 50-ms bins, so the agent used them instead of the reference analysis's native sliding-window parameters.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, trial `start_time`, go-cue times, and the centers of the neural bins.

ii.
```python
samp = _event_times(f, 'sample_start_times')
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. The last sample onset before go is selected because early licks can cause sample-epoch replay.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Sample events are mapped to trials, the latest eligible event is selected, and its absolute timestamp is subtracted from every absolute bin center.

ii.
```python
if np.isnan(out[i]) or s > out[i]:
    out[i] = s
dt_tone = (centers - tone[trials][:, None]).astype(np.float32)
```

iii. The agent avoided clipping even though replayed trials can produce large elapsed times, because clipping was not specified.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same absolute bin centers used by the neural trial grid, so it has one value per neural bin.

ii.
```python
centers = 0.5 * (edges[:, :-1] + edges[:, 1:])
dt_tone = centers - tone[trials][:, None]
```

iii. Shared session timestamps and a shared bin grid provide direct alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`.

ii.
```python
ps = _event_times(f, 'photostim_start_times')
pe = _event_times(f, 'photostim_stop_times')
```

iii. The agent preferred the explicit event streams, which it found occur in the final delay period.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A neural bin is set to 1 if its half-open interval overlaps any stimulation interval; otherwise it remains 0.

ii.
```python
ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
stim_on[ov] = 1.0
```

iii. This produces the requested binary, time-varying input directly from onset/offset events.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are compared to the same absolute per-trial edges used for spike binning.

ii.
```python
ov = (edges[:, 1:] > s_) & (edges[:, :-1] < e_)
```

iii. All streams use the same NWB session clock.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `outcome` and `trial_instruction`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
choice[hit & (instruction == 'left')] = 0
choice[miss & (instruction == 'left')] = 1
```

iii. The trajectory notes that no direct choice column exists and that instruction plus outcome uniquely determines the requested class.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, 2 no lick and repeated across all 80 time bins.

ii.
```python
choice = np.full(len(trials), 2, dtype=np.int64)
np.repeat(choice[:, None], N_BINS, axis=1)
```

iii. Repetition lets per-trial and time-varying outputs share one `(4, 80)` array per trial.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` field.

ii.
```python
outcome = info['outcome'][trials]
```

iii. The stored categories already match the task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Ignore, miss, and hit are encoded as 0, 1, and 2 respectively, then repeated across time.

ii.
```python
outcome_code = np.zeros(len(trials), dtype=np.int64)
outcome_code[miss] = 1
outcome_code[hit] = 2
```

iii. The encoding follows the requested output order.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
early = info['early'][trials]
```

iii. The NWB table explicitly records the flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other expected value (`no early`) maps to 0; the code is repeated across bins.

ii.
```python
early_code = (early == 'early').astype(np.int64)
```

iii. This implements the requested no/yes ordering.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`: data column 1 is y, column 2 is DeepLabCut likelihood, and the series timestamps align frames.

ii.
```python
data = np.asarray(grp[key]['data'][:])
ts = np.asarray(grp[key]['timestamps'][:])
y = data[:, 1]
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
```

iii. The agent identified side-view tracking as the relevant tongue stream used by the paper.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at most 0.9 or nonfinite y are excluded. Visible y values are summed and counted by neural bin using cumulative arrays, producing a mean visible y per bin; bins without visible frames receive class 3.

ii.
```python
vis = data[:, 2] > TONGUE_LIKELIHOOD_THRESH
mean_y = np.where(n > 0, s / np.maximum(n, 1), np.nan)
code = np.full((ntrials, nbins), 3, dtype=np.int64)
```

iii. The agent observed a strongly bimodal likelihood distribution and concluded that 0.9 would reliably identify protruded/visible tongue frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over all visible raw frame y-values. Each bin's mean is class 0 below the lower edge, class 1 inclusively between edges, class 2 above the upper edge, or class 3 if unseen.

ii.
```python
lo, hi = np.percentile(y[vis], [40.0, 60.0]) if vis.sum() else (0., 0.)
code[seen & (mean_y < lo)] = 0
code[seen & (mean_y >= lo) & (mean_y <= hi)] = 1
code[seen & (mean_y > hi)] = 2
```

iii. The agent followed the requested per-session percentile boundaries, but chose raw visible frames as the percentile population.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are searched at every absolute neural bin edge, and frame sums/counts between adjacent edge indices form the corresponding bin value.

ii.
```python
pos = np.searchsorted(ts, edges.ravel()).reshape(ntrials, nedges)
n = cum_n[pos[:, 1:]] - cum_n[pos[:, :-1]]
```

iii. The agent relied on the shared NWB clock and exact neural edges, avoiding interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing go/tone trials, water trials, trials outside ephys observation intervals, wholly silent trials, sessions without good units, and sessions left with fewer than two trials are removed. Missing tongue series or bins become class 3. Unsorted spikes are sorted. Worker exceptions are caught and cause the session to be skipped.

ii.
```python
if spikes.size and np.any(np.diff(spikes) < 0):
    spikes = np.sort(spikes)
if key not in grp:
    return np.full((ntrials, nbins), 3, dtype=np.int64)
```

iii. The trajectory documents empirical checks for truncated ephys coverage and one all-zero terminal trial; the agent preferred exclusion rather than representing absent neural recordings as genuine zero firing.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is reading large HDF5 spike/video arrays, per-unit spike binning, writing per-session temporary pickles, assembling them, and writing the roughly 7.2-GB final pickle. Multiprocessing makes session conversion fast.

ii.
```python
with Pool(args.nproc) as pool:
    files = pool.map(_worker, paths)
```

iii. The trajectory reports full conversion in under a minute from cached files and identifies the full decoder training, rather than conversion, as the longest validation step.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Event-to-trial loops (`go_cue_times`, `tone_onset_times`), the loop over good units, keyword-based region mapping, and photostimulation interval loop remain. Trial spike searches are already vectorized within each unit, while tongue binning is fully vectorized by cumulative sums.

ii.
```python
for k, iu in enumerate(good):
    ...
    pos = np.searchsorted(spikes, flat_edges).reshape(len(trials), N_BINS + 1)
```

iii. Ragged spike trains make a unit loop natural; the trajectory prioritized correctness and session-level parallelism.

## 10-c. What processing does the code repeat multiple times?

i. Every retained NWB is opened once by `session_info` and again by `process_session`. Data is also serialized to a temporary session pickle and later deserialized before final pickling. Region-list membership and index lookup are repeatedly linear.

ii.
```python
info = session_info(path)
...
with h5py.File(path, 'r') as f:
```

iii. The agent used a small first read to decide session inclusion and temporary files to limit interprocess transfer/assembly complexity; it did not explicitly discuss the duplicated I/O as a drawback.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Session-selection diagnostics (`performance`, correct-left/right counts), `frac_observed`, detailed metadata text, and coarse-region derivation are not neural-decoder features, although some are preserved as metadata. Temporary per-session serialization and construction of `centers`/`edges` copies are conversion overhead. No major computed signal is silently discarded.

ii.
```python
performance=perf,
ncorrect_left=ncl, ncorrect_right=ncr,
frac_observed=float(...),
```

iii. The agent retained these values for provenance and sanity checking; its trajectory emphasizes verification and paper-consistent documentation rather than minimizing metadata work.
