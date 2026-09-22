# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes each file with `h5py`, normally using a 16-process pool. It reads trials, units, events, and video directly from each NWB hierarchy.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with Pool(min(args.nproc, len(files))) as pool:
    results = pool.map(_worker, list(zip(files, plot_flags)), chunksize=1)
with h5py.File(path, 'r') as f:
    u = f['units']
    t = f['intervals/trials']
```

iii. The notes say the release contains 174 NWB files under 28 subject folders and that one NWB file represents one session. Direct HDF5 access, single-pass reads, and session parallelism were chosen for speed.

## 1-b. How are the data split into subjects?

i. Each session reads `general/subject/subject_id`; assembly sorts unique IDs and creates a session-wise integer index.

ii.
```python
subject = f['general/subject/subject_id'][()].decode()
subjects = sorted({r['subject'] for r in results})
subject_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([subject_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The agent treats the NWB subject field as canonical and reports 28 mice, matching the paper.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Its `identifier` and start time are retained; output session order is sorted pathname order, except unusable sessions are omitted.

ii.
```python
sess_id = f['identifier'][()].decode()
session_start = f['session_start_time'][()].decode()
```

iii. The notes identify 174 files but 173 sessions with at least one good unit, matching the published 173-session count.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`; one go-cue timestamp is required per row. A retained-index array (`trial_idx`) is applied consistently to all streams.

ii.
```python
trial_start = t['start_time'][:]
ntrials_file = len(trial_start)
go_all = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
if len(go_all) != ntrials_file:
    return None
trial_idx = np.where(keep)[0]
```

iii. The trials table is the explicit NWB trial definition; the one-to-one go-cue check guards against ambiguous alignment.

## 1-e. How are trials filtered based on quality controls?

i. The agent removes auto-water and free-water trials, nonfinite go cues, trials not covered by every selected unit's `obs_intervals`, and later any trial whose entire selected-neuron window contains zero spikes. Sessions with fewer than two survivors are dropped. Early-lick, ignore/miss, and photostimulation trials are retained.

ii.
```python
keep = ~(auto_water | free_water)
keep &= np.isfinite(go_all)
keep &= observed_trial_mask(u, unit_ids, trial_start, trial_stop)
has_spikes = fr.sum(axis=(0, 2)) > 0
```

iii. The notes justify water-trial removal using the reference regular-trial mask, retain target/input-defining conditions, and exclude unobserved/all-zero neural trials to avoid representing missing recordings as 0 Hz. This produced 89,544 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values derive from `units/spike_times` and its ragged index, restricted by `units/classification`, `units/anno_name`, and aligned using go-cue timestamps.

ii.
```python
classification = u['classification'][:]
good = (classification == b'good')
good &= np.array([a.strip() != '' for a in anno])
spikes = u['spike_times'][:]
s0, s1 = _spike_slices(u['spike_times_index'][:])
```

iii. The agent identifies spike times as the raw neural representation and the classifier verdict as the paper's region-specific QC output.

## 2-b. How is the `neural` data processed?

i. For each good unit, absolute bin edges for every trial are searched in its sorted spike vector. Adjacent cumulative indices are differenced into counts and divided by 0.05 s to obtain float32 firing rates in Hz; there is no smoothing or normalization.

ii.
```python
idx = np.searchsorted(sp, flat).reshape(ntrials, NBINS + 1)
out[i] = np.diff(idx, axis=1) / BIN_SIZE
```

iii. The notes state this matches the reference `sliding_histogram(rate=True)` rate definition while using the decoder-required nonoverlapping 50-ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == b'good'` and a nonempty CCF annotation. Sessions with no such units are skipped; no 2-Hz firing-rate cutoff is used.

ii.
```python
good = (classification == b'good')
good &= np.array([a.strip() != '' for a in anno])
unit_ids = np.where(good)[0]
if len(unit_ids) == 0:
    return None
```

iii. The agent says classifier-good plus histology matches the reference classifier mode; it rejects the encoding-analysis-specific 2-Hz cutoff because the decoder can use all QC-passing units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each relative edge from -2.5 to +1.5 s is added to that trial's absolute go-cue timestamp, and absolute spike times are binned against those edges.

ii.
```python
go = go_all[trial_idx]
edges_abs = go[:, None] + BIN_EDGES[None, :]
fr = bin_spikes(spikes, s0, s1, unit_ids, edges_abs)
```

iii. The notes explain that spikes, behavioral events, and video share the NWB session clock, so adding go time performs the required alignment without interpolation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins spanning [-2.5, +1.5) s around go cue. Raw spike times are directly histogrammed to this grid; the paper's 40-ms/3.4-ms sliding representation is not first computed and rebinned.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. This is explicitly required by the decoder task; retaining the reference rate formula preserves applicable processing.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses behavioral-event `sample_start_times`, trials' start times, and go cues. The last sample onset assigned to each trial before its go cue is the tone onset.

ii.
```python
sample_start = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
tone = tone_onset_times(trial_start, go_all, sample_start)[trial_idx]
```

iii. Early licks can replay the sample epoch, so the agent selects the final instruction tone preceding go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Sample events are assigned to trials by `searchsorted`; the latest valid event is kept. Missing tone values are imputed as go minus 1.85 s. Each value is the absolute neural-bin center minus tone onset.

ii.
```python
idx = np.searchsorted(trial_start, sample_start, side='right') - 1
tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
time_from_tone = (centers_abs - tone[:, None]).astype(np.float32)
```

iii. The 1.85-s fallback is the nominal 0.65-s sample plus 1.2-s delay and is described as rare; otherwise no transformation beyond elapsed time is needed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same absolute 50-ms bin centers used by each go-aligned neural trial.

ii.
```python
centers_abs = go[:, None] + BIN_CENTERS[None, :]
time_from_tone = centers_abs - tone[:, None]
```

iii. Sharing bin centers guarantees one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It preferentially uses BehavioralEvents `photostim_start_times` and `photostim_stop_times`; if unavailable, it reconstructs intervals from trials' `photostim_onset`, `photostim_duration`, and `start_time`.

ii.
```python
on = be['photostim_start_times/timestamps'][:]
off = be['photostim_stop_times/timestamps'][:]
# fallback
a = trial_start[i] + float(onset[i])
iv.append([a, a + float(dur[i])])
```

iii. The event streams provide absolute intervals directly; the trial-table path is a robustness fallback.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A float32 binary matrix is initialized to zero. A bin is set to one if its interval has any strict overlap with any photostimulation interval.

ii.
```python
photostim = np.zeros((len(trial_idx), NBINS), dtype=np.float32)
for a, b in stim_iv:
    photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. The agent chose overlap so partially illuminated bins count as on, yielding a time-varying binary decoder input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation intervals are compared against the same absolute go-aligned bin edges used for spikes.

ii.
```python
edges_abs = go[:, None] + BIN_EDGES[None, :]
photostim[(edges_abs[:, 1:] > a) & (edges_abs[:, :-1] < b)] = 1.0
```

iii. All timestamps share the session clock, so no offset correction is applied.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial `outcome` and `trial_instruction`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
oc = outcome[trial_idx]
instr = instruction[trial_idx]
licked_left = ((oc == 'hit') & (instr == 'left')) | ((oc == 'miss') & (instr == 'right'))
```

iii. The notes say choice is not stored explicitly and report 99.4% agreement with a first-post-go-lick cross-check.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Codes are 0 no lick, 1 left, and 2 right, then repeated across all 80 time bins.

ii.
```python
choice = np.zeros(len(trial_idx), dtype=np.int64)
choice[licked_left] = 1
choice[licked_right] = 2
o[0] = choice[i]
```

iii. Repetition lets per-trial and time-varying outputs share one `(4, 80)` array; `OUTPUT_VALUES` documents the chosen code order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome` after applying `trial_idx`.

ii.
```python
outcome = _decode(t['outcome'][:])
oc = outcome[trial_idx]
```

iii. The NWB values already are the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Ignore, miss, and hit map to 0, 1, and 2 with `np.select`; the scalar trial label is repeated through time.

ii.
```python
outcome_code = np.select([oc == 'ignore', oc == 'miss', oc == 'hit'], [0, 1, 2], default=0)
o[1] = outcome_code[i]
```

iii. The mapping follows the requested ordering. The agent uses default 0 as a defensive fallback.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` strings.

ii.
```python
early = _decode(t['early_lick'][:])
el_tr = early[trial_idx]
```

iii. The source explicitly stores the requested per-trial flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1 and every other value to 0; it is repeated across bins.

ii.
```python
early_code = (el_tr == 'early').astype(np.int64)
o[2] = early_code[i]
```

iii. This implements no/yes categorical encoding and the common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It searches for side-camera `Camera0_side_TongueTracking` or `Camera3_side_TongueTracking`; column 1 is y, column 2 likelihood, with the series timestamps.

ii.
```python
for k in ('Camera0_side_TongueTracking', 'Camera3_side_TongueTracking'):
    if k in bt: key = k; break
data = bt[key]['data'][:]
ts = bt[key]['timestamps'][:]
y, lik = data[:, 1], data[:, 2]
```

iii. The side-camera DLC stream is the dataset's tongue measurement; Camera3 support is a missing-stream fallback.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only when likelihood is greater than 0.9 and y is finite. Visible y values are averaged inside each trial's neural bins using cumulative sums; bins without a visible frame become NaN/not visible.

ii.
```python
visible = (lik > TONGUE_LIKELIHOOD_THRESHOLD) & np.isfinite(y)
idx = np.searchsorted(ts, edges_abs)
nvis = cs_n[hi] - cs_n[lo]
mean_y = np.where(nvis > 0, sumy / np.maximum(nvis, 1), np.nan)
```

iii. The agent says DLC likelihood is strongly bimodal, making the exact cutoff largely irrelevant, and averaging matches the 50-ms target resolution.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Within each session, percentiles are computed from the mean-y values in visible bins of retained trial windows. Values below p40 are 0, p40 through p60 inclusive are 1, above p60 are 2, and invisible bins are 3.

ii.
```python
p40, p60 = np.percentile(mean_y[visible], [40, 60])
cls[visible & (mean_y < p40)] = 0
cls[visible & (mean_y >= p40) & (mean_y <= p60)] = 1
cls[visible & (mean_y > p60)] = 2
```

iii. The notes interpret “per-session” as all visible converted bins for that session and use bin means so thresholds apply to the same quantity being categorized.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are searched at every absolute neural-bin edge; frames between successive edges are averaged into the corresponding bin.

ii.
```python
idx = np.searchsorted(ts, edges_abs)
lo, hi = idx[:, :-1], idx[:, 1:]
```

iii. The agent relies on the shared session clock and identical edges, requiring neither interpolation nor a stream offset.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Workers catch session exceptions and skip failures; sessions lacking QC units are skipped; mismatched go counts skip the session; too few usable trials skip it; absent tongue streams yield all “not visible”; missing tone events use a nominal 1.85-s offset; photostimulation can fall back to trial columns; and unrecorded/all-zero neural trials are dropped.

ii.
```python
except Exception as exc:
    return None
if key is None:
    return (np.full((ntrials, NBINS), np.nan), np.zeros((ntrials, NBINS), dtype=bool))
tone[missing] = go[missing] - DEFAULT_SAMPLE_TO_GO
```

iii. The notes distinguish explicit missing categories (tongue) from absent neural recordings, which are excluded, and favor continuing conversion despite a bad file.

## 10-a. What are the most time-consuming steps of the code?

i. Reading the large spike/video datasets, per-unit spike binning, and writing the multi-gigabyte pickle are the principal costs. Session multiprocessing reduced the measured full processing phase to 24.2 s; serialization remains separate.

ii.
```python
spikes = u['spike_times'][:]
for i, k in enumerate(unit_ids):
    idx = np.searchsorted(sp, flat)
with Pool(...) as pool:
    results = pool.map(...)
```

iii. The agent's notes identify I/O and repeated spike searches as scale-dependent bottlenecks and report the achieved parallel speedup.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorizes trials in spike binning and all trial/time bins in tongue averaging. Remaining loops are over units, selected-unit observation intervals, stimulation intervals, sample events, unit region mapping, and final per-trial output assembly. Some bookkeeping loops could be vectorized; the ragged per-unit spike loop is difficult to eliminate cleanly.

ii.
```python
for i, k in enumerate(unit_ids):
    sp = spike_times[starts[k]:stops[k]]
for k in unit_ids:
    iv = oi[starts[k]:oi_index[k]]
for i in range(len(trial_idx)):
    outputs.append(o)
```

iii. The notes specifically say nested trial/bin neural loops were replaced by one search per unit and video binning by cumulative sums; session-level multiprocessing handles coarse parallelism.

## 10-c. What processing does the code repeat multiple times?

i. Per session it repeatedly scans selected units for QC regions, observation coverage, and spike binning; it also builds each trial's output/list object individually. With `--show-processing`, selected sessions reopen the NWB file and reread unit metadata/spikes for plots. Ordinary full conversion otherwise reads each source file once.

ii.
```python
for k in unit_ids:  # region assignment
for k in unit_ids:  # observed_trial_mask
for i, k in enumerate(unit_ids):  # bin_spikes
with h5py.File(path, 'r') as f:  # plot_processing reopens file
```

iii. The agent characterizes the normal conversion as single-pass and intentionally reuses module-level bin grids; plotting is optional diagnostic work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal full conversion retains all main computed arrays, but computes timing dictionaries and `trial_idx` that are only diagnostic and are not copied into the final pickle. It also reads `trial_stop`, passes unused `go` parameters to helpers, and calculates session durations. Optional plotting recomputes/reads data solely for figures and is discarded by the decoder.

ii.
```python
trial_stop = t['stop_time'][:]
timing['neural'] = time.time() - t0
'trial_idx': trial_idx,
# final data omits timing and trial_idx
```

iii. The notes emphasize diagnostic validation and timing; these costs are small, while `--show-processing` is explicitly optional rather than part of the full saved dataset.
