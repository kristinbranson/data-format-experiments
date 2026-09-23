# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent finds every `*.nwb` file one directory below `/app/data`, sorts the paths, and processes each file with `h5py` in a multiprocessing pool. Within each file it reads the trials, behavioral events, units, electrodes, and side-camera tracking directly from NWB HDF5 groups.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, '*', '*.nwb')))
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, [(fn, name2anc) for fn in files]):
```
```python
with h5py.File(fname, 'r') as f:
    out = _process_session(f, fname, name2anc, info)
```

iii. The trajectory established that the release consists of 174 NWB session files and that NWB contains all required streams on a common clock. The agent chose parallel session processing because the complete output is large but the machine had ample RAM and CPUs.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. Unique IDs are sorted into `subjects`, and each retained session receives the corresponding index in `subject_idx`.

ii.
```python
subject = str(np.asarray(f['general/subject/subject_id'][()]).astype(str))
subjects = sorted({r['subject'] for r in results})
'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
```

iii. The NWB subject field is the canonical identifier, so the agent did not infer subjects from file names.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Retained session results become one element of each of the top-level `neural`, `input`, and `output` lists. The file-name session timestamp is recorded in metadata.

ii.
```python
info['session'] = os.path.basename(fname).split('_')[1].replace('ses-', '')
'neural': [r['neural'] for r in results],
```

iii. The agent found that all probes for a recording are already merged into the file's units table and therefore used the NWB file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. Rows in `intervals/trials` define trials. Go-cue events are assigned to those rows by locating the trial interval containing each event; the final retained trial indices are used consistently for all streams.

ii.
```python
trial_start = tr['start_time'][:]
trial_stop = tr['stop_time'][:]
gidx = map_events_to_trials(go_all, trial_start, trial_stop)
go = np.full(ntrials_all, np.nan)
go[gidx[gidx >= 0]] = go_all[gidx >= 0]
```

iii. The trajectory verified exactly one go cue per trial, while sample events can repeat after an early lick, so the trials table—not sample-event count—was used as authoritative.

## 1-e. How are trials filtered based on quality controls?

i. The agent first excludes entire sessions unless control-trial hit rate is above 65% with at least 50 correct trials per direction. It excludes sessions with unusable video. Within retained sessions it drops auto-water, free-water, missing-go, poor-video-coverage, and trials not observed by every selected unit. After constructing rates it also drops trials with a missing tone/nonfinite input or zero population spikes. Early-lick, ignore, and photostimulation trials are retained.

ii.
```python
if not (perf > PERF_THRESH and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    return None
keep = (~auto_water) & (~free_water) & good_video & np.isfinite(go)
keep = keep & observed
valid = [i for i, x in enumerate(input_trials)
         if np.all(np.isfinite(x)) and neural_trials[i].sum() > 0]
```

iii. The agent interpreted the papers' behavioral criteria as required curation because they select exactly the reported 106 sessions. Auto/free-water trials were deemed to lack meaningful choice/outcome; missing ephys/video/tone data were treated as absent observations. Required decoder targets motivated retaining early-lick and ignore trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times`, restricted to classifier-good units, and from per-trial `go_start_times`, which place bin edges.

ii.
```python
sp_index = f['units/spike_times_index'][:]
spike_ds = f['units/spike_times']
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. The agent found spike times to be the NWB neural representation and verified its bin counts against brute-force histograms.

## 2-b. How is the `neural` data processed?

i. For each retained unit, sorted spike times are searched at all trial-bin edges. Adjacent cumulative counts are differenced and divided by 0.05 s to produce unsmoothed firing rates in spikes/s (`float32`).

ii.
```python
counts_sorted = np.searchsorted(st, sorted_edges)
counts[order] = counts_sorted
counts = counts.reshape(len(trials), NBINS + 1)
rates[i] = np.diff(counts, axis=1).astype(np.float32) / BIN_SIZE
```

iii. This implements the requested firing-rate representation while changing the papers' original binning only where the decoder instructions explicitly require 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'`. In addition, a unit is removed if `is_good_trials` is false on any analyzed trial. Sessions with no surviving units are removed.

ii.
```python
good_unit = classification == 'good'
unit_idx = np.flatnonzero(good_unit)
...
unit_ok[i] = bool(flag[keep].all())
unit_idx = unit_idx[unit_ok]
```

iii. The agent tied `classification` to the region-specific classifier in the spike-sorting paper. It interpreted `is_good_trials` as instability/drift information and required every retained neuron to be well isolated on every retained trial.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges spanning -2.5 to +1.5 s are formed by adding relative offsets to each trial's go-cue timestamp. Spikes and events are already on the same session clock.

ii.
```python
edges = (go[trials][:, None] + OFF_START
         + BIN_SIZE * np.arange(NBINS + 1)[None, :])
```

iii. Inspection showed no stream-specific synchronization was necessary; direct use of the shared clock aligns every trial to go-cue onset.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There are 80 non-overlapping 50 ms bins over the four-second window. Raw spikes are binned directly; there is no smoothing, interpolation, baseline subtraction, or later rebinning.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The task explicitly specifies 50 ms bins, overriding the papers' 40 ms sliding-window analysis.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial intervals, and each trial's go-cue time. The last sample/tone onset no later than the go cue is selected.

ii.
```python
sam_all = be['sample_start_times/timestamps'][:]
sidx = map_events_to_trials(sam_all, trial_start, trial_stop)
```

iii. Early licking can replay the sample epoch, so the agent chose the last pre-go tone as the tone relevant to the eventual response.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each 50 ms bin center, the absolute center time is subtracted from the selected tone onset, yielding seconds elapsed since tone onset.

ii.
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
time_from_tone = (g + bin_centers) - tone_onset[t]
```

iii. No further transformation was considered necessary because the requested input is continuous elapsed time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the centers of the same go-aligned bins used for neural firing rates.

ii.
```python
bin_centers = OFF_START + BIN_SIZE * (np.arange(NBINS) + 0.5)
inp = np.stack([time_from_tone.astype(np.float32), photostim])
```

iii. Both streams use the same global timestamps and bin grid, so each input column corresponds to the neural column at that timepoint.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, mapped to trials using trial start/stop intervals and then expressed relative to the go cue.

ii.
```python
pst = be['photostim_start_times/timestamps'][:]
psp = be['photostim_stop_times/timestamps'][:]
pidx = map_events_to_trials(pst, trial_start, trial_stop)
```

iii. The trajectory checked these event times against the trial-table photostimulation fields and chose the explicit event intervals already on the shared session clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. It produces a binary vector. A bin is 1 whenever its interval has any overlap with the photostimulation interval; all bins are 0 on unstimulated trials.

ii.
```python
a, b = ps_on[t] - g, ps_off[t] - g
photostim = ((bin_hi > a) & (bin_lo < b)).astype(np.float32)
```

iii. The agent represented stimulation as a time-varying on/off input as required and used interval overlap so partially stimulated bins count as on.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Start and stop timestamps are shifted by the same go cue used for neural alignment and compared with the same 50 ms bin bounds.

ii.
```python
bin_lo = OFF_START + BIN_SIZE * np.arange(NBINS)
bin_hi = bin_lo + BIN_SIZE
a, b = ps_on[t] - g, ps_off[t] - g
```

iii. Because the event and spike timestamps share a clock, no interpolation or synchronization correction is applied.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
if outcome[t] == 'hit':
    ch = instruction[t]
elif outcome[t] == 'miss':
    ch = other[instruction[t]]
else:
    ch = 'no lick'
```

iii. No direct choice column exists. The agent cross-checked this inference with recorded lick times and reported greater than 99% agreement.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The strings are mapped to 0 left, 1 right, and 2 no lick, then the per-trial category is repeated over all 80 bins.

ii.
```python
choice_map = {'left': 0, 'right': 1, 'no lick': 2}
np.full(NBINS, choice_map[ch], dtype=np.int64)
```

iii. Repetition allows the scalar trial label to share one `(n_output, n_timepoints)` array with the time-varying tongue output.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome` (`ignore`, `miss`, or `hit`).

ii.
```python
outcome = _str(tr['outcome'][:])
```

iii. The raw categories exactly match the requested output, so no behavioral inference is needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Categories are mapped to 0 ignore, 1 miss, and 2 hit and repeated across all bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
np.full(NBINS, outcome_map[outcome[t]], dtype=np.int64)
```

iii. The fixed codes follow the requested order; repetition is for uniform output shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`, whose values are `early` and `no early`.

ii.
```python
early = _str(tr['early_lick'][:])
```

iii. The NWB table already explicitly labels this per-trial behavior.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` maps to 1 and everything else (`no early`) maps to 0; the value is repeated across 80 bins.

ii.
```python
np.full(NBINS, 1 if early[t] == 'early' else 0, dtype=np.int64)
```

iii. The binary mapping matches the requested no/yes categories and uses the common output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and columns 1 (tongue y) and 2 (tracking likelihood) of `Camera0_side_TongueTracking`.

ii.
```python
vts = f[key + '/timestamps'][:]
vdata = f[key + '/data'][:]
tongue_y = vdata[:, 1].astype(np.float64)
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. Dataset inspection showed the three columns are tongue x, tongue y, and likelihood and that this is the relevant side-view tongue series.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only above likelihood 0.9. A five-standard-deviation velocity rule flags jumps and interpolates their y values. Visible y values are averaged within each trial's 50 ms bins; bins with no visible frame remain class 3. Sessions/trials with insufficient or malformed video are excluded.

ii.
```python
LIKELIHOOD_THRESH = 0.9
tongue_y, tongue_vis = clean_marker(tongue_y, tongue_vis)
ysum = np.bincount(b_idx[vis], weights=tongue_y[lo:hi][vis], minlength=NBINS)
ymean = np.where(nvis > 0, ysum / np.maximum(nvis, 1), np.nan)
```

iii. The agent viewed low-likelihood positions as retracted/not visible and adopted the method paper's five-sigma velocity cleaning. Averaging visible frames gives one value on the same time scale as neural bins.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The 40th and 60th percentiles are computed per session over all cleaned, visible **frame-level** y values. A visible bin mean is class 0 below p40, class 1 from p40 through p60, and class 2 above p60; a bin without a visible frame is class 3.

ii.
```python
p40, p60 = np.percentile(tongue_y[tongue_vis], [40, 60])
tclass[seen] = np.where(ymean[seen] < p40, 0,
                        np.where(ymean[seen] <= p60, 1, 2))
```

iii. The agent followed the requested per-session percentile split, reasoning that raw visible frames describe the session distribution; it explicitly reserved a fourth category for invisibility.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go-2.5, go+1.5)` are found with `searchsorted`, then assigned by floor division to the same 50 ms go-relative bins as neural data.

ii.
```python
lo = np.searchsorted(vts, g + OFF_START)
hi = np.searchsorted(vts, g + OFF_END)
b_idx = np.floor((vts[lo:hi] - (g + OFF_START)) / BIN_SIZE).astype(int)
```

iii. Camera, spike, and event timestamps share the session clock, so applying the identical go-aligned grid guarantees column-wise alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent drops sessions for exceptions, poor behavior, absent/short/nonmonotonic video, too few visible tongue frames, or no good units. It drops trials with missing go/tone, inadequate video, absent ephys observation, or zero population spikes. Low-confidence tongue frames become “not visible”; velocity outliers are interpolated. Unexpected `is_good_trials` layouts are conservatively treated as all true.

ii.
```python
except Exception as exc:
    info['excluded'] = 'error: %s' % exc
    return None, info
```
```python
if np.any(np.diff(vts) <= 0): return None
if len(row) == len(ti): flag[ti] = row
else: flag[:] = True
```

iii. The trajectory distinguishes missing acquisition from biological silence and therefore excludes absent-data sessions/trials. Legitimate tongue invisibility is categorical, while brief tracking artifacts are imputed according to the cited marker-cleaning method.

## 10-a. What are the most time-consuming steps of the code?

i. The expensive work is reading large NWB spike/video arrays, per-unit spike binning, assembling the multi-gigabyte object, and pickling it. Multiprocessing reduces wall-clock session conversion; decoder training, performed only for validation, took substantially longer than conversion.

ii.
```python
with Pool(args.nproc) as pool:
    for res, info in pool.imap(process_session, ...):
...
with open(args.out, 'wb') as fh:
    pickle.dump(data, fh, protocol=4)
```

iii. The trajectory measured a 6.5 GB final artifact and identified array I/O and spike searches as scale-dependent operations; it deliberately parallelized independent sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops include per-unit observation/QC mapping, per-unit region mapping, per-unit spike binning, event-to-trial assignment, and per-trial output assembly. Trial bin edges and all bins for a unit are already vectorized. The event and trial-output loops could be replaced with grouped reductions/broadcasting; ragged spike arrays make the per-unit loop less amenable to simple vectorization.

ii.
```python
for u in unit_idx: ...
for i, u in enumerate(unit_idx): ...
for t, i in zip(sam_all, sidx): ...
for k, t in enumerate(trials): ...
```

iii. The agent prioritized transparent handling of ragged unit/event data and parallelized across sessions. It did vectorize the high-volume time-bin dimension with `searchsorted`, `bincount`, and broadcasting.

## 10-c. What processing does the code repeat multiple times?

i. Per session it repeatedly maps ragged per-unit observation/QC data into full trial masks and repeatedly performs list lookups (`subjects.index`, `brain_regions.index`) during assembly. It also iterates over trials once to build all neural/input/output lists, though major source arrays are read only once per session.

ii.
```python
for u in unit_idx:
    m = np.zeros(ntrials_all, dtype=bool)
...
'subject_idx': np.array([subjects.index(r['subject']) for r in results]),
```

iii. The trajectory did not identify these as bottlenecks; the repeated work supports unit-specific ragged layouts and output construction and was retained for clarity.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes extensive exclusion diagnostics (`info` fields), ontology/probe-target region derivations, video coverage metrics, and trial-QC bookkeeping not used by decoder features. Some diagnostics are retained in `session_info`, but exclusion information for dropped sessions is only printed. It also sorts each unit's spike times even though NWB spike times are normally already sorted, and computes an unused `frame_dt`.

ii.
```python
st = np.sort(st)
frame_dt = np.median(np.diff(vts))
info['n_unobserved_trials'] = int((~observed[trials]).sum())
```

iii. These checks were used to validate data integrity and reproduce paper criteria, so the agent accepted modest extra computation even where the final decoder does not consume the result.
