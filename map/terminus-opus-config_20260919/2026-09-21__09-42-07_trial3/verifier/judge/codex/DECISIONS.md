# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent glob-sorts every `/app/data/sub-*/*.nwb` file, processes each file as one session, and uses an HDF5 reader. Full mode processes the files in a 16-worker pool; sample mode deliberately selects two named files.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with h5py.File(fname, 'r') as f:
    tr = f['intervals/trials']
    u = f['units']
...
with Pool(min(args.nproc, len(tasks))) as pool:
    for i, res in enumerate(pool.imap(process_session, tasks)):
```

iii. The notes say NWB contains all probes for a session in one units table, so direct HDF5 loading is equivalent to the paper code's per-probe MAT loading. Sorting is deterministic and multiprocessing reduced the full run to 36.9 seconds.

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from each NWB filename, unique IDs are sorted, and each retained session receives an index into that list.

ii.
```python
'subject': os.path.basename(fname).split('_')[0].replace('sub-', ''),
subjects = sorted({s['subject'] for s in sessions})
subject_idx = np.array([subjects.index(s['subject']) for s in sessions], dtype=np.int64)
```

iii. The notes identify the `sub-*` folder/filename ID as the mouse ID and report 28 mice, matching the dataset and papers.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session ID is parsed from the filename, but sessions failing behavioral performance, correct-trial count, unit, video, or minimum-trial criteria are omitted.

ii.
```python
'session_id': os.path.basename(fname).split('_')[1].replace('ses-', ''),
if performance <= MIN_PERFORMANCE: return None, info
if n_correct_left < MIN_CORRECT_PER_SIDE or n_correct_right < MIN_CORRECT_PER_SIDE:
    return None, info
```

iii. The agent justified the extra selection as the data-paper criteria (>65% performance and at least 50 correct trials per side), plus requirements for QC units and usable video. This retained 138 of 174 files.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials; their start/stop times are used with one go cue per row. Retained trial indices are then used to create lists of per-trial arrays.

ii.
```python
tr = f['intervals/trials']
start_time = tr['start_time'][:]
stop_time = tr['stop_time'][:]
go = be['go_start_times/timestamps'][:]
assert len(go) == ntrials_all, 'go cue count != trial count'
for k, ti in enumerate(trials):
    neural_list.append(rates[:, k, :])
```

iii. The trials table is the explicit NWB trial definition, and the one-to-one go-cue assertion was used as a sanity check.

## 1-e. How are trials filtered based on quality controls?

i. The agent removes auto-water and free-water trials; trials observed/stable for fewer than 90% of good units; trials lacking a video frame in any of 80 bins; and wholly spike-empty trials. It requires at least 10 retained trials. Early-lick, miss/ignore, and photostimulation trials are kept.

ii.
```python
usable = observed & igt
trial_stable = usable.mean(axis=0) >= IGT_TRIAL_FRAC
keep = (auto_water == 0) & (free_water == 0) & trial_stable
video_ok = np.all(frames_per_bin > 0, axis=1)
keep = keep & video_ok
nonempty = rates.sum(axis=(0, 2)) > 0
```

iii. The notes say water trials are not normal choice trials, `obs_intervals` avoids treating absent recordings as silence, `is_good_trials` captures manual stability curation, and full video coverage distinguishes missing video from an invisible tongue. Required decoder classes are intentionally retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, restricted to units whose `classification` is `good`; go-cue timestamps define absolute bin edges.

ii.
```python
classification = decode_array(u['classification'][:])
good = np.where(classification == 'good')[0]
st_index = u['spike_times_index'][:]
st_all = u['spike_times'][:]
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]
```

iii. The agent says `classification == 'good'` is the NWB encoding of the white-paper QC classifier and spike times are the available neural representation.

## 2-b. How is the `neural` data processed?

i. For every retained unit, all retained trial edges are searched at once; adjacent cumulative indices are differenced into spike counts and divided by 0.05 seconds to obtain float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
counts = np.searchsorted(sp, flat_edges).reshape(nkept, NBINS + 1)
rates[i] = np.diff(counts, axis=1).astype(np.float32)
rates /= BIN_SIZE
```

iii. The notes explicitly connect this to the reference `sliding_histogram(..., rate=True)` estimator, with the task-prescribed nonoverlapping 50-ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It first keeps classifier-good units. It then uses each unit's `obs_intervals` and `is_good_trials`; after trial selection, any unit not usable on every retained trial is removed. Sessions with no remaining units are rejected.

ii.
```python
good = np.where(classification == 'good')[0]
usable = observed & igt
unit_ok = usable[:, trials].all(axis=1)
good = good[unit_ok]
if unit_ok.sum() == 0: return None, info
```

iii. The agent considered `is_good_trials` an additional manual stability QC made possible by the NWB release and used `obs_intervals` to avoid fabricated zero activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All timestamps share session time. Relative edges from -2.5 to +1.5 seconds are added to each trial's go cue, and spike times are binned directly against those absolute edges.

ii.
```python
go = be['go_start_times/timestamps'][:]
edges_abs = go[:, None] + BIN_EDGES_REL[None, :]
edges_kept = edges_abs[trials]
```

iii. The notes say no interpolation or clock correction is needed because NWB events, spikes, and video use the same absolute clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins over a four-second window. Raw spikes are newly binned at that resolution; no later rebinning is applied.

ii.
```python
BIN_SIZE = 0.05
NBINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES_REL = T_START + BIN_SIZE * np.arange(NBINS + 1)
```

iii. This exactly follows the decoder instructions; the notes explain why the paper's 40-ms/3.4-ms sliding representation was not used.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and each trial's go cue, selecting the last sample start before that go cue.

ii.
```python
sample_on = be['sample_start_times/timestamps'][:]
si = np.searchsorted(sample_on, go) - 1
tone_abs = np.where(si >= 0, sample_on[np.clip(si, 0, len(sample_on) - 1)], np.nan)
```

iii. Early licks can replay the sample epoch; the last tone is therefore the tone associated with the ultimately executed trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone time is expressed relative to the go cue, then subtracted from each go-relative bin center to produce seconds since tone onset.

ii.
```python
tone_rel = tone_abs - go
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. The notes describe this as the signed time of each bin center relative to the last tone and assert that retained trials have finite tone times.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the exact same 80 go-cue-relative bin centers as the neural bin edges.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel[trials][:, None]
```

iii. The shared go-cue-relative grid guarantees one input value per neural time bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It preferentially uses behavioral-event photostimulation start/stop timestamps, assigns events to trials via `start_time`, and falls back to trials-table onset/duration for stimulated trials missing event records. `photostim_power` identifies stimulated trials.

ii.
```python
pstart = be['photostim_start_times/timestamps'][:]
pstop = be['photostim_stop_times/timestamps'][:]
pt = np.searchsorted(start_time, pstart, side='right') - 1
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
stim_rel[miss, 0] = start_time[miss] + ps_onset[miss] - go[miss]
```

iii. The agent cross-checked both representations and used the table as a robustness fallback where the event stream is missing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Start/stop times are converted to go-relative intervals. Each bin center inside the closed interval is labeled 1, otherwise 0.

ii.
```python
stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
stim_rel[pt[ok], 1] = pstop[ok] - go[pt[ok]]
stim_input[k] = ((BIN_CENTERS_REL >= stim_rel[ti, 0]) &
                 (BIN_CENTERS_REL <= stim_rel[ti, 1])).astype(np.float32)
```

iii. The notes report that independent event/table checks agreed, stimulation lasted 0.5 seconds, and occupied exactly 10 bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation timestamps are shifted by the corresponding go cue and compared with the same go-relative neural bin centers.

ii.
```python
stim_rel[pt[ok], 0] = pstart[ok] - go[pt[ok]]
stim_input[k] = (BIN_CENTERS_REL >= stim_rel[ti, 0]) & ...
```

iii. The common go-cue reference and common bin centers were validated by plotting and spot checks.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `left_lick_times`, `right_lick_times`, the go cue, and trial stop time: the first post-go lick within the trial determines left/right; none gives no lick.

ii.
```python
l0 = left_lick[np.searchsorted(left_lick, lo)] if ... else np.inf
r0 = right_lick[np.searchsorted(right_lick, lo)] if ... else np.inf
l0 = l0 if l0 <= hi else np.inf
r0 = r0 if r0 <= hi else np.inf
```

iii. The notes cite the paper pipeline's lick directions/times and report 99.8% agreement with the independent outcome-by-instruction derivation.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earliest valid left/right lick is coded 0/1; no valid lick is 2. This scalar trial label is broadcast across all 80 output bins.

ii.
```python
if np.isinf(l0) and np.isinf(r0): choice[ti] = 2
elif l0 <= r0: choice[ti] = 0
else: choice[ti] = 1
...
out[0] = choice[ti]
```

iii. Direct lick timing was chosen as the behavioral definition of actual choice; broadcasting permits all outputs to share a `(4, 80)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome = decode_array(tr['outcome'][:])
```

iii. The NWB field already contains the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2 and the per-trial value is broadcast over time.

ii.
```python
outcome_code = np.select([outcome == 'ignore', outcome == 'miss', outcome == 'hit'],
                         [0, 1, 2], default=0).astype(np.int64)
out[1] = outcome_code[ti]
```

iii. The mapping matches the requested category order and the unified time-varying output representation.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early_lick = decode_array(tr['early_lick'][:])
```

iii. The trials table explicitly supplies the flag.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` becomes 1 and every other stored value becomes 0; the result is broadcast across 80 bins.

ii.
```python
early_code = (early_lick == 'early').astype(np.int64)
out[2] = early_code[ti]
```

iii. The notes describe the requested no/yes encoding and retain early-lick trials because this is a target variable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking`: timestamps, data column 1 for y, and column 2 for tracking likelihood.

ii.
```python
tongue = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
vts = tongue['timestamps'][:]
vdata = tongue['data'][:]
vy = vdata[:, 1]
vlik = vdata[:, 2]
```

iii. The agent identified this as the dataset's tongue measurement and used likelihood to distinguish visible from untracked tongue positions.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood above 0.9 are visible. Session 40th/60th thresholds are computed from all visible raw-frame y values. Within each trial/bin, visible y values are averaged; the mean is categorized, while bins with no visible frame remain class 3.

ii.
```python
visible_all = vlik > LIKELIHOOD_THRESH
y_p40, y_p60 = np.percentile(vy[visible_all], [40, 60])
cnt = np.bincount(b[fvis], minlength=NBINS)
ysum = np.bincount(b[fvis], weights=fy[fvis], minlength=NBINS)
ymean[has] = ysum[has] / cnt[has]
```

iii. The notes say likelihood filters occluded/retracted frames, percentiles must be per session, and bin averaging aligns the high-rate camera to 50-ms decoder bins. They state 0.9 is a conservative DLC confidence cutoff.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. `np.digitize` against per-session raw-visible-frame p40/p60 produces 0 below p40, 1 between thresholds, and 2 above p60; no visible frame is 3.

ii.
```python
cls = np.full(NBINS, 3, dtype=np.int64)
cls[has] = np.digitize(ymean[has], [y_p40, y_p60])
```

iii. This implements the four requested labels, though the notes' planning table inconsistently says “median” while the code and later validation correctly describe a mean.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go-2.5, go+1.5)` are selected and assigned to the same 50-ms go-relative bins as neural activity.

ii.
```python
i0, i1 = np.searchsorted(vts, [go[ti] + T_START, go[ti] + T_END])
ft = vts[i0:i1] - go[ti]
b = np.clip(((ft - T_START) / BIN_SIZE).astype(int), 0, NBINS - 1)
```

iii. Shared absolute timestamps and an identical go-relative grid make explicit interpolation unnecessary.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/non-numeric table values become NaN; missing photostim events fall back to trial fields; unobserved/unstable neural trials and all-zero trials are removed; absent video bins cause trial removal; no visible tongue in an otherwise recorded bin is class 3. Sessions without QC units, tracking, or enough retained trials are rejected. Exceptions are recorded and the run continues.

ii.
```python
except Exception: return np.nan
miss = is_stim & ~np.isfinite(stim_rel[:, 0])
keep = keep & video_ok
if visible_all.sum() < 100: return None, info
except Exception as e:
    info['error'] = '%s: %s' % (type(e).__name__, e)
```

iii. The agent's principle was not to fabricate silence or confuse absent video with an invisible tongue; it documented each discovered edge case and added independent checks.

## 10-a. What are the most time-consuming steps of the code?

i. Bulk NWB/HDF5 reading, per-unit spike binning, multiprocessing result transfer/assembly, and writing the roughly 9-GB pickle are the costly operations. Session processing was parallelized; the reported full conversion took 36.9 seconds before validation/training.

ii.
```python
st_all = u['spike_times'][:]
for i, ui in enumerate(good):
    counts = np.searchsorted(sp, flat_edges)
with Pool(min(args.nproc, len(tasks))) as pool:
```

iii. The notes describe bulk reads and spike searches as the scale-dependent work and estimate large gains from 16-way session parallelism and flattened-edge searches.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops include per-unit coverage construction, expansion of `is_good_trials`, per-unit spike searches, per-trial photostim labeling, per-trial first-lick choice, per-trial tongue binning, and final list assembly. Several trial loops could be further vectorized; ragged per-unit spike arrays make the unit loop less straightforward.

ii.
```python
for i, ui in enumerate(good): ...
for k, ti in enumerate(trials):
    stim_input[k] = ...
for ti in trials: ...
for k, ti in enumerate(trials):
    cnt = np.bincount(...)
```

iii. The agent says it already eliminated loops over trials/bins for spike counting and bins within tongue trials; it favored clear small outer loops where ragged arrays or per-trial event logic remain.

## 10-c. What processing does the code repeat multiple times?

i. Choice lookup calls `np.searchsorted` twice for each side in its conditional expression. Trial-wise loops are separately repeated for photostim, choice, tongue, and output assembly. With diagnostic plotting enabled, unit arrays/classification are reread and several summaries are recomputed.

ii.
```python
l0 = left_lick[np.searchsorted(left_lick, lo)] if np.searchsorted(left_lick, lo) < len(left_lick) else np.inf
...
classification = decode_array(u['classification'][:])  # in make_processing_plot
```

iii. The notes broadly claim each core quantity is read/computed once, but the implementation contains these minor repetitions; optional plotting repetition was accepted as diagnostic-only overhead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `win_lo`, `win_hi`, electrode `ey`, and some intermediate annotations are computed/read but not used in the target arrays. Extensive diagnostics, statistics, `info` records, and plots do not affect decoder inputs, though much of that material supports validation and metadata.

ii.
```python
ex, ey, ez = el['x'][:], el['y'][:], el['z'][:]
win_lo = go + T_START
win_hi = go + T_END
agree = np.array([i['choice_agreement'] for i in infos if 'choice_agreement' in i])
```

iii. The agent viewed validation and diagnostic computation as purposeful rather than unnecessary, but the two window arrays and `ey` are genuinely unused; plotting is optional and was justified for alignment inspection.
