# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes one NWB file per session with `pynwb.NWBHDF5IO`. By default it uses a 16-process pool; `--sample` limits processing to the first two files.

ii.
```python
def list_sessions():
    import glob
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))

with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes say NWB is the published format, there are 174 files under 28 subject folders, and parallel session processing makes the full conversion fast while complying with the required `pynwb` API.

## 1-b. How are the data split into subjects?

i. The subject is read from each file's `nwb.subject.subject_id`. Unique IDs are sorted, and each retained session receives an integer index into that list.

ii.
```python
subject = str(nwb.subject.subject_id)
subjects = sorted({r['subject'] for r in good})
subject_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. The notes identify this field as the canonical subject identifier and report 28 unique mice; the descriptive mouse name is retained only in session metadata.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session order follows the sorted file list, and files with no QC-passing units or fewer than two usable trials are dropped.

ii.
```python
files = list_sessions()
ident = nwb.identifier
good = [r for r in results if 'neural' in r]
```

iii. The agent states that the dataset layout is one file per session. Dropping the single file with no `classification == 'good'` units yields 173 sessions, matching the paper.

## 1-d. How are the data split into trials?

i. Trial rows come from `nwb.trials`; one `BehavioralEvents/go_start_times` timestamp is paired by row index with every trial. A retained row index array (`kidx`) is used consistently for every stream.

ii.
```python
trial_start = np.asarray(tr['start_time'][:], dtype=float)
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
assert len(go) == len(trial_start)
kidx = np.where(keep)[0]
```

iii. The trajectory verified that go-cue count equals trial count, while sample events may repeat following early licks; therefore the trials table, not sample events, defines trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. The agent first removes both auto-water and free-water trials. After binning, it also removes any trial with zero spikes across every retained unit and all 80 bins. Early-lick, ignore, miss, and photostimulation trials are retained, and sessions with fewer than two remaining trials are dropped.

ii.
```python
keep = (auto_water == 0) & (free_water == 0)
...
spikes_per_trial = fr.sum(axis=(0, 2))
rec = spikes_per_trial > 0
fr = fr[:, rec, :]
kidx = kidx[rec]
```

iii. The notes cite `get_regular_trial_mask` for removing water-delivery trials and explain that trials with no spikes represent ephys acquisition gaps. Required decoder classes are deliberately retained. The full run removed 3,789 water trials and 1,657 zero-spike trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `nwb.units['spike_times']` for units whose `classification` is `good`, together with go-cue timestamps used to position bins.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
good = np.where(classification == 'good')[0]
spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
fr = bin_spikes(spike_lists, go[kidx])
```

iii. The agent identifies spike times as the raw neural representation and `classification == 'good'` as the embedded output of the paper's QC classifiers.

## 2-b. How is the `neural` data processed?

i. For each unit, session-absolute spike times are searched against all trial bin edges at once. Adjacent cumulative indices are differenced to obtain counts, then divided by 0.05 s to produce Hz. There is no smoothing, normalization, or baseline subtraction.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
for k, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
    out[k] = np.diff(idx, axis=1)
out /= np.float32(BIN_SIZE)
```

iii. The notes describe this as a vectorized analogue of the reference `sliding_histogram(..., rate=True)` with stride equal to bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units.classification == 'good'` are retained; no extra firing-rate or metric threshold is imposed. A session with no such units is discarded.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'identifier': ident, 'dropped': 'no good units'}
```

iii. The agent ties this field to the white-paper QC classifier and notes that the retained 69,453 units closely match published totals. It explicitly rejects an additional 2 Hz cutoff as specific to encoding analyses rather than this decoder task.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative edges from -2.5 to +1.5 s are added to each trial's session-absolute go-cue time, and spikes are counted between those absolute edges.

ii.
```python
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
```

iii. The notes state that spikes and behavioral events share a session clock, so no offset correction or interpolation is required.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50 ms bins over `[-2.5, 1.5)` s relative to the go cue. Raw spike timestamps are binned directly; no subsequent temporal rebinning occurs.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. These values directly implement the decoder specification and ensure identical temporal dimensions across sessions and trials.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, trial start times, and go-cue times. The helper associates sample events with trials and takes the last sample onset at or before that trial's go cue.

ii.
```python
tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
tone_rel_go = tone - go[kidx]
```

iii. The agent found that early licking can replay the sample epoch, making the last pre-go sample onset the relevant tone for the eventual response.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each go-relative bin center is shifted by the tone's go-relative time, yielding seconds elapsed since tone onset.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. The notes report a typical last-tone onset of -1.85 s relative to go and describe this as a direct clock subtraction with no further transformation.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the centers of the same 80 go-cue-relative bins used for neural firing rates.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]
```

iii. Both timestamps are on the session clock and both arrays use the shared go-relative grid, so index `t` refers to the same 50 ms neural bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, with trial start and go-cue timestamps used to associate and align each interval. The trials-table photostim power is read only for a consistency diagnostic.

ii.
```python
stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
ps_all = photostim_binary(stim_on, stim_off, trial_start, go)
```

iii. The agent chose exact event timestamps because they directly encode the laser interval; it used the table flag to investigate 21 apparent mismatches, all caused by stimulation during an aborted replay outside the exported window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each event is assigned to a trial. A bin is labeled 1 if its interval has any overlap with the photostimulation interval, otherwise 0. Reversed corrupt bounds would be swapped defensively.

ii.
```python
tidx = np.searchsorted(trial_starts, stim_on, side='right') - 1
on_rel, off_rel = on - go_times[i], off - go_times[i]
overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
ps[i, overlap] = 1.0
```

iii. The notes say this produces the required time-varying binary indicator and that plotted intervals line up with the raw event shading.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute laser timestamps are converted to offsets from the same trial go cue, then compared with the exact neural bin edges.

ii.
```python
on_rel, off_rel = on - go_times[i], off - go_times[i]
overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
```

iii. Because all event and neural timestamps share the session clock, the resulting photostimulation row is aligned bin-for-bin with firing rates.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `outcome` and `trial_instruction`: hit means instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
out_k = outcome[kidx]
instr_k = instruction[kidx]
choice = np.where(out_k == 'ignore', 2,
                  np.where(out_k == 'hit',
                           np.where(instr_k == 'left', 0, 1),
                           np.where(instr_k == 'left', 1, 0)))
```

iii. The agent notes that there is no direct choice column but that instruction and outcome determine it; trajectory checks found agreement with the first post-go lick in more than 99% of trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as 0 left, 1 right, and 2 no lick, then broadcast across all 80 time bins of the trial.

ii.
```python
np.full(NBINS, choice[i])
OUTPUT_VALUES[0] = ['left', 'right', 'no lick']
```

iii. This matches the requested categories. Broadcasting allows all outputs to share a `(4, 80)` representation while preserving the per-trial meaning.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` field.

ii.
```python
outcome = np.asarray(tr['outcome'][:]).astype(str)
out_k = outcome[kidx]
```

iii. The field already contains exactly the requested `ignore`, `miss`, and `hit` categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are encoded as ignore=0, miss=1, hit=2 and broadcast across 80 bins.

ii.
```python
outcome_code = np.where(out_k == 'ignore', 0,
                        np.where(out_k == 'miss', 1, 2)).astype(np.int64)
np.full(NBINS, outcome_code[i])
```

iii. The coding order follows the decoder specification; repetition expresses a trial-level output in the common time-series array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
early = np.asarray(tr['early_lick'][:]).astype(str)
```

iii. The agent found the table explicitly labels trials as `early` or `no early`, so no lick-time inference is necessary.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` is encoded as 1 and all other expected values (`no early`) as 0, then broadcast across 80 bins.

ii.
```python
early_code = (early[kidx] == 'early').astype(np.int64)
np.full(NBINS, early_code[i])
```

iii. This gives the requested no/yes coding and the same temporal shape as other outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamped column 1 is y-position and column 2 is DeepLabCut likelihood.

ii.
```python
tongue = bts['Camera0_side_TongueTracking']
tt = np.asarray(tongue.timestamps[:], dtype=float)
tdata = np.asarray(tongue.data[:], dtype=float)
ty_binned = bin_tongue(tt, tdata[:, 1], tdata[:, 2], go[kidx])
```

iii. The notes identify this as the available side-camera tongue measurement and report approximately 3.4 ms frame spacing.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible only when likelihood is greater than 0.5. Among visible frames, the frame following a y-velocity jump beyond five standard deviations is rejected. Remaining y values are averaged within each 50 ms trial bin; bins without a valid frame remain NaN. Percentiles are then computed from all finite binned values in the retained trials of that session.

ii.
```python
visible = lik > TONGUE_LIK_THRESH
dy = np.diff(y[iv]) / np.maximum(np.diff(ts[iv]), 1e-6)
bad = np.abs(dy) > VELOCITY_SIGMA * s
valid[iv[1:][bad]] = False
...
res[i, nz] = (sm[nz] / cnt[nz]).astype(np.float32)
```

iii. The agent cites the method paper's 5-sigma movement outlier rule, while intentionally not mean-imputing occluded tongue frames because the decoder requires a `not visible` class. It argues that percentiles should be defined on the binned quantity being classified.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles of finite binned y values from retained trial windows define thresholds. Values below p40 are class 0, values from p40 through p60 are class 1, values above p60 are class 2, and NaN bins are class 3.

ii.
```python
p40, p60 = np.percentile(y_binned[vis], [40, 60])
c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
cls = np.full(y_binned.shape, 3, dtype=np.int64)
```

iii. The thresholds and per-session scope follow the task. The notes report the expected 40/20/40 visible-bin class balance and reserve class 3 for no valid visible frame.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames between `go-2.5` and `go+1.5` are selected by timestamp and assigned to the same 50 ms go-relative bins as spikes.

ii.
```python
t0 = go_times + OFF_START
lo = np.searchsorted(ts, t0)
hi = np.searchsorted(ts, go_times + OFF_END)
bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
```

iii. Camera, go-cue, and spike timestamps use the same session clock; applying the shared interval grid guarantees binwise alignment without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. A session without good units is dropped; water trials and all-zero-neural trials are dropped; sessions falling below two trials are dropped. Missing/low-confidence/outlier tongue frames become NaN and ultimately class 3. Photostim intervals with reversed endpoints are swapped. Worker exceptions are captured and excluded from assembly. Bins outside recorded trial intervals remain zero spikes.

ii.
```python
if len(good) == 0: return {'identifier': ident, 'dropped': 'no good units'}
rec = fr.sum(axis=(0, 2)) > 0
cls = np.full(y_binned.shape, 3, dtype=np.int64)
if off_rel < on_rel: on_rel, off_rel = off_rel, on_rel
```

iii. The notes distinguish absent ephys, which is excluded, from legitimately invisible tongue, which receives an explicit category. They also document that partially unobserved neural bins are left at zero because no further information exists.

## 10-a. What are the most time-consuming steps of the code?

i. Reading ragged spike arrays, binning spikes, accumulating the approximately 12 GB result, and writing the pickle are the major costs. Optional diagnostic plotting is also substantial when enabled. Parallel session conversion reduced the reported full run to 42 s (22 s conversion plus 19 s pickle writing).

ii.
```python
spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
with ProcessPoolExecutor(min(args.workers, len(files))) as ex:
    for i, r in enumerate(ex.map(_worker, tasks)):
        results.append(r)
pickle.dump(data, f, protocol=4)
```

iii. The notes' timing measurements identify NWB/spike I/O and the very large pickle as dominant, with vectorized binning taking only about 0.16 s per typical session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural binning retains a per-neuron loop because spike arrays are ragged, though trials are vectorized. Tongue binning retains a per-trial loop; photostimulation retains a per-event loop; output assembly and summary counting retain per-trial loops. Some of these could be replaced by global indices, grouped reductions, or array broadcasting.

ii.
```python
for k, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
...
for i in range(ntr):
    ...
    cnt = np.bincount(bi, minlength=NBINS)
```

iii. The agent says ragged units prevent a simple single search, while the tongue trial loop is cheap relative to I/O. It prioritized clarity after vectorizing the expensive trial dimension of spike binning.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly accesses ragged spike times once per good unit, repeatedly constructs per-trial neural/input/output views, and iterates over outputs again for summary statistics. With diagnostics enabled, raw spikes are independently histogrammed again for plots. Core conversion quantities are otherwise computed once per session.

ii.
```python
spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
neural_trials = [np.ascontiguousarray(fr[:, i, :]) for i in range(ntr)]
for r in good:
    for o in r['output']:
        ...
```

iii. The notes characterize the conversion as a single pass and the repeated diagnostic histogram as an intentional independent sanity check. Repeated summary traversal occurs only after conversion and is inexpensive compared with the data write.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes CCF y and z coordinates, retains full CCF arrays and fine annotations in each intermediate result, and derives subject descriptions and extensive timing/QC metadata; `ccf` and `anno` are not copied into the final dataset. It also converts/records some diagnostic values, such as trial stop and the table photostim flag, that do not drive decoder arrays. Optional plots recompute histograms solely for validation.

ii.
```python
ccf_y = edf['y'].values.astype(float)[elec_ids]
ccf_z = edf['z'].values.astype(float)[elec_ids]
result = {'ccf': np.stack([ccf_x, ccf_y, ccf_z], axis=1), 'anno': anno, ...}
stim_flag_tbl = np.asarray(tr['photostim_power'][:]).astype(str) != 'N/A'
```

iii. The agent justifies these as provenance, region-validation, consistency checks, and optional visual sanity checks. They are useful during development but the CCF/fine-annotation intermediate fields are discarded during final assembly and are not used by decoder training.
