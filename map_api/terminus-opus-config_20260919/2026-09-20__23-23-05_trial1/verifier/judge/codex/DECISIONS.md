# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globs every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes each file with `pynwb.NWBHDF5IO`. Full mode uses a multiprocessing pool and later sorts retained results by subject and session.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
with Pool(args.nproc) as pool:
    for i, res in enumerate(pool.imap_unordered(process_session, jobs)):
        results.append(res)
```
```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
```

iii. The notes state that there is one NWB file per session, that `pynwb` is required, and that parallel session processing reduces wall-clock time. The full run found all 174 files.

## 1-b. How are the data split into subjects?

i. The subject for each session is read from `nwb.subject.description`, falling back to `subject_id`. Unique names are sorted, and each retained session receives an index into that list.

ii.
```python
subject = nwb.subject.description or nwb.subject.subject_id
subjects = sorted({r['subject'] for r in kept})
subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The notes prefer names such as `SC015` because those are the mouse names used by the papers. They report that all 28 subjects remain represented.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session and identified by `nwb.identifier`. The agent additionally excludes sessions with no good units, sessions failing behavioral criteria, and sessions with inadequate side-camera coverage. It retained 138 of 174 sessions.

ii.
```python
sess_name = nwb.identifier
if n_good == 0:
    return info
if not (performance > MIN_PERFORMANCE
        and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    return info
if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
    return info
```

iii. The agent says the behavioral thresholds reproduce the paper's 84% selected-session performance and that six sessions were removed because tongue output would be unavailable over the response period.

## 1-d. How are the data split into trials?

i. Trial rows come from `nwb.trials`. Go cues are assigned to trial intervals by locating the preceding trial start; a session is rejected if any row lacks a go cue.

ii.
```python
n_trials = len(trials)
start_time = np.asarray(trials['start_time'].data[:])
go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
gi = np.searchsorted(start_time, go_times_all, side='right') - 1
go[gi[ok]] = go_times_all[ok]
```

iii. The agent reports verifying one go cue per trial in every session. Interval assignment was chosen rather than assuming positional correspondence.

## 1-e. How are trials filtered based on quality controls?

i. It intersects `obs_intervals` across every good unit, requires at least 90% video coverage and a finite tone onset, then removes trials having no spikes from any good unit. It keeps photostimulation, early-lick, ignore, auto-water, and free-water trial types. Sessions with fewer than two final trials are dropped.

ii.
```python
observed = np.ones(n_trials, dtype=bool)
for i in good:
    ...
    observed &= m
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]
```

iii. The notes explain that unobserved trials would otherwise become artificial all-zero neural records, and video-poor trials cannot supply tongue output. Required decoder classes motivated retaining early-lick, ignore, and photostimulation trials. The full run dropped 4,540 trials within retained sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for units whose `classification` is `good`, using `BehavioralEvents/go_start_times` to define trial-relative windows.

ii.
```python
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
```

iii. The agent identifies `classification == 'good'` as the QC classifier list used in the papers and spike times as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. For each good unit, spikes in all retained trial windows are located with `searchsorted`, assigned to non-overlapping bins with `floor`, counted with `bincount`, stored as `float32`, and divided by 0.05 s to obtain Hz. No smoothing or normalization is applied.

ii.
```python
lo = np.searchsorted(st, w0)
hi = np.searchsorted(st, w1)
rel = st[pos] - gk[trial_ids]
bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
neural_all /= BIN_SIZE
```

iii. The notes describe this as the task-required, non-overlapping 50-ms counterpart of the reference `sliding_histogram(..., rate=True)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only classifier-good units are retained. No additional metric or low-firing-rate threshold is used; sessions with zero such units are excluded.

ii.
```python
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
```

iii. The agent says the classifier integrates the relevant QC metrics and that the paper's 2-Hz threshold was analysis-specific. Its raw count of 69,453 good units is close to the reported 69,943.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are sliced from `go - 2.5` through `go + 1.5`, and each selected spike has that trial's go time subtracted before bin assignment.

ii.
```python
w0 = gk + T_START
w1 = gk + T_END
rel = st[pos] - gk[trial_ids]
```

iii. The agent notes that NWB spikes and behavioral events share a session clock, so subtraction of the go timestamp gives the same alignment as the already-aligned reference exports.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 non-overlapping 50-ms bins over `[-2.5, 1.5)` seconds. Raw spikes are newly binned at this resolution; the original paper's overlapping 40-ms/3.4-ms-stride analysis is not reused.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
```

iii. The agent correctly identifies the change from the paper as required by the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, trial start times, and go-cue times. The last sample onset assigned to a trial and occurring before its go cue is selected.

ii.
```python
sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
si = np.searchsorted(start_time, sample_times, side='right') - 1
if ... and s < go[tr_i]:
    tone_onset[tr_i] = s
```

iii. The agent explains that an early lick can replay the sample epoch, so the final pre-go tone is the instructing tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone is expressed relative to the go cue, then subtracted from every go-relative bin center, yielding seconds elapsed since tone onset.

ii.
```python
tone_rel = tone_onset[keep_trials] - gk
input_all[:, 0, :] = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. The notes state that this makes the input zero at tone onset and continuous across all 80 bins.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 bin centers relative to the same per-trial go cue used for neural binning.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

iii. The agent's plots and independent checks reportedly verified that zero coincides with the marked tone onset.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses the absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, assigning each event pair to the preceding trial start and subtracting that trial's go cue.

ii.
```python
photostim_on = np.asarray(bev['photostim_start_times'].timestamps[:])
photostim_off = np.asarray(bev['photostim_stop_times'].timestamps[:])
stim_on[tr_i] = on - go[tr_i]
stim_off[tr_i] = off - go[tr_i]
```

iii. The agent verified that these event timestamps agree with the trials-table onset/duration representation and preferred the already-absolute events.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The value is 1 when a bin center lies inclusively between stimulation onset and offset, otherwise 0. Trials without a finite onset remain all zero.

ii.
```python
input_all[has_stim, 1, :] = (
    (BIN_CENTERS[None, :] >= son[has_stim][:, None])
    & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)
```

iii. The rationale is to represent stimulation as the required time-varying binary input rather than a trial flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation bounds are converted to go-relative seconds and compared with the same go-relative bin centers as the neural bins.

ii.
```python
stim_on[tr_i] = on - go[tr_i]
stim_off[tr_i] = off - go[tr_i]
```

iii. The agent reports that the input is active only in expected pre-go bins and aligns with shaded intervals in its diagnostic plots.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trials-table `outcome` and `trial_instruction`.

ii.
```python
out[0] = choice_from_trial(outcome[tr], instruction[tr])
```

iii. The notes say there is no direct choice column, but instruction plus correctness fully determines the response side, with ignore meaning no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Ignore maps to 2; hit maps to the instructed side; miss maps to the opposite side. Codes are 0 left, 1 right, 2 no lick and are repeated across time.

ii.
```python
if outcome == 'ignore': return 2
right = (instruction == 'right')
if outcome == 'miss': right = not right
return 1 if right else 0
```

iii. The derivation was cross-checked against the first post-go lick and reportedly agreed on 99.75% of responded trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials-table `outcome` column.

ii.
```python
outcome = np.asarray(trials['outcome'].data[:])
```

iii. The raw categories already match the requested ignore/miss/hit output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers 0 ignore, 1 miss, and 2 hit, then repeated across all time bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
out[1] = OUTCOME_CODE[outcome[tr]]
```

iii. The ordering follows the task specification, and repetition permits all outputs to share a `(4, 80)` array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_lick = np.asarray(trials['early_lick'].data[:])
```

iii. The file already explicitly labels early versus no-early trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other stored value maps to 0; the result is repeated across time.

ii.
```python
out[2] = 1 if early_lick[tr] == 'early' else 0
```

iii. The agent says this implements the requested no/yes category coding while retaining the early-lick trials needed for decoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` timestamps and data column 1 for y-position; data column 2 is the visibility likelihood.

ii.
```python
tongue = bts['Camera0_side_TongueTracking']
vtime = np.asarray(tongue.timestamps[:])
vdata = np.asarray(tongue.data[:])
tongue_y = vdata[:, 1]
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. The notes identify this as the side-view DeepLabCut series described by the reference alignment code.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood at most 0.9 are excluded. Visible y-values are averaged within each retained trial's 50-ms bins. Percentiles are computed over all finite binned values from retained windows in that session; empty bins become not visible.

ii.
```python
sel = tongue_vis[lo:hi]
sums = np.bincount(bidx, weights=vy, minlength=N_BINS)
cnts = np.bincount(bidx, minlength=N_BINS)
tongue_bin_y[k, nz] = sums[nz] / cnts[nz]
p40, p60 = np.percentile(tongue_bin_y[visible], [40, 60])
```

iii. The agent says likelihood is strongly bimodal, making 0.9 insensitive in practice, and that computing percentiles on visible binned values matches the quantity being classified and yields the intended 40/20/40 balance.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. For each retained session, visible bin means below p40 map to 0, values from p40 through p60 map to 1, values above p60 map to 2, and bins with no visible frame map to 3.

ii.
```python
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)
tongue_class[visible & (tongue_bin_y < p40)] = 0
tongue_class[visible & (tongue_bin_y >= p40) & (tongue_bin_y <= p60)] = 1
tongue_class[visible & (tongue_bin_y > p60)] = 2
```

iii. This directly implements the requested four labels; the agent reports an approximately 40/20/40 split among visible bins.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected with absolute bounds `go + T_START` and `go + T_END`, converted to time relative to that go cue, and assigned to the same 50-ms grid as spikes.

ii.
```python
lo = np.searchsorted(vtime, g + T_START)
hi = np.searchsorted(vtime, g + T_END)
vt = vtime[lo:hi][sel] - g
bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
```

iii. The agent uses stored timestamps rather than an assumed frame index spacing, noting that this is robust to dropped frames while preserving the common NWB clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions lacking a go cue, good units, adequate behavior, or adequate video are skipped. Trials outside intersected unit observation intervals, with insufficient video, missing tone onset, or no spikes are removed. Invisible tongue bins use category 3 rather than imputation. Sessions with fewer than two trials are removed. `is_good_trials` is deliberately ignored.

ii.
```python
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
nonempty = neural_all.any(axis=(1, 2))
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)
```

iii. The agent distinguishes absent recordings, which it excludes, from legitimate nonvisibility, which it encodes explicitly. Its notes document partially recorded sessions, interrupted recordings, stopped video, and the rationale for not applying per-unit `is_good_trials` masks.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identifies NWB reads (especially spike and video arrays), per-unit spike binning, and final pickling as the main costs. Multiprocessing reduced the full conversion to 35 seconds; pickling the 9.63-GB result took 15 seconds.

ii.
```python
timings['video_read'] = time.time() - t0
timings['spikes_read'] = time.time() - t0
timings['bin_trials'] = time.time() - t0
pickle.dump(data, f, protocol=4)
```

iii. The notes estimate roughly 60 seconds serial for spike reads and binning before parallelism, and measured the full parallel run and serialization separately.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining loops are over good units for observation-mask intersection and neural binning, over trials for tongue binning and output construction, and over units for region labels. The agent vectorized bins and all trials within each unit but did not eliminate ragged unit/trial loops.

ii.
```python
for i in good:
    observed &= m
for ui, st in enumerate(spikes):
    counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
for k in range(nk):
    ...
```

iii. The notes say earlier per-bin and repeated per-trial spike reads were replaced by `bincount`, `searchsorted`, and one read per unit. The tongue loop was retained for clarity over ragged frame slices.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly performs searches and binning per unit or trial, and constructs the large neural/input/output lists after array computation. Each file and its spike/video buffers are nevertheless read only once per session; the global bin grid is reused.

ii.
```python
for ui, st in enumerate(spikes):
    lo = np.searchsorted(st, w0)
    hi = np.searchsorted(st, w1)
for k in range(nk):
    lo = np.searchsorted(vtime, g + T_START)
```

iii. The agent emphasizes that it removed repeated spike reads and bin-level Python loops. Remaining repetition reflects ragged per-unit spike trains and per-trial camera slices rather than recomputing whole data products.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal full conversion, it reads and uses `auto_water`, `free_water`, stop times, session-performance statistics, detailed timings, trial indices, and multiple diagnostics that are mostly retained only in metadata or logs rather than decoder tensors. Optional plotting performs substantial extra work only when requested.

ii.
```python
auto_water = np.asarray(trials['auto_water'].data[:])
free_water = np.asarray(trials['free_water'].data[:])
n_unobserved = int(np.sum(...))
timings=timings,
if show_processing:
    plot_session(...)
```

iii. The agent's notes frame these calculations as curation checks, paper-consistency validation, and user-facing diagnostics. They are not required by decoder training itself, but most are used to select data or document conversion quality rather than silently discarded.
