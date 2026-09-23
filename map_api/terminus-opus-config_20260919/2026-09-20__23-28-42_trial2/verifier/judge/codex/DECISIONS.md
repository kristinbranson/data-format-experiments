# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `/app/data/sub-*/*.nwb`, sorts the paths, and processes each NWB file with `pynwb.NWBHDF5IO`. Inside each file it reads `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`.

ii.
```python
def list_sessions():
    import glob
    return sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    ident = nwb.identifier
    subject = str(nwb.subject.subject_id)
    ...
    units = nwb.units
    ...
    tr = nwb.trials
    ...
    be = nwb.acquisition['BehavioralEvents'].time_series
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI justifies this as the published DANDI/NWB layout: one NWB file per session, all required variables already present in the NWB file, and `pynwb` is the required reader.

## 1-b. How are the data split into subjects?

i. Subjects are split by `nwb.subject.subject_id`. The final `subjects` list is the sorted unique set of those IDs, and `subject_idx` maps each session to its subject.

ii.
```python
subject = str(nwb.subject.subject_id)
```

```python
subjects = sorted({r['subject'] for r in good})
subject_idx = np.array([subjects.index(r['subject']) for r in good], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says `nwb.subject.subject_id` is the canonical per-mouse identifier in the NWB release and matches the dataset’s per-subject directory structure.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and each session is identified by `nwb.identifier`.

ii.
```python
files = list_sessions()
```

```python
ident = nwb.identifier
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI notes that the DANDI release is already one file per session, so file boundaries define session boundaries directly.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. The AI reads the per-trial columns from `nwb.trials`, and it checks that the number of `go_start_times` matches the number of trial starts.

ii.
```python
tr = nwb.trials
trial_start = np.asarray(tr['start_time'][:], dtype=float)
trial_stop = np.asarray(tr['stop_time'][:], dtype=float)
...
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
assert len(go) == len(trial_start), 'go cue count != trial count in %s' % ident
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI states that `go_start_times` exists once per trial in all sessions, so the NWB trials table plus the one-to-one go-cue count gives an unambiguous trial split.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps early-lick, ignore, miss, and photostim trials, but filters out `auto_water` and `free_water` trials first. After neural binning, it also removes any trial whose entire neural tensor is zero across all good units and all 80 bins, treating those as missing-ephys trials. It drops a session if fewer than 2 trials remain.

ii.
```python
# trial curation: drop auto-water / free-water trials
keep = (auto_water == 0) & (free_water == 0)
...
if ntr < 2:
    return {'identifier': ident, 'dropped': 'fewer than 2 usable trials'}
```

```python
spikes_per_trial = fr.sum(axis=(0, 2))
rec = spikes_per_trial > 0
...
if n_dropped_norec:
    fr = fr[:, rec, :]
    kidx = kidx[rec]
    ntr = int(rec.sum())
if ntr < 2:
    return {'identifier': ident, 'dropped': 'fewer than 2 trials with ephys'}
```

iii. In `CONVERSION_NOTES.md` Steps 5, 6, 9, and trajectory steps 93-97, the AI argues that `auto_water`/`free_water` should be excluded because reward is not behaviorally meaningful there, while all-zero trials are acquisition gaps with no usable neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `units['spike_times']` for units whose `classification` is `'good'`, and from `BehavioralEvents/go_start_times` to place trial windows around the go cue.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
good = np.where(classification == 'good')[0]
```

```python
spike_lists = [np.asarray(units['spike_times'][int(i)]) for i in good]
...
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
fr = bin_spikes(spike_lists, go[kidx])
```

iii. In `CONVERSION_NOTES.md` Steps 3 and 5, the AI justifies this as the faithful NWB equivalent of the reference pipeline: classifier-approved units plus raw spike times on the session clock, aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue, counts spikes per bin with `np.searchsorted`, and divides by 0.05 s to convert counts to firing rates in Hz. It applies no smoothing, normalization, or baseline subtraction.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES_REL = OFF_START + BIN_SIZE * np.arange(NBINS + 1)
```

```python
def bin_spikes(spike_times_list, go_times):
    ntr = len(go_times)
    edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
    out = np.empty((len(spike_times_list), ntr, NBINS), dtype=np.float32)
    for k, st in enumerate(spike_times_list):
        idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
        out[k] = np.diff(idx, axis=1)
    out /= np.float32(BIN_SIZE)
    return out
```

iii. In `CONVERSION_NOTES.md` Steps 1, 5, and 6, the AI says this is the natural 50 ms analogue of the reference `sliding_histogram(..., rate=True)` while obeying the decoder task’s required bin width.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units.classification == 'good'`. If a session has zero such units, it drops the whole session.

ii.
```python
classification = np.asarray(units['classification'][:]).astype(str)
good = np.where(classification == 'good')[0]
if len(good) == 0:
    return {'identifier': ident, 'dropped': 'no good units'}
```

iii. In `CONVERSION_NOTES.md` Steps 3 and 5, the AI argues that `classification` is the published output of the Chen/Liu QC classifier and is the closest NWB equivalent to the reference good-unit lists.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to go-cue onset. The AI uses absolute `go_start_times`, adds the fixed relative bin edges, and bins spikes directly on that shared session clock.

ii.
```python
go = np.asarray(be['go_start_times'].timestamps[:], dtype=float)
fr = bin_spikes(spike_lists, go[kidx])
```

```python
edges = (go_times[:, None] + BIN_EDGES_REL[None, :]).ravel()
```

iii. In `CONVERSION_NOTES.md` Steps 3 and 5, the AI justifies this by noting that spikes and behavioral events already share the NWB session clock, so go-cue alignment only requires subtracting the per-trial go time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins. The AI defines 80 bins over the 4 s window and does not perform any additional temporal rebinning beyond that single binning step.

ii.
```python
BIN_SIZE = 0.05
OFF_START = -2.5
OFF_END = 1.5
NBINS = int(round((OFF_END - OFF_START) / BIN_SIZE))   # 80
```

```python
'time_bin_size': BIN_SIZE * 1000.0,
```

iii. In `CONVERSION_NOTES.md` Steps 1, 3, and 5, the AI cites the decoder instructions: 2.5 s before to 1.5 s after the go cue, with 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `trials.start_time`, and `BehavioralEvents/go_start_times`. The AI assigns each sample-start event to a trial and keeps the last sample-start at or before that trial’s go cue.

ii.
```python
sample_starts = np.asarray(be['sample_start_times'].timestamps[:], dtype=float)
```

```python
def tone_onset_times(sample_starts, trial_starts, go_times):
    out = np.full(ntr, np.nan)
    if len(sample_starts):
        tidx = np.searchsorted(trial_starts, sample_starts, side='right') - 1
        ...
        ok2 = ev <= go_times[tidx]
        ...
        out[tidx[order]] = ev[order]
    return out
```

iii. In `CONVERSION_NOTES.md` Steps 2, 4, and 5, the AI says early licks can replay the sample epoch, so the relevant tone for a trial is the last sample-start before the final go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding the tone onset per trial, the AI computes time since tone onset at each go-cue-centered bin center by subtracting the trial’s tone-relative offset from the shared bin-center grid.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
tone = tone_onset_times(sample_starts, trial_start, go)[kidx]
tone_rel_go = tone - go[kidx]
time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as the same go-cue-relative time axis used for neural binning, re-expressed as seconds since the trial’s last pre-go tone onset.

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned on the exact same 80 bin centers relative to the go cue that the neural data use.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE / 2
...
time_from_tone = (BIN_CENTERS_REL[None, :] - tone_rel_go[:, None]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly says the value is evaluated at the same go-cue-centered bin centers as the neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times`, `BehavioralEvents/photostim_stop_times`, `trials.start_time`, and `BehavioralEvents/go_start_times`. It also cross-checks against `trials.photostim_power`.

ii.
```python
stim_flag_tbl = np.asarray(tr['photostim_power'][:]).astype(str) != 'N/A'
...
stim_on = np.asarray(be['photostim_start_times'].timestamps[:], dtype=float)
stim_off = np.asarray(be['photostim_stop_times'].timestamps[:], dtype=float)
```

```python
def photostim_binary(stim_on, stim_off, trial_starts, go_times):
    tidx = np.searchsorted(trial_starts, stim_on, side='right') - 1
    for i, on, off in zip(tidx, stim_on, stim_off):
        ...
        on_rel, off_rel = on - go_times[i], off - go_times[i]
```

iii. In `CONVERSION_NOTES.md` Steps 4, 5, and 9, the AI justifies using event timestamps because they are the exact stimulation times on the shared session clock and can be checked against the trial-table stim fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts stim events into a binary `(n_trials, 80)` time series. A bin is marked 1 when the stimulation interval overlaps that 50 ms bin; otherwise it is 0.

ii.
```python
def photostim_binary(stim_on, stim_off, trial_starts, go_times):
    ps = np.zeros((ntr, NBINS), dtype=np.float32)
    ...
    for i, on, off in zip(tidx, stim_on, stim_off):
        ...
        overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
        ps[i, overlap] = 1.0
    return ps
```

iii. In `CONVERSION_NOTES.md` Step 5 and trajectory step 110, the AI argues this is the most direct way to represent whether light is on during each neural bin, including replay-aborted stim pulses that may fall outside the exported window.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The AI assigns each stim event to a trial, converts onset and offset to times relative to that trial’s go cue, and compares them against the same 50 ms go-cue-centered bins used for neural data.

ii.
```python
on_rel, off_rel = on - go_times[i], off - go_times[i]
...
overlap = (BIN_EDGES_REL[:-1] < off_rel) & (BIN_EDGES_REL[1:] > on_rel)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI explicitly describes `photostim_on` as go-cue-relative and bin-aligned with the neural tensor.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trials.outcome` and `trials.trial_instruction`. There is no direct choice column in the NWB file.

ii.
```python
outcome = np.asarray(tr['outcome'][:]).astype(str)
instruction = np.asarray(tr['trial_instruction'][:]).astype(str)
```

```python
choice = np.where(out_k == 'ignore', 2,
                  np.where(out_k == 'hit',
                           np.where(instr_k == 'left', 0, 1),
                           np.where(instr_k == 'left', 1, 0))).astype(np.int64)
```

iii. In `CONVERSION_NOTES.md` Steps 4 and 5, the AI says hit means lick the instructed side, miss means lick the opposite side, and ignore means no valid lick, so choice is deterministically reconstructible from those two fields.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0=left`, `1=right`, `2=no lick` and broadcasts the per-trial value across all 80 bins in output row 0.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```

```python
output_trials = [np.stack([np.full(NBINS, choice[i]),
                           np.full(NBINS, outcome_code[i]),
                           np.full(NBINS, early_code[i]),
                           tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies repeating per-trial categorical outputs across bins so all outputs share one `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `trials.outcome`.

ii.
```python
outcome = np.asarray(tr['outcome'][:]).astype(str)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI notes that the NWB trials table already stores the exact three requested outcome categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeats the code across all 80 bins in output row 1.

ii.
```python
outcome_code = np.where(out_k == 'ignore', 0, np.where(out_k == 'miss', 1, 2)).astype(np.int64)
```

```python
output_trials = [np.stack([np.full(NBINS, choice[i]),
                           np.full(NBINS, outcome_code[i]),
                           np.full(NBINS, early_code[i]),
                           tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says the coding follows the task specification and keeps the output array rectangular.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `trials.early_lick`.

ii.
```python
early = np.asarray(tr['early_lick'][:]).astype(str)
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI notes that early lick is explicitly stored in the trials table and must be kept because it is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `early` as 1 and `no early` as 0, then repeats that code across all 80 bins in output row 2.

ii.
```python
early_code = (early[kidx] == 'early').astype(np.int64)
```

```python
output_trials = [np.stack([np.full(NBINS, choice[i]),
                           np.full(NBINS, outcome_code[i]),
                           np.full(NBINS, early_code[i]),
                           tongue_cls[i]]).astype(np.int64) for i in range(ntr)]
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as the required binary categorical output format.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps plus the `y` and likelihood columns from the `(x, y, likelihood)` data array.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries'].time_series
tongue = bts['Camera0_side_TongueTracking']
tt = np.asarray(tongue.timestamps[:], dtype=float)
tdata = np.asarray(tongue.data[:], dtype=float)
```

```python
ty_binned = bin_tongue(tt, tdata[:, 1], tdata[:, 2], go[kidx])
```

iii. In `CONVERSION_NOTES.md` Steps 2 and 5, the AI identifies this NWB time series as the side-view DeepLabCut tongue signal used for the decoder’s tongue output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI keeps frames whose likelihood exceeds 0.5, removes 5-sigma velocity outliers among visible frames, bins the surviving `y` values into 50 ms means separately for each trial window, and leaves bins with no valid frame as NaN.

ii.
```python
TONGUE_LIK_THRESH = 0.5
VELOCITY_SIGMA = 5.0
```

```python
def bin_tongue(ts, y, lik, go_times):
    visible = lik > TONGUE_LIK_THRESH
    valid = visible.copy()
    if visible.sum() > 10:
        iv = np.where(visible)[0]
        dy = np.diff(y[iv]) / np.maximum(np.diff(ts[iv]), 1e-6)
        s = np.std(dy)
        if s > 0:
            bad = np.abs(dy) > VELOCITY_SIGMA * s
            valid[iv[1:][bad]] = False
    ...
    res[i, nz] = (sm[nz] / cnt[nz]).astype(np.float32)
```

iii. In `CONVERSION_NOTES.md` Steps 3, 5, and 6, the AI justifies this by citing the method paper’s DeepLabCut outlier handling and the task requirement to preserve a separate “not visible” category instead of imputing hidden frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After binning tongue `y`, the AI computes the 40th and 60th percentiles over all finite binned tongue values in the kept trial windows of that session, assigns class 0 below the 40th percentile, class 1 from the 40th through the 60th percentile, class 2 above the 60th percentile, and class 3 to bins with no visible tongue.

ii.
```python
def discretize_tongue(y_binned):
    vis = np.isfinite(y_binned)
    cls = np.full(y_binned.shape, 3, dtype=np.int64)
    if vis.sum() > 0:
        p40, p60 = np.percentile(y_binned[vis], [40, 60])
        v = y_binned[vis]
        c = np.where(v < p40, 0, np.where(v <= p60, 1, 2))
        cls[vis] = c
    else:
        p40 = p60 = np.nan
    return cls, float(p40), float(p60)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI justifies this as a per-session discretization that yields a 40/20/40 split over visible bins, with class 3 reserved for occluded or rejected bins.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue frames to the same go-cue-centered 50 ms window as the neural data. For each trial it finds the camera-frame range between `go + OFF_START` and `go + OFF_END`, then bins frame times by offset from `go + OFF_START`.

ii.
```python
lo = np.searchsorted(ts, t0)
hi = np.searchsorted(ts, go_times + OFF_END)
for i in range(ntr):
    ...
    bi = np.floor((tsel[vsel] - t0[i]) / BIN_SIZE).astype(int)
    np.clip(bi, 0, NBINS - 1, out=bi)
```

iii. In `CONVERSION_NOTES.md` Steps 3 and 5, the AI says the camera timestamps are already on the same session clock as spikes and go cues, so no extra interpolation or clock correction is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several cases: sessions with no good QC-passing units are dropped; trials with zero spikes across the entire neural window are dropped as missing-ephys gaps; low-likelihood or velocity-outlier tongue frames are excluded and become class 3 when a bin has no valid frame; photostim table/event mismatches are investigated but not changed when they fall outside the exported window.

ii.
```python
if len(good) == 0:
    return {'identifier': ident, 'dropped': 'no good units'}
```

```python
spikes_per_trial = fr.sum(axis=(0, 2))
rec = spikes_per_trial > 0
```

```python
visible = lik > TONGUE_LIK_THRESH
...
cls = np.full(y_binned.shape, 3, dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md` Steps 6, 9, and trajectory steps 91-110, the AI argues that missing neural acquisition should be removed rather than encoded as all-zero activity, while legitimately absent tongue observations should become an explicit category.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies per-session NWB I/O, reading large spike-time arrays, neural binning, and pickling the final multi-GB dataset as the dominant costs. It also notes video/tongue processing, but as a smaller share.

ii.
```python
timing['spike_read'] = time.time() - t0
...
timing['binning'] = time.time() - t0
...
timing['outputs'] = time.time() - t0
```

```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. In `CONVERSION_NOTES.md` Steps 6 and 7, the AI explicitly reports run-time breakdowns and says the wall-clock cost is dominated by NWB reads plus writing the ~10-12 GB pickle.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several Python loops: a per-unit loop in `bin_spikes`, a per-event loop in `photostim_binary`, and a per-trial loop in `bin_tongue`. These are partly vectorized already over the time-bin dimension but not fully collapsed.

ii.
```python
for k, st in enumerate(spike_times_list):
    idx = np.searchsorted(st, edges).reshape(ntr, NBINS + 1)
    out[k] = np.diff(idx, axis=1)
```

```python
for i, on, off in zip(tidx, stim_on, stim_off):
    ...
```

```python
for i in range(ntr):
    ...
    cnt = np.bincount(bi, minlength=NBINS)
    sm = np.bincount(bi, weights=y[a:b][vsel], minlength=NBINS)
```

iii. In `CONVERSION_NOTES.md` Step 6, the AI justifies keeping these loops because the heavy inner math is vectorized and the remaining loops are over ragged units or per-trial frame groups.

## 10-c. What processing does the code repeat multiple times?

i. The core conversion is mostly single-pass, but the script repeats some nonessential work: after assembling the output it loops again to compute summary statistics, output distributions, and region counts; optional `--show-processing` also recomputes independent PSTHs and plotting diagnostics from raw spikes.

ii.
```python
reg = Counter()
for r in good:
    reg.update(r['region_labels'].tolist())
...
for r in good:
    for o in r['output']:
        outs['choice'][int(o[0, 0])] += 1
```

```python
for k in range(sub):
    s = spike_lists[k]
    for i in kidx[:200]:
        m = s[(s >= go[i] + OFF_START) & (s < go[i] + OFF_END)] - go[i]
        raw += np.histogram(m, bins=BIN_EDGES_REL)[0]
```

iii. The AI’s notes mostly emphasize single-pass conversion, but `CONVERSION_NOTES.md` Step 6 also presents the diagnostic plots and summary accounting as explicit extra validation work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and carries some data that never make it into the final saved dataset: per-session `ccf` coordinates and fine `anno` labels are stored in the intermediate `result` dict but dropped during final assembly. It also does optional diagnostic plotting and end-of-run summaries that are not used downstream by the decoder.

ii.
```python
result = {
    'identifier': ident,
    'subject': subject,
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'region_labels': region_labels,
    'ccf': np.stack([ccf_x, ccf_y, ccf_z], axis=1).astype(np.float32),
    'anno': anno,
    'info': info,
}
```

```python
data = {
    'neural': [r['neural'] for r in good],
    'input': [r['input'] for r in good],
    'output': [r['output'] for r in good],
    ...
}
```

iii. In `CONVERSION_NOTES.md` Steps 6, 10, and 13, the AI treats these extra computations as diagnostics and traceability aids rather than part of the decoder payload.
