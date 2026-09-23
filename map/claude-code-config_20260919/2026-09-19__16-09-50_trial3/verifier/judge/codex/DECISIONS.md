# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI script treats the dataset as one NWB file per session under `/app/data/sub-*/*.nwb`. It first finds all files with a sorted glob, then does a cheap first pass over every file to read the trial table and decide whether the session passes its inclusion criteria. In a second pass it fully processes only the selected sessions with direct `h5py` reads instead of `pynwb`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
for fp in files:
    with h5py.File(fp, 'r') as f:
        sid = f['identifier'][()].decode()
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
```

```python
with h5py.File(filepath, 'r') as f:
    session_id = f['identifier'][()].decode()
    subject_id = f['general/subject/subject_id'][()].decode()
    tr = read_trial_table(f)
```

iii. In `CONVERSION_NOTES.md`, the AI states that the published dataset is distributed as one NWB per recording session and that all fields needed by the reference pipeline have direct NWB counterparts. It also documents a deliberate session-selection pass before full conversion so that the output matches the data paper's reported behavioral-session criteria.

## 1-b. How are the data split into subjects?

i. The AI groups sessions into subjects by the mouse name prefix embedded in `identifier` (for example `SC015_...` becomes subject `SC015`), not by the NWB `subject_id` field. It still reads the numeric `subject_id` and asserts that mouse name and numeric subject ID are in one-to-one correspondence before assembling `subjects` and `subject_idx`.

ii.
```python
session_id = f['identifier'][()].decode()
subject_id = f['general/subject/subject_id'][()].decode()
mouse = session_id.split('_')[0]
```

```python
subjects = sorted({r['subject'] for r in results})
sub_to_idx = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([sub_to_idx[r['subject']] for r in results], dtype=np.int64),
```

iii. The notes justify this indirectly by reporting subjects as mouse names throughout and by adding a consistency check that mouse name and DANDI subject ID form a 1:1 mapping. The apparent rationale is interpretability: the session identifier already uses the mouse name, and the mapping is verified not to collapse animals.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. After filtering, each retained file becomes one session entry in `neural`, `input`, `output`, and metadata. Final session order is deterministic because the files are globbed and then the converted session records are sorted by `session_id`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
results = [r for r in results if not r.get('excluded', False)]
...
results.sort(key=lambda r: r['session_id'])
```

iii. In the notes, the AI states that the dandiset is already organized as one NWB per recording session, so no extra session-boundary inference is needed.

## 1-d. How are the data split into trials?

i. Trials are taken directly from the NWB trials table. The script reads `intervals/trials`, checks that the number of `go_start_times` matches the number of trial rows, and then uses one row per trial throughout the rest of the processing.

ii.
```python
t = f['intervals/trials']
tr = {
    'start_time': t['start_time'][:],
    'stop_time': t['stop_time'][:],
    'outcome': _dec(t['outcome'][:]),
    ...
}
go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
if len(go) != ntrials:
    raise ValueError('number of go cues (%d) != number of trials (%d)' % (len(go), ntrials))
tr['go_time'] = go
```

iii. The notes say the NWB trials table is the authoritative source for trial structure, and that exactly one go cue is expected per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials inside the units' observation intervals and additionally excludes both `auto_water` and `free_water` trials. After neural binning it also drops any retained trial whose entire neural matrix sums to zero. Beyond trial-level filtering, the script performs session-level filtering first: only sessions with control-trial performance above 65%, at least 50 correct left trials, at least 50 correct right trials, at least one good unit, and at least two usable trials are processed.

ii.
```python
def trial_mask(f, tr):
    return ((tr['auto_water'] == 0) & (tr['free_water'] == 0) &
            observed_trial_mask(f, tr))
```

```python
if perf <= MIN_PERFORMANCE:
    reason = 'performance <= %.2f' % MIN_PERFORMANCE
elif min(ncorrect_left, ncorrect_right) < MIN_CORRECT_PER_DIRECTION:
    reason = 'fewer than %d correct trials in one direction' % MIN_CORRECT_PER_DIRECTION
elif ngood == 0:
    reason = 'no units passed quality control'
elif nkept < MIN_TRIALS_PER_SESSION:
    reason = 'fewer than %d usable trials' % MIN_TRIALS_PER_SESSION
```

```python
silent = fr.sum(axis=(1, 2)) == 0
if n_silent:
    kept_positions = np.flatnonzero(keep)
    keep[kept_positions[silent]] = False
    fr = fr[~silent]
```

iii. The notes explicitly justify the session filter as matching the data paper's session-selection criteria. They justify excluding `auto_water` and `free_water` because those trials do not reflect a real behavioral decision, while early-lick, ignore, and photostim trials are deliberately kept because the decoder must model them. The silent-trial drop is justified as a data-integrity fix for a residual recording-coverage issue.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times`, with trial-specific go-cue times from `BehavioralEvents/go_start_times` used to place the bin edges. Only units whose `units/classification` is `'good'` are kept.

ii.
```python
classification = _dec(u['classification'][:])
good = np.flatnonzero(classification == 'good')
...
spike_times = u['spike_times'][:]
spike_index = u['spike_times_index'][:]
```

```python
go = tr['go_time'][keep]
fr = bin_spikes(spikes, go)
```

iii. The notes say the reference processing uses classifier-approved good units and session-absolute spike times aligned to the go cue, so the NWB `spike_times` field is the direct raw source to reproduce.

## 2-b. How is the `neural` data processed?

i. The AI converts absolute spike times into per-trial firing rates in Hz. It constructs the same bin edges for all trials relative to each trial's go cue, uses `np.searchsorted` and `np.diff` to get spike counts per bin for each unit, and divides by the 50 ms bin width. No smoothing, baseline subtraction, or normalization is applied.

ii.
```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
fr = np.empty((ntrials, len(spikes), N_BINS), dtype=np.float32)
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    fr[:, i, :] = np.diff(pos, axis=1)
fr /= BIN_SIZE
```

iii. The notes explicitly justify firing rates in Hz as matching the reference's `sliding_histogram(..., rate=True)` convention while adapting the bin width to the task-mandated 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by the QC classifier verdict: `units/classification == 'good'`. Sessions with zero such units are excluded entirely. The script does not apply extra thresholds on individual QC metrics and does not use `unit_quality`.

ii.
```python
classification = _dec(u['classification'][:])
good = np.flatnonzero(classification == 'good')
```

```python
elif ngood == 0:
    reason = 'no units passed quality control'
```

iii. The notes justify this as the paper's intended curation rule and claim it reproduces the paper's per-area unit counts, so additional metric thresholds were treated as unnecessary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to go-cue onset. The script uses absolute go-cue timestamps for each trial, adds the fixed relative bin edges `[-2.5, 1.5]`, and bins spikes directly against those absolute windows.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
```

```python
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. The notes justify this by stating that all NWB streams share one session-absolute clock, so alignment to the go cue only requires subtracting or offsetting by each trial's `go_start_times` timestamp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms non-overlapping bins from -2.5 s to +1.5 s relative to go cue, yielding 80 bins per trial. This is a rebinning from raw spike times to fixed-width firing-rate bins.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

iii. The notes say the reference pipeline uses different binning, but the decoder task explicitly mandates 50 ms bins, so the AI rebinned to that grid.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from the sample-tone onset times in `BehavioralEvents/sample_start_times` and the per-trial go-cue times. For each trial, the script chooses the last sample-start event before the go cue and stores it as `tone_time`.

ii.
```python
sample_start = np.sort(f['acquisition/BehavioralEvents/sample_start_times']['timestamps'][:])
idx = np.searchsorted(sample_start, go, side='right') - 1
tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
tone = np.where(np.isnan(tone), go - 1.85, tone)
tr['tone_time'] = tone
```

iii. The notes justify taking the last sample start before the go cue because early licks can replay the sample or delay epoch, so a trial may contain multiple sample-start events and the one immediately preceding the go cue is the relevant tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, the AI computes elapsed time since tone onset on the go-cue-aligned grid. Algebraically this is `bin_center - (tone - go)`, equivalent to `bin_center + (go - tone)`.

ii.
```python
inp = np.zeros((n, 2, N_BINS), dtype=np.float32)
inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)
```

iii. The notes describe this as a continuous ramp representing time since the instruction-tone onset at each neural bin center. They also document a fallback to `go - 1.85 s` for any trial lacking a preceding tone event.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone-time input is defined directly on the same 80 bin centers used for neural activity, so each timepoint is already aligned to the neural bins.

ii.
```python
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)
...
inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)
```

iii. The notes explicitly say all inputs and outputs are made time-varying on the same go-cue-aligned 50 ms grid as the neural array.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table fields `photostim_onset` and `photostim_duration`, together with `start_time` and `go_time` to express the stimulation interval on the go-cue-centered axis.

ii.
```python
onset_str = tr['photostim_onset'][keep]
dur_str = tr['photostim_duration'][keep]
start = tr['start_time'][keep]
has_stim = onset_str != 'N/A'
```

```python
on_rel = start + onset - go
off_rel = on_rel + dur
```

iii. The notes say the NWB trial fields are direct counterparts of the reference task-stimulation variables, and that onset must be translated from trial-start-relative time into go-cue-relative time to match the neural grid.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI turns photostimulation into a binary time series over the 80 bins. For stimulated trials, any bin center between stimulation onset and offset is marked 1; all other bins are 0. Trials with `photostim_onset == 'N/A'` remain all zero.

ii.
```python
if has_stim.any():
    onset = np.zeros(n)
    dur = np.zeros(n)
    onset[has_stim] = np.array([float(x) for x in onset_str[has_stim]])
    dur[has_stim] = np.array([float(x) for x in dur_str[has_stim]])
    on_rel = start + onset - go
    off_rel = on_rel + dur
    inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
              (BIN_CENTERS[None, :] < off_rel[:, None]))
    inp[:, 1, :] = (inside & has_stim[:, None]).astype(np.float32)
```

iii. The notes justify a time-varying binary input because the decoder specification asks whether photostimulation is on at every time point, not merely whether a trial was stimulated.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation onset and offset are converted into time relative to each trial's go cue, then compared against the same bin centers used for the neural matrices.

ii.
```python
on_rel = start + onset - go
off_rel = on_rel + dur
inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
          (BIN_CENTERS[None, :] < off_rel[:, None]))
```

iii. The notes explicitly state that all inputs are expressed on the go-cue-aligned 50 ms bin grid shared with neural activity.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated NWB field. The AI derives it from the trial outcome (`hit`, `miss`, `ignore`) and the instructed side (`left` or `right`) in the trial table.

ii.
```python
outcome = tr['outcome'][keep]
instr = tr['instruction'][keep]
instr_code = np.where(instr == 'left', CHOICE_LEFT, CHOICE_RIGHT)
opposite = np.where(instr == 'left', CHOICE_RIGHT, CHOICE_LEFT)
```

iii. The notes justify this by saying the actual lick direction is logically determined by instruction and outcome: hit means correct side, miss means opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps choice to three codes: left = 0, right = 1, no lick = 2. This per-trial code is then broadcast across all 80 bins so the output tensor is time-varying in shape even though the value is constant within a trial.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
...
choice = np.where(outcome == 'hit', instr_code,
                  np.where(outcome == 'miss', opposite, CHOICE_NOLICK))
return choice.astype(np.int64)
```

```python
out[:, 0, :] = choice_codes(tr, keep)[:, None]
```

iii. The notes say constant-across-time replication was intentional so that all outputs share a common `(n_output, 80)` trial format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` column.

ii.
```python
'outcome': _dec(t['outcome'][:]),
...
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. The notes state that the NWB trial table already stores the exact three outcome categories required by the decoder task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The strings are mapped to integer classes `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all bins for the trial.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. The notes justify this as a direct categorical encoding of the trial-level outcome requested by the task.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial-table `early_lick` field.

ii.
```python
'early_lick': _dec(t['early_lick'][:]),
...
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```

iii. The notes treat this as an explicitly stored per-trial label and keep early-lick trials because early lick is one of the decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts the string label to a binary code, with `no early -> 0` and `early -> 1`, then repeats that value across the 80 bins.

ii.
```python
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```

iii. The notes justify the constant-over-time representation the same way as for choice and outcome: all outputs are stored on the shared 80-bin grid.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-camera tongue tracking time series at `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. The script uses `timestamps`, the y coordinate in column 1, and the DeepLabCut likelihood in column 2.

ii.
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
data = tt['data']
ts = tt['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. The notes identify this NWB series as the direct counterpart of the reference tongue marker stream.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI bins the side-camera tongue y trace into the go-cue-centered 50 ms trial grid. Within each trial and bin, it averages y over frames whose likelihood exceeds a visibility threshold. Bins with no visible frames are left as `NaN` during averaging and later converted to the special class 3. Session-level percentile cutoffs are then computed from all visible binned values in the retained trials of that session, and each visible bin is discretized against those cutoffs.

ii.
```python
vis = lik > TONGUE_LIKELIHOOD_THRESHOLD
...
for i in range(ntrials):
    sl = slice(lo[i], hi[i])
    ...
    b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
    counts = np.bincount(b, minlength=N_BINS)
    sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
    nz = counts > 0
    y_bin[i, nz] = sums[nz] / counts[nz]
```

```python
visible = ~np.isnan(y_bin)
vals = y_bin[visible]
p40, p60 = np.percentile(vals, [TONGUE_PCTL_LOW, TONGUE_PCTL_HIGH])
```

iii. The notes justify this as matching the decoder task's per-session discretization while treating low-likelihood frames as not visible. They specifically argue that using visible frames only avoids mixing protrusions with noise and that a visible-frame mean per 50 ms bin better matches the discretized quantity than raw frames do.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After computing session-specific 40th and 60th percentile cutoffs, the AI assigns class 0 to bins below `p40`, class 1 to bins between the cutoffs, class 2 to bins above `p60`, and class 3 to bins where the tongue was not visible.

ii.
```python
out = np.full(y_bin.shape, 3, dtype=np.int64)
...
code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
out[visible] = code[visible]
```

iii. The notes explicitly cite the decoder-task discretization rule and add that the fourth class is needed to represent bins with no visible tongue data rather than imputing them.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are aligned to the same go-cue-centered trial windows as the spikes. For each trial, the script selects frames between `go + OFF_START` and `go + OFF_END`, converts timestamps to time relative to go cue, and places them into the same 50 ms bins as the neural data.

ii.
```python
lo = np.searchsorted(ts, go_times + OFF_START)
hi = np.searchsorted(ts, go_times + OFF_END)
for i in range(ntrials):
    sl = slice(lo[i], hi[i])
    rel = ts[sl] - go_times[i]
    ...
    b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
```

iii. The notes justify this by saying spikes, video, and behavioral events all live on the same session-absolute time base, so no extra interpolation or clock correction is needed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases explicitly. A trial with no preceding tone event falls back to `go - 1.85 s`. Photostimulation `'N/A'` strings are treated as no stimulation. Sessions with no good units are excluded. Trials outside observation intervals are excluded, and trials whose entire neural matrix is zero are dropped as recording-coverage artifacts. For tongue tracking, bins with no visible frames become class 3, and an all-invisible session would return all 3s instead of failing.

ii.
```python
tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
tone = np.where(np.isnan(tone), go - 1.85, tone)
```

```python
has_stim = onset_str != 'N/A'
```

```python
silent = fr.sum(axis=(1, 2)) == 0
if n_silent:
    kept_positions = np.flatnonzero(keep)
    keep[kept_positions[silent]] = False
    fr = fr[~silent]
```

```python
out = np.full(y_bin.shape, 3, dtype=np.int64)
if visible.sum() == 0:
    return out, (np.nan, np.nan)
```

iii. The notes justify these choices as explicit handling of real data anomalies discovered during conversion. The fallback and exclusion rules are described as preferable to fabricating data, while missing tongue visibility is represented by a dedicated category because the decoder task asks for categorical output.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies full-session I/O and neural binning as the dominant costs. Concretely, the notes say unit loading (`spike_times`), spike binning, and input/output construction including video processing are the main runtime components, with session-level multiprocessing used to reduce wall-clock time.

ii.
```python
spikes, region_idx, anno, ccf_xyz = load_good_units(f, ontology)
...
fr = bin_spikes(spikes, go)
...
inp = build_inputs(tr, keep)
out, y_bin, pctls = build_outputs(f, tr, keep)
```

iii. `CONVERSION_NOTES.md` contains a runtime table estimating that spike loading, spike binning, and video/input-output processing dominate per-session cost, and it reports measured speedups from vectorized binning and multiprocessing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining explicit loops are the per-unit `searchsorted` loop in `bin_spikes` and the per-trial tongue-binning loop in `bin_tongue`. The AI treats the first as already vectorized across trials and the second as acceptable because it uses `np.bincount` inside the trial loop.

ii.
```python
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    fr[:, i, :] = np.diff(pos, axis=1)
```

```python
for i in range(ntrials):
    sl = slice(lo[i], hi[i])
    ...
    counts = np.bincount(b, minlength=N_BINS)
    sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
```

iii. The notes explicitly say neural binning is already vectorized as far as practical and that the tongue code is vectorized within each trial. No stronger justification for further vectorization is documented.

## 10-c. What processing does the code repeat multiple times?

i. The script repeats trial-table parsing and session-selection logic in a two-pass design. `read_trial_table` and `session_passes` run once in the initial selection pass over all files and again inside `process_session` for every selected file. The script also loads the ontology once in the main process and once per worker process.

ii.
```python
for fp in files:
    with h5py.File(fp, 'r') as f:
        sid = f['identifier'][()].decode()
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
```

```python
with h5py.File(filepath, 'r') as f:
    ...
    tr = read_trial_table(f)
    ok, perf, ncl, ncr, reason = session_passes(f, tr)
```

iii. No explicit justification for this repetition appears in the notes, but the code comments describe the first pass as a cheap session-selection stage before full conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does some work that is not needed by the final decoder-facing arrays. It computes session-selection statistics in pass 1 and then recomputes them in pass 2; it reads and returns `anno` and `ccf_xyz` from `load_good_units` even though they are not written into the final output; and optional `--show-processing` generates large diagnostic plots that are entirely outside the saved dataset. The saved pickle also carries diagnostic metadata such as timing and session performance that the decoder itself does not use.

ii.
```python
def load_good_units(f, ontology):
    ...
    anno = _dec(u['anno_name'][:])[good]
    ...
    ccf_xyz = exyz[eidx]
    ...
    return spikes, region_idx, anno, ccf_xyz
```

```python
if show_processing:
    plot_processing(f, tr, keep, spikes, fr, inp, out, y_bin, pctls,
                    region_idx, session_id)
```

iii. The notes justify most of this as validation and documentation: the diagnostics were used to check alignment and data integrity, and the session-selection pass was intended to enforce the paper's behavioral-session criteria before expensive conversion.
