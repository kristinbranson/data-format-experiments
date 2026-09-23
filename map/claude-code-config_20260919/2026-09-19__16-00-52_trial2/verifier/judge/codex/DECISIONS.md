# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats the dataset as one NWB file per session under `/app/data/sub-*/*.nwb`, discovers files with a glob, and reads each file once with `h5py`. Within each file it loads the trials table, behavioral events, tongue tracking stream, unit metadata, and spike times into plain NumPy arrays before processing the session.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
```

```python
with h5py.File(path, 'r') as f:
    trials = f['intervals/trials']
    ev = f['acquisition/BehavioralEvents']
    ...
    d['subject'] = f['general/subject/subject_id'][()].decode()
    ...
    d['go_time'] = ev['go_start_times']['timestamps'][:][et]
    ...
    tongue = tk['data'][:]
    ...
    spike_times = u['spike_times'][:]
```

iii. `CONVERSION_NOTES.md` says the archive is organized as one NWB per recording session and that `read_session` should make one pass over each file, pulling trials, behavioral events, tongue tracking, and spike times.

## 1-b. How are the data split into subjects?

i. The AI reads each subject from `general/subject/subject_id` inside the NWB file, stores it per session, then builds `subjects` as the sorted unique subject ids and `subject_idx` as the index for each kept session.

ii.
```python
d['subject'] = f['general/subject/subject_id'][()].decode()
```

```python
subjects = sorted({r['subject'] for r in results})
sub_index = {s: i for i, s in enumerate(subjects)}
...
'subjects': subjects,
'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The notes identify `general/subject/subject_id` as the NWB counterpart of the mouse id and explicitly map it to `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. It derives a session id from the filename, processes each file independently, and later sorts accepted sessions by that derived id before assembly.

ii.
```python
def session_id_from_path(path):
    base = os.path.basename(path)
    sub = base.split('_')[0].replace('sub-', '')
    ses = base.split('_')[1].replace('ses-', '')
    return f'{sub}_{ses}'
```

```python
results.sort(key=lambda r: r['session_id'])
```

iii. `CONVERSION_NOTES.md` Step 2 states that the archive is one NWB file per recording session, so file boundaries define session boundaries.

## 1-d. How are the data split into trials?

i. The AI starts from the NWB trials table and the per-trial go cue timestamps, then immediately restricts to the subset of behavioral trials covered by ephys using `units/obs_intervals`. After that, later filtering is applied for water/video/session criteria.

ii.
```python
trial_start_all = trials['start_time'][:]
...
obs = u['obs_intervals'][0:oi_index[0]]
ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
...
d['trial_start'] = trial_start_all[et]
...
d['go_time'] = ev['go_start_times']['timestamps'][:][et]
```

```python
go = d['go_time']
assert len(go) == len(d['trial_start'])
win_start = go + T_START
```

iii. The notes say the AI found that some sessions have behavioral trials without spike coverage and therefore reconstructed the observed-trial subset from `obs_intervals`, asserting alignment to the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in three stages. First, it only keeps the trials covered by ephys (`obs_intervals`). Second, it drops `auto_water` and `free_water` trials. Third, it drops trials whose tongue video covers less than 90% of the extracted window. Sessions are then rejected if post-filter control-trial performance is too low or there are too few correct left/right trials.

ii.
```python
obs = u['obs_intervals'][0:oi_index[0]]
ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
...
et = ephys_trials
```

```python
keep = (~d['auto_water']) & (~d['free_water'])
cov = video_coverage(d, win_start)
keep &= cov >= MIN_VIDEO_COVERAGE
trial_idx = np.where(keep)[0]
```

```python
perf, n_left, n_right = session_performance(d, trial_idx)
if perf <= MIN_PERFORMANCE:
    return None, info
if n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    return None, info
```

iii. The notes justify the `obs_intervals` restriction as necessary because some behavioral trials have no spikes, justify dropping water trials because reward is not contingent on behavior, and justify the video filter because the tongue output cannot otherwise be defined. They also intentionally keep photostim, early-lick, and ignore trials because those are required decoder inputs/outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data are derived from unit spike times in `units/spike_times`, together with `units/spike_times_index` to unrag the data, and the go cue timestamps in `BehavioralEvents/go_start_times` to define trial windows.

ii.
```python
st_index = u['spike_times_index'][:]
starts = np.concatenate([[0], st_index[:-1]])
spike_times = u['spike_times'][:]
unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]
```

```python
go = d['go_time']
win_start = go + T_START
fr = bin_spikes(d['unit_spikes'], win_start)
```

iii. The notes map `units/spike_times` plus go-cue times directly to the target `neural` field and describe this as the NWB counterpart of the reference spike data.

## 2-b. How is the `neural` data processed?

i. The AI converts spike times to firing rates in Hz in 80 non-overlapping 50 ms bins from -2.5 s to +1.5 s around the go cue. It uses a vectorized single-pass histogram over all spikes in the session: assign each spike to a trial window, assign it to a bin within that trial, count with `np.bincount`, and divide by the bin width. No smoothing or normalization is applied.

ii.
```python
def bin_spikes(unit_spikes, win_start):
    ...
    ti = np.searchsorted(win_start, t, side='right') - 1
    ...
    b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
    ...
    flat = (uidx * n_trials + ti) * N_BINS + b
    counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)
    counts = counts.reshape(n_units, n_trials, N_BINS)
    return (counts / BIN_WIDTH).astype(np.float32)
```

iii. The notes explicitly say neural activity should match the reference `sliding_histogram(..., rate=True)` conceptually while using a faster vectorized implementation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps units whose `classification` is `'good'`, whose CCF annotation maps to one of its predefined major regions, and that are not silent across all extracted bins after trial selection. It also adds hemisphere labels based on CCF `x` coordinate.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
good = classification == 'good'
...
regions = np.array([annotation_to_region(a) or '' for a in anno])
keep_unit = good & (regions != '')
```

```python
fr = bin_spikes(d['unit_spikes'], win_start)
fr = fr[:, trial_idx, :]
active = fr.any(axis=(1, 2))
fr = fr[active]
unit_region = d['unit_region'][active]
```

iii. The notes justify the `'good'` classifier filter as matching the QC white paper, justify mapped-region filtering so every kept neuron has a named region, and justify dropping silent units as matching the intent of `check_fr`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to go cue onset. For each trial the AI computes `win_start = go_time - 2.5` and bins spikes on that session-time grid, so the same bin index corresponds to the same time relative to the go cue across all trials.

ii.
```python
go = d['go_time']
win_start = go + T_START
```

```python
b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
```

iii. The notes repeatedly state that all streams share the session clock and that the go cue is the alignment event used by the reference processing and required by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins, producing 80 time bins per trial over a 4 s window. There is no additional temporal rebinning after this; the spike times are directly binned onto that grid.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_STOP = 1.5
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. The notes describe this as a deliberate decoder-task deviation from the paper’s 40 ms / 3.4 ms processing.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from `sample_start_times` and `go_start_times`. The AI uses the last sample-epoch onset at or before each trial’s go cue as the tone onset for that trial.

ii.
```python
def tone_onset_rel(d):
    si = np.searchsorted(d['sample_start'], d['go_time'], side='right') - 1
    assert (si >= 0).all(), 'a trial has no preceding sample epoch'
    return d['sample_start'][si] - d['go_time']
```

iii. The notes say early licks can replay the sample epoch, so the relevant instruction tone is the last sample onset before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI first computes the tone onset relative to the go cue, then converts bin centers from go-cue-relative time to seconds since the tone onset by subtracting that relative offset. The result is a continuous ramp with slope 1 across the 80 bins.

ii.
```python
tone_rel = tone_onset_rel(d)[trial_idx]
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. The notes describe this as “bin centre − (tone_onset − go_cue)” and show it as a decoder-specific continuous input.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on exactly the same 80 go-cue-centered bin centers as the neural data, so each input timepoint corresponds to the same time bin used for firing rates.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
...
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. The notes state that neural activity, decoder inputs, and decoder outputs all share the same go-cue-relative bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the behavioral-event streams `photostim_start_times` and `photostim_stop_times`, then maps those absolute session times back onto trials using `trial_start`.

ii.
```python
d['stim_on'] = ev['photostim_start_times']['timestamps'][:] \
    if 'photostim_start_times' in ev else np.zeros(0)
d['stim_off'] = ev['photostim_stop_times']['timestamps'][:] \
    if 'photostim_stop_times' in ev else np.zeros(0)
```

```python
ti = np.searchsorted(d['trial_start'], d['stim_on'], side='right') - 1
...
on[ti] = s_on - d['go_time'][ti]
off[ti] = s_off - d['go_time'][ti]
```

iii. The notes justify using recorded photostim event times rather than assuming a fixed delay-epoch stimulation window.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts each trial’s stimulation on/off times into go-cue-relative times, then samples the binary laser state at each 50 ms bin center. Bins whose centers fall between stimulation onset and offset are set to 1, otherwise 0.

ii.
```python
on, off = photostim_windows(d)
on, off = on[trial_idx], off[trial_idx]
has_stim = np.isfinite(on)
...
stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)
        ).astype(np.float32)
```

iii. In Step 10 the notes say this was chosen after finding that raw off timestamps can overshoot the go cue by ~1 ms; center sampling avoids spuriously lighting the go-cue bin and yields the expected 10 bins for a 0.5 s photostim epoch.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim times are converted to times relative to each trial’s go cue, and the binary signal is sampled at the same bin centers used for the neural firing rates.

ii.
```python
on[ti] = s_on - d['go_time'][ti]
off[ti] = s_off - d['go_time'][ti]
...
stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)
        ).astype(np.float32)
```

iii. The notes explicitly describe all time-varying streams as sharing the go-cue-relative binning grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read from a dedicated field. The AI derives it from `outcome` plus `trial_instruction`: hit means the instructed side, miss means the opposite side, ignore means no lick.

ii.
```python
def choice_from_trials(d):
    ch = np.full(len(d['outcome']), 2, dtype=np.int64)
    hit = d['outcome'] == 'hit'
    miss = d['outcome'] == 'miss'
    left = d['instruction'] == 'left'
    ch[hit & left] = 0
    ch[hit & ~left] = 1
    ch[miss & left] = 1
    ch[miss & ~left] = 0
    return ch
```

iii. The notes say this matches the reference logic and was checked against the first post-go lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The derived choice is encoded as `0=left`, `1=right`, `2=no lick`, then repeated across all 80 bins so it can share the common `(n_output, n_timepoints)` output format with the tongue trace.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ...
]
```

```python
choice = choice_from_trials(d)[trial_idx]
...
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. The notes explicitly say per-trial outputs are broadcast over time so all outputs have a common shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trial-table `outcome` field.

ii.
```python
d['outcome'] = _decode(trials['outcome'][:])[et]
```

iii. The notes list `outcome` as the direct NWB counterpart of the reference behavioral report variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps outcome strings to integers `ignore=0`, `miss=1`, `hit=2`, then repeats the per-trial value across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_map[o] for o in d['outcome']])[trial_idx]
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. The notes describe this as a direct categorical output and part of the shared output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial-table `early_lick` field.

ii.
```python
d['early_lick'] = _decode(trials['early_lick'][:])[et]
```

iii. The notes map `early_lick` directly to the target output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI converts `early` to 1 and `no early` to 0, then repeats the per-trial category across all 80 bins.

ii.
```python
early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]
```

```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. The notes say per-trial outputs are time-broadcast so that the whole output block has one uniform shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The tongue output is derived from the side-view DeepLabCut stream `Camera0_side_TongueTracking`: timestamps, y position, and likelihood. Only frames above a likelihood threshold are treated as visible.

ii.
```python
tk = bts['Camera0_side_TongueTracking']
tongue = tk['data'][:]
d['tongue_t'] = tk['timestamps'][:]
d['tongue_y'] = tongue[:, 1]
d['tongue_lik'] = tongue[:, 2]
```

iii. The notes identify this as the side-view tongue marker stream used by the reference video-processing code.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI thresholds visibility at likelihood `> 0.5`, averages visible tongue-y values within each 50 ms trial bin, and then computes session-level 40th/60th percentiles from the visible bin means of the kept converted trials. Bins with no visible frame remain class 3.

ii.
```python
visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH
...
nvis = np.bincount(flat, minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
ysum = np.bincount(flat, weights=vy,
                   minlength=n_trials * N_BINS).reshape(n_trials, N_BINS)
has = nvis > 0
ybar[has] = ysum[has] / nvis[has]
```

```python
out = np.full(ybar.shape, 3, dtype=np.int64)
if has.sum() < 2:
    return out, np.nan, np.nan
p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
out[has] = cls[has]
```

iii. The notes justify likelihood-thresholding because occluded frames have meaningless coordinates. They also say the percentiles were intentionally changed during iteration from whole-session raw frames to per-bin values over the converted trials so the visible bins would land at an exact 40/20/40 split.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four categories: `0` below the 40th percentile, `1` between the 40th and 60th percentiles, `2` above the 60th percentile, and `3` when no visible frame is present in the bin.

ii.
```python
OUTPUT_VALUES = [
    ['left', 'right', 'no lick'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['<40th pct', '40th-60th pct', '>60th pct', 'not visible'],
]
```

```python
cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
out[has] = cls[has]
```

iii. The notes frame this as implementing the decoder task’s required per-session discretization plus an explicit “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI bins tongue frames on the same go-cue-relative 50 ms grid as the neural data. Frame times are assigned to trial windows using `win_start`, then to bins within those windows by offset from `win_start`.

ii.
```python
win_end = win_start + N_BINS * BIN_WIDTH
ti = np.searchsorted(win_start, vt, side='right') - 1
...
b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)
```

iii. The notes emphasize that spikes, events, and video timestamps are all on the same session clock, so no extra cross-stream alignment is required beyond using the common go-cue-relative grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by dropping trials or sessions when it considers the missingness fatal, and by using explicit placeholder categories otherwise. Sessions with poor behavioral performance or implausible tongue tracking are rejected. Trials without enough video are dropped. Trials without photostim get all-zero photostim traces via NaN bounds, and bins without visible tongue frames become class 3.

ii.
```python
if perf <= MIN_PERFORMANCE:
    info['rejected'] = 'behavioural performance'
    return None, info
...
if vis_frac > MAX_TONGUE_VISIBLE_FRACTION:
    info['rejected'] = 'tongue tracking failure (implausible visible fraction)'
    return None, info
```

```python
keep &= cov >= MIN_VIDEO_COVERAGE
```

```python
on = np.full(n, np.nan)
off = np.full(n, np.nan)
...
out = np.full(ybar.shape, 3, dtype=np.int64)
```

iii. Step 10 of the notes documents these as edge-case fixes: ephys-prefix sessions are restricted by `obs_intervals`, photostim timestamp overshoot is handled by bin-center sampling, degenerate tongue tracking causes session rejection, and missing video causes trial rejection.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies NWB I/O, spike-time loading, tongue-tracking loading, and naive per-unit/per-frame temporal binning as the main expensive operations. Its final implementation is organized around reducing those costs and prints timing breakdowns per session.

ii.
```python
t_read = time.time() - t0
...
t_neural = time.time() - t1
...
t_io = time.time() - t2
...
info.update(..., t_read=t_read, t_neural=t_neural, t_io=t_io, t_pack=t_pack,
            t_total=time.time() - t0)
```

```python
print(f'[{k+1}/{len(files)}] {i["session_id"]}: {i["n_neurons"]} neurons, '
      f'{i["n_trials"]} trials, perf={i["performance"]:.3f}, '
      f'{i["t_total"]:.1f}s (read {i["t_read"]:.1f} / bin {i["t_neural"]:.1f} '
      f'/ io {i["t_io"]:.2f} / pack {i["t_pack"]:.2f})', flush=True)
```

iii. Step 6 of the notes says the expected bottlenecks were reading `spike_times`, reading the video stream, and binning spikes/video; it then explains the optimizations added to reduce those costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI’s own implementation already vectorizes the expensive spike and tongue binning that a naive solution would loop over. The remaining notable Python loops are minor ones for hemisphere assignment, per-trial packing into lists, and optional plotting.

ii.
```python
for j, i in enumerate(idx):
    x = ccf_x[i]
    if np.isfinite(x):
        hemi[j] = 'left' if x >= ML_MIDLINE else 'right'
    else:
        ml = probe_ml.get(gnames[i], np.nan)
        hemi[j] = 'left' if (np.isfinite(ml) and ml < 0) else 'right'
```

```python
neural = [np.ascontiguousarray(fr[:, i, :]) for i in range(n_tr)]
inputs = [np.stack([time_from_tone[i], stim[i]]).astype(np.float32)
          for i in range(n_tr)]
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. The notes explicitly say the naive per-unit/per-trial spike binning and analogous video binning were the obvious vectorization targets, and that the final code replaced them with one-session `searchsorted` + `bincount` passes.

## 10-c. What processing does the code repeat multiple times?

i. Core conversion work is mostly done once per session: each file is read once, spike binning is run once, and each input/output stream is derived once. The main repeated work outside the core conversion is optional debug plotting and summary statistics generation after assembly.

ii.
```python
with h5py.File(path, 'r') as f:
    ...
```

```python
fr = bin_spikes(d['unit_spikes'], win_start)
...
tone_rel = tone_onset_rel(d)[trial_idx]
on, off = photostim_windows(d)
...
tongue, p_lo, p_hi = tongue_classes(d, win_start[trial_idx])
```

iii. Step 6 of the notes describes the pipeline as a single-pass per-session conversion and presents the optimizations as avoiding redundant passes over spikes and video.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does some extra processing for diagnostics and provenance that is not required for downstream decoding: optional `--show-processing` debug plots, per-session timing/provenance bookkeeping, and a separate `_session_info.json` sidecar. When plotting is enabled, it also constructs `debug` payloads and recomputes derived displays purely for visualization.

ii.
```python
if want_debug:
    result['debug'] = dict(
        go=go, trial_idx=trial_idx, win_start=win_start,
        tone_rel=tone_rel, stim_on=on, stim_off=off,
        tongue_t=d['tongue_t'], tongue_y=d['tongue_y'], tongue_lik=d['tongue_lik'],
        ...
    )
```

```python
if 'debug' in res:
    if n_plots < 2:
        make_processing_plot(res, f'/app/processing_{res["session_id"]}.png')
        n_plots += 1
    del res['debug']
```

```python
with open(os.path.splitext(args.outfile)[0] + '_session_info.json', 'w') as f:
    json.dump(infos, f, indent=1, default=float)
```

iii. The notes justify these extra computations as verification and sanity-check tooling rather than part of the final decoder dataset itself.
