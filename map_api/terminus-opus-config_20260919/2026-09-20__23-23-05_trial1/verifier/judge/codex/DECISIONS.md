# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files with a sorted glob over `/app/data/sub-*/*.nwb`, then opens each file once with `pynwb.NWBHDF5IO`. Inside each file it reads `nwb.trials`, `nwb.units`, `BehavioralEvents`, `BehavioralTimeSeries`, and `nwb.electrodes`.

ii. 
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
...
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    trials = nwb.trials
    units = nwb.units
    bev = nwb.acquisition['BehavioralEvents'].time_series
    bts = nwb.acquisition['BehavioralTimeSeries'].time_series
    electrodes = nwb.electrodes
```

iii. The notes and trajectory say the dataset is one NWB file per session, `pynwb` is required, and the sorted glob makes the processing deterministic.

## 1-b. How are the data split into subjects?

i. The AI assigns each session to a subject using `nwb.subject.description` when present, otherwise `nwb.subject.subject_id`. It then builds `subjects` as the sorted unique set of those labels and `subject_idx` by lookup into that list.

ii. 
```python
subject = nwb.subject.description or nwb.subject.subject_id
...
subjects = sorted({r['subject'] for r in kept})
subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. In `CONVERSION_NOTES.md`, the AI justifies this as matching the mouse nicknames used in the papers and reference `.mat` exports (for example `SC015`), while still being one-to-one with NWB subject IDs.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session identity is `nwb.identifier`, and the final output session order is the kept sessions sorted by `(subject, sess)`.

ii. 
```python
sess_name = nwb.identifier
...
kept.sort(key=lambda r: (r['subject'], r['sess']))
```

iii. The notes state that DANDI 000363 is organized as one NWB file per session, so the file boundary is the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table as the trial structure, then maps the go-cue timestamps onto those trial rows with `searchsorted(start_time, go_times_all)`. It requires every trial to receive a go cue; otherwise it skips the session.

ii. 
```python
trials = nwb.trials
n_trials = len(trials)
start_time = np.asarray(trials['start_time'].data[:])
...
go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
go = np.full(n_trials, np.nan)
gi = np.searchsorted(start_time, go_times_all, side='right') - 1
ok = (gi >= 0) & (gi < n_trials)
go[gi[ok]] = go_times_all[ok]
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
```

iii. The trajectory says the AI verified there is exactly one go cue per trial across the dataset, so trial rows plus go-cue assignment were treated as unambiguous.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies several trial filters. First it builds an `observed` mask by intersecting `units['obs_intervals']` across all good units, dropping trials with no ephys recording. Then it computes video coverage of the `[-2.5, 1.5] s` window around the go cue and keeps only trials with at least 90% coverage and finite tone onset. After neural binning it drops trials whose pooled neural matrix is entirely zero. It does **not** explicitly exclude `auto_water` or `free_water` trials in code.

ii. 
```python
observed = np.ones(n_trials, dtype=bool)
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    ...
    observed &= m
...
coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
keep_trials = np.where(keep_mask)[0]
...
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]
```

iii. The notes justify dropping unobserved trials because otherwise they appear as all-zero neural data, and dropping low-coverage video trials because the tongue output would be undefined. The trajectory also records a bug fix for all-zero neural trials caused by partial recording coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for quality-controlled units, with go-cue times from `BehavioralEvents/go_start_times` used to define the alignment windows.

ii. 
```python
go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
...
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
```

iii. The notes say spike times are the only neural representation in the NWB files, so the firing-rate matrix must be derived from them directly.

## 2-b. How is the `neural` data processed?

i. The AI subtracts each kept trial's go cue from each unit's session-clock spike times, clips to the `[-2.5, 1.5] s` window, bins with 50 ms non-overlapping bins, and divides counts by `BIN_SIZE` to get firing rates in spikes/s.

ii. 
```python
w0 = gk + T_START
w1 = gk + T_END
for ui, st in enumerate(spikes):
    ...
    rel = st[pos] - gk[trial_ids]
    bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
    np.clip(bidx, 0, N_BINS - 1, out=bidx)
    flat = trial_ids * N_BINS + bidx
    counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
    neural_all[:, ui, :] = counts
neural_all /= BIN_SIZE
```

iii. The notes and docstring say this is intended to reproduce `sliding_histogram(..., rate=True)` from the reference code, with the task-required 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `units['classification'] == 'good'`. It does not apply explicit thresholds on QC metrics and does not use `is_good_trials`.

ii. 
```python
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
```

iii. The notes argue that `classification == 'good'` is the QC-classifier verdict described in the white paper and used in the papers, and that `is_good_trials` was empirically all-true in nearly all checked cases.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by using `go_start_times` as `t = 0` for every trial and binning spikes in a fixed `[-2.5, 1.5] s` window around that event.

ii. 
```python
T_START = -2.5
T_END = 1.5
...
w0 = gk + T_START
w1 = gk + T_END
...
rel = st[pos] - gk[trial_ids]
```

iii. The notes say the NWB timestamps are on a shared session clock, so alignment only requires subtracting the trial's go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping 50 ms bins spanning 4 s. No additional smoothing or rebinning is applied after counting spikes into those bins.

ii. 
```python
BIN_SIZE = 0.05
N_BINS = int(round((T_END - T_START) / BIN_SIZE))
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The notes explicitly describe this as a deliberate deviation from the method paper's 40 ms / 3.4 ms scheme because the decoder instructions require 50 ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from the sample-epoch onset events in `BehavioralEvents/sample_start_times` together with the per-trial go-cue times. For each trial, the AI takes the last sample onset before the go cue.

ii. 
```python
sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
...
tone_onset = np.full(n_trials, np.nan)
si = np.searchsorted(start_time, sample_times, side='right') - 1
for tr_i, s in zip(si, sample_times):
    if 0 <= tr_i < n_trials and s < go[tr_i]:
        if np.isnan(tone_onset[tr_i]) or s > tone_onset[tr_i]:
            tone_onset[tr_i] = s
```

iii. The notes justify "last sample onset" by the task structure: early licks can replay the sample epoch, so the final sample onset before the go cue is the instructing tone for that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI converts the per-trial tone time into a time-varying signal by subtracting tone onset from each go-cue-centered bin center. The resulting value is continuous and measured in seconds.

ii. 
```python
tone_rel = tone_onset[keep_trials] - gk
input_all = np.zeros((nk, 2, N_BINS), dtype=np.float32)
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

iii. The notes say this matches the decoder requirement for a continuous, time-varying "time from tone onset" input.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same `BIN_CENTERS` grid used for neural binning, so each timepoint of this input corresponds directly to a neural bin.

ii. 
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
...
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

iii. The notes treat this as the natural consequence of aligning everything to the go cue and using one shared time grid for all streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, then maps those events to trials with `searchsorted(start_time, photostim_on)`.

ii. 
```python
photostim_on = np.asarray(bev['photostim_start_times'].timestamps[:])
photostim_off = np.asarray(bev['photostim_stop_times'].timestamps[:])
...
stim_on = np.full(n_trials, np.nan)
stim_off = np.full(n_trials, np.nan)
pi = np.searchsorted(start_time, photostim_on, side='right') - 1
for tr_i, on, off in zip(pi, photostim_on, photostim_off):
    if 0 <= tr_i < n_trials:
        stim_on[tr_i] = on - go[tr_i]
        stim_off[tr_i] = off - go[tr_i]
```

iii. The notes say the event timestamps exactly match the trial-table photostim fields, but are easier to use because they are already on the session clock.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the stimulation interval into a binary time series: bins whose centers fall inside `[stim_on, stim_off]` are set to 1, all others remain 0. Trials without stimulation stay all zero because their onset/offset remain `NaN`.

ii. 
```python
has_stim = np.isfinite(son)
if has_stim.any():
    input_all[has_stim, 1, :] = (
        (BIN_CENTERS[None, :] >= son[has_stim][:, None])
        & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)
```

iii. The notes justify this as satisfying the instruction that a stimulus-onset-type variable should be represented as a binary time series rather than a per-trial scalar.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are expressed relative to the same go cue used for the neural alignment, and then compared directly to `BIN_CENTERS`.

ii. 
```python
stim_on[tr_i] = on - go[tr_i]
stim_off[tr_i] = off - go[tr_i]
...
(BIN_CENTERS[None, :] >= son[has_stim][:, None])
```

iii. The notes say photostim is converted to go-cue-relative time because the decoder window is itself defined relative to the go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is not read directly from the NWB file. The AI derives it from `trials['outcome']` and `trials['trial_instruction']`.

ii. 
```python
def choice_from_trial(outcome, instruction):
    if outcome == 'ignore':
        return 2
    right = (instruction == 'right')
    if outcome == 'miss':
        right = not right
    return 1 if right else 0
```

iii. The notes and trajectory say this mirrors the behavioral logic of the task: hit means the instructed side was chosen, miss means the opposite side, and ignore means no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps choice to integer classes `0 = left`, `1 = right`, `2 = no lick`, then repeats that single-trial value across all 80 bins in output row 0.

ii. 
```python
out = np.zeros((4, N_BINS), dtype=np.int64)
out[0] = choice_from_trial(outcome[tr], instruction[tr])
```

iii. The notes justify repeating per-trial outputs across bins so all outputs share one `(n_output, n_timepoints)` format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `trials['outcome']`.

ii. 
```python
outcome = np.asarray(trials['outcome'].data[:])
```

iii. The notes say the NWB trials table already stores the three required categories directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore'`, `'miss'`, and `'hit'` to `0`, `1`, and `2`, then tiles that per-trial label across time in output row 1.

ii. 
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out[1] = OUTCOME_CODE[outcome[tr]]
```

iii. The notes describe this as a direct categorical recoding that follows the decoder specification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `trials['early_lick']`.

ii. 
```python
early_lick = np.asarray(trials['early_lick'].data[:])
```

iii. The notes say the NWB trial table already marks early-lick status explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early'` to `0` and `'early'` to `1`, then repeats that label across all 80 bins in output row 2.

ii. 
```python
out[2] = 1 if early_lick[tr] == 'early' else 0
```

iii. The notes justify the repeated-per-bin representation the same way as for choice and outcome.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position comes from `BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 as `y` and column 2 as the DeepLabCut likelihood for visibility.

ii. 
```python
tongue = bts['Camera0_side_TongueTracking']
vtime = np.asarray(tongue.timestamps[:])
vdata = np.asarray(tongue.data[:])
tongue_y = vdata[:, 1]
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
```

iii. The notes justify this as the session's side-view tongue tracking stream, aligned to the same session clock as the spikes and events.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first thresholds tongue visibility with `likelihood > 0.9`. For each kept trial it averages visible-frame `y` values within each 50 ms bin, leaving `NaN` if no visible frame falls in that bin. After all trial windows are binned, it computes per-session 40th and 60th percentiles over the visible binned values from those kept windows, then digitizes the bins into classes.

ii. 
```python
LIKELIHOOD_THRESH = 0.9
...
tongue_bin_y = np.full((nk, N_BINS), np.nan)
for k in range(nk):
    ...
    sel = tongue_vis[lo:hi]
    if not sel.any():
        continue
    vt = vtime[lo:hi][sel] - g
    vy = tongue_y[lo:hi][sel]
    bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
    ...
    tongue_bin_y[k, nz] = sums[nz] / cnts[nz]
...
visible = np.isfinite(tongue_bin_y)
if visible.sum() >= 10:
    p40, p60 = np.percentile(tongue_bin_y[visible], [40, 60])
```

iii. The notes say this differs from the papers because the decoder task requires a distinct "not visible" class; the AI therefore uses likelihood thresholding rather than mean-value imputation for occluded tongue frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses four classes: `0` for visible bins below the session 40th percentile, `1` for visible bins between the 40th and 60th percentiles, `2` for visible bins above the 60th percentile, and `3` for bins with no visible frame.

ii. 
```python
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)
if np.isfinite(p40):
    tongue_class[visible & (tongue_bin_y < p40)] = 0
    tongue_class[visible & (tongue_bin_y >= p40) & (tongue_bin_y <= p60)] = 1
    tongue_class[visible & (tongue_bin_y > p60)] = 2
```

iii. The notes justify this with the task specification's percentile-based per-session discretization and the explicit class for "not visible."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are treated as session-clock times, and each kept trial is windowed from `go + T_START` to `go + T_END`. The selected frames are shifted by that trial's go cue and binned onto the same 50 ms grid as the neural data.

ii. 
```python
lo = np.searchsorted(vtime, g + T_START)
hi = np.searchsorted(vtime, g + T_END)
...
vt = vtime[lo:hi][sel] - g
bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
```

iii. The notes say using the stored timestamps is equivalent to the reference alignment but more robust than reconstructing frame times from frame index.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data mainly by exclusion. Sessions are skipped for missing go cues, no good units, poor behavioral performance, or insufficient video coverage. Trials are dropped if they are outside `obs_intervals`, have too little video coverage, lack a detectable tone onset, or yield an all-zero neural matrix. For tongue data, missing visible frames are not imputed; those bins become class 3 (`not visible`).

ii. 
```python
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
if not (performance > MIN_PERFORMANCE
        and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    info['skipped'] = 'behavioural session criteria'
    return info
...
if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
    info['skipped'] = 'video does not cover the analysis window'
    return info
...
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
...
nonempty = neural_all.any(axis=(1, 2))
```

iii. The notes justify these exclusions as preventing fabricated all-zero neural trials, undefined tongue outputs, and low-quality sessions that the papers excluded. The trajectory shows the observed-trial and all-zero-trial filters were added after decoder warnings.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are opening/reading each NWB file, reading the full spike trains for all kept units, reading the video arrays, and the per-unit spike binning loop. The script records timings for these stages per session.

ii. 
```python
timings['open'] = time.time() - t_open
...
timings['video_read'] = time.time() - t0
...
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
timings['spikes_read'] = time.time() - t0
...
for ui, st in enumerate(spikes):
    ...
timings['bin_trials'] = time.time() - t0
```

iii. The notes say the runtime is dominated by NWB I/O and per-unit spike processing, not by the later bookkeeping.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain. The AI still loops over good units to build the `observed` mask, loops over sample and photostim events to assign them to trials, loops over units for neural binning, loops over kept trials for tongue binning, and loops again over kept trials to assemble output arrays.

ii. 
```python
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    ...

for tr_i, s in zip(si, sample_times):
    ...

for tr_i, on, off in zip(pi, photostim_on, photostim_off):
    ...

for ui, st in enumerate(spikes):
    ...

for k in range(nk):
    ...

for k, tr in enumerate(keep_trials):
    ...
```

iii. The notes justify the remaining loops as clarity/performance tradeoffs, and emphasize that the expensive neural loop is already vectorized across trials with one `bincount` per unit.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats some indexing and windowing work. It searches video timestamps once to estimate coverage and again to bin tongue frames, repeatedly applies `searchsorted`-based event-to-trial assignment for go, tone, and photostim, and later performs `subjects.index(...)` / `all_regions.index(...)` style lookups during assembly.

ii. 
```python
vlo = np.searchsorted(vtime, win0)
vhi = np.searchsorted(vtime, win1)
...
for k in range(nk):
    lo = np.searchsorted(vtime, g + T_START)
    hi = np.searchsorted(vtime, g + T_END)
```

```python
subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
brain_region_idx = [np.array([all_regions.index(l) for l in r['region_labels']], dtype=np.int64)
                    for r in kept]
```

iii. The notes still frame the converter as a single pass over files with one read of spikes and video per session; they do not explicitly justify these smaller repeated computations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code performs extra processing that is not needed by the decoder inputs/outputs themselves: behavioral session-performance screening, video-coverage diagnostics, detailed timing/diagnostic bookkeeping, and subject/region naming logic aimed at reproducing paper conventions. Some of this only affects filtering or metadata.

ii. 
```python
performance = hit[responded].sum() / max(responded.sum(), 1)
n_correct_left = int((hit & ctrl & (instruction == 'left')).sum())
n_correct_right = int((hit & ctrl & (instruction == 'right')).sum())
...
coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)
...
timings['open'] = time.time() - t_open
...
'session_info': [{'session': r['sess'], 'subject': r['subject'],
                  'n_trials_total': r['n_trials_total'], ...
                  'performance': r['performance'], ...} for r in kept],
```

iii. The notes justify these additions as consistency checks and paper matching, but they are not required to construct the decoder-ready arrays themselves.
