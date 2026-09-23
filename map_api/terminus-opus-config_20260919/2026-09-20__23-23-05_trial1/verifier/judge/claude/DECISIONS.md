# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` and processes each with `pynwb.NWBHDF5IO`. Each file is one session. Trials, units, behavioral events, and video data are extracted from within each NWB file. Multiprocessing (16 workers by default) is used to parallelize across sessions.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with NWBHDF5IO(path, mode='r', load_namespaces=True) as io:
    nwb = io.read()
    sess_name = nwb.identifier
    subject = nwb.subject.description or nwb.subject.subject_id
    trials = nwb.trials
    # ...
    bev = nwb.acquisition['BehavioralEvents'].time_series
    go_times_all = np.asarray(bev['go_start_times'].timestamps[:])
    sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
    # ...
    units = nwb.units
```

iii. The AI documented in CONVERSION_NOTES.md that NWB is the published format and `pynwb` is its standard reader. The 174 files and 28 subjects match `dandiset.yaml`. Multiprocessing was added for speed.

## 1-b. How are the data split into subjects?

i. The AI identifies subjects using `nwb.subject.description` (mouse nickname like "SC015"), falling back to `nwb.subject.subject_id` if description is empty. The unique sorted set of subject names forms `subjects`, and each session is indexed into that list.

ii.
```python
subject = nwb.subject.description or nwb.subject.subject_id
# ...
subjects = sorted({r['subject'] for r in kept})
subject_idx = np.array([subjects.index(r['subject']) for r in kept], dtype=np.int64)
```

iii. The AI chose `description` (e.g., "SC015") because it matches the mouse names used in the papers and in the reference `.mat` file names.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. Sessions are identified by `nwb.identifier`. The AI sorts sessions by `(subject, session_name)` after processing. The AI applies three session-level filters: (1) at least 1 good unit, (2) the data paper's behavioral session-selection criteria (performance > 65% and >= 50 correct lick-left and lick-right trials), (3) side-view video covering the analysis window. This yields 138 sessions from the original 174.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
if not (performance > MIN_PERFORMANCE
        and n_correct_left >= MIN_CORRECT_PER_DIRECTION
        and n_correct_right >= MIN_CORRECT_PER_DIRECTION):
    info['skipped'] = 'behavioural session criteria'
    return info
# ...
if np.mean(coverage[observed] >= MIN_VIDEO_COVERAGE) < MIN_SESSION_VIDEO_FRAC:
    info['skipped'] = 'video does not cover the analysis window'
    return info
```

iii. The AI justified the behavioral criteria by noting these are the papers' own session-selection criteria. The video criterion was added because 6 sessions have video stopping at the go cue, making the tongue output undefined for the post-go period.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`nwb.trials`). The AI assigns go-cue times to trials using `searchsorted` on trial start times. Sessions where any trial lacks a go cue are skipped.

ii.
```python
n_trials = len(trials)
start_time = np.asarray(trials['start_time'].data[:])
# ...
go = np.full(n_trials, np.nan)
gi = np.searchsorted(start_time, go_times_all, side='right') - 1
ok = (gi >= 0) & (gi < n_trials)
go[gi[ok]] = go_times_all[ok]
if np.any(np.isnan(go)):
    return dict(sess=sess_name, skipped='missing go cue')
```

iii. The AI verified that there is exactly one go cue per trial in all 174 sessions. The `searchsorted` approach finds which trial each go-cue event belongs to.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three trial-level filters: (1) trials must be within the `obs_intervals` of ALL good units (intersection), (2) video must cover >= 90% of the analysis window, (3) trials must have at least one spike from any good unit. No behavioral quality filter (no free_water, no auto_water filter) is applied. A session must have >= 2 usable trials.

ii.
```python
# obs_intervals intersection across all good units
observed = np.ones(n_trials, dtype=bool)
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    oi = oi[(oi >= 0) & (oi < n_trials)]
    m = np.zeros(n_trials, dtype=bool)
    m[oi] = True
    observed &= m

# video coverage
coverage = (vhi - vlo) / ((T_END - T_START) / VIDEO_DT)
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)

# empty spike trials
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]
```

iii. The AI justified the obs_intervals filter because in 8 sessions the ephys covers only part of the behavioral session. The video coverage filter ensures the tongue output can be computed. The empty-spike filter catches genuine acquisition dropouts (2,150 trials). Free_water and auto_water trials are deliberately kept because the AI states the decoder benefits from all trial types.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` for units with `units['classification'] == 'good'`. Go-cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
# ...
lo = np.searchsorted(st, w0)
hi = np.searchsorted(st, w1)
```

iii. `spike_times` is the only neural representation in the file, and good units are identified by the QC classifier.

## 2-b. How is the `neural` data processed?

i. For each unit, spike times within each trial's window [go + T_START, go + T_END] are identified with `searchsorted`. Within each window, spikes are binned into 50ms bins using `np.floor((rel - T_START) / BIN_SIZE)` and `np.bincount`. Counts are divided by `BIN_SIZE` (0.05 s) to get firing rates in Hz. No smoothing, normalization, or baseline subtraction.

ii.
```python
neural_all = np.zeros((nk, n_good, N_BINS), dtype=np.float32)
for ui, st in enumerate(spikes):
    lo = np.searchsorted(st, w0)
    hi = np.searchsorted(st, w1)
    cnt = hi - lo
    tot = int(cnt.sum())
    # ... builds flat index arrays ...
    rel = st[pos] - gk[trial_ids]
    bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
    np.clip(bidx, 0, N_BINS - 1, out=bidx)
    flat = trial_ids * N_BINS + bidx
    counts = np.bincount(flat, minlength=nk * N_BINS).reshape(nk, N_BINS)
    neural_all[:, ui, :] = counts
neural_all /= BIN_SIZE      # spikes / s
```

iii. The AI states this is equivalent to the reference `sliding_histogram(..., rate=True)` with bin_width == stride == BIN_SIZE.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units['classification'] == 'good'` are kept. No additional metric thresholds (e.g., firing rate) are applied. Sessions with no good units are dropped.

ii.
```python
classification = np.asarray(units['classification'].data[:])
good = np.where(classification == 'good')[0]
n_good = len(good)
if n_good == 0:
    info['skipped'] = 'no good units'
    return info
```

iii. The AI notes this is the QC-classifier verdict from the ChenLiu white paper. The method paper's 2 Hz firing rate threshold is analysis-specific, not dataset curation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times and go-cue times are on the same session-absolute clock. The window for each trial is computed as `[go + T_START, go + T_END]`, and spike times within this window are made relative to the go cue by subtracting it: `rel = st[pos] - gk[trial_ids]`.

ii.
```python
w0 = gk + T_START
w1 = gk + T_END
# ...
rel = st[pos] - gk[trial_ids]
bidx = np.floor((rel - T_START) / BIN_SIZE).astype(np.int64)
```

iii. Everything in the NWB file is on one global clock, so alignment only requires subtracting the go-cue time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins spanning -2.5 s to +1.5 s relative to the go cue, yielding 80 bins per trial. Bin edges are defined once and reused for all trials and sessions.

ii.
```python
T_START = -2.5           # s relative to go cue
T_END = 1.5              # s relative to go cue
BIN_SIZE = 0.05          # s (50 ms bins)
N_BINS = int(round((T_END - T_START) / BIN_SIZE))  # 80
BIN_EDGES = T_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_SIZE / 2
```

iii. The 50 ms bin width and -2.5 to +1.5 s window are set by the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onsets) and the go-cue time for each trial. The tone for a trial is the last `sample_start_times` event before the go cue, found by iterating through sample times and assigning each to a trial.

ii.
```python
sample_times = np.asarray(bev['sample_start_times'].timestamps[:])
# ...
tone_onset = np.full(n_trials, np.nan)
si = np.searchsorted(start_time, sample_times, side='right') - 1
for tr_i, s in zip(si, sample_times):
    if 0 <= tr_i < n_trials and s < go[tr_i]:
        if np.isnan(tone_onset[tr_i]) or s > tone_onset[tr_i]:
            tone_onset[tr_i] = s
```

iii. An early lick replays the sample epoch, so a trial can have multiple tone onsets; the last one before the go cue is the instructing tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The value at each bin is `BIN_CENTER - tone_rel`, where `tone_rel = tone_onset - go` (negative, since tone is before go). This gives the time from tone onset to the bin center.

ii.
```python
tone_rel = tone_onset[keep_trials] - gk
input_all[:, 0, :] = (BIN_CENTERS[None, :] - tone_rel[:, None]).astype(np.float32)
```

iii. Algebraically: `BIN_CENTER - (tone - go) = BIN_CENTER + (go - tone)`, which equals the time from tone onset to the bin center (since bin centers are relative to go cue).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin grid defined relative to the go cue (`BIN_CENTERS`), ensuring each input bin corresponds to the same time interval as the corresponding neural bin.

ii. Same `BIN_CENTERS` array is used for both neural binning and input computation.

iii. N/A.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times` (session-absolute timestamps). These are assigned to trials using `searchsorted` on trial start times.

ii.
```python
photostim_on = np.asarray(bev['photostim_start_times'].timestamps[:])
photostim_off = np.asarray(bev['photostim_stop_times'].timestamps[:])
# ...
stim_on = np.full(n_trials, np.nan)
stim_off = np.full(n_trials, np.nan)
pi = np.searchsorted(start_time, photostim_on, side='right') - 1
for tr_i, on, off in zip(pi, photostim_on, photostim_off):
    if 0 <= tr_i < n_trials:
        stim_on[tr_i] = on - go[tr_i]
        stim_off[tr_i] = off - go[tr_i]
```

iii. The AI uses the BehavioralEvents timestamps directly rather than the trials table `photostim_onset`/`photostim_duration` columns. The AI verified that both sources agree exactly (`np.allclose`).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is 1 if its center falls within [onset, offset] (inclusive on both ends) relative to the go cue, and 0 otherwise. Non-stimulated trials have NaN onset/offset and remain 0.

ii.
```python
son = stim_on[keep_trials]
soff = stim_off[keep_trials]
has_stim = np.isfinite(son)
if has_stim.any():
    input_all[has_stim, 1, :] = (
        (BIN_CENTERS[None, :] >= son[has_stim][:, None])
        & (BIN_CENTERS[None, :] <= soff[has_stim][:, None])).astype(np.float32)
```

iii. The AI documented that photostim always occurs in the late delay and ends before the go cue.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are expressed relative to the go cue (`on - go[tr_i]`), which is the same reference as the neural bin centers. Comparing bin centers directly against these relative times ensures alignment.

ii. See 4-b code: `BIN_CENTERS >= son` and `BIN_CENTERS <= soff` where son/soff are go-cue-relative.

iii. N/A.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. From `trials['outcome']` (hit/miss/ignore) and `trials['trial_instruction']` (left/right). There is no explicit choice column in the NWB file.

ii.
```python
outcome = np.asarray(trials['outcome'].data[:])
instruction = np.asarray(trials['trial_instruction'].data[:])
# ...
def choice_from_trial(outcome, instruction):
    if outcome == 'ignore':
        return 2
    right = (instruction == 'right')
    if outcome == 'miss':
        right = not right
    return 1 if right else 0
```

iii. Hit means the animal licked the instructed side, miss means the opposite side, ignore means no lick. The AI verified this derivation agrees with actual lick events on 99.75% of responded trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right, 2=no lick. It is a per-trial value repeated across all 80 bins (constant across time within a trial).

ii.
```python
out = np.zeros((4, N_BINS), dtype=np.int64)
out[0] = choice_from_trial(outcome[tr], instruction[tr])
```

iii. Choice is one value per trial, so it is broadcast across bins so all outputs share a single `(n_output, n_timepoints)` array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials['outcome']`, which holds strings 'ignore', 'miss', 'hit'.

ii.
```python
outcome = np.asarray(trials['outcome'].data[:])
# ...
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
out[1] = OUTCOME_CODE[outcome[tr]]
```

iii. The trials table stores outcome explicitly with the three required categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: 0=ignore, 1=miss, 2=hit. Per-trial value repeated across all 80 bins.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
out[1] = OUTCOME_CODE[outcome[tr]]
```

iii. Direct mapping, no additional processing needed.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials['early_lick']`, which holds strings 'early' and 'no early'.

ii.
```python
early_lick = np.asarray(trials['early_lick'].data[:])
# ...
out[2] = 1 if early_lick[tr] == 'early' else 0
```

iii. The trials table flags early licking explicitly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Per-trial value repeated across all 80 bins.

ii.
```python
out[2] = 1 if early_lick[tr] == 'early' else 0
```

iii. Direct mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which has `(n_frames, 3)` = (x, y, likelihood) with timestamps. Column 1 (y) is the position; column 2 (likelihood) determines visibility.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries'].time_series
tongue = bts['Camera0_side_TongueTracking']
vtime = np.asarray(tongue.timestamps[:])
vdata = np.asarray(tongue.data[:])  # (n_frames, 3): x, y, likelihood
tongue_y = vdata[:, 1]
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH  # LIKELIHOOD_THRESH = 0.9
```

iii. This is the only tongue measurement in the file. Tracking is present in all 174 sessions at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.9 are excluded. Per trial, visible frames are binned into 50ms bins by averaging y-positions within each bin. A fourth class (3 = "not visible") is assigned to bins with no visible frames.

ii.
```python
LIKELIHOOD_THRESH = 0.9
tongue_vis = vdata[:, 2] > LIKELIHOOD_THRESH
# ...
for k in range(nk):
    # ...
    sel = tongue_vis[lo:hi]
    vt = vtime[lo:hi][sel] - g
    vy = tongue_y[lo:hi][sel]
    bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
    np.clip(bidx, 0, N_BINS - 1, out=bidx)
    sums = np.bincount(bidx, weights=vy, minlength=N_BINS)
    cnts = np.bincount(bidx, minlength=N_BINS)
    nz = cnts > 0
    tongue_bin_y[k, nz] = sums[nz] / cnts[nz]
```

iii. The AI documented that tongue likelihood is bimodal (~1 when visible, ~1e-4 when not), so the exact threshold is immaterial.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles are computed over all visible (non-NaN) binned tongue y-values across the kept trials. Then: values < p40 become class 0, p40 <= value <= p60 become class 1, values > p60 become class 2, and bins with no visible frame are class 3.

ii.
```python
visible = np.isfinite(tongue_bin_y)
if visible.sum() >= 10:
    p40, p60 = np.percentile(tongue_bin_y[visible], [40, 60])
tongue_class = np.full(tongue_bin_y.shape, 3, dtype=np.int64)  # 3 = not visible
if np.isfinite(p40):
    tongue_class[visible & (tongue_bin_y < p40)] = 0
    tongue_class[visible & (tongue_bin_y >= p40) & (tongue_bin_y <= p60)] = 1
    tongue_class[visible & (tongue_bin_y > p60)] = 2
```

iii. The 40th/60th percentile split and per-session scope follow the instructions. Percentiles are computed over the visible binned y-values from extracted trial windows.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as spikes and go cues. Each trial's frame range is found by `searchsorted` on camera timestamps at `go + T_START` and `go + T_END`, and frames are assigned to the same go-cue-relative bin grid used for firing rates.

ii.
```python
lo = np.searchsorted(vtime, g + T_START)
hi = np.searchsorted(vtime, g + T_END)
vt = vtime[lo:hi][sel] - g
bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
```

iii. The same bin grid guarantees bin k of the tongue output covers the same interval as bin k of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- **Session with no quality-controlled units** (1 session): `classification` values that are not 'good' result in 0 good units, and the session is skipped.
- **Trials outside obs_intervals**: Intersected across all good units; unobserved trials are dropped.
- **Trials with zero spikes from all good units**: Dropped as genuine acquisition dropouts (2,150 trials).
- **Video not covering the analysis window**: Trials with < 90% coverage dropped; sessions where the video stops at the go cue are dropped entirely.
- **Tongue not visible**: Bins with no visible frame get class 3 ("not visible").
- **Trials with no tone onset**: Filtered by `np.isfinite(tone_onset)` in the keep mask.

ii.
```python
# obs_intervals intersection
observed = np.ones(n_trials, dtype=bool)
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    # ...
    observed &= m

# empty spike trials
nonempty = neural_all.any(axis=(1, 2))
keep_trials = keep_trials[nonempty]

# video coverage
keep_mask = observed & (coverage >= MIN_VIDEO_COVERAGE) & np.isfinite(tone_onset)
```

iii. The AI documented each case in CONVERSION_NOTES.md with counts and justifications.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file and extracting spike times dominate. With 16 workers, the full conversion takes ~35 seconds. Per session: opening the file (~0.16s), reading spike times (~0.08-0.5s depending on unit count), reading video (~0.03s), binning/processing (~0.04-0.1s). Pickling the ~9.6 GB result also takes significant time.

ii. N/A (timing info from CONVERSION_NOTES.md Step 7).

iii. The AI reported timing breakdowns per session and noted the work is dominated by I/O.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain: (1) a per-unit loop in neural binning that uses searchsorted per unit across all trials, and (2) a per-trial loop for tongue y-position binning. The per-unit loop is necessary because each unit has a different-length spike array. The tongue loop iterates over trials.

ii.
```python
# per-unit loop
for ui, st in enumerate(spikes):
    lo = np.searchsorted(st, w0)
    hi = np.searchsorted(st, w1)
    # ...

# per-trial tongue loop
for k in range(nk):
    # ...
    bidx = np.floor((vt - T_START) / BIN_SIZE).astype(np.int64)
    # ...
```

iii. The AI noted the per-unit loop cannot be collapsed because spike arrays are ragged. The tongue loop could be vectorized but is not a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The obs_intervals are read and processed for EVERY good unit (looping through all good units to intersect their obs_intervals), which involves repeated `searchsorted` and array operations. Spike times are also read one-by-one per unit.

ii.
```python
# reads obs_intervals for every good unit
for i in good:
    obs = np.asarray(units['obs_intervals'][int(i)])
    oi = np.searchsorted(start_time, obs[:, 0], side='right') - 1
    # ...

# reads spike times per unit
spikes = [np.asarray(units['spike_times'][int(i)]) for i in good]
```

iii. The obs_intervals intersection is done per-unit when in most sessions all units share the same intervals. The spike times reading could potentially be done in bulk.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores several diagnostic quantities in each session's result dict that are not part of the final output format: `frac_unobserved_bins`, `tongue_p40`/`tongue_p60`, `n_photostim`, `n_dropped_video`, `n_dropped_unobserved`, `n_dropped_empty`, `n_auto_water`, `n_free_water`, `timings`, `trial_idx`, `path`, `performance`, etc. These are stored in `metadata['session_info']` but are not needed by the decoder. The video coverage computation (`coverage`) is also done for all trials before filtering.

ii.
```python
res = dict(
    # ... many diagnostic fields ...
    frac_unobserved_bins=float(n_unobserved / (len(keep_trials) * N_BINS)),
    tongue_p40=float(p40), tongue_p60=float(p60),
    n_photostim=int(np.isfinite(stim_on[keep_trials]).sum()),
    # ...
)
```

iii. These are used for validation/documentation purposes but not for the decoder itself.
