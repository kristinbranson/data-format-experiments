# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds all NWB files by globbing `sub-*/*.nwb` under the data directory. Each file is opened with `h5py` (not `pynwb`) and processed by `process_session()`. The AI performs a **two-pass approach**: first it reads every file to check session-level behavioral criteria (performance > 65%, >= 50 correct per direction), then in a second pass it fully converts only the selected sessions, optionally using multiprocessing.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# --- pass 1: session selection ---
for fp in files:
    with h5py.File(fp, 'r') as f:
        sid = f['identifier'][()].decode()
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)
    if ok:
        selected.append(fp)
# --- pass 2: full conversion ---
for i, job in enumerate(jobs):
    res = process_session(job[0], ...)
```

iii. The AI chose h5py over pynwb for direct HDF5 access, citing performance and control. The two-pass approach was chosen to first identify sessions passing the data paper's behavioral criteria before doing the expensive full conversion. The multiprocessing pool was added for efficiency.

## 1-b. How are the data split into subjects?

i. Each NWB file's `identifier` field (e.g. `SC015_20190207_120657_s1`) is split on `_` and the first element (the mouse name, e.g. `SC015`) is used as the subject identifier. This differs from the reference, which uses the numeric DANDI `subject_id` (e.g. `440956`).

ii.
```python
session_id = f['identifier'][()].decode()
subject_id = f['general/subject/subject_id'][()].decode()
mouse = session_id.split('_')[0]
# ...
result = {
    'subject': mouse,
    'subject_id': subject_id,
    ...
}
# Assembly:
subjects = sorted({r['subject'] for r in results})
```

iii. The AI uses the mouse name (from the session identifier) rather than the numeric DANDI subject_id. Both identify the same animals; the AI verified a 1:1 mapping between mouse names and subject IDs with an assertion: `assert len(pairs) == len(subjects), 'mouse name / subject id mapping is not 1:1'`.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI identifies sessions by `f['identifier']` (e.g. `SC015_20190207_120657_s1`). Sessions are sorted by file path, which gives chronological order within each subject. The AI keeps sessions that pass behavioral performance criteria (150 of 174).

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
session_id = f['identifier'][()].decode()
```

iii. The file-per-session structure is inherent to the dataset. The AI's two-pass approach first selects sessions by behavioral criteria, then processes only those.

## 1-d. How are the data split into trials?

i. Trials come from the NWB `intervals/trials` table, one row per behavioral trial. The AI verifies that the number of go cue events matches the number of trials.

ii.
```python
def read_trial_table(f):
    t = f['intervals/trials']
    tr = {
        'start_time': t['start_time'][:],
        'stop_time': t['stop_time'][:],
        'outcome': _dec(t['outcome'][:]),
        # ...
    }
    go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
    if len(go) != ntrials:
        raise ValueError('number of go cues (%d) != number of trials (%d)' % (len(go), ntrials))
```

iii. The trials table directly provides trial boundaries. The go-cue count check ensures consistency between the behavioral events and the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies **three levels** of filtering:

1. **Session-level**: Sessions must have behavioral performance > 65% on control trials (no photostim, no early lick, no auto/free water) AND at least 50 correct lick-left and 50 correct lick-right trials. This removes 23 sessions plus 1 session with no good units, leaving 150 of 174.

2. **Trial-level**: Within selected sessions, auto-water and free-water trials are excluded (citing `get_regular_trial_mask`), plus trials outside `units/obs_intervals`.

3. **Post-processing**: Trials where no neuron fires at all (sum of all firing rates = 0) are dropped.

ii.
```python
# Session filtering:
MIN_PERFORMANCE = 0.65
MIN_CORRECT_PER_DIRECTION = 50
def session_passes(f, tr):
    perf = session_performance(tr)
    ncorrect_left = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'left')).sum())
    ncorrect_right = int(((tr['outcome'] == 'hit') & (tr['instruction'] == 'right')).sum())
    # ...

# Trial filtering:
def trial_mask(f, tr):
    return ((tr['auto_water'] == 0) & (tr['free_water'] == 0) &
            observed_trial_mask(f, tr))

# Silent trial filtering:
silent = fr.sum(axis=(1, 2)) == 0
if n_silent:
    keep[kept_positions[silent]] = False
    fr = fr[~silent]
```

iii. The AI justified session filtering by matching the data paper's stated criteria ("We selected experimental sessions for analysis based on following criteria: overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."). Auto-water exclusion was justified by citing `get_regular_trial_mask` and the argument that on these trials "outcome and choice do not reflect a decision." Silent trial dropping was discovered empirically when the last trial in `obs_intervals` extended past the end of the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged sorted spike time arrays) and `units/spike_times_index` (offsets for the ragged array). Only units with `units/classification == 'good'` are included.

ii.
```python
def load_good_units(f, ontology):
    u = f['units']
    classification = _dec(u['classification'][:])
    good = np.flatnonzero(classification == 'good')
    spike_times = u['spike_times'][:]
    spike_index = u['spike_times_index'][:]
    starts = np.concatenate([[0], spike_index[:-1]]).astype(np.int64)
    ends = spike_index.astype(np.int64)
    spikes = [spike_times[starts[i]:ends[i]] for i in good]
    return spikes, region_idx, anno, ccf_xyz
```

iii. `spike_times` is the only neural representation in the NWB files. The ragged array is read once as a bulk buffer rather than per-unit to improve I/O performance.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning -2.5 s to +1.5 s relative to the go cue. Bin edges for all trials are computed as a flat array, then `np.searchsorted` gives the cumulative spike count at each edge, and differencing gives the per-bin count. Counts are divided by the bin width (0.05 s) to get firing rates in Hz.

ii.
```python
def bin_spikes(spikes, go_times):
    edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()    # (ntrials*81,)
    fr = np.empty((ntrials, len(spikes), N_BINS), dtype=np.float32)
    for i, st in enumerate(spikes):
        pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
        fr[:, i, :] = np.diff(pos, axis=1)
    fr /= BIN_SIZE
    return fr
```

iii. This is equivalent to the reference code's `sliding_histogram(..., rate=True)` which returns `binSpikes / bin_width`. No smoothing, normalization, or baseline subtraction is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == 'good'` are kept. No additional QC metric thresholds are applied. Sessions with zero good units are excluded. The AI verified that this reproduces the paper's per-area unit counts exactly (thalamus 12,808; orbital 10,223; striatum 7,664; etc.).

ii.
```python
classification = _dec(u['classification'][:])
good = np.flatnonzero(classification == 'good')
```

iii. The AI cited the spike-sorting QC white paper (ChenLiuEtAl2023) and verified the criterion by matching per-area unit counts from the data paper's Fig. 1J. The AI noted that `units/unit_quality` is an older label and was deliberately not used.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. The bin edges are defined as offsets from the go cue (-2.5 s to +1.5 s), then for each trial, these relative edges are added to the trial's absolute go cue time to get absolute bin boundaries. Spike times are already on the same session-absolute clock.

ii.
```python
go = f['acquisition/BehavioralEvents/go_start_times']['timestamps'][:]
# ...
edges = (go_times[:, None] + BIN_EDGES[None, :]).ravel()
```

iii. All timestamps in the NWB file are on the same session-absolute clock, so no resampling or offset correction is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 50 ms (0.05 s), producing 80 non-overlapping bins over the 4 s trial window (-2.5 to +1.5 s). No temporal rebinning or sliding window is applied. The bin grid is defined once and reused for all trials and sessions.

ii.
```python
OFF_START = -2.5
OFF_END = 1.5
BIN_SIZE = 0.05
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))  # 80
BIN_EDGES = OFF_START + BIN_SIZE * np.arange(N_BINS + 1)
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)
```

iii. The 50 ms bin width is mandated by the Decoder Task instructions. The reference code uses 40 ms bins with 3.4 ms stride, but the task specification overrides this.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times` (the tone onset events) and the go cue time of each trial. For trials with early licks that replay the sample epoch, the last `sample_start` before the go cue is used.

ii.
```python
sample_start = np.sort(f['acquisition/BehavioralEvents/sample_start_times']['timestamps'][:])
idx = np.searchsorted(sample_start, go, side='right') - 1
tone = np.where(idx >= 0, sample_start[np.maximum(idx, 0)], np.nan)
tone = np.where(np.isnan(tone), go - 1.85, tone)  # fallback
```

iii. Early licks replay the sample epoch, so a trial can have multiple tone onsets. The last one before the go cue is the relevant one. A fallback to `go - 1.85` (nominal sample + delay duration) is provided but was never triggered.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset at each bin center is computed as `bin_center - (tone_time - go_time)`, which equals `bin_center + (go - tone)`. This gives a continuous ramp starting at a negative value (before tone onset) and increasing through the trial.

ii.
```python
inp[:, 0, :] = (BIN_CENTERS[None, :] - (tone - go)[:, None]).astype(np.float32)
```

iii. This is algebraically equivalent to the reference's formula `CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers (aligned to the go cue) are used for both the neural data binning and the time-from-tone computation, so they are inherently aligned.

ii.
```python
BIN_CENTERS = OFF_START + BIN_SIZE * (np.arange(N_BINS) + 0.5)
# Used for both neural binning and input computation
```

iii. No separate alignment step is needed since both use the same temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` (onset relative to trial start, as a string), `intervals/trials/photostim_duration` (duration as a string), and `intervals/trials/start_time` (trial start in session time). The go cue time is used to re-express these relative to the alignment event.

ii.
```python
onset_str = tr['photostim_onset'][keep]
dur_str = tr['photostim_duration'][keep]
start = tr['start_time'][keep]
has_stim = onset_str != 'N/A'
```

iii. Photostim onset/duration are stored as strings with `'N/A'` for non-stimulated trials. The onset is relative to trial start, not the go cue, so it must be transformed.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is constructed: for each bin, 1 if the bin center falls within [stim_onset, stim_offset) relative to the go cue, 0 otherwise. Non-stimulated trials (onset == 'N/A') remain all zeros.

ii.
```python
onset[has_stim] = np.array([float(x) for x in onset_str[has_stim]])
dur[has_stim] = np.array([float(x) for x in dur_str[has_stim]])
on_rel = start + onset - go
off_rel = on_rel + dur
inside = ((BIN_CENTERS[None, :] >= on_rel[:, None]) &
          (BIN_CENTERS[None, :] < off_rel[:, None]))
inp[:, 1, :] = (inside & has_stim[:, None]).astype(np.float32)
```

iii. The extra `& has_stim[:, None]` mask ensures non-stimulated trials stay 0 even when onset/duration values default to zero. The half-open interval `[onset, offset)` is used for the bin-center test.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim onset and offset are expressed relative to the go cue, which is the same alignment event used for the neural bin edges. The bin centers are compared against the go-cue-relative onset/offset.

ii.
```python
on_rel = start + onset - go  # relative to go cue
```

iii. Same temporal reference frame as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/outcome` (`hit`/`miss`/`ignore`) and `intervals/trials/trial_instruction` (`left`/`right`). There is no explicit choice column in the NWB file.

ii.
```python
def choice_codes(tr, keep):
    outcome = tr['outcome'][keep]
    instr = tr['instruction'][keep]
    instr_code = np.where(instr == 'left', CHOICE_LEFT, CHOICE_RIGHT)
    opposite = np.where(instr == 'left', CHOICE_RIGHT, CHOICE_LEFT)
    choice = np.where(outcome == 'hit', instr_code,
                      np.where(outcome == 'miss', opposite, CHOICE_NOLICK))
    return choice.astype(np.int64)
```

iii. Hit means the animal licked the instructed side, miss means it licked the other side, ignore means no lick. The AI verified this derivation against the first lick after the go cue (agreement 1.000 in spot-checked sessions).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0 (left), 1 (right), 2 (no lick) and is a per-trial value repeated across all 80 time bins.

ii.
```python
CHOICE_LEFT, CHOICE_RIGHT, CHOICE_NOLICK = 0, 1, 2
# ...
out[:, 0, :] = choice_codes(tr, keep)[:, None]
```

iii. Per-trial values are broadcast across all bins to maintain the `(n_output, n_timepoints)` shape required by the target format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `intervals/trials/outcome`, which holds the strings `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
OUTCOME_CODE = {'ignore': 0, 'miss': 1, 'hit': 2}
# ...
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. The trials table stores outcome explicitly with exactly the three categories the instructions require.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2. The value is per-trial, repeated across all 80 time bins.

ii.
```python
out[:, 1, :] = np.array([OUTCOME_CODE[o] for o in tr['outcome'][keep]])[:, None]
```

iii. Direct mapping, no additional processing needed.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `intervals/trials/early_lick`, which holds `'early'` or `'no early'`.

ii.
```python
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```

iii. The trials table explicitly flags early licking.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early lick) or 1 (early lick) via boolean comparison. Per-trial value repeated across all 80 time bins.

ii.
```python
out[:, 2, :] = (tr['early_lick'][keep] == 'early').astype(np.int64)[:, None]
```

iii. Simple binary encoding. Equivalent to the reference's dictionary-based mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data = (tongue_x, tongue_y, tongue_likelihood) with corresponding timestamps. Column 1 (tongue_y) is the position; column 2 (likelihood) determines visibility.

ii.
```python
tt = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking']
data = tt['data']
ts = tt['timestamps'][:]
y = data[:, 1]
lik = data[:, 2]
```

iii. This is the only tongue measurement in the NWB files. The tracking runs at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Three steps:

1. Frames with `likelihood <= 0.9` are marked as not visible (the AI uses 0.9 as the DLC default cut-off, vs the reference's 0.5).
2. Per trial, visible frames within the [-2.5, 1.5] window are averaged into 50 ms bins using `np.bincount`.
3. Per-session 40th and 60th percentiles are computed over all visible bin-mean values across all trials of the session.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
# ...
vis = lik > TONGUE_LIKELIHOOD_THRESHOLD
# bin within each trial window:
b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
counts = np.bincount(b, minlength=N_BINS)
sums = np.bincount(b, weights=y[sl][v], minlength=N_BINS)
y_bin[i, nz] = sums[nz] / counts[nz]

# Percentiles over trial-window bin means:
def discretize_tongue(y_bin):
    vals = y_bin[visible]
    p40, p60 = np.percentile(vals, [TONGUE_PCTL_LOW, TONGUE_PCTL_HIGH])
    code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
    out[visible] = code[visible]
```

iii. The AI chose 0.9 as the threshold, noting the bimodal distribution makes it insensitive. The percentiles are taken over per-trial visible bin means (not the entire session's frames), which differs from the reference's session-wide binning approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Discretized into 4 classes:
- 0: y < 40th percentile
- 1: 40th percentile <= y <= 60th percentile
- 2: y > 60th percentile
- 3: not visible (no visible frame in the bin)

ii.
```python
code = np.where(y_bin < p40, 0, np.where(y_bin > p60, 2, 1))
out[visible] = code[visible]
# non-visible bins default to 3
out = np.full(y_bin.shape, 3, dtype=np.int64)
```

iii. The boundary condition at exactly p60 assigns class 1 (the reference's `np.digitize` would assign class 2), but this affects negligible numbers of data points.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session-absolute clock as the spikes and go cues. For each trial, frames within `[go + OFF_START, go + OFF_END)` are found via `searchsorted` on the camera timestamps, then assigned to bins by their offset from the trial window start using the same bin grid as the neural data.

ii.
```python
lo = np.searchsorted(ts, go_times + OFF_START)
hi = np.searchsorted(ts, go_times + OFF_END)
for i in range(ntrials):
    rel = ts[sl] - go_times[i]
    b = np.floor((rel[v] - OFF_START) / BIN_SIZE).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
```

iii. Same go-cue-relative grid as the neural data ensures bin k of the tongue output covers the same interval as bin k of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:

- **Session with no good units** (1 session): excluded by `session_passes` check for `ngood == 0`.
- **Sessions not meeting behavioral criteria** (23 sessions): excluded by performance/correct-count criteria.
- **Trials without spike coverage**: excluded via `units/obs_intervals` matching.
- **Auto-water and free-water trials**: excluded as behaviorally uninformative.
- **Silent trials** (no neuron fires): dropped post-binning (1 trial dataset-wide).
- **Tongue not visible**: bins with no visible frame get class 3 ("not visible").
- **Trial with no preceding tone onset**: falls back to `go - 1.85` (never triggered).

ii.
```python
# Session with no good units:
elif ngood == 0:
    reason = 'no units passed quality control'
# Silent trials:
silent = fr.sum(axis=(1, 2)) == 0
if n_silent:
    keep[kept_positions[silent]] = False
# Tongue fallback:
tone = np.where(np.isnan(tone), go - 1.85, tone)
```

iii. The AI documented each edge case in CONVERSION_NOTES.md and verified handling through sanity checks.

## 10-a. What are the most time-consuming steps of the code?

i. According to the AI's timing data: unit loading (reading spike times from HDF5) takes 0.06-0.6 s per session, spike binning takes 0.1-0.8 s per session, and inputs/outputs (including video reading) take 0.16-0.5 s per session. Total per session is 0.2-2 s. With 24 workers, full conversion takes ~22 s wall clock for 150 sessions. Pickling the output takes additional time for the ~10 GB file.

ii. N/A (timing is diagnostic output, not in the conversion logic itself)

iii. The AI measured and documented timing at each step and determined that the total was well within the 15-minute budget without needing further optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops remain:
1. The per-unit loop in `bin_spikes` that runs one `np.searchsorted` per unit (all trials vectorized).
2. The per-trial loop in `bin_tongue` that processes one trial's frames at a time.

ii.
```python
# Per-unit loop (trials vectorized):
for i, st in enumerate(spikes):
    pos = np.searchsorted(st, edges).reshape(ntrials, N_BINS + 1)
    fr[:, i, :] = np.diff(pos, axis=1)

# Per-trial tongue loop:
for i in range(ntrials):
    sl = slice(lo[i], hi[i])
    # ... bincount per trial
```

iii. The per-unit loop cannot be eliminated because each unit has a different number of spikes (ragged storage). The per-trial tongue loop could theoretically be vectorized with global bin indexing but is not a measurable fraction of runtime.

## 10-c. What processing does the code repeat multiple times?

i. The AI's two-pass approach reads the trials table from each NWB file twice: once during session selection (pass 1) and once during full conversion (pass 2). This doubles the I/O for the trials table but is a small fraction of total time.

ii.
```python
# Pass 1:
for fp in files:
    with h5py.File(fp, 'r') as f:
        tr = read_trial_table(f)
        ok, perf, ncl, ncr, reason = session_passes(f, tr)

# Pass 2:
def process_session(filepath, ...):
    with h5py.File(filepath, 'r') as f:
        tr = read_trial_table(f)
        # ... full processing
```

iii. The two-pass design avoids loading expensive spike/video data for sessions that will be excluded, but repeats the cheap trials-table read.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are performed that may not be needed downstream:

1. **Session performance calculation** (`session_performance`) and the session selection criteria — these filter 24 sessions that the reference solution keeps.
2. **Allen CCF ontology loading and structure-graph traversal** for brain region assignment — the reference uses simple string parsing instead.
3. **CCF coordinate extraction** from the electrode table for the ALM carve-out.
4. **Extensive diagnostic metadata** (timing info, excluded session details, tongue percentile values per session).

ii.
```python
# Allen ontology loading (unnecessary if using simple region labels):
def load_ontology(path=ONTOLOGY_PATH):
    root = json.load(open(path))['msg'][0]
    # ... recursive tree walk

# Session performance (unnecessary if not filtering):
def session_performance(tr):
    control = ((tr['early_lick'] == 'no early') & ...)
    return float((tr['outcome'][control] == 'hit').sum() / responded.sum())
```

iii. The brain region assignment using the Allen CCF ontology is more complex than needed but produces coarser (14 vs ~267) region labels that match the data paper's Fig. 1J grouping. The session filtering reduces the dataset from 173 to 150 sessions, discarding usable data.
