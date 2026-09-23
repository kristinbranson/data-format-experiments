# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files under `data/sub-*/` and processes each file with `read_session()` using `h5py` (not `pynwb`). Each file is read once, extracting trials, behavioral events, tongue tracking, unit spike times, and metadata into a plain-numpy dictionary. Sessions are processed in parallel with `ProcessPoolExecutor`.

ii.
```python
files = sorted(glob.glob(os.path.join(DATA_DIR, 'sub-*', '*.nwb')))
# ...
with ProcessPoolExecutor(max_workers=nworkers) as ex:
    for k, (res, info) in enumerate(ex.map(_worker, jobs)):
```

```python
def read_session(path):
    with h5py.File(path, 'r') as f:
        trials = f['intervals/trials']
        ev = f['acquisition/BehavioralEvents']
        d = {}
        d['subject'] = f['general/subject/subject_id'][()].decode()
        # ... reads all relevant fields
```

iii. The AI chose h5py over pynwb for speed, since all needed fields can be accessed by their HDF5 paths directly. The CONVERSION_NOTES document that 174 NWB files were found across 28 subjects, matching the dandiset.

## 1-b. How are the data split into subjects?

i. Each NWB file's `general/subject/subject_id` is read as the subject identifier (a numeric string like `'440956'`). At assembly, a sorted unique set of subjects is built and each session gets an index into that list.

ii.
```python
d['subject'] = f['general/subject/subject_id'][()].decode()
# ...
subjects = sorted({r['subject'] for r in results})
sub_index = {s: i for i, s in enumerate(subjects)}
'subject_idx': np.array([sub_index[r['subject']] for r in results], dtype=np.int64),
```

iii. The subject_id is the canonical animal identifier in the NWB file. This yields 28 subjects, matching the papers.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. The AI identifies sessions by a derived session ID (`<subject>_<datetime>`). Additionally, the AI applies **session-level curation** based on the data paper's STAR Methods: sessions must have behavioral performance > 65% on control non-early-lick trials, and at least 50 correct lick-left and 50 correct lick-right trials. A session with degenerate tongue tracking (>50% frames visible) is also rejected. This reduces 174 sessions to 136.

ii.
```python
def session_performance(d, sel=slice(None)):
    ps = d['photostim_power'][sel] != 'N/A'
    ctrl = (~ps) & (d['early_lick'][sel] == 'no early')
    hit = d['outcome'][sel] == 'hit'
    miss = d['outcome'][sel] == 'miss'
    m = ctrl & (hit | miss)
    perf = hit[m].sum() / m.sum() if m.sum() > 0 else 0.0
    # ...

if perf <= MIN_PERFORMANCE:
    info['rejected'] = 'behavioural performance'
    return None, info
if n_left < MIN_CORRECT_PER_DIRECTION or n_right < MIN_CORRECT_PER_DIRECTION:
    info['rejected'] = 'too few correct lick-left / lick-right trials'
    return None, info
if vis_frac > MAX_TONGUE_VISIBLE_FRACTION:
    info['rejected'] = 'tongue tracking failure'
    return None, info
```

iii. The AI justifies session curation by citing the data paper's explicit session-selection criteria and noting it reproduces the reported 84% mean correct rate (83.5% after curation). 25 sessions rejected for low performance, 12 for too few correct trials, 1 for tongue tracking failure.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The AI first restricts to trials covered by the electrophysiology recording using `units/obs_intervals`, then applies further trial-level filters.

ii.
```python
oi_index = u['obs_intervals_index'][:]
n_int = np.diff(np.concatenate([[0], oi_index]))
assert n_int.min() == n_int.max(), 'units disagree on observed trials'
obs = u['obs_intervals'][0:oi_index[0]]
ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
assert np.allclose(trial_start_all[ephys_trials], obs[:, 0])
```

iii. The obs_intervals restriction handles sessions where the electrophysiology recording doesn't cover all behavioral trials. The AI verifies that all units agree on which trials were observed and that obs_intervals start times match trial start times.

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level filters are applied:
1. **obs_intervals**: Trials not covered by the electrophysiology recording are excluded (handled at read time by restricting to `ephys_trials`).
2. **auto_water and free_water**: Both are excluded because reward is not contingent on the animal's choice, so `outcome` is not a behavioral report.
3. **Video coverage**: Trials where the video covers less than 90% of the [-2.5, +1.5]s window are dropped (254 trials across the dataset).

Photostimulation, early-lick, and no-response (ignore) trials are kept because they are required decoder inputs/outputs.

ii.
```python
keep = (~d['auto_water']) & (~d['free_water'])
info['n_dropped_water'] = int((~keep).sum())
cov = video_coverage(d, win_start)
keep &= cov >= MIN_VIDEO_COVERAGE
info['n_dropped_video'] = int(((cov < MIN_VIDEO_COVERAGE)).sum())
trial_idx = np.where(keep)[0]
```

iii. The AI cites the reference code's `get_regular_trial_mask` for auto_water/free_water exclusion and `get_bad_trial_inds` for video coverage filtering. Final trial count: 70,949 across 136 sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays) for units passing quality control. Go-cue times from `BehavioralEvents/go_start_times` define the bin edges.

ii.
```python
st_index = u['spike_times_index'][:]
starts = np.concatenate([[0], st_index[:-1]])
spike_times = u['spike_times'][:]
unit_spikes = [spike_times[starts[i]:st_index[i]] for i in idx]
```

iii. `spike_times` is the only neural representation in the NWB files. The AI reads all spike times at once and slices per unit, rather than reading per-unit.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50ms bins spanning [-2.5, +1.5)s relative to the go cue. Spike counts per bin are divided by the bin width (0.05s) to yield firing rates in Hz. No smoothing, normalization, or baseline subtraction is applied.

The AI uses a vectorized single-pass approach: all spikes from all units are concatenated, their trial index is found by one `np.searchsorted` over the window-start array, their bin index by integer division, and the full (unit, trial, bin) tensor is built with one `np.bincount`.

ii.
```python
def bin_spikes(unit_spikes, win_start):
    t = np.concatenate(unit_spikes)
    uidx = np.repeat(np.arange(n_units), lens)
    ti = np.searchsorted(win_start, t, side='right') - 1
    # ... validity checks ...
    b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
    np.clip(b, 0, N_BINS - 1, out=b)
    flat = (uidx * n_trials + ti) * N_BINS + b
    counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)
    return (counts / BIN_WIDTH).astype(np.float32)
```

iii. Firing rate in Hz matches the reference code's `sliding_histogram(..., rate=True)` which returns `binSpikes / bin_width`. The vectorized bincount approach is O(n_spikes) instead of per-unit per-trial searchsorted.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three unit-level filters:
1. `units/classification == 'good'` (the QC classifier verdict from the spike-sorting white paper).
2. The unit's CCF annotation (`anno_name`) must map to one of 14 major brain regions via a `region_map.py` module.
3. After computing firing rates, units that are completely silent across all extracted windows are dropped.

ii.
```python
classification = _decode(u['classification'][:])
anno = _decode(u['anno_name'][:])
good = classification == 'good'
regions = np.array([annotation_to_region(a) or '' for a in anno])
keep_unit = good & (regions != '')
# ...
# after binning:
active = fr.any(axis=(1, 2))
fr = fr[active]
```

iii. The AI's CONVERSION_NOTES state that all 293 distinct annotations of good units map successfully, so the CCF filter does not actually remove any units in practice. The activity filter matches the intent of the reference code's `check_fr` (drops neurons with zero across-trial variance). The AI reports 54,629 good units across 136 kept sessions, median 392 per session vs the paper's 393.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go-cue time from `BehavioralEvents/go_start_times` serves as time zero. Bin edges are computed as `go_time + T_START` to `go_time + T_STOP` in absolute session time. All NWB timestamps share a single session-absolute clock, so no interpolation or offset correction is needed.

ii.
```python
go = d['go_time']
win_start = go + T_START
# in bin_spikes:
ti = np.searchsorted(win_start, t, side='right') - 1
b = ((t - win_start[ti]) / BIN_WIDTH).astype(np.int64)
```

iii. The reference code also aligns everything to the go cue. The AI verifies that trial windows never overlap (go cues are >= 4.58s apart).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins spanning [-2.5, +1.5)s relative to the go cue. This matches the task specification. No rebinning is applied since spikes are binned directly from raw spike times.

ii.
```python
BIN_WIDTH = 0.05
T_START = -2.5
T_STOP = 1.5
N_BINS = int(round((T_STOP - T_START) / BIN_WIDTH))   # 80
BIN_EDGES = T_START + BIN_WIDTH * np.arange(N_BINS + 1)
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
```

iii. The reference paper used 40ms bins with 3.4ms stride, but the task specifies 50ms bins. The AI notes this as a required deviation.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `BehavioralEvents/sample_start_times` (tone onset timestamps) and the go-cue time. The tone for a trial is the **last** sample-epoch onset at or before the go cue.

ii.
```python
def tone_onset_rel(d):
    si = np.searchsorted(d['sample_start'], d['go_time'], side='right') - 1
    assert (si >= 0).all(), 'a trial has no preceding sample epoch'
    return d['sample_start'][si] - d['go_time']
```

iii. Early-lick trials replay the sample epoch, so there can be multiple sample onsets per trial. The last one before the go cue is the relevant one.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the value at each bin center is: `bin_center_relative_to_go - tone_onset_relative_to_go`. Since `tone_onset_rel` returns a negative value (tone is before go), `bin_center - tone_rel` gives seconds since the tone.

ii.
```python
tone_rel = tone_onset_rel(d)[trial_idx]
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. This is a continuous, time-varying input. For standard trials the tone is at -1.85s relative to the go cue, so the value at the go-cue bin is 1.875s (bin center at 0.025s).

## 3-c. How is `input` *Time from tone onset in seconds* aligned with the neural data?

i. The `BIN_CENTERS` array is defined relative to the go cue, the same event the neural data is aligned to. So the time-from-tone value at bin k corresponds to the same 50ms interval as the neural data at bin k.

ii.
```python
BIN_CENTERS = BIN_EDGES[:-1] + BIN_WIDTH / 2
time_from_tone = BIN_CENTERS[None, :] - tone_rel[:, None]
```

iii. Both neural and input share the same bin grid, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `BehavioralEvents/photostim_start_times` and `photostim_stop_times` (the actual recorded laser on/off timestamps in session time). These are mapped to trials by `np.searchsorted` against trial start times, then expressed relative to the go cue.

ii.
```python
d['stim_on'] = ev['photostim_start_times']['timestamps'][:]
d['stim_off'] = ev['photostim_stop_times']['timestamps'][:]
# ...
def photostim_windows(d):
    ti = np.searchsorted(d['trial_start'], d['stim_on'], side='right') - 1
    on[ti] = s_on - d['go_time'][ti]
    off[ti] = s_off - d['go_time'][ti]
    return on, off
```

iii. The AI uses the event-level timestamps rather than the trials table fields (`photostim_onset`/`photostim_duration`). The AI documents that this avoids issues with the laser-off timestamp overshooting the go cue by ~1ms.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1.0 where the bin center falls within the [on, off] interval, 0.0 elsewhere. The AI uses `>=` for the onset and `<=` for the offset (closed interval).

ii.
```python
o = np.where(has_stim, on, np.inf)[:, None]
f_ = np.where(has_stim, off, -np.inf)[:, None]
stim = ((BIN_CENTERS[None, :] >= o) & (BIN_CENTERS[None, :] <= f_)).astype(np.float32)
```

iii. Non-stimulated trials get inf/neg-inf bounds so the condition is always False. The AI notes each 0.5s stimulation covers exactly 10 bins (matching the data paper).

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim on/off times are expressed relative to the go cue (same as the bin centers), so the binary trace is automatically aligned with the neural bins.

ii.
```python
on[ti] = s_on - d['go_time'][ti]
off[ti] = s_off - d['go_time'][ti]
# compared against BIN_CENTERS which are go-cue-relative
```

iii. Same alignment mechanism as all other streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from two trials-table columns: `outcome` (hit/miss/ignore) and `trial_instruction` (left/right). Choice is not stored directly in the NWB file.

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

iii. Hit means the animal licked the instructed side; miss means the opposite side; ignore means no lick. The AI cross-validated this derivation against actual lick timestamps (99.92% agreement).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Mapped to integers: 0=left, 1=right, 2=no lick. The per-trial value is broadcast across all 80 time bins.

ii.
```python
outputs = [np.stack([np.full(N_BINS, choice[i]), np.full(N_BINS, outcome[i]),
                     np.full(N_BINS, early[i]), tongue[i]]).astype(np.int64)
           for i in range(n_tr)]
```

iii. Broadcasting ensures all four outputs share a single (4, 80) array per trial.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings `'ignore'`, `'miss'`, `'hit'`.

ii.
```python
d['outcome'] = _decode(trials['outcome'][:])[et]
# ...
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_map[o] for o in d['outcome']])[trial_idx]
```

iii. No derivation needed; the trials table stores the outcome directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: 0=ignore, 1=miss, 2=hit. Broadcast across all 80 bins (per-trial value).

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome = np.array([outcome_map[o] for o in d['outcome']])[trial_idx]
# then broadcast via np.full(N_BINS, outcome[i])
```

iii. Straightforward categorical encoding matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains `'early'` or `'no early'`.

ii.
```python
d['early_lick'] = _decode(trials['early_lick'][:])[et]
# ...
early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]
```

iii. The trials table stores this flag directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: 0=no, 1=yes. Broadcast across all 80 bins.

ii.
```python
early = (d['early_lick'] == 'early').astype(np.int64)[trial_idx]
# then broadcast via np.full(N_BINS, early[i])
```

iii. Simple binary encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(n_frames, 3)` data: tongue_x, tongue_y, tongue_likelihood, plus corresponding timestamps.

ii.
```python
tk = bts['Camera0_side_TongueTracking']
tongue = tk['data'][:]           # (n_frames, 3): x, y, likelihood
d['tongue_t'] = tk['timestamps'][:]
d['tongue_y'] = tongue[:, 1]
d['tongue_lik'] = tongue[:, 2]
```

iii. This is the only tongue measurement in the NWB files. Video runs at ~300 Hz (dt=3.4ms).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing steps:
1. Frames with likelihood <= 0.5 are excluded (tongue not visible).
2. Visible frames within each trial's window are assigned to 50ms bins by `np.searchsorted` and integer division, then averaged per bin.
3. The 40th and 60th percentiles of all visible bin-means **across the kept trial windows** (not the whole session) serve as class boundaries.
4. Each bin is classified: 0 (< 40th pct), 1 (40th-60th pct), 2 (> 60th pct), or 3 (not visible, no visible frame in the bin).

ii.
```python
def tongue_bin_position(d, win_start):
    visible = d['tongue_lik'] > TONGUE_LIKELIHOOD_THRESH
    vt = d['tongue_t'][visible]
    vy = d['tongue_y'][visible]
    # ... assign to (trial, bin) using searchsorted + integer division ...
    ybar[has] = ysum[has] / nvis[has]
    return ybar, has

def tongue_classes(d, win_start):
    ybar, has = tongue_bin_position(d, win_start)
    out = np.full(ybar.shape, 3, dtype=np.int64)
    if has.sum() < 2:
        return out, np.nan, np.nan
    p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
    cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
    out[has] = cls[has]
    return out, float(p_lo), float(p_hi)
```

iii. The AI initially computed percentiles over the whole session but found it gave an unbalanced class split (44/31/24). They switched to computing percentiles over the per-bin means of the kept trial windows, yielding a 40/20/40 split. The likelihood threshold of 0.5 is justified by the strongly bimodal distribution (median ~6e-5 for occluded, ~1.0 for visible).

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th/60th percentiles of visible bin means (from kept trial windows) define two thresholds. Values below the 40th percentile are class 0, between 40th and 60th are class 1, above 60th are class 2. Bins with no visible frame are class 3.

ii.
```python
p_lo, p_hi = np.percentile(ybar[has], [TONGUE_LOW_PCT, TONGUE_HIGH_PCT])
cls = np.where(ybar < p_lo, 0, np.where(ybar > p_hi, 2, 1))
out[has] = cls[has]
# default is 3 (not visible)
```

iii. The boundary conditions: class 0 for strictly < p_lo, class 2 for strictly > p_hi, class 1 for values in [p_lo, p_hi].

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps share the session-absolute clock. The trial's window start (`go + T_START`) and end (`go + T_STOP`) define which frames belong to each trial, and their offset from the window start determines the bin assignment. This is the same go-cue-relative grid used for neural data.

ii.
```python
ti = np.searchsorted(win_start, vt, side='right') - 1
ok = (ti >= 0) & (ti < n_trials)
# ...
ok &= vt < win_end[ti_c]
b = ((vt - win_start[ti]) / BIN_WIDTH).astype(np.int64)
np.clip(b, 0, N_BINS - 1, out=b)
```

iii. Same alignment mechanism as the neural data (go-cue-relative bin grid), ensuring bin k of the tongue output covers the same interval as bin k of the firing rates.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several data issues are handled:
- **Ephys not covering all behavioral trials** (9 of 174 sessions): Only trials within `obs_intervals` are processed.
- **Auto-water and free-water trials**: Excluded (reward not contingent on choice).
- **Missing video coverage**: Trials with <90% of expected frames in the window are dropped.
- **Tongue not visible**: Bins with no visible frame get class 3 ("not visible").
- **Spikes only within [trial.start, trial.stop]**: For error trials, the trial ends early (~0.2-0.5s after go cue), so late bins necessarily have 0 Hz. Documented as a known limitation.
- **Photostim timestamp jitter**: Laser-off timestamps overshoot go cue by ~1ms; handled by bin-center sampling.
- **Degenerate tongue tracking**: One session rejected where >50% of frames show tongue as visible.
- **Sample epoch replays**: Last tone onset before go cue is used.

ii.
```python
# obs_intervals handling
ephys_trials = np.searchsorted(trial_start_all, obs[:, 0])
# video coverage
cov = video_coverage(d, win_start)
keep &= cov >= MIN_VIDEO_COVERAGE
# degenerate tongue tracking
if vis_frac > MAX_TONGUE_VISIBLE_FRACTION:
    info['rejected'] = 'tongue tracking failure'
    return None, info
```

iii. The AI documents each issue in CONVERSION_NOTES.md and implements specific handling for each case, with assertions and verification.

## 10-a. What are the most time-consuming steps of the code?

i. Reading each NWB file via h5py is the main cost (~0.45s CPU per session). Spike binning takes ~0.50s CPU per session. However, with 24 parallel workers the wall-clock time is only ~0.11s per session, totaling 19s for all 174 files plus 17s to write the 9.45 GB pickle.

ii. N/A (timing is reported in the conversion output, not in specific code)

iii. The AI documents detailed timing breakdowns in CONVERSION_NOTES.md Step 7 and notes that total conversion takes ~36s wall-clock, well within the 15-minute budget.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI's code is highly vectorized. The spike binning uses a single `np.bincount` over all spikes from all units in the session (no per-unit or per-trial loops). The tongue binning similarly uses vectorized operations. No significant loops remain that could be further vectorized.

ii.
```python
# Spike binning - fully vectorized
t = np.concatenate(unit_spikes)
uidx = np.repeat(np.arange(n_units), lens)
flat = (uidx * n_trials + ti) * N_BINS + b
counts = np.bincount(flat, minlength=n_units * n_trials * N_BINS)
```

iii. The AI explicitly designed this approach to avoid the naive per-(unit, trial) loop, achieving ~100x speedup on the binning step.

## 10-c. What processing does the code repeat multiple times?

i. No significant repeated processing. Each NWB file is read once; each quantity is computed once. The bin grid is defined once at module level. Session-level tongue percentiles are computed in the same pass as the per-trial binning.

ii. N/A

iii. Single-pass design avoids redundant computation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce information not needed by the decoder:
1. **Session-level performance metrics** and selection criteria (behavioral performance, correct trial counts) are computed but only used for session rejection.
2. **Video coverage** per trial is computed for trial filtering.
3. **Hemisphere determination** from CCF coordinates and probe metadata is used to prefix brain region names (e.g., "left ALM"), which is more granular than strictly necessary.
4. **Per-session JSON info** files with detailed provenance are written alongside the pickle.
5. The `region_map.py` module maps CCF annotations to 14 major regions, which adds complexity beyond simple annotation parsing.

ii.
```python
# Session curation (not needed for decoder)
perf, n_left, n_right = session_performance(d, trial_idx)
# Hemisphere determination
hemi[j] = 'left' if x >= ML_MIDLINE else 'right'
d['unit_region'] = np.array([f'{h} {r}' for h, r in zip(hemi, regions[idx])])
```

iii. These are overhead from the AI's more conservative data curation approach, which follows the data paper more closely but adds complexity and processing time.
