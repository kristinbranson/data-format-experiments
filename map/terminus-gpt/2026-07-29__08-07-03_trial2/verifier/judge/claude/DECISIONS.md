# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data directly from NWB (Neurodata Without Borders) files stored in the `data/` directory. It uses `h5py` to read the HDF5-based NWB files. All `.nwb` files are discovered recursively with `Path('data').rglob('*.nwb')`, sorted alphabetically, and processed sequentially. Each NWB file is treated as one session. The reference code, by contrast, loads from preprocessed `.mat` files exported from DataJoint, not directly from NWB.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
# ...
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```
and within `extract_session`:
```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    # ...
    units = f['units']
```

iii. The AI justified this by noting that the reference code operates on preprocessed arrays and external paths not available in the working environment, so NWB files were the source of truth. This is documented in CONVERSION_NOTES.md Step 4: "Use NWB as source of truth; use code/papers to guide curation and alignment."

## 1-b. How are the data split into subjects (mice)?

i. Subject ID is extracted from each NWB file's `general/subject/subject_id` field. If that field is missing, the parent directory name is used as fallback. A `subject_to_idx` dictionary maps unique subject IDs to indices, building the `subjects` list and `subject_idx` array.

ii.
```python
def get_subject_id(f, path):
    try:
        sid = f['general/subject/subject_id'][()]
        return _decode(sid)
    except Exception:
        return path.parent.name
```
```python
sid = info['subject']
if sid not in subject_to_idx:
    subject_to_idx[sid] = len(subjects)
    subjects.append(sid)
subject_idx.append(subject_to_idx[sid])
```

iii. The AI noted that the NWB files are organized in subject-specific directories (`data/sub-<id>/`), so subject IDs could be reliably extracted from either the metadata or directory structure.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes all 174 NWB files found in the data directory. The reference paper reports 173 behavioral sessions; the AI acknowledged this 174 vs 173 discrepancy in CONVERSION_NOTES but did not resolve it (no session-level filtering was applied).

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    paths = paths[:2]
# Each path is one session:
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

iii. The AI noted the mismatch with the paper (174 vs 173 sessions) in CONVERSION_NOTES Step 9 but did not investigate which session should be excluded. The reference paper states sessions were selected based on "overall behavioral performance (> 65%), and at least 50 correct lick left and lick right trials each."

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` group within each NWB file. All trial metadata (start_time, trial_instruction, outcome, early_lick, photostim_onset, photostim_duration) are read from this group. The number of trials is determined by `len(trials['id'])`.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. The AI determined from NWB exploration that all trial metadata needed for the decoder task was available in the `intervals/trials` group.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three trial filters:
1. **Label validity**: trials must have `trial_instruction` in ('left','right'), `outcome` in ('ignore','miss','hit'), and `early_lick` in ('no early','early').
2. **Neural coverage**: the go-cue-aligned window [-2.5s, +1.5s] must fall within the recorded spike time range.
3. **Non-zero neural**: trials where all binned spike counts are zero are skipped.

Notably **missing** filters compared to the reference code's `get_regular_trial_mask()`:
- No exclusion of early lick trials (reference excludes them)
- No exclusion of auto-water trials
- No exclusion of free-water trials
- No exclusion of photostimulation/laser trials (reference excludes them)
- No session-level performance threshold (>65%)
- No minimum trial count requirement (>=50 correct each direction)
- No exclusion of first 10 "warm-up" trials

ii.
```python
# Label validity filter
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)

# Neural coverage filter
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok

# All-zero neural skip
if np.all(fr == 0):
    continue
```

iii. The AI justified keeping early lick and photostim trials because the decoder task explicitly requests early_lick as an output and photostimulation as an input. The neural coverage filter was added after discovering that in some sessions, behavioral timestamps extend far beyond recorded spike times. The all-zero filter was added after finding ~3,819 such trials in the full dataset.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` in each NWB file, filtered by `units/unit_quality`.

ii.
```python
units = f['units']
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The AI identified `units/spike_times` as the primary neural data source from NWB exploration.

## 2-b. How is the `neural` data processed?

i. Spike times for each good unit are histogrammed into 50ms bins spanning [-2.5s, +1.5s] relative to the derived go cue time. Spike counts are divided by bin width (0.05s) to convert to firing rates in Hz. The result is a (n_neurons, 80) matrix per trial.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))  # = 80
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)

# Per trial:
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The AI used 50ms bins as specified by the decoder task instructions, noting that the reference code uses 40ms bins with 3.4ms stride for their analyses, but the decoder task explicitly requires 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons using `unit_quality == 'good'` from the NWB `units` table. This yields 154,948 "good" units across 174 sessions. The paper reports 69,943 good units from 173 sessions using a more stringent classifier-based QC procedure (region-specific logistic regression classifiers trained on manual curation). The AI acknowledged this 2.2x discrepancy but did not resolve it.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
```

iii. From CONVERSION_NOTES: "unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper." The AI noted the `unit_quality` field in NWB likely uses different (less stringent) criteria than the classifier-based QC described in the paper and the `ChenLiuEtAl2023_SpikeSortingQC.pdf` white paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Go cue onset is derived as `trial_start_time + 1.85s`, based on fixed task timing from methods.txt (sample epoch 0.65s + delay epoch 1.2s = 1.85s). The neural window is [-2.5s, +1.5s] relative to this derived go cue. The reference code's spike times are already aligned to go cue = 0 in the raw .mat files; the NWB files store absolute timestamps requiring this derivation.

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85
TONE_ONSET_FROM_TRIAL_START = 0.0

trial_start = trials['start_time'][:].astype(float)
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
# ...
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The AI noted that explicit go-cue timestamp columns were absent from the inspected NWB trial tables, so timing was derived from the documented task structure. CONVERSION_NOTES Step 4: "If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (BIN_SIZE = 0.05s), yielding 80 time bins per trial over the 4s window. No rebinning is applied — firing rates are computed directly from spike times into the final 50ms bins.

ii.
```python
BIN_SIZE = 0.05
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))  # 80
```

iii. The 50ms bin size is specified directly by the decoder task instructions. The reference code uses 40ms bins with 3.4ms stride, but the decoder task overrides this.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `intervals/trials/start_time` (trial start time) with the assumption that tone onset coincides with trial start (`TONE_ONSET_FROM_TRIAL_START = 0.0`).

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The AI determined from methods.txt that the sample epoch (tone onset) occurs at the start of the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the time from tone onset is computed as the difference between each bin center (in absolute time) and the tone onset time. This produces a continuous, monotonically increasing time series per trial, ranging from approximately -0.625s to 3.325s (since tone onset is 1.85s before go cue, and the window is [-2.5s, +1.5s] relative to go cue).

ii.
```python
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The verification output confirms input range [-0.6, 3.3] which is consistent with this computation (bin centers at -2.475s to +1.475s relative to go cue, plus 1.85s offset to tone onset).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone-onset input is computed at the same bin centers as the neural data, ensuring perfect temporal alignment. Both use `centers_abs = align + BIN_CENTERS`.

ii.
```python
centers_abs = align + centers_rel  # same centers used for neural histogramming
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. No separate alignment step is needed since both are defined on the same time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` (onset time relative to trial start) and `intervals/trials/photostim_duration` (duration in seconds) from the NWB trial table.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. The AI identified these fields during NWB exploration in Step 2.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series is constructed. For each trial, if both photostim_onset and photostim_duration are valid (not None/N/A), time bins whose centers fall within [trial_start + onset, trial_start + onset + duration) are set to 1.0; all others are 0.0. For trials without photostimulation, the entire array is zeros.

ii.
```python
photo = np.zeros(N_BINS, dtype=np.float32)
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
inp = np.stack([time_from_tone, photo], axis=0)
```

iii. The AI treated photostimulation as a binary on/off indicator, consistent with the decoder task instructions requiring a discrete, time-varying input.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary signal is computed at the same bin centers as the neural data (`centers_abs`), ensuring temporal alignment.

ii.
```python
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Same time grid as neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI uses `intervals/trials/trial_instruction` from the NWB file. This field contains the **instructed** lick direction (the direction the mouse was cued to lick), NOT the animal's actual lick choice. On miss trials (where the mouse licks the wrong port), `trial_instruction` gives the opposite of what the mouse actually did.

ii.
```python
trial_instruction = np.array([_decode(x) for x in trials['trial_instruction'][:]])
# ...
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The AI used `trial_instruction` as the source for "choice" without discussing the distinction between instructed direction and actual behavioral choice. The CONVERSION_NOTES do not address this mapping decision explicitly.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The string values 'left' and 'right' are mapped to integers 0 and 1, respectively. The value is per-trial (constant across all 80 time bins).

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The mapping follows the decoder task specification (left=0, right=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` in the NWB file. Values are 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. The AI identified the outcome field during NWB exploration.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values mapped to integers: 'ignore'→0, 'miss'→1, 'hit'→2. Per-trial value repeated across all time bins.

ii.
```python
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. Mapping follows the decoder task specification.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This variable does not exist in this dataset. The decoder task defines Outcome (not "Distance to reward zone") as a per-trial categorical output. It is simply broadcast to all time bins.

ii. N/A — no distance-to-reward-zone variable. Outcome is per-trial:
```python
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. N/A for this dataset.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` in the NWB file. Values are 'no early' and 'early'.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
```

iii. The AI identified this field during NWB exploration.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values mapped to integers: 'no early'→0, 'early'→1. Per-trial value repeated across all time bins.

ii.
```python
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. Mapping follows the decoder task specification (no=0, yes=1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` in the NWB file. The `data` array (assumed shape [n_frames, 3] for [x, y, likelihood]) provides the y-coordinate (column index 1) and the `timestamps` array provides frame times.

ii.
```python
tongue_group = choose_tongue_group(f)
# ...
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
    tongue_y = tongue_data[:, 1].astype(float)
    if tongue_data.shape[1] >= 3:
        lik = tongue_data[:, 2].astype(float)
        tongue_y[lik < 0.5] = np.nan
```

iii. The AI selected Camera0 (side view) as the preferred tongue tracking source, consistent with the paper's mention of side-view camera tracking.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The y-coordinate (column 1) is extracted. Low-confidence frames (likelihood column < 0.5) are masked with NaN. No further smoothing or filtering is applied to the raw tracking data.

ii.
```python
tongue_y = tongue_data[:, 1].astype(float)
if tongue_data.shape[1] >= 3:
    lik = tongue_data[:, 2].astype(float)
    tongue_y = tongue_y.copy()
    tongue_y[lik < 0.5] = np.nan
```

iii. The AI used a likelihood threshold of 0.5 to remove unreliable tracking frames, which is a common practice with DeepLabCut outputs.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles of ALL valid (non-NaN) tongue y-values across the entire session (not just trial-aligned periods) are computed. Values below the 40th percentile → 0, between 40th-60th → 1, above 60th → 2. Missing/unavailable tongue data defaults to class 1 (middle).

ii.
```python
# Session-wide percentiles (from ALL tongue data, not just trial periods)
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])

# Per time bin discretization:
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)  # default middle
# ...
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. The decoder task specifies these exact percentile thresholds. The resulting class distribution is severely imbalanced: lt_40pct=8.0%, 40_to_60pct=90.1%, gt_60pct=1.9%. This is because tongue tracking has many NaN/missing frames during the trial window, and the default for missing data is class 1. The percentiles are computed from all session data (including inter-trial periods when the tongue may be more active), but during trial-aligned windows much of the tongue data is missing or far from any valid sample.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Nearest-neighbor interpolation via `np.searchsorted` maps each neural time bin center to the closest valid tongue tracking sample. If the nearest valid sample is more than 0.1s away, the value is set to NaN and defaults to class 1.

ii.
```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
vt = tongue_t[valid_idx]
vy = tongue_y[valid_idx]
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y[far] = np.nan
```

iii. The 0.1s threshold is somewhat arbitrary but reasonable given the 300 Hz camera frame rate (~3.3ms per frame) and 50ms bin width.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Missing photostim onset/duration**: `parse_optional_time()` converts 'N/A' or invalid values to None; trials without photostim get all-zero photostim input.
- **Trials outside neural coverage**: Excluded via `window_ok` check comparing go-cue window against spike time range.
- **All-zero neural trials**: Explicitly skipped after binning.
- **Missing tongue data**: Default to class 1 (middle percentile category).
- **Missing brain region metadata**: Initially used 'unknown' fallback; later fixed by deriving regions from electrode location metadata.
- **Sessions with <2 valid trials or 0 good units**: Skipped entirely.
- **Byte-string encoding**: `_decode()` helper handles both bytes and string values from HDF5.

ii.
```python
def parse_optional_time(x):
    x = _decode(x)
    if isinstance(x, str) and x == 'N/A':
        return None
    try:
        return float(x)
    except Exception:
        return None

# Session skip
if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
    print(f'Skipping {path.name}: insufficient kept trials or no good units')
    continue
```

iii. The AI documented edge case handling in CONVERSION_NOTES, particularly the neural coverage issue where behavioral timestamps extended beyond spike recording periods in some sessions.

## 10-a. What are the most time-consuming steps of the code?

i. The per-neuron spike histogramming loop is the main bottleneck, iterating over each neuron individually for each trial. The AI reported ~17.7s per session for the first few sessions, projecting >50 minutes for the full 174-session dataset.

ii.
```python
# This loop runs for each trial, for each neuron:
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. From CONVERSION_NOTES Step 6: "Per-neuron histogram loop is a bottleneck on full dataset; observed ~17.7 s/session over first 3 sessions."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron histogram loop (lines 175-179) iterates over each neuron individually with `np.histogram`. This could potentially be vectorized by concatenating all spike times with neuron indices and using a 2D histogram or sparse matrix approach. Additionally, the trial loop (line 165) processes trials sequentially when independent trials could be processed in parallel.

ii.
```python
for n, st in enumerate(good_spike_times):  # loops over neurons
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. CONVERSION_NOTES acknowledged the inefficiency but only minor optimizations were applied (pre-slicing spike times, skipping empty sessions).

## 10-c. What processing does the code repeat multiple times?

i. For every trial, the code reloads and re-filters the same set of good spike times. The `good_spike_times` list is built once per session, but for each trial the same spike times are re-masked and re-histogrammed with overlapping time regions. The tongue tracking data (`tongue_t`, `tongue_y`) is also fully loaded once but the valid-index computation and searchsorted are repeated per trial.

ii.
```python
# good_spike_times loaded once per session
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
# But for each trial, each neuron's full spike array is searched:
for i in range(n_trials):  # trial loop
    for n, st in enumerate(good_spike_times):  # neuron loop
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
```

iii. No explicit mention in CONVERSION_NOTES of repeated computation beyond the general bottleneck acknowledgment.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code converts spike counts to firing rates by dividing by bin width (`/ BIN_SIZE`). Whether this is "unnecessary" depends on the decoder — the decoder likely works equally well with raw counts since dividing by a constant doesn't change relative patterns. The code also computes and stores `session_info` metadata for every session including timing information that isn't used by the decoder.

ii.
```python
fr[n] = counts.astype(np.float32) / BIN_SIZE  # dividing by 0.05
```

iii. Converting to firing rates is a standard neuroscience convention and the AI did not discuss whether raw counts would suffice.
