# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by globbing for all `*.nwb` files under `data/` using `pathlib.Path.rglob`, then opens each file with `h5py.File` (not `pynwb`). It accesses trial metadata from `intervals/trials`, unit data from `units`, and behavioral data from `acquisition/BehavioralEvents` and `acquisition/BehavioralTimeSeries`. Each NWB file corresponds to one session.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    paths = paths[:2]
data = build_dataset(paths, show_processing=args.show_processing)
```

```python
with h5py.File(session_path, 'r') as f:
    trials = f['intervals/trials']
    n_trials = len(trials['id'])
    trial_start = trials['start_time'][:].astype(float)
    ...
    units = f['units']
    unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
```

iii. The AI chose `h5py` over `pynwb` for direct HDF5 access. The CONVERSION_NOTES.md states that the code directory analysis revealed that NWB files contain the relevant data groups. The AI identified 174 NWB files across 28 subjects.

## 1-b. How are the data split into subjects?

i. The AI reads subject IDs from `general/subject/subject_id` in each NWB file, falling back to the parent directory name if that fails. Subjects are accumulated into a list as new IDs are encountered, with an index mapping maintained via a dictionary.

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

iii. The AI identified `subject_id` from the NWB file structure. The subjects list is built in encounter order rather than sorted.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are identified by the file stem (e.g., `sub-440956_ses-20190208T120657_behavior+ecephys+ogen`). The file list is sorted, giving deterministic order.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
```

```python
'session_id': session_path.stem,
```

iii. The AI noted in CONVERSION_NOTES.md that the NWB file organization maps one file to one session.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. The number of trials is determined by `len(trials['id'])`. Each trial has a `start_time` and associated metadata columns.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
```

iii. The AI reads trials directly from the NWB trials table, iterating over each trial index in a for loop.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) trials with invalid labels (instruction not left/right, outcome not ignore/miss/hit, or early_lick not 'no early'/'early') are excluded; (2) trials whose neural window falls outside the spike time coverage are excluded; (3) after binning, trials with all-zero neural data are dropped.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)

go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok
```

```python
if np.all(fr == 0):
    continue
```

iii. The AI discovered during debugging (documented in CONVERSION_NOTES.md Steps 10/12) that many trials had all-zero neural data because behavioral timestamps extended beyond spike recording. It added the neural coverage filter and explicit zero-trial removal. The AI did not use `obs_intervals` or `free_water` filtering from the NWB file.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index`, which store the sorted spike times for each unit. Only units with `unit_quality == 'good'` are included.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The AI identified `spike_times` as the neural data source and `unit_quality` as the quality filter field.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram` for each unit per trial. The spike counts are divided by the bin size (0.05s) to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The AI converts spike counts to firing rates (Hz) by dividing by the bin width, consistent with the reference approach.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units using `units/unit_quality == 'good'`. This retained 154,948 units across all sessions.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
```

iii. The CONVERSION_NOTES.md acknowledges this is a major discrepancy: "unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper." The AI did not discover or use the `classification` field, which is the QC classifier verdict described in the spike sorting white paper. The `unit_quality` field is an older, more permissive label.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI derives the go cue time as a fixed offset from trial start: `go_cue_times = trial_start + 1.85` (1.85 seconds after trial start). This is based on the task timing described in methods.txt (0.65s sample + 1.2s delay = 1.85s).

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85

go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

```python
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The CONVERSION_NOTES.md states: "If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start." The AI believed the NWB files lacked explicit go cue timestamps. However, the actual go cue timestamps are available in `acquisition/BehavioralEvents/go_start_times/timestamps`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins spanning -2.5s to +1.5s relative to the (derived) go cue, producing 80 time bins per trial. No rebinning is applied; spikes are binned directly from spike times.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
```

iii. The bin size and window match the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The tone onset is derived as a fixed offset from trial start: `tone_onset_times = trial_start + 0.0` (i.e., tone onset equals trial start time).

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0

tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The AI derived tone onset from the documented task structure, assuming the tone starts at trial start (time 0). The actual tone onset times are available in `acquisition/BehavioralEvents/sample_start_times/timestamps`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial and time bin, the value is computed as the absolute time of the bin center minus the tone onset time.

ii.
```python
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

Where `centers_abs = align + centers_rel` and `align = go_cue_times[i]`.

iii. The computation is straightforward subtraction, but both the go cue and tone onset are derived from fixed offsets rather than actual timestamps.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same bin centers as the neural data, so they share the same time grid.

ii.
```python
centers_abs = align + centers_rel  # same alignment as neural bins
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. Both neural and input data use the same bin center times, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. The AI correctly identified the photostimulation fields in the NWB trials table.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset is interpreted as relative to trial start. A binary time series is created: 1.0 where bin centers fall within the stimulation window, 0.0 otherwise.

ii.
```python
photo = np.zeros(N_BINS, dtype=np.float32)
p_on = photostim_onset[i]
p_dur = photostim_duration[i]
if p_on is not None and p_dur is not None:
    p_start = trial_start[i] + p_on
    p_stop = p_start + p_dur
    photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. The AI uses absolute times (trial_start + onset) for the photostim window, which is correct since onset is stored relative to trial start.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim binary is evaluated at the same bin centers as the neural data, using absolute timestamps.

ii.
```python
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Same bin grid as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `trial_instruction` in the trials table, mapping 'left' to 0 and 'right' to 1. This represents the **instructed** side, not the animal's actual lick direction.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}

choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The AI maps the trial instruction directly to the choice output, without considering whether the animal actually licked the instructed direction or not.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed direction is mapped to 0 (left) or 1 (right) and repeated across all 80 time bins. There are only 2 classes, with no separate class for no-lick (ignore) trials.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. The AI's output has 2 classes for choice, whereas the instructions specify "left = 0, right = 1" which could be interpreted as 2 classes, but the reference solution uses 3 classes (left, right, no lick) to handle ignore trials where the animal didn't lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. The outcome strings are read directly from the NWB file.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to 0 (ignore), 1 (miss), 2 (hit) and repeated across all 80 time bins.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}

outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. The mapping follows the instructions exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. The early lick flag is read directly from the NWB file.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes) and repeated across all time bins.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}

early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. The mapping follows the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 (y-coordinate) and column 2 (likelihood) of the data array, plus timestamps.

ii.
```python
tongue_group = choose_tongue_group(f)
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
tongue_y = tongue_data[:, 1].astype(float)
if tongue_data.shape[1] >= 3:
    lik = tongue_data[:, 2].astype(float)
    tongue_y[lik < 0.5] = np.nan
```

iii. The AI correctly identifies the tongue tracking data source and applies a likelihood threshold of 0.5.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.5 are set to NaN. Session-wide 40th and 60th percentiles are computed from the **raw valid frame values** (not bin means). For each trial, the AI uses nearest-neighbor interpolation from valid frames to bin centers (with a 0.1s distance threshold), then discretizes into 3 classes based on the percentile thresholds.

ii.
```python
# Session-wide percentiles from raw frames
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])

# Per-trial nearest-neighbor interpolation
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
vt = tongue_t[valid_idx]
vy = tongue_y[valid_idx]
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y[far] = np.nan

# Discretize
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)  # default to middle class
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. The CONVERSION_NOTES.md notes that "tongue discretization improved after lowering likelihood threshold and nearest-valid sampling, but remains strongly skewed toward the middle class." The default value for bins without valid data is class 1 (middle), rather than a separate 'not visible' class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three classes: 0 (< 40th percentile), 1 (40th-60th percentile or not visible), 2 (> 60th percentile). The percentiles are computed from raw valid frame y-values across the whole session. Bins without nearby valid frames default to class 1 (middle).

ii.
```python
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)  # default middle class
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

```python
'output_values': [
    ...
    ['lt_40pct', '40_to_60pct', 'gt_60pct'],
]
```

iii. The AI uses 3 classes with no separate "not visible" class. The reference uses 4 classes with a distinct "not visible" class (value 3) for bins with no tongue data.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses nearest-neighbor interpolation: for each bin center, it finds the closest valid tongue frame (within 0.1s) and uses that value. This differs from the reference approach of averaging all frames within each 50ms bin.

ii.
```python
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y[far] = np.nan
```

iii. The nearest-neighbor approach with 0.1s threshold is an approximation that could introduce artifacts compared to proper bin averaging.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled: (1) Sessions where no good units exist are skipped; (2) Trials outside neural recording coverage are excluded; (3) Trials with all-zero neural data are dropped after binning; (4) Tongue frames with low likelihood are set to NaN and bins without nearby valid frames default to the middle tongue class.

ii.
```python
if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
    print(f'Skipping {path.name}: insufficient kept trials or no good units')
    continue

if np.all(fr == 0):
    continue
```

iii. The AI's CONVERSION_NOTES.md extensively documents the debugging process for all-zero neural trials, which was caused by behavioral timestamps extending beyond spike recording coverage. The AI added the spike time coverage check and all-zero trial filter to handle this.

## 10-a. What are the most time-consuming steps of the code?

i. The per-neuron histogram loop inside `extract_session` is the main bottleneck. The AI noted ~17.7s/session for early sessions, projecting 50+ minutes for the full dataset. The AI acknowledged needing optimization but the final code still uses per-trial, per-neuron histogram calls.

ii.
```python
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
        if np.any(mask):
            counts, _ = np.histogram(st[mask], bins=edges_abs)
            fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The CONVERSION_NOTES.md states: "Per-neuron histogram loop is a bottleneck on full dataset; observed ~17.7 s/session over first 3 sessions."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The nested loop over trials and neurons (for spike binning) could be significantly improved. The reference solution vectorizes across all trials simultaneously by flattening bin edges and using a single `searchsorted` per unit. The AI's code runs `np.histogram` once per unit per trial, which is O(n_trials * n_units) calls.

ii.
```python
# AI's nested loop approach:
for i in range(n_trials):
    for n, st in enumerate(good_spike_times):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
```

iii. The AI identified this as a bottleneck but did not vectorize it.

## 10-c. What processing does the code repeat multiple times?

i. Spike times are read into a list of arrays per unit (`get_unit_spike_times`), then for each trial, each unit's spike times are filtered by the trial window and histogrammed. This repeated filtering/masking of the same spike arrays is redundant work.

ii.
```python
def get_unit_spike_times(units_group):
    spikes = units_group['spike_times'][:]
    index = units_group['spike_times_index'][:]
    out = []
    start = 0
    for stop in index:
        out.append(spikes[start:stop])
        start = stop
    return out
```

iii. The spike times are loaded once but processed repeatedly per trial.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads spike times for ALL units before filtering to good units, which wastes memory and I/O. It also computes `spike_t_min` and `spike_t_max` across all good units for trial coverage filtering, which is an approximation of the more precise `obs_intervals` approach.

ii.
```python
spike_times_list = get_unit_spike_times(units)  # loads ALL units
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]  # then filters
```

iii. Loading all spike times before filtering is wasteful but functionally correct.
