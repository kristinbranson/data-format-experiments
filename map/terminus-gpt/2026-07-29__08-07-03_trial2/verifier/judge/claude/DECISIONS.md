# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `data/` using `h5py` (not `pynwb`). It finds all `.nwb` files using `Path('data').rglob('*.nwb')`, sorts them, and processes each sequentially via `extract_session()`. Each file is opened with `h5py.File()` and data is read from `intervals/trials`, `units`, `acquisition/BehavioralTimeSeries`, and `acquisition/BehavioralEvents`.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
...
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
```

```python
def extract_session(session_path, show_processing=False):
    with h5py.File(session_path, 'r') as f:
        trials = f['intervals/trials']
        ...
```

iii. The AI chose `h5py` instead of `pynwb` to read NWB files. Its CONVERSION_NOTES state: "Use NWB as source of truth; use code/papers to guide curation and alignment." No explicit justification for `h5py` over `pynwb` was given.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject ID from `general/subject/subject_id` in each NWB file. If that field is unavailable, it falls back to the parent directory name. Subjects are accumulated in encounter order (not sorted).

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

iii. The AI noted that subject ID is available from the NWB subject metadata. No explicit justification for the encounter-order (vs sorted) subject list was provided.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI identifies sessions by `session_path.stem` (the filename without extension). Sessions are processed in sorted file-path order.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
...
'session_id': session_path.stem,
```

iii. The AI's CONVERSION_NOTES state that NWB files contain session-level data and the file boundary defines the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. Each row is one trial. The number of trials is determined by `len(trials['id'])`.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
```

iii. The AI used the NWB trials table directly. No assertion is made that go cue events match trial count (because go cue times are not read from the file).

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two ways: (1) requiring valid categorical labels (`trial_instruction` in left/right, `outcome` in ignore/miss/hit, `early_lick` in no early/early), and (2) requiring the neural recording window to cover the trial's go-cue-aligned window (checking spike time min/max). Additionally, trials where the binned neural matrix is all zeros are dropped post-hoc. Sessions with fewer than 2 surviving trials or no good units are skipped.

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

iii. The AI's CONVERSION_NOTES document that trials outside neural coverage and all-zero neural trials were excluded to fix an issue where many trials had no concurrent neural data. The AI does NOT filter based on `obs_intervals` or `free_water` as the reference does.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (via `units/spike_times` and `units/spike_times_index`). Only units with `unit_quality == 'good'` are included.

ii.
```python
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The AI identified spike times as the raw neural data source, consistent with the reference.

## 2-b. How is the `neural` data processed?

i. For each trial and each good unit, spikes are binned into 50ms bins spanning -2.5s to +1.5s around the (derived) go cue using `np.histogram`. Spike counts are divided by the bin width (0.05s) to get firing rates in Hz. No smoothing or normalization is applied.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The AI processes spikes into firing rates in Hz using standard histogram binning, consistent with the general approach of the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters units using `units/unit_quality == 'good'`. This yields 154,948 good units across all sessions, which is significantly more than the 69,943 units obtained using `units/classification == 'good'` (the QC classifier verdict used in the reference).

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
```

iii. The AI's CONVERSION_NOTES acknowledge the discrepancy: "unit curation from available NWB fields (`unit_quality`) yields 154,948 units, which does not match the 69,943 good units reported in the paper." The agent noted it could not find the `classification` field through its h5py-based exploration and left this as an unresolved issue.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI derives go cue time as `trial_start + 1.85s` using a hardcoded offset from the methods text, rather than reading the actual go cue timestamps from `BehavioralEvents/go_start_times`. Bin edges are computed as `align + BIN_EDGES` where `align = go_cue_times[i]`.

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The AI's CONVERSION_NOTES state: "If explicit event timestamps are absent, derive tone/go cue times from documented task structure relative to trial start." The agent determined from its NWB exploration that the trials table lacked explicit go cue columns, but missed that `BehavioralEvents/go_start_times` provides actual go cue timestamps. This is problematic because early-lick trials replay the sample/delay epochs, making the actual go cue time differ from trial_start + 1.85s.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50ms bins spanning -2.5s to +1.5s, yielding 80 bins per trial. This matches the instructions exactly.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
```

iii. The binning parameters follow the instructions directly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives tone onset time as `trial_start + 0.0` (i.e., tone onset equals trial start time), using a hardcoded constant from the methods text. It does NOT read `sample_start_times` from `BehavioralEvents`.

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The AI's CONVERSION_NOTES state: "derive tone onset from trial `start_time`" based on the methods description. The agent believed explicit tone onset timestamps were absent from the NWB file.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the difference between the absolute bin center time and the derived tone onset time, giving a continuous time-varying input.

ii.
```python
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. This is a straightforward subtraction. Since the AI assumes tone onset = trial start, the values represent time from trial start rather than time from actual tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and the time-from-tone input use the same bin centers derived from `align + BIN_CENTERS`, where `align` is the derived go cue time. This ensures temporal alignment between neural and input data.

ii.
```python
centers_abs = align + centers_rel
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. The bin grid is shared across all data streams.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` used to compute absolute onset times.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time-varying signal is constructed: 1.0 where the bin center falls between `trial_start + photostim_onset` and `trial_start + photostim_onset + photostim_duration`, 0.0 elsewhere. Trials without photostimulation (where onset is 'N/A') have all zeros.

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

iii. The approach is conceptually the same as the reference, computing absolute onset/offset and checking bin centers. The AI uses absolute times while the reference converts to go-cue-relative times, but the result should be equivalent.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation onset/offset are compared against the same absolute bin centers (`centers_abs`) used for neural binning, ensuring alignment.

ii.
```python
centers_abs = align + centers_rel
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Same bin grid is used for all data streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` (the instructed lick direction), NOT from the actual lick direction. It maps 'left' to 0 and 'right' to 1. There is no "no lick" category.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
...
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The AI equates choice with instruction. This is incorrect: choice should reflect what the animal actually did (which differs from the instruction on miss trials, and is absent on ignore trials).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed direction is mapped to 0 (left) or 1 (right) and repeated across all 80 bins. The output_values list only has `['left', 'right']` with no "no lick" option.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
...
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. No justification was given for using instruction as choice or for omitting the "no lick" category. The instructions explicitly state "Lick direction choice (left, right, no lick, per-trial)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which holds 'ignore', 'miss', and 'hit'.

ii.
```python
outcome = np.array([_decode(x) for x in trials['outcome'][:]])
...
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. Correct source variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to integers: ignore=0, miss=1, hit=2, and repeated across all 80 bins.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
...
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. Consistent with instructions and reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, holding 'no early' and 'early'.

ii.
```python
early_lick = np.array([_decode(x) for x in trials['early_lick'][:]])
```

iii. Correct source variable.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no) and 1 (yes), repeated across all 80 bins.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
...
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. Consistent with instructions and reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, using column 1 (y-position) and column 2 (likelihood) from the `data` array, with matching `timestamps`.

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

iii. Same source variable as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood < 0.5 are set to NaN. The 40th and 60th percentiles are computed over ALL valid (non-NaN) raw frames across the entire session (not bin means). For each trial, the AI uses nearest-valid-frame sampling (searchsorted) to assign a tongue y value to each bin center, with frames farther than 0.1s marked as NaN. Values below q40 get class 0, between q40 and q60 get class 1, above q60 get class 2. Bins with no nearby valid frame remain at the default of class 1 (40th-60th percentile), NOT a "not visible" class.

ii.
```python
q40, q60 = np.nanpercentile(tongue_y, [40, 60])
...
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)  # default to middle class
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y[far] = np.nan
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. The AI's approach differs from the reference in several ways: (1) percentiles are computed over raw frames, not 50ms bin means; (2) nearest-frame sampling is used instead of bin averaging; (3) no "not visible" class exists - bins without valid tongue data default to class 1 (middle), and output_values only lists 3 classes.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories only: 0 (below 40th percentile), 1 (40th to 60th percentile, also the default), 2 (above 60th percentile). No "not visible" (class 3) category. The instructions specify 4 categories including "3: not visible."

ii.
```python
'output_values': [
    ...
    ['lt_40pct', '40_to_60pct', 'gt_60pct'],
]
```

iii. The AI did not implement the "not visible" category specified in the instructions. Bins where the tongue is not visible default to class 1, which conflates actual middle-range tongue positions with missing data.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the same bin centers (`centers_abs`) to look up the nearest valid tongue frame for each bin, ensuring temporal alignment with the neural data.

ii.
```python
idx = np.searchsorted(vt, centers_abs, side='left')
```

iii. The bin grid is shared across all data streams. However, the nearest-frame approach differs from the reference's bin-mean approach.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three approaches: (1) Sessions where no `subject_id` is available fall back to the directory name. (2) Trials with all-zero neural activity are dropped post-hoc. (3) Tongue frames with low likelihood are set to NaN, and bins without nearby valid frames default to class 1. The AI does NOT handle the session with NaN `classification` values (since it doesn't use that field), and does not use `obs_intervals` or `free_water` for trial filtering.

ii.
```python
# Subject fallback
except Exception:
    return path.parent.name

# All-zero neural trial drop
if np.all(fr == 0):
    continue

# Tongue missing data
tongue_y[lik < 0.5] = np.nan
```

iii. The AI's CONVERSION_NOTES document the discovery and resolution of all-zero neural trials and the neural coverage filtering approach.

## 10-a. What are the most time-consuming steps of the code?

i. The AI's CONVERSION_NOTES report that the per-neuron histogram loop was the main bottleneck, with initial processing taking ~17.7 s/session (projected 50+ min for the full dataset). The AI noted this exceeded the 15-minute budget and attempted optimizations.

ii. N/A (performance observation, not code logic)

iii. The AI documented timing information and noted the bottleneck in its CONVERSION_NOTES.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop is the per-neuron per-trial histogram loop. Each trial is processed independently in a Python for-loop over `n_trials`, and within each trial, each neuron is processed in another loop. The reference vectorizes across all trials simultaneously using `searchsorted` on a flattened edge array.

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

iii. The AI's code has nested loops over trials and neurons, whereas the reference processes all trials at once per neuron using vectorized searchsorted.

## 10-c. What processing does the code repeat multiple times?

i. The code computes `go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START` twice (lines 83 and 109). Other than that, processing is done in a single pass.

ii.
```python
# Line 83
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
# Line 109
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
```

iii. Minor redundancy, not significant.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI reads all spike times for all units (including non-good units) before filtering. `get_unit_spike_times` reads all spike times into a list, then selects the good ones. The reference reads the raw buffer once and indexes into it per-unit. Also, brain region extraction via electrode metadata involves JSON parsing of location strings, which is extra work compared to reading `anno_name` directly.

ii.
```python
spike_times_list = get_unit_spike_times(units)  # reads ALL units
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]  # then filters
```

iii. Reading all spike times and then filtering wastes memory and time for the ~45% of units that are not 'good' (under unit_quality) or ~75% that are not 'good' (under classification).
