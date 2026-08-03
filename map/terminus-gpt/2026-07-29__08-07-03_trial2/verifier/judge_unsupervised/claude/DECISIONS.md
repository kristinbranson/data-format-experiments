# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (HDF5) files using `h5py`. It recursively globs all `.nwb` files under the `data/` directory, sorts them, and processes each one via `extract_session()`. Each NWB file represents one session. The main entry point is `build_dataset()` which iterates over all session paths.

ii.
```python
paths = sorted(Path('data').rglob('*.nwb'))
if args.sample:
    paths = paths[:2]
data = build_dataset(paths, show_processing=args.show_processing)
```
```python
def extract_session(session_path, show_processing=False):
    with h5py.File(session_path, 'r') as f:
        trials = f['intervals/trials']
        units = f['units']
        ...
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code operates on preprocessed arrays and external paths, so conversion loads raw NWB data directly while matching paper/code semantics for curation and alignment. This is a reasonable approach since the NWB files are the primary data source.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject ID from each NWB file using `f['general/subject/subject_id']`, falling back to the parent directory name if that field is unavailable. Subject IDs are collected into a unique list, and each session is mapped to a subject index.

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

iii. The AI did not provide explicit justification for this approach, but it follows standard NWB conventions. The data directory structure (`data/sub-XXXXXX/`) naturally groups files by subject.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI processes each file independently via `extract_session()` and appends the results to lists. Sessions are included only if they have at least 2 valid trials and at least one good unit.

ii.
```python
for path in paths:
    ntr, itr, otr, info = extract_session(path, show_processing=show_processing)
    if len(ntr) < 2 or (len(ntr) > 0 and ntr[0].shape[0] == 0):
        print(f'Skipping {path.name}: insufficient kept trials or no good units')
        continue
```

iii. The AI noted this is consistent with the NWB file organization and the decoder requirement of at least 2 trials per session. The final count is 174 sessions.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` group in each NWB file. The AI reads `start_time` and associated trial metadata fields (`trial_instruction`, `outcome`, `early_lick`, `photostim_onset`, `photostim_duration`). Each valid trial produces one neural matrix, one input array, and one output array.

ii.
```python
trials = f['intervals/trials']
n_trials = len(trials['id'])
trial_start = trials['start_time'][:].astype(float)
...
for i in range(n_trials):
    if not valid_trials[i]:
        continue
    ...
```

iii. The AI uses the NWB trial table directly, which is the standard source for trial definitions in this dataset.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) required labels must be present (`trial_instruction` in left/right, `outcome` in ignore/miss/hit, `early_lick` in no early/early), (2) the trial's aligned neural window must fall within the temporal coverage of spike recordings, and (3) trials where all neurons have zero spikes are excluded.

ii.
```python
valid_trials = np.array([
    ti in ('left', 'right') and oc in ('ignore', 'miss', 'hit') and el in ('no early', 'early')
    for ti, oc, el in zip(trial_instruction, outcome, early_lick)
], dtype=bool)

go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
window_ok = (go_cue_times + WINDOW_START >= spike_t_min) & (go_cue_times + WINDOW_END <= spike_t_max)
valid_trials = valid_trials & window_ok
...
if np.all(fr == 0):
    continue
```

iii. The AI documented discovering that in some sessions, behavioral timestamps extend well beyond neural recording coverage (e.g., spikes end at ~1107s but trials continue to ~3705s), which originally caused many all-zero neural trials. The neural coverage check was added to address this.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` in the NWB files. These contain the spike timestamps for all recorded units.

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

iii. The AI identified that spike_times is the raw neural data source, consistent with electrophysiology recordings stored in NWB format.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram` with edges defined by the trial window (-2.5s to +1.5s around go cue). The counts are converted to firing rates by dividing by the bin size (0.05s). This produces an (n_neurons, 80) matrix per trial.

ii.
```python
fr = np.zeros((len(good_spike_times), N_BINS), dtype=np.float32)
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The AI chose 50ms bins as specified in the instructions and converts to firing rates (spikes/second). The reference code used 40ms bins, but the decoder task explicitly requires 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `unit_quality == 'good'` are kept. This is the sole neuron filtering criterion applied by the AI.

ii.
```python
unit_quality = np.array([_decode(x) for x in units['unit_quality'][:]])
good_mask = unit_quality == 'good'
spike_times_list = get_unit_spike_times(units)
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
```

iii. The AI noted that the papers describe using only units labeled 'good' by region-specific QC classifiers. However, using unit_quality=='good' from the NWB file yielded 154,948 units, far exceeding the 69,943 good units reported in the paper. The AI acknowledged this discrepancy but did not resolve it, noting it as a remaining unresolved issue. The paper likely used additional filtering (e.g., the region-specific classifiers described in ChenLiuEtAl2023_SpikeSortingQC.pdf) that goes beyond the NWB `unit_quality` field.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset, which is derived as `trial_start + 1.85s`. The 1.85s offset comes from the task structure described in methods.txt: sample epoch (3 x 150ms tones + 100ms inter-tone intervals = 0.65s) + delay epoch (1.2s) = 1.85s.

ii.
```python
GO_CUE_FROM_TRIAL_START = 1.85
go_cue_times = trial_start + GO_CUE_FROM_TRIAL_START
...
align = go_cue_times[i]
edges_abs = align + BIN_EDGES
```

iii. The AI noted that the NWB trial tables lack explicit go cue timestamp columns, so the go cue time must be derived from the documented task structure. The methods describe the auditory delayed-response task timing: tone onset at trial start, go cue at 1.85s after trial start.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms, producing 80 bins for the 4-second window (-2.5s to +1.5s). Spike times are directly binned at this resolution — there is no intermediate rebinning step.

ii.
```python
BIN_SIZE = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_BINS = int(round((WINDOW_END - WINDOW_START) / BIN_SIZE))  # = 80
BIN_EDGES = np.arange(WINDOW_START, WINDOW_END + BIN_SIZE, BIN_SIZE)
```

iii. The instructions explicitly require 50ms bins. The reference analysis code used 40ms bins, but the AI correctly followed the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from the trial `start_time` field in `intervals/trials`. The AI assumes tone onset occurs at trial start (offset = 0.0s).

ii.
```python
TONE_ONSET_FROM_TRIAL_START = 0.0
tone_onset_times = trial_start + TONE_ONSET_FROM_TRIAL_START
```

iii. The AI derived this from the methods description of the auditory delayed-response task, where the sample epoch (tones) begins at the start of the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as the difference between each bin center (in absolute time) and the tone onset time. This produces a linearly increasing time series across the trial window.

ii.
```python
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. This produces values ranging from about -0.625s (2.5s before go cue, which is 1.85s after tone onset, so -2.5+1.85 = -0.65s) to about 3.325s (1.5s after go cue = 1.85+1.5 = 3.35s). The verification output confirms the range is [-0.6, 3.3], consistent with bin centers offset from the edges.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by computing the values at the same bin centers used for neural data binning. Both neural and input data share the same time grid defined by `centers_abs = align + BIN_CENTERS`.

ii.
```python
centers_abs = align + centers_rel  # align = go_cue_times[i]
time_from_tone = (centers_abs - tone_onset_times[i]).astype(np.float32)
```

iii. Using the same time grid ensures alignment between neural activity and this input variable.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `photostim_onset` and `photostim_duration` fields in `intervals/trials`, combined with the trial `start_time`.

ii.
```python
photostim_onset = [parse_optional_time(x) for x in trials['photostim_onset'][:]]
photostim_duration = [parse_optional_time(x) for x in trials['photostim_duration'][:]]
```

iii. These fields indicate when photostimulation occurred relative to trial start and for how long. Many trials have 'N/A' values, indicating no photostimulation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary (0/1) time series is created. For trials with valid photostim onset and duration, bins whose centers fall within the photostim window are set to 1. The photostim start is computed as `trial_start + photostim_onset`, and the stop as `start + duration`.

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

iii. The AI treats photostim_onset as relative to trial_start, consistent with the NWB convention. Trials without photostimulation (N/A values) remain all-zero.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Like time-from-tone, photostimulation is evaluated at the same bin centers as the neural data, ensuring temporal alignment.

ii.
```python
photo[(centers_abs >= p_start) & (centers_abs < p_stop)] = 1.0
```

iii. Using bin centers for comparison ensures the photostim binary signal is aligned to the same time grid as the neural firing rates.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from `trial_instruction` in `intervals/trials`, which indicates the instructed lick direction (left or right), NOT the animal's actual lick direction.

ii.
```python
CHOICE_MAP = {b'left': 0, b'right': 1, 'left': 0, 'right': 1}
choice_val = CHOICE_MAP[trial_instruction[i]]
```

iii. The AI maps left=0 and right=1 as specified in the instructions. However, `trial_instruction` represents the correct/instructed direction, not the animal's actual choice. For hit trials, the actual choice equals the instruction, but for miss trials, the animal chose the opposite side, and for ignore trials, the animal made no choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The choice value is mapped to an integer (left=0, right=1) and repeated across all 80 time bins as a constant per-trial time series.

ii.
```python
choice_val = CHOICE_MAP[trial_instruction[i]]
choice_ts = np.full(N_BINS, choice_val, dtype=np.int64)
```

iii. The AI makes choice time-invariant within a trial, which matches the per-trial nature of the choice output specified in the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `outcome` field in `intervals/trials`.

ii.
```python
OUTCOME_MAP = {b'ignore': 0, b'miss': 1, b'hit': 2, 'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = OUTCOME_MAP[outcome[i]]
```

iii. Direct mapping from the NWB trial metadata field, consistent with the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome string is mapped to an integer (ignore=0, miss=1, hit=2) and repeated across all time bins.

ii.
```python
outcome_val = OUTCOME_MAP[outcome[i]]
outcome_ts = np.full(N_BINS, outcome_val, dtype=np.int64)
```

iii. Mapping matches the instructions: ignore=0, miss=1, hit=2.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the `early_lick` field in `intervals/trials`.

ii.
```python
EARLY_MAP = {b'no early': 0, b'early': 1, 'no early': 0, 'early': 1}
early_val = EARLY_MAP[early_lick[i]]
```

iii. Direct mapping from the NWB trial metadata.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The early lick string is mapped to an integer (no=0, yes=1) and repeated across all time bins.

ii.
```python
early_val = EARLY_MAP[early_lick[i]]
early_ts = np.full(N_BINS, early_val, dtype=np.int64)
```

iii. Mapping matches the instructions: no early lick=0, early lick=1.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` in the NWB file. The data array has columns [x, y, likelihood], and the y-coordinate (column index 1) is used.

ii.
```python
tongue_group = choose_tongue_group(f)
...
tongue_data = tongue_group['data'][:]
tongue_t = tongue_group['timestamps'][:].astype(float)
if tongue_data.ndim == 2 and tongue_data.shape[1] >= 2:
    tongue_y = tongue_data[:, 1].astype(float)
    if tongue_data.shape[1] >= 3:
        lik = tongue_data[:, 2].astype(float)
        tongue_y = tongue_y.copy()
        tongue_y[lik < 0.5] = np.nan
```

iii. The AI prefers Camera0 if available. Low-confidence frames (likelihood < 0.5) are masked as NaN.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The tongue y-position is interpolated to the neural time grid using nearest-neighbor lookup via `np.searchsorted`. Points where the nearest tongue tracking sample is more than 0.1s away from the bin center are set to NaN.

ii.
```python
valid_idx = np.flatnonzero(np.isfinite(tongue_y))
...
vt = tongue_t[valid_idx]
vy = tongue_y[valid_idx]
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
far = np.abs(vt[idx] - centers_abs) > 0.1
y = y.copy()
y[far] = np.nan
```

iii. The 0.1s tolerance ensures only temporally proximate tongue tracking data contributes to the output.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles are computed from all valid (non-NaN) tongue y values. The continuous y-position is discretized: below 40th percentile = 0, between 40th and 60th = 1, above 60th = 2. Time points without valid data default to category 1 (middle).

ii.
```python
if tongue_y is not None and np.isfinite(tongue_y).any():
    q40, q60 = np.nanpercentile(tongue_y, [40, 60])
...
tongue_disc = np.full(N_BINS, 1, dtype=np.int64)
...
tongue_disc[finite & (y < q40)] = 0
tongue_disc[finite & (y > q60)] = 2
```

iii. The instructions specify session-wise percentile discretization: 0 for <40th, 1 for 40th-60th, 2 for >60th. The AI implements this correctly in logic, but the percentiles are computed over ALL tongue y data for the session, not just the data within trial windows. The default of 1 for missing/invalid data biases the distribution heavily toward the middle category (90.1% of values are category 1).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The tongue y-position is interpolated to the same bin centers as the neural data using nearest-neighbor lookup. This ensures temporal alignment between tongue tracking data and neural firing rates.

ii.
```python
idx = np.searchsorted(vt, centers_abs, side='left')
idx = np.clip(idx, 0, len(vt) - 1)
y = vy[idx]
```

iii. The alignment uses the same `centers_abs` time grid as neural and input data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/problematic data:
- Missing photostim values ('N/A'): treated as no photostimulation (all zeros)
- Missing tongue tracking data: defaults to category 1 (middle percentile bin)
- Low-confidence tongue tracking frames (likelihood < 0.5): masked as NaN
- Trials outside neural recording coverage: excluded
- All-zero neural trials: excluded
- Missing brain region metadata: initially used 'unknown', later resolved by deriving from electrode location metadata

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
```

iii. The AI documented discovering and fixing issues iteratively. The neural coverage check was added after discovering that some sessions had trial timestamps extending far beyond neural recording time.

## 10-a. What are the most time-consuming steps of the code?

i. The per-neuron spike histogramming loop is the main bottleneck, with per-session times ranging from ~5s to ~143s. The total conversion takes approximately 90 minutes for 174 sessions.

ii.
```python
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    if np.any(mask):
        counts, _ = np.histogram(st[mask], bins=edges_abs)
        fr[n] = counts.astype(np.float32) / BIN_SIZE
```

iii. The AI identified this as a bottleneck in CONVERSION_NOTES.md (projected ~50+ min) but did not fully optimize it. Each trial requires histogramming all good neurons' spike times.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the per-neuron histogramming inside each trial. Currently, for each trial, every neuron's spike times are independently masked and histogrammed in a Python for-loop.

ii.
```python
for n, st in enumerate(good_spike_times):
    mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
    ...
```

iii. This could be vectorized by concatenating all spike times with neuron labels, then using a 2D histogram or digitize-based approach. The trial loop itself cannot easily be vectorized since bin edges change per trial.

## 10-c. What processing does the code repeat multiple times?

i. For each trial, the full spike time arrays for all neurons are re-masked and re-histogrammed. The masking operation `(st >= edges_abs[0]) & (st < edges_abs[-1])` is done independently for every trial-neuron combination, even though the spike times are the same — only the edges change.

ii. The per-trial loop reads the same `good_spike_times` arrays repeatedly:
```python
for i in range(n_trials):
    ...
    for n, st in enumerate(good_spike_times):
        mask = (st >= edges_abs[0]) & (st < edges_abs[-1])
```

iii. A more efficient approach would be to sort spike times once and use binary search for each trial's time window, or pre-compute a session-wide spike count matrix.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads ALL spike times for the entire session into memory for every unit, even though only spikes within the trial windows (-2.5s to +1.5s around go cue) are needed. Also, the AI reads the full tongue tracking data for the entire session, even though only samples within valid trial windows contribute to the output.

ii.
```python
spike_times_list = get_unit_spike_times(units)  # loads all spike times
good_spike_times = [spike_times_list[i] for i in np.where(good_mask)[0]]
...
tongue_data = tongue_group['data'][:]  # loads all tongue data
tongue_t = tongue_group['timestamps'][:].astype(float)
```

iii. Loading all data upfront is simpler but memory-intensive. For sessions with spike recordings much longer than the trial windows, a significant portion of the loaded spike data is never used in histograms. The `--show-processing` option is also accepted but not implemented (no plots are generated).
