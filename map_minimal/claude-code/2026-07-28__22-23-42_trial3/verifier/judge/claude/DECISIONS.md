# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI globs all NWB files matching `data/sub-*/sub-*.nwb` (174 files across 28 subjects) and processes each file sequentially using h5py. Each NWB file represents one recording session. Trial data, neural data (spike times), behavioral events, and tongue tracking are all extracted from within each NWB file.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
for nwb_path in nwb_files:
    result = process_session(nwb_path)
```
```python
f = h5py.File(nwb_path, 'r')
# Trial info
n_trials = len(f['intervals/trials/id'][:])
outcomes = np.array([o.decode() if isinstance(o, bytes) else o for o in f['intervals/trials/outcome'][:]])
# Go cue times
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# Spike times
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
# Tongue tracking
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
```

iii. The AI notes in CONVERSION_NOTES.md: "174 NWB files across 28 subjects in `data/sub-*/sub-*.nwb`. Each NWB file represents one recording session." The reference code loads .mat files exported from DataJoint, but the task provides NWB data, so this is the appropriate loading approach.

## 1-b. How are the data split into subjects (mice)?

i. Subject IDs are extracted from NWB filenames by parsing the `sub-XXXXXX` prefix. Unique subject IDs are collected and sorted, and a `subject_idx` array maps each session to its subject.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
# ...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. CONVERSION_NOTES.md: "174 NWB files across 28 subjects." The subject splitting is straightforward from the filename convention used in the DANDI dataset.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The AI processes each file independently in `process_session()` and collects results. Sessions that don't meet quality criteria are skipped.

ii.
```python
for nwb_path in nwb_files:
    result = process_session(nwb_path)
    if result is None:
        n_skipped += 1
        continue
    all_neural.append(result['neural'])
    # ...
```

iii. CONVERSION_NOTES.md: "Each NWB file represents one recording session." The reference code similarly treats each session-probe combination as a unit, then combines probes within a session. Here, NWB files already contain combined probe data per session.

## 1-d. How are the data split into trials?

i. Trial information is read from the NWB `intervals/trials/` table. Go cue times are from `acquisition/BehavioralEvents/go_start_times/timestamps`. A trial mask selects valid trials, and data is extracted per-trial within each session.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# ...
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
trial_indices = np.where(trial_mask)[0]
# ...
for ti in trial_indices:
    gc = go_cue_times[ti]
    # process each trial...
```

iii. The AI verifies that go cue count matches trial count with an assertion: `assert len(go_cue_times) == n_trials`.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of filtering:
- **Session-level**: Performance > 65% on control trials, and >= 50 correct left AND >= 50 correct right on control trials. Control trials are those with no early lick, no auto_water, no free_water, and no photostim.
- **Trial-level**: Excludes `auto_water` and `free_water` trials. Also excludes trials outside the neural recording observation window. Keeps early lick trials, ignore/no-response trials, and photostimulation trials.

ii.
```python
# Session filtering
is_control = ((early_lick == 'no early') & (auto_water == 0) & (free_water == 0) & (ps_onset_str == 'N/A'))
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None
# Trial filtering
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```

iii. CONVERSION_NOTES.md: "Excluded auto_water and free_water trials (as in reference code). Kept early lick trials (needed for early_lick decoder output). Kept ignore/no-response trials (needed for outcome decoder output). Kept photostimulation trials (needed for photostim decoder input)." The reference code's `get_regular_trial_mask` also excludes early lick, no-response, and stimulation trials, but the AI chose to keep them because the decoder task requires them as inputs/outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in the NWB units table (`/units/spike_times` with ragged array indexing via `/units/spike_times_index`).

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
# Build per-unit spike time arrays
unit_spike_times = []
for ui in unit_indices:
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. CONVERSION_NOTES.md: "Loaded from `/units/spike_times` (ragged array with index in `/units/spike_times_index`)."

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, then binned into 50ms non-overlapping bins spanning -2.5s to 1.5s relative to go cue (80 bins total). Spike counts per bin are divided by the bin width (0.05s) to produce firing rates in Hz.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)

def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

iii. CONVERSION_NOTES.md: "50ms non-overlapping bins (as specified in decoder task). Spike counts divided by bin width (0.05s) to get rates in Hz. 80 time bins per trial." The reference code uses `sliding_histogram` with 100ms bandwidth and 50ms stride (or 40ms bandwidth with 3.4ms stride in `preprocess_all_ephys.py`), but the decoder task specifies 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by classifier-based quality control (`classification == 'good'` in the NWB units table). Additionally, units must have a valid brain region annotation (via `map_anno_name_to_region`). Units with empty or unmappable annotation names are excluded.

ii.
```python
classification = np.array([c.decode() if isinstance(c, bytes) else c for c in f['units/classification'][:]])
good_mask = classification == 'good'
# Map brain regions and filter units with valid regions
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

iii. CONVERSION_NOTES.md: "Classifier-based QC (classification='good' in NWB units table). Matches reference: The reference code uses qc_mode='classifier' which applies trained logistic regression classifiers per brain area. Units with empty or unmappable annotation names excluded (490 units, <1%)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data (spike times) is aligned to the go cue onset. Each spike time is subtracted from the go cue time to produce go-cue-relative spike times, which are then binned.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    # ...
```
```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue for each trial." The AI follows this exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (BIN_WIDTH = 0.05s), producing 80 bins per trial covering -2.5s to 1.5s. No rebinning is applied -- spikes are directly binned at this target resolution from the raw spike times.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. CONVERSION_NOTES.md: "50ms non-overlapping bins (as specified in decoder task)... Reference code uses 100ms bandwidth with 50ms stride (Gaussian-weighted), but decoder task specifies 50ms bins." The reference code uses a sliding histogram with different parameters, but the decoder task specification takes precedence.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from two NWB variables:
- `acquisition/BehavioralEvents/sample_start_times/timestamps` (the times when sample tones begin)
- `acquisition/BehavioralEvents/go_start_times/timestamps` (go cue times, used as reference for alignment)

ii.
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. CONVERSION_NOTES.md: "Computed as seconds since the first tone of the (successful) sample epoch. For trials with early lick replays, uses the last sample_start before the go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI finds the last `sample_start_time` before the go cue (to handle early lick replays), computes the tone onset time relative to the go cue, then computes `time_from_tone = BIN_CENTERS - tone_onset_rel` -- the time elapsed since tone onset at each bin center. If no sample_start is found, a fallback value of -1.85s is used.

ii.
```python
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time  # negative value
    return None

# In processing loop:
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel  # time since tone onset at each bin
```

iii. CONVERSION_NOTES.md: "Typical value at go cue: ~1.85s (0.65s sample + 1.2s delay). Range varies across trials due to early lick replays that extend the sample/delay epochs."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Time from tone onset is computed at the same bin centers as the neural data (BIN_CENTERS), which are the midpoints of the 50ms bins spanning -2.5s to 1.5s relative to go cue. This ensures temporal alignment between input and neural data.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
# ...
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The input has shape (2, n_bins) matching the neural data's n_timepoints dimension.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from three NWB variables:
- `intervals/trials/photostim_onset` (onset time relative to trial start; 'N/A' for non-photostim trials)
- `intervals/trials/photostim_duration` (duration of photostimulation)
- `intervals/trials/start_time` (trial start time, needed to convert photostim_onset to absolute time)

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. CONVERSION_NOTES.md: "`photostim_onset` in NWB is relative to trial start (converted to absolute then to go-cue-relative)."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostimulation (photostim_onset != 'N/A'):
1. Parse photostim_onset and photostim_duration as floats
2. Convert photostim_onset from trial-start-relative to absolute time: `ps_abs_onset = trial_start_times[ti] + ps_onset_val`
3. Convert to go-cue-relative: `ps_start_rel = ps_abs_onset - gc`
4. Mark bin centers within [ps_start_rel, ps_start_rel + ps_dur) as 1, others as 0.

ii.
```python
photostim_binary = np.zeros(N_BINS, dtype=np.float64)
if ps_onset_str[ti] != 'N/A':
    ps_onset_val = float(ps_onset_str[ti])
    ps_dur = float(ps_dur_str[ti])
    ps_abs_onset = trial_start_times[ti] + ps_onset_val
    ps_start_rel = ps_abs_onset - gc
    ps_end_rel = ps_start_rel + ps_dur
    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. CONVERSION_NOTES.md: "Photostim typically occurs during late delay epoch (-1.2s to -0.7s relative to go cue). Duration: 0.5s (last 0.5s of delay including 100ms ramp-down)."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is evaluated at the same bin centers as the neural data (BIN_CENTERS), producing a binary time series that is temporally aligned with the neural firing rate bins.

ii.
```python
photostim_binary = ((BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The photostim binary signal has the same number of time points (80) as the neural data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/trial_instruction` in the NWB file, which encodes the instructed lick direction (left or right) based on the tone stimulus.

ii.
```python
instructions = np.array([i.decode() if isinstance(i, bytes) else i
                        for i in f['intervals/trials/trial_instruction'][:]])
# ...
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. CONVERSION_NOTES.md: "Based on `trial_instruction` field (which port the animal should lick)." This uses the INSTRUCTED direction (stimulus identity), not the actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple binary mapping: 'left' -> 0, 'right' -> 1. The value is per-trial, broadcast across all 80 time bins.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
# Later:
np.full(N_BINS, output_choice[i], dtype=np.int64),  # choice (per-trial, broadcast)
```

iii. The reference code also uses trial_type (stimulus identity) rather than actual lick direction: `sess_dict['trial_type'] = 1*(trial_type=='l')` where 1=left, 0=right. Note the AI uses opposite encoding (left=0, right=1) matching the decoder task specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome` in the NWB file, which contains string values: 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. The outcome field directly encodes the trial result.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: 'ignore' -> 0, 'miss' -> 1, 'hit' -> 2. Per-trial value, broadcast across all time bins.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:  # hit
    outcome = 2
output_outcome.append(outcome)
# Later:
np.full(N_BINS, output_outcome[i], dtype=np.int64),  # outcome (per-trial, broadcast)
```

iii. The mapping matches the decoder task specification: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)."

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Outcome is a per-trial scalar value that is broadcast (repeated) across all 80 time bins to match the neural data's temporal dimension. No temporal alignment is needed since it is a per-trial variable.

ii.
```python
np.full(N_BINS, output_outcome[i], dtype=np.int64)
```

iii. The instructions describe outcome as "per-trial", so broadcasting is the appropriate approach.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick` in the NWB file, which contains string values like 'early' or 'no early'.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. The early_lick field directly encodes whether the animal licked prematurely during the sample/delay epoch.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'early' -> 1, anything else -> 0. Per-trial value, broadcast across all time bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
output_early_lick.append(el)
# Later:
np.full(N_BINS, output_early_lick[i], dtype=np.int64),  # early lick (per-trial, broadcast)
```

iii. The mapping matches the decoder task specification: "Early lick (no = 0, yes = 1, per-trial)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from the side-view camera tongue tracking data:
- `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 = y-position)
- `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps`

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. CONVERSION_NOTES.md: "Side-view camera tracking from `/acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`. Shape (N, 3): [x, y, likelihood] from DeepLabCut."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue tracking timestamps are aligned to go cue, and the mean y-position is computed within each 50ms time bin. Bins with no tracking data get NaN.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    n_bins = len(bin_edges) - 1
    tongue_y_binned = np.full(n_bins, np.nan)
    t_abs_start = go_cue_time + bin_edges[0]
    t_abs_end = go_cue_time + bin_edges[-1]
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_window = tongue_timestamps[idx[0]:idx[1]]
    y_window = tongue_y[idx[0]:idx[1]]
    ts_rel = ts_window - go_cue_time
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
    return tongue_y_binned
```

iii. CONVERSION_NOTES.md: "All DLC tracking data used (no confidence filtering)." The data column index 2 is the DLC confidence/likelihood, but it is not used for filtering.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization using the 40th and 60th percentiles of all valid (non-NaN) tongue y-position values across all trials in the session. Categories: 0 (<40th percentile), 1 (40th-60th percentile), 2 (>60th percentile). NaN bins (no tracking data) default to category 1 (middle).

ii.
```python
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)
tongue_y_discrete_trials = []
for ty in tongue_y_all_trials:
    ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
    valid_mask = ~np.isnan(ty)
    ty_disc[valid_mask & (ty < p40)] = 0
    ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
    ty_disc[valid_mask & (ty > p60)] = 2
    tongue_y_discrete_trials.append(ty_disc)
```

iii. CONVERSION_NOTES.md: "0 = below 40th percentile, 1 = 40th-60th percentile, 2 = above 60th percentile. Percentiles computed over all tongue y-position values across all trials in the session."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking data is aligned to the go cue using the same bin edges as the neural data. The mean y-position per bin is computed at the same temporal resolution (50ms bins, -2.5s to 1.5s), ensuring exact alignment with the neural data.

ii.
```python
ty = extract_tongue_y_for_trial(tongue_ts, tongue_y, gc, BIN_EDGES)
# BIN_EDGES is the same used for neural firing rate computation
```

iii. The tongue data uses the same `BIN_EDGES` as the neural data, ensuring temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing sample_start_times**: If no sample_start is found before the go cue, a fallback tone_onset_rel of -1.85s is used.
- **Missing tongue tracking**: NaN bins default to the middle category (1) for tongue y discretization. Sessions with no valid tongue tracking default all bins to middle category.
- **Missing brain region annotation**: Units with empty or unmappable annotation names are excluded.
- **Sessions with no good units**: Skipped (returns None).
- **Trials outside recording window**: Excluded via `obs_intervals` check.
- **Photostim parsing errors**: Caught with try/except, defaulting to no photostimulation.
- **Sessions with fewer than 2 valid trials**: Skipped.

ii.
```python
# Missing tone onset fallback
if tone_onset_rel is None:
    tone_onset_rel = -1.85

# Missing tongue data
if len(valid_tongue_y) > 0:
    # ... discretize normally
else:
    tongue_y_discrete_trials = [np.ones(N_BINS, dtype=np.int64) for _ in range(len(trial_indices))]

# Photostim parsing error handling
try:
    ps_onset_val = float(ps_onset_str[ti])
    # ...
except (ValueError, TypeError):
    pass

# Missing region annotation
region = map_anno_name_to_region(anno_names[i])
if region is not None:
    unit_regions.append(region)
```

iii. CONVERSION_NOTES.md documents the handling of missing data: "Units with empty or unmappable annotation names excluded (490 units, <1%)."

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading NWB files**: Each file is opened with h5py and multiple large arrays (spike_times, tongue tracking) are loaded into memory.
2. **Computing firing rates**: A nested loop iterates over all units and all trials, calling `compute_firing_rates` for each unit-trial pair. For 400+ units and 500+ trials per session, this is O(n_units * n_trials) histogram operations.
3. **Extracting tongue y-position**: Per-trial extraction with inner loop over bins.

ii.
```python
# Nested loop over units and trials for firing rates
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. No explicit justification given; the code prioritizes correctness and simplicity over performance.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. **Firing rate computation** (lines 492-493): The inner loop over units could be vectorized by concatenating all spike times with unit labels and using a 2D histogram or vectorized binning approach.
2. **Tongue y-position binning** (lines 313-316 in `extract_tongue_y_for_trial`): The per-bin loop computing mean y-position could be vectorized using `np.digitize` or `scipy.stats.binned_statistic`.

ii.
```python
# Inner loop over units (could be vectorized)
for j, st in enumerate(unit_spike_times):
    fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)

# Per-bin loop for tongue y (could use np.digitize)
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
    if np.any(bin_mask):
        tongue_y_binned[b] = np.mean(y_window[bin_mask])
```

iii. No justification given for the loop-based approach.

## 10-c. What processing does the code repeat multiple times?

i. The main repeated processing is:
1. **Spike time slicing**: For each trial, each unit's full spike time array is passed to `compute_firing_rates`, which first masks to the time window. The masking could be precomputed per-trial window.
2. **Tongue timestamp searching**: `np.searchsorted` is called per-trial on the full tongue timestamp array, though the data is already sorted so subsequent trials could narrow the search range.

ii.
```python
# Each trial re-processes all unit spike times
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The code loads spike times once and reuses them, but doesn't optimize for temporal locality across trials.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Key unnecessary processing:
1. **Broadcasting per-trial outputs across time bins**: Choice, outcome, and early_lick are per-trial values broadcast to full (4, 80) arrays per trial. These could be stored as scalar values, as the decoder framework likely handles per-trial vs. time-varying outputs differently.
2. **All tongue tracking data without confidence filtering**: Using DLC tracking data without any confidence threshold may include unreliable points that add noise rather than signal.
3. **Observation window checking**: While useful for correctness, computing `min_obs_end` and `max_obs_start` across all units for each session adds overhead that may not change the trial selection in most cases.

ii.
```python
# Broadcasting per-trial scalars to full time series
np.full(N_BINS, output_choice[i], dtype=np.int64),       # choice
np.full(N_BINS, output_outcome[i], dtype=np.int64),      # outcome
np.full(N_BINS, output_early_lick[i], dtype=np.int64),   # early lick
```

iii. The instructions specify outputs can be "(n_output, n_timepoints) or (n_output)" -- scalar per-trial outputs are allowed but the AI chose to broadcast them. CONVERSION_NOTES.md does not discuss this design choice.
