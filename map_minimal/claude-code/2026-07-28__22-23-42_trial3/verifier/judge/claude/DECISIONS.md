# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from the `data/sub-*/sub-*.nwb` directory using `h5py` (not `pynwb`). Each NWB file corresponds to one session. The AI iterates over all sorted NWB files, calling `process_session()` on each, and collects results into lists.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
for nwb_path in nwb_files:
    result = process_session(nwb_path)
```

```python
f = h5py.File(nwb_path, 'r')
# ...
n_trials = len(f['intervals/trials/id'][:])
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
```

iii. The AI's CONVERSION_NOTES.md states: "174 NWB files across 28 subjects in data/sub-*/sub-*.nwb. Each NWB file represents one recording session." The AI uses h5py for direct HDF5 access rather than pynwb.

## 1-b. How are the data split into subjects?

i. The AI extracts the subject ID from the NWB filename by parsing the path: removing `sub-` prefix and splitting at `_ses-`. This gives numeric IDs like `440956`. Unique subjects are sorted and an index array maps sessions to subjects.

ii.
```python
basename = os.path.basename(nwb_path)
subject_id = basename.split('_ses-')[0].replace('sub-', '')
```

```python
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The CONVERSION_NOTES.md mentions 28 subjects. The AI derives subject IDs from filenames rather than reading `nwb.subject.subject_id` from within the file.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. The AI processes each file independently. Session identity is tracked via the basename of the file.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
# ...
all_session_files.append(result['session_file'])
```

```python
'session_file': basename,
```

iii. The CONVERSION_NOTES.md states each NWB file represents one recording session. 144 sessions are kept after filtering (30 skipped due to session-level quality filters).

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The number of trials is determined by `intervals/trials/id`, and the AI verifies that go cue count matches trial count.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
# ...
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
# ...
assert len(go_cue_times) == n_trials
```

iii. The AI uses the trials table directly. No re-derivation of trial boundaries is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple trial filters: (1) excludes `auto_water` trials, (2) excludes `free_water` trials, (3) excludes trials outside the neural recording observation window (where go cue + analysis window falls outside obs_intervals). Additionally, the AI applies **session-level filters**: sessions with performance <= 65% on control trials or fewer than 50 correct trials per direction are dropped entirely.

ii.
```python
# Session-level filtering
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))
# ...
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
if performance <= MIN_PERFORMANCE:
    return None
if control_hits_left < MIN_CORRECT_PER_DIRECTION or control_hits_right < MIN_CORRECT_PER_DIRECTION:
    return None

# Trial-level filtering
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```

```python
# Observation window check
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
```

iii. The CONVERSION_NOTES.md states: "Performance threshold: >65% correct on control trials" and "Minimum correct trials: >=50 correct left AND >=50 correct right trials", citing the methods text. The AI also states it excluded auto_water and free_water trials "as in reference code."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (with `units/spike_times_index` for ragged indexing). Go cue times from `acquisition/BehavioralEvents/go_start_times/timestamps` are used to place bin edges.

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]

unit_spike_times = []
for ui in unit_indices:
    start_idx = 0 if ui == 0 else spike_times_index[ui - 1]
    end_idx = spike_times_index[ui]
    unit_spike_times.append(spike_times_data[start_idx:end_idx])
```

iii. The AI uses spike times as the raw neural data, which is the only neural representation available.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram`, then divided by bin width to get firing rates in Hz. This is done per-unit, per-trial in a nested loop.

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
    bin_width = bin_edges[1] - bin_edges[0]
    return counts.astype(np.float64) / bin_width
```

```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
    neural_trials.append(fr_matrix)
```

iii. The CONVERSION_NOTES.md confirms: "50ms non-overlapping bins... Spike counts divided by bin width (0.05s) to get rates in Hz."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Additionally, units whose `anno_name` cannot be mapped to one of the 14 predefined brain region categories are excluded. A session with no remaining units is dropped.

ii.
```python
classification = np.array([c.decode() if isinstance(c, bytes) else c
                           for c in f['units/classification'][:]])
good_mask = classification == 'good'

unit_regions = []
unit_indices = []
for i in range(len(classification)):
    if not good_mask[i]:
        continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_regions.append(region)
        unit_indices.append(i)
```

iii. The CONVERSION_NOTES.md states: "Classifier-based QC (classification='good')" and notes "Units with empty or unmappable annotation names excluded (490 units, <1%)".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go cue time to get relative times, then histogramming into bins defined by BIN_EDGES (which are relative to the go cue at t=0).

ii.
```python
def compute_firing_rates(spike_times, go_cue_time, bin_edges):
    rel_times = spike_times - go_cue_time
    mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
    rel_times = rel_times[mask]
    counts, _ = np.histogram(rel_times, bins=bin_edges)
```

iii. Alignment is straightforward since all timestamps share the same clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, 80 bins spanning -2.5s to +1.5s relative to go cue. Bin edges are computed with `np.linspace`. No rebinning is applied; spike times are directly binned.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
```

iii. The CONVERSION_NOTES.md states 50ms bins as specified in the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (tone onset timestamps) and `go_cue_times`. The tone onset for each trial is determined as the last `sample_start` before the go cue.

ii.
```python
sample_start_times = f['acquisition/BehavioralEvents/sample_start_times/timestamps'][:]
```

```python
def get_tone_onset_relative_to_go(sample_start_times, go_cue_time):
    before = sample_start_times[sample_start_times < go_cue_time]
    if len(before) > 0:
        return before[-1] - go_cue_time
    return None
```

iii. The CONVERSION_NOTES.md explains: "For trials with early lick replays, uses the last sample_start before the go cue."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset relative to the go cue is computed, then subtracted from BIN_CENTERS to get time from tone onset at each bin center. If no sample_start is found before the go cue, a fallback value of -1.85s is used.

ii.
```python
tone_onset_rel = get_tone_onset_relative_to_go(sample_start_times, gc)
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel  # time since tone onset at each bin
```

iii. The AI notes the typical value at go cue is ~1.85s (0.65s sample + 1.2s delay).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same BIN_CENTERS array (bin centers relative to go cue), so alignment is inherent.

ii.
```python
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2
# ...
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. Using the same bin center grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` and go cue times for coordinate conversion.

ii.
```python
ps_onset_str = np.array([p.decode() if isinstance(p, bytes) else p
                        for p in f['intervals/trials/photostim_onset'][:]])
ps_dur_str = np.array([p.decode() if isinstance(p, bytes) else p
                      for p in f['intervals/trials/photostim_duration'][:]])
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. The CONVERSION_NOTES.md explains photostim_onset is relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The onset is converted from trial-relative to go-cue-relative coordinates. A binary array marks bins where photostim is on (bin center falls between onset and offset).

ii.
```python
if ps_onset_str[ti] != 'N/A':
    ps_onset_val = float(ps_onset_str[ti])
    ps_dur = float(ps_dur_str[ti])
    ps_abs_onset = trial_start_times[ti] + ps_onset_val
    ps_start_rel = ps_abs_onset - gc
    ps_end_rel = ps_start_rel + ps_dur
    photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                       (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The AI correctly handles the coordinate conversion and marks it as binary.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim onset/offset are expressed relative to the go cue, and the same BIN_CENTERS array is used for comparison, ensuring alignment with neural bins.

ii.
```python
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                   (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. Same bin grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from the `trial_instruction` column, mapping 'left' to 0 and 'right' to 1. It does NOT use the outcome to determine the actual lick direction.

ii.
```python
# --- Output: choice ---
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
output_choice.append(choice)
```

iii. The CONVERSION_NOTES.md states: "Based on trial_instruction field (which port the animal should lick)." This is the instructed direction, not the actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping: 'left' -> 0, 'right' -> 1. No handling of 'ignore' trials (where the animal didn't lick). The value is broadcast across all time bins. Only 2 output values defined: ['left', 'right'].

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

```python
'output_values': [
    ['left', 'right'],           # choice: 0=left, 1=right
    ...
]
```

iii. The AI does not account for the difference between instructed and actual lick direction on miss trials, nor does it handle the no-lick case for ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column in the trials table, holding strings 'ignore', 'miss', 'hit'.

ii.
```python
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. Direct from trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Broadcast across all time bins.

ii.
```python
if outcomes[ti] == 'ignore':
    outcome = 0
elif outcomes[ti] == 'miss':
    outcome = 1
else:  # hit
    outcome = 2
```

iii. Straightforward mapping matching the instructions.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column in the trials table, holding strings 'early' and 'no early'.

ii.
```python
early_lick = np.array([e.decode() if isinstance(e, bytes) else e
                      for e in f['intervals/trials/early_lick'][:]])
```

iii. Direct from trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) or 1 (early). Broadcast across time bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
```

iii. Straightforward mapping matching the instructions.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data`, column 1 (y-position), with corresponding timestamps.

ii.
```python
tongue_ts = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]  # y-position
```

iii. The CONVERSION_NOTES.md confirms: "Side-view camera tracking from Camera0_side_TongueTracking, Shape (N, 3): [x, y, likelihood]."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI does NOT apply confidence/likelihood filtering on the tongue tracking data. All tracking values are used regardless of confidence. The raw y-positions are averaged per bin for each trial. Percentiles (40th and 60th) are computed over all non-NaN tongue y values from all trial windows concatenated (raw frame values, not bin means). Values are discretized into 3 classes: 0 (<p40), 1 (p40-p60), 2 (>p60). Bins with no data default to class 1 (middle).

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    # No confidence filtering - uses all tracking data
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
        if np.any(bin_mask):
            tongue_y_binned[b] = np.mean(y_window[bin_mask])
    return tongue_y_binned
```

```python
# Percentiles from concatenated raw values across trials
all_tongue_y_vals = np.concatenate(tongue_y_all_trials)
valid_tongue_y = all_tongue_y_vals[~np.isnan(all_tongue_y_vals)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)

# Discretization
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The CONVERSION_NOTES.md explicitly states: "All tracking data used (no confidence filtering) for percentile computation and discretization."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th-60th percentile), 2 (above 60th percentile). No "not visible" class is defined. Bins with no tongue data default to class 1 (middle). The percentiles are computed from raw frame values concatenated across all trial windows (not from session-wide bin means).

ii.
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
valid_mask = ~np.isnan(ty)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

```python
'output_values': [
    ...
    ['low', 'middle', 'high'],   # tongue_y: 0=<p40, 1=p40-p60, 2=>p60
],
```

iii. The AI uses only 3 classes; bins without visible tongue data default to the middle category rather than having a distinct "not visible" class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are searched using the same bin edges (relative to go cue) used for neural data. Mean y-position per bin is computed from frames falling within each bin's time window.

ii.
```python
def extract_tongue_y_for_trial(tongue_timestamps, tongue_y, go_cue_time, bin_edges):
    t_abs_start = go_cue_time + bin_edges[0]
    t_abs_end = go_cue_time + bin_edges[-1]
    idx = np.searchsorted(tongue_timestamps, [t_abs_start, t_abs_end])
    ts_rel = ts_window - go_cue_time
    for b in range(n_bins):
        bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. Same bin grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Sessions with no good units with valid regions are skipped. (2) Trials outside recording observation windows are excluded. (3) For tongue tracking, bins with no data get NaN which defaults to class 1 (middle). (4) If no sample_start is found before a go cue, a fallback tone onset of -1.85s is used. (5) Photostim parsing errors are silently caught with try/except.

ii.
```python
if len(unit_indices) == 0:
    return None

if tone_onset_rel is None:
    tone_onset_rel = -1.85

try:
    ps_onset_val = float(ps_onset_str[ti])
    # ...
except (ValueError, TypeError):
    pass

ty_disc = np.ones(N_BINS, dtype=np.int64)  # default to middle category
```

iii. The AI handles missing data through defaults and skipping, though the fallback tone onset and tongue defaulting to middle are notable choices.

## 10-a. What are the most time-consuming steps of the code?

i. The nested per-unit, per-trial loop for computing firing rates is the most time-consuming step. Each trial calls `compute_firing_rates` for each unit separately, involving subtraction, masking, and histogramming per call. Reading each NWB file with h5py also takes time.

ii.
```python
for ti in trial_indices:
    gc = go_cue_times[ti]
    fr_matrix = np.zeros((n_units, N_BINS), dtype=np.float64)
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The AI does not discuss performance in CONVERSION_NOTES.md.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial, per-unit firing rate computation loop is doubly nested (units x trials). Each call to `compute_firing_rates` does masking and histogramming independently. The tongue tracking per-bin loop could also be vectorized. The reference code vectorizes the trial dimension by flattening all trial edges into one array.

ii.
```python
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

```python
for b in range(n_bins):
    bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. No justification provided for this design.

## 10-c. What processing does the code repeat multiple times?

i. The spike time masking (`rel_times >= bin_edges[0]`) is repeated for every trial of every unit. The tongue timestamp searching is done per-bin rather than vectorized. String decoding is done for every column separately with the same pattern.

ii.
```python
# This pattern is repeated for outcomes, early_lick, instructions, etc.
outcomes = np.array([o.decode() if isinstance(o, bytes) else o
                    for o in f['intervals/trials/outcome'][:]])
```

iii. No justification provided.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session-level performance metrics (control trial hit rates, performance percentage) for session filtering. These statistics are not saved in the output data. The auto_water filter is applied but auto_water trials may already be a subset of what's handled by obs_intervals.

ii.
```python
is_control = ((early_lick == 'no early') &
              (auto_water == 0) &
              (free_water == 0) &
              (ps_onset_str == 'N/A'))
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
```

iii. The session filtering adds overhead and also reduces the dataset from 173 to 144 sessions.
