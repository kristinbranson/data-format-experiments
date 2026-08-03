# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `data/sub-*/` directories using `h5py` (not `pynwb`). It lists subject directories, then iterates over NWB files within each, collecting file paths. Each file is opened with `h5py.File()` and fields are read directly from the HDF5 structure.

ii.
```python
def get_nwb_files():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                'path': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file
            })
    return all_files
```

```python
f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. The AI chose h5py over pynwb for direct HDF5 access. From CONVERSION_NOTES.md: "Used h5py for NWB file reading." The AI noted 174 NWB files across 28 subjects, matching the dandiset.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the directory name (`sub-XXXXXX`). The AI passes the subject directory name as `subject_id` to the processing function. Unique subjects are collected in order of first appearance.

ii.
```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(nwb_info['path'], nwb_info['subject'], ...)
```
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. From CONVERSION_NOTES.md: "28 subjects (sub-440956 through sub-484677)." The AI uses the directory name `sub-XXXXXX` as the subject identifier rather than reading `nwb.subject.subject_id` from within the file.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Files are sorted within each subject directory. Sessions that don't meet behavioral criteria (correct rate, correct trials per side) or that have no good units are skipped.

ii.
```python
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```

iii. From CONVERSION_NOTES.md: "174 NWB files total" and "143 sessions pass" after filtering. The AI applies session-level behavioral selection criteria that filter out 31 sessions.

## 1-d. How are the data split into trials?

i. Trials come from `intervals/trials` in each NWB file. The AI reads trial-level columns (trial_instruction, outcome, early_lick, etc.) and go-cue times from `BehavioralEvents/go_start_times`.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. The AI reads trials directly from the NWB trials table and uses `go_start_times` for temporal alignment. No assertion is made that the number of go cues equals the number of trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by three criteria: (1) `auto_water == 0`, (2) `free_water == 0`, and (3) neural validity based on spike time range coverage. Additionally, entire sessions are filtered by behavioral performance criteria: correct rate >= 65% and at least 50 correct trials per side (left and right).

ii.
```python
# Trial-level filtering
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid

# Session-level filtering
if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    return None
```

```python
# Neural validity check
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
```

iii. From CONVERSION_NOTES.md: "Exclude auto_water and free_water trials" and "Session selection: >65% correct rate, >=50 correct left and right trials each." The AI applied session selection criteria from the methods paper that were used for the paper's own analyses, but the reference solution does not apply these criteria because the decoder task doesn't require them.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times) and `units/spike_times_index` (ragged array offsets). Only units with `classification == b'good'` are used.

ii.
```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
classification = f['units']['classification'][:]
good_mask = classification == b'good'
```

iii. From CONVERSION_NOTES.md: "QC filtering uses classifier-based 'good' unit labels (NWB: classification field)."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning -2.5 to +1.5 s relative to the go cue. For each unit and each trial, spikes are found within the window using `searchsorted`, then histogrammed with `np.histogram`, and divided by bin width to get firing rates in Hz.

ii.
```python
def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                               good_indices, trial_indices, ...):
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    for trial_idx in trial_indices:
        go_time = go_cue_times[trial_idx]
        for i, unit_spikes in enumerate(good_spike_times):
            idx_lo = np.searchsorted(unit_spikes, abs_start)
            idx_hi = np.searchsorted(unit_spikes, abs_end)
            if idx_hi > idx_lo:
                aligned = unit_spikes[idx_lo:idx_hi] - go_time
                counts = np.histogram(aligned, bins=bin_edges)[0]
                fr_trial[i, :] = counts / bin_width
```

iii. From CONVERSION_NOTES.md: "Computed firing rates as spike counts / bin_width (Hz)." The approach is functionally equivalent to the reference's `searchsorted` + `diff` method, but uses `np.histogram` per unit per trial rather than vectorizing across trials.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b'good'` are kept. Sessions with no good units are dropped.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
if n_good == 0:
    return None
```

iii. From CONVERSION_NOTES.md: "Use classifier-based QC: keep only units with classification='good'". This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the go-cue time from each spike time. Bin edges are defined relative to the go cue (-2.5 to +1.5 s), and spikes within the window are histogrammed using those relative bin edges.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end = go_time + window_end
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. The go-cue alignment matches both the instructions and the reference approach.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning -2.5 to +1.5 s. No rebinning is applied; spikes are binned directly from the raw spike times.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
```

iii. Matches the task specification exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset timestamps) and `go_start_times` (go cue timestamps).

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. From CONVERSION_NOTES.md: "Time from tone onset" is listed as input[0].

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last `sample_start_times` before the go cue is found using `searchsorted`. The tone onset relative to the go cue is computed, then time from tone is `bin_centers - tone_onset_rel`.

ii.
```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. The AI finds the last sample onset before the go cue, which handles early-lick retrials correctly (same logic as the reference). The fallback value of -1.85 is used if no sample start is found, though this should never happen in practice.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers derived from the go-cue-aligned time axis, so alignment is inherent.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
```

iii. The bin centers are computed the same way for neural and input data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, plus `start_time` and go cue times.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostim (onset != 'N/A'), the onset is converted from trial-relative to go-cue-relative time. A binary time series is created where bin centers falling within [onset, onset+duration) are set to 1.

ii.
```python
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The logic matches the reference: onset is relative to trial start, converted to go-cue-relative, and a binary mask is created based on bin centers.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim on/off times are expressed relative to the go cue, matching the bin center axis used for neural data.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. Alignment is through the shared go-cue-relative time axis.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` only, encoding the instructed lick direction (left=0, right=1). It does NOT use `outcome` to determine the actual lick direction.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. From CONVERSION_NOTES.md variable mapping: "trial_instruction -> output[0] choice: left=0, right=1." The AI treats choice as the instructed side, not the actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The instructed side is mapped to 0 (left) or 1 (right). There is no third class for "no lick" (ignore trials). The value is repeated across all 80 time bins.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
output_trial[0, :] = choice
```

```python
'output_values': [
    ['left', 'right'],  # only 2 values, no "no lick"
    ...
]
```

iii. The AI uses only 2 classes for choice. The reference uses 3 classes (left, right, no lick) and derives the actual lick direction from instruction x outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which stores strings `b'ignore'`, `b'miss'`, `b'hit'`.

ii.
```python
out = outcome[trial_idx]
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
```

iii. Direct mapping from the trials table, matching the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
output_trial[1, :] = outcome_val
```

iii. Matches the reference encoding exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which stores `b'early'` or `b'no early'`.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. Direct mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: `b'early'` -> 1, anything else -> 0. The value is repeated across all time bins.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
output_trial[2, :] = early
```

iii. Matches the reference encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, which contains (x, y, likelihood) data with timestamps.

ii.
```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI filters tongue tracking frames by likelihood > 0.1 (different from reference's < 0.5). Valid tongue y values are collected from all trials, and the 40th and 60th percentiles are computed from individual frame values (not bin means). Per-trial tongue y is averaged within each 50 ms bin, then discretized against these percentiles. NaN bins (no valid frames) default to class 1 (middle category).

ii.
```python
# Likelihood threshold
high_conf = lk_slice > 0.1   # reference uses < 0.5 to exclude

# Percentile computation from individual values
all_tongue_y_arr = np.array(all_tongue_y)
p40 = np.percentile(all_tongue_y_arr, 40)
p60 = np.percentile(all_tongue_y_arr, 60)

# Discretization
tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default to 1 (middle)
tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. From CONVERSION_NOTES.md: "Tongue y percentiles: Computed per-session from all valid (likelihood>0.1) tracking data." The likelihood threshold (0.1 vs 0.5), the percentile computation basis (individual values vs bin means), and the handling of missing data (default to class 1 vs separate class 3) all differ from the reference.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th to 60th percentile), 2 (above 60th percentile). Bins with no valid tongue data default to category 1 (middle). There is no fourth "not visible" category. The boundary condition uses `<=` for the 60th percentile, differing from the reference's `np.digitize` which uses `<`.

ii.
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default = 1 (middle)
tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

```python
'output_values': [
    ...
    ['below_p40', 'p40_to_p60', 'above_p60'],  # only 3 classes
]
```

iii. The reference uses 4 classes (including "not visible" class 3 for bins with no tracked tongue), while the AI uses only 3 classes and defaults missing data to the middle category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are on the same session-absolute clock. For each trial, frames within the go-cue-relative window are found by `searchsorted`, and tongue y is averaged within each 50 ms bin defined by the same bin edges as neural data.

ii.
```python
abs_start = go_time + window_start - bin_width   # slight padding
abs_end = go_time + window_end + bin_width
idx_start = np.searchsorted(tongue_timestamps, abs_start)
idx_end = np.searchsorted(tongue_timestamps, abs_end)
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. Alignment is correct via the shared go-cue-relative time axis.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled:
- Sessions with no good units are skipped.
- Trials without neural coverage (based on spike time range) are excluded.
- Tongue tracking bins with no valid frames default to class 1 (middle category) rather than a separate "not visible" class.
- Sessions that don't meet behavioral performance criteria are skipped.
- auto_water and free_water trials are excluded.

ii.
```python
if n_good == 0:
    return None

neural_valid[t_idx] = True  # only if spike coverage sufficient

tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # NaN -> class 1
```

iii. From CONVERSION_NOTES.md Step 10: "Missing tongue tracking: Defaults to middle category (1) for NaN values." The reference instead uses a fourth class (3 = "not visible") for bins with no tongue tracking data.

## 10-a. What are the most time-consuming steps of the code?

i. Based on CONVERSION_NOTES.md, firing rate computation dominates at 3-14s per session. Data loading takes 0.2-0.5s per session. Total estimated ~16 minutes for full conversion.

ii.
```python
# The nested loop: per trial, per unit
for trial_idx in trial_indices:
    for i, unit_spikes in enumerate(good_spike_times):
        idx_lo = np.searchsorted(unit_spikes, abs_start)
        idx_hi = np.searchsorted(unit_spikes, abs_end)
        ...
        counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. From CONVERSION_NOTES.md: "Firing rate computation: 3-14s" per session, "Total: ~7s avg, ~16 min" estimated total.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two nested loops remain in the firing rate computation: an outer loop over trials and an inner loop over units. The reference vectorizes the trial dimension by flattening all bin edges across trials into one array and calling `searchsorted` once per unit. The tongue tracking also has a nested loop over trials and bins.

ii.
```python
# AI: loop over trials AND units (doubly nested)
for trial_idx in trial_indices:
    for i, unit_spikes in enumerate(good_spike_times):
        ...

# Reference: loop only over units, trials vectorized
edges = (go[:, None] + REL_EDGES[None, :]).ravel()
for r, u in enumerate(good):
    pos = np.searchsorted(s, edges).reshape(n_trials, N_BINS + 1)
```

```python
# Tongue: loop over trials AND bins
for trial_idx in trial_indices:
    for b in range(n_bins):
        bin_mask = ...
```

iii. The doubly-nested loop in firing rate computation is the main performance bottleneck and could have been vectorized across trials as the reference does.

## 10-c. What processing does the code repeat multiple times?

i. The code re-extracts spike times for good units once at the start of `compute_firing_rates_fast`, which is efficient. However, the per-trial loop recomputes `abs_start` and `abs_end` for each trial, and the tongue tracking function recomputes bin edges per call. These are minor repeated computations.

ii.
```python
# Pre-extracted once (good)
good_spike_times = []
for unit_idx in good_indices:
    good_spike_times.append(spike_times_flat[starts[unit_idx]:ends[unit_idx]])
```

iii. No major repeated computation beyond the loop overhead.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes session selection statistics (correct_rate, correct_left, correct_right) and uses them to filter sessions. This filtering is unnecessary and actually discards valid data. The `is_good_trials` matrix is read but only used for its shape, not its content. Brain region extraction via JSON parsing is more complex than needed compared to using `anno_name` directly.

ii.
```python
# Session selection criteria not needed for decoder task
correct_rate = hits / (hits + misses)
if correct_rate < MIN_CORRECT_RATE:
    return None
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    return None

# is_good_trials read but only shape used
n_recorded_trials = f['units']['is_good_trials'].shape[1]
```

iii. The session selection criteria from the methods paper were applied even though the decoder task doesn't require them, resulting in 143 sessions instead of 173.
