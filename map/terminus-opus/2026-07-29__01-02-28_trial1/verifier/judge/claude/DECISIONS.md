# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files using `h5py` (not `pynwb`). It lists subject directories under `data/`, then lists NWB files within each. Each file is opened with `h5py.File()` and data is extracted by directly accessing HDF5 groups and datasets.

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

iii. The AI chose h5py over pynwb for direct HDF5 access. The CONVERSION_NOTES.md states: "Used h5py for NWB file reading." No further justification is given for this choice over pynwb.

## 1-b. How are the data split into subjects?

i. The subject ID is taken from the directory name (e.g., `sub-440956`), not from the NWB file's internal subject field. Subjects are accumulated as an ordered list (not sorted) during processing.

ii.
```python
def get_nwb_files():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    ...
    for subj in subjects:
        ...
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                ...
            })
```

```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. No explicit justification given. The directory names correspond to subject IDs in the DANDI archive.

## 1-c. How are the data split into sessions?

i. One NWB file is one session. Sessions are processed sequentially from the file list. Session filtering applies performance criteria: correct rate >= 65% and >= 50 correct trials per side for left and right.

ii.
```python
if correct_rate < MIN_CORRECT_RATE:
    print(f"    Skipping: correct rate too low")
    f.close()
    return None

if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:
    print(f"    Skipping: insufficient correct trials per side")
    f.close()
    return None
```

iii. The CONVERSION_NOTES.md states: "Session selection: >65% correct rate, >=50 correct left and right trials each" under Step 3 and Step 5. These criteria come from the data paper's methods section describing how sessions were selected for the original study. The AI retained 143 of 174 sessions, vs. the 173 sessions reported in the paper.

## 1-d. How are the data split into trials?

i. Trials are read from `intervals/trials` in the NWB file. The trial count is taken from `len(trials['id'])`. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
...
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
```

iii. No explicit justification beyond reading from the standard NWB trials table.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on three criteria: (1) `auto_water == 0`, (2) `free_water == 0`, and (3) neural validity, determined by checking that the trial index is within `is_good_trials.shape[1]` and that the go cue falls within the recorded spike time range (with 1s buffer). The AI does NOT use `obs_intervals` for trial filtering. Additionally, session-level performance filtering is applied (correct rate >= 65%, >= 50 correct trials per side).

ii.
```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
...
neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
```

iii. The CONVERSION_NOTES.md states: "Exclude auto_water and free_water trials" and "Filter trials without neural recording coverage (using is_good_trials and spike time range)." The session-level filtering criteria are stated as from the methods paper.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` (the ragged array indexing). Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

iii. No explicit justification beyond that spike_times is the standard neural data source.

## 2-b. How is the `neural` data processed?

i. For each trial, spike times are windowed relative to the go cue, then histogrammed into 50ms bins using `np.histogram`. Counts are divided by bin width to get firing rates in Hz. The loop iterates per trial, then per unit within each trial.

ii.
```python
def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                               good_indices, trial_indices, ...):
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
    ...
    for trial_idx in trial_indices:
        go_time = go_cue_times[trial_idx]
        fr_trial = np.zeros((n_good, n_bins), dtype=np.float32)
        abs_start = go_time + window_start
        abs_end = go_time + window_end
        for i, unit_spikes in enumerate(good_spike_times):
            idx_lo = np.searchsorted(unit_spikes, abs_start)
            idx_hi = np.searchsorted(unit_spikes, abs_end)
            if idx_hi > idx_lo:
                aligned = unit_spikes[idx_lo:idx_hi] - go_time
                counts = np.histogram(aligned, bins=bin_edges)[0]
                fr_trial[i, :] = counts / bin_width
        firing_rates_list.append(fr_trial)
```

iii. CONVERSION_NOTES states: "Computed firing rates as spike counts / bin_width (Hz)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b'good'` are kept. This is the same QC classifier approach as the reference.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
good_indices = np.where(good_mask)[0]
```

iii. CONVERSION_NOTES states: "Use classifier-based QC: keep only units with classification='good'."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue. For each trial, the go cue time is used to compute absolute window boundaries, spikes within the window are extracted, and their times are expressed relative to the go cue before histogramming.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end = go_time + window_end
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. CONVERSION_NOTES states: "Temporal alignment: go cue = time 0."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins are used, spanning -2.5s to +1.5s relative to the go cue, giving 80 time bins. No rebinning is applied (spikes are directly histogrammed).

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
```

iii. CONVERSION_NOTES states: "50ms non-overlapping bins (task spec, not 40ms sliding from reference)."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents/sample_start_times/timestamps` (tone onsets) and go cue times. The last tone onset before the go cue is selected using `searchsorted`.

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. No explicit justification beyond using the sample_start_times event stream.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the last sample_start_time before the go cue is found. The tone onset is expressed relative to the go cue. Then time_from_tone = bin_centers - tone_onset_rel, which gives the time from tone onset at each bin center.

ii.
```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. No explicit justification. The fallback value of -1.85 is used when no tone onset is found before the go cue.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin_centers array (80 bin centers from -2.475 to 1.475 relative to go cue), so alignment is inherent.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
time_from_tone = bin_centers - tone_onset_rel
```

iii. No explicit justification needed; same time grid is used.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, along with `start_time` and go cue times.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
...
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
```

iii. No explicit justification beyond following the NWB data fields.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostimulation, onset and offset are computed relative to the go cue. A binary time series is created where bins with centers between onset and offset are 1, others 0. Non-stimulated trials have all-zero vectors.

ii.
```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
...
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. No explicit justification.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Uses the same bin_centers as the neural data, ensuring alignment.

ii.
```python
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. Same time grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived solely from `trial_instruction` (left/right). It does NOT use `outcome` to determine actual lick direction. All trials are assigned either left (0) or right (1) based on the instruction, even ignore trials where the animal never licked.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. No explicit justification. CONVERSION_NOTES maps it as: "trial_instruction -> output[0] choice: left=0, right=1."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping: left instruction -> 0, right instruction -> 1. The value is repeated across all 80 time bins. There is no "no lick" category; `output_values` lists only `['left', 'right']`.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
output_trial[0, :] = choice
```

```python
'output_values': [
    ['left', 'right'],
    ...
]
```

iii. No justification for omitting the "no lick" category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains `b'hit'`, `b'miss'`, and `b'ignore'`.

ii.
```python
out = outcome[trial_idx]
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0
```

iii. No explicit justification.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
output_trial[1, :] = outcome_val
```

iii. No explicit justification.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains `b'early'` or `b'no early'`.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. No explicit justification.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: `b'early'` -> 1, `b'no early'` -> 0. Repeated across all 80 time bins.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
output_trial[2, :] = early
```

iii. No explicit justification.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains (x, y, likelihood) per frame with associated timestamps.

ii.
```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
```

iii. No explicit justification.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood > 0.1 are considered valid (threshold 0.1, not the reference's 0.5). For each trial, tongue y is averaged within each 50ms bin. Session-level percentiles (40th, 60th) are computed from all valid y values across all trial bins (not from session-wide bin means). Discretization: below p40 -> 0, between p40 and p60 -> 1, above p60 -> 2. Bins with NaN (no valid frames) default to category 1 (middle category), rather than a separate "not visible" class.

ii.
```python
high_conf = lk_slice > 0.1
...
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
...
tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default to category 1
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

```python
if len(all_tongue_y) > 0:
    all_tongue_y_arr = np.array(all_tongue_y)
    p40 = np.percentile(all_tongue_y_arr, 40)
    p60 = np.percentile(all_tongue_y_arr, 60)
```

iii. CONVERSION_NOTES states: "Tongue y percentiles: Computed per-session from all valid (likelihood>0.1) tracking data." The choice of 0.1 threshold is not justified vs the 0.5 used in the reference.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th to 60th percentile), 2 (above 60th percentile). Bins with no valid tongue data default to category 1 (middle). There is no "not visible" (category 3) as specified in the instructions.

ii.
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default to category 1
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

```python
'output_values': [
    ...
    ['below_p40', 'p40_to_p60', 'above_p60'],
]
```

iii. No justification for omitting the "not visible" category. The instructions explicitly state "3: not visible" as a category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are used directly (same session-absolute clock). For each trial, frames within the go cue-aligned window are found via searchsorted, and averaged into the same 50ms bins as neural data.

ii.
```python
idx_start = np.searchsorted(tongue_timestamps, abs_start)
idx_end = np.searchsorted(tongue_timestamps, abs_end)
...
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. No explicit justification; same time grid ensures alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- Sessions with no good units: skipped (return None).
- Trials without neural coverage: filtered via spike time range and is_good_trials check.
- Tongue tracking with no valid frames in a bin: defaults to category 1 (middle) rather than a distinct "not visible" class.

ii.
```python
if n_good == 0:
    print(f"    Skipping: no good units")
    f.close()
    return None
```

```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default to category 1
```

iii. CONVERSION_NOTES states under Step 10: "Missing tongue tracking: Defaults to middle category (1) for NaN values."

## 10-a. What are the most time-consuming steps of the code?

i. The firing rate computation is the bottleneck, with the AI's code looping per trial AND per unit (doubly nested). The CONVERSION_NOTES estimates ~7s per session average, ~16 minutes total for 143 sessions. The firing rate computation alone takes 3-14s per session.

ii.
```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...
```

iii. CONVERSION_NOTES states firing rate computation takes 3-14s per session.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing rate computation has a doubly nested loop (per trial, per unit). The reference code vectorizes the trial dimension by flattening all trial edges and using a single searchsorted per unit. The tongue tracking also loops per trial and per bin.

ii.
```python
# Per trial, per unit loop:
for trial_idx in trial_indices:
    for i, unit_spikes in enumerate(good_spike_times):
        ...

# Per trial, per bin loop for tongue:
for trial_idx in trial_indices:
    for b in range(n_bins):
        ...
```

iii. No justification for the nested loop structure. CONVERSION_NOTES mentions "searchsorted instead of full array masking: ~2x speedup" but doesn't address the doubly-nested loop issue.

## 10-c. What processing does the code repeat multiple times?

i. The per-trial loop for constructing outputs iterates over trials a second time after tongue tracking is already computed per trial. The photostim and time_from_tone inputs are also computed in a separate per-trial loop. These could be combined but the redundancy is minor.

ii.
```python
# Loop 1: inputs
for trial_idx in trial_indices:
    # compute time_from_tone, photostim

# Loop 2: tongue tracking
for trial_idx in trial_indices:
    # compute tongue y per trial

# Loop 3: outputs
for i, trial_idx in enumerate(trial_indices):
    # construct output array
```

iii. No explicit discussion of redundant processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The session-level performance statistics (correct_rate, correct_left, correct_right) are computed for session filtering but are only stored as `correct_rate` in the result dict and not included in the final output. The `is_good_trials` matrix is loaded to determine neural validity but is otherwise unused. These are minor.

ii.
```python
correct_rate = hits / (hits + misses)
correct_left = np.sum(...)
correct_right = np.sum(...)
```

iii. No explicit discussion.
