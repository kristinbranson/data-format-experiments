# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files stored in `/app/data/`. Each subject has a directory (`sub-XXXXXX`), and each session is a single `.nwb` file within that directory. The AI uses `pynwb.NWBHDF5IO` to read each file. It iterates over all subjects and their NWB files sequentially, processing one session at a time.

ii.
```python
def get_nwb_files(data_dir):
    """Get list of all NWB files organized by subject."""
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files
```

```python
io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The AI identified that data is in NWB format (not the .mat format the reference code uses) and uses pynwb to load it. This is documented in CONVERSION_NOTES.md Step 1: "Reference code processes .mat files from DataJoint export; our data is in NWB format."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the directory names under `/app/data/` (e.g., `sub-440956`). The AI maintains a `subjects_seen` dictionary to track unique subjects and assign indices. The final `subjects` list preserves the sorted order of subject directories.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
# ...
if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
```

iii. The AI noted 28 subjects in the data directory, matching the papers ("28 mice"). Subject IDs are preserved from directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes each NWB file independently via the `process_session()` function. Sessions are accumulated into a list, with sessions that have zero good units skipped.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    if result is None:
        print("SKIPPED (no good units or trials)")
        continue
    all_sessions.append(result)
```

iii. CONVERSION_NOTES.md Step 4 notes: "174 NWB - 1 with 0 good units = 173. Matches." The AI found 174 NWB files total but 1 session had 0 good units, yielding 173 sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB trials table (`nwb.trials`). For each session, all trials are enumerated and then filtered (see 1-e). Each trial is identified by its index in the trials table. Trial-level events (go cue, sample onset, etc.) are accessed from `BehavioralEvents`.

ii.
```python
n_trials_total = len(nwb.trials)
# ...
for ti in trial_indices:
    go_cue = go_start_times[ti]
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The AI uses the NWB trials table which contains one row per trial, with associated metadata (outcome, instruction, early_lick, etc.).

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out trials where `auto_water == 1` or `free_water == 1`. It also excludes trials that fall outside the recording period (using `obs_intervals`). Crucially, the AI keeps early lick trials (since early_lick is a decoder output), photostimulation trials (since photostim is a decoder input), and ignore/no-response trials (since outcome includes ignore=0).

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0)
# Intersect with valid recording trials
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

if len(trial_indices) < 2:
    io.close()
    return None
```

iii. CONVERSION_NOTES.md Step 4: "For our decoder: keep early lick (output), photostim (input), ignore (outcome=0). Exclude only auto_water and free_water." The reference code's `get_regular_trial_mask` excludes early lick, no response, and stimulation trials, but those are needed as decoder inputs/outputs per the task instructions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']` — the spike times for each unit that passes quality control (`classification == 'good'`). Only units with valid brain region annotations (`anno_name` mappable to one of 14 coarse regions) are included.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
# ...
good_unit_indices = np.where(good_mask)[0]
# ...
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. CONVERSION_NOTES.md Step 1: "In NWB: `classification == 'good'` corresponds to QC classifier pass." Step 5: "classification=='good' only (all have anno_name)."

## 2-b. How is the `neural` data processed?

i. Spike times are binned into non-overlapping 50ms time bins spanning -2.5s to 1.5s relative to the go cue (80 bins total). Spike counts per bin are converted to firing rates (Hz) by dividing by the bin width. Each neuron's spikes are filtered to the trial's time window, assigned to bins via integer division, and accumulated using `np.add.at`.

ii.
```python
def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_neurons = len(spike_times_list)
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end
    for i, st in enumerate(spike_times_list):
        if len(st) == 0:
            continue
        mask = (st >= abs_start) & (st < abs_end)
        spikes_in_window = st[mask]
        if len(spikes_in_window) == 0:
            continue
        bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)
        np.add.at(fr[i], bin_indices, 1.0)
    fr /= bin_size
    return fr
```

iii. CONVERSION_NOTES.md Step 5: "Spike binning: 50ms non-overlapping bins (per task spec), NOT 40ms/3.4ms stride from reference." The task instructions require 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by: (1) `classification == 'good'` (QC classifier pass), and (2) having a valid brain region annotation (`anno_name` that maps to one of 14 coarse regions). No firing rate threshold is applied. The AI does not apply the 2 Hz firing rate threshold mentioned in the method paper.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
# ...
for i, anno in enumerate(anno_names):
    region = map_anno_to_region(str(anno))
    if region is None:
        valid_neuron_mask[i] = False
```

iii. CONVERSION_NOTES.md Step 5: "classification=='good' only... No firing rate threshold (2 Hz was analysis-specific)." The AI determined the 2 Hz threshold was used only for the video prediction analysis in the method paper, not for general preprocessing.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the Go cue onset. The go cue time for each trial is obtained from `BehavioralEvents` → `go_start_times`. Spike times are absolute, so the window is computed as `[go_cue + T_START, go_cue + T_END]` where T_START=-2.5s and T_END=1.5s.

ii.
```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
# ...
go_cue = go_start_times[ti]
fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The task instructions specify: "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue." CONVERSION_NOTES.md Step 5 confirms: "Time window: -2.5s to 1.5s relative to go cue (per task spec)."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50 ms (non-overlapping bins), yielding 80 time bins over the 4-second window. No rebinning is applied — spikes are directly binned at 50ms resolution. The reference code uses 40ms bins with 3.4ms stride, but the task spec requires 50ms bins.

ii.
```python
BIN_SIZE_S = 0.05  # 50 ms bins
T_START = -2.5     # seconds before go cue
T_END = 1.5        # seconds after go cue
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. CONVERSION_NOTES.md Step 3: "Our task requires: 50ms bins, aligned -2.5s to 1.5s relative to go cue → 80 time bins." Step 5: "50ms non-overlapping bins (per task spec), NOT 40ms/3.4ms stride from reference."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (from `BehavioralEvents`) and `go_start_times`. The `sample_start_times` timestamps contain the onset times of the sample epoch (tone). For early lick trials with replays, the AI takes the last sample start before the go cue.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
# ...
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

iii. CONVERSION_NOTES.md Step 5: "Tone onset: Last sample_start_time before each trial's go cue (accounting for early lick replays)."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial: (1) Find the last `sample_start_time` before the go cue. (2) Compute bin centers relative to the go cue. (3) Subtract the tone onset time (relative to go cue) from each bin center to get "time from tone onset." If no sample_start_time is found before the go cue, a default of `go_cue - 1.85` is used (based on typical sample epoch timing).

ii.
```python
def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time  # tone onset relative to go cue
    time_from_tone = bin_centers - tone_rel  # time since tone onset
    return time_from_tone.astype(np.float32)

def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]
```

iii. The fallback to `go_cue - 1.85` is documented in CONVERSION_NOTES.md Step 3: "Sample epoch (tone): 3 tones x 150ms + 100ms gaps = 650ms, starts ~-1.85s."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same bin centers as the neural data (50ms bins, centers at bin_start + 25ms). This ensures temporal alignment: both neural firing rates and the time-from-tone input share the same 80-timepoint grid.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
# Same bin structure as neural data
```

iii. The AI uses the same `t_start`, `t_end`, and `bin_size` parameters for both neural and input computations, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `photostim_start_times` and `photostim_stop_times` from `BehavioralEvents`. These contain the absolute onset and offset times of photostimulation events across the session.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. CONVERSION_NOTES.md Step 5: "Binary time series - 1 during photostim, 0 otherwise. Use absolute photostim_start/stop_times aligned to go cue."

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial: (1) Filter photostim events that overlap the trial's time window. (2) For each time bin center, check if it falls within any photostim [start, stop] interval. (3) Set the bin to 1 if photostim is active, 0 otherwise.

ii.
```python
def compute_photostim_timeseries(go_cue_time, photostim_starts, photostim_stops,
                                  t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    abs_centers = bin_centers + go_cue_time
    stim = np.zeros(n_bins, dtype=np.float32)
    for start, stop in zip(photostim_starts, photostim_stops):
        mask = (abs_centers >= start) & (abs_centers <= stop)
        stim[mask] = 1.0
    return stim
```

iii. CONVERSION_NOTES.md Step 3: "Photostim: last 0.5s of delay epoch (~-0.5s to 0s), bilateral/unilateral ALM silencing."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Same bin centers as neural data (50ms bins from -2.5s to 1.5s relative to go cue). Photostim events are evaluated at each bin center to determine the binary on/off state.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
abs_centers = bin_centers + go_cue_time
```

iii. Same temporal grid as neural and other input data.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` in the NWB trials table. This represents the instructed lick direction (which port the mouse should lick), not the actual lick direction.

ii.
```python
instructions = trials['trial_instruction'][:]
# ...
choice = 0 if instructions[ti] == 'left' else 1
```

iii. CONVERSION_NOTES.md Step 5 maps `trial_instruction` → `output[0]: choice`, noting "left=0, right=1." The reference code similarly uses `trial_type` which represents the instruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Simple mapping: `'left'` → 0, `'right'` → 1. The value is per-trial (constant across time bins) and replicated across all 80 time bins in the output array.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
# ...
full_output[0, :] = out_dict['choice']  # replicated across time
```

iii. Task instructions specify: "Lick direction choice (left = 0, right = 1, per-trial)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from the `outcome` field in the NWB trials table, which contains string values: 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes = trials['outcome'][:]
# ...
outcome_str = outcomes[ti]
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. CONVERSION_NOTES.md Step 5 maps `outcome` → `output[1]: outcome` with "ignore=0, miss=1, hit=2."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String-to-integer mapping: 'ignore' → 0, 'miss' → 1, 'hit' → 2. Default to 0 for any unrecognized value. Per-trial value replicated across all time bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
full_output[1, :] = out_dict['outcome']
```

iii. Task instructions specify: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)."

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Note: The template question mentions "Distance to reward zone" which is not an output in this dataset. For *Outcome*, it is a per-trial scalar value that is broadcast (replicated) across all 80 time bins to match the neural data's temporal dimension.

ii.
```python
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[1, :] = out_dict['outcome']  # same value at every time bin
```

iii. Per-trial outputs are constant across time, so alignment is trivial — the value is copied to all time bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from the `early_lick` field in the NWB trials table, which contains string values like 'no early' or 'early'.

ii.
```python
early_lick = trials['early_lick'][:]
# ...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. CONVERSION_NOTES.md Step 5: maps `early_lick` → `output[2]: early_lick` with "no=0, yes=1."

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'no early' → 0, anything else → 1. Per-trial value replicated across all time bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
full_output[2, :] = out_dict['early_lick']
```

iii. Task instructions specify: "Early lick (no = 0, yes = 1, per-trial)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `Camera0_side_TongueTracking` in `BehavioralTimeSeries`. This contains 3 columns: (x, y, likelihood). The AI uses column index 1 (y-position) and column index 2 (likelihood for quality filtering).

ii.
```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. CONVERSION_NOTES.md Step 5: "Use column 1 (tongue_y) from TongueTracking."

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial: (1) Extract tongue y-position data within the trial's time window. (2) Filter by likelihood threshold (< 0.9 → set to NaN). (3) Bin into 50ms time bins by averaging valid values within each bin using `np.digitize` and `np.bincount`. Bins with no valid data remain NaN.

ii.
```python
def get_tongue_y_for_trial(tongue_data, tongue_timestamps, tongue_likelihood,
                            go_cue_time, t_start, t_end, bin_size):
    # ...
    y_in_window[like_in_window < 0.9] = np.nan
    # Bin the tongue y-position (vectorized)
    tongue_y_binned = np.full(n_bins, np.nan, dtype=np.float32)
    bin_indices = np.digitize(t_in_window, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    valid_mask = ~np.isnan(y_in_window)
    if valid_mask.any():
        valid_bins = bin_indices[valid_mask]
        valid_vals = y_in_window[valid_mask]
        sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
        counts = np.bincount(valid_bins, minlength=n_bins)
        has_data = counts > 0
        tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
    return tongue_y_binned
```

iii. CONVERSION_NOTES.md Step 3: "Tongue tracking: DLC at 300 Hz, columns (x, y, likelihood); outlier correction with 5-sigma velocity threshold." Note: the AI uses a likelihood threshold of 0.9 but does not implement the 5-sigma velocity outlier correction described in the methods.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session discretization using 40th and 60th percentiles of all valid tongue y-values across all trials in the session. Values below 40th percentile → 0, between 40th-60th → 1, above 60th → 2. NaN values (from low likelihood) are set to category 0.

ii.
```python
if len(tongue_y_session) > 0:
    tongue_y_arr = np.array(tongue_y_session)
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)
else:
    p40, p60 = 0, 0

# ...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. Task instructions: "Tongue y-position, per-session discretization: 0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile." The AI follows this exactly.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking data (at ~300 Hz) is binned into the same 50ms time bins as neural data. Within each bin, tongue y-values are averaged. The discretization is then applied to the binned values, producing one category per time bin aligned with the neural data.

ii.
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
# digitize assigns each tongue frame to a bin
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The temporal alignment uses the same window (-2.5s to 1.5s relative to go cue) and bin size (50ms) as neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- **Missing tone onset**: If no `sample_start_time` is found before the go cue, defaults to `go_cue - 1.85` (typical sample epoch timing).
- **Missing tongue data**: Bins without valid tongue data are set to NaN, then discretized as category 0.
- **Low-likelihood tongue tracking**: Frames with likelihood < 0.9 are treated as NaN.
- **Unmapped brain regions**: Neurons whose `anno_name` doesn't map to any of the 14 coarse regions are excluded.
- **Empty spike trains**: Neurons with no spikes in a trial get zero firing rate.
- **Sessions with no good units**: Skipped entirely.
- **Sessions with fewer than 2 valid trials**: Skipped.
- **Recording gaps**: `obs_intervals` are used to exclude trials outside recording periods.

ii.
```python
# Missing tone onset
if tone_onset is None:
    tone_onset = go_cue - 1.85

# Low likelihood tongue data
y_in_window[like_in_window < 0.9] = np.nan

# Unmapped regions
if region is None:
    valid_neuron_mask[i] = False

# Empty sessions
if n_good == 0:
    io.close()
    return None
```

iii. CONVERSION_NOTES.md Step 10 documents: "time_from_tone_onset max=7.94: Some trials have very early or misdetected tone onsets. Not critical for decoder." And: "2 all-zero neural trials: Negligible (0.002% of trials)."

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the per-trial spike binning in `compute_firing_rates()`, which loops over all neurons for each trial. With ~400 neurons/session and ~500 trials/session, this involves ~200,000 loop iterations per session. Loading NWB files and reading tongue tracking data are also significant. Total processing time was ~38 minutes for 173 sessions.

ii.
```python
for i, st in enumerate(spike_times_list):  # loops over neurons
    if len(st) == 0:
        continue
    mask = (st >= abs_start) & (st < abs_end)
    # ...
```

iii. CONVERSION_NOTES.md Step 7 estimated ~6.5s per session, ~19 min total. Actual full conversion took 38 min (longer than estimated due to varying session sizes).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main loop that could be vectorized is the neuron-level loop in `compute_firing_rates()`. Instead of looping over each neuron, all spikes could be processed in a single batch using 2D histogram operations. The brain region mapping loop (iterating over `anno_names`) could also be vectorized with a lookup table. The photostim time series loop over start/stop pairs could use vectorized interval operations.

ii.
```python
# Current loop over neurons:
for i, st in enumerate(spike_times_list):
    # ... per-neuron spike binning

# Current loop for photostim:
for start, stop in zip(photostim_starts, photostim_stops):
    mask = (abs_centers >= start) & (abs_centers <= stop)
    stim[mask] = 1.0
```

iii. The AI did vectorize some operations (tongue y binning uses `np.digitize` and `np.bincount`), but the main spike binning loop remains per-neuron.

## 10-c. What processing does the code repeat multiple times?

i. The code reads and processes tongue tracking data for the entire session once, then iterates over it for each trial. The `go_start_times`, `sample_start_times`, and other event arrays are read once per session but indexed per trial — this is efficient. The output distribution statistics are computed twice: once during conversion (printing summary) and once could be computed from the saved data. The `find_last_sample_before_go` function linearly scans the sample_start_times array for every trial.

ii.
```python
# sample_start_times searched for every trial:
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
```

iii. No specific note in CONVERSION_NOTES.md about repeated processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and stores tongue y-position as a time-varying signal with per-session percentile discretization. However, when tongue data has low likelihood (< 0.9), those bins default to category 0, which conflates "no tongue data" with "tongue below 40th percentile." Additionally, the code computes brain region mapping for neurons that may ultimately be excluded if their annotation doesn't match any coarse region. The obs_intervals validation check processes all unique observation interval patterns across good units, even though many share the same intervals.

ii.
```python
# NaN tongue values default to 0 (same as "below 40th percentile")
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
# Only valid (non-NaN) values get assigned categories; rest stay 0
```

iii. CONVERSION_NOTES.md doesn't specifically discuss unnecessary processing or data loss from tongue discretization defaults.
