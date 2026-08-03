# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over all NWB files organized by subject directories under `data/`. It scans for directories starting with `sub-`, then finds `.nwb` files within each. Each NWB file is opened with `h5py` and processed individually via `process_session()`. Data is read from HDF5 groups: `intervals/trials` for trial info, `units` for spike data, `acquisition/BehavioralEvents` for event timestamps, `acquisition/BehavioralTimeSeries` for tongue tracking, and `general/extracellular_ephys/electrodes` for brain region info.

ii.
```python
def get_nwb_files():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({'subject': subj, 'path': os.path.join(subj_dir, nwb_file), 'filename': nwb_file})
    return all_files
```

iii. The AI noted that "our data is in NWB format" and mapped NWB fields to equivalent fields from the reference code which used .mat files from DataJoint. This is documented in CONVERSION_NOTES Step 1 and Step 4.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the directory name (e.g., `sub-440956`). Each NWB file's parent directory name is used as the subject ID. The AI maintains a running list of unique subjects and assigns a subject index to each session.

ii.
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The AI noted 28 subjects in the data, matching reference papers. Subjects are stored in the `subjects` list and indexed via `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are processed sequentially and must pass quality criteria (>65% correct rate, >=50 correct trials per side for both left and right) to be included. Out of 174 NWB files, 143 sessions pass these criteria (23 skipped for low correct rate, 8 for insufficient correct trials per side).

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

iii. The AI documented session selection criteria in CONVERSION_NOTES Step 3: "Performance threshold: >65%" and "Min correct trials/side: 50 each", derived from the reference methods. The correct rate is computed as hits/(hits+misses) on control (non-photostim) trials without early lick.

## 1-d. How are the data split into trials?

i. Trials are read from `intervals/trials` in each NWB file. Each row represents a trial with associated metadata (instruction, outcome, early_lick, photostim, etc.). The AI filters trials to exclude auto_water and free_water trials, and also checks for neural recording coverage.

ii.
```python
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
```

iii. The AI justified including early lick, ignore, and photostim trials (unlike the reference code which excludes them) because the decoder task specification requires early lick as an output, photostim as an input, and all outcomes as outputs.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering has two levels: (1) Individual trial filtering: exclude auto_water and free_water trials, and trials without sufficient neural coverage (checked via spike time range). (2) Session-level filtering: sessions must have >65% correct rate and >=50 correct trials per side (computed on control, non-early-lick trials).

ii.
```python
# Trial-level filtering
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid

# Neural coverage check
neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

# Session-level correct rate (on control non-early-lick trials)
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
hits = np.sum(outcome[control_non_early] == b'hit')
misses = np.sum(outcome[control_non_early] == b'miss')
correct_rate = hits / (hits + misses)
```

iii. The AI documented these criteria in CONVERSION_NOTES Steps 3 and 5, noting that they match the reference paper's methods. The neural coverage check uses a 1.0s buffer around the spike time range, which is a reasonable tolerance.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (flat array of all spike times across all units) and `units/spike_times_index` (ragged array indices delimiting each unit's spike times). Only units with `classification == b'good'` are included.

ii.
```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
```

iii. The AI noted in CONVERSION_NOTES Step 1 that the reference code uses QC classifier-based "good" unit labels, and mapped this to the NWB `classification` field.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins over a [-2.5s, 1.5s] window relative to the go cue, then converted to firing rates (Hz) by dividing counts by bin width. This produces an 80-timepoint vector per neuron per trial.

ii.
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
# For each trial and unit:
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. The AI noted in CONVERSION_NOTES Step 5 that the reference code uses 40ms sliding bins with 3.4ms stride, but the decoder task specification requires 50ms bins. This difference is justified by the task requirements.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == b'good'` are included. This corresponds to the 5 region-specific logistic regression classifiers described in the spike sorting QC paper. No additional firing rate thresholds or other neuron-level quality controls are applied.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
good_indices = np.where(good_mask)[0]
```

iii. The AI documented in CONVERSION_NOTES Step 3 that "Good unit rate = 25.9%" matching the QC paper's report, and that this classifier-based approach matches the reference code's `helper_get_neuron_id_area` function.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, spike times are centered on the go cue time by subtracting `go_times[trial_idx]`, then binned over the [-2.5, 1.5]s window.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start  # go_time - 2.5
abs_end = go_time + window_end      # go_time + 1.5
# ...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
```

iii. The AI documented in CONVERSION_NOTES Step 5: "Temporal alignment: go cue = time 0" and noted this matches both the reference code and the decoder task specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms bins (BIN_WIDTH = 0.05s), producing 80 time bins over the 4s window. No rebinning is applied; spike times are directly histogrammed into the 50ms bins.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
```

iii. The AI noted this differs from the reference code's 40ms sliding bins but matches the decoder task specification. The metadata stores `time_bin_size: 50.0` (in ms).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (the tone/sample onset time for each trial) and `acquisition/BehavioralEvents/go_start_times/timestamps` (the go cue time).

ii.
```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
```

iii. The AI mapped "sample_start_times" to tone onset based on the task structure where the sample epoch begins with the first tone.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the tone onset time relative to the go cue is computed by finding the nearest sample_start_time before the go cue using `searchsorted`. Then time_from_tone is computed as `bin_centers - tone_onset_rel`, where tone_onset_rel is the tone onset relative to go cue (a negative number, typically around -1.85s). If no sample_start is found, a default of -1.85s is used.

ii.
```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. The AI noted that the sample epoch is 650ms (3 tones x 150ms + 2 gaps x 100ms) and typically starts ~1.85s before the go cue. The searchsorted approach is used to handle potential edge cases in timestamp alignment.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone values are computed at the same bin_centers as the neural data (centered within each 50ms bin), ensuring perfect temporal alignment. Both use the go cue as the common reference point.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
```

iii. The AI uses a shared time grid (bin_centers) for both neural and input data, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `intervals/trials/photostim_onset` (onset time relative to trial start, stored as bytes string or 'N/A'), `intervals/trials/photostim_duration` (duration in seconds), and `intervals/trials/start_time` (trial start time).

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

iii. The AI noted that photostim_onset is stored as bytes (e.g., `b'1.936'` or `b'N/A'`) and requires parsing.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, if photostim_onset is not 'N/A', the photostim onset and end times are computed relative to the go cue. The onset is `trial_start + photostim_onset - go_time`, and the end is onset + duration. A binary vector is created where bins with centers falling within the photostim window are set to 1.

ii.
```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The AI verified that photostim typically occurs during the last 0.5s of the delay period (about -1.2s to -0.7s relative to go cue), matching the methods description.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostim binary uses the same bin_centers as neural data, ensuring temporal alignment via the common go cue reference.

ii. Same `bin_centers` array is used as for neural and time_from_tone inputs.

iii. Alignment is ensured by using the go cue as the common reference point for all time series.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `intervals/trials/trial_instruction`, which encodes the instructed lick direction (b'left' or b'right').

ii.
```python
trial_instruction = trials['trial_instruction'][:]
# ...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. The AI maps trial_instruction directly to choice (left=0, right=1).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A simple binary mapping: 'left' -> 0, 'right' -> 1. The value is repeated across all 80 time bins as a per-trial constant.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
output_trial[0, :] = choice
```

iii. The AI treats choice as a per-trial variable repeated across time bins, as specified in the target format.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `intervals/trials/outcome`, which contains values b'hit', b'miss', or b'ignore'.

ii.
```python
outcome = trials['outcome'][:]
# ...
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

iii. The mapping follows the instruction specification: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A categorical mapping from string values to integers: ignore=0, miss=1, hit=2. Unknown values default to 0 (ignore). The value is repeated across all time bins.

ii.
```python
output_trial[1, :] = outcome_val
```

iii. The AI follows the decoder task specification exactly for outcome encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `intervals/trials/early_lick`, which contains values b'early' or b'no early'.

ii.
```python
early_lick = trials['early_lick'][:]
# ...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. Maps to binary: no early lick = 0, early lick = 1.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'no early' -> 0, 'early' -> 1. Repeated across all time bins.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
output_trial[2, :] = early
```

iii. Follows the decoder task specification: early lick (no = 0, yes = 1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (3 columns: x, y, likelihood) and its associated `timestamps`.

ii.
```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
```

iii. The data has 3 columns: x-position, y-position, and tracking likelihood, sampled at ~300Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue y-position data is extracted within the trial window, filtered by likelihood > 0.1, and averaged within each 50ms time bin. The result is a binned tongue y-position time series with NaN for bins without valid data.

ii.
```python
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
high_conf = lk_slice > 0.1

for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

iii. The likelihood threshold of 0.1 is used to filter low-confidence tracking data before binning.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) are computed from all valid (non-NaN) tongue y values pooled across all trials and time bins. Values are then discretized: < p40 -> 0, p40 to p60 -> 1, > p60 -> 2. NaN values (no valid tracking) default to category 1 (middle).

ii.
```python
all_tongue_y_arr = np.array(all_tongue_y)
p40 = np.percentile(all_tongue_y_arr, 40)
p60 = np.percentile(all_tongue_y_arr, 60)

tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # Default to 1 (middle)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. The instructions specify "per-session discretization" with 40th and 60th percentiles. The AI pools all valid tongue y data across the session for percentile computation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking is binned using the same bin_edges as neural data, aligned to the go cue. Each 50ms bin's tongue y is the mean of all valid tracking samples within that bin.

ii.
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. Alignment is ensured by using the same go-cue-relative time grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Missing tongue tracking: defaults to category 1 (middle) for all time bins
- Sessions without photostim: photostim binary is all zeros
- Partial neural recording (trials extending beyond recording): checked via spike time range with 1.0s buffer
- Trials exceeding recorded trial count: filtered via `min(n_trials_total, n_recorded_trials)`
- Unknown outcome values: default to 0 (ignore)
- Missing sample_start_times: default tone onset of -1.85s relative to go cue

ii.
```python
# Missing tongue tracking
if not has_tongue:
    tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]

# Neural coverage
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True

# Missing tone onset
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
```

iii. The AI documented edge case handling in CONVERSION_NOTES Step 10, Check 5.

## 10-a. What are the most time-consuming steps of the code?

i. Firing rate computation is by far the most time-consuming step, taking 3-14s per session. Total conversion takes about 935s (~16 minutes) for all 174 NWB files. Tongue tracking extraction takes 0.2-0.7s per session. Data loading takes 0.2-0.5s per session.

ii.
```python
# Timing is printed for each step:
print(f"    Firing rate computation: {t2-t1:.1f}s")
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

iii. The AI documented timing in CONVERSION_NOTES Step 7 and used searchsorted for optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be vectorized:
1. The inner loop over units in `compute_firing_rates_fast` (line 101-109): iterates over each good unit per trial, applying searchsorted and histogram individually.
2. The inner loop over time bins in `get_tongue_y_for_trials` (line 144-148): iterates over each of the 80 bins to compute mean tongue y per bin.

ii.
```python
# Unit loop (could be partially vectorized with batch histogram operations)
for i, unit_spikes in enumerate(good_spike_times):
    idx_lo = np.searchsorted(unit_spikes, abs_start)
    idx_hi = np.searchsorted(unit_spikes, abs_end)
    if idx_hi > idx_lo:
        aligned = unit_spikes[idx_lo:idx_hi] - go_time
        counts = np.histogram(aligned, bins=bin_edges)[0]
        fr_trial[i, :] = counts / bin_width

# Bin loop for tongue tracking
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    n_in_bin = np.sum(bin_mask)
    if n_in_bin > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

iii. The AI noted in CONVERSION_NOTES Step 6 that searchsorted was used as an optimization over full array masking. The tongue bin loop could be replaced with np.digitize + groupby operations.

## 10-c. What processing does the code repeat multiple times?

i. Several computations are repeated:
1. The same spike time slicing per unit is done for every trial (though pre-extraction of per-unit spike times mitigates this).
2. Bin edges are recomputed in both `compute_firing_rates_fast` and `get_tongue_y_for_trials` (minor, but redundant).
3. The searchsorted for tongue timestamps is done per-trial in a loop when it could be batched.

ii.
```python
# bin_edges computed in both functions
bin_edges = np.linspace(window_start, window_end, n_bins + 1)  # in compute_firing_rates_fast
bin_edges = np.linspace(window_start, window_end, n_bins + 1)  # in get_tongue_y_for_trials
```

iii. These redundancies are minor and don't significantly impact performance.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes several things that may not be needed downstream:
1. `all_tongue_y` list is accumulated for percentile computation but contains raw tongue y values from all trials/time bins - these are discarded after computing p40/p60.
2. The processing plots (in `--show-processing` mode) are computed but only for the first 2 sessions.
3. Session-level statistics like correct_rate are computed and printed but not stored in the output data structure.
4. Brain region information is extracted via JSON parsing of electrode locations for ALL units, then filtered to good units only.
5. The tongue tracking data bins an extra `bin_width` padding on each side of the window (`abs_start = go_time + window_start - bin_width`, `abs_end = go_time + window_end + bin_width`) which is read but never used.

ii.
```python
# Extra padding read but unused
abs_start = go_time + window_start - bin_width
abs_end = go_time + window_end + bin_width

# Brain regions for all units computed, then filtered
unit_regions = []
for i in range(n_total):
    # computes for all units
    ...
good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
```

iii. These are minor inefficiencies that don't substantially impact correctness or runtime.
