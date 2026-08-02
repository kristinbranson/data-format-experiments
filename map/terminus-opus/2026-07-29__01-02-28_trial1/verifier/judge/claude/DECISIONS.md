# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates through subject directories in the `data/` folder (sorted alphabetically, each named `sub-XXXXXX`), finds all `.nwb` files within each subject directory, and loads each NWB file using `h5py`. Each NWB file represents one session. The AI extracts trial information from `intervals/trials`, spike times from `units/spike_times`, go cue times from `acquisition/BehavioralEvents/go_start_times`, sample start times from `acquisition/BehavioralEvents/sample_start_times`, tongue tracking from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, and electrode/region information from `general/extracellular_ephys/electrodes`.

ii.
```python
def get_nwb_files():
    """Get all NWB file paths organized by subject."""
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

iii. The AI noted that the reference code works with `.mat` files from DataJoint export, not NWB directly. The AI mapped NWB fields to equivalent `.mat` fields to achieve the same data loading. 174 NWB files were found across 28 subjects.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the directory name under `data/` (e.g., `sub-440956`). Each NWB file records which subject directory it belongs to. The AI collects unique subject IDs and builds a `subject_idx` array mapping each session to its subject.

ii.
```python
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The AI recognized that each subject has its own directory containing one or more session NWB files. The subject ordering follows alphabetical sorting of directory names.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes each NWB file independently via `process_session()`. Sessions that pass quality criteria are included; others are skipped.

ii.
```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(
        nwb_info['path'],
        nwb_info['subject'],
        show_processing=args.show_processing,
        session_idx=session_count
    )
    if result is None:
        skipped += 1
        continue
```

iii. The AI identified 174 NWB files total. After session-level quality filtering (correct rate > 65%, >= 50 correct trials per side, sufficient neural coverage), 143 sessions passed. The reference papers report 173 sessions.

## 1-d. How are the data split into trials?

i. Trials are read from `intervals/trials` in each NWB file. The trial table contains metadata for each trial including instruction, outcome, early lick, auto_water, free_water, and photostim information. The AI creates a trial mask to select valid trials.

ii.
```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
trial_instruction = trials['trial_instruction'][:]
outcome = trials['outcome'][:]
early_lick = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
# ...
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
```

iii. The AI chose to include all trial types except auto_water and free_water trials, based on the decoder task requirements (early lick and photostim trials are needed as decoder outputs/inputs).

## 1-e. How are trials filtered based on quality controls?

i. Trial-level filtering: auto_water and free_water trials are excluded. Trials must also pass a neural validity check (trial index within recorded range and go cue time within spike time range). Session-level filtering: correct rate > 65% on control non-early-lick trials, and >= 50 correct left and right trials each (computed on control non-early-lick trials).

ii.
```python
# Trial filtering
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid

# Session selection criteria
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
hits = np.sum(outcome[control_non_early] == b'hit')
misses = np.sum(outcome[control_non_early] == b'miss')
correct_rate = hits / (hits + misses)
correct_left = np.sum((outcome == b'hit') & (trial_instruction == b'left') & control_non_early)
correct_right = np.sum((outcome == b'hit') & (trial_instruction == b'right') & control_non_early)

if correct_rate < MIN_CORRECT_RATE:  # 0.65
    # skip session
if correct_left < MIN_CORRECT_TRIALS_PER_SIDE or correct_right < MIN_CORRECT_TRIALS_PER_SIDE:  # 50
    # skip session
```

iii. The AI derived session selection criteria from the methods: ">65% correct rate" and "at least 50 correct lick left and lick right trials each." Correct rate is computed as hits/(hits+misses) on control (no photostim), non-early-lick trials, matching the paper's description.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (a ragged array of absolute spike timestamps for all units) and `units/spike_times_index` (cumulative index into the flat spike_times array for each unit).

ii.
```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
```

iii. The AI identified that NWB stores spike times as absolute timestamps in a flat array with an index array to delineate units, matching the NWB ragged array convention.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins in a window of -2.5s to +1.5s relative to the go cue (80 bins). Spike counts are divided by bin width (0.05s) to get firing rates in Hz. For each trial, spikes within the window are found using `searchsorted`, aligned to go cue, and histogrammed.

ii.
```python
def compute_firing_rates_fast(spike_times_flat, spike_times_index, go_cue_times,
                               good_indices, trial_indices, ...):
    bin_edges = np.linspace(window_start, window_end, n_bins + 1)
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

iii. The AI noted that the reference code uses bw=0.04s (40ms) sliding window with stride=0.0034s, but the decoder task specification requires 50ms bins. The AI followed the decoder task specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by `classification == 'good'`. Only units labeled as 'good' by the classifier-based QC system are included. Units labeled 'unlabelled' are excluded.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
n_good = int(np.sum(good_mask))
good_indices = np.where(good_mask)[0]
```

iii. The AI identified that the reference code uses a classifier-based QC system (5 region-specific logistic regression classifiers) that labels units as 'good' or 'unlabelled'. The NWB `classification` field contains these labels directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to go cue onset. For each trial, the go cue time is obtained from `acquisition/BehavioralEvents/go_start_times/timestamps`, and spikes are extracted in the window [-2.5, +1.5]s relative to the go cue.

ii.
```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
# In compute_firing_rates_fast:
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start  # go_time - 2.5
abs_end = go_time + window_end      # go_time + 1.5
aligned = unit_spikes[idx_lo:idx_hi] - go_time
```

iii. The decoder task instructions specify "Temporally align based on Go cue onset" and "Extract 2.5 s before to 1.5 s after the go cue." The AI follows this exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s). There are 80 bins covering the 4s window (-2.5 to +1.5s). No rebinning is applied; firing rates are computed directly from spike times using `np.histogram` with these bin edges.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
WINDOW_START = -2.5  # seconds relative to go cue
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))  # 80 bins
```

iii. The decoder task specifies "Use 50-ms-width bins for computing firing rates." The reference code uses 40ms bins with 3.4ms stride, but the AI correctly follows the decoder task specification instead.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Time from tone onset is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps` (absolute timestamps of all sample epoch onsets, including replays) and the go cue times.

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
```

iii. The AI noted that `sample_start_times` contains all sample epoch starts including replays due to early licks. The relevant onset for each trial is the last sample start before the go cue.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the AI finds the last `sample_start_time` before the go cue using `searchsorted`. It computes the tone onset time relative to the go cue, then subtracts this from `bin_centers` to get time-from-tone-onset at each time bin.

ii.
```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85  # default fallback
time_from_tone = bin_centers - tone_onset_rel
```

iii. The AI justified using `searchsorted` to handle the fact that `sample_start_times` is a global array (not per-trial) and may include replayed epochs. The fallback of -1.85s is the expected default sample-to-go interval (650ms sample + 1200ms delay = 1850ms).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The input uses the same `bin_centers` array as the neural data, ensuring temporal alignment. `bin_centers` are computed as the center of each 50ms bin within the [-2.5, 1.5]s window.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. By using `bin_centers` for both neural and input data, temporal alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` (onset time relative to trial start), `intervals/trials/photostim_duration` (duration in seconds), and `intervals/trials/start_time` (absolute trial start time).

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

iii. The AI identified that photostim onset is stored relative to trial start time, requiring conversion to go-cue-relative time.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial with photostim (onset != 'N/A'), the AI converts the onset from trial-start-relative to go-cue-relative time, computes the end time using onset + duration, and creates a binary array indicating whether photostim is on at each bin center.

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

iii. The AI verified that the absolute photostim time computed this way matches `photostim_start_times` events in the NWB file. Control trials (no photostim) have all-zero binary arrays.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim binary array uses the same `bin_centers` as the neural data, ensuring temporal alignment.

ii. Same `bin_centers` array is used:
```python
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. Alignment is achieved by evaluating photostim timing at the same bin centers used for firing rate computation.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Lick direction choice is derived from `intervals/trials/trial_instruction`, which records the instructed lick direction ('left' or 'right') for each trial.

ii.
```python
trial_instruction = trials['trial_instruction'][:]
# In output construction:
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. The AI uses `trial_instruction` (the instructed/correct direction) rather than actual lick direction. This means for "miss" trials (incorrect licks), the recorded "choice" reflects what the mouse was supposed to do, not what it actually did.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Left is encoded as 0, right as 1. The value is constant across all time bins for a given trial (per-trial output repeated across time).

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
output_trial[0, :] = choice  # repeated across all time bins
```

iii. The encoding matches the decoder task specification: "left = 0, right = 1, per-trial."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome`, which records the trial outcome as 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = trials['outcome'][:]
out = outcome[trial_idx]
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0  # fallback to ignore
```

iii. The AI maps outcome values to integers matching the decoder task specification: "ignore = 0, miss = 1, hit = 2."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as a categorical integer (0=ignore, 1=miss, 2=hit). The value is constant across all time bins for a given trial. Any unexpected outcome values default to 0 (ignore).

ii.
```python
output_trial[1, :] = outcome_val  # repeated across all time bins
```

iii. The encoding matches the decoder task specification.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. There is no "Distance to reward zone" output in this dataset. The question appears to be from a template. The `Outcome` output is per-trial and constant across time, so alignment is trivial (the same value is replicated across all 80 time bins).

ii.
```python
output_trial[1, :] = outcome_val  # same value for all time bins
```

iii. Per-trial outputs do not require temporal alignment since they are constant.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`, which records whether the mouse licked early during the sample/delay epoch.

ii.
```python
early_lick = trials['early_lick'][:]
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. The NWB field contains string values 'early' or 'no early'.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. 'early' is encoded as 1, 'no early' as 0. The value is constant across time bins per trial.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
output_trial[2, :] = early
```

iii. The encoding matches the decoder task specification: "no = 0, yes = 1, per-trial."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains tracking data with columns for x-position, y-position, and likelihood (confidence), sampled at ~300 Hz.

ii.
```python
has_tongue = 'Camera0_side_TongueTracking' in f['acquisition']['BehavioralTimeSeries']
if has_tongue:
    tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
    tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
```

iii. The AI correctly identified the tongue tracking data source and its three-column structure (x, y, likelihood).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The raw tongue y-position (column index 1) is extracted and filtered by a likelihood threshold of 0.1 (column index 2). Only data points with likelihood > 0.1 are considered valid. The valid y-values are averaged within each 50ms time bin.

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

iii. The likelihood threshold of 0.1 filters out low-confidence tracking data. Bins with no valid data get NaN values.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) are computed from ALL valid (non-NaN) tongue y values across all bins and all selected trials. Values below 40th percentile are category 0, between 40th and 60th are category 1, above 60th are category 2. NaN values (no valid tracking data) default to category 1 (middle).

ii.
```python
all_tongue_y_arr = np.array(all_tongue_y)
p40 = np.percentile(all_tongue_y_arr, 40)
p60 = np.percentile(all_tongue_y_arr, 60)

tongue_y_disc = np.ones(n_bins, dtype=np.int64)  # default to 1 (middle)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. The AI follows the decoder task specification for discretization thresholds. The default of category 1 for NaN values is a design choice not specified in the instructions.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking data is binned using the same bin edges as the neural data (derived from the same window and bin width), ensuring temporal alignment.

ii.
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The bin edges match the neural data bins, ensuring the tongue y values are aligned in time.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **Missing tongue tracking**: Sessions without `Camera0_side_TongueTracking` default to all-NaN tongue y values, which are then categorized as 1 (middle).
- **Missing tone onset**: If no sample start time is found before the go cue (`ss_idx < 0`), a default of -1.85s is used.
- **Partial neural recording**: Trials beyond the recorded range (`n_recorded_trials`) or with go cue times outside the spike time range are marked as invalid.
- **Unexpected outcome values**: Any outcome not in {hit, miss, ignore} defaults to 0 (ignore).
- **Sessions with too few trials**: Sessions with < 2 valid trials are skipped.
- **Sessions with no good units**: Skipped.

ii.
```python
# Missing tongue tracking
if not has_tongue:
    tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]
    all_tongue_y = []

# Missing tone onset
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85

# Partial neural recording
neural_valid = np.zeros(n_trials_total, dtype=bool)
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10.

## 10-a. What are the most time-consuming steps of the code?

i. The firing rate computation is the most time-consuming step, taking ~3-14s per session. This involves nested loops over units and trials, with `searchsorted` and `np.histogram` operations for each.

ii.
```python
# Nested loop: for each trial, for each unit
for trial_idx in trial_indices:
    for i, unit_spikes in enumerate(good_spike_times):
        idx_lo = np.searchsorted(unit_spikes, abs_start)
        idx_hi = np.searchsorted(unit_spikes, abs_end)
        if idx_hi > idx_lo:
            aligned = unit_spikes[idx_lo:idx_hi] - go_time
            counts = np.histogram(aligned, bins=bin_edges)[0]
            fr_trial[i, :] = counts / bin_width
```

iii. The AI estimated ~7s average per session and ~16 minutes total for all sessions. The firing rate computation dominates.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two key loops could be vectorized:
1. **Firing rate computation**: The inner loop over units could be vectorized by concatenating all spikes with unit labels and using vectorized histogram operations.
2. **Tongue y binning**: The per-bin loop in `get_tongue_y_for_trials` could be replaced with `np.digitize` or vectorized bin assignment.

ii.
```python
# Current: loop over bins for tongue y
for b in range(n_bins):
    bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
    if np.sum(bin_mask) > 0:
        tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

iii. The AI noted optimization with `searchsorted` for firing rates but did not fully vectorize the inner loops. The tongue y binning loop is particularly amenable to vectorization.

## 10-c. What processing does the code repeat multiple times?

i. Several operations are repeated:
1. **NWB file I/O**: Each session file is opened, read, and closed independently. No caching or parallel loading.
2. **Bin edges computation**: `np.linspace(window_start, window_end, n_bins + 1)` is computed in both `compute_firing_rates_fast` and `get_tongue_y_for_trials`, though with identical results.
3. **Bin centers computation**: Computed both in the main function and used implicitly in firing rate computation.

ii.
```python
# In compute_firing_rates_fast:
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
# In get_tongue_y_for_trials:
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
```

iii. These redundancies are minor since the arrays are small. The main repeated work is the per-session processing pattern.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several outputs are computed but not fully utilized:
1. **Per-trial outputs replicated across time**: Choice, outcome, and early lick are per-trial values but are replicated across all 80 time bins in the output array. The decoder likely only needs one value per trial for these, making the replication unnecessary storage overhead.
2. **Tongue y values collected for percentile computation then discarded**: `all_tongue_y` is built up across trials for percentile computation, then not used further.
3. **Correct rate and side counts**: Computed for session filtering but could be computed once rather than for every session.
4. **`is_good_trials` shape**: The AI reads this field to determine `n_recorded_trials` but doesn't use the actual per-unit per-trial validity data, which could inform more targeted trial-unit filtering.

ii.
```python
# Per-trial outputs replicated across 80 time bins
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice      # same value in all 80 bins
output_trial[1, :] = outcome_val # same value in all 80 bins
output_trial[2, :] = early       # same value in all 80 bins
```

iii. The format specification requires `(n_output, n_timepoints)` shape, so replication is needed to match the spec even if the per-trial outputs don't vary over time.
