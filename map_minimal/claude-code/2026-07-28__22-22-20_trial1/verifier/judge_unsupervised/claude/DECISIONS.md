# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories (`sub-*`) in the data directory, then iterates over all `.nwb` files within each subject directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and read into memory. The `trials`, `units`, and `acquisition` fields are extracted from each NWB file. Data is processed one session at a time sequentially.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    for fname in files:
        fpath = os.path.join(subj_dir, fname)
        result = process_session(fpath, sample_mode=sample_mode)
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
trials = nwb.trials
units = nwb.units
```

iii. The AI noted the NWB format and identified 28 subjects and 174 sessions in the data directory. It processes each NWB file by extracting trials, units, behavioral events, and tongue tracking data.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by reading `nwb.subject.description` from each NWB file, which contains the mouse name (e.g., 'SC015'). Unique subject IDs are collected across all sessions using `OrderedDict.fromkeys` to maintain insertion order.

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015'
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The AI identified the subject information from the NWB file's `subject.description` field. The final dataset contains 25 unique subjects (after session selection filtering removed some subjects entirely).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are identified by `nwb.session_id` or the filename if session_id is absent. Sessions that fail quality criteria are excluded.

ii.
```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. From the CONVERSION_NOTES: 174 total sessions were found, of which 105 passed selection criteria (69 excluded).

## 1-d. How are the data split into trials?

i. Trials are defined by the NWB `trials` table. The AI iterates over all trials using the NWB trial table indices. Trials must also have corresponding entries in the `obs_intervals` of the units table to be included (meaning neural data is available for that trial).

ii.
```python
trials = nwb.trials
n_trials = len(trials)

valid_trial_indices = []
for i in range(n_trials):
    if i not in trials_with_neural:
        continue
    outcome = trials['outcome'][i]
    early = trials['early_lick'][i]
    if early != 'no early':
        continue
    if outcome == 'ignore':
        continue
    valid_trial_indices.append(i)
```

iii. The AI maps obs_intervals to trials to ensure neural data is available. It builds a bidirectional mapping between obs_interval indices and trial table indices.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two ways: (1) early lick trials are excluded (`early_lick != 'no early'`), and (2) no-response/ignore trials are excluded (`outcome == 'ignore'`). Additionally, trials without matching neural data (not in `obs_intervals`) are excluded. The AI also requires at least 2 valid trials per session.

ii.
```python
if early != 'no early':
    continue
if outcome == 'ignore':
    continue
if i not in trials_with_neural:
    continue
```

iii. From CONVERSION_NOTES: "Following the paper's methodology: Early lick trials excluded, No-response (ignore) trials excluded." This matches the methods.txt statement: "Early lick trials and no response trials were excluded for analysis."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spike_times` field in the NWB `units` table. For each unit classified as 'good', the spike times are loaded and used to compute firing rates.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
# where:
return [nwb.units['spike_times'][uid] for uid in unit_indices]
```

iii. The AI identified spike times from the NWB units table as the raw neural data source, consistent with the NWB format for storing electrophysiology recordings.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins aligned to the go cue. Spike counts in each bin are divided by the bin width (0.05s) to obtain firing rates in Hz. This uses `np.histogram` with linearly spaced bin edges from -2.5s to +1.5s relative to the go cue, producing 80 time bins.

ii.
```python
n_bins = int(round((end_time - begin_time) / bin_width))
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
counts, _ = np.histogram(st_window, bins=bin_edges)
fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. From CONVERSION_NOTES: "Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to get firing rates in Hz. This is simpler than the sliding kernel approach in the original preprocessing code but appropriate for the decoder task."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` in the NWB units table are included. This corresponds to the classifier-based QC described in the paper, where trained logistic regression classifiers labeled units as 'good' or 'unlabeled'. Sessions with 0 good units are skipped.

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    io.close()
    return None
```

iii. From CONVERSION_NOTES: "QC method: Classifier-based ('good' classification in NWB)." The methods.txt describes the classifier-based QC process using region-specific logistic regression classifiers.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset time. The go cue time is extracted from `BehavioralEvents['go_start_times']` using the trial index. Bin edges are computed relative to the go cue time, spanning -2.5s to +1.5s.

ii.
```python
go_cue = go_start_times[trial_idx]
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
# In compute_firing_rates:
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The instructions specify: "Temporally align based on Go cue onset. Extract 2.5 s before to 1.5 s after the go cue." The AI follows this exactly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms bins (BIN_WIDTH = 0.05s), producing 80 time bins per trial. No temporal rebinning is applied; spikes are directly histogrammed into 50ms bins. The reference code uses 40ms Gaussian kernel with 3.4ms stride, but the AI chose 50ms rectangular bins as specified in the decoder task instructions.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
n_bins = int(round((end_time - begin_time) / bin_width))  # = 80
```

iii. The instructions specify "50-ms-width bins for computing firing rates." The reference code (`preprocess_all_ephys.py`) uses `bw = 0.04` (40ms) and `stride = 0.0034` with a sliding kernel, but the AI correctly followed the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The tone onset time is derived from `BehavioralEvents['sample_start_times']` timestamps. The AI finds the last `sample_start_time` that occurs between the trial start time and the go cue time.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
sample_start_times = be.time_series['sample_start_times'].timestamps[:]

t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback
```

iii. The AI uses `sample_start_times` from the NWB behavioral events, which corresponds to the onset of the sample epoch (tone presentation). A fallback of `go_cue - 1.85` is used when no matching sample start is found.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset time is converted to a relative time (relative to go cue), then for each time bin, the time since tone onset is computed as `bin_center - tone_onset_relative`. This produces a continuous, linearly increasing time series.

ii.
```python
tone_onset_rel = tone_onset - go_cue  # relative to go cue (should be ~-1.85)
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
```

iii. The AI represents time from tone onset as a continuous variable that increases linearly across bins, rather than as a binary onset indicator. This matches the instruction specification "Time from tone onset in seconds (continuous, time-varying)."

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time from tone onset is computed at the same bin centers as the neural data (both aligned to go cue), so the alignment is inherent. The `bin_centers` array used for neural data is reused for computing the tone onset input.

ii.
```python
time_from_tone = bin_centers - tone_onset_rel  # same bin_centers as neural data
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)
```

iii. Since both neural data and tone onset input use the same temporal grid (80 bins from -2.5s to +1.5s relative to go cue), alignment is guaranteed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from two fields in the NWB trials table: `photostim_onset` (onset time relative to trial start) and `photostim_duration` (duration). Control trials have `photostim_onset == 'N/A'`.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
```

iii. The AI correctly identified that photostimulation information is stored in the NWB trials table, with 'N/A' indicating control trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary array is created (0=off, 1=on). For photostimulation trials, the onset time (relative to trial start) is converted to absolute time, then the duration is added to get the offset. These are converted to times relative to go cue, and bins falling within the [start, stop) interval are set to 1.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
    ps_start_rel = photostim_start_abs - go_cue
    ps_stop_rel = photostim_stop_abs - go_cue
    photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The AI documents this as: "Binary input (0=off, 1=on). Photostim onset times from NWB are converted from absolute times to relative-to-go-cue times."

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned using the same bin_centers as the neural data. The photostimulation time window is converted to go-cue-relative coordinates and compared against bin_centers.

ii.
```python
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. Same temporal grid as neural data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from two fields in the NWB trials table: `trial_instruction` (which side should be licked: 'left' or 'right') and `outcome` ('hit' or 'miss'). The AI infers the actual choice from the combination of instruction and outcome.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0  # miss means licked wrong side
```

iii. From CONVERSION_NOTES: "Hit trials: choice = instruction side. Miss trials: choice = opposite of instruction side."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as left=0, right=1. For hit trials, the choice matches the instruction. For miss trials, the choice is the opposite of the instruction (since miss means the animal licked the wrong side). The output is expanded to a time-varying array (constant across all time bins).

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
# Later expanded:
output_trial[0, :] = base_output[0]  # choice (constant across time)
```

iii. The AI correctly infers the actual lick direction from instruction + outcome, since the NWB data doesn't have a direct "choice" field.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the `outcome` field in the NWB trials table, which contains values 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = trials['outcome'][trial_idx]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. The AI maps the string outcomes to integers matching the specification: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped to integers: ignore=0, miss=1, hit=2. Since ignore trials are filtered out, only miss (1) and hit (2) appear in the final data. The output is expanded to time-varying (constant across all time bins).

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```

iii. The output distribution shows outcome has values [1.0, 2.0], confirming ignore trials (0) are absent after filtering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the `early_lick` field in the NWB trials table, which contains either 'no early' or 'early'.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The raw data contains an explicit early_lick field in the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Early lick is encoded as no=0, yes=1. However, since early lick trials are filtered out during trial selection, this output is always 0 in the final dataset. The output is expanded to time-varying (constant across all time bins).

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

iii. From CONVERSION_NOTES: "After filtering, early_lick output is always 0." The verification confirms: "early_lick: {no (1.000)}". This makes the early_lick output trivially decodable (100% accuracy) and uninformative.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the `BehavioralTimeSeries['Camera0_side_TongueTracking']` time series in the NWB acquisition. This contains (x, y, likelihood) columns from DeepLabCut tracking of the side camera.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The AI identified the tongue tracking data from the side camera (Camera0_side), extracting the y-position (column index 1) and likelihood (column index 2).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, tongue tracking data within the trial window (go_cue + begin_time - bin_width to go_cue + end_time + bin_width) is extracted. Frames with likelihood > 0.5 are kept. Good frames are binned into the same 50ms time bins as neural data using `np.digitize`, with mean y-position computed per bin. Bins without valid tracking data are set to NaN initially.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
lh_mask = tw_lh > 0.5
tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
if len(tw_times_good) > 0:
    abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
    bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
    for b in range(len(bin_centers)):
        b_mask = bin_assignments == b
        if np.any(b_mask):
            tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The AI uses a likelihood threshold of 0.5 to filter DeepLabCut tracking confidence, which is a reasonable choice for DLC outputs.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) are computed from all valid (non-NaN) tongue y values across all trials in the session. Then each time bin's y-position is categorized: 0 if below 40th percentile, 1 if between 40th and 60th, 2 if above 60th. NaN values (no valid tracking) default to category 0.

ii.
```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)

def discretize_tongue_y(tongue_y_values, p40, p60):
    for i, val in enumerate(tongue_y_values):
        if np.isnan(val):
            result[i] = 0  # no tongue visible = low position
        elif val < p40:
            result[i] = 0
        elif val <= p60:
            result[i] = 1
        else:
            result[i] = 2
```

iii. The instructions specify: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile" with per-session discretization. The AI follows this, but uses `<= p60` for the middle bin, meaning exactly p60 goes to category 1, while exactly p40 goes to category 0. The output distribution shows tongue_y_position is heavily skewed: {low: 83.1%, mid: 5.6%, high: 11.3%}, likely because NaN values (no tongue visible) default to 0.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are binned into the same temporal grid as neural data (80 bins of 50ms from -2.5s to +1.5s relative to go cue). Mean y-position is computed per bin.

ii.
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The alignment uses the same bin edges as the neural data, ensuring temporal correspondence.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used:
- **Missing tongue tracking**: Bins with no valid tracking data (or low likelihood) are set to NaN, then defaulted to category 0 (low) during discretization.
- **Missing tone onset**: If no `sample_start_time` is found before the go cue, a fallback of `go_cue - 1.85` is used.
- **obs_intervals mismatch**: When the number of obs_intervals doesn't match the number of trials, a temporal matching algorithm is used (matching by start time with 0.5s tolerance). Trials without matching neural data are excluded.
- **All-zero neural data**: Some trials have all-zero neural data (no spikes in the window) - these are retained but flagged as warnings during verification.
- **Missing brain region annotations**: Units with missing/invalid `anno_name` are labeled 'unknown'.

ii.
```python
# Missing tone onset fallback:
tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration

# Missing tongue data:
if np.isnan(val):
    result[i] = 0  # no tongue visible = low position

# Missing brain region:
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
```

iii. From CONVERSION_NOTES: The AI documented several known issues including all-zero neural data in sessions 40-42 and the obs_intervals handling approach.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading spike times** for all good units from NWB files (`preload_spike_times`), which reads potentially millions of spike times from disk.
2. **Computing firing rates** for each trial, which involves histogramming spike times for each neuron.
3. **Building obs_intervals to trial mapping**, which involves iterating over observation intervals and matching to trial start times.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
# preloading all spike times into memory

fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
# histogramming per neuron per trial
```

iii. The AI preloads spike times once per session (rather than reading from NWB per trial), which is a good optimization. However, the per-trial processing is still sequential.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **Firing rate computation**: The inner loop over neurons (`for i, st in enumerate(spike_times_by_neuron)`) could potentially be vectorized if spike times were combined.
2. **Tongue y binning**: The inner loop `for b in range(len(bin_centers))` computing mean y per bin is a loop that could be vectorized using array operations.
3. **Trial filtering loop**: The loop checking trial validity could use vectorized boolean operations on the trials table.
4. **obs_to_trial mapping**: The loop matching obs_intervals to trials via argmin could be vectorized with broadcasting.

ii.
```python
# Neuron loop in compute_firing_rates:
for i, st in enumerate(spike_times_by_neuron):
    counts, _ = np.histogram(st_window, bins=bin_edges)
    fr[i, :] = counts.astype(np.float32) / bin_width

# Tongue y binning loop:
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. These loops operate over potentially hundreds of neurons and 80 time bins per trial across thousands of trials, so vectorization would provide meaningful speedup.

## 10-c. What processing does the code repeat multiple times?

i. The `map_region_to_major` function is called twice for each neuron's brain region label:
1. Once when collecting all region labels into a set (`all_region_labels`)
2. Again when building `brain_region_idx` arrays

ii.
```python
# First call:
for s in all_sessions:
    for label in s['brain_region_labels']:
        all_region_labels.add(map_region_to_major(label))

# Second call:
for s in all_sessions:
    idx = np.array([brain_regions.index(map_region_to_major(label))
                    for label in s['brain_region_labels']], dtype=np.int64)
```

iii. The `map_region_to_major` function performs string matching with many `any(x in anno for x in [...])` checks. Calling it twice per neuron is redundant but not a major performance issue compared to spike processing.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several aspects of processing produce unused or trivially uninformative results:
1. **Early lick output**: Since early lick trials are filtered out, this output is always 0. It provides no information for the decoder (trivially 100% accuracy).
2. **Obs_intervals trial-level spike time filtering**: The `get_spike_times_for_trial_from_cache` function filters spike times by obs_interval boundaries, but then `compute_firing_rates` further clips to the [-2.5, 1.5] window around go cue. The obs_interval filtering is redundant if the firing rate window is always within the obs_interval.
3. **Brain region mapping complexity**: The detailed region mapping (15 categories) involves many string pattern checks, but the decoder uses all neurons regardless of region. The region info is metadata rather than used for processing.

ii.
```python
# Early lick is always 0 after filtering:
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
# This output is always 0 and the decoder achieves 100% trivially

# Redundant spike time filtering:
spike_times_list = []
t_start, t_stop = obs_intervals[obs_idx]
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
# Then further filtered in compute_firing_rates:
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
```

iii. The early lick output being always 0 is explicitly noted in CONVERSION_NOTES and verified in the output statistics. The decoder achieves 100% "accuracy" on this trivially constant output.
