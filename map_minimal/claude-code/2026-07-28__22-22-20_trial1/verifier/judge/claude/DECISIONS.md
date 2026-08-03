# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories (`sub-*`) under the data directory, then iterates over `.nwb` files within each subject directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and processed by `process_session()`. Trials, units, and behavioral events are read from within each NWB file.

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
be = nwb.acquisition['BehavioralEvents']
```

iii. The AI chose to iterate subject directories and then files within them, which is functionally equivalent to a glob over all NWB files. The CONVERSION_NOTES.md states the dataset has 28 subjects and 174 sessions, matching the dandiset.

## 1-b. How are the data split into subjects?

i. Subject identity is read from `nwb.subject.description` (e.g. `'SC015'`), which is the mouse name rather than the numeric subject_id. Unique subjects are collected in order of first appearance and indexed.

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
```

```python
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The AI chose `nwb.subject.description` which gives the mouse name (e.g. 'SC015') rather than `nwb.subject.subject_id` which gives the numeric DANDI ID (e.g. '440956'). Both are valid identifiers for the same animal. The CONVERSION_NOTES.md does not explicitly discuss this choice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Session identity comes from `nwb.session_id`. Sessions are processed in sorted filename order within each subject directory.

ii.
```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. Since the dataset stores one session per NWB file, this is straightforward. The AI notes in CONVERSION_NOTES.md that 105 out of 174 sessions pass their selection criteria.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB trials table (`nwb.trials`). Go cue times come from `BehavioralEvents/go_start_times`. The AI builds a mapping between `obs_intervals` indices and trial table indices using temporal matching.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

```python
def build_obs_to_trial_map(nwb, good_indices):
    obs = units['obs_intervals'][good_indices[0]]
    if n_obs == n_trials:
        return list(range(n_trials))
    # Match by start time
    trial_starts = np.array([trials['start_time'][i] for i in range(n_trials)])
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0.5:
            obs_to_trial.append(int(min_idx))
```

iii. The AI uses a tolerance-based matching approach (0.5s) to map obs_intervals to trials, rather than exact matching. This handles sessions where obs_intervals cover fewer trials than the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) trials must have neural data (be in obs_intervals), (2) early lick trials are excluded (`early_lick != 'no early'`), and (3) no-response/ignore trials are excluded (`outcome == 'ignore'`). Additionally, entire sessions are filtered based on performance criteria: >65% correct on control (non-photostim) non-early-lick trials, and at least 50 correct lick-left and 50 correct lick-right trials.

ii.
```python
# Session selection
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None

# Trial filtering
for i in range(n_trials):
    if i not in trials_with_neural:
        continue
    if early != 'no early':
        continue
    if outcome == 'ignore':
        continue
    valid_trial_indices.append(i)
```

iii. The AI's CONVERSION_NOTES.md states: "Following the paper's methodology: Early lick trials excluded, No-response (ignore) trials excluded." The session selection criteria (>65% correct, >=50 correct per side) are taken from the reference code for the method paper. The AI also notes that `free_water` trials are not explicitly filtered (unlike the reference solution).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `nwb.units['spike_times']`, the sorted spike times for each unit. Only units with `classification == 'good'` are used.

ii.
```python
all_spike_times = [nwb.units['spike_times'][uid] for uid in good_indices]
```

iii. The AI uses spike times as the source for computing firing rates, consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Spike times are histogrammed into 50ms bins aligned to the go cue, then divided by bin width to produce firing rates in Hz. The AI uses `np.histogram` per neuron per trial.

ii.
```python
def compute_firing_rates(spike_times_by_neuron, go_cue_time, begin_time, end_time, bin_width):
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
    for i, st in enumerate(spike_times_by_neuron):
        mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
        st_window = st[mask]
        if len(st_window) > 0:
            counts, _ = np.histogram(st_window, bins=bin_edges)
            fr[i, :] = counts.astype(np.float32) / bin_width
    return fr, bin_centers
```

iii. CONVERSION_NOTES.md states: "Spike counts are histogrammed into 50ms bins and divided by bin width (0.05s) to get firing rates in Hz."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with 0 good units are skipped. No additional quality metric thresholds are applied.

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    return None
```

iii. CONVERSION_NOTES.md states: "QC method: Classifier-based ('good' classification in NWB)." This matches the reference approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. Bin edges are computed as `go_cue_time + [BEGIN_TIME, ..., END_TIME]`. Spikes are histogrammed into these absolute-time bins.

ii.
```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The instructions specify alignment to go cue onset, which the AI follows.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins spanning -2.5s to +1.5s relative to go cue, giving 80 time bins per trial. No rebinning is applied; spikes are binned directly from spike times.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
n_bins = int(round((end_time - begin_time) / bin_width))  # 80
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

iii. The 50ms bin width and [-2.5, 1.5]s window match the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset events) and `go_start_times` (go cue times). The AI finds the last `sample_start_time` that falls between the trial's `start_time` and its go cue.

ii.
```python
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

iii. The AI filters sample_start_times to those within the trial boundaries before taking the last one. A fallback of `go_cue - 1.85` is used if no valid sample is found (though the AI notes this shouldn't normally happen).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue, then bin centers (also relative to go cue) are shifted by this offset to give time-from-tone-onset at each bin.

ii.
```python
tone_onset_rel = tone_onset - go_cue  # relative to go cue
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
```

iii. This produces a continuous time-varying input giving seconds elapsed since tone onset at each time bin.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both neural data and time_from_tone use the same bin centers defined relative to the go cue, so they are inherently aligned.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
time_from_tone = bin_centers - tone_onset_rel
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `trials['photostim_onset']` and `trials['photostim_duration']` in the trials table, along with `trials['start_time']` and go cue time for coordinate conversion.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. Same source variables as the reference solution.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: bins where the bin center falls between photostim onset and offset are set to 1, all others to 0.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. CONVERSION_NOTES.md states: "Binary input (0=off, 1=on)."

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are converted to go-cue-relative times, then compared against the same bin centers used for neural data.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials['trial_instruction']` (left/right) and `trials['outcome']` (hit/miss/ignore). Since ignore trials are excluded, choice is only left or right.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0  # shouldn't happen after filtering
```

iii. The AI derives choice from the combination of instruction and outcome, since the actual lick direction is not stored. Because ignore trials are excluded, only hit/miss outcomes are present, so choice is always 0 (left) or 1 (right).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right (2 classes). It is stored as a per-trial scalar that is then broadcast across all 80 time bins.

ii.
```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]  # choice (constant across time)
```

```python
'output_values': [
    ['left', 'right'],           # choice
    ...
]
```

iii. The AI uses only 2 output values for choice (left, right) rather than 3, because ignore trials have been excluded.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From `trials['outcome']`, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = trials['outcome'][trial_idx]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. Direct mapping from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. However, since ignore trials are excluded, only miss=1 and hit=2 values appear in the output. The value is broadcast across all time bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```

```python
'output_values': [
    ...
    ['ignore', 'miss', 'hit'],   # outcome
    ...
]
```

iii. The mapping follows the instructions (ignore=0, miss=1, hit=2), but ignore trials are never present due to trial filtering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials['early_lick']`, which contains 'no early' or 'early'.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. Direct mapping from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0 (no early) or 1 (early). However, since early lick trials are excluded, the value is always 0 in practice. The AI acknowledges this in CONVERSION_NOTES.md: "early_lick output is always 0 and outcome never has value 0 (ignore)."

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

iii. CONVERSION_NOTES.md: "After filtering, `early_lick` output is always 0."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (tongue_y) and column 2 (tongue_likelihood) of the (n_frames, 3) data array, along with timestamps.

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Same source as the reference solution.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood <= 0.5 are excluded. The remaining tongue_y values are averaged per 50ms bin within each trial window. Session-level 40th and 60th percentiles are computed from the valid (non-NaN) bin-averaged values across all valid trials (not the whole session), and used to discretize into 3 classes: 0 (below 40th), 1 (40th-60th), 2 (above 60th).

ii.
```python
lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]

# Bin and average
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])

# Session percentiles from valid trial values only
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)
```

iii. CONVERSION_NOTES.md: "Per-session discretization using 40th and 60th percentiles."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th to 60th percentile), 2 (above 60th percentile). Bins with no valid tracking data (NaN) are mapped to class 0 rather than a separate "not visible" class.

ii.
```python
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
    return result
```

```python
'output_values': [
    ...
    ['low', 'mid', 'high'],  # tongue_y_position
]
```

iii. CONVERSION_NOTES.md: "Time bins without valid tracking data default to 0 (low)."

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking timestamps are filtered to the trial window [go + BEGIN_TIME - BIN_WIDTH, go + END_TIME + BIN_WIDTH] (slightly wider than the neural window), then digitized into the same bin edges as the neural data.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The tongue data is binned onto the same time grid as the neural data, ensuring alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with 0 good units are skipped. (2) Trials without neural data (outside obs_intervals) are excluded via the obs_to_trial mapping. (3) Tongue tracking bins with no valid frames (likelihood <= 0.5 or no frames in the bin) are set to class 0 (low). The AI does not explicitly handle NaN classification values (the loop checks `== 'good'` which would be False for NaN).

ii.
```python
if len(good_indices) == 0:
    return None

if i not in trials_with_neural:
    continue

if np.isnan(val):
    result[i] = 0  # no tongue visible = low position
```

iii. CONVERSION_NOTES.md discusses obs_intervals handling and all-zero neural data warnings but does not specifically mention NaN classification handling.

## 10-a. What are the most time-consuming steps of the code?

i. Loading spike times for each unit individually via `nwb.units['spike_times'][uid]` (one HDF5 read per unit) and processing each trial individually for firing rates (one `np.histogram` call per neuron per trial) are the most expensive operations.

ii.
```python
all_spike_times = [nwb.units['spike_times'][uid] for uid in good_indices]

for trial_idx in valid_trial_indices:
    fr, bin_centers = compute_firing_rates(spike_times_trial, go_cue, ...)
```

iii. N/A

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-neuron-per-trial firing rate computation could be vectorized. The reference solution computes all trials at once per neuron using a flattened edge array and `searchsorted`. The AI's code loops over both trials (outer) and neurons (inner via `compute_firing_rates`). Additionally, the tongue discretization loop over individual values could use `np.digitize`.

ii.
```python
# Per-trial loop
for trial_idx in valid_trial_indices:
    fr, bin_centers = compute_firing_rates(spike_times_trial, go_cue, ...)

# Per-neuron loop inside compute_firing_rates
for i, st in enumerate(spike_times_by_neuron):
    counts, _ = np.histogram(st_window, bins=bin_edges)

# Per-value loop in discretize_tongue_y
for i, val in enumerate(tongue_y_values):
    if np.isnan(val): ...
```

iii. N/A

## 10-c. What processing does the code repeat multiple times?

i. Bin edges and bin centers are recomputed for every trial inside `compute_firing_rates`, though they only depend on the bin width and window (which are constant). The `map_region_to_major` function is called twice for each unit's region label (once during initial collection and once during index building).

ii.
```python
# Called once per trial:
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

```python
# Called twice per unit:
all_region_labels.add(map_region_to_major(label))  # first pass
idx = np.array([brain_regions.index(map_region_to_major(label)) ...])  # second pass
```

iii. N/A

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI filters out early lick and ignore trials, which means the `early_lick` output is always 0 and `outcome` never has value 0 (ignore). Computing and storing these outputs wastes space and provides no information to the decoder. Additionally, spike times are clipped to the trial window per neuron per trial via a mask (`(st >= bin_edges[0]) & (st < bin_edges[-1])`), which is redundant since `np.histogram` handles out-of-range values.

ii.
```python
# Redundant masking before histogram
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
st_window = st[mask]
counts, _ = np.histogram(st_window, bins=bin_edges)
```

iii. N/A
