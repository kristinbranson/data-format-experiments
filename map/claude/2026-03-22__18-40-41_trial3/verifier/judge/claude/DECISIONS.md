# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files in the `data/` directory, organized by subject subdirectories (`sub-XXXXXX/`). It lists all subject directories, then lists NWB files within each, collecting `(subject, path)` tuples. Each file is opened with `pynwb.NWBHDF5IO` and processed via `process_session()`. Trials, units, behavioral events, and behavioral time series are all extracted from within each NWB file.

ii.
```python
def list_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir)
                      if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('sub-')])
    all_files = []
    for sub in subjects:
        sub_dir = os.path.join(data_dir, sub)
        nwb_files = sorted([os.path.join(sub_dir, f)
                           for f in os.listdir(sub_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append((sub, nwb_file))
    return all_files
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
trials = nwb.trials
units = nwb.units
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The AI noted in CONVERSION_NOTES.md that the dataset consists of 174 NWB files from 28 subjects organized in the DANDI 000363 format, and that pynwb is the standard reader for NWB files.

## 1-b. How are the data split into subjects?

i. Subject ID is extracted from `nwb.subject.subject_id` for each session. Subjects are collected in the order they are first encountered during processing, and each session is assigned an index into the subjects list.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
subjects = []
subject_idx = []
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI recognized that subject IDs come from the NWB subject field. The subjects list is built in encounter order rather than sorted, which could lead to non-deterministic ordering if session processing order changes.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Each session is identified by `nwb.identifier`. The AI processes all NWB files found in the data directory, applying session-level selection criteria to determine which sessions to keep.

ii.
```python
session_id = nwb.identifier
```

iii. The AI identified that sessions correspond to individual NWB files in the DANDI dataset.

## 1-d. How are the data split into trials?

i. Trials are taken from `nwb.trials`, which provides a table with one row per trial. The AI extracts trial metadata columns (start_time, stop_time, trial_instruction, outcome, early_lick, etc.) and verifies that the number of go cue events matches the number of trials.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials, f"Go times ({len(go_times)}) != trials ({n_trials})"
```

iii. Noted in CONVERSION_NOTES.md that go cue times match trial count, providing a 1:1 mapping.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of filtering:
1. **Session-level**: Sessions are skipped if correct rate < 65%, or if there are fewer than 50 correct left or 50 correct right trials (computed on "regular" trials excluding early lick, stim, auto/free water, and ignore trials).
2. **Trial-level**: Trials with `auto_water == 1` or `free_water == 1` are excluded. Trials with missing tone onset are excluded. Trials beyond the neural recording period (`go_time + T_END > max_recording_time + 1.0`) are excluded.

The AI does NOT use `obs_intervals` to filter trials without spike data.

ii.
```python
# Session selection criteria
MIN_CORRECT_RATE = 0.65
MIN_CORRECT_LEFT = 50
MIN_CORRECT_RIGHT = 50
```

```python
behav_valid = (auto_water == 0) & (free_water == 0)
...
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. The AI documented in CONVERSION_NOTES.md that the session selection criteria come from the paper (>65% correct, >=50 correct L/R). It stated that auto_water and free_water trials should be excluded, while keeping early lick, ignore, and photostim trials for the decoder task. The AI's approach to checking recording coverage uses `max_recording_time` from `obs_intervals` but does not use the per-trial obs_intervals matching used by the reference.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, accessed per unit via indexing. Only units where `classification == 'good'` and `anno_name` is non-empty are included. Go cue times from `BehavioralEvents/go_start_times` are used for temporal alignment.

ii.
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]
...
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. The AI noted that `classification == 'good'` in NWB is equivalent to passing the QC classifier. It additionally requires `anno_name` to be non-empty, based on the reference code's requirement for histology data.

## 2-b. How is the `neural` data processed?

i. For each trial, spike times are binned into 50ms windows around the go cue (-2.5s to +1.5s, 80 bins). Spike counts per bin are converted to firing rates in Hz by dividing by bin width. Processing loops over each neuron and each trial independently.

ii.
```python
def compute_firing_rates_vectorized(spike_times_list, go_time, t_start, t_end, bin_width, n_bins):
    n_neurons = len(spike_times_list)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
    bin_edges_end = bin_edges_start + bin_width

    for i, spk in enumerate(spike_times_list):
        if len(spk) == 0:
            continue
        mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
        spk_window = spk[mask]
        if len(spk_window) == 0:
            continue
        bin_idx = np.floor((spk_window - (go_time + t_start)) / bin_width).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)
        np.add.at(fr[i], bin_idx, 1)

    fr /= bin_width
    return fr
```

iii. The AI documented using 50ms bins as specified by the decoder task, which differs from the reference code's 40ms bins. This is a required difference per the instructions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by two criteria: `classification == 'good'` AND `anno_name` must be non-empty (not '' and not None). Sessions with no qualifying units are skipped.

ii.
```python
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code requires both ephys QC and histology (CCF coordinates), so anno_name must not be empty.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue. For each trial, the go cue time is used to define the absolute time window `[go_time + T_START, go_time + T_END]`, and spikes within that window are binned relative to the go cue.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
fr = compute_firing_rates_vectorized(
    spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
)
```

Inside `compute_firing_rates_vectorized`:
```python
bin_edges_start = go_time + t_start + np.arange(n_bins) * bin_width
```

iii. The AI correctly identified that NWB spike times are in absolute time and must be re-referenced to the go cue for each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins, 80 bins per trial covering -2.5s to +1.5s relative to go cue. No rebinning is applied — spikes are directly binned from raw spike times into the 50ms windows.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The AI noted this matches the decoder task specification, which overrides the reference code's 40ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (tone onset events from BehavioralEvents) and `go_start_times`. For each trial, the last `sample_start` event that falls within the trial window (between `trial_starts[i]` and `go_times[i]`) is used as the tone onset.

ii.
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]

tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]
```

iii. The AI documented that the last sample onset within each trial is used to handle cases where early licking causes the sample epoch to replay.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Time from tone onset is computed as: `bin_centers - tone_relative`, where `tone_relative = tone_time - go_time` (tone onset relative to go cue, typically negative). This gives the elapsed time since tone onset at each bin center.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2

tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time
time_from_tone = bin_centers - tone_relative
```

iii. The AI documented this as a continuous variable representing time since tone onset in go-cue-relative coordinates.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers defined relative to the go cue (`T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2`), so they share the same time axis.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
```

iii. The same bin grid is used for neural data and inputs, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `trials['photostim_onset']` and `trials['photostim_duration']` in the trials table. The onset is relative to trial start time (`trial_starts[trial_idx]`).

ii.
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
```

iii. The AI noted that photostim_onset values are stored as strings and can be 'N/A' for non-stimulated trials. It identified that the onset values are small numbers suggesting they are relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset is converted from trial-relative to go-cue-relative coordinates. A binary time series is created where bins with centers falling within [onset, onset+duration) are set to 1.0. Non-stimulated trials ('N/A') remain all zeros. The processing loops over each bin individually.

ii.
```python
if photostim_onset[trial_idx] != 'N/A':
    ps_onset = float(photostim_onset[trial_idx])
    ps_duration = float(photostim_duration[trial_idx])
    ps_onset_abs = trial_starts[trial_idx] + ps_onset
    ps_end_abs = ps_onset_abs + ps_duration
    ps_onset_rel = ps_onset_abs - go_time
    ps_end_rel = ps_end_abs - go_time

    for b in range(N_BINS):
        bc = bin_centers[b]
        if ps_onset_rel <= bc < ps_end_rel:
            photostim[b] = 1.0
```

iii. The AI documented this as a binary time series input indicating when photostimulation is active.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim binary values are assigned to the same bin centers as the neural data (relative to go cue), so they share the same temporal grid.

ii.
```python
bc = bin_centers[b]  # same bin_centers used for neural data
```

iii. Same bin grid ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trials['trial_instruction']` (left/right) and `trials['outcome']` (hit/miss/ignore). For hits, choice matches the instruction. For misses, choice is the opposite of instruction. For ignore trials, choice is assigned the instruction direction.

ii.
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0
else:  # ignore
    choice = 0 if instr == 'left' else 1  # assign instruction direction
```

iii. The AI decided to assign the instruction direction for ignore trials since there is no actual lick to determine a choice. CONVERSION_NOTES.md states: "Choice for ignore trials: Set to instruction direction (the 'correct' choice), since there's no actual lick."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as left=0, right=1 (two categories). It is a per-trial value broadcast across all 80 bins. The output has only 2 output values defined.

ii.
```python
output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
], dtype=np.int64)
```

```python
'output_values': [
    ['left', 'right'],           # choice
    ...
],
```

iii. The AI chose a 2-class encoding matching the instructions "left = 0, right = 1".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials['outcome']`, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes = trials['outcome'][:]
...
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

iii. Direct mapping from the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Per-trial value broadcast across all 80 bins.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
...
np.full(N_BINS, outcome_val, dtype=np.int64),
```

iii. Follows the instruction specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From `trials['early_lick']`, which contains 'early' or 'no early'.

ii.
```python
early_licks = trials['early_lick'][:]
...
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. Direct mapping from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1. Per-trial value broadcast across all 80 bins.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
...
np.full(N_BINS, early_val, dtype=np.int64),
```

iii. Follows the instruction specification.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `BehavioralTimeSeries/Camera0_side_TongueTracking`, which has columns (x, y, likelihood). Column 1 (y) is used for position and column 2 (likelihood) for visibility filtering.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The AI correctly identified the tongue tracking time series and its column structure.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI filters visible frames (likelihood > 0.5), computes 40th and 60th percentiles of the visible y-values across the entire session (raw frames, not binned means), and discretizes tongue y into 3 categories (0: below 40th, 1: 40th-60th, 2: above 60th). There is no "not visible" class.

ii.
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
```

iii. The AI documented using per-session percentiles with 40th/60th thresholds, consistent with the instructions. However, it takes percentiles over raw visible frames rather than over 50ms bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th-60th), 2 (above 60th). No fourth "not visible" category is used. When the tongue is not visible, the nearest frame's value is used instead due to `searchsorted` finding the closest timestamp.

ii.
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
    if ty < p40:
        tongue_y_trial[b] = 0
    elif ty < p60:
        tongue_y_trial[b] = 1
    else:
        tongue_y_trial[b] = 2
```

iii. The AI uses only 3 output values for tongue_y: `['low', 'mid', 'high']`. It does not distinguish between bins where the tongue is visible vs. not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin, the AI finds the closest tongue tracking frame using `searchsorted` on the absolute time of the bin center. This is a nearest-neighbor approach rather than averaging all frames within the bin.

ii.
```python
bc_abs = go_time + bin_centers[b]
t_idx = np.searchsorted(tongue_ts, bc_abs)
t_idx = min(t_idx, len(tongue_ts) - 1)
ty = tongue_y[t_idx]
```

iii. The AI uses nearest-frame lookup rather than averaging frames within each 50ms bin window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Sessions with no good neurons**: Skipped entirely.
- **Sessions failing behavioral criteria**: Skipped (correct rate < 65%, insufficient correct trials).
- **Trials with missing tone onset**: Excluded via `~np.isnan(tone_onset_per_trial)`.
- **Trials beyond recording period**: Excluded by comparing go_time + T_END to max_recording_time.
- **Unmapped brain regions**: Fall back to 'OtherCortex' with a warning printed.
- **NWB file with NaN classification** (session s4 of sub-440958): Skipped because no units pass the good filter.

ii.
```python
valid_mask &= ~np.isnan(tone_onset_per_trial)
...
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

```python
# Fallback for unmapped annotations
print(f'  WARNING: Unmapped annotation: "{anno_name}"')
return 'OtherCortex'
```

iii. The AI handles multiple edge cases but uses a coarser approach to recording coverage than the reference (max_recording_time + 1.0 tolerance vs. per-trial obs_intervals matching).

## 10-a. What are the most time-consuming steps of the code?

i. The code took 30.2 minutes for 174 sessions (144 processed). The main bottleneck is the per-trial, per-neuron loop in `compute_firing_rates_vectorized`, which is called once per trial rather than vectorized across all trials simultaneously. Individual session processing times range from ~3s to ~24s depending on neuron and trial counts.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

iii. The AI noted timing information in the output but the code is significantly slower than the reference (~30 min vs ~4 min) due to calling the firing rate computation per trial.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be improved:
1. The outer loop over trials in `process_session` calls `compute_firing_rates_vectorized` once per trial, requiring each neuron's spikes to be re-filtered for every trial. The reference vectorizes this by building all trial edges at once.
2. The photostim binary computation loops over bins individually instead of using vectorized comparison.
3. The tongue y-position computation loops over bins with `searchsorted` per bin.

ii.
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
```

```python
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. The reference solution vectorizes the neural binning across all trials simultaneously using `searchsorted` on a flattened edge array, which is dramatically faster.

## 10-c. What processing does the code repeat multiple times?

i. The most significant repetition is loading each neuron's spike times and filtering them for every trial individually, rather than loading once and binning across all trials. Each call to `compute_firing_rates_vectorized` re-reads the spike time arrays and re-applies the window mask.

ii.
```python
for trial_idx in valid_indices:
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
```

Inside the function, for each neuron:
```python
mask = (spk >= go_time + t_start) & (spk < go_time + t_end)
```

iii. This repeated filtering of the same spike arrays for each trial is the primary source of inefficiency.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The session-level behavioral performance metrics (correct_rate, correct_left, correct_right) are computed for session selection but are not included in the final output beyond a `correct_rate` field in session results. The per-session `processing_plots` functionality exists but is only called when `--show-processing` is specified. The extensive `REGION_MAPPING` dictionary maps annotations to broad categories, which is extra processing compared to using the annotation directly.

ii.
```python
correct_rate = correct_regular / n_regular
correct_left = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'left'))
correct_right = np.sum(regular_mask & (outcomes == 'hit') & (instructions == 'right'))
```

iii. The session selection criteria computation is the main unnecessary processing since the reference solution does not apply these filters.
