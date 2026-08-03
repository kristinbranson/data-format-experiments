# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB (Neurodata Without Borders) files stored in `/app/data/sub-XXXXXX/` directories. It scans all subdirectories starting with `sub-` under the data directory, and within each subject directory, collects all `.nwb` files. Each NWB file corresponds to one behavioral session. The files are opened using `pynwb.NWBHDF5IO`, and trial information, neural units, behavioral events, and behavioral time series are extracted from the NWB structure.

ii.
```python
def list_nwb_files(data_dir):
    """List all NWB files organized by subject."""
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
```

iii. The AI noted in CONVERSION_NOTES.md that the reference code works with `.mat` files from DataJoint export, not NWB directly. Since the available data was in NWB format, the AI mapped NWB fields to the equivalent variables used in the reference code. This is a necessary adaptation since the data format differs from the reference code's expected format.

## 1-b. How are the data split into subjects?

i. Subjects are identified by their subdirectory names under the data directory (e.g., `sub-440956`). Each NWB file stores the subject ID in `nwb.subject.subject_id`. Unique subjects are collected during dataset building, and each session is mapped to its subject via `subject_idx`.

ii.
```python
subject_id = nwb.subject.subject_id if nwb.subject else os.path.basename(nwb_path).split('_')[0]
```

```python
# In build_dataset():
for sess in session_results:
    if sess['subject_id'] not in subjects:
        subjects.append(sess['subject_id'])
    subject_idx.append(subjects.index(sess['subject_id']))
```

iii. The AI uses the NWB file's subject metadata field directly. The fallback to parsing the filename is a reasonable safeguard for edge cases.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The AI processes each NWB file independently via `process_session()`, which extracts all data for that session. Sessions are identified by `nwb.identifier`.

ii.
```python
session_id = nwb.identifier
```

iii. The AI noted that DANDI archive has 174 NWB files while the paper reports 173 sessions. After applying session selection criteria, 144 sessions pass.

## 1-d. How are the data split into trials?

i. Trials are extracted from the NWB `trials` table. Each trial has a start_time, stop_time, and associated metadata (instruction, outcome, early_lick, etc.). Go cue times from `BehavioralEvents` are used as alignment points.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
trial_starts = trials['start_time'][:]
trial_stops = trials['stop_time'][:]
instructions = trials['trial_instruction'][:]
outcomes = trials['outcome'][:]
early_licks = trials['early_lick'][:]
auto_water = trials['auto_water'][:]
free_water = trials['free_water'][:]
```

```python
be = nwb.acquisition['BehavioralEvents']
go_times = be.time_series['go_start_times'].timestamps[:]
assert len(go_times) == n_trials
```

iii. The AI verified that the number of go cue events matches the number of trials in the trials table.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two levels of trial filtering:
1. **Session-level selection**: Correct rate must be >65%, with >=50 correct left and >=50 correct right trials. Correct rate is computed on "regular" trials only (excluding auto_water, free_water, photostim, early_lick, and ignore trials).
2. **Trial-level filtering**: Only auto_water and free_water trials are excluded. Early lick, photostim, and ignore trials are KEPT (because they are needed as decoder inputs/outputs). Trials without a valid tone onset (NaN) are excluded. Trials beyond the neural recording period are excluded.

ii.
```python
# Session selection
behav_valid = (auto_water == 0) & (free_water == 0)
regular_mask = behav_valid.copy()
for i in range(n_trials):
    if photostim_onset[i] != 'N/A':
        regular_mask[i] = False
    if early_licks[i] == 'early':
        regular_mask[i] = False
regular_mask &= (outcomes != 'ignore')

correct_rate = correct_regular / n_regular
if correct_rate < MIN_CORRECT_RATE:  # 0.65
    return None

# Trial filtering
valid_mask = behav_valid.copy()
valid_mask &= ~np.isnan(tone_onset_per_trial)
for i in range(n_trials):
    trial_end_abs = go_times[i] + T_END
    if trial_end_abs > max_recording_time + 1.0:
        valid_mask[i] = False
```

iii. The AI justified keeping early lick, photostim, and ignore trials because the decoder task specifies early_lick as an output, photostim as an input, and outcome (including ignore) as an output. The session selection criteria (>65% correct, >=50 correct L/R) match the paper's criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']` in the NWB file, filtered by `units['classification']` (must be 'good') and `units['anno_name']` (must be non-empty for brain region assignment).

ii.
```python
units = nwb.units
classification = units['classification'][:]
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

spike_times_all = units['spike_times']
spike_times_good = [spike_times_all[idx] for idx in good_indices]
```

iii. The AI noted that in NWB, `classification == 'good'` is equivalent to passing the QC classifier used in the reference code, and that neurons must have valid CCF annotations.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins within a [-2.5, +1.5]s window around the go cue. Spike counts per bin are divided by the bin width (0.05s) to produce firing rates in Hz.

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
    fr /= bin_width  # convert to firing rate (Hz)
    return fr
```

iii. The AI chose 50ms bins as specified by the decoder task instructions, noting this differs from the reference code's 40ms bins with 3.4ms stride. The AI documented this as an intentional deviation required by the decoder task specification.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by two criteria:
1. `classification == 'good'` (QC classifier filter)
2. Non-empty `anno_name` (must have CCF brain region annotation)

Sessions with zero good neurons are skipped.

ii.
```python
good_mask_units = classification == 'good'
anno_names = units['anno_name'][:]
good_mask_units &= np.array([a != '' and a is not None for a in anno_names])
good_indices_units = np.where(good_mask_units)[0]

if len(good_indices_units) == 0:
    print(f'  SKIP: no good neurons')
    io.close()
    return None
```

iii. The AI noted that the NWB `classification` field is equivalent to the reference code's QC classifier output. This is consistent with the spike sorting QC paper's approach.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue time is used as the reference point (t=0), and spike times are binned in a window of [-2.5s, +1.5s] relative to the go cue. The go cue times come from `BehavioralEvents['go_start_times']`.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
# ...
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(
        spike_times_good, go_time, T_START, T_END, BIN_WIDTH, N_BINS
    )
```

iii. The AI correctly identified the go cue as the alignment event specified in the decoder task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.05s), producing 80 bins for the 4-second window [-2.5s, +1.5s]. No temporal rebinning is applied -- spikes are directly binned into the 50ms bins from raw spike times, not rebinned from a finer resolution.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins (decoder task spec)
T_START = -2.5
T_END = 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)  # 80 bins
```

iii. The reference code uses 40ms bins with 3.4ms stride (overlapping bins). The AI chose 50ms non-overlapping bins as specified by the decoder task. This is an allowed deviation per the instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents['sample_start_times'].timestamps` (the tone/sample onset event times) and `go_start_times` (the go cue times for alignment).

ii.
```python
sample_starts = be.time_series['sample_start_times'].timestamps[:]
```

iii. The AI identifies the sample (tone) onset within each trial by finding events within the trial's time window.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial:
1. Find all sample_start events between the trial start and go cue.
2. Take the LAST sample onset (handles replays from early licking).
3. Compute the tone onset time relative to the go cue.
4. For each time bin, compute `time_from_tone = bin_center - tone_onset_relative_to_go_cue`.

ii.
```python
tone_onset_per_trial = np.full(n_trials, np.nan)
for i in range(n_trials):
    in_trial = sample_starts[(sample_starts >= trial_starts[i]) & (sample_starts <= go_times[i])]
    if len(in_trial) > 0:
        tone_onset_per_trial[i] = in_trial[-1]

# Per trial:
tone_time = tone_onset_per_trial[trial_idx]
tone_relative = tone_time - go_time  # negative
time_from_tone = bin_centers - tone_relative  # time since tone onset
```

iii. The AI correctly handles early lick replays by taking the last sample onset. Trials with no valid tone onset are excluded via `valid_mask &= ~np.isnan(tone_onset_per_trial)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time_from_tone is computed at the same bin centers as the neural data, so they are inherently aligned. Both use the same 80 time bins defined by `bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2`.

ii.
```python
bin_centers = T_START + np.arange(N_BINS) * BIN_WIDTH + BIN_WIDTH / 2
# ...
time_from_tone = bin_centers - tone_relative
```

iii. The alignment is automatic since both neural and input data use the same time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from `trials['photostim_onset']` and `trials['photostim_duration']` in the NWB trials table. Also uses `trials['start_time']` to convert photostim onset (relative to trial start) to absolute time.

ii.
```python
photostim_onset = trials['photostim_onset'][:]
photostim_duration = trials['photostim_duration'][:]
```

iii. The AI identified that photostim_onset values are stored as strings ('N/A' for no stimulation, or numeric values).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For trials with photostimulation:
1. Convert photostim_onset from trial-relative to absolute time: `ps_onset_abs = trial_starts[trial_idx] + ps_onset`
2. Compute photostim end: `ps_end_abs = ps_onset_abs + ps_duration`
3. Convert to go-cue-relative coordinates
4. For each time bin, set photostim=1 if the bin center falls within the stimulation window, else 0.

ii.
```python
photostim = np.zeros(N_BINS, dtype=np.float32)
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

iii. The AI noted uncertainty about whether photostim_onset is relative to trial start or absolute, but concluded from data exploration that values are small (~1.8s), suggesting they are relative to trial start.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary signal is computed at the same bin centers as the neural data, ensuring temporal alignment.

ii.
```python
for b in range(N_BINS):
    bc = bin_centers[b]
    if ps_onset_rel <= bc < ps_end_rel:
        photostim[b] = 1.0
```

iii. Same bin centers used for neural and input data ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials['trial_instruction']` (left/right) and `trials['outcome']` (hit/miss/ignore).

ii.
```python
instr = instructions[trial_idx]
outcome = outcomes[trial_idx]
```

iii. The AI uses both instruction and outcome to infer the animal's actual lick direction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is determined by combining instruction and outcome:
- Hit: choice matches instruction (left instruction -> left lick = 0, right instruction -> right lick = 1)
- Miss: choice is opposite of instruction (left instruction -> right lick = 1, right instruction -> left lick = 0)
- Ignore: choice is set to instruction direction (no lick occurred)

The choice value is constant across all time bins within a trial.

ii.
```python
if outcome == 'hit':
    choice = 0 if instr == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instr == 'left' else 0  # wrong lick = opposite
else:  # ignore
    choice = 0 if instr == 'left' else 1  # assign instruction direction

output_data = np.array([
    np.full(N_BINS, choice, dtype=np.int64),
    ...
])
```

iii. The AI's logic for hit and miss trials is reasonable: on hit trials the animal licked correctly (matching instruction), on miss trials the animal licked the wrong direction. For ignore trials, the AI assigns the instruction direction since there was no lick. This is a debatable choice -- there is no actual lick direction on ignore trials.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived directly from `trials['outcome']` which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcomes = trials['outcome'][:]
```

iii. Direct mapping from the NWB trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple categorical mapping: ignore=0, miss=1, hit=2. The value is constant across all time bins in a trial.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}[outcome]
```

iii. The mapping matches the decoder task specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `trials['early_lick']` which contains 'early' or 'no early'.

ii.
```python
early_licks = trials['early_lick'][:]
```

iii. Direct mapping from the NWB trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: 'no early' -> 0, 'early' -> 1. Constant across all time bins in a trial.

ii.
```python
early_val = 0 if early_licks[trial_idx] == 'no early' else 1
```

iii. The mapping matches the decoder task specification exactly.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `BehavioralTimeSeries['Camera0_side_TongueTracking']` in the NWB file. This time series contains (x, y, likelihood) for each video frame.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
has_tongue = 'Camera0_side_TongueTracking' in bts.time_series
if has_tongue:
    tongue_data = bts.time_series['Camera0_side_TongueTracking'].data[:]
    tongue_ts = bts.time_series['Camera0_side_TongueTracking'].timestamps[:]
    tongue_y = tongue_data[:, 1]
    tongue_likelihood = tongue_data[:, 2]
```

iii. The AI correctly identifies the tongue tracking data from the side camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The tongue y-position is extracted from the tracking data at each time bin, then discretized into 3 categories per session:
1. Compute per-session thresholds using 40th and 60th percentiles of y-position for frames with tongue likelihood > 0.5.
2. For each time bin, find the closest tongue tracking frame using `np.searchsorted`.
3. Classify: y < p40 -> 0 (low), p40 <= y < p60 -> 1 (mid), y >= p60 -> 2 (high).

ii.
```python
tongue_visible = tongue_likelihood > 0.5
if np.sum(tongue_visible) > 100:
    visible_y = tongue_y[tongue_visible]
    p40 = np.percentile(visible_y, 40)
    p60 = np.percentile(visible_y, 60)
```

iii. The AI uses a likelihood threshold of 0.5 to determine when the tongue is visible, and computes percentiles only on visible frames. If fewer than 100 visible frames exist, it falls back to all frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentile thresholds (40th and 60th) on visible tongue y-positions. For each time bin:
- 0 (low): y < 40th percentile
- 1 (mid): 40th <= y < 60th percentile
- 2 (high): y >= 60th percentile

ii.
```python
ty = tongue_y[t_idx]
if ty < p40:
    tongue_y_trial[b] = 0
elif ty < p60:
    tongue_y_trial[b] = 1
else:
    tongue_y_trial[b] = 2
```

iii. The thresholds match the decoder task specification. The percentiles are computed over the entire session's visible tongue data, not per-trial.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each neural time bin, the closest tongue tracking frame (by timestamp) is found using `np.searchsorted`, and the tongue y-position at that frame is discretized.

ii.
```python
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    t_idx = min(t_idx, len(tongue_ts) - 1)
    ty = tongue_y[t_idx]
```

iii. The AI uses nearest-neighbor interpolation (searchsorted finds the next frame). Since the video frame rate is ~300 Hz (3.3ms between frames), this is much finer than the 50ms neural bins, so the alignment is adequate.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
1. **Missing tongue tracking**: If `Camera0_side_TongueTracking` is not present, tongue_y defaults to 1 (mid) for all bins.
2. **Missing tone onset**: Trials with NaN tone onset (no valid sample_start found) are excluded.
3. **Trials beyond recording**: Trials where `go_time + 1.5s` exceeds `max_recording_time + 1.0s` are excluded.
4. **Empty spike trains**: Neurons with no spikes in the window get zero firing rate.
5. **Zero-neural-data trials**: 126/74894 trials (0.17%) have all-zero neural data, concentrated in session 34. These are kept but flagged as warnings.
6. **Unmapped brain regions**: CCF annotations not in the mapping dictionary fall back to 'OtherCortex'.

ii.
```python
# Missing tongue
if not has_tongue:
    tongue_y_trial = np.ones(N_BINS, dtype=np.float32)  # default to middle

# Missing tone onset
valid_mask &= ~np.isnan(tone_onset_per_trial)

# Recording coverage
if trial_end_abs > max_recording_time + 1.0:
    valid_mask[i] = False

# Empty spikes
if len(spk) == 0:
    continue
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md and verified their impact is minimal.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the firing rate computation in `compute_firing_rates_vectorized()`, which loops over all neurons for each trial. The tongue y-position discretization also loops over time bins per trial. Each session takes 5-10 seconds, and the full conversion takes approximately 30 minutes for 144 sessions.

ii.
```python
for i, spk in enumerate(spike_times_list):
    # per-neuron processing
    ...

for b in range(N_BINS):
    # per-bin tongue processing
    ...
```

iii. The AI estimated conversion time at 5-10s per session, ~30 min total, which was within the 15-minute guideline (approximately).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops could be further vectorized:
1. The neuron loop in `compute_firing_rates_vectorized()` iterates per-neuron. The inner operations (masking, binning) could potentially be vectorized using multi-dimensional histogramming.
2. The tongue y-position discretization loop over bins could be vectorized using `np.searchsorted` on the full array of bin times at once and then applying vectorized comparisons.

ii.
```python
# Neuron loop (could be vectorized)
for i, spk in enumerate(spike_times_list):
    ...

# Tongue bin loop (could be vectorized)
for b in range(N_BINS):
    bc_abs = go_time + bin_centers[b]
    t_idx = np.searchsorted(tongue_ts, bc_abs)
    ...
```

iii. Despite these opportunities, the AI kept the per-neuron loop because spike times are ragged arrays (different lengths per neuron), making full vectorization non-trivial.

## 10-c. What processing does the code repeat multiple times?

i. The code reloads the full tongue tracking data for each session but processes it per-trial within a loop. The `compute_firing_rates_vectorized()` function is called once per trial, re-applying the same spike windowing and binning logic each time. The spike times for each neuron are re-filtered with `(spk >= go_time + t_start) & (spk < go_time + t_end)` for every trial.

ii.
```python
for trial_idx in valid_indices:
    go_time = go_times[trial_idx]
    fr = compute_firing_rates_vectorized(spike_times_good, go_time, ...)
```

iii. A more efficient approach would be to pre-sort spike times and use binary search to find the window boundaries, reducing redundant filtering. However, the current approach works and completes in reasonable time.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `bin_edges_end` in `compute_firing_rates_vectorized()` but never uses it. The code also loads `trial_stops` but only uses `trial_starts` for photostim computation. The photostim binary signal is computed with a Python loop over bins, which could be done with a simple vectorized comparison. Processing plots are generated only for the first 2 sessions but the infrastructure exists for all sessions.

ii.
```python
bin_edges_end = bin_edges_start + bin_width  # computed but never used

trial_stops = trials['stop_time'][:]  # loaded but not used
```

iii. These are minor inefficiencies that don't affect correctness, only code cleanliness.
