# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over subject directories (`sub-*`) in sorted order, then over NWB files within each subject directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and processed by `process_session()`. Trials, units, behavioral events, and tongue tracking data are all read from within each NWB file.

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

iii. The AI recognized NWB as the published format and used `pynwb` to read it. The iteration over subject directories and then NWB files is functionally equivalent to a glob over the dataset.

## 1-b. How are the data split into subjects?

i. The AI uses `nwb.subject.description` (e.g., `'SC015'`) as the subject identifier, rather than `nwb.subject.subject_id` (a numeric string like `'440956'`). Unique subjects are collected in order of first appearance using `OrderedDict.fromkeys`.

ii.
```python
subject_id = nwb.subject.description  # e.g. 'SC015' - mouse name from description
```

```python
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The agent's trajectory shows it chose `nwb.subject.description` to get the mouse name (e.g., SC015) rather than the numeric `subject_id`. The agent noted this as the "mouse name from description."

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI uses `nwb.session_id` as the session identifier.

ii.
```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. The AI correctly identified the one-file-per-session structure of the dataset.

## 1-d. How are the data split into trials?

i. Trials are read from the NWB trials table. The AI builds a mapping from `obs_intervals` to trial indices to identify which trials have neural data.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
```

```python
obs_to_trial = build_obs_to_trial_map(nwb, good_indices)
trials_with_neural = set(t for t in obs_to_trial if t >= 0)
```

iii. The AI recognized that `obs_intervals` may have fewer entries than trials and built a mapping between them using start time matching with 0.5s tolerance.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three layers of filtering:
1. **Session-level selection**: Sessions must have >65% correct on control non-early-lick trials, and ≥50 correct lick-left and lick-right trials.
2. **Trial-level filtering**: Early lick trials and no-response (ignore) trials are excluded.
3. **Neural data availability**: Trials must be in `obs_intervals`.

The AI does NOT filter `free_water` trials.

ii.
```python
# Session selection
if perf <= MIN_PERF or correct_left < MIN_CORRECT_PER_SIDE or correct_right < MIN_CORRECT_PER_SIDE:
    return None
```

```python
# Trial filtering
if early != 'no early':
    continue
if outcome == 'ignore':
    continue
```

iii. From the trajectory, the agent stated: "Per methods: 'Early lick trials and no response trials were excluded for analysis'" and applied the session selection criteria from the method paper (>65% performance, ≥50 correct per side). The agent also explicitly decided not to filter `free_water` trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units['spike_times']`, the sorted spike times of each unit. Only units with `classification == 'good'` are used. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
# which calls:
return [nwb.units['spike_times'][uid] for uid in unit_indices]
```

iii. The AI correctly identified `spike_times` as the source of neural data.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms bins using `np.histogram` and divided by the bin width to produce firing rates in Hz. The function `compute_firing_rates` is called per trial, per neuron, computing rates for the window from -2.5s to +1.5s relative to the go cue.

ii.
```python
def compute_firing_rates(spike_times_by_neuron, go_cue_time, begin_time, end_time, bin_width):
    n_bins = int(round((end_time - begin_time) / bin_width))
    bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
    ...
    for i, st in enumerate(spike_times_by_neuron):
        ...
        counts, _ = np.histogram(st_window, bins=bin_edges)
        fr[i, :] = counts.astype(np.float32) / bin_width
    return fr, bin_centers
```

iii. The AI implemented standard spike binning and rate computation. No smoothing or normalization is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. Sessions with 0 good units are skipped.

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    return None
```

iii. The AI uses the same `classification` field as the reference, which is the spike-sorting QC classifier from ChenLiuEtAl2023.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. The bin edges are placed at `go_cue_time + begin_time` to `go_cue_time + end_time`. However, the AI also clips spike times to the `obs_intervals` for each trial before computing firing rates.

ii.
```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
```

```python
spike_times_list = []
t_start, t_stop = obs_intervals[obs_idx]
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```

iii. The AI correctly aligns to the go cue. The additional clipping to obs_intervals is an extra step not in the reference but shouldn't affect results since obs_intervals covers the trial window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms bins from -2.5s to +1.5s relative to go cue, giving 80 bins per trial. No rebinning is applied — spike times are binned directly into this grid.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5  # relative to go cue
END_TIME = 1.5    # relative to go cue
```

iii. Matches the instructions exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` in `BehavioralEvents`, together with the go cue time and trial start time. The AI selects the last `sample_start_times` that falls between the trial's `start_time` and its go cue.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback
```

iii. The AI recognized that early licks replay the sample epoch, so a trial can have multiple tone onsets. It takes the last one within the trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The bin centers (relative to go cue) are shifted by the tone onset time (also relative to go cue) to give time from tone onset at each bin.

ii.
```python
tone_onset_rel = tone_onset - go_cue  # relative to go cue
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
```

iii. Straightforward subtraction to convert from go-cue-relative to tone-onset-relative time.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers defined relative to the go cue, so they share the same time grid.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. N/A

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, with `start_time` used to convert from trial-relative to absolute times.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. The AI correctly identified that photostim_onset is stored as a string relative to trial start, with 'N/A' for non-stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series is created where bins with centers falling between the onset and offset of photostimulation are set to 1.0, all others to 0.0.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
...
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The AI correctly creates a time-varying binary input.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The onset and offset are converted to go-cue-relative time and compared against the same bin centers used for neural data.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. N/A

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore). A hit means the animal licked the instructed side; a miss means it licked the opposite side. Since ignore trials are filtered out, choice is always left or right.

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

iii. The agent noted that choice is not stored directly and must be derived from instruction and outcome. Because ignore trials are excluded, the "no lick" category does not appear.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is coded as 0=left, 1=right (only 2 values). It is repeated across all time bins to create a (1, n_bins) shaped output.

ii.
```python
output_trial[0, :] = base_output[0]  # choice (constant across time)
```

```python
'output_values': [
    ['left', 'right'],           # choice
    ...
]
```

iii. The AI only defines 2 output values for choice (left, right), missing the "no lick" category since ignore trials are excluded.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains 'hit', 'miss', or 'ignore'.

ii.
```python
outcome = trials['outcome'][trial_idx]
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. The AI reads outcome directly from the trials table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to integers: ignore=0, miss=1, hit=2. Repeated across time bins. However, since ignore trials are filtered out, outcome value 0 never appears in the data.

ii.
```python
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```

```python
'output_values': [
    ...
    ['ignore', 'miss', 'hit'],   # outcome
    ...
]
```

iii. The mapping matches the instructions, but the 'ignore' category is effectively absent due to trial filtering.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'no early' or 'early'.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. Read directly from the trials table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to 0=no, 1=yes. Repeated across time bins. However, since early lick trials are filtered out, the value is always 0.

ii.
```python
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

```python
'output_values': [
    ...
    ['no', 'yes'],               # early_lick
    ...
]
```

iii. The mapping matches the instructions, but the 'yes' category is effectively absent due to trial filtering.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, column 1 (tongue_y) and column 2 (likelihood).

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The AI correctly identified the tongue tracking data source.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with `likelihood > 0.5` are kept (note: strict inequality). Tongue y is averaged within each 50ms time bin. Session-level 40th and 60th percentiles are computed from the concatenated valid bin means across all valid trials (not from the full session recording). The bin means are then discretized into 3 classes using those percentiles.

ii.
```python
lh_mask = tw_lh > 0.5
tw_y_good = tw_y[lh_mask]
...
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)
```

iii. The AI computes percentiles from trial bin means rather than session-wide bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Three categories: 0 (below 40th percentile), 1 (40th to 60th percentile), 2 (above 60th percentile). NaN values (no visible tongue) are mapped to class 0 rather than a separate "not visible" class.

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
    ['low', 'mid', 'high'],      # tongue_y_position
]
```

iii. The AI maps NaN (no visible tongue) to class 0 ("low") rather than defining a 4th "not visible" class. The output_values labels are 'low', 'mid', 'high' rather than descriptive percentile-based names.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are used to assign tongue frames to the same go-cue-relative bins as the neural data. The AI uses `np.digitize` against the bin edges.

ii.
```python
trial_window_start = go_cue + BEGIN_TIME - BIN_WIDTH
trial_window_end = go_cue + END_TIME + BIN_WIDTH
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
...
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. The AI adds a buffer of one bin width on each side of the trial window when selecting tongue frames, which may include frames slightly outside the intended window but doesn't affect the binning since `np.digitize` assigns to the correct bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases:
- **Sessions with 0 good units**: Skipped entirely.
- **Trials without neural data**: Excluded via `obs_intervals` mapping.
- **Tongue frames with low likelihood**: Set to NaN; bins with no visible frames are assigned class 0 (not a separate "not visible" class).
- **Missing tone onset**: Falls back to `go_cue - 1.85` as the typical sample-delay duration.
- **`free_water` trials**: Not filtered out.

ii.
```python
if len(good_indices) == 0:
    return None
```

```python
if i not in trials_with_neural:
    continue
```

```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback
```

iii. The AI handles missing data by either skipping (sessions/trials) or using fallback values (tone onset). The `free_water` trials are not filtered, and tongue NaN is mapped to class 0 rather than a separate class.

## 10-a. What are the most time-consuming steps of the code?

i. Reading spike times from each NWB file and computing firing rates per trial are the most time-consuming. The agent initially had performance issues with per-trial spike time extraction and had to optimize by preloading all spike times for good units.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
```

iii. From the trajectory, the agent noted that "reading spike times individually for each neuron and trial from NWB is very slow" and optimized by batch-loading spike times.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two main loops: (1) the per-neuron, per-trial firing rate computation using `np.histogram` per neuron per trial, and (2) the per-bin tongue y averaging loop. Both could potentially be vectorized.

ii.
```python
# Per-neuron loop in compute_firing_rates
for i, st in enumerate(spike_times_by_neuron):
    ...
    counts, _ = np.histogram(st_window, bins=bin_edges)
```

```python
# Per-bin tongue loop
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The `compute_firing_rates` function is called per trial AND loops per neuron — this is a double loop (trials x neurons) that could be vectorized by flattening bin edges across trials as the reference does.

## 10-c. What processing does the code repeat multiple times?

i. The `compute_firing_rates` function is called once per trial, repeating the bin edge computation each time. The `map_region_to_major` function is called once per neuron per session during assembly AND was already applied during initial processing.

ii.
```python
# Called once per trial
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
```

```python
# Called twice for brain regions
brain_region_labels.append(str(anno))  # in process_session
map_region_to_major(label)  # in convert_data assembly
```

iii. Bin edges and centers are recomputed for each trial, though they share the same relative structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI clips spike times to `obs_intervals` before computing firing rates, which is an extra filtering step. Since the neural binning window is within the trial window (which is within obs_intervals), this clipping is unnecessary. Additionally, the `perf` (performance) value computed per session is stored in the result but not included in the final output.

ii.
```python
# Spike times clipped to obs_intervals
t_start, t_stop = obs_intervals[obs_idx]
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```

```python
'perf': perf,  # stored but not used in final output
```

iii. The obs_intervals clipping doesn't change results but adds unnecessary computation.
