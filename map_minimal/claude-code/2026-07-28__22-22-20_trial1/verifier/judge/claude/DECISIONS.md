# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads NWB files from `/app/data/` using PyNWB. It iterates over subject directories (`sub-*`), then over `.nwb` files within each directory. Each NWB file is opened with `pynwb.NWBHDF5IO` and the trial table, units table, behavioral events, and tongue tracking time series are extracted.

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

iii. The AI identified that data are stored in NWB format on DANDI, with one NWB file per session. The reference code used MATLAB `.mat` files exported from DataJoint, but the data content is equivalent. The AI adapted the loading accordingly.

## 1-b. How are the data split into subjects (mice)?

i. Each NWB file contains a subject description field. The AI extracts the subject ID from `nwb.subject.description` (e.g., `'SC015'`). Unique subjects are collected across all sessions, and a `subject_idx` array maps each session to its subject.

ii.
```python
subject_id = nwb.subject.description
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The AI used `nwb.subject.description` which contains the mouse name (e.g., `'SC015'`), rather than `nwb.subject.subject_id` which contains a numeric ID (e.g., `'440956'`). This gives human-readable subject identifiers consistent with the paper.

## 1-c. How are the data split into sessions?

i. Each NWB file represents one session. The session ID is taken from `nwb.session_id`. All data within one NWB file (trials, units, behavioral events) belong to that session.

ii.
```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
```

iii. The DANDI dataset organizes data as one NWB file per session, so this is a natural split.

## 1-d. How are the data split into trials?

i. Trials are defined by the NWB trial table. The AI iterates over valid trial indices and processes each trial individually, extracting neural data, inputs, and outputs per trial.

ii.
```python
for trial_idx in valid_trial_indices:
    go_cue = go_start_times[trial_idx]
    obs_idx = trial_to_obs[trial_idx]
    spike_times_trial = get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx)
    fr, bin_centers = compute_firing_rates(spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH)
```

iii. The AI handles the obs_intervals-to-trial mapping carefully, since some NWB files have fewer obs_intervals than trials (due to multi-session recordings within one file). Only trials with corresponding neural data are processed.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes early lick trials (`early_lick != 'no early'`) and no-response/ignore trials (`outcome == 'ignore'`). Additionally, only trials with neural data coverage (present in obs_intervals) are included. The AI does NOT filter auto_water or free_water trials.

ii.
```python
if early != 'no early':
    continue
if outcome == 'ignore':
    continue
```

iii. The AI cited the methods text: "Early lick trials and no response trials were excluded for analysis." The reference code's downstream analysis (`get_regular_trial_mask`) additionally filters auto_water, free_water, and stimulation trials, but the AI only applied the filters explicitly mentioned in the methods text. Stimulation trials are kept because the instructions require photostimulation as a decoder input.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times of individual units in the NWB `units` table, specifically the `spike_times` column for units classified as `'good'`.

ii.
```python
all_spike_times = preload_spike_times(nwb, good_indices)
# ...
return [nwb.units['spike_times'][uid] for uid in unit_indices]
```

iii. The AI preloads all spike times for good units into memory for efficiency, then extracts per-trial spike times based on obs_intervals.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins and divided by bin width to produce firing rates in Hz. The window is -2.5s to +1.5s relative to go cue onset, yielding 80 time bins per trial.

ii.
```python
n_bins = int(round((end_time - begin_time) / bin_width))  # 80
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
counts, _ = np.histogram(st_window, bins=bin_edges)
fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. The AI noted this is "simpler than the sliding kernel approach in the original preprocessing code but appropriate for the decoder task." The reference code used sliding histogram with 40ms width and 3.4ms stride, but the instructions specify 50ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` in the NWB units table are included. Sessions with 0 good units are skipped.

ii.
```python
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    io.close()
    return None
```

iii. The AI identified that the `'good'` classification in NWB corresponds to the classifier-based QC described in the paper and reference code (`qc_mode='classifier'`). The paper describes region-specific logistic regression classifiers trained on manually curated labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. Spike times are absolute in the NWB file; the AI subtracts the go cue time to create relative bin edges, then histograms spikes into those bins.

ii.
```python
go_cue = go_start_times[trial_idx]
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. The go cue times are extracted from `BehavioralEvents['go_start_times']`. The reference code's raw data had spike times already aligned to go cue, so this step is an adaptation to the NWB format.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 50ms (0.05s), producing 80 bins for the 4.0s window (-2.5 to +1.5s). No rebinning is applied; spikes are directly histogrammed into these bins.

ii.
```python
BIN_WIDTH = 0.05  # 50 ms bins
BEGIN_TIME = -2.5
END_TIME = 1.5
```

iii. The instructions specify "50-ms-width bins for computing firing rates." The reference code used 40ms width with 3.4ms stride (sliding window), but the AI correctly follows the instructions for the decoder task.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `BehavioralEvents['sample_start_times']` timestamps and `BehavioralEvents['go_start_times']` timestamps, plus trial start times.

ii.
```python
be = nwb.acquisition['BehavioralEvents']
go_start_times = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
```

iii. The AI uses `sample_start_times` as the tone onset events, matching the paper's description of the auditory sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI finds the last `sample_start_time` that falls between trial start and go cue (to avoid replayed epochs from early licking). It then computes time from tone onset at each bin center. A fallback of `go_cue - 1.85` is used if no sample_start is found.

ii.
```python
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85  # fallback

tone_onset_rel = tone_onset - go_cue
time_from_tone = bin_centers - tone_onset_rel
```

iii. The agent identified that early licking triggers tone replays, so there can be multiple sample_start_times per trial. Taking the last one before go cue gives the actual (non-replayed) tone onset. The typical offset is ~1.85s before go cue (0.65s sample + 1.2s delay).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The tone onset time is converted to go-cue-relative coordinates, then subtracted from the bin centers (which are also go-cue-relative) to produce time-from-tone-onset at each bin. This is stored as a continuous time-varying signal with shape `(n_bins,)`.

ii.
```python
time_from_tone = bin_centers - tone_onset_rel  # time since tone onset at each bin
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)  # (2, n_bins)
```

iii. Both neural bin centers and tone onset are in the same go-cue-relative coordinate system, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Derived from the trial table columns `photostim_onset`, `photostim_duration`, and trial `start_time`.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
# ...
onset_val = float(photostim_onset_trial)
dur_val = float(trials['photostim_duration'][trial_idx])
```

iii. The AI identified that `photostim_onset` is `'N/A'` for control trials and a numeric value (relative to trial start) for stimulation trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, if photostim_onset is not `'N/A'`, the onset time is converted from trial-start-relative to go-cue-relative coordinates. A binary vector is created where bins within the photostim window are set to 1.0, all others to 0.0.

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

iii. The AI verified that photostim onset is ~-1.2s relative to go cue with 0.5s duration, consistent with the paper's description of late-delay photoinhibition.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim on/off times are converted to go-cue-relative coordinates and compared against the same bin centers used for neural data. The binary photostim vector has the same shape `(n_bins,)` as the neural data's time dimension.

ii.
```python
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. Same coordinate system (go-cue-relative) ensures alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Derived from `trials['trial_instruction']` (left/right) and `trials['outcome']` (hit/miss/ignore).

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
```

iii. The AI reasons that on hit trials, the mouse licked the instructed side. On miss trials, the mouse licked the wrong side, so choice is opposite of instruction.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is inferred from the combination of instruction and outcome: hit = licked correct side (instruction direction), miss = licked wrong side (opposite of instruction). Left = 0, right = 1. The NWB data does not have an explicit "lick direction" column, so this inference is necessary.

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0  # shouldn't happen after filtering
```

iii. Since ignore trials are filtered out, only hit and miss remain. The mapping is: hit+left instruction -> left choice (0), hit+right instruction -> right choice (1), miss+left instruction -> right choice (1), miss+right instruction -> left choice (0).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from `trials['outcome']` which contains `'ignore'`, `'miss'`, or `'hit'`.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. Direct mapping from string values to integers per the instructions.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple string-to-integer mapping: ignore=0, miss=1, hit=2. After filtering out ignore and early lick trials, only miss (1) and hit (2) actually appear.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. The output_values list includes all three values (`['ignore', 'miss', 'hit']`) for completeness, even though ignore never appears after trial filtering.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. N/A - This dataset does not have a "distance to reward zone" output. The outcome variable is per-trial (constant across time bins), broadcast to shape `(n_bins,)` for the time-varying output format.

ii.
```python
output_trial[1, :] = base_output[1]  # outcome (constant across time)
```

iii. Outcome is a per-trial label, replicated across all time bins.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Derived from `trials['early_lick']` which contains `'no early'` or `'early'`.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. Direct binary mapping from the trial table field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Binary mapping: `'no early'` -> 0, anything else -> 1. Since early lick trials are filtered out, this is always 0, making it a trivially decodable variable (100% accuracy).

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
output_trial[2, :] = base_output[2]  # early_lick (constant across time)
```

iii. The AI acknowledged this is trivial after filtering but kept it as required by the output specification.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Derived from `BehavioralTimeSeries['Camera0_side_TongueTracking']` which contains (x, y, likelihood) per frame at 300 Hz from DeepLabCut tracking.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. Uses the side camera (Camera0) tongue tracking data, column index 1 for y-position.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Tongue y values are extracted per trial within the time window, filtered by likelihood > 0.5, averaged within each 50ms bin, then discretized per session using 40th and 60th percentiles.

ii.
```python
# Filter by likelihood > 0.5
lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]

# Bin into time bins using np.digitize
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The AI uses a likelihood threshold of 0.5 to filter out low-confidence tracking points, then computes mean y-position per time bin.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles (40th and 60th) are computed from all valid (non-NaN) tongue y values across all trials in the session. Values below p40 get class 0 ("low"), between p40 and p60 get class 1 ("mid"), above p60 get class 2 ("high"). NaN values (no valid tracking) are mapped to class 0.

ii.
```python
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)

# In discretize_tongue_y:
if np.isnan(val):
    result[i] = 0  # no tongue visible = low position
elif val < p40:
    result[i] = 0
elif val <= p60:
    result[i] = 1
else:
    result[i] = 2
```

iii. The percentile computation matches the instructions. The NaN-to-0 mapping is a design choice: the AI reasoned that no tongue visible implies the tongue is retracted (low position).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue tracking frames are binned into the same 50ms time bins used for neural data, using go-cue-relative coordinates. The tongue y output has shape `(n_bins,)`, matching the neural time dimension.

ii.
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

iii. Same bin edges as neural data ensure temporal alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several missing data scenarios are handled:
- **obs_intervals mismatch**: When NWB files have fewer obs_intervals than trials (multi-session recordings), the AI builds a mapping from obs_intervals to trials and only processes trials with neural data.
- **Missing tongue tracking**: Bins without valid tongue tracking (likelihood < 0.5 or no frames) get NaN, which is discretized to class 0.
- **Missing tone onset**: Falls back to `go_cue - 1.85` if no sample_start_times found.
- **Missing brain region annotation**: Mapped to `'unknown'`.
- **Empty spike times**: Neurons with no spikes get zero firing rates.

ii.
```python
# obs_intervals mapping
if diffs[min_idx] < 0.5:
    obs_to_trial.append(int(min_idx))
else:
    obs_to_trial.append(-1)  # no match

# Tongue tracking fallback
tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)

# Tone onset fallback
tone_onset = go_cue - 1.85  # fallback: typical sample-delay duration

# Brain region fallback
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md and verified them against the data.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is reading spike times from NWB files. The AI initially read spike times individually per unit per trial, which was extremely slow (>30 min for full dataset). After optimization with batch preloading, the bottleneck became the overall I/O of reading large NWB files.

ii.
```python
# Optimized: batch preload
def preload_spike_times(nwb, unit_indices):
    return [nwb.units['spike_times'][uid] for uid in unit_indices]
```

iii. The agent documented: "Original code read spike times individually for each unit from NWB per trial (very slow, >30 min for full dataset). Fix: Added preload_spike_times() to batch-read all spike times upfront."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
- The `compute_firing_rates` function loops over neurons individually to histogram spikes. This could use 2D histogram or vectorized operations.
- The `discretize_tongue_y` function loops over individual values.
- The `build_obs_to_trial_map` function loops over obs_intervals to find matches.
- The per-bin tongue y averaging loop (`for b in range(len(bin_centers))`) could be replaced with groupby-style operations.

ii.
```python
# Per-neuron loop in compute_firing_rates
for i, st in enumerate(spike_times_by_neuron):
    counts, _ = np.histogram(st_window, bins=bin_edges)
    fr[i, :] = counts.astype(np.float32) / bin_width

# Per-value loop in discretize_tongue_y
for i, val in enumerate(tongue_y_values):
    if np.isnan(val):
        result[i] = 0
    elif val < p40:
        result[i] = 0
```

iii. The neuron-by-neuron histogram is inherently hard to vectorize since each neuron has different numbers of spikes. The discretize_tongue_y function is easily vectorizable with np.digitize or np.where.

## 10-c. What processing does the code repeat multiple times?

i. The `map_region_to_major` function is called twice for each neuron: once when collecting all region labels (to build the `brain_regions` list) and again when creating the `brain_region_idx` arrays.

ii.
```python
# First call: collecting labels
for label in s['brain_region_labels']:
    all_region_labels.add(map_region_to_major(label))

# Second call: creating indices
idx = np.array([brain_regions.index(map_region_to_major(label))
                for label in s['brain_region_labels']], dtype=np.int64)
```

iii. This duplication is minor in terms of performance since string matching is fast, but it could be avoided by caching the mapped region per neuron.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several outputs are computed but have limited utility after trial filtering:
- **Early lick output**: Always 0 after filtering (trivially decodable at 100%).
- **Outcome ignore class (0)**: Never appears after filtering, yet the output_values includes it.
- **Brain region mapping to major regions**: The detailed-to-major region mapping is extensive (15+ categories with many string matching rules) but could be simplified since many regions have few neurons.
- **Tongue y-position per-bin computation**: The detailed per-bin tongue y computation is expensive, and bins with NaN (no tongue visible, which is the majority of time bins since the tongue is only out during licking) all map to class 0.

ii.
```python
# Early lick - always 0 after filtering
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1

# Tongue y: many bins will be NaN -> class 0
tongue_y_bins = np.full(len(bin_centers), np.nan, dtype=np.float32)
```

iii. The early lick output is required by the instructions even though it provides no information after filtering. The detailed tongue y computation is necessary per the instructions but produces mostly class-0 values during non-licking periods.
