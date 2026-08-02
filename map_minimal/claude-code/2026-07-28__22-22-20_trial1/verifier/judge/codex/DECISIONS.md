# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script scans `/app/data` for subject directories named `sub-*`, then scans each subject directory for `.nwb` files. Each NWB file is treated as one session, opened with `pynwb.NWBHDF5IO`, and its `trials`, `units`, `BehavioralEvents`, and `BehavioralTimeSeries` tables are read. Trials are then processed one-by-one inside `process_session`.

ii. 
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])

for subj in subjects:
    subj_dir = os.path.join(data_dir, subj)
    files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
    ...
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
bts = nwb.acquisition['BehavioralTimeSeries']
```

iii. The trajectory says the agent decided the “Data source” was “NWB files in `/app/data/`” and then built the converter around that representation.

## 1-b. How are the data split into subjects?

i. Iteration is organized by `sub-*` directories, but the stored subject identity comes from `nwb.subject.description` (for example `SC015`). After all passing sessions are collected, the script builds a unique ordered subject list and a `subject_idx` per session.

ii.
```python
subject_id = nwb.subject.description
...
unique_subjects = list(OrderedDict.fromkeys(s['subject_id'] for s in all_sessions))
subject_idx = np.array([unique_subjects.index(s['subject_id']) for s in all_sessions], dtype=np.int64)
```

iii. The notes state that the output summarizes mice, not DANDI subject folder names, and the code reflects that by using the NWB subject metadata.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. The session ID is taken from `nwb.session_id` when available, otherwise the filename is used. Sessions failing the filtering rules are skipped and never added to the final dataset.

ii.
```python
session_id = nwb.session_id if nwb.session_id else os.path.basename(nwb_path)
...
result = process_session(fpath, sample_mode=sample_mode)
if result is not None:
    all_sessions.append(result)
```

iii. The trajectory explicitly lists “NWB files in `/app/data/`” as the session-level source and the notes report final counts in sessions, confirming that file-level split.

## 1-d. How are the data split into trials?

i. Trial identities come from the NWB `trials` table. Neural coverage is associated to trial indices through a custom `obs_intervals` to trial mapping: if the number of observation intervals equals the number of trials, it uses identity mapping; otherwise it matches each `obs_interval` to the nearest trial start within a 0.5 s tolerance. Only mapped trials are processed further.

ii.
```python
def build_obs_to_trial_map(nwb, good_indices):
    trials = nwb.trials
    ...
    obs = units['obs_intervals'][good_indices[0]]
    if n_obs == n_trials:
        return list(range(n_trials))
    ...
    for i in range(n_obs):
        diffs = np.abs(trial_starts - obs[i, 0])
        min_idx = np.argmin(diffs)
        if diffs[min_idx] < 0.5:
            obs_to_trial.append(int(min_idx))
        else:
            obs_to_trial.append(-1)
```

iii. The notes justify this as handling sessions where `obs_intervals` cover only a subset of trials because recording started or stopped mid-session.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if it has mapped neural coverage, is not an early-lick trial, and is not a no-response (`ignore`) trial. The script does not additionally exclude `auto_water` or `free_water` trials, and it keeps photostimulation trials because photostimulation is later used as a decoder input.

ii.
```python
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

iii. The notes and trajectory both say the agent followed the paper’s rule that “Early lick trials and no response trials were excluded for analysis.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `nwb.units['spike_times']` for units whose `classification` is `'good'`. Trial assignment also depends on `units['obs_intervals']`, and temporal alignment uses the trial’s go cue from `BehavioralEvents/go_start_times`.

ii.
```python
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)
...
all_spike_times = preload_spike_times(nwb, good_indices)
obs_intervals = units['obs_intervals'][good_indices[0]]
go_start_times = be.time_series['go_start_times'].timestamps[:]
```

iii. The trajectory says the key neural decision was “Spike times: Absolute times, need to align to go cue” with classifier-based QC via `classification == 'good'`.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the script first restricts each good unit’s absolute spike times to that trial’s `obs_interval`, then histograms spikes into non-overlapping 50 ms bins from `-2.5` to `+1.5` s around the go cue, and finally divides counts by 0.05 s to produce firing rates in Hz.

ii.
```python
def get_spike_times_for_trial_from_cache(all_spike_times, obs_intervals, obs_idx):
    t_start, t_stop = obs_intervals[obs_idx]
    for st in all_spike_times:
        mask = (st >= t_start) & (st < t_stop)
        spike_times_list.append(st[mask])
```

```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
...
counts, _ = np.histogram(st_window, bins=bin_edges)
fr[i, :] = counts.astype(np.float32) / bin_width
```

iii. The notes explicitly justify this as “Spike counts are histogrammed into 50ms bins and divided by bin width” and acknowledge that this is “simpler than the sliding kernel approach in the original preprocessing code.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered solely by the NWB `classification` field: only units labeled `'good'` are retained. Sessions with zero such units are discarded.

ii.
```python
good_indices = []
for i in range(n_units):
    if units['classification'][i] == 'good':
        good_indices.append(i)

if len(good_indices) == 0:
    print(f"  Skipping {os.path.basename(nwb_path)}: 0 good units")
    ...
    return None
```

iii. The notes describe this as “Classifier-based (`'good'` classification in NWB)” and tie it to the spike-sorting QC paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial is aligned to that trial’s go cue. Bin edges are created by adding the `[-2.5, 1.5]` window to the absolute `go_cue_time`, and the returned bin centers are converted back to time relative to go cue.

ii.
```python
go_cue = go_start_times[trial_idx]
fr, bin_centers = compute_firing_rates(
    spike_times_trial, go_cue, BEGIN_TIME, END_TIME, BIN_WIDTH
)
```

```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0 - go_cue_time
```

iii. Both the file header and the notes say the converter intentionally aligns all neural data to “go cue onset.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins, giving 80 time bins over the 4.0 s window. No additional temporal rebinning is applied after histogramming.

ii.
```python
BIN_WIDTH = 0.05
BEGIN_TIME = -2.5
END_TIME = 1.5
...
n_bins = int(round((end_time - begin_time) / bin_width))
```

iii. The notes list “Bin width | 50 ms” and “Time bins per trial | 80,” matching the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, `BehavioralEvents/go_start_times`, and the trial’s `start_time` from the trials table. The sample start events are searched within the bounds of the current trial and before the go cue.

ii.
```python
go_start_times = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
t_start = trials['start_time'][trial_idx]
valid_samples = sample_start_times[
    (sample_start_times >= t_start) & (sample_start_times < go_cue)
]
```

iii. The trajectory says the agent investigated sample starts and concluded that the last sample start before the go cue corresponds to the real tone onset after any replayed epochs.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script takes the last sample start before the go cue as tone onset. If none is found, it falls back to `go_cue - 1.85`. It then subtracts this tone onset from each go-cue-aligned bin center so that each bin contains “time since tone onset.”

ii.
```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85

tone_onset_rel = tone_onset - go_cue
time_from_tone = bin_centers - tone_onset_rel
```

iii. The notes say tone onset is “the last `sample_start_time` before the go cue,” and the trajectory says early licks create earlier replay sample starts, so the final one is the actual tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is sampled on exactly the same `bin_centers` used for the neural firing rates, so every time point in this input corresponds directly to a neural time bin.

ii.
```python
fr, bin_centers = compute_firing_rates(...)
...
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_on], axis=0)
```

iii. The agent’s stated plan was to use go-cue-aligned bins for all streams, and this input is computed directly from those bin centers.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. `photostimulation` is derived from per-trial `photostim_onset`, `photostim_duration`, and `start_time` in the trials table. It does not use the separate behavioral event time series for photostim.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][trial_idx]
if photostim_onset_trial != 'N/A':
    onset_val = float(photostim_onset_trial)
    dur_val = float(trials['photostim_duration'][trial_idx])
    photostim_start_abs = t_start + onset_val
    photostim_stop_abs = photostim_start_abs + dur_val
```

iii. The trajectory says the agent specifically checked photostim timing and then decided the decoder only needed a binary “on/off” regressor.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the script initializes an all-zero vector, converts the photostim onset and offset from trial-start-relative time to go-cue-relative time, and fills bins within that interval with `1.0`. Control trials with `'N/A'` remain all zeros.

ii.
```python
photostim_on = np.zeros(len(bin_centers), dtype=np.float32)
...
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The notes summarize the decision as “Binary input (0=off, 1=on)” and say any bin within the photostim interval is marked on.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostim interval is converted into the same go-cue-relative coordinates as the neural bins, then rasterized on the same `bin_centers`.

ii.
```python
ps_start_rel = photostim_start_abs - go_cue
ps_stop_rel = photostim_stop_abs - go_cue
photostim_on[(bin_centers >= ps_start_rel) & (bin_centers < ps_stop_rel)] = 1.0
```

iii. The notes explicitly say photostim onset times are “converted from absolute times to relative-to-go-cue times.”

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from the trial instruction and the trial outcome. The code does not read lick timestamps or an explicit final lick-direction label.

ii.
```python
instruction = trials['trial_instruction'][trial_idx]
outcome = trials['outcome'][trial_idx]
```

iii. The notes justify the inference rule directly: hit trials use the instructed side and miss trials use the opposite side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The output is encoded as left `0`, right `1`. For `hit` trials, choice equals the instructed side. For `miss` trials, choice is flipped to the opposite side. Other outcomes default to `0`, though those should already have been filtered out.

ii.
```python
if outcome == 'hit':
    choice = 0 if instruction == 'left' else 1
elif outcome == 'miss':
    choice = 1 if instruction == 'left' else 0
else:
    choice = 0
```

iii. The notes call this out as a dedicated decision under “Choice Determination.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials table field `trials['outcome']`.

ii.
```python
outcome = trials['outcome'][trial_idx]
```

iii. No extra justification was needed beyond matching the task’s requested categorical outcome.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps trial outcomes to integer categories: `ignore -> 0`, `miss -> 1`, `hit -> 2`. Because `ignore` trials were filtered earlier, `0` is defined but normally absent from retained trials.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome, 0)
```

iii. The notes explicitly say the output values still list all categories “for completeness” even though `ignore` trials were excluded.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The script does not compute a “distance to reward zone” output at all. The only relevant alignment step here is that `outcome` is eventually broadcast across all neural time bins when the final `output_trial` matrix is assembled.

ii.
```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]
output_trial[1, :] = base_output[1]
output_trial[2, :] = base_output[2]
output_trial[3, :] = tongue_y_disc
```

iii. The task instructions requested `Outcome`, not distance-to-reward-zone. The agent therefore never justified or implemented a reward-zone distance variable.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii.
```python
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The notes frame early lick as both a trial filter and an output variable, with the consequence that retained trials all end up labeled `0`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string label is binarized as `no early -> 0` and anything else -> `1`. However, because early-lick trials were filtered before trial processing, the stored output is effectively always `0`.

ii.
```python
if early != 'no early':
    continue
...
early_lick_val = 0 if trials['early_lick'][trial_idx] == 'no early' else 1
```

iii. The notes explicitly acknowledge this: “After filtering, `early_lick` output is always 0.”

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from the side-camera DeepLabCut tongue tracking stream `Camera0_side_TongueTracking`, using column 1 as `y` and column 2 as tracking likelihood, together with that time series’ timestamps.

ii.
```python
tongue_ts = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts.data[:]
tongue_times = tongue_ts.timestamps[:]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
```

iii. The trajectory says the agent checked the tongue tracking stream and concluded it contained `(x, y, likelihood)` from the side camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the script collects tongue samples from a slightly expanded trial window, removes samples with likelihood `<= 0.5`, assigns the remaining timestamps into go-cue-aligned neural bins, and stores the mean y-position per bin. Raw binned values are saved first and discretized later at the session level.

ii.
```python
tw_mask = (tongue_times >= trial_window_start) & (tongue_times < trial_window_end)
tw_times = tongue_times[tw_mask]
tw_y = tongue_y[tw_mask]
tw_lh = tongue_likelihood[tw_mask]

lh_mask = tw_lh > 0.5
tw_times_good = tw_times[lh_mask]
tw_y_good = tw_y[lh_mask]
```

```python
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The notes justify the side-camera choice and say only positions with tracking likelihood `> 0.5` are used.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After all trials of a session are processed, the script concatenates all non-NaN binned tongue y values from that session, computes the 40th and 60th percentiles, and discretizes each bin into low/mid/high. NaN bins are forced to class `0`.

ii.
```python
all_tongue_y = np.concatenate([v for v in s['tongue_y_trial_values']])
valid_tongue = all_tongue_y[~np.isnan(all_tongue_y)]
...
p40 = np.percentile(valid_tongue, 40)
p60 = np.percentile(valid_tongue, 60)
```

```python
if np.isnan(val):
    result[i] = 0
elif val < p40:
    result[i] = 0
elif val <= p60:
    result[i] = 1
else:
    result[i] = 2
```

iii. The notes say the per-session 40th/60th percentile rule came from the task specification, and also record the extra heuristic that missing bins default to low.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are converted into the same absolute bin edges used around each trial’s go cue, then binned onto the same 50 ms neural time base. In the final output, tongue position remains the only genuinely time-varying output dimension.

ii.
```python
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
bin_assignments = np.digitize(tw_times_good, abs_bin_edges) - 1
```

```python
output_trial[3, :] = tongue_y_disc
```

iii. The notes describe tongue y-position as a time-varying output derived from the side video and discretized after alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several heuristics. Missing or empty region annotations become `'unknown'`. Unmatched `obs_intervals` get `-1` and corresponding trials are dropped. Missing tone onset falls back to `go_cue - 1.85`. Missing tongue samples stay `NaN` until discretization, then become class `0`. Sessions with no good units, no qualifying control trials, poor performance, or fewer than two valid trials are skipped.

ii.
```python
if diffs[min_idx] < 0.5:
    obs_to_trial.append(int(min_idx))
else:
    obs_to_trial.append(-1)
```

```python
if len(valid_samples) > 0:
    tone_onset = valid_samples[-1]
else:
    tone_onset = go_cue - 1.85
```

```python
if anno is None or anno == '' or anno == 'nan':
    anno = 'unknown'
...
if np.isnan(val):
    result[i] = 0
```

iii. The notes call out unusual timing cases, partial `obs_intervals`, and all-zero neural trials as known issues, so these heuristics were deliberate rather than accidental.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive work is reading each NWB file, preloading spike times for all good units, repeatedly slicing those spike times per trial, histogramming spikes neuron-by-neuron for every trial, and binning tongue tracking frame-by-frame into neural bins. On the full dataset, these nested trial/neuron loops dominate runtime.

ii.
```python
print(f"  Loading spike times for {len(good_indices)} good units...")
all_spike_times = preload_spike_times(nwb, good_indices)
```

```python
for i, st in enumerate(spike_times_by_neuron):
    ...
    counts, _ = np.histogram(st_window, bins=bin_edges)
```

```python
for trial_idx in valid_trial_indices:
    ...
    for b in range(len(bin_centers)):
        b_mask = bin_assignments == b
        if np.any(b_mask):
            tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The conversion logs and code structure show the runtime is dominated by loading and per-trial processing rather than final assembly.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are still scalar or repeatedly masked: the per-neuron firing-rate loop in `compute_firing_rates`, the repeated per-trial slicing over all unit spike trains in `get_spike_times_for_trial_from_cache`, the per-bin tongue averaging loop, and the loops that gather good units and brain regions. These are correct but not especially efficient.

ii.
```python
for st in all_spike_times:
    mask = (st >= t_start) & (st < t_stop)
    spike_times_list.append(st[mask])
```

```python
for i, st in enumerate(spike_times_by_neuron):
    ...
    counts, _ = np.histogram(st_window, bins=bin_edges)
```

```python
for b in range(len(bin_centers)):
    b_mask = bin_assignments == b
    if np.any(b_mask):
        tongue_y_bins[b] = np.mean(tw_y_good[b_mask])
```

iii. The notes repeatedly describe some optimizations as “for speed,” which implies the agent recognized performance pressure but stopped short of broader vectorization.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly scans the trials table for session filtering and then again for trial extraction, repeatedly masks spike-time arrays by observation interval for each trial, and repeatedly recomputes absolute bin edges for neural and tongue alignment on every trial. It also computes time-varying output matrices by broadcasting constant per-trial labels on every trial.

ii.
```python
for i in range(n_trials):
    ...
    if is_control and is_no_early:
        ...
```

```python
for i in range(n_trials):
    if i not in trials_with_neural:
        continue
    ...
    valid_trial_indices.append(i)
```

```python
bin_edges = np.linspace(begin_time, end_time, n_bins + 1) + go_cue_time
...
abs_bin_edges = np.linspace(BEGIN_TIME, END_TIME, len(bin_centers) + 1) + go_cue
```

iii. This follows naturally from the script’s straightforward per-session, per-trial design; there is no sign in the notes that the agent tried to deduplicate these passes.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script stores raw per-bin `tongue_y_trial_values` for every trial only to discard them after session-level discretization. It also keeps `ignore` and `early lick = yes` categories in metadata even though its own filtering makes those labels absent from the final retained trials. Finally, it expands constant outputs (`choice`, `outcome`, `early_lick`) across all 80 time bins, which is convenient for format uniformity but redundant for downstream use.

ii.
```python
tongue_y_trial_values.append(tongue_y_bins)
...
tongue_y_bins = s['tongue_y_trial_values'][t_idx]
tongue_y_disc = discretize_tongue_y(tongue_y_bins, p40, p60)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no', 'yes'],
    ['low', 'mid', 'high'],
],
```

```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = base_output[0]
output_trial[1, :] = base_output[1]
output_trial[2, :] = base_output[2]
```

iii. The notes explicitly acknowledge that `ignore` never appears and `early_lick` is always `0`, so some of this processing was knowingly carried along only for completeness or output-format convenience.
