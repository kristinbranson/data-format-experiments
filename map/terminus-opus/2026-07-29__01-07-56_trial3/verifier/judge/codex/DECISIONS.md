# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all NWB session files by globbing `data/sub-*/sub-*.nwb`, then processes each file one at a time with `pynwb.NWBHDF5IO`. Within each NWB it reads the trials table, units table, `BehavioralEvents`, and `BehavioralTimeSeries`.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))

for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
```

iii. In `CONVERSION_NOTES.md` Step 2 the agent records that the dataset consists of 174 NWB files under subject directories. In trajectory Steps 18-25 it explicitly switches from the reference `.mat` layout to the NWB layout and treats one NWB file as one behavioral session.

## 1-b. How are the data split into subjects?

i. Subjects are identified from `nwb.subject.subject_id`. After all sessions are processed, the agent creates a sorted unique subject list and stores a per-session `subject_idx`.

ii.
```python
subject_id = nwb.subject.subject_id
...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
```

iii. `CONVERSION_NOTES.md` Step 9 says the converted dataset has 28 subjects and uses session-level subject metadata. The trajectory repeatedly refers to “28 subjects, 174 sessions,” then builds the final `subjects`/`subject_idx` arrays from `subject_id`.

## 1-c. How are the data split into sessions?

i. The agent treats each NWB file as one session. `process_session()` extracts one session dictionary, and the final output stores lists of sessions in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
def process_session(nwb_file, show_processing=False, session_idx=0):
    ...
    return {
        'neural': neural_trials,
        'input': input_trials,
        'output': output_trials,
        'subject_id': subject_id,
        ...
    }
```

```python
neural_list = [s['neural'] for s in all_sessions]
input_list = [s['input'] for s in all_sessions]
output_list = [s['output'] for s in all_sessions]
```

iii. The notes and trajectory both describe the NWB layout as one file per session, replacing the reference code’s “multiple probe files per session” concatenation with a single NWB read.

## 1-d. How are the data split into trials?

i. Trials are indexed from the NWB trial table and aligned using `go_start_times`. After filtering to `valid_trial_indices`, each remaining index becomes one trial in the session’s `neural`, `input`, and `output` lists.

ii.
```python
n_trials_total = len(nwb.trials)
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
valid_trial_indices = np.where(valid_trial_mask)[0]
...
for local_idx, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
    ...
    input_trials.append(input_data)
    ...
    tongue_y_trials.append(tongue_y)
```

iii. In trajectory Steps 21-25 and 54-58 the agent decides that trials should be keyed by go-cue timestamps and later narrowed to those with usable neural coverage.

## 1-e. How are trials filtered based on quality controls?

i. The agent does not apply the reference code’s regular-trial mask. Instead it keeps all trials whose go-cue window falls inside a coarse recording range, then uses session-level behavioral criteria computed on control, non-early, non-ignore trials: correct rate at least 0.65 and at least 50 correct left and 50 correct right trials. It loads `auto_water` and `free_water` but does not use them.

ii.
```python
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
```

```python
is_control = np.array([str(p) == 'N/A' for p in photostim_power])
is_no_early = np.array([str(e) == 'no early' for e in early_lick])
control_no_early = is_control & is_no_early
is_not_ignore = outcome != 'ignore'

denom = np.sum(control_no_early & is_not_ignore)
hits = np.sum(control_no_early & (outcome == 'hit'))
correct_rate = hits / denom
correct_left = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'left'))
correct_right = np.sum(control_no_early & (outcome == 'hit') & (trial_instruction == 'right'))
```

iii. `CONVERSION_NOTES.md` Step 5 says “Include all trials” and explicitly contrasts that choice with `get_regular_trial_mask`. Trajectory Steps 15-16 show the agent understood the reference mask excludes early lick, auto water, free water, no-response, and stimulation, but Step 34/Step 43 argues that those trial types should be kept because some are decoder inputs/outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `nwb.units['spike_times'][:]` after filtering units by `nwb.units['classification'][:] == 'good'`. Brain-region labels come from `anno_name`, but they are metadata, not the neural signal itself.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
...
spike_times_all = nwb.units['spike_times'][:]
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

iii. In `CONVERSION_NOTES.md` Steps 1, 3, and 5 the agent says the white paper’s classifier-based QC should be used and maps `spike_times -> neural`.

## 2-b. How is the `neural` data processed?

i. For each valid trial, the agent builds 50 ms bins spanning -2.5 s to +1.5 s around the go cue, counts spikes for each good unit in each bin, and divides by bin width to produce firing rates in Hz. No smoothing or additional normalization is applied.

ii.
```python
BIN_WIDTH = 0.050
WINDOW_START = -2.5
WINDOW_END = 1.5
```

```python
for trial_idx in range(n_trials):
    go_time = go_times[trial_idx]
    bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
    ...
    counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
    fr[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Step 5 maps `spike_times` to “Bin into 50ms windows, compute firing rate (Hz), aligned to go cue, -2.5 to +1.5s.” Trajectory Step 43 repeats the same plan.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by the classifier QC label `classification == 'good'`. Sessions with fewer than two good units are skipped. The final code does not use `is_good_trials`, does not remove zero-variance neurons, and only filters trials by a session-wide spike-time range.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)

if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    ...
```

iii. `CONVERSION_NOTES.md` Step 3 states “Neuron curation: classification == 'good' (classifier-based QC).” In trajectory Step 21 the agent noticed `is_good_trials`, and Step 43 even says “Use is_good_trials per-unit mask,” but that decision was not carried into the final code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. For each trial, bin edges are formed by adding the relative window `[-2.5, 1.5]` to that trial’s absolute `go_time`.

ii.
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The notes’ Step 3 and Step 5 both say temporal alignment is to go-cue onset, matching the instructions and the reference code’s go-cue-centered timing.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins and 80 bins per trial window. There is no secondary rebinning step; the initial histogram output is the final neural time base.

ii.
```python
BIN_WIDTH = 0.050  # 50 ms bins for firing rates
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
```

iii. The notes and trajectory repeatedly cite the task requirement “50-ms-width bins” and treat it as an override of the reference code’s other binning settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code it is not derived from any raw NWB variable. The agent hard-codes tone onset as a constant `TONE_ONSET_REL_GO = -1.85` seconds relative to go cue and derives the input from that constant plus bin centers.

ii.
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. Trajectory Steps 29-30 say the agent inferred tone onset from the paper: sample start is always 1.85 s before go cue, so “time from tone onset = time_from_go + 1.85.” `CONVERSION_NOTES.md` Step 5 repeats that decision instead of citing `sample_start_times`.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent constructs a single fixed 80-sample vector equal to each bin center minus the hard-coded tone-onset offset. The same vector is reused for every trial, regardless of session or trial-specific event timestamps.

ii.
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
...
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The justification is explicit in trajectory Steps 29-35: the agent decided sample onset was “consistently at -1.85s relative to go cue,” so it did not read `sample_start_times` from the NWB file.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It shares the exact same per-trial 50 ms bin grid as the neural data. The vector is stored as one row of `input_data` for each trial, with one value per neural time bin.

ii.
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
input_trials.append(input_data)
```

iii. The notes’ Step 5 says this input is aligned to the go-cue-centered trial window, and the code uses the same `BIN_CENTERS` that define the neural bins.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The photostimulation input is derived from the absolute event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`. `photostim_power` is only used earlier to define control trials for session filtering.

ii.
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. Trajectory Steps 24-25 describe discovering the separate start/stop event streams and choosing them for the binary time series.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the agent finds photostim events that overlap the trial window, converts the bin centers to absolute time, and marks each bin as 1 if its center falls between any selected start/stop pair, otherwise 0.

ii.
```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
if np.any(stim_mask):
    trial_stim_starts = photostim_starts_all[stim_mask]
    trial_stim_stops = photostim_stops_all[stim_mask]
    photostim_ts = get_photostim_timeseries(trial_stim_starts, trial_stim_stops, go_time)
else:
    photostim_ts = np.zeros(N_TIMEBINS, dtype=np.float32)
```

```python
abs_bin_centers = go_time + bin_centers
for start, stop in zip(photostim_start_times, photostim_stop_times):
    mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
    photostim[mask] = 1.0
```

iii. `CONVERSION_NOTES.md` Step 5 maps `photostim_start/stop -> binary 1 if photostim on, 0 otherwise`, and trajectory Step 43 says the same.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is aligned on the same 80 go-cue-centered bins used for neural firing rates by comparing absolute photostim times to `go_time + bin_centers`.

ii.
```python
abs_bin_centers = go_time + bin_centers
...
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The notes treat photostimulation as a time-varying per-bin decoder input, and the code uses the neural bin centers directly.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The code derives this output from `nwb.trials['trial_instruction']`, not from lick events or any actual choice variable. It therefore uses the instructed side rather than the animal’s observed lick direction.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `trial_instruction -> output[0]`, and trajectory Step 43 repeats that mapping. There is no evidence in the final code of using `left_lick_times`, `right_lick_times`, or `lick_directions`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent string-maps `left -> 0` and `right -> 1`, then broadcasts that scalar across all 80 bins for the trial.

ii.
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
...
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64)
```

iii. The notes justify this as a simple categorical per-trial output. The trajectory never revisits the distinction between instructed side and actual lick side once this mapping is chosen.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from `nwb.trials['outcome']`.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `outcome -> output[1]` with values `ignore/miss/hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code maps strings to integers `ignore=0`, `miss=1`, `hit=2`, then broadcasts the scalar outcome label over all time bins in the trial.

ii.
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
```

iii. The notes’ Step 5 lists exactly this mapping and says the output is per-trial rather than event-timed.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. `convert_data.py` does not create any “distance to reward zone” output. This appears to be a questionnaire/template error. If this item was intended to ask about `outcome`, the agent aligns outcome by broadcasting one trial-level label across all 80 neural bins.

ii.
```python
'output_names': ['choice', 'outcome', 'early_lick', 'tongue_y_position']
```

```python
out_arr = np.stack([
    np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
    np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64),
    np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64),
    tongue_y_discrete[t_idx].astype(np.int64),
], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 lists only four decoder outputs and does not mention reward-zone distance anywhere.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `nwb.trials['early_lick']`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. The notes’ Step 5 maps `early_lick -> output[2]`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then broadcasts the per-trial label across all 80 time bins.

ii.
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
...
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 and trajectory Step 43 both specify this exact mapping.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps plus the second data column (`[:, 1]`) as y position.

ii.
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
...
local_y = tongue_data[idx_start:idx_end, 1]
```

iii. Trajectory Steps 22-25 identify `Camera0_side_TongueTracking` as `(x, y, likelihood)` at roughly 300 Hz and choose the y coordinate.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the agent extracts the video samples falling in the neural window, bins them into the same 50 ms bins, and stores the mean y value in each bin. Empty bins are left as `NaN`. The code ignores the tracking likelihood channel.

ii.
```python
t_start = go_time + bin_centers[0] - bin_width / 2
t_end = go_time + bin_centers[-1] + bin_width / 2
...
local_ts = tongue_ts[idx_start:idx_end]
local_y = tongue_data[idx_start:idx_end, 1]
...
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. In trajectory Step 35 the agent explicitly considered using only high-likelihood detections, but `CONVERSION_NOTES.md` Step 5 records the final decision as “Use all tongue y values regardless of likelihood for percentile computation.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent pools all non-NaN tongue-y samples from the session, computes the 40th and 60th percentiles, then labels each valid bin as low/mid/high. Missing bins are left at the default middle category `1`.

ii.
```python
all_values = np.concatenate(all_values)
p_low = np.percentile(all_values, percentile_low)
p_high = np.percentile(all_values, percentile_high)
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
valid = ~np.isnan(y)
if np.any(valid):
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
```

iii. `CONVERSION_NOTES.md` Step 5 states the 40th/60th percentile discretization rule exactly as in the task, but also notes the extra implementation choice to ignore likelihood and default missing values to the middle class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned to the same go-cue-centered 50 ms bins as the neural data. Each tongue category sequence has the same 80-bin length as the neural matrix for that trial.

ii.
```python
tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
...
out_arr = np.stack([
    ...,
    tongue_y_discrete[t_idx].astype(np.int64),
], axis=0)
```

iii. The notes’ Step 5 describes tongue y as a time-varying output, and the helper function explicitly bins it using the neural bin boundaries.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing or irregular data with a set of ad hoc defaults: blank region names become `'unknown'`; unrecognized choice/outcome/early-lick strings fall back to label `0`; missing tongue bins remain `NaN` and later become category `1`; sessions with too few good units or valid trials are skipped; trial coverage is approximated by the min/max spike time over good units. It does not explicitly handle internal recording gaps and does not use tracking likelihood.

ii.
```python
if not anno_name or anno_name.strip() == '':
    return 'unknown'
```

```python
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

```python
tongue_y = np.full(n_bins, np.nan, dtype=np.float32)
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
```

iii. Trajectory Steps 54-71 show the agent investigating recording gaps, first considering `obs_intervals`, then replacing that with a min/max spike-time range. `CONVERSION_NOTES.md` Steps 10 and 12 acknowledge residual all-zero neural trials rather than removing them.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is neural firing-rate computation in `compute_firing_rates_session`, which loops over every valid trial and every good unit and histograms spikes. A secondary cost is per-trial tongue-y extraction/binning. The agent’s own runtime table also identifies firing-rate computation as the main bottleneck.

ii.
```python
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
```

```python
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. `CONVERSION_NOTES.md` Step 7 estimates full-conversion time and specifically breaks out “Firing rates computed” as the largest cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest vectorization opportunities are the nested trial-by-neuron loop in `compute_firing_rates_session`, the per-bin loop in `get_tongue_y_for_trial`, the per-trial scan over all photostim events, and the final loop that rebuilds broadcast output arrays one trial at a time.

ii.
```python
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
```

```python
for b in range(n_bins):
    mask = bin_indices == b
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    ...
    stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
```

iii. The notes never claim these loops are optimized; instead Step 7 reports long runtimes and Step 6 only describes the current implementation as “searchsorted optimization.”

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly rebuilds absolute bin edges for every trial, repeatedly scans the full photostim event arrays for each trial, repeatedly bins tongue video samples per trial, and repeatedly materializes constant per-trial categorical outputs with `np.full`.

ii.
```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
```

```python
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64)
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64)
```

iii. This follows directly from the implementation style in `process_session()`. The notes’ runtime breakdown is consistent with these repeated per-trial operations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `auto_water`, `free_water`, and `subject_desc` but never uses them in the final dataset; computes logging-only timing and summary statistics after saving; and supports optional diagnostic plotting that is not part of downstream decoder analysis. It also returns `subject_desc` from `process_session()` but drops it when building the final pickle.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
subject_desc = nwb.subject.description
```

```python
return {
    'neural': neural_trials,
    'input': input_trials,
    'output': output_trials,
    'subject_id': subject_id,
    'subject_desc': subject_desc,
    ...
}
```

```python
all_c, all_o, all_e = [], [], []
for s_idx in range(len(all_sessions)):
    for t_idx in range(len(output_list[s_idx])):
        all_c.append(output_list[s_idx][t_idx][0, 0])
        ...
print(f'\nOutput distributions:')
```

iii. The notes and trajectory emphasize diagnostic logging and validation. Those checks are useful for debugging, but they are not consumed by `train_decoder.py` or preserved as part of the core converted dataset.
