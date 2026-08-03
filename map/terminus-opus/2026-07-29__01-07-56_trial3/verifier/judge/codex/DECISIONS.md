# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing all NWB files under `data/sub-*/sub-*.nwb`, sorting the list, and calling `process_session` once per file. Inside `process_session` it opens the NWB file with `pynwb`, then reads subject metadata, trial columns, unit columns, behavioral event series, and tongue-tracking time series.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(
        nwb_file,
        show_processing=args.show_processing and i < 2,
        session_idx=i
    )
```

```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()
...
trial_instruction = nwb.trials['trial_instruction'][:]
classification = nwb.units['classification'][:]
beh_events = nwb.acquisition['BehavioralEvents']
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md` Step 0 and Step 2, the AI states that the dataset consists of 28 subject directories with 174 NWB files and that each NWB file contains trials, units, and behavioral streams. The trajectory shows it intentionally adopted the NWB layout as the master source and treated one file as one processing unit.

## 1-b. How are the data split into subjects?

i. The AI splits data into subjects using `nwb.subject.subject_id` from each NWB file. After processing all sessions, it builds `subjects` as the sorted unique subject ids and `subject_idx` as the per-session lookup into that list.

ii. 
```python
subject_id = nwb.subject.subject_id
```

```python
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
...
'subjects': unique_subjects,
'subject_idx': np.array(subject_idx_list, dtype=np.int64),
```

iii. The notes describe the NWB layout as one directory per subject and one file per session. The trajectory shows the AI treated `subject_id` as the canonical subject label from the files.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Sessions are processed in sorted file order, and each surviving `process_session` result becomes one entry in the top-level `neural`, `input`, and `output` session lists.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
...
if session_data is not None:
    all_sessions.append(session_data)
```

```python
neural_list = [s['neural'] for s in all_sessions]
input_list = [s['input'] for s in all_sessions]
output_list = [s['output'] for s in all_sessions]
```

iii. In `CONVERSION_NOTES.md` Step 2, the AI describes the file naming scheme as one NWB file per session and uses that file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the index of `go_start_times` as the trial index and assumes those indices correspond to the rows of `nwb.trials`. It then keeps only `valid_trial_indices` whose go-cue-centered window falls inside the inferred recording range and iterates over those indices to build one neural/input/output example per retained trial.

ii. 
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
    ...
    choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
    out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The trajectory shows the AI explored the NWB trial table and event streams, then assumed trial rows and go-cue events are aligned by index. Its justification for the later filtering step was that only some trials were covered by neural recording.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials in two ways. First, it keeps only trials whose `[-2.5, +1.5]` s window around the go cue lies inside a session-level recording range inferred from the earliest and latest observed spike times across good units. Second, it drops entire sessions unless they have at least 2 good units, at least 2 valid trials, behavioral correct rate `>= 0.65` on control no-early non-ignore trials, and at least 50 correct left and 50 correct right control-no-early trials. It does not explicitly remove `free_water` trials at the trial level.

ii. 
```python
if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
...
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
...
if len(valid_trial_indices) < 2:
    print(f'  Skipping: only {len(valid_trial_indices)} trials with neural coverage')
    io.close()
    return None
```

```python
is_control = np.array([str(p) == 'N/A' for p in photostim_power])
is_no_early = np.array([str(e) == 'no early' for e in early_lick])
control_no_early = is_control & is_no_early
is_not_ignore = outcome != 'ignore'
...
if correct_rate < 0.65:
    ...
if correct_left < 50 or correct_right < 50:
    ...
```

iii. `CONVERSION_NOTES.md` Step 3-5 says the AI adopted the paper’s session selection criteria (`>65%` correct and `>=50` correct left/right) and decided to include all trial types after session selection. The trajectory around steps 45-56 and 95-96 shows the AI explicitly justified the coverage filter as a response to sessions where spike data ended well before the session ended, and justified the session-performance filter using the methods text.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `nwb.units['spike_times'][:]` for units whose `classification` equals `'good'`, using `BehavioralEvents/go_start_times` to define trial alignment windows.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
...
spike_times_all = nwb.units['spike_times'][:]
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the AI states that neural data should come from spike times binned around the go cue and filtered with classifier-based QC.

## 2-b. How is the `neural` data processed?

i. The AI sorts spike times for each kept unit, then computes per-trial firing rates in 50 ms bins from `-2.5` s to `+1.5` s around the go cue. For each trial and neuron it builds bin edges, counts spikes with `np.histogram`, and divides by bin width to convert to Hz. No smoothing or normalization is applied.

ii. 
```python
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

```python
for trial_idx in range(n_trials):
    go_time = go_times[trial_idx]
    bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
    ...
    for i, spikes in enumerate(spike_times_list):
        if len(spikes) > 0:
            left = np.searchsorted(spikes, bin_edges[0])
            right = np.searchsorted(spikes, bin_edges[-1])
            if left < right:
                counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
                fr[i, :] = counts / bin_width
```

iii. `CONVERSION_NOTES.md` Step 5 and Step 6 says the neural mapping is “spike_times -> firing rates with 50ms windows, aligned to go cue” and the trajectory repeatedly describes this as the intended replication of the requested decoder format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'`. It then skips any session with fewer than 2 such units.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
n_good = np.sum(good_mask)

if n_good < 2:
    print(f'  Skipping: only {n_good} good units')
    io.close()
    return None
```

iii. The notes and trajectory explicitly say the AI chose the classifier-based QC mode from the reference methods and not the older `unit_quality` field. The extra `n_good < 2` rule appears to be the AI’s own safeguard rather than something justified from the papers.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go cue onset by taking each trial’s `go_start_times` timestamp and building the `[-2.5, +1.5]` s window around that event. The bin edges are absolute session times derived from `go_time + relative_offsets`.

ii. 
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 5 says temporal alignment is at go cue onset. The trajectory also records the AI reading reference timing code and then using go cue as time zero.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 50 ms bins and 80 total bins covering the 4 s window from `-2.5` to `+1.5` s. It does not apply any additional temporal rebinning beyond this direct binning from spike times.

ii. 
```python
BIN_WIDTH = 0.050
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
```

iii. The trajectory step 118 explicitly re-checks that `0.050` s means 50 ms and that a 4 s window yields 80 bins, so this was an intentional decision tied directly to the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code, the AI does not derive this input from a raw NWB variable. Instead it hard-codes tone onset as a fixed constant `TONE_ONSET_REL_GO = -1.85` seconds relative to the go cue, based on task timing inferred from the methods text.

ii. 
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
```

```python
# Time from tone onset (same for all trials)
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The notes and trajectory (especially steps 29-30, 35, 43, 118) say the AI concluded tone onset was consistently `-1.85 s` relative to go cue and therefore treated it as constant across trials rather than reading `sample_start_times` from the raw data.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes this input as the go-cue-relative bin centers shifted by `+1.85 s`. The same 80-value vector is reused for every trial in a session.

ii. 
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The AI’s justification in the trajectory is that if tone onset is fixed relative to go cue, then “time from tone onset” is also fixed once the go-cue-centered bins are fixed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI aligns this input by putting it on the same 80 go-cue-relative bins as the neural data. Because it uses a fixed vector, each trial gets the same time-from-tone trace.

ii. 
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
input_trials.append(input_data)
```

iii. The trajectory states that the bins are defined relative to go cue and that time-from-tone should therefore be expressed on that same grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the behavioral event streams `photostim_start_times` and `photostim_stop_times`, both read as absolute timestamps from `BehavioralEvents`.

ii. 
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory around steps 24-25 and 43 shows the AI inspected both trial-table photostim fields and event streams, then chose the start/stop event timestamps because they already gave absolute times for building a binary time series.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each retained trial, the AI finds all photostimulation intervals that overlap the trial’s `[-2.5, +1.5]` s go-cue-centered window. It then marks each 50 ms bin as `1.0` if the absolute bin center lies between any stimulation start and stop times, otherwise `0.0`.

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

iii. `CONVERSION_NOTES.md` Step 5 says photostimulation should be a per-time-bin binary variable. The trajectory shows the AI explicitly chose absolute start/stop times to create that binary trace.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostimulation to the neural data by comparing absolute photostim event times against absolute bin centers computed from the same `go_time` and `BIN_CENTERS` used for neural alignment.

ii. 
```python
abs_bin_centers = go_time + bin_centers
...
mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
```

iii. The trajectory shows the AI understood both spikes and behavioral events share the NWB session timebase, so it aligned them on the common go-cue-centered bin grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice only from `trial_instruction`. It does not use `outcome` to infer whether a miss means the opposite lick or whether an ignore means no lick.

ii. 
```python
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `trial_instruction` directly to choice, and trajectory step 43 explicitly says “Choice: left=0, right=1 (from trial_instruction).”

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps `'left' -> 0` and `'right' -> 1`, stores that single value per trial, and repeats it across all 80 bins in the output tensor. It defines only two choice categories in `output_values`.

ii. 
```python
CHOICE_MAP = {'left': 0, 'right': 1}
```

```python
output_choice.append(choice)
...
out_arr = np.stack([
    np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
    ...
], axis=0)
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no_early', 'early'],
    ['low', 'mid', 'high'],
],
```

iii. The AI’s notes justify this by treating the instructed side as the decoded choice. There is no evidence in the notes or trajectory that it revisited this after inspecting `outcome`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The AI derives outcome directly from `nwb.trials['outcome'][:]`.

ii. 
```python
outcome = nwb.trials['outcome'][:]
```

iii. The notes list `outcome` as a direct mapping from the trial table, with no derivation needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore' -> 0`, `'miss' -> 1`, `'hit' -> 2`, then repeats that per-trial value across all 80 bins.

ii. 
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

```python
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 describes exactly this mapping and notes that outcome is a per-trial categorical decoder target.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. The AI derives early lick directly from `nwb.trials['early_lick'][:]`.

ii. 
```python
early_lick = nwb.trials['early_lick'][:]
```

iii. The notes list `early_lick` as a direct trial-table field mapped into the decoder outputs.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'no early' -> 0` and `'early' -> 1`, then repeats the resulting per-trial value across all 80 bins.

ii. 
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

```python
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Step 5 gives the same mapping and treats this as a direct categorical target.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue y-position from `Camera0_side_TongueTracking`, specifically the timestamps and column 1 of the `data` array. It does not use the likelihood column in the final code.

ii. 
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
```

```python
local_ts = tongue_ts[idx_start:idx_end]
local_y = tongue_data[idx_start:idx_end, 1]
```

iii. The trajectory shows the AI inspected the tongue tracking columns and knew column 2 was likelihood, but `CONVERSION_NOTES.md` Step 5 explicitly says it decided to “use all tongue y values regardless of likelihood for percentile computation.”

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first averages raw tongue y values within each 50 ms trial bin. It then concatenates all non-NaN per-bin values from all retained trials in the session, computes the 40th and 60th percentiles on that pooled set, and discretizes each trial’s bins into low/mid/high based on those thresholds. Missing bins remain `NaN` until discretization, where the default class is the middle class.

ii. 
```python
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

```python
all_values = []
for y in tongue_y_session:
    valid = y[~np.isnan(y)]
    if len(valid) > 0:
        all_values.append(valid)
...
all_values = np.concatenate(all_values)
p_low = np.percentile(all_values, percentile_low)
p_high = np.percentile(all_values, percentile_high)
```

iii. The notes say the AI chose per-session discretization and explicitly decided to use all tongue y values without likelihood filtering. The trajectory step 35 mentions it saw tongue likelihood as a visibility signal but the final notes record the contrary decision.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three categories only: `< 40th percentile -> 0`, `40th-60th percentile -> 1`, `> 60th percentile -> 2`. Bins without a value are left at the default middle category `1`; there is no separate “not visible” class.

ii. 
```python
discrete = np.ones(len(y), dtype=np.int64)  # default middle
valid = ~np.isnan(y)
if np.any(valid):
    discrete[valid & (y < p_low)] = 0
    discrete[valid & (y >= p_low) & (y <= p_high)] = 1
    discrete[valid & (y > p_high)] = 2
```

```python
'output_values': [
    ['left', 'right'],
    ['ignore', 'miss', 'hit'],
    ['no_early', 'early'],
    ['low', 'mid', 'high'],
],
```

iii. `CONVERSION_NOTES.md` Step 5 describes the output as a 3-way discretization based on session percentiles. There is no justification in the notes for collapsing missing or not-visible bins into the middle category, but that is what the code does.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue y-position to the neural data by taking frames whose timestamps fall in the trial’s `[-2.5, +1.5]` s go-cue-centered window and binning them with the same 50 ms edges used for the neural traces.

ii. 
```python
t_start = go_time + bin_centers[0] - bin_width / 2
t_end = go_time + bin_centers[-1] + bin_width / 2
...
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
```

iii. The trajectory records that the AI recognized camera timestamps and go-cue times share the NWB clock, so it reused the go-cue-centered bin grid for tongue alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by dropping sessions with too few good units, too few valid trials, or low behavioral performance, and by dropping trials outside the inferred spike-time coverage range. For tongue data, missing bins are not represented explicitly; they effectively become the middle category because `discretize_tongue_y` initializes output bins to `1`. If an entire session has no valid tongue values, the function returns all ones. The code does not have dedicated handling for `free_water` trials or for text missingness beyond comparisons silently failing.

ii. 
```python
if n_good < 2:
    ...
if len(valid_trial_indices) < 2:
    ...
if correct_rate < 0.65:
    ...
if correct_left < 50 or correct_right < 50:
    ...
```

```python
if len(all_values) == 0:
    return [np.ones(len(y), dtype=np.int64) for y in tongue_y_session]
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
```

iii. The notes and trajectory justify dropping sessions and uncovered trials as necessary quality control. For tongue data, the final notes explicitly acknowledge using all values regardless of likelihood, and the code’s default-middle behavior appears to be an implementation choice rather than a separately justified policy.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading large NWB arrays, computing firing rates with the nested trial-by-neuron loop in `compute_firing_rates_session`, and building per-trial inputs/outputs including tongue extraction. The notes estimate firing-rate computation as the main runtime contributor.

ii. 
```python
fr_all = []
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    ...
    photostim_ts = get_photostim_timeseries(...)
    ...
    tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
```

iii. `CONVERSION_NOTES.md` Step 7 reports per-session timing and specifically attributes most of the runtime to data loading and firing-rate computation, with input/output construction as a smaller but still nontrivial cost.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the nested trial-by-neuron loop in `compute_firing_rates_session`, the per-bin loop inside `get_tongue_y_for_trial`, the per-trial loop that rebuilds photostim and tongue arrays, and the final summary loops over all trials.

ii. 
```python
for trial_idx in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
```

```python
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    ...
```

iii. The notes say the AI implemented `compute_firing_rates_session` with a “searchsorted optimization,” but the final code still retains several Python loops that could have been reduced or fused.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several passes over the same information. It scans all good-unit spike trains once in `get_recording_range` to find min/max spike times and then again to sort and bin spikes. It scans tongue tracking once per retained trial in `get_tongue_y_for_trial` instead of binning the session once. It also builds outputs in multiple stages: first separate lists for choice/outcome/early/tongue, then a second pass to pack them into output arrays.

ii. 
```python
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]
    ...
    for i in good_indices:
        st = np.array(spike_times_all[i])
```

```python
spike_times_all = nwb.units['spike_times'][:]
spike_times_good = []
for i in good_indices:
    st = np.array(spike_times_all[i], dtype=np.float64)
    spike_times_good.append(np.sort(st))
```

```python
for local_idx, trial_idx in enumerate(valid_trial_indices):
    ...
    tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
    tongue_y_trials.append(tongue_y)
...
for t_idx in range(n_valid):
    out_arr = np.stack([
```

iii. The notes do not present these repetitions as deliberate optimizations; they are artifacts of the implementation structure.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does several things that are not needed for the final decoder dataset: it loads `auto_water`, `free_water`, and `subject_desc` without using them in the saved output; it includes optional plotting machinery for `--show-processing`; it computes and prints summary distributions solely for logging; and it stores session `correct_rate` metadata even though downstream decoding uses only the converted arrays and structural metadata.

ii. 
```python
subject_desc = nwb.subject.description
...
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
```

```python
if show_processing:
    plot_processing(nwb_file, neural_trials, input_trials, output_trials, session_idx)
```

```python
all_c, all_o, all_e = [], [], []
for s_idx in range(len(all_sessions)):
    for t_idx in range(len(output_list[s_idx])):
        all_c.append(output_list[s_idx][t_idx][0, 0])
```

iii. `CONVERSION_NOTES.md` Step 13 explicitly mentions creating plots/README/cleanup artifacts, and the code shows additional logging-oriented work beyond what is needed to produce `converted_data.pkl`.
