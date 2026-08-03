# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script glob-loads every NWB file under `data/sub-*/sub-*.nwb`, opens each file with `pynwb.NWBHDF5IO`, and processes one file at a time in `process_session`. Within each session it reads the `trials`, `units`, `BehavioralEvents`, and `BehavioralTimeSeries` tables/streams.

ii. 
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))

for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
```

```python
io = pynwb.NWBHDF5IO(nwb_file, 'r')
nwb = io.read()

trial_instruction = nwb.trials['trial_instruction'][:]
classification = nwb.units['classification'][:]
beh_events = nwb.acquisition['BehavioralEvents']
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
```

iii. In `CONVERSION_NOTES.md` Step 2 and the trajectory, the agent justified this as adapting the reference analysis from `.mat` files to NWB while extracting the same conceptual streams: trials, units, event times, and video tracking.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from `nwb.subject.subject_id`. After processing all accepted sessions, the script creates a sorted unique subject list and a `subject_idx` entry mapping each kept session to one subject.

ii. 
```python
subject_id = nwb.subject.subject_id
...
all_subject_ids = [s['subject_id'] for s in all_sessions]
unique_subjects = sorted(set(all_subject_ids))
subject_to_idx = {s: i for i, s in enumerate(unique_subjects)}
subject_idx_list = [subject_to_idx[s['subject_id']] for s in all_sessions]
```

iii. The notes say the dataset contains 28 subjects and that subject IDs should come directly from the NWB metadata so the session order can later be indexed back to mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A file contributes one session to the final output only if `process_session` returns data instead of `None`; otherwise the whole file/session is skipped.

ii. 
```python
for i, nwb_file in enumerate(nwb_files):
    session_data = process_session(nwb_file, ...)
    if session_data is not None:
        all_sessions.append(session_data)
```

iii. In the notes and trajectory, the agent consistently treated one behavioral NWB file as one behavioral session, matching the dataset organization it observed.

## 1-d. How are the data split into trials?

i. Trials are indexed from the `go_start_times` event vector. The script builds `valid_trial_indices` from those go cues, then constructs one neural/input/output item per retained go cue.

ii. 
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
valid_trial_mask = (go_times + WINDOW_START >= rec_start) & (go_times + WINDOW_END <= rec_end)
valid_trial_indices = np.where(valid_trial_mask)[0]
...
valid_go_times = go_times[valid_trial_indices]
fr_all = compute_firing_rates_session(spike_times_good, valid_go_times)

for local_idx, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
    ...
```

iii. The trajectory states the agent decided to use go cue onset as the per-trial anchor because the task instructions explicitly required go-cue alignment.

## 1-e. How are trials filtered based on quality controls?

i. At the trial level, the code only keeps trials whose `[-2.5 s, +1.5 s]` go-cue window falls between the minimum and maximum spike times across good units. It does **not** apply the reference `regular trial` exclusions for early lick, auto-water, free-water, ignore/no-response, or stimulation. Those factors are only used indirectly when computing a session-level inclusion rule (`>65%` correct, at least `50` correct left and `50` correct right on control no-early non-ignore trials).

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
...
if correct_rate < 0.65: return None
if correct_left < 50 or correct_right < 50: return None
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent explicitly chose to “include all trials” because `early_lick`, `outcome`, and `photostim` were decoder targets/inputs. The trajectory also shows it knew the reference code had a stricter `get_regular_trial_mask`, but chose not to replicate it.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `nwb.units['spike_times']`, restricted to units whose `classification` equals `'good'`. Trial alignment additionally uses `BehavioralEvents/go_start_times`.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
...
spike_times_all = nwb.units['spike_times'][:]
...
go_times = beh_events.time_series['go_start_times'].timestamps[:]
```

iii. The notes call this “classifier-based QC,” matching the paper’s good-unit filtering, and describe spike times plus go-cue times as the key ingredients for the neural tensor.

## 2-b. How is the `neural` data processed?

i. For each retained trial, the code builds 50 ms bins from `go_time - 2.5 s` to `go_time + 1.5 s`, counts spikes from each good unit in each bin, and divides by bin width to convert counts to firing rates in Hz.

ii. 
```python
BIN_WIDTH = 0.050
WINDOW_START = -2.5
WINDOW_END = 1.5
```

```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
...
counts, _ = np.histogram(spikes[left:right], bins=bin_edges)
fr[i, :] = counts / bin_width
```

iii. The notes and trajectory say the agent matched the decoder specification here, even though the reference code uses other binning choices in other contexts.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit QC is `classification == 'good'`. Session QC requires at least 2 good units. Trial QC is only a coarse recording-range filter based on the global minimum and maximum spike time across good units. The code does not use `is_good_trials`, does not inspect `obs_intervals` gaps, and does not drop zero-neural trials that still pass the global range check.

ii. 
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
if n_good < 2:
    return None
```

```python
def get_recording_range(nwb, good_indices):
    spike_times_all = nwb.units['spike_times'][:]
    ...
    rec_start = min(rec_start, st.min())
    rec_end = max(rec_end, st.max())
```

iii. The notes justify the unit filter as matching classifier-based QC, but also document that the agent switched from `obs_intervals` to global spike-time range after debugging end-of-recording issues. The trajectory shows it noticed `is_good_trials`, even planned to use it, but the final script never does.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial is aligned to go cue onset. The code computes absolute bin edges by adding `WINDOW_START..WINDOW_END` offsets to that trial’s `go_time`.

ii. 
```python
go_times = beh_events.time_series['go_start_times'].timestamps[:]
...
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

iii. The instructions explicitly required go-cue alignment, and the notes/trajectory repeatedly mention this as a deliberate match to the task specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins. No second-stage temporal rebinning is applied; the neural data are produced directly at that resolution.

ii. 
```python
BIN_WIDTH = 0.050
N_TIMEBINS = int((WINDOW_END - WINDOW_START) / BIN_WIDTH)  # 80 bins
```

iii. The notes say this was chosen directly from the decoder instructions, not from the reference scripts’ alternate 100 ms / 40 ms preprocessing settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. In the final code it is not computed from a raw variable at run time. It is derived from the hard-coded constant `TONE_ONSET_REL_GO = -1.85`, which the agent inferred from the task structure and from exploratory comparisons of sample and go times.

ii. 
```python
TONE_ONSET_REL_GO = -1.85  # tone onset relative to go cue (seconds)
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
```

iii. The trajectory shows the justification: the agent concluded tone onset was “consistently at -1.85s relative to go cue” and therefore treated it as constant. `CONVERSION_NOTES.md` Step 5 repeats that assumption.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The script takes the fixed go-aligned bin centers and subtracts the fixed relative tone time (`-1.85 s`), yielding a per-bin elapsed-time-from-tone vector that is reused for every trial in a session.

ii. 
```python
BIN_CENTERS = WINDOW_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
...
time_from_tone = (BIN_CENTERS - TONE_ONSET_REL_GO).astype(np.float32)
...
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The trajectory shows the agent believed tone onset was fixed relative to go cue, so no per-trial recomputation from `sample_start_times` was necessary.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is aligned by sharing the same 80 go-cue-centered bins as the neural tensor. However, because the vector is fixed across trials, trial-to-trial variation in actual sample/tone timing is not represented.

ii. 
```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
```

iii. The notes justify this by assuming the tone always occurs `1.85 s` before go cue. The trajectory records that same assumption explicitly.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`.

ii. 
```python
photostim_starts_all = beh_events.time_series['photostim_start_times'].timestamps[:]
photostim_stops_all = beh_events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory shows the agent inspected these event streams and concluded they were the correct way to represent “whether photostimulation is on at every time point.”

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each retained trial, the script finds stimulation intervals that overlap the `[-2.5, 1.5] s` trial window and marks each bin center as `1` if it falls inside any overlapping stim interval, else `0`.

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

iii. The notes say the agent wanted a binary time series matching the decoder input specification. The trajectory also notes photostim typically occurs in late delay and should be represented continuously over bins.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The code converts the same go-aligned bin centers used for neural data into absolute timestamps and checks whether each center lies inside a photostim interval.

ii. 
```python
abs_bin_centers = go_time + bin_centers
for start, stop in zip(photostim_start_times, photostim_stop_times):
    mask = (abs_bin_centers >= start) & (abs_bin_centers <= stop)
    photostim[mask] = 1.0
```

iii. The trajectory explicitly describes this as aligning photostim to the same go-cue-centered frame as spikes.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The final code derives “choice” from `nwb.trials['trial_instruction']`, not from actual lick-direction behavior. It therefore uses the instructed side rather than the animal’s chosen side.

ii. 
```python
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly maps `trial_instruction -> output[0]`. The trajectory shows the agent had discovered `left_lick_times` and `right_lick_times`, but still chose the trial instruction as the decoder output.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script maps `'left' -> 0` and `'right' -> 1`, stores one scalar per trial, and then broadcasts that scalar across all 80 time bins.

ii. 
```python
CHOICE_MAP = {'left': 0, 'right': 1}
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
...
np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64)
```

iii. The notes justify this as a simple categorical per-trial output, but they justify the wrong source variable: instructed side, not behavioral choice.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from `nwb.trials['outcome']`.

ii. 
```python
outcome = nwb.trials['outcome'][:]
...
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
```

iii. The notes and trajectory consistently describe the NWB `outcome` field as containing `hit`, `miss`, and `ignore`, which matches the target output categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps `ignore -> 0`, `miss -> 1`, `hit -> 2`, then broadcasts the result across all time bins for that trial.

ii. 
```python
OUTCOME_MAP = {'ignore': 0, 'miss': 1, 'hit': 2}
...
np.full(N_TIMEBINS, output_outcome[t_idx], dtype=np.int64)
```

iii. The notes explicitly record this mapping in Step 5 as the intended output encoding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `nwb.trials['early_lick']`.

ii. 
```python
early_lick = nwb.trials['early_lick'][:]
...
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

iii. The trajectory documents that the agent read `early_lick` values as `'early'` or `'no early'` from the trial table and used that field directly.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `'no early' -> 0` and `'early' -> 1`, then broadcasts the categorical result across the whole trial.

ii. 
```python
EARLY_LICK_MAP = {'no early': 0, 'early': 1}
...
np.full(N_TIMEBINS, output_early_lick[t_idx], dtype=np.int64)
```

iii. The notes list exactly this mapping in the Step 5 planning table.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: the code uses the timestamps and the second column of the tracking matrix (`[:, 1]`).

ii. 
```python
tongue_ts_obj = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
...
local_y = tongue_data[idx_start:idx_end, 1]
```

iii. The trajectory shows the agent inspected the video stream and concluded the columns were `(x, y, likelihood)`, so column 1 should be the y-position.

## 8-b. How is `output` *Tongue y-position* processed?

i. For each trial, the script extracts the continuous tongue trace inside the neural window, assigns video samples to 50 ms go-aligned bins, and stores the mean y-value in each bin. It does not use the likelihood/confidence column to mask unreliable points.

ii. 
```python
idx_start = np.searchsorted(tongue_ts, t_start, side='left')
idx_end = np.searchsorted(tongue_ts, t_end, side='right')
...
bin_indices = np.digitize(local_ts, abs_bin_edges) - 1
for b in range(n_bins):
    mask = bin_indices == b
    if np.any(mask):
        tongue_y[b] = np.mean(local_y[mask])
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent deliberately used “all tongue y values regardless of likelihood for percentile computation.” The trajectory shows it considered likelihood filtering, then rejected it.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After collecting all trial traces for one session, the script pools all non-NaN bin values from that session, computes the 40th and 60th percentiles, and discretizes each valid bin as low/mid/high. Missing bins stay at the default middle category `1`.

ii. 
```python
all_values = np.concatenate(all_values)
p_low = np.percentile(all_values, percentile_low)
p_high = np.percentile(all_values, percentile_high)
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
discrete[valid & (y < p_low)] = 0
discrete[valid & (y >= p_low) & (y <= p_high)] = 1
discrete[valid & (y > p_high)] = 2
```

iii. The notes justify the percentile thresholds as matching the task instructions. They also note the deliberate choice not to filter by likelihood first.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is cut from the same absolute `[-2.5, 1.5] s` interval around each trial’s go cue and binned onto the same 50 ms grid as the neural tensor.

ii. 
```python
t_start = go_time + bin_centers[0] - bin_width / 2
t_end = go_time + bin_centers[-1] + bin_width / 2
abs_bin_edges = go_time + bin_centers[0] - bin_width/2 + np.arange(n_bins + 1) * bin_width
```

iii. The trajectory explicitly says the agent wanted tongue output on the same go-aligned frame as spikes and photostim.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The script mostly handles issues with silent defaults or coarse exclusion:
- sessions with fewer than 2 good units or too few valid trials are skipped;
- sessions failing performance criteria are skipped;
- unknown categorical values default to `0` via `.get(..., 0)`;
- missing tongue bins are left as `NaN` initially, then become the default middle class `1` after discretization;
- sessions with no tongue samples at all return all-middle tongue labels;
- global warnings are suppressed with `warnings.filterwarnings('ignore')`.

ii. 
```python
warnings.filterwarnings('ignore')
...
if n_good < 2: return None
if len(valid_trial_indices) < 2: return None
...
choice = CHOICE_MAP.get(str(trial_instruction[trial_idx]), 0)
out = OUTCOME_MAP.get(str(outcome[trial_idx]), 0)
el = EARLY_LICK_MAP.get(str(early_lick[trial_idx]), 0)
```

```python
if len(all_values) == 0:
    return [np.ones(len(y), dtype=np.int64) for y in tongue_y_session]
...
discrete = np.ones(len(y), dtype=np.int64)  # default middle
```

iii. The notes justify some of these as practical safeguards, especially skipping bad sessions and defaulting missing tongue values to the middle bucket. The trajectory also records that the agent accepted thousands of all-zero neural trials as a “minor issue” instead of fixing the coverage logic.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is the neural firing-rate computation, especially the nested loop over every retained trial and every good neuron. The per-trial tongue/video extraction is a smaller but still repeated cost.

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

iii. `CONVERSION_NOTES.md` Step 7 gives runtime estimates showing firing-rate computation as the slowest stage by far, with input/output assembly much cheaper.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization targets are:
- the nested `trial x neuron` histogram loop in `compute_firing_rates_session`;
- the per-bin loop in `get_tongue_y_for_trial`;
- the per-trial construction of constant output arrays and repeated photostim overlap checks.

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
for t_idx in range(n_valid):
    out_arr = np.stack([
        np.full(N_TIMEBINS, output_choice[t_idx], dtype=np.int64),
        ...
    ], axis=0)
```

iii. The notes mention runtime pressure and estimate full conversion time, but do not record any successful vectorization beyond using `searchsorted` before each histogram.

## 10-c. What processing does the code repeat multiple times?

i. The code repeatedly rebuilds per-trial bin edges, repeatedly scans stimulation intervals against each trial window, repeatedly rebins tongue data trial by trial, and repeatedly broadcasts trial-level outputs into 80-bin arrays.

ii. 
```python
bin_edges = go_time + window_start + np.arange(n_bins + 1) * bin_width
```

```python
stim_mask = (photostim_stops_all > trial_window_start) & (photostim_starts_all < trial_window_end)
```

```python
tongue_y = get_tongue_y_for_trial(tongue_ts_all, tongue_data_all, go_time)
```

iii. No separate explicit justification was given for these repetitions; they are simply how the final implementation is structured.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script loads `subject_desc`, `auto_water`, and `free_water` but never uses them in the final dataset. It also computes and prints detailed summary statistics, optionally creates processing plots, and builds temporary trial-level output lists before restacking them. `subject_desc` is returned from `process_session` but then discarded when the final pickle is assembled.

ii. 
```python
subject_desc = nwb.subject.description
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
...
return {
    ...
    'subject_desc': subject_desc,
    ...
}
```

```python
if show_processing:
    plot_processing(...)
...
print(f'\nOutput distributions:')
```

iii. No explicit justification for these extra steps appears in the notes beyond debugging, validation, and documentation convenience.
