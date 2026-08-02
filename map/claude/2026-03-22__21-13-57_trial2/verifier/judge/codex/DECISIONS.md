# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans `/app/data` for `sub-*` directories, then scans each subject directory for `.nwb` files. It treats each NWB file as one session candidate, opens it with `NWBHDF5IO`, and reads session contents directly from NWB tables and acquisitions inside `process_session()`.

ii. ```python
def get_nwb_files(data_dir):
    subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
    nwb_files = []
    for subj in subjects:
        subj_dir = os.path.join(data_dir, subj)
        files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for fname in files:
            nwb_files.append((subj, os.path.join(subj_dir, fname)))
    return nwb_files

io = NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. In `CONVERSION_NOTES.md`, the agent states that the provided dataset is organized as NWB files by subject, not as the `.mat` exports used by the reference preprocessing code, so it decided to load directly from NWB while trying to mirror the reference logic after loading.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the top-level `sub-*` directory names. The script keeps a `subjects_seen` map and assigns each retained session a `subject_idx` pointing into `subjects_list`.

ii. ```python
if subject_id not in subjects_seen:
    subjects_seen[subject_id] = len(subjects_seen)
    subjects_list.append(subject_id)
subj_idx = subjects_seen[subject_id]
subject_idx_list.append(subj_idx)
```

iii. The notes say the NWB files are "organized by subject" and that `subject_id` should populate `subjects` / `subject_idx` in the target decoder format.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Sessions are kept only if `process_session()` returns usable data; sessions with no good units, no mapped neurons, or fewer than two kept trials are dropped.

ii. ```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    result = process_session(nwb_path, subject_id)
    if result is None:
        print("SKIPPED (no good units or trials)")
        continue
    all_sessions.append(result)
```

iii. In the notes, the agent explicitly chose to "Include all 173 sessions with good units," interpreting the paper's 173 behavioral sessions as the dataset-level session count rather than a behavior-filtered analysis subset.

## 1-d. How are the data split into trials?

i. Trials are defined from rows of `nwb.trials`, then filtered by a boolean `trial_mask`. Trial-specific neural/input/output arrays are built by iterating over the surviving `trial_indices`.

ii. ```python
n_trials_total = len(nwb.trials)
trials = nwb.trials
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]

for ti in trial_indices:
    go_cue = go_start_times[ti]
    ...
```

iii. The notes describe the trial structure as coming from the NWB `trials` table plus event streams, and later note that `obs_intervals` forced an additional recording-coverage filter so only trials with valid neural recording are retained.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes only `auto_water` and `free_water` trials, then further excludes trials not covered by the retained units' recording intervals (`obs_intervals`). It keeps early-lick, ignore, and photostim trials.

ii. ```python
trial_mask = (auto_water == 0) & (free_water == 0)
recording_mask = np.zeros(n_trials_total, dtype=bool)
recording_mask[valid_trial_idx] = True
trial_mask = trial_mask & recording_mask
trial_indices = np.where(trial_mask)[0]
```

iii. The notes justify this as a deliberate deviation from `get_regular_trial_mask()` in the reference code: early lick is a decoder output, photostimulation is a decoder input, and ignore is a requested outcome class, so the agent kept them and removed only auto/free water plus trials outside valid recording periods.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `nwb.units['spike_times']` for units with `classification == 'good'`, after region mapping via `anno_name` and trial-validity filtering via `obs_intervals`.

ii. ```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
anno_names = nwb.units['anno_name'][:][good_mask]
...
for idx in good_unit_indices:
    st = nwb.units['spike_times'][idx]
    spike_times_per_unit.append(st)
```

iii. The notes repeatedly say the NWB `classification == 'good'` field corresponds to the reference QC classifier output and that spike times are the raw neural variable to convert into per-trial firing-rate features.

## 2-b. How is the `neural` data processed?

i. For each kept trial, the script takes absolute spike times from each retained neuron, restricts them to the trial window relative to that trial's go cue, bins them into 50 ms non-overlapping bins, and divides by bin width to get firing rates in Hz.

ii. ```python
def compute_firing_rates(spike_times_list, go_cue_time, t_start, t_end, bin_size):
    n_bins = int((t_end - t_start) / bin_size)
    fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
    abs_start = go_cue_time + t_start
    abs_end = go_cue_time + t_end
    ...
    np.add.at(fr[i], bin_indices, 1.0)
    fr /= bin_size
    return fr
```

iii. The notes state that the reference code uses `sliding_histogram()` with 40 ms width and 3.4 ms stride, but the agent chose 50 ms non-overlapping bins because the decoder instructions explicitly required 50 ms bins from `-2.5 s` to `1.5 s`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if they are NWB-classified as `good`, can be mapped from fine `anno_name` annotations into one of 14 coarse regions, and survive session-level checks. No explicit low-firing-rate threshold is applied.

ii. ```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
...
region = map_anno_to_region(str(anno))
if region is None:
    valid_neuron_mask[i] = False
...
good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
```

iii. The notes justify `classification == 'good'` as the NWB analogue of the reference QC classifier. The notes also explicitly say the method-paper 2 Hz cutoff was treated as analysis-specific and therefore not applied here.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go cue onset. For each trial, the trial's `go_start_times` timestamp defines time zero, and the script extracts `[-2.5, +1.5)` seconds around that go cue.

ii. ```python
go_start_times = events.time_series['go_start_times'].timestamps[:]
...
go_cue = go_start_times[ti]
fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
```

iii. The notes cite the decoder task requirement to align everything to Go cue onset and treat the reference code's go-cue alignment as conceptually consistent with this choice.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 50 ms bins over a 4.0 s window, giving 80 time bins per trial. The script does not first create a higher-resolution representation and then rebin it; it bins spikes directly into 50 ms trial bins.

ii. ```python
BIN_SIZE_S = 0.05
T_START = -2.5
T_END = 1.5
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)  # 80
```

iii. The notes explicitly call out this difference from the reference code's 40 ms / 3.4 ms sliding histogram and say it was chosen to satisfy the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times` and `go_start_times` in `BehavioralEvents`. For each trial, the script finds the last sample onset before that trial's go cue.

ii. ```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
go_start_times = events.time_series['go_start_times'].timestamps[:]

def find_last_sample_before_go(sample_start_times, go_cue_time):
    valid = sample_start_times[sample_start_times < go_cue_time]
    if len(valid) == 0:
        return None
    return valid[-1]
```

iii. The notes and trajectory say the agent noticed there are more sample-start events than trials because early licks trigger replays, so it chose the last sample onset before each go cue as the tone onset for that trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After finding a per-trial tone onset, the script computes the center of each 50 ms go-cue-aligned bin and subtracts the tone onset time relative to go cue. If no earlier sample onset is found, it falls back to `go_cue - 1.85`.

ii. ```python
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
if tone_onset is None:
    tone_onset = go_cue - 1.85
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)

def compute_time_from_tone(go_cue_time, tone_onset_time, t_start, t_end, bin_size):
    bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
    tone_rel = tone_onset_time - go_cue_time
    time_from_tone = bin_centers - tone_rel
```

iii. The notes justify the "last sample start" logic using replayed sample epochs after early licks. The trajectory also shows the agent recognized some timing outliers and kept a hard-coded `1.85 s` fallback as a safety default.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone trace is evaluated on the same 80 bin centers as the neural firing rates, using the same per-trial go cue as time zero.

ii. ```python
input_data = np.stack([time_from_tone, photostim_ts], axis=0)
...
fr = compute_firing_rates(..., go_cue, T_START, T_END, BIN_SIZE_S)
time_from_tone = compute_time_from_tone(go_cue, tone_onset, T_START, T_END, BIN_SIZE_S)
```

iii. The notes frame this as one of the central consistency requirements: all streams should be aligned to the same go-cue-based time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the absolute `photostim_start_times` and `photostim_stop_times` event streams in `BehavioralEvents`.

ii. ```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The trajectory shows the agent inspected both trial-level photostim fields and event-level photostim timestamps, then decided the absolute event timestamps were the safer source for bin-level alignment.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, the script selects photostim intervals that overlap the trial window and creates a binary time series whose entries are 1 for bin centers inside any overlapping photostim interval and 0 otherwise.

ii. ```python
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
ps_starts = photostim_start_abs[stim_mask]
ps_stops = photostim_stop_abs[stim_mask]
photostim_ts = compute_photostim_timeseries(go_cue, ps_starts, ps_stops,
                                            T_START, T_END, BIN_SIZE_S)
```

iii. The notes say the decoder needs photostimulation as a time-varying input, so the agent kept stimulated trials and encoded photostim as a binary series rather than filtering them out.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The script uses the same trial-specific go cue as neural time zero, computes the same 50 ms bin centers, converts them back to absolute time, and checks whether each bin center falls inside any photostim interval.

ii. ```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
abs_centers = bin_centers + go_cue_time
for start, stop in zip(photostim_starts, photostim_stops):
    mask = (abs_centers >= start) & (abs_centers <= stop)
    stim[mask] = 1.0
```

iii. The notes explicitly say photostim should be "aligned to go cue" using absolute `photostim_start/stop_times`.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent did not derive choice from actual lick behavior. It uses `trials['trial_instruction']` and treats that instruction label as the output choice.

ii. ```python
instructions = trials['trial_instruction'][:]
...
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The notes map `trial_instruction` to `output[0]: choice`, and the README later describes this output as "Trial instruction (lick direction)," showing the agent consciously equated instructed side with lick choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The script performs a direct binary mapping: `'left' -> 0`, anything else -> `1`. It then repeats this scalar across all 80 time bins so the output becomes time-varying in shape, even though the quantity is per-trial.

ii. ```python
choice = 0 if instructions[ti] == 'left' else 1
...
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[0, :] = out_dict['choice']
```

iii. The notes say per-trial outputs were replicated across time because the decoder accepts time-varying outputs, but they do not justify the semantic choice of using instruction instead of observed lick behavior.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `trials['outcome']`.

ii. ```python
outcomes = trials['outcome'][:]
...
outcome_str = outcomes[ti]
```

iii. The notes map the NWB `outcome` field to decoder output `ignore / miss / hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script maps string outcomes to integer classes `ignore=0`, `miss=1`, `hit=2`, with unknown strings defaulting to `0`, and then repeats the trial-level value across all time bins.

ii. ```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
...
full_output[1, :] = out_dict['outcome']
```

iii. The notes say this mapping was chosen to match the decoder specification exactly.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. The script does not compute any "distance to reward zone" output. Instead, it stores only per-trial outcome and repeats that outcome label across all neural time bins.

ii. ```python
full_output = np.zeros((4, N_TIMEBINS), dtype=np.int64)
full_output[1, :] = out_dict['outcome']
```

iii. Neither the notes nor the script mention a reward-zone distance variable. This appears to be a mismatch between the evaluation prompt and the actual decoder task, which asked for `Outcome`, not reward-zone distance.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from `trials['early_lick']`.

ii. ```python
early_lick = trials['early_lick'][:]
...
early_val = 0 if early_lick[ti] == 'no early' else 1
```

iii. The notes explicitly say early lick was kept because it is one of the requested decoder outputs, even though reference analysis code usually filters such trials out.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The script maps `'no early'` to 0 and all other values to 1, then repeats the result across all 80 time bins.

ii. ```python
early_val = 0 if early_lick[ti] == 'no early' else 1
...
full_output[2, :] = out_dict['early_lick']
```

iii. The notes justify this as a direct implementation of the decoder spec's binary early-lick label.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the side-camera tongue tracking time series: `Camera0_side_TongueTracking.data[:, 1]` for y, `[:, 2]` for likelihood, plus the corresponding timestamps.

ii. ```python
tongue_ts = nwb.acquisition['BehavioralTimeSeries'].time_series['Camera0_side_TongueTracking']
tongue_data_all = tongue_ts.data[:]  # (n_frames, 3): x, y, likelihood
tongue_timestamps = tongue_ts.timestamps[:]
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. The notes say the agent identified the side-camera tongue tracking array as `(x, y, likelihood)` and selected column 1 as the requested y-position signal.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the script extracts tongue samples inside the go-cue-aligned window, discards samples with likelihood `< 0.9`, bins the remaining y-values into the 50 ms neural bins by average value, and leaves empty bins as `NaN`.

ii. ```python
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
...
y_in_window[like_in_window < 0.9] = np.nan
...
sums = np.bincount(valid_bins, weights=valid_vals, minlength=n_bins)
counts = np.bincount(valid_bins, minlength=n_bins)
tongue_y_binned[has_data] = (sums[has_data] / counts[has_data]).astype(np.float32)
```

iii. The notes justify this as using the raw DLC tongue trace with a likelihood filter. The trajectory shows the agent recognized the 300 Hz tongue data and chose simple bin averaging rather than reusing the reference video-processing pipeline.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After collecting all valid tongue y values from the session, the script computes the session's 40th and 60th percentiles and assigns each binned value to class 0, 1, or 2. Bins left as `NaN` are not specially represented; because the discretized array is initialized to zeros, missing bins effectively become class 0.

ii. ```python
if len(tongue_y_session) > 0:
    p40 = np.percentile(tongue_y_arr, 40)
    p60 = np.percentile(tongue_y_arr, 60)
...
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
valid = ~np.isnan(tongue_y_raw)
tongue_y_disc[valid & (tongue_y_raw < p40)] = 0
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. The notes explicitly say the discretization uses per-session 40th and 60th percentiles over all valid time points, matching the task instructions. They do not discuss the consequence that missing bins collapse to class 0.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue timestamps are cut to the same `[-2.5, +1.5)` window around the per-trial go cue and then averaged within the same 50 ms bins used for neural firing rates.

ii. ```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The notes treat this as part of the same global go-cue alignment requirement used for all neural, input, and output streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or problematic data are handled with ad hoc defaults: sessions with no good units or too few kept trials are skipped; neurons whose annotations do not map to a coarse region are dropped; missing sample-onset matches fall back to `go_cue - 1.85`; low-likelihood tongue samples become `NaN`; empty tongue bins stay `NaN` until discretization, where they effectively become class 0; unknown outcome strings default to class 0.

ii. ```python
if n_good == 0:
    io.close()
    return None
...
if region is None:
    valid_neuron_mask[i] = False
...
if tone_onset is None:
    tone_onset = go_cue - 1.85
...
y_in_window[like_in_window < 0.9] = np.nan
...
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
```

iii. The notes and trajectory justify these as pragmatic fixes for imperfect NWB/event alignment and missing tracking values, but they do not show a principled treatment for all missing-data cases.

## 10-a. What are the most time-consuming steps of the code?

i. The slowest work is per-session trial processing: repeated spike binning in `compute_firing_rates()`, trial-validity determination from `obs_intervals`, and per-trial tongue binning. Full conversion took about 2301 seconds for 173 sessions.

ii. ```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, T_START, T_END, BIN_SIZE_S)
    ...
    tongue_y_trial = get_tongue_y_for_trial(...)
```

iii. The trajectory explicitly calls out `get_valid_trial_indices()` as a bottleneck, and the notes estimate ~6.5 seconds per session before full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining vectorization opportunities are the per-neuron spike loop inside `compute_firing_rates()`, the outer per-trial session loop, the annotation-to-region mapping loop, and the repeated Python-level construction of output statistics at the end of `main()`.

ii. ```python
for i, st in enumerate(spike_times_list):
    ...
for ti in trial_indices:
    ...
for i, anno in enumerate(anno_names):
    ...
for session_outputs in data['output']:
    for trial_out in session_outputs:
        ...
```

iii. The trajectory shows the agent thinking explicitly about vectorization and bottlenecks, but the final script remains mostly serial at the session and neuron loop levels.

## 10-c. What processing does the code repeat multiple times?

i. The script repeatedly recomputes the same bin counts / bin centers inside helper functions, rescans `sample_start_times` for every trial, builds trial-level photostim masks separately for every trial, and replicates trial-level labels across all 80 bins.

ii. ```python
n_bins = int((t_end - t_start) / bin_size)
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
...
tone_onset = find_last_sample_before_go(sample_start_times, go_cue)
...
stim_mask = (photostim_stop_abs > trial_abs_start) & (photostim_start_abs < trial_abs_end)
...
full_output[0, :] = out_dict['choice']
full_output[1, :] = out_dict['outcome']
full_output[2, :] = out_dict['early_lick']
```

iii. The notes mention runtime concerns and some vectorization, but they also document the final design as repeated per-trial computations around the common go-cue-aligned bin grid.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra work that is not needed by the saved dataset or most downstream analysis: it optionally generates processing plots, computes output / brain-region summary counters only for printing, stores trial-level categorical outputs as full 80-bin repeated arrays, and computes raw tongue traces only to discard them after percentile discretization.

ii. ```python
if args.show_processing and i < 2:
    plot_processing(result, i, nwb_path)

for session_outputs in data['output']:
    for trial_out in session_outputs:
        all_choices.append(int(trial_out[0, 0]))
...
output_trials_raw.append({
    'choice': choice,
    'outcome': outcome_val,
    'early_lick': early_val,
    'tongue_y_raw': tongue_y_trial
})
```

iii. The notes describe the plotting and summary printing as validation aids, and the time-repeated outputs were justified as a decoder-format convenience rather than as information-preserving processing.
