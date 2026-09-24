# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `.nwb` file beneath sorted `/app/data/sub-*` directories, opens each file once with `pynwb.NWBHDF5IO`, and processes it as a session.

ii.
```python
for subdir in sorted(os.listdir(data_dir)):
    ...
    for f in sorted(os.listdir(sub_path)):
        if f.endswith('.nwb'):
            nwb_files.append(os.path.join(sub_path, f))
...
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. The notes identify 28 subject directories and 174 NWB session files and justify direct NWB loading as access to the published data.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the containing directory name (for example, `sub-440956`), deduplicated and sorted; each session gets an index into that list.

ii.
```python
subject_id = nwb_path.split('/')[-2]
all_subjects = sorted(set(s['subject_id'] for s in all_sessions_data))
subject_idx.append(subject_to_idx[sess['subject_id']])
```

iii. The agent treats the DANDI `sub-*` layout as the subject boundary and reports 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its filename (without `.nwb`) is the session ID, and one result is appended per surviving file.

ii.
```python
session_id = os.path.basename(nwb_path).replace('.nwb', '')
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, ...)
    if result is not None:
        all_sessions.append(result)
```

iii. The notes state that 174 files exist and 173 remain after the session with zero classifier-good units is skipped, matching the paper's usable-session count.

## 1-d. How are the data split into trials?

i. Rows of the NWB `trials` interval table define trials. Same-position entries in the go-cue array and trial-table columns are subset with `valid_indices`.

ii.
```python
trials = nwb.intervals['trials']
n_trials_total = len(trials)
trial_starts = trials['start_time'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
...
go_times_valid = go_times[valid_indices]
```

iii. The agent relies on the NWB trial table and corresponding BehavioralEvents arrays as the dataset's explicit trial organization.

## 1-e. How are trials filtered based on quality controls?

i. It removes `auto_water` and `free_water` trials and any trial whose entire `[-2.5, 1.5]` s go-aligned window is not within the intersection of the first-to-last observation range of every good unit. Sessions with fewer than two remaining trials are dropped. Early-lick, ignore, and photostimulation trials are retained.

ii.
```python
valid_trials = (auto_water == 0) & (free_water == 0)
recording_covered = (go_times + t_start >= rec_start) & (go_times + t_end <= rec_end)
valid_trials = valid_trials & recording_covered
...
if n_valid < 2:
    return None
```

iii. The notes call water-delivery trials nongenuine, retain variables required as decoder inputs/outputs, and add coverage filtering after discovering 1,061 all-zero neural trials. This leaves 89,532 trials and two all-zero edge cases.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units whose `classification` equals `good`; go-cue timestamps define trial alignment.

ii.
```python
classifications = units['classification'][:]
good_indices = np.where(classifications == 'good')[0]
for idx in good_indices:
    spike_times_list.append(units['spike_times'][idx])
```

iii. The notes identify the NWB classification as the published region-specific spike-sorting QC classifier and report 69,453 retained units.

## 2-b. How is the `neural` data processed?

i. For each good neuron and trial, spikes in the requested window are made relative to the go cue, histogrammed into nonoverlapping bins, and counts are divided by bin width to produce Hz. There is no smoothing or normalization.

ii.
```python
relative_spikes = spk_times[lo:hi] - go
counts, _ = np.histogram(relative_spikes, bins=bin_edges)
fr_all[i, j, :] = counts / bin_width
```

iii. The agent chose 50-ms count/rate bins to satisfy the decoder instructions, while noting that the paper code used a 100-ms window and 50-ms stride.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only `classification == 'good'` units are retained; a session with none is skipped. No further metric or zero-variance filter is applied.

ii.
```python
good_mask = classifications == 'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    return None
```

iii. The notes justify this column as the precomputed QC-classifier output and explain that the one unclassified session is excluded.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are shifted by each trial's go-cue timestamp before histogramming over `[-2.5, 1.5]` s.

ii.
```python
go = go_times[i]
lo = np.searchsorted(spk_times, go + t_start)
hi = np.searchsorted(spk_times, go + t_end)
relative_spikes = spk_times[lo:hi] - go
```

iii. The notes state that NWB spike and event timestamps share the session clock, so subtracting the go time performs the required alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins from -2.5 to +1.5 s. Raw spike times are binned once; no later rebinning is applied.

ii.
```python
n_bins = int(round((t_end - t_start) / bin_width))
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
...
bin_width = 0.05
```

iii. The task explicitly says 50-ms-width bins, so the agent uses 50-ms width and stride instead of the paper's overlapping 100-ms windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from BehavioralEvents `sample_start_times`, trial start times, go-cue times, and the common bin centers. The last sample event between trial start and go is selected.

ii.
```python
samp_in_trial = sample_start_times[(sample_start_times >= ts) & (sample_start_times < go)]
tone_onsets[i] = samp_in_trial[-1]
```

iii. The notes say sample/tone onset is typically 1.85 s before go and that the last onset handles replayed sample epochs after early licks.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone's go-relative time is subtracted from each go-relative bin center. If no sample event is found, tone onset is imputed as `go - 1.85` s.

ii.
```python
tone_onsets[i] = go - 1.85
...
tone_onset_rel = tone_onsets[i] - go_times_valid[i]
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. The fallback reflects the typical fixed trial timing; the notes explicitly list it as an edge-case safeguard.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the identical go-relative 50-ms bin centers used for neural firing rates.

ii.
```python
fr_list, bin_centers = compute_firing_rates_vectorized(...)
...
inputs[0, :] = bin_centers - tone_onset_rel
```

iii. The agent's processing plots and direct NWB sanity check were used to verify the shared time axis.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus go-cue timestamps.

ii.
```python
photostim_onset_str = trials['photostim_onset'][:]
photostim_duration_str = trials['photostim_duration'][:]
stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
```

iii. The notes describe onset as trial-relative metadata that must be placed onto the go-aligned time axis.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stimulated trials, onset and duration are converted to a half-open interval; bins whose centers lie in it are 1. Nonstimulated (`N/A`) trials remain all zero.

ii.
```python
if photostim_onset_valid[i] != 'N/A':
    stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
    stim_end_go_rel = stim_onset_go_rel + stim_dur
    inputs[1, :] = ((bin_centers >= stim_onset_go_rel) &
                    (bin_centers < stim_end_go_rel)).astype(np.float32)
```

iii. The agent chose a binary time-varying signal, as required, rather than a trial-level stimulation flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-relative onset is converted to an absolute timestamp and then to go-relative time, after which it is tested at the neural bin centers.

ii.
```python
stim_onset_abs = trial_starts_valid[i] + float(photostim_onset_valid[i])
stim_onset_go_rel = stim_onset_abs - go_times_valid[i]
```

iii. This puts stimulation and neural activity on the same go-cue-aligned grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from BehavioralEvents `left_lick_times` and `right_lick_times`, together with each go cue and trial stop, rather than from trial instruction and outcome.

ii.
```python
left_after = left_lick_times[(left_lick_times > go_time) & (left_lick_times < trial_stop)]
right_after = right_lick_times[(right_lick_times > go_time) & (right_lick_times < trial_stop)]
```

iii. The notes say choice should be the first lick after the go cue and use no lick when neither stream contains a response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The earlier qualifying left/right lick selects code 0/1; a tie or absence gives 2. This per-trial value is repeated across all bins.

ii.
```python
if first_left < first_right:
    return 0
elif first_right < first_left:
    return 1
else:
    return 2
...
outputs[0, :] = choice
```

iii. The three codes are documented as left, right, and no lick; repetition makes all outputs share a time-shaped array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the trials-table `outcome` strings.

ii.
```python
outcomes = trials['outcome'][:]
outcomes_valid = outcomes[valid_indices]
```

iii. The raw field already contains the requested hit, miss, and ignore categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped as hit=0, miss=1, ignore=2 (unknown values also default to 2) and repeated over all bins.

ii.
```python
outcome_map = {'hit': 0, 'miss': 1, 'ignore': 2}
outcome = outcome_map.get(outcomes_valid[i], 2)
outputs[1, :] = outcome
```

iii. The chosen code order is consistently reflected in `output_values`; the agent uses a per-trial label expanded across time.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` field.

ii.
```python
early_lick = trials['early_lick'][:]
early_lick_valid = early_lick[valid_indices]
```

iii. The notes retain early-lick trials specifically because this flag is a requested decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other value to 0; the result is repeated across bins.

ii.
```python
early = 1 if early_lick_valid[i] == 'early' else 0
outputs[2, :] = early
```

iii. The codes are documented as `no_early` and `early`, consistent with a binary per-trial output.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `Camera0_side_TongueTracking` data column 1 (y), column 2 (tracking likelihood), and its timestamps, plus go-cue times.

ii.
```python
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
...
tongue_y = tongue_data[:, 1]
tongue_lk = tongue_data[:, 2]
```

iii. The notes identify the side-camera tongue tracker as the relevant time series and use likelihood to decide visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session-wide p40/p60 are computed over raw y values from frames with likelihood at least 0.9. Within each trial and 50-ms bin, visible-frame y values are averaged and categorized; bins without a visible frame become class 3. Missing tongue streams make every bin class 3.

ii.
```python
visible_mask = tongue_data[:, 2] >= 0.9
y_visible = tongue_data[visible_mask, 1]
p40 = np.percentile(y_visible, 40)
p60 = np.percentile(y_visible, 60)
...
visible = trial_lk_v[mask] >= likelihood_threshold
mean_y = np.mean(trial_y_v[mask][visible])
```

iii. The agent chose 0.9 as a visibility-confidence threshold and reports an optimized timestamp/bin lookup after the initial implementation was slow.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. A visible bin mean is 0 below p40, 1 from p40 through p60, and 2 above p60; 3 means not visible. Thresholds are per-session percentiles of visible raw frames.

ii.
```python
if mean_y < p40:
    result[i, b] = 0
elif mean_y <= p60:
    result[i, b] = 1
else:
    result[i, b] = 2
```

iii. The labels directly follow the requested 40th/60th-percentile category descriptions, although the percentile population is raw frames rather than binned means.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected between `go+t_start` and `go+t_end`, then assigned to the same go-relative 50-ms intervals as neural data.

ii.
```python
abs_edges = go + bin_edges
idx_start = np.searchsorted(tongue_timestamps, abs_edges[0])
idx_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
bin_indices = np.searchsorted(abs_edges, trial_ts, side='right') - 1
```

iii. Shared absolute timestamps allow direct go-cue alignment without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units and sessions with fewer than two valid trials are skipped; trials lacking full common recording coverage are removed; absent tone events are imputed at go-1.85 s; absent/unusable tongue data becomes `not_visible`; unknown outcomes default to ignore. The code suppresses all warnings globally.

ii.
```python
if n_good == 0: return None
if n_valid < 2: return None
tone_onsets[i] = go - 1.85
tongue_y_list = [np.full(n_bins, 3, dtype=np.int64) ...]
outcome = outcome_map.get(outcomes_valid[i], 2)
warnings.filterwarnings('ignore')
```

iii. The notes describe these as edge-case protections and document the coverage fix prompted by zero-neural trials; two such trials nevertheless remain.

## 10-a. What are the most time-consuming steps of the code?

i. NWB/spike loading and the nested neuron-by-trial firing-rate histogram loop dominate; plotting is also costly when explicitly requested. The notes measured about 1.8-3.1 s/session for firing rates and about 0.2-0.3 s/session for tongue processing after optimization.

ii.
```python
for j, spk_times in enumerate(spike_times_list):
    for i in range(n_trials):
        ...
        counts, _ = np.histogram(relative_spikes, bins=bin_edges)
```

iii. The notes report a 22x improvement over the initial masking/tongue implementation and 589.8 s for full conversion.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The inner per-trial neural histogram loop could be replaced by one search over flattened edges per unit. The per-bin tongue loop could use grouped sums/counts (`bincount`), and the per-trial input/output construction and lick selection could also be array-oriented.

ii.
```python
for j, spk_times in enumerate(spike_times_list):
    for i in range(n_trials):
        ...
for i in range(n_trials):
    ...
    for b in range(n_bins):
```

iii. The agent calls the routines vectorized because frame/spike window lookup uses `searchsorted`, but its notes acknowledge the original loops as bottlenecks and only partially vectorize them.

## 10-c. What processing does the code repeat multiple times?

i. It rebuilds identical bin edges in both neural and tongue functions, repeatedly filters global lick arrays per trial, and repeatedly performs per-trial/per-bin selection. Per-trial outputs and three scalar labels are also copied across 80 bins by design.

ii.
```python
bin_edges = t_start + np.arange(n_bins + 1) * bin_width
...
left_after = left_lick_times[(left_lick_times > go_time) & ...]
...
outputs[0, :] = choice
outputs[1, :] = outcome
outputs[2, :] = early
```

iii. Repetition mostly arises from keeping processing straightforward and producing the common `(n_output, n_timepoints)` layout.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `instructions_valid` only for optional plots, computes and returns `bin_centers` for every session but discards them during assembly, and computes several diagnostic counts/timings used only for console output. With `--show-processing`, it performs extensive plotting that is not part of the converted dataset.

ii.
```python
instructions_valid = instructions[valid_indices]
...
'bin_centers': bin_centers,
...
if show_processing:
    plot_processing(...)
```

iii. These operations support diagnostics, validation, and documentation rather than decoder consumption; plotting is optional.
