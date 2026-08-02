# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `sub-*` directory under `data/`, treats every `.nwb` file inside as one candidate session, and opens each file with `h5py`. Within each session it reads trial metadata from `intervals/trials`, behavioral event timestamps from `acquisition/BehavioralEvents`, unit spike data from `units`, and tongue tracking from `acquisition/BehavioralTimeSeries`.

ii.

```python
def get_nwb_files():
    subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
    all_files = []
    for subj in subjects:
        subj_dir = os.path.join(DATA_DIR, subj)
        nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
        for nwb_file in nwb_files:
            all_files.append({
                'subject': subj,
                'path': os.path.join(subj_dir, nwb_file),
                'filename': nwb_file
            })

...

f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
```

iii. In `CONVERSION_NOTES.md` Step 1 and Step 2, the agent says the reference code used DataJoint-exported `.mat` files but the provided dataset is NWB, so it mapped NWB fields to the reference variables and loaded all 174 NWB files directly.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined by the NWB parent directory name, e.g. `sub-440956`. The final `subjects` list is built from unique subject IDs among sessions that pass filtering, and `subject_idx` stores the per-session index into that list.

ii.

```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
...
for subj in subjects:
    subj_dir = os.path.join(DATA_DIR, subj)
...
subj = result['subject']
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The notes explicitly record 28 subject directories and describe them as the mice. The trajectory also shows the agent exploring the directory structure before coding.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. A session only enters the output dataset if `process_session(...)` returns a valid result after the session-level filters are applied.

ii.

```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(
        nwb_info['path'],
        nwb_info['subject'],
        show_processing=args.show_processing,
        session_idx=session_count
    )

    if result is None:
        skipped += 1
        continue

    all_neural.append(result['neural'])
```

iii. `CONVERSION_NOTES.md` Step 2 says there are 174 NWB files total and Step 9 reports that 143 sessions survived the filters, so the agent clearly equated session identity with file identity.

## 1-d. How are the data split into trials?

i. Trials start from the NWB trial table rows (`intervals/trials/id` length). The agent keeps a trial index array `trial_indices = np.where(trial_mask)[0]`; every selected trial becomes one trial entry in the session lists for `neural`, `input`, and `output`.

ii.

```python
trials = f['intervals']['trials']
n_trials_total = len(trials['id'])
...
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
trial_indices = np.where(trial_mask)[0]
...
for trial_idx in trial_indices:
    ...
    inputs_list.append(input_trial)
...
for i, trial_idx in enumerate(trial_indices):
    ...
    outputs_list.append(output_trial)
```

iii. The notes describe `intervals/trials/` as the native trial table and say the agent needed to filter that table down to the subset with usable neural coverage.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes only `auto_water` and `free_water` trials from the converted dataset, plus trials without enough neural recording coverage (`neural_valid`). It deliberately does not exclude early-lick, ignore/no-response, or photostim trials from the exported dataset, but it does use only non-early, control trials to compute session inclusion statistics.

ii.

```python
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
...
is_control = (photostim_onset_trial == b'N/A') & trial_mask
is_not_early = early_lick == b'no early'
control_non_early = is_control & is_not_early
...
correct_rate = hits / (hits + misses)
```

iii. The trajectory explicitly says the reference `get_regular_trial_mask` excludes early lick, no response, and photostim, but the agent chose to retain them because the decoder task requires `early_lick`, `outcome`, and `photostimulation` as output/input variables. The notes repeat that rationale in Step 3 and Step 5.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrices are derived primarily from `units/spike_times` and `units/spike_times_index`, with `go_start_times/timestamps` providing the alignment event and `classification` defining which units are kept. Brain region metadata comes from `units/electrodes`, `units/electrodes_index`, and the electrode `location` JSON.

ii.

```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
classification = f['units']['classification'][:]
...
electrodes_idx = f['units']['electrodes'][:]
electrode_locations = f['general']['extracellular_ephys']['electrodes']['location'][:]
```

iii. The notes say the agent mapped the NWB spike-time representation to the reference spike-time arrays and used `classification='good'` as the QC-equivalent of the reference `goodunits` classifier output.

## 2-b. How is the `neural` data processed?

i. For each selected trial and each retained unit, the agent extracts spikes in a `[-2.5, 1.5] s` window around the go cue, subtracts the trial's go time, histograms the aligned spikes into 50 ms non-overlapping bins, and divides by bin width to obtain firing rates in Hz.

ii.

```python
abs_start = go_time + window_start
abs_end = go_time + window_end
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. Step 5 and Step 6 of the notes state that the agent intentionally followed the decoder specification rather than the reference code's `bw=0.04, stride=0.0034` sliding histogram, so it switched to 50 ms bins over `[-2.5, 1.5]` while keeping the same go-cue alignment logic.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Unit-level QC keeps only `classification == b'good'`. Trial-level neural availability is approximated by requiring the trial index to fall within `units/is_good_trials.shape[1]` and the requested window to lie within the global spike-time range for the session.

ii.

```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
...
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    go = go_times[t_idx]
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
...
good_mask = classification == b'good'
```

iii. The trajectory shows the agent first discovering that some sessions continue behaviorally after ephys stops, then deciding to use `is_good_trials.shape[1]` plus spike-time coverage as a pragmatic neural-coverage filter. The notes say the `classification` field is the NWB form of the QC classifier output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to go cue onset. For each trial, spikes are converted from absolute timestamps to time-from-go-cue by subtracting `go_times[trial_idx]`, and only the `[-2.5, 1.5] s` window around that event is retained.

ii.

```python
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
...
go_time = go_cue_times[trial_idx]
...
aligned = unit_spikes[idx_lo:idx_hi] - go_time
```

iii. Both the task instructions and the notes explicitly name go cue onset as the temporal alignment event. The trajectory shows the agent verifying that NWB spike times are absolute and therefore need this subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural and behavioral time axes use 50 ms bins, giving 80 bins over the 4 s window. There is no second-stage rebinning; the spike histogram itself is directly computed at that resolution.

ii.

```python
BIN_WIDTH = 0.05
WINDOW_START = -2.5
WINDOW_END = 1.5
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))
...
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. The notes explicitly contrast the reference code's 40 ms sliding window with the decoder task's required 50 ms bins and say the agent followed the task specification here.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The agent derives tone onset from the global `sample_start_times` event stream and each trial's `go_start_times`. It assumes the last sample-start event before a given go cue corresponds to that trial's tone onset.

ii.

```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
...
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
tone_onset_rel = sample_starts_all[ss_idx] - go_time
```

iii. In the trajectory, the agent states that sample start is the tone onset and that it is typically 1.85 s before the go cue. It then implemented the searchsorted lookup on the NWB `sample_start_times` array.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The agent finds the most recent sample-start event before the go cue, computes its time relative to go cue, falls back to `-1.85` if no such event exists, and then converts the per-bin go-cue-centered time axis into elapsed time since tone onset via `bin_centers - tone_onset_rel`.

ii.

```python
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
time_from_tone = bin_centers - tone_onset_rel
```

iii. The notes say the agent believed tone onset was consistently 1.85 s before go cue, but the trajectory also shows it noticing suspicious `time_from_tone_onset` values up to about 11.9 s after the conversion. It never repaired this logic.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is represented on the same 80 go-cue-aligned bin centers as the neural data; each bin stores the elapsed time since the inferred tone onset for that trial.

ii.

```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
time_from_tone = bin_centers - tone_onset_rel
input_trial = np.stack([time_from_tone.astype(np.float32), photostim_binary], axis=0)
```

iii. The notes and trajectory both describe the design as 'same time bins as neural data, aligned to go cue'. The only disputed part is the tone-onset source, not the use of the shared bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the per-trial fields `photostim_onset`, `photostim_duration`, and `start_time`, together with `go_start_times` so the stimulation window can be converted into go-cue-relative time.

ii.

```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
...
go_time = go_times[trial_idx]
```

iii. The trajectory shows the agent comparing trial-table photostim metadata against `BehavioralEvents/photostim_start_times` and concluding that the trial-table fields are trial-relative, matching the reference `.mat` `task_stimulation` representation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stim trials, the onset and duration strings are cast to floats, converted from trial-start-relative time to go-cue-relative time, and then rasterized into a binary 50 ms time series using the shared bin centers. Non-stim trials remain all zeros.

ii.

```python
photostim_binary = np.zeros(n_bins, dtype=np.float32)
ps_onset_val = photostim_onset_trial[trial_idx]
if ps_onset_val != b'N/A':
    ps_onset_float = float(ps_onset_val)
    ps_dur_float = float(photostim_duration_trial[trial_idx])
    trial_start = trial_start_times[trial_idx]
    ps_rel_start = (trial_start + ps_onset_float) - go_time
    ps_rel_end = ps_rel_start + ps_dur_float
    photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The agent's notes say photostim trials are kept because photostimulation is a decoder input, and the trajectory shows it verifying that photostim onset is about `-0.5 s` relative to go cue, matching the methods section.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The binary photostim signal is put on the exact same go-cue-centered 50 ms bin grid as the neural firing rates and the tone-onset input.

ii.

```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2, WINDOW_END - BIN_WIDTH/2, n_bins)
...
photostim_binary = ((bin_centers >= ps_rel_start) & (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The notes describe a shared time base across `neural`, `input`, and `output`. The trajectory explicitly mentions that photostim ends before the go cue, consistent with the methods text.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from the trial-table field `trial_instruction`, which stores `left` or `right` for each trial.

ii.

```python
trial_instruction = trials['trial_instruction'][:]
...
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. The notes map `trial_instruction` directly to the decoder output `choice`, and the trajectory shows the agent inspecting that field early in data exploration.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The agent maps `left -> 0` and `right -> 1` to match the decoder specification, then broadcasts that per-trial label across all 80 time bins in `output_trial[0, :]`.

ii.

```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
...
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[0, :] = choice
```

iii. The notes explicitly record the left/right mapping. This deliberately differs from one convention in the reference code because the task instructions explicitly require left = 0 and right = 1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the NWB trial-table field `outcome`, whose values are strings such as `ignore`, `miss`, and `hit`.

ii.

```python
outcome = trials['outcome'][:]
...
out = outcome[trial_idx]
```

iii. The notes map the NWB `outcome` field directly to the decoder output and preserve the three categories requested by the task.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The agent maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, with an `else` fallback to 0, then repeats the categorical value across all 80 bins for the trial.

ii.

```python
if out == b'ignore':
    outcome_val = 0
elif out == b'miss':
    outcome_val = 1
elif out == b'hit':
    outcome_val = 2
else:
    outcome_val = 0
...
output_trial[1, :] = outcome_val
```

iii. The notes list exactly this mapping in Step 5 because it matches the decoder-task output coding.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This appears to be a template error in the evaluation prompt: the agent never created a `distance to reward zone` output, and the task instructions never requested one. For the implemented `Outcome` output, the agent aligned it by broadcasting the per-trial categorical label across the same 80 go-cue-centered bins used for the neural data.

ii.

```python
output_trial = np.zeros((4, n_bins), dtype=np.int64)
output_trial[1, :] = outcome_val
```

iii. The notes and code only define four outputs: choice, outcome, early lick, and tongue y-position. No distance-to-reward-zone variable appears anywhere.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from the trial-table field `early_lick`.

ii.

```python
early_lick = trials['early_lick'][:]
...
early = 1 if early_lick[trial_idx] == b'early' else 0
```

iii. The notes map `early_lick` directly to the decoder output and explain that these trials were intentionally retained so the decoder could predict early licking.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The agent maps `early -> 1` and anything else (effectively `no early`) to 0, then repeats that per-trial label across all neural time bins.

ii.

```python
early = 1 if early_lick[trial_idx] == b'early' else 0
...
output_trial[2, :] = early
```

iii. The notes list the requested coding as `no = 0, yes = 1`, and the trajectory explains why early-lick trials were retained even though the reference analysis often excluded them.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data`, using column 1 as `y` and column 2 as DeepLabCut-like likelihood/confidence, plus the matching timestamp vector for alignment.

ii.

```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['timestamps'][:]
...
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. The trajectory shows the agent explicitly checking the column semantics and concluding that the columns are `(tongue_x, tongue_y, tongue_likelihood)`.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each trial, the agent extracts a slightly padded time window around go cue, converts timestamps to go-cue-relative time, keeps only frames with likelihood greater than 0.1, and averages the y-values falling inside each 50 ms bin. Bins without accepted frames remain `NaN`.

ii.

```python
abs_start = go_time + window_start - bin_width
abs_end = go_time + window_end + bin_width
...
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
high_conf = lk_slice > 0.1
...
if n_in_bin > 0:
    tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```

iii. The notes say the output uses per-session tongue-y percentiles and that the underlying tracking is about 300 Hz. The trajectory also records the agent's explicit check that column 1 is tongue y and column 2 is likelihood.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The agent pools all valid binned tongue-y values from the session, computes the 40th and 60th percentiles, and assigns bin labels `0` below `p40`, `1` between `p40` and `p60` inclusive, and `2` above `p60`. Missing bins are left at the initialization value `1`, so they become the middle category.

ii.

```python
if len(all_tongue_y) > 0:
    all_tongue_y_arr = np.array(all_tongue_y)
    p40 = np.percentile(all_tongue_y_arr, 40)
    p60 = np.percentile(all_tongue_y_arr, 60)
...
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
if np.any(valid_mask):
    tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
    tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
    tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. The notes explicitly say 'per-session percentiles' because that part is dictated by the task instructions. The trajectory later remarks that the middle class is overrepresented because missing bins default to category 1.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue positions are aligned by subtracting each trial's go cue from the video timestamps and then binning the result onto the same 50 ms go-cue-centered time grid as the neural data.

ii.

```python
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
...
bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The notes describe the entire converted dataset as sharing a single go-cue-aligned time base. The trajectory shows the agent verifying that the video stream is about 300 Hz and separate from the ephys timestamps.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent uses a series of pragmatic fallbacks and skips. Entire sessions are skipped if there are no spikes, too few valid trials, no usable control trials, too few good units, low performance, or too few correct trials per side. Missing/parse-failed brain regions become `'unknown'`. If no prior sample-start event is found, tone onset defaults to `-1.85 s`. If tongue bins have no accepted frames they stay `NaN` during aggregation, but missing bins become tongue category `1` when discretized. Unknown outcomes also fall back to class `0`.

ii.

```python
except (json.JSONDecodeError, AttributeError):
    return 'unknown'
...
if len(spike_times_flat) == 0:
    f.close()
    return None
...
if ss_idx >= 0:
    tone_onset_rel = sample_starts_all[ss_idx] - go_time
else:
    tone_onset_rel = -1.85
...
tongue_y_binned = np.full(n_bins, np.nan)
...
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
...
else:
    outcome_val = 0
```

iii. The notes emphasize session skipping as the main protection against bad data, and the trajectory documents the additional ad hoc fallbacks the agent added while debugging partial recordings and missing tongue observations.

## 10-a. What are the most time-consuming steps of the code?

i. The agent identified firing-rate computation as the main runtime cost, with tongue-tracking alignment the second-largest cost. The script even prints timings for data loading, firing-rate computation, input construction, tongue tracking, and output construction on each session.

ii.

```python
t1 = time.time()
print(f"    Data loading: {t1-t0:.1f}s")
...
firing_rates = compute_firing_rates_fast(...)
...
print(f"    Firing rate computation: {t2-t1:.1f}s")
...
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 7 explicitly report firing-rate computation at roughly 3-14 s per session and tongue tracking around 0.4-0.7 s per session after optimization.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The most obvious vectorization targets are the nested per-trial/per-unit loop in `compute_firing_rates_fast`, the per-trial/per-bin loop in `get_tongue_y_for_trials`, and the repeated pure-Python loops over trials for building inputs and outputs. Brain-region extraction also loops unit-by-unit.

ii.

```python
for trial_idx in trial_indices:
    ...
    for i, unit_spikes in enumerate(good_spike_times):
        ...

for trial_idx in trial_indices:
    ...
    for b in range(n_bins):
        ...

for trial_idx in trial_indices:
    ...

for i, trial_idx in enumerate(trial_indices):
    ...
```

iii. The notes discuss optimization work centered on `searchsorted` and pre-extraction but do not eliminate the fundamental nested loops, so the remaining vectorization opportunities are visible directly in the code.

## 10-c. What processing does the code repeat multiple times?

i. The code makes several separate passes over the same selected trial set: one to compute firing rates, one to build the tone/photostim inputs, one inside `get_tongue_y_for_trials`, and one to construct outputs. It also repeatedly computes searchsorted lookups and repeatedly broadcasts constant per-trial labels across 80 bins.

ii.

```python
firing_rates = compute_firing_rates_fast(...)
...
for trial_idx in trial_indices:
    ...
    inputs_list.append(input_trial)
...
tongue_y_trials, all_tongue_y = get_tongue_y_for_trials(...)
...
for i, trial_idx in enumerate(trial_indices):
    ...
    outputs_list.append(output_trial)
```

iii. This follows directly from the structure of `process_session`, and the agent's notes about runtime optimization show that it was aware of repeated work at least in the firing-rate path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns logging-oriented quantities like `correct_rate`, `n_good`, and `n_trials` that are not stored in the final dataset. It also expands trial-constant outputs (`choice`, `outcome`, `early_lick`) into full 80-bin time series, which increases compute and storage even though they contain no within-trial temporal structure. When plotting is enabled it also generates diagnostic figures that are unrelated to downstream decoding.

ii.

```python
return {
    'neural': firing_rates,
    'input': inputs_list,
    'output': outputs_list,
    'brain_regions': good_regions,
    'subject': subject_id,
    'n_good': n_good,
    'n_trials': len(firing_rates),
    'correct_rate': correct_rate,
}
...
output_trial[0, :] = choice
output_trial[1, :] = outcome_val
output_trial[2, :] = early
...
if show_processing and session_idx < 2:
    ...
    plt.savefig(...)
```

iii. The notes emphasize those fields for sanity checking and runtime monitoring, but they are not part of the final serialized output except indirectly through global summary counts. The full-bin broadcasting is visible in the code itself.
