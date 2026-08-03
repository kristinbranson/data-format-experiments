# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the dataset by globbing all NWB files under `data/sub-*/*.nwb`, sorting them, and processing each file with `pynwb.NWBHDF5IO`. Within each file it reads trials from `nwb.trials`, events from `nwb.acquisition['BehavioralEvents']`, time series from `nwb.acquisition['BehavioralTimeSeries']`, and units from `nwb.units`.

ii.
```python
def get_nwb_files(data_dir='data'):
    return sorted(glob.glob(os.path.join(data_dir, 'sub-*', '*.nwb')))
```

```python
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
    subject_id = nwb.subject.subject_id
    n_trials = len(nwb.trials)
```

iii. The notes say the reference `.mat` pipeline must be mapped onto the published NWB files, and they identify one NWB file per session under `data/sub-{id}/...`. The trajectory shows the AI explicitly choosing the NWB layout as the complete dataset inventory.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `nwb.subject.subject_id` as the subject identity for each session, then builds `subjects` as the sorted unique set of those ids and `subject_idx` as an index per session.

ii.
```python
subject_id = nwb.subject.subject_id
```

```python
subjects = sorted(set(s['subject_id'] for s in all_sessions))
...
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The notes state there are 28 subject directories and use `subject_id` as the canonical subject field from NWB. No alternative grouping rule is described.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. `process_session` is called once per file, and each returned session becomes one element of the top-level `neural`, `input`, and `output` lists.

ii.
```python
nwb_files = get_nwb_files()
...
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE,
                       show_processing=args.show_processing and i < 2, session_idx=i)
```

```python
neural=[]; inputs=[]; outputs=[]
for s in all_sessions:
    neural.append(s['neural']); inputs.append(s['input']); outputs.append(s['output'])
```

iii. The notes repeatedly describe the dataset as “174 NWB files” and “one file per session,” so the AI uses the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table. The AI reads trial-level columns from `nwb.trials`, uses `go_start_times` as the per-trial alignment event array, and keeps trial indices as the basic trial identity throughout processing.

ii.
```python
n_trials = len(nwb.trials)

trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
```

```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
for t_idx, trial_idx in enumerate(valid_trials):
    go = go_times[trial_idx]
```

iii. The notes map the reference trial variables onto NWB trial-table columns and event arrays. There is no attempt to re-derive trials from spikes or video.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are both in `obs_intervals` and pass a “regular trial” mask matching the reference code’s `get_regular_trial_mask`: no early lick, no auto water, no free water, no `ignore`, and no photostimulation. Sessions with fewer than two remaining trials are dropped.

ii.
```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
```

```python
mask_no_auto = (auto_water == 0)
mask_no_free = (free_water == 0)
mask_no_ignore = (outcome != 'ignore')
regular_mask = mask_no_early & mask_no_auto & mask_no_free & mask_no_ignore & mask_no_photostim

recorded_set = set(recorded_trials)
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
...
if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
```

iii. The notes explicitly say “Trial filtering: Apply get_regular_trial_mask (no early lick, no auto water, no free water, no ignore, no photostim)” and treat this as a key decision. Later notes also say early-lick and photostim outputs become trivial because those trials are filtered out.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives neural data from `nwb.units['spike_times']` for units whose `classification` is `'good'`, with `go_start_times` providing the alignment times.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
...
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

iii. The notes identify classifier-based QC and `units.spike_times` as the neural source, with go-cue alignment.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 50 ms bins from -2.5 s to +1.5 s around the go cue and converts counts to firing rates in Hz. It does this by histogramming each neuron's spikes inside each trial window.

ii.
```python
def bin_spikes_all_trials(spike_times_list, go_times_arr, t_start, t_end, bin_size):
    n_bins = int(round((t_end - t_start) / bin_size))
    ...
    for go in go_times_arr:
        bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
        fr = np.zeros((n_neurons, n_bins), dtype=np.float32)
        for i, st in enumerate(spike_times_list):
            ...
            counts, _ = np.histogram(st_w, bins=bin_edges)
            fr[i] = counts.astype(np.float32) / bin_size
```

iii. The notes describe 50 ms go-cue-aligned firing-rate bins as the intended output, and call out `np.histogram` as the implementation choice.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are kept. If a session has no such units, the session is dropped.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. The notes say the NWB `classification` field is the classifier-based QC result and should be used directly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned directly to go-cue onset. For each retained trial the AI constructs absolute bin edges by adding the relative window `[-2.5, 1.5]` to that trial’s go-cue time.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

```python
for go in go_times_arr:
    bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The notes say spike times and events are on the same absolute session clock, so alignment is just windowing around the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 50 ms bins. The AI does not apply any extra rebinning beyond this histogramming step.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
```

```python
n_bins = int(round((t_end - t_start) / bin_size))
bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The notes explicitly say 50 ms bins were chosen to match the decoder task specification.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `sample_start_times`, `trial_start`, `trial_stop`, and `go_times`. It finds the first `sample_start_time` inside each trial and combines it with go-cue-relative bin centers.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
```

```python
def get_sample_start_for_trial(sample_start_times, trial_start, trial_stop):
    mask = (sample_start_times >= trial_start) & (sample_start_times <= trial_stop)
    return sample_start_times[mask][0] if np.any(mask) else None
```

iii. In the trajectory and notes, the AI says it is computing time from the “first sample_start_time for each trial” or “first tone onset in each trial,” motivated by repeated sample events when trials are replayed.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI computes bin centers relative to the go cue, finds the trial’s tone onset as the first sample start inside the trial, and then converts those go-cue-relative centers into time-since-tone values. If no sample start is found, it fills the entire input with `NaN`.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The notes say this field should be “continuous seconds” from sample start. The trajectory explicitly says “time from first tone onset in each trial.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated on the same 80 go-cue-relative bin centers used for the neural trial windows, so each timepoint lines up with a neural bin.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

```python
input_trials.append(np.stack([tft.astype(np.float32), ps_on]))
```

iii. The notes say temporal alignment is to the go cue for all streams, and the code uses the shared `bin_centers` array for both neural and input construction.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI uses `photostim_power`, `photostim_onset`, `photostim_duration`, `trial_start`, and `go_times`. `photostim_power` is used to decide whether stimulation is present; onset and duration are then converted into a go-cue-relative interval.

ii.
```python
photostim_power_raw = nwb.trials['photostim_power'][:]
photostim_onset_raw = nwb.trials['photostim_onset'][:]
photostim_dur_raw = nwb.trials['photostim_duration'][:]
has_photostim = np.array([p != 'N/A' and float(p) > 0 for p in photostim_power_raw])
```

```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
d = float(photostim_dur_raw[trial_idx])
```

iii. The notes say photostim should be a binary time-varying input and that onset must be converted from trial-start-relative to go-cue-relative time.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI creates a binary time series that is 1 when a bin center falls within the stimulation interval and 0 otherwise. In the final dataset this is effectively always zero because photostim trials are removed by trial filtering.

ii.
```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes explicitly say “Photostim input: Always 0 because photostim trials are filtered out. This is correct per reference code.” The binary time-series formulation itself is also listed in the mapping plan.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts photostim onset from trial-start-relative time into go-cue-relative time and compares that interval to the same `bin_centers` used for the neural data.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
...
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes say the onset conversion is an edge case that was checked explicitly.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice directly from `trial_instruction` alone: `'left'` becomes 0 and `'right'` becomes 1. It does not use `outcome` to flip miss trials or add a no-lick class.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The mapping table in the notes says `trial_instruction -> choice (left=0, right=1)`, and a later manual check says a miss trial with `trial_instruction='left'` correctly produced `choice=0`, which confirms the AI intentionally treated choice as instructed side rather than actual lick side.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI maps left/right to 0/1 and repeats that per-trial value across all 80 bins. It defines only two output values for this variable.

ii.
```python
out = np.zeros((4, n_bins), dtype=np.int64)
out[0, :] = choice
```

```python
'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
```

iii. The notes justify this as following the decoder spec’s `left = 0, right = 1`, but do not discuss miss trials or no-lick trials beyond filtering out `ignore`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the NWB trials-table `outcome` column.

ii.
```python
outcome = nwb.trials['outcome'][:]
```

iii. The notes map NWB `outcome` directly onto the decoder output categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `'ignore'`, `'miss'`, and `'hit'` to 0, 1, and 2 respectively, then repeats the per-trial value across all bins.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
...
out[1, :] = out_val
```

iii. The notes explicitly state this coding.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the NWB trials-table `early_lick` column.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
```

iii. The notes map `'no early'/'early'` directly to the output field.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `'early'` to 1 and `'no early'` to 0, then repeats the per-trial value across all bins.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
...
out[2, :] = early_val
```

iii. The notes state `early_lick -> no=0, yes=1`. They also note that this output becomes trivial because early-lick trials are excluded upstream.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from the `Camera0_side_TongueTracking` behavioral time series: timestamps, y coordinate (`data[:, 1]`), and DeepLabCut likelihood (`data[:, 2]`).

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
...
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr = td[:, 1]
tongue_lk_arr = td[:, 2]
```

iii. The trajectory records the AI’s inspection of this stream and notes “(x, y, likelihood) columns” at ~300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first pools all high-confidence tongue frames (`likelihood > 0.9`) from retained trials in the go-cue window and computes the 40th and 60th percentiles of those raw y values. Then, for each trial and bin, it averages high-confidence y values within a centered 50 ms window and thresholds that mean against the two session-wide percentiles.

ii.
```python
if tongue_ts is not None:
    all_ty = []
    for ti in valid_trials:
        ...
        lk = tongue_lk_arr[i_lo:i_hi]
        good = lk > 0.9
        if np.any(good):
            all_ty.append(tongue_y_arr[i_lo:i_hi][good])
    if all_ty:
        concat = np.concatenate(all_ty)
        tongue_y_p40 = np.percentile(concat, 40)
        tongue_y_p60 = np.percentile(concat, 60)
```

```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
    if ih > il:
        lk = tongue_lk_arr[il:ih]
        gd = lk > 0.9
        if np.any(gd):
            my = np.mean(tongue_y_arr[il:ih][gd])
            ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The notes say the AI chose a stricter DLC threshold (`> 0.9`) and a three-level session-wise discretization. The trajectory indicates this was based on exploratory checks of the tongue tracking quality.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses three classes only: 0 for below the 40th percentile, 1 for between the 40th and 60th percentiles, and 2 for above the 60th percentile. Bins without any high-confidence tongue data default to class 1 (“mid”).

ii.
```python
ty = np.ones(n_bins, dtype=np.int64)
```

```python
if np.any(gd):
    my = np.mean(tongue_y_arr[il:ih][gd])
    ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

```python
'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
```

iii. The notes explicitly say “default to class 1 (mid) when no high-confidence DLC detection” and describe only three tongue categories.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to the same go-cue-relative 50 ms bins as neural data. For each bin center it looks up camera frames within `center ± 25 ms` in absolute time.

ii.
```python
half_bin = bin_size / 2.0
...
bc = go + bin_centers[b]
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The notes say all streams share the go-cue alignment, and the trajectory explicitly identifies camera timestamps and go times as being on the same absolute clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by exclusion or fallback defaults. Sessions with no `classification == 'good'` units are dropped. Trials without neural recording coverage are removed using `obs_intervals`. Missing tone onset within a trial yields an all-`NaN` time-from-tone input. Missing high-confidence tongue data within a bin yields the default tongue class 1. Sessions with fewer than two valid trials are dropped.

ii.
```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

```python
recorded_trials = get_recorded_trial_indices(nwb, good_indices)
...
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
```

```python
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

```python
ty = np.ones(n_bins, dtype=np.int64)
```

iii. The notes justify the `obs_intervals` handling as necessary because some sessions record only a subset of trials. They also explicitly state the tongue fallback to mid class.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by NWB I/O, loading all good-unit spike trains and tongue tracking arrays, nested spike-binning loops over trials and neurons, and per-bin tongue processing. The notes estimate roughly 10 to 12 seconds per session.

ii.
```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
...
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

```python
for go in go_times_arr:
    ...
    for i, st in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(st_w, bins=bin_edges)
```

```python
for b in range(n_bins):
    ...
    if np.any(gd):
        my = np.mean(tongue_y_arr[il:ih][gd])
```

iii. The notes call out `np.histogram`, `np.searchsorted`, and `obs_intervals` handling as the main implementation details and give runtime estimates around “~10s/session” and “~12s/session”.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several obvious Python loops unvectorized: the outer per-trial and inner per-neuron loops in `bin_spikes_all_trials`, the list comprehension over units to load spikes, the loop collecting tongue frames across trials, and the per-bin tongue loop within each trial.

ii.
```python
for go in go_times_arr:
    ...
    for i, st in enumerate(spike_times_list):
```

```python
for ti in valid_trials:
    ...
```

```python
for b in range(n_bins):
    ...
```

iii. The notes frame the implementation as “efficient” because it uses `np.histogram` and `np.searchsorted`, but they do not claim these remaining loops were vectorized.

## 10-c. What processing does the code repeat multiple times?

i. The AI repeats several computations. It scans tongue frames once to estimate session percentiles and again to compute per-bin outputs. It also recomputes per-trial spike histograms separately for every trial instead of reusing a shared edge structure. In addition, it computes behavioral performance and left/right correct counts that are only reported or stored in metadata.

ii.
```python
if tongue_ts is not None:
    all_ty = []
    for ti in valid_trials:
        ...
```

```python
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

```python
ctrl_out = outcome[control]
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
...
correct_left = np.sum(...)
correct_right = np.sum(...)
```

iii. The notes present performance calculation and tongue processing as sanity checks or metadata-oriented work, but the code computes them during every session conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes several quantities not needed for downstream decoder inputs and outputs: behavioral performance, `correct_left`, `correct_right`, plotting support via `show_processing`, and photostim timing logic even though the retained trials exclude photostimulation. It also stores extra session metadata not required by the target format.

ii.
```python
ctrl_out = outcome[control]
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
...
print(f"  Performance: {perf:.1%} ({n_hit}/{n_hit+n_miss}), Correct L={correct_left}, R={correct_right}")
```

```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    ...
```

```python
if show_processing:
    try:
        plot_processing(...)
```

iii. The notes themselves say the photostim input is always zero after filtering and describe performance calculations mainly as session statistics and validation checks rather than decoder features.
