# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all data by globbing every NWB file under `data/sub-*/*.nwb`, sorting the paths, and processing each file with `pynwb.NWBHDF5IO`. Within each file it reads trials, units, and behavioral event/time-series objects directly from the NWB structure.

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

iii. In `CONVERSION_NOTES.md`, the AI states that the dataset consists of 28 subject directories and 174 NWB files and treats those files as the complete dataset.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `nwb.subject.subject_id` for each session, then deduplicated and sorted when assembling the final dataset. Each session stores a `subject_id`, and `subject_idx` is built by indexing into the sorted subject list.

ii.
```python
subject_id = nwb.subject.subject_id
```

```python
subjects = sorted(set(s['subject_id'] for s in all_sessions))
...
subj_idx.append(subjects.index(s['subject_id']))
```

iii. The notes identify the subject folders as `sub-{id}` and explicitly describe the dataset as 28 subjects, so the AI used the NWB subject field as the canonical subject split.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. It loops over the sorted NWB file list, processes each file once, and records the basename as `session_name` in the output metadata.

ii.
```python
nwb_files = get_nwb_files()
...
for i, path in enumerate(nwb_files):
    r = process_session(path, T_START, T_END, BIN_SIZE,
                       show_processing=args.show_processing and i < 2, session_idx=i)
```

```python
basename = os.path.basename(nwb_path)
...
'session_name': basename,
```

iii. In the notes, the AI documents the NWB filename layout and treats each `sub-..._ses-...nwb` file as a session boundary.

## 1-d. How are the data split into trials?

i. Trials are taken from the NWB trials table. Trial-aligned variables such as `trial_instruction`, `outcome`, `early_lick`, `start_time`, `stop_time`, and `go_start_times` are read as parallel arrays indexed by trial. The final output includes only the subset `valid_trials`.

ii.
```python
n_trials = len(nwb.trials)

trial_instruction = nwb.trials['trial_instruction'][:]
outcome = nwb.trials['outcome'][:]
early_lick = nwb.trials['early_lick'][:]
trial_starts = nwb.trials['start_time'][:]
trial_stops = nwb.trials['stop_time'][:]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

```python
valid_trials = np.array([i for i in range(n_trials) if i in recorded_set and regular_mask[i]])
...
for t_idx, trial_idx in enumerate(valid_trials):
    go = go_times[trial_idx]
```

iii. The notes say the NWB trials table holds the behavioral trial structure, and that `obs_intervals` must be used to determine which behavioral trials also have neural recordings.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that both have neural recording coverage and satisfy a "regular trial" mask: no early lick, no auto water, no free water, no ignore/no-response outcome, and no photostimulation. Sessions with fewer than 2 surviving trials are dropped.

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

iii. The notes say the AI chose to "Apply get_regular_trial_mask" from the reference analysis code and also to use `obs_intervals` so only recorded trials are processed. The trajectory shows the AI explicitly removing an earlier session-level performance filter but keeping this trial-level mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `nwb.units['spike_times']` for units whose `classification` is `'good'`, together with the behavioral go-cue timestamps used to define the trial windows.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
```

```python
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
...
go_times = be.time_series['go_start_times'].timestamps[:]
```

iii. In the notes, the AI maps `units.spike_times (good)` to `neural` and states that the data should be binned around go-cue onset.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 50 ms firing rates for each retained trial. For each trial it builds absolute bin edges around the trial's go cue, histograms each neuron's spikes in that window, and divides counts by bin width to get Hz. No smoothing or normalization is applied.

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

iii. The notes describe `np.histogram` as the core spike-binning step and document the intended 50 ms firing-rate representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered only by the NWB `classification` field: a unit is kept if `classification == 'good'`. Sessions with zero such units are skipped entirely.

ii.
```python
classifications = nwb.units['classification'][:]
good_mask = np.array(classifications) == 'good'
good_indices = np.where(good_mask)[0]
...
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
```

iii. The notes explicitly say the AI followed classifier-based QC and equated this with `classification == 'good'`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is aligned to go-cue onset. For each retained trial, the AI uses the absolute go-cue time from `BehavioralEvents/go_start_times` and bins spikes from `go - 2.5 s` to `go + 1.5 s`.

ii.
```python
go_times = be.time_series['go_start_times'].timestamps[:]
...
go_valid = go_times[valid_trials]
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
```

```python
bin_edges = np.linspace(t_start + go, t_end + go, n_bins + 1)
```

iii. The notes say spike times, trial starts, and go cues are on the same absolute session clock, so alignment is done directly relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses 50 ms bins over a 4 s window (`-2.5` to `+1.5` s), producing 80 time bins per trial. No additional temporal rebinning is applied beyond this initial binning.

ii.
```python
T_START = -2.5; T_END = 1.5; BIN_SIZE = 0.05
...
n_bins = int(round((t_end - t_start) / bin_size))
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
```

iii. The notes explicitly call out "Bin 50ms, align go cue, [-2.5, 1.5]s" as the mapping for neural data.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times`, `trials/start_time`, `trials/stop_time`, and the go-cue times. It looks for the first `sample_start_times` event that falls within the trial boundaries.

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

iii. The notes map "time from sample_start" to `time_from_tone_onset`, and the trajectory shows the AI using trial boundaries because `sample_start_times` can contain repeated events when early licks replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each retained trial, the AI computes the time-from-tone value at each bin center as `bin_center - (sample_start - go)`, which is equivalent to go-relative bin time plus the go-to-tone offset. If no sample onset is found for a trial, the input is filled with `NaN`.

ii.
```python
ss = get_sample_start_for_trial(sample_start_times, trial_starts[trial_idx], trial_stops[trial_idx])
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

iii. The notes describe this channel as a continuous time input rather than a binary event marker and say it should reflect seconds from tone onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same 80 go-cue-centered time bins as the neural data. The AI computes the value at each neural bin center, trial by trial.

ii.
```python
bin_centers = np.linspace(t_start + bin_size/2, t_end - bin_size/2, n_bins)
...
tft = bin_centers - (ss - go)
```

iii. The notes state that all decoder variables are aligned to go cue onset, and the code uses the same `bin_centers` array for both input and neural channels.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the trials-table fields `photostim_power`, `photostim_onset`, and `photostim_duration`, together with `start_time` and the go-cue time. `photostim_power` is used to determine whether a trial has stimulation at all.

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

iii. The notes explicitly map the reference stimulation field onto NWB photostim metadata and say photostim should become a binary per-timepoint input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI constructs a binary time series per trial: bins are `1` while the bin center lies within the stimulation interval and `0` otherwise. Because the earlier trial filter removes all photostim trials, the retained dataset ends up with all-zero photostim inputs.

ii.
```python
ps_on = np.zeros(n_bins, dtype=np.float32)
if has_photostim[trial_idx]:
    o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
    d = float(photostim_dur_raw[trial_idx])
    ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes say the photostim input is "Binary per timepoint," and later acknowledge that it is always zero because photostim trials were filtered out.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI converts photostim onset from trial-start-relative time into go-cue-relative time, then compares that interval to the same bin centers used for the neural data.

ii.
```python
o_rel = float(photostim_onset_raw[trial_idx]) - (go - trial_starts[trial_idx])
...
ps_on = ((bin_centers >= o_rel) & (bin_centers < o_rel + d)).astype(np.float32)
```

iii. The notes say photostim onset needs to be converted to the go-cue-aligned frame because the decoder window is aligned to go cue.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives `choice` only from `trials['trial_instruction']`. It does not use `outcome` to distinguish correct versus incorrect responses or to create a "no lick" class.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
...
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
```

iii. The notes map `trial_instruction` directly to choice (`left=0, right=1`) and do not describe reconstructing actual lick direction from trial outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0` for left and `1` for right, repeats that value across all time bins in the trial, and exposes only two choice labels in `output_values`.

ii.
```python
choice = 1 if trial_instruction[trial_idx] == 'right' else 0
...
out = np.zeros((4, n_bins), dtype=np.int64)
out[0, :] = choice
```

```python
'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
```

iii. The notes describe this as a simple left/right mapping and do not mention a third no-lick category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is read directly from `trials['outcome']`.

ii.
```python
outcome = nwb.trials['outcome'][:]
...
out_val = out_map.get(outcome[trial_idx], 0)
```

iii. The notes map NWB `outcome` directly onto the decoder output categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, and repeats the resulting per-trial category across all time bins.

ii.
```python
out_map = {'ignore': 0, 'miss': 1, 'hit': 2}
out_val = out_map.get(outcome[trial_idx], 0)
...
out[1, :] = out_val
```

iii. The notes say this output should follow the task specification exactly.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `trials['early_lick']`.

ii.
```python
early_lick = nwb.trials['early_lick'][:]
...
early_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The notes map the NWB early-lick annotation directly to the decoder output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI encodes `early` as `1` and `no early` as `0`, then repeats that per-trial value across all time bins.

ii.
```python
early_val = 1 if early_lick[trial_idx] == 'early' else 0
...
out[2, :] = early_val
```

iii. The notes say this output should be categorical and per-trial; later notes also acknowledge it becomes trivial because early-lick trials were filtered out.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI uses the `BehavioralTimeSeries/Camera0_side_TongueTracking` series. It takes timestamps from `tt_obj.timestamps`, tongue y-position from column 1 of `tt_obj.data`, and tracking confidence from column 2.

ii.
```python
tt_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_ts = tt_obj.timestamps[:]
td = tt_obj.data[:]
tongue_y_arr = td[:, 1]
tongue_lk_arr = td[:, 2]
```

iii. The notes explicitly describe the side-camera tongue tracking stream as the source for this output.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI first gathers all high-confidence tongue y samples from the retained trial windows only, using a confidence threshold of `0.9`. It computes session-level 40th and 60th percentiles from the concatenated raw y samples. Then, for each trial and each bin, it averages the high-confidence tongue y values in that 50 ms window and assigns a low/mid/high class from those two thresholds.

ii.
```python
if tongue_ts is not None:
    all_ty = []
    for ti in valid_trials:
        go = go_times[ti]
        i_lo = np.searchsorted(tongue_ts, go + t_start)
        i_hi = np.searchsorted(tongue_ts, go + t_end)
        if i_hi > i_lo:
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
lk = tongue_lk_arr[il:ih]
gd = lk > 0.9
if np.any(gd):
    my = np.mean(tongue_y_arr[il:ih][gd])
    ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The notes justify this as using "DLC likelihood > 0.9 threshold" and later report the session percentiles as part of the processing summary.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI uses only three categories: `0` for below the 40th percentile, `1` for between the 40th and 60th percentiles, and `2` for above the 60th percentile. If no high-confidence tongue samples are found for a bin, it leaves that bin at the default middle class `1`; it does not create a fourth "not visible" class.

ii.
```python
ty = np.ones(n_bins, dtype=np.int64)
...
if np.any(gd):
    my = np.mean(tongue_y_arr[il:ih][gd])
    ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

```python
'output_values': [['left','right'], ['ignore','miss','hit'], ['no','yes'], ['low','mid','high']],
```

iii. The notes explicitly say the AI chose to "default to class 1 (mid) when no high-confidence detection" rather than representing invisibility as its own category.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each retained trial and each 50 ms bin, the AI centers a window on `go + bin_center`, finds all tongue frames in that half-bin interval, averages the high-confidence tongue y values, and writes the result into the corresponding decoder bin. This places tongue bins on the same go-cue-centered grid as the neural data.

ii.
```python
half_bin = bin_size / 2.0
...
for b in range(n_bins):
    bc = go + bin_centers[b]
    il = np.searchsorted(tongue_ts, bc - half_bin)
    ih = np.searchsorted(tongue_ts, bc + half_bin)
```

iii. The notes describe all streams as being aligned to go cue onset and processed in 50 ms bins.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several data issues by exclusion or fallback values. Sessions with no good neurons are skipped. Trials outside recorded `obs_intervals` or failing the regular-trial mask are removed. If no sample onset is found for a retained trial, `time_from_tone_onset` is filled with `NaN`. If tongue tracking is present but a bin has no high-confidence frames, the tongue output remains at the default middle class `1`.

ii.
```python
if n_good == 0:
    print(f"  SKIPPING: No good neurons"); return None
...
if len(valid_trials) < 2:
    print(f"  SKIPPING: <2 valid trials"); return None
```

```python
if ss is not None:
    tft = bin_centers - (ss - go)
else:
    tft = np.full(n_bins, np.nan, dtype=np.float32)
```

```python
ty = np.ones(n_bins, dtype=np.int64)
...
if np.any(gd):
    my = np.mean(tongue_y_arr[il:ih][gd])
    ty[b] = 0 if my < tongue_y_p40 else (2 if my > tongue_y_p60 else 1)
```

iii. The notes mention fixing `obs_intervals`, skipping sessions with no good neurons, and intentionally defaulting missing tongue detections to the middle class.

## 10-a. What are the most time-consuming steps of the code?

i. The most expensive steps are loading spike times from NWB, trial-by-trial spike histogramming for every neuron, and the per-bin tongue-processing loops. Optional plotting can add extra time when enabled.

ii.
```python
t_load = time.time()
spike_times_list = [nwb.units['spike_times'][idx] for idx in good_indices]
print(f"  Loaded spike times ({time.time()-t_load:.1f}s)")
```

```python
neural_trials = bin_spikes_all_trials(spike_times_list, go_valid, t_start, t_end, bin_size)
...
for b in range(n_bins):
    bc = go + bin_centers[b]
```

iii. In the notes, the AI says the implementation spends about 10 s per session and explicitly calls out spike loading, histogramming, and tongue processing as the main costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The code contains several loops that could be vectorized further: the nested trial-by-neuron loop in `bin_spikes_all_trials`, the per-trial/per-bin tongue loop, the `obs_intervals` matching loop, and the repeated `list.index` lookups during assembly.

ii.
```python
for go in go_times_arr:
    ...
    for i, st in enumerate(spike_times_list):
        ...
        counts, _ = np.histogram(st_w, bins=bin_edges)
```

```python
for t_idx, trial_idx in enumerate(valid_trials):
    ...
    for b in range(n_bins):
        bc = go + bin_centers[b]
```

```python
for i in range(n_obs):
    diffs = np.abs(trial_starts - obs[i, 0])
    best = np.argmin(diffs)
```

iii. The notes describe the implementation as "efficient," but the final code still relies heavily on Python loops over trials, neurons, and bins.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several computations that could be shared: it recomputes trial-specific spike masks and histograms for every neuron and every trial; it performs `searchsorted` for every tongue bin of every trial; and it repeatedly searches subject and region lists with `list.index` while assembling the final structure.

ii.
```python
mask = (st >= bin_edges[0]) & (st < bin_edges[-1])
st_w = st[mask]
if len(st_w) > 0:
    counts, _ = np.histogram(st_w, bins=bin_edges)
```

```python
il = np.searchsorted(tongue_ts, bc - half_bin)
ih = np.searchsorted(tongue_ts, bc + half_bin)
```

```python
subj_idx.append(subjects.index(s['subject_id']))
br_idx.append(np.array([regions.index(r) for r in s['neuron_regions']]))
```

iii. These repeated computations are not highlighted in the notes, but they follow directly from the AI's implementation choices.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some session-level diagnostics that are not used to build decoder arrays: control-trial performance, `correct_left`, `correct_right`, and optional plotting. It also computes metadata such as `n_total` and `performance` that are descriptive but not used downstream by the decoder itself.

ii.
```python
control = mask_no_early & mask_no_photostim
ctrl_out = outcome[control]
n_hit = np.sum(ctrl_out == 'hit')
n_miss = np.sum(ctrl_out == 'miss')
perf = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0.0
...
correct_left = np.sum((trial_instruction[ctrl_responded] == 'left') & (outcome[ctrl_responded] == 'hit'))
correct_right = np.sum((trial_instruction[ctrl_responded] == 'right') & (outcome[ctrl_responded] == 'hit'))
```

```python
if show_processing:
    try:
        plot_processing(...)
```

iii. The notes present these as sanity checks and summaries rather than required decoder features.
