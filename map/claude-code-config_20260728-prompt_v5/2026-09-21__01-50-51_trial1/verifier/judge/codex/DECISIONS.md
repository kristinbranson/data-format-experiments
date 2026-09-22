# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all sessions by globbing every NWB file under `/app/data/sub-*/*.nwb`, sorting the paths, and processing each file once with `pynwb.NWBHDF5IO`. Within each file it reads the full NWB object and then accesses `nwb.subject`, `nwb.units`, `nwb.trials`, `nwb.acquisition['BehavioralEvents']`, and `nwb.acquisition['BehavioralTimeSeries']`.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, bin_edges, n_bins,
                             show_processing=args.show_processing, session_idx=i)
```

```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The notes say the dataset is organized as one NWB file per session and explicitly list the file layout under `/app/data/`, so the AI treated a sorted file glob as the complete dataset.

## 1-b. How are the data split into subjects (mice)?

i. The AI uses `nwb.subject.subject_id` from each session file as the subject identifier, then builds `subjects` as the sorted unique set of those IDs and `subject_idx` as a per-session lookup into that list.

ii.
```python
subject_id = nwb.subject.subject_id
...
subject_ids = sorted(set(r['subject_id'] for r in all_results))
subjects = [str(sid) for sid in subject_ids]
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
...
subject_idx.append(subject_id_to_idx[r['subject_id']])
```

iii. The notes say to map unique NWB subject IDs into `subjects/subject_idx`, and the dataset exploration step identifies subject IDs as the native mouse identifiers.

## 1-c. How are the data split into sessions?

i. The AI treats each NWB file as one session. Session order follows the sorted file list, and each surviving file contributes one entry to `neural`, `input`, `output`, `subject_idx`, and `brain_region_idx`.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
...
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, bin_edges, n_bins,
                             show_processing=args.show_processing, session_idx=i)
    if result is not None:
        all_results.append(result)
```

```python
for r in all_results:
    neural.append(r['neural'])
    inputs.append(r['input'])
    outputs.append(r['output'])
```

iii. The notes describe each NWB file as “one session with behavior + electrophysiology,” so the AI used the file boundary as the session boundary.

## 1-d. How are the data split into trials?

i. The AI uses the NWB trials table and parallel event arrays. It reads per-trial columns such as `auto_water`, `free_water`, `trial_instruction`, `early_lick`, and `outcome`, and it indexes those arrays by `valid_trial_indices`. Go-cue times come from the `go_start_times` event stream and are assumed to be aligned by row index to the trials table.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
trial_instruction = nwb.trials['trial_instruction'][:]
early_lick_arr = nwb.trials['early_lick'][:]
outcome_arr = nwb.trials['outcome'][:]
...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
valid_trial_indices = np.where(valid_trial_mask)[0]
go_cue_times = go_cue_times_all[valid_trial_indices]
```

iii. The notes describe the trials table as the native per-trial structure and list `BehavioralEvents/go_start_times` as the trial-alignment signal, so the AI used those two sources together.

## 1-e. How are trials filtered based on quality controls?

i. The AI’s notes say the intended trial filter was to exclude only `auto_water` and `free_water`, while keeping early-lick, ignore, and photostim trials because they are decoder targets or inputs. The code actually excludes `auto_water`, excludes `free_water`, and also drops any trial whose go-cue-aligned `[-2.5, 1.5] s` window falls outside the first good unit’s overall `obs_intervals` span. A session is dropped if fewer than 2 trials remain.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
...
obs_intervals = nwb.units['obs_intervals'][good_indices[0]]
obs_start = obs_intervals[0, 0]
obs_end = obs_intervals[-1, 1]
within_obs = ((go_cue_times_all + begin_time) >= obs_start) & \
             ((go_cue_times_all + end_time) <= obs_end)
valid_trial_mask = valid_trial_mask & within_obs
...
if n_valid_trials < 2:
    io.close()
    return None
```

iii. The notes justify keeping early-lick, ignore, and photostim trials on task grounds, and explicitly say “Only exclude auto_water and free_water.” They also say “All 174 sessions included,” although later notes acknowledge one skipped session.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `nwb.units['spike_times']` for quality-filtered units, with `go_start_times` providing the alignment times used to place the spike-count bins.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
...
all_spike_times_obj = nwb.units['spike_times']
...
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
```

iii. The notes map `units.spike_times (good only)` to the target `neural` field and describe the spikes as absolute session time that must be aligned to go cue.

## 2-b. How is the `neural` data processed?

i. For each good unit and each kept trial, the AI extracts spikes in the `[-2.5, 1.5] s` window around the go cue, histograms them into non-overlapping 50 ms bins, and divides by bin width to convert counts to firing rates.

ii.
```python
def bin_spikes_all_trials(spike_times_unit, go_cue_times, bin_edges):
    n_trials = len(go_cue_times)
    n_bins = len(bin_edges) - 1
    bin_width = bin_edges[1] - bin_edges[0]
    fr = np.zeros((n_trials, n_bins), dtype=np.float32)
    ...
    for t in range(n_trials):
        gc = go_cue_times[t]
        mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
        if np.any(mask):
            aligned = spike_times_unit[mask] - gc
            counts, _ = np.histogram(aligned, bins=bin_edges)
            fr[t] = counts / bin_width
```

iii. The notes say to bin good-unit spike times into 50 ms bins aligned to go cue and emphasize that the decoder task requires 50 ms non-overlapping bins rather than the reference paper’s finer video-analysis binning.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == 'good'`. It does not apply an additional firing-rate threshold. If a session has zero such units, it is dropped.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
n_good = len(good_indices)

if n_good == 0:
    io.close()
    return None
```

iii. The notes explicitly justify using classifier QC (`classification == 'good'`) and explicitly reject the method paper’s 2 Hz threshold as analysis-specific rather than a general data-quality rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns every trial to `BehavioralEvents/go_start_times`. For each trial it builds bin edges relative to the go cue and subtracts the go-cue time from spikes before histogramming.

ii.
```python
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
...
gc = go_cue_times[t]
mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
if np.any(mask):
    aligned = spike_times_unit[mask] - gc
    counts, _ = np.histogram(aligned, bins=bin_edges)
```

iii. The notes say “Spike times aligned to go cue (time 0)” and “Verified temporal alignment (go cue at t=0).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping bins of width 50 ms spanning `[-2.5, 1.5] s` around go cue. No additional temporal smoothing or rebinning is applied.

ii.
```python
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_WIDTH = 0.05
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
bin_edges = np.linspace(BEGIN_TIME, END_TIME, n_bins + 1)
```

iii. The notes justify this as a decoder-task requirement and explicitly contrast it with the different binning used in the source code base.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI derives this input from `BehavioralEvents/sample_start_times` and `BehavioralEvents/go_start_times`. For each kept trial it picks the last sample-start timestamp at or before the go cue and treats that as the tone onset.

ii.
```python
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
...
candidates = sample_start_times[sample_start_times <= gc + 0.01]
if len(candidates) > 0:
    tone_onsets_rel[t] = candidates[-1] - gc
```

iii. The notes say “sample_start relative to go cue” and describe the output as a time-varying ramp derived from sample onset and go cue timing.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The AI first computes the tone onset relative to the go cue for each trial. It then subtracts that offset from the go-cue-relative bin centers so each bin stores elapsed time since tone onset. If no candidate sample onset exists, it falls back to `-1.85 s`.

ii.
```python
tone_onsets_rel = np.zeros(n_valid_trials, dtype=np.float64)
for t, trial_idx in enumerate(valid_trial_indices):
    gc = go_cue_times_all[trial_idx]
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
    if len(candidates) > 0:
        tone_onsets_rel[t] = candidates[-1] - gc
    else:
        tone_onsets_rel[t] = -1.85  # fallback
...
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. The notes justify the transformation as “t - (sample_start - go_cue) for each time bin t,” and separately note that the sample-to-go-cue interval is consistently about 1.85 s in the dataset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The AI uses the same 80 go-cue-relative bin centers used for neural firing rates, so the time-from-tone value is defined at exactly the same time points as the neural bins.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
...
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
...
neural_trials.append(fr_all[:, t, :])  # (n_good, n_bins)
```

iii. The notes explicitly describe `time_from_tone_onset` as a time-varying quantity on the same 50 ms trial grid as the neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, not from per-trial `photostim_onset` and `photostim_duration` in the trials table.

ii.
```python
has_photostim = 'photostim_start_times' in be.time_series
if has_photostim:
    ps_start_times = be.time_series['photostim_start_times'].timestamps[:]
    ps_stop_times = be.time_series['photostim_stop_times'].timestamps[:]
else:
    ps_start_times = np.array([])
    ps_stop_times = np.array([])
```

iii. The notes say “Photostim timing: Photostim times are in absolute session time. Convert to go-cue-relative for each trial,” and the dataset-exploration notes list `photostim_start/stop_times` among the available event streams.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts each absolute photostim interval into a go-cue-relative interval for each trial, then sets a binary time series to 1 in bins whose centers fall inside any overlapping stim interval. Multiple overlapping intervals are clipped back to 1.

ii.
```python
photostim_all = np.zeros((n_valid_trials, n_bins), dtype=np.float32)
if len(ps_start_times) > 0:
    for t in range(n_valid_trials):
        gc = go_cue_times[t]
        for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
            ps_s_rel = ps_s - gc
            ps_e_rel = ps_e - gc
            if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
                photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
    photostim_all = np.clip(photostim_all, 0, 1)
```

iii. The notes justify photostimulation as a time-varying binary input and state that the event times should be converted into the go-cue-relative frame.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The AI aligns photostim to the neural data by subtracting the trial’s go-cue time from each photostim start and stop timestamp, then comparing those relative times to the same bin centers used for neural and other inputs.

ii.
```python
gc = go_cue_times[t]
for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
    ps_s_rel = ps_s - gc
    ps_e_rel = ps_e - gc
    if ps_s_rel < bin_edges[-1] and ps_e_rel > bin_edges[0]:
        photostim_all[t] += ((bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)).astype(np.float32)
```

iii. The notes explicitly say to convert absolute photostim times into go-cue-relative time for each trial.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from two trial-table variables: `trial_instruction` and `outcome`. There is no direct choice column.

ii.
```python
trial_instruction = nwb.trials['trial_instruction'][:]
outcome_arr = nwb.trials['outcome'][:]
...
instr = trial_instruction[trial_idx]
out = outcome_arr[trial_idx]
```

iii. The notes explicitly state “trial_instruction + outcome -> choice” and define the coding as left, right, or no lick.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A hit is coded as the instructed side, a miss as the opposite side, and anything else as no lick (`2`). The per-trial value is then repeated across all 80 bins.

ii.
```python
if out == 'hit':
    choices[t] = 0 if instr == 'left' else 1
elif out == 'miss':
    choices[t] = 1 if instr == 'left' else 0
else:
    choices[t] = 2  # no lick
...
np.full(n_bins, int(choices[t]), dtype=np.int64)
```

iii. The notes justify this directly: “hit → lick matches instruction; miss → lick opposite to instruction; ignore → no lick.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the NWB trials-table field `outcome`.

ii.
```python
outcome_arr = nwb.trials['outcome'][:]
```

iii. The notes list `outcome` as a direct source variable with categories `ignore`, `miss`, and `hit`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore -> 0`, `miss -> 1`, and `hit -> 2`, then repeats the resulting per-trial label across all 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
outcomes[t] = outcome_map.get(out, 0)
...
np.full(n_bins, int(outcomes[t]), dtype=np.int64)
```

iii. The notes specify the same categorical mapping under the planned variable mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the NWB trials-table field `early_lick`.

ii.
```python
early_lick_arr = nwb.trials['early_lick'][:]
```

iii. The notes map `early_lick` directly onto the decoder output with categories `no` and `yes`.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `early -> 1` and everything else to `0`, then repeats that per-trial label across the 80 bins.

ii.
```python
early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0
...
np.full(n_bins, int(early_vals[t]), dtype=np.int64)
```

iii. The notes specify the intended coding as `no=0, yes=1`.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. The AI derives tongue position from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of `data` is the tongue y-value and column 2 is the tracking likelihood; `timestamps` provide frame times.

ii.
```python
bts = nwb.acquisition['BehavioralTimeSeries']
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
```

iii. The dataset-exploration notes identify this exact series as the tongue-tracking source and describe it as `(x, y, likelihood at ~300 Hz/3.4 ms)`.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI uses a likelihood threshold of `0.9`, computes session-wide 40th and 60th percentiles from all raw y-values in frames whose likelihood exceeds that threshold, then bins each trial’s frames into 50 ms bins. A bin is assigned a visible tongue class only if the mean likelihood of frames in that bin is at least `0.9`; then the bin’s mean y is compared to the session percentiles.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
...
visible_mask_all = tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH
if np.sum(visible_mask_all) > 0:
    visible_y = tongue_data[visible_mask_all, 1]
    tongue_y_p40 = np.percentile(visible_y, 40)
    tongue_y_p60 = np.percentile(visible_y, 60)
```

```python
avg_likelihood = np.mean(frames[:, 2])
if avg_likelihood >= likelihood_thresh:
    avg_y = np.mean(frames[:, 1])
    if avg_y < tongue_y_p40:
        result[t, b] = 0
    elif avg_y <= tongue_y_p60:
        result[t, b] = 1
    else:
        result[t, b] = 2
```

iii. The notes explicitly justify `0.9` as a “DeepLabCut convention,” say that tongue frames below that threshold should become “not visible,” and say the 40th and 60th percentiles should be computed over “all visible tongue y-positions in the session.”

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI creates four categories: `0` if the bin-mean y is below the session 40th percentile, `1` if between the 40th and 60th percentiles inclusive, `2` if above the 60th percentile, and `3` if the bin fails the visibility test or has no frames.

ii.
```python
result = np.full((n_trials, n_bins), 3, dtype=np.int64)  # default: not visible
...
if avg_likelihood >= likelihood_thresh:
    avg_y = np.mean(frames[:, 1])
    if avg_y < tongue_y_p40:
        result[t, b] = 0
    elif avg_y <= tongue_y_p60:
        result[t, b] = 1
    else:
        result[t, b] = 2
```

iii. The notes specify the same four output categories and tie the visibility rule to the `0.9` likelihood threshold.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each trial, the AI selects frames between `go + T_START` and `go + T_STOP`, assigns them to bins on that same go-cue-relative grid, and returns one tongue class per neural time bin.

ii.
```python
abs_edges = gc + bin_edges
i_start = np.searchsorted(tongue_timestamps, abs_edges[0])
i_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
...
bin_idx = np.searchsorted(abs_edges, trial_ts, side='right') - 1
bin_idx = np.clip(bin_idx, 0, n_bins - 1)
```

iii. The notes describe tongue y-position as a time-varying output aligned to the same 50 ms go-cue-centered grid as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases with fallbacks or dropping. Sessions with no good units are dropped. Trials outside the inferred observation window are dropped. If no tone onset is found before a go cue, the code substitutes `-1.85 s`. If a session has no visible tongue frames above the threshold, the code sets the session percentiles to `0.0`. If a tongue bin has no frames or low average likelihood, it is labeled “not visible.”

ii.
```python
if n_good == 0:
    io.close()
    return None
```

```python
if len(candidates) > 0:
    tone_onsets_rel[t] = candidates[-1] - gc
else:
    tone_onsets_rel[t] = -1.85  # fallback
```

```python
if np.sum(visible_mask_all) > 0:
    visible_y = tongue_data[visible_mask_all, 1]
    tongue_y_p40 = np.percentile(visible_y, 40)
    tongue_y_p60 = np.percentile(visible_y, 60)
else:
    tongue_y_p40 = 0.0
    tongue_y_p60 = 0.0
```

iii. The notes justify some of these choices by saying invalid recording periods should be excluded, the sample-to-go-cue gap is consistently about 1.85 s, and low-likelihood tongue frames should become “not visible.” There is no separate note defending the `0.0` percentile fallback.

## 10-a. What are the most time-consuming steps of the code?

i. The AI instrumented per-session timings and appears to regard spike binning and tongue processing as the main costs within a session, with file reading reported separately. The code structure makes per-unit spike binning the dominant compute loop, followed by tongue binning.

ii.
```python
t_read = time.time()
...
for i, unit_idx in enumerate(good_indices):
    st = all_spike_times_obj[unit_idx]
    fr_all[i] = bin_spikes_all_trials(st, go_cue_times, bin_edges)

t_spike = time.time()
...
tongue_disc = bin_tongue_all_trials(...)
t_tongue = time.time()
...
print(f'  {os.path.basename(nwb_path)}: {n_good} units, {n_valid_trials} trials, '
      f'{t1-t0:.1f}s (read:{t_read-t0:.1f}s spike:{t_spike-t_read:.1f}s '
      f'tongue:{t_tongue-t_spike:.1f}s assemble:{t1-t_tongue:.1f}s)')
```

iii. The notes say the script includes “optimized vectorized binning” and record sample runtime by session, but they do not give a more granular written justification than the timing output itself.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain unvectorized: spike binning loops over trials inside each unit, tongue processing loops over trials and then bins, tone-onset extraction scans candidate sample times trial by trial, and photostim construction loops over every trial and every photostim interval.

ii.
```python
for t in range(n_trials):
    gc = go_cue_times[t]
    mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
```

```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        frame_mask = bin_idx == b
```

```python
for t in range(n_valid_trials):
    gc = go_cue_times[t]
    for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
```

iii. The notes describe the code as “optimized vectorized binning,” but they do not explicitly defend these remaining nested loops.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats several scans that could have been shared or preindexed. It rescans the full `sample_start_times` array for every trial, rescans all photostim intervals for every trial, and loops through all bins again inside each trial of tongue processing.

ii.
```python
for t, trial_idx in enumerate(valid_trial_indices):
    gc = go_cue_times_all[trial_idx]
    candidates = sample_start_times[sample_start_times <= gc + 0.01]
```

```python
for t in range(n_valid_trials):
    gc = go_cue_times[t]
    for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
```

```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        frame_mask = bin_idx == b
```

iii. There is no explicit note acknowledging these repeated computations; the notes instead characterize the script as optimized.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code does extra work for plotting and diagnostics that is not required for the converted dataset itself. It imports `matplotlib` unconditionally, computes `subject_desc` but never uses it downstream, carries `tongue_y_p40` and `tongue_y_p60` in the per-session return only for plots/debugging, and has an optional `show_processing_plots` path that is unrelated to the final pickle contents.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```

```python
subject_desc = nwb.subject.description
...
return {
    ...
    'subject_desc': subject_desc,
    ...
    'tongue_y_p40': tongue_y_p40,
    'tongue_y_p60': tongue_y_p60,
}
```

```python
def show_processing_plots(session_result, bin_edges, session_idx, nwb_path):
    ...
```

iii. The notes justify the plotting path as part of the requested `--show-processing` mode, but they do not claim these values are needed by downstream decoder analyses.
