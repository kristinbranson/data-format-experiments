# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds every NWB file under `/app/data/sub-*/*.nwb`, sorts the paths, and processes each file as one session with `pynwb.NWBHDF5IO`. Within a file it reads the subject, units, trial columns, behavioral events, and tongue-tracking series. `--sample` restricts this to the first two files.

ii.
```python
nwb_files = sorted(glob.glob('/app/data/sub-*/*.nwb'))
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, bin_edges, n_bins, ...)
```
```python
io = pynwb.NWBHDF5IO(nwb_path, 'r')
nwb = io.read()
```

iii. The notes identify the source as 174 NWB files for 28 subjects and state that each file contains one session. NWB is treated as the authoritative published representation; sorting makes traversal deterministic.

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `nwb.subject.subject_id`. After all sessions are processed, unique IDs are sorted into `subjects`, and each retained session gets an integer `subject_idx`.

ii.
```python
subject_id = nwb.subject.subject_id
subject_ids = sorted(set(r['subject_id'] for r in all_results))
subjects = [str(sid) for sid in subject_ids]
subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
subject_idx.append(subject_id_to_idx[r['subject_id']])
```

iii. The notes describe `subject_id` as the NWB source variable and report 28 unique subjects, matching the paper and dataset inventory.

## 1-c. How are the data split into sessions?

i. One NWB file is taken to be one session. Every successfully processed file contributes one element to each session-level output list; sessions with no good units or fewer than two valid trials are skipped.

ii.
```python
for i, nwb_path in enumerate(nwb_files):
    result = process_session(nwb_path, bin_edges, n_bins, ...)
    if result is not None:
        all_results.append(result)
```

iii. The notes explicitly say each NWB file is one session. They report 173 retained sessions and one skipped session, which had no valid trials after the script's filters.

## 1-d. How are the data split into trials?

i. Trial-indexed columns come from `nwb.trials`, and go-cue timestamps are assumed to have the same indexing. A Boolean mask is applied to both via `valid_trial_indices`; each retained index becomes one trial matrix in the output.

ii.
```python
auto_water = nwb.trials['auto_water'][:]
free_water = nwb.trials['free_water'][:]
go_cue_times_all = be.time_series['go_start_times'].timestamps[:]
valid_trial_indices = np.where(valid_trial_mask)[0]
go_cue_times = go_cue_times_all[valid_trial_indices]
```

iii. The agent relied on the NWB trials table and behavioral events as parallel trial representations. Its notes describe the trials-table fields but do not document an explicit assertion that trial rows and go cues have equal length.

## 1-e. How are trials filtered based on quality controls?

i. The script excludes `auto_water` and `free_water` trials. It also retains only trials whose entire go-aligned `[-2.5, 1.5]` window lies between the first and last observation bounds of the first good unit. Early-lick, ignore, and photostimulation trials are deliberately retained. Sessions with fewer than two surviving trials are dropped.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
obs_intervals = nwb.units['obs_intervals'][good_indices[0]]
within_obs = ((go_cue_times_all + begin_time) >= obs_start) & \
             ((go_cue_times_all + end_time) <= obs_end)
valid_trial_mask = valid_trial_mask & within_obs
if n_valid_trials < 2:
    return None
```

iii. The notes justify keeping early-lick, ignore, and photostim trials because they are required targets/inputs, while calling auto/free-water trials nonstandard. The full-window observation filter is intended to prevent all-zero neural windows; it produced 89,532 trials, versus 90,860 in the reference conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `nwb.units['spike_times']` for units whose `classification` is `good`, together with go-cue timestamps used for alignment.

ii.
```python
classification = nwb.units['classification'][:]
good_indices = np.where(classification == 'good')[0]
all_spike_times_obj = nwb.units['spike_times']
st = all_spike_times_obj[unit_idx]
```

iii. The notes say spike times are the NWB neural source and classifier-good units reproduce the reference classifier-based QC.

## 2-b. How is the `neural` data processed?

i. For each good unit and trial, spikes in the absolute go-aligned window are selected, shifted relative to the go cue, histogrammed into fixed bins, and divided by bin width to obtain Hz. No smoothing, normalization, or baseline subtraction is applied.

ii.
```python
mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
aligned = spike_times_unit[mask] - gc
counts, _ = np.histogram(aligned, bins=bin_edges)
fr[t] = counts / bin_width
```

iii. The agent chose 50-ms nonoverlapping firing rates because the decoder task overrides the papers' alternate 40-ms/3.4-ms or 100-ms/50-ms settings.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `classification == 'good'` are retained. No 2-Hz firing-rate cutoff is imposed. A session with no good units is discarded.

ii.
```python
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    return None
```

iii. The notes identify this as the classifier QC from the spike-sorting reference. They reject the method paper's 2-Hz cutoff as specific to its video-prediction analysis, not general QC, and report 69,453 retained units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each unit's session-absolute spike times are selected between `go cue - 2.5 s` and `go cue + 1.5 s`, then the go-cue timestamp is subtracted before histogramming against relative edges.

ii.
```python
gc = go_cue_times[t]
mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
aligned = spike_times_unit[mask] - gc
counts, _ = np.histogram(aligned, bins=bin_edges)
```

iii. The notes state that NWB spike and event timestamps share an absolute session clock and that go cue is time zero, so subtraction provides direct alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The analysis window is divided into 80 nonoverlapping 50-ms bins. Raw spike timestamps are binned once into rates; there is no later rebinning.

ii.
```python
BEGIN_TIME = -2.5
END_TIME = 1.5
BIN_WIDTH = 0.05
n_bins = int(round((END_TIME - BEGIN_TIME) / BIN_WIDTH))
bin_edges = np.linspace(BEGIN_TIME, END_TIME, n_bins + 1)
```

iii. The 50-ms bins and four-second window are direct decoder-task requirements.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times`, each trial's `go_start_times`, and the common bin centers. The selected tone is the last sample-start event no later than 10 ms after the go cue.

ii.
```python
sample_start_times = be.time_series['sample_start_times'].timestamps[:]
candidates = sample_start_times[sample_start_times <= gc + 0.01]
tone_onsets_rel[t] = candidates[-1] - gc
```

iii. The notes report a consistent sample-to-go delay of about 1.85 s and select the last onset because early licks can replay task epochs.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone time is stored relative to go cue, then subtracted from every neural bin center. If no candidate exists, the script substitutes `-1.85` seconds.

ii.
```python
tone_onsets_rel[t] = candidates[-1] - gc
# fallback
tone_onsets_rel[t] = -1.85
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. The notes justify the transform as a continuous time-from-tone ramp and cite the empirically consistent 1.85-s interval; they do not demonstrate that the fallback is ever needed.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same go-relative bin centers used by the neural histogram, producing one value for each of the 80 neural bins.

ii.
```python
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
time_from_tone = (bin_centers - tone_onsets_rel[t]).astype(np.float32)
```

iii. The shared bin centers guarantee one-to-one temporal correspondence.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It is derived from the absolute timestamps of `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, plus each trial's go cue.

ii.
```python
ps_start_times = be.time_series['photostim_start_times'].timestamps[:]
ps_stop_times = be.time_series['photostim_stop_times'].timestamps[:]
ps_s_rel = ps_s - gc
ps_e_rel = ps_e - gc
```

iii. The notes state that these event times are absolute and therefore must be expressed relative to the trial's go cue.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, every global stimulation interval overlapping the analysis window is tested. Bin centers within `[start, stop)` are added to a binary vector, which is clipped to 0/1. Sessions without event streams remain all zero.

ii.
```python
photostim_all[t] += ((bin_centers >= ps_s_rel) &
                     (bin_centers < ps_e_rel)).astype(np.float32)
photostim_all = np.clip(photostim_all, 0, 1)
```

iii. This implements the required time-varying indicator and handles overlapping/global event scanning defensively.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute stimulation boundaries are shifted by the same trial go cue and compared with the same relative bin centers as neural activity.

ii.
```python
ps_s_rel = ps_s - gc
ps_e_rel = ps_e - gc
(bin_centers >= ps_s_rel) & (bin_centers < ps_e_rel)
```

iii. The notes explicitly describe converting absolute event times to go-relative time.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trials-table `trial_instruction` and `outcome`: hit means instructed side, miss means the opposite side, and all other outcomes mean no lick.

ii.
```python
if out == 'hit':
    choices[t] = 0 if instr == 'left' else 1
elif out == 'miss':
    choices[t] = 1 if instr == 'left' else 0
else:
    choices[t] = 2
```

iii. The agent notes that choice is not directly stored but is determined by instruction and outcome.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The inferred classes are encoded as left=0, right=1, no lick=2 and repeated across all 80 time bins.

ii.
```python
np.full(n_bins, int(choices[t]), dtype=np.int64)
```

iii. The code follows the requested category order; repetition lets per-trial and time-varying outputs share one rectangular array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `nwb.trials['outcome']`.

ii.
```python
outcome_arr = nwb.trials['outcome'][:]
out = outcome_arr[trial_idx]
```

iii. The NWB field already supplies the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped to ignore=0, miss=1, hit=2, with unknown values defaulting to 0, and the value is repeated across bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcomes[t] = outcome_map.get(out, 0)
np.full(n_bins, int(outcomes[t]), dtype=np.int64)
```

iii. The mapping matches the requested order. The default is a defensive choice but conflates an unexpected/missing value with a genuine ignore.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `nwb.trials['early_lick']`.

ii.
```python
early_lick_arr = nwb.trials['early_lick'][:]
```

iii. The trials table explicitly records early-lick status.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other value to 0; that per-trial class is repeated across 80 bins.

ii.
```python
early_vals[t] = 1 if early_lick_arr[trial_idx] == 'early' else 0
np.full(n_bins, int(early_vals[t]), dtype=np.int64)
```

iii. This gives the requested no/yes coding, though the catch-all else would treat unexpected missing labels as no.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and the `(x, y, likelihood)` data from `BehavioralTimeSeries/Camera0_side_TongueTracking`; column 1 is y and column 2 is tracking likelihood.

ii.
```python
tongue_ts_obj = bts.time_series['Camera0_side_TongueTracking']
tongue_data = tongue_ts_obj.data[:]
tongue_timestamps = tongue_ts_obj.timestamps[:]
visible_mask_all = tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH
visible_y = tongue_data[visible_mask_all, 1]
```

iii. The notes identify this as the session's tongue tracking and use likelihood to represent visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The script uses a 0.9 confidence threshold. Session percentiles are computed from all individually visible raw-frame y values. Within each trial/bin it averages likelihood across all frames; only if that average is at least 0.9 does it average y across all frames and classify the result. Otherwise the bin stays category 3.

ii.
```python
TONGUE_LIKELIHOOD_THRESH = 0.9
visible_y = tongue_data[tongue_data[:, 2] >= TONGUE_LIKELIHOOD_THRESH, 1]
tongue_y_p40 = np.percentile(visible_y, 40)
tongue_y_p60 = np.percentile(visible_y, 60)
avg_likelihood = np.mean(frames[:, 2])
if avg_likelihood >= likelihood_thresh:
    avg_y = np.mean(frames[:, 1])
```

iii. The notes call 0.9 a DeepLabCut convention and say percentiles should cover visible y positions over the session. This differs from the reference, which first removes low-confidence frames, computes 50-ms bin means, and takes percentiles of those means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session raw visible-frame 40th and 60th percentiles define class 0 below p40, class 1 from p40 through p60, class 2 above p60, and class 3 when mean confidence is below 0.9 or no frames exist.

ii.
```python
if avg_y < tongue_y_p40:
    result[t, b] = 0
elif avg_y <= tongue_y_p60:
    result[t, b] = 1
else:
    result[t, b] = 2
```

iii. The per-session 40/60 thresholds and four labels follow the task, but the statistic used for the thresholds (raw frames) does not match the statistic classified (bin averages) or the reference implementation.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in each absolute `go + [-2.5, 1.5]` interval are found with `searchsorted` and assigned to the same go-relative edges as neural bins.

ii.
```python
abs_edges = gc + bin_edges
i_start = np.searchsorted(tongue_timestamps, abs_edges[0])
i_end = np.searchsorted(tongue_timestamps, abs_edges[-1])
bin_idx = np.searchsorted(abs_edges, trial_ts, side='right') - 1
```

iii. Shared absolute timestamps and identical edges provide direct neural/video alignment.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units and sessions with fewer than two valid trials are skipped; windows outside the first good unit's observation range are removed; absent photostim streams become zero; missing tone onsets are imputed as -1.85 s; sessions with no visible tongue get zero percentile edges; invisible/empty tongue bins become class 3; unknown outcomes become ignore and non-`early` early-lick values become no.

ii.
```python
if n_good == 0: return None
if n_valid_trials < 2: return None
tone_onsets_rel[t] = -1.85  # fallback
if np.sum(visible_mask_all) == 0:
    tongue_y_p40 = tongue_y_p60 = 0.0
result = np.full((n_trials, n_bins), 3, dtype=np.int64)
outcomes[t] = outcome_map.get(out, 0)
```

iii. The general intent is to avoid fabricating neural data and preserve explicit tongue invisibility. Some fallbacks silently turn malformed/missing data into legitimate classes or fixed values rather than raising, excluding, or explicitly marking it.

## 10-a. What are the most time-consuming steps of the code?

i. The agent's timers separate file reading, spike binning, tongue binning, and assembly. The nested per-unit/per-trial spike selection and histogramming is the principal compute-heavy step; reading large NWB spike/video arrays and writing the roughly 12-GB pickle are also expensive.

ii.
```python
for i, unit_idx in enumerate(good_indices):
    st = all_spike_times_obj[unit_idx]
    fr_all[i] = bin_spikes_all_trials(st, go_cue_times, bin_edges)
```

iii. The code labels this optimized, but the full conversion logs and per-stage timers are the evidence. The notes report about 3.7 s/session on a sample and a large full output.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Spike binning loops over units and then trials, repeatedly scanning an entire unit's spike vector. Tongue binning loops over trials and then all 80 bins. Tone selection loops over trials, and photostim loops over trials and every stimulation event. These could use flattened edge `searchsorted`, indexed reductions, or trial/event association.

ii.
```python
for t in range(n_trials):
    mask = (spike_times_unit >= gc + begin) & (spike_times_unit < gc + end)
```
```python
for t in range(n_trials):
    ...
    for b in range(n_bins):
        frame_mask = bin_idx == b
```

iii. Although the notes call the pipeline optimized, the reference demonstrates that neural trials can be vectorized per unit by flattening all bin edges. The agent did not discuss these remaining loop costs critically.

## 10-c. What processing does the code repeat multiple times?

i. It rescans each unit's full spike array once per trial; scans all sample events once per trial; tests all session photostim intervals once per trial; and constructs 80-element constant arrays separately for three per-trial outputs. Optional plotting also concatenates/derives summaries already available from processed arrays.

ii.
```python
candidates = sample_start_times[sample_start_times <= gc + 0.01]
for ps_s, ps_e in zip(ps_start_times, ps_stop_times):
    ...
np.full(n_bins, int(choices[t]), dtype=np.int64)
```

iii. These choices favor straightforward code. The notes' statement that the script is optimized overlooks repeated scans inside nested loops.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `subject.description`, stores diagnostic-only values in each temporary session result (`nwb_path`, tongue percentile values), computes stage timestamps and extensive summary distributions, and imports plotting machinery even when plotting is disabled. The optional plotting path computes large flattened firing-rate summaries and image files not used by the decoder.

ii.
```python
subject_desc = nwb.subject.description
return {'subject_desc': subject_desc, 'nwb_path': nwb_path,
        'tongue_y_p40': tongue_y_p40, 'tongue_y_p60': tongue_y_p60, ...}
```

iii. These support diagnostics and documentation rather than the serialized decoder fields. They are modest except for optional plotting, but are discarded when the final `data` dictionary is assembled.
