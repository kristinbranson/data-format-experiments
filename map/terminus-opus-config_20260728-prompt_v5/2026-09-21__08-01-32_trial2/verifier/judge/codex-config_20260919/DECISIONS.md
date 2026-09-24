# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `sub-*` directory, collects every NWB file in sorted order, and opens each file once with `pynwb.NWBHDF5IO`. It reports 174 files from 28 subjects and processes all files unless sample mode is requested.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
...
nwb_files = sorted(glob.glob(os.path.join(subj_dir, '*.nwb')))
...
with pynwb.NWBHDF5IO(nwb_path, 'r') as io:
    nwb = io.read()
```

iii. The notes identify NWB as the supplied data structure and state that `/app/data/` contains 28 subject directories and 174 files. The full run was used to include all sessions.

## 1-b. How are the data split into subjects?

i. Subject directories organize file discovery, but the output subject identity is read from `nwb.subject.subject_id`. Unique IDs are sorted and each retained session is mapped to its index.

ii.
```python
subject_id = nwb.subject.subject_id
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The agent treated the NWB subject field as canonical and validated that the converted output has 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Retained session lists are appended in sorted file order and identified with `nwb.identifier`.

ii.
```python
session_id = nwb.identifier
...
for i, (subj, fpath) in enumerate(all_files):
    result = process_session(fpath, ...)
    ...
    all_neural.append(result['neural'])
```

iii. The notes say 174 files represent sessions and that the one file without good units is skipped, producing the expected 173 sessions.

## 1-d. How are the data split into trials?

i. Trial-indexed values come from `nwb.trials`; go-cue timestamps are assumed to have the same order and count. After a validity mask is constructed, each retained trial index selects the corresponding go cue and behavioral labels.

ii.
```python
trials = nwb.trials
n_trials = len(trials)
go_times = be.time_series['go_start_times'].timestamps[:]
...
valid_trial_indices = np.where(valid_trial_mask)[0]
for i, trial_idx in enumerate(valid_trial_indices):
    go_time = go_times[trial_idx]
```

iii. The agent relied on the NWB trial table and corresponding event ordering; its sanity checks compared converted trial labels against raw NWB values.

## 1-e. How are trials filtered based on quality controls?

i. It excludes `auto_water` and `free_water` trials, then excludes any trial whose entire -2.5-to-+1.5 s go-cue window is not within the first good unit's overall recording range. Sessions with fewer than two surviving trials are skipped. It retains early-lick, ignore, and photostimulation trials.

ii.
```python
valid_trial_mask = (auto_water == 0) & (free_water == 0)
for i in range(n_trials):
    if valid_trial_mask[i]:
        window_start = go_times[i] + ALIGN_START
        window_end = go_times[i] + ALIGN_END
        if window_start < rec_min or window_end > rec_max:
            valid_trial_mask[i] = False
...
if n_valid_trials < 2:
    return None
```

iii. The notes justify excluding water-delivery trials and trials outside recording coverage, while retaining all behavioral conditions needed as decoder targets. They describe this as use of `obs_intervals`, although the code collapses the intervals to one minimum/maximum range.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units['spike_times']` for units whose `units['classification']` equals `good`; go-cue timestamps define trial windows.

ii.
```python
classification = units['classification'][:]
good_indices = np.where(classification == 'good')[0]
...
st = units['spike_times'][idx]
good_spike_times.append(np.sort(st))
```

iii. The notes identify spike times as the NWB neural source and `classification == "good"` as matching classifier-mode QC in the reference material.

## 2-b. How is the `neural` data processed?

i. For every retained trial and good unit, sorted spikes in the aligned window are histogrammed into non-overlapping bins and divided by 0.05 s to produce float32 firing rates in Hz. There is no smoothing or normalization.

ii.
```python
counts, _ = np.histogram(spikes_window, bins=bin_edges)
fr[i] = counts / bin_width
```

iii. The agent states that it followed the requested 50-ms firing-rate representation and spot-checked the result against raw spike times with exact agreement.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units labeled `good` in `classification` are retained. A session with zero such units is skipped; no additional firing-rate or quality-metric filter is applied.

ii.
```python
good_mask = classification == 'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    return None
```

iii. The notes connect this field to the region-specific classifier QC and explain that 173 sessions survive, consistent with the paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute bin edges are obtained by adding offsets from -2.5 to +1.5 s to each trial's go-cue timestamp; spike timestamps are histogrammed against those edges.

ii.
```python
bin_offsets = align_start + np.arange(n_bins + 1) * bin_width
bin_edges = go_cue_times[t] + bin_offsets
```

iii. The notes explicitly identify go-cue onset as the alignment event and report raw-data spot checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data have 80 non-overlapping 50-ms bins over four seconds. Raw spikes are binned directly; there is no further temporal rebinning.

ii.
```python
BIN_WIDTH = 0.050
ALIGN_START = -2.5
ALIGN_END = 1.5
N_TIMEBINS = int((ALIGN_END - ALIGN_START) / BIN_WIDTH)
```

iii. The notes explain that 50 ms is the task requirement, replacing the different bandwidth/stride used in an analysis from the reference repository.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and the current trial's `go_start_times`. The last sample timestamp earlier than `go + 0.01` s is selected; if none exists, a synthetic onset 1.85 s before go is used.

ii.
```python
prev_samples = sample_start_times[sample_start_times < go_time + 0.01]
tone_onset = prev_samples[-1] if len(prev_samples) > 0 else go_time - 1.85
```

iii. The agent notes that repeated sample epochs can occur after early licks, so it chose the last tone before go. The fallback and 10-ms tolerance are not documented in the notes.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each neural bin center in session-absolute time, tone onset is subtracted to yield elapsed seconds since the selected tone.

ii.
```python
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. The mapping plan calls this a continuous time-varying input, and a raw-data spot check reportedly matched exactly.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at `go_time + BIN_CENTERS`, the centers of the same 80 go-aligned bins used for neural firing rates.

ii.
```python
BIN_CENTERS = ALIGN_START + BIN_WIDTH/2 + np.arange(N_TIMEBINS) * BIN_WIDTH
time_from_tone = (go_time + BIN_CENTERS) - tone_onset
```

iii. The common go-cue-relative bin grid was chosen to ensure input and neural columns correspond.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `BehavioralEvents/photostim_start_times` and `photostim_stop_times`, plus the trial-table `photostim_power` field to identify stimulated trials.

ii.
```python
photostim_event_starts = be.time_series['photostim_start_times'].timestamps[:]
photostim_event_stops = be.time_series['photostim_stop_times'].timestamps[:]
...
if has_photostim_events and photostim_power_str[trial_idx] != 'N/A':
```

iii. The notes describe photostimulation events as the source and the desired result as a binary, time-varying input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each stimulated trial, the code scans all session photostimulation intervals and marks a bin as 1 if its interval has any overlap with a stimulation interval; otherwise it remains 0.

ii.
```python
if bin_start_abs < ps_stop and bin_end_abs > ps_start:
    photostim_binary[b] = 1.0
```

iii. The agent intended an on/off time series. It did not document why any-overlap bin labeling was preferred over sampling state at bin centers.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation timestamps and go timestamps remain on the absolute NWB clock, and each stimulation interval is compared with the absolute start/end of every go-aligned neural bin.

ii.
```python
bin_start_abs = go_time + ALIGN_START + b * BIN_WIDTH
bin_end_abs = bin_start_abs + BIN_WIDTH
```

iii. The notes identify the output as time-varying and aligned to the same go-cue window.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from trial-table `trial_instruction` and `outcome`: hit means the instructed side, miss means the opposite side, and ignore means no lick.

ii.
```python
if outcome == 'ignore': return 2
elif outcome == 'hit': return 0 if trial_instruction == 'left' else 1
elif outcome == 'miss': return 1 if trial_instruction == 'left' else 0
```

iii. The notes explicitly map instruction plus outcome to left, right, or no lick and report label spot checks.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded left=0, right=1, no lick=2, then repeated across all 80 time bins.

ii.
```python
output_per_trial.append(np.array([choice, outcome_val, early_lick_val], dtype=np.int64))
...
per_trial_expanded = np.repeat(per_trial[:, np.newaxis], N_TIMEBINS, axis=1)
```

iii. The mapping table documents the three categories; repetition lets scalar trial outputs share one `(4, 80)` output array with tongue position.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the NWB trials-table `outcome` column.

ii.
```python
outcome = trials['outcome'][:]
```

iii. The requested categories already exist in the raw table, so the agent used them directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 (unknown values default to 0), and the trial value is repeated across 80 bins.

ii.
```python
outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
outcome_val = outcome_map.get(outcome[trial_idx], 0)
```

iii. The notes give this exact category mapping and state that trial outputs were spot-checked.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trials-table `early_lick` column.

ii.
```python
early_lick = trials['early_lick'][:]
```

iii. The raw field directly supplies the requested binary trial label.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string `early` maps to 1 and every other value maps to 0; the value is repeated across all time bins.

ii.
```python
early_lick_val = 1 if early_lick[trial_idx] == 'early' else 0
```

iii. The notes define no=0 and yes=1 and report successful spot checks.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses `BehavioralTimeSeries/Camera0_side_TongueTracking`: timestamps align frames, column 1 is y-position, and column 2 is tracking likelihood.

ii.
```python
tongue_data_all = tongue_ts_obj.data[:]
tongue_ts_all = tongue_ts_obj.timestamps[:]
...
visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
tongue_y_binned[b] = frames[visible, 1].mean()
```

iii. The mapping plan identifies tongue y as the raw source and likelihood as visibility confidence.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Within each 50-ms trial bin, frames with likelihood at least 0.9 are averaged. The code also stores the fraction of all frames that are visible. It collects bin means only when at least half the frames are visible, and uses those values from retained trial windows to calculate session percentiles. Missing tracking yields NaNs/zero visibility and eventually class 3.

ii.
```python
visible = frames[:, 2] >= TONGUE_LIKELIHOOD_THRESHOLD
visible_binned[b] = visible.mean()
if visible.sum() > 0:
    tongue_y_binned[b] = frames[visible, 1].mean()
...
visible_mask = tongue_vis >= 0.5
tongue_y_all_visible.extend(tongue_y[visible_mask].tolist())
```

iii. The notes justify 0.9 as a DLC confidence threshold and specify per-session discretization, but do not justify the additional requirement that at least half of a bin's frames be visible or restricting percentile data to retained trial windows.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per session, the 40th and 60th percentiles of selected trial-bin y means define categories: `<p40` is 0, `p40..p60` inclusive at the upper edge is 1, `>p60` is 2, and bins with visibility fraction below 0.5 are 3. If no selected values exist, both edges are zero.

ii.
```python
p40 = np.percentile(tongue_y_arr, 40)
p60 = np.percentile(tongue_y_arr, 60)
...
tongue_discrete[visible_mask] = np.where(ty < p40, 0,
                                         np.where(ty <= p60, 1, 2))
```

iii. The 40/60 per-session split and not-visible class come directly from the task. The precise visibility rule and percentile population were agent choices not explained in the notes.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected between the same absolute go-cue-relative trial endpoints as neural data, digitized into the same 50-ms absolute bin edges, and reduced to one value per neural time bin.

ii.
```python
bin_edges = go_cue_time + align_start + np.arange(n_bins + 1) * bin_width
left_idx = np.searchsorted(tongue_ts, bin_edges[0])
right_idx = np.searchsorted(tongue_ts, bin_edges[-1])
bin_idx = np.digitize(ts_window, bin_edges) - 1
```

iii. The agent designed the binned camera output specifically to match the neural grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions without good units and sessions with fewer than two valid trials are skipped. Trials outside the overall recording span and water trials are removed. Missing tone onset is imputed as 1.85 s before go. Missing tongue tracking makes every bin not visible; absent visible values use percentile edges of zero. Unknown outcome/early-lick labels silently map to baseline classes. Warnings are globally suppressed.

ii.
```python
warnings.filterwarnings('ignore')
...
if n_good == 0: return None
...
tone_onset = prev_samples[-1] if len(prev_samples) > 0 else go_time - 1.85
...
tongue_y_per_trial.append(np.full(N_TIMEBINS, np.nan))
```

iii. The notes mainly justify exclusion of unrecorded data and explicit not-visible tongue coding. They mention one retained all-zero neural trial as valid, but do not discuss silent fallbacks or global warning suppression.

## 10-a. What are the most time-consuming steps of the code?

i. Reading ragged spike arrays and the full tongue series, then firing-rate computation, dominate per-session work; the full conversion and final pickle write also handle a very large payload. The script times unit reading, tongue reading, firing-rate calculation, and I/O/output processing separately.

ii.
```python
print(f"    Reading units: {t_units_end - t_units_start:.1f}s")
print(f"    Reading tongue: {t_tongue_end - t_tongue_start:.1f}s")
print(f"    Computing firing rates: {t_fr_end - t_fr_start:.1f}s")
```

iii. The notes estimate about 3.5 seconds per session and roughly 10 minutes overall; no finer performance justification is provided.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The firing-rate function loops over trials and neurons even though all trial edges for one unit could be searched together. Tongue binning loops over all 80 bins per trial. Photostimulation uses nested trial, session-event, and bin loops. Subject/region index construction repeatedly calls list `index` rather than dictionary lookups.

ii.
```python
for t in range(n_trials):
    ...
    for i, spikes in enumerate(spike_times_list):
...
for b in range(n_bins):
...
for ps_idx in range(len(photostim_event_starts)):
    for b in range(N_TIMEBINS):
```

iii. The agent called its firing-rate routine “vectorized” because histogramming is vectorized within a neuron/trial, but did not discuss the remaining loops or possible flattened-edge `searchsorted` approach.

## 10-c. What processing does the code repeat multiple times?

i. Every stimulated trial scans all photostimulation events and rebuilds every absolute bin interval. Tongue data is first binned/collected, then traversed again for discretization and output assembly. Constant dictionaries are recreated per trial. Summary statistics later traverse all output trials again, and region/subject lookup repeatedly performs linear searches.

ii.
```python
for i, trial_idx in enumerate(valid_trial_indices):
    ...
    outcome_map = {'ignore': 0, 'miss': 1, 'hit': 2}
...
for i in range(len(output_per_trial)):
...
for sess_outputs in data['output']:
    for trial_out in sess_outputs:
```

iii. The notes do not identify repeated work; they focus on validation and total runtime.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `auto_water`, computes a control-trial correct rate solely for logging, computes/stores tongue visible fractions only as an intermediate threshold, sorts spike times that should already be sorted, and optionally creates extensive diagnostic plots. The `(subj, path)` directory label and `session_idx` argument are unused in conversion logic.

ii.
```python
correct_rate = n_hit / (n_hit + n_miss) if (n_hit + n_miss) > 0 else 0
...
good_spike_times.append(np.sort(st))
...
visible_binned[b] = visible.mean()
```

iii. Correct-rate reporting and plots were sanity-check aids, but neither is part of the saved target data. The notes frame these checks as validation rather than necessary conversion processing.
