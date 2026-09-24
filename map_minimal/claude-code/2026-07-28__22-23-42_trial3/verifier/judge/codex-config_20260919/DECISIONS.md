# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script finds sorted NWB files under `data/sub-*/sub-*.nwb`, opens each with `h5py`, and reads trials, events, units, spikes, and tongue tracking directly from NWB datasets.

ii.
```python
nwb_files = sorted(glob.glob('data/sub-*/sub-*.nwb'))
for nwb_path in nwb_files:
    result = process_session(nwb_path)
```
```python
f = h5py.File(nwb_path, 'r')
n_trials = len(f['intervals/trials/id'][:])
```

iii. The trajectory recognized that each NWB file is one session and planned a single pass across all files. It used `h5py` as a direct NWB/HDF5 reader and sorted paths for deterministic processing.

## 1-b. How are the data split into subjects?

i. The numeric subject ID is parsed from each NWB filename, unique IDs are sorted, and every retained session gets an index into that list.

ii.
```python
subject_id = basename.split('_ses-')[0].replace('sub-', '')
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The trajectory identified 28 subjects and treated the `sub-...` directory/filename component as the mouse identifier.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. Each surviving `process_session` result becomes one element of the outer `neural`, `input`, and `output` lists; files failing unit, recording, performance, or trial-count checks are omitted.

ii.
```python
all_neural.append(result['neural'])
all_input.append(result['input'])
all_output.append(result['output'])
all_session_files.append(result['session_file'])
```

iii. The agent explicitly reasoned that each NWB file represents a session. It later reported retaining 144 of 174 files because of its added filtering criteria.

## 1-d. How are the data split into trials?

i. Rows of `intervals/trials` define trials and are matched positionally to `go_start_times`; an assertion checks equal counts. Retained row indices drive all per-trial extraction.

ii.
```python
n_trials = len(f['intervals/trials/id'][:])
go_cue_times = f['acquisition/BehavioralEvents/go_start_times/timestamps'][:]
assert len(go_cue_times) == n_trials
trial_indices = np.where(trial_mask)[0]
for ti in trial_indices:
```

iii. The trajectory found one go cue per trial and used the NWB trials table rather than reconstructing trials from event timestamps.

## 1-e. How are trials filtered based on quality controls?

i. It retains trials with neither `auto_water` nor `free_water` and whose entire -2.5 to +1.5 s window lies inside the common observation range across selected units. It also filters whole sessions to performance above 65% and at least 50 control hits in each direction, and requires at least two surviving trials.

ii.
```python
recording_valid = ((go_cue_times + T_START) >= max_obs_start - 0.1) & \
                  ((go_cue_times + T_END) <= min_obs_end + 0.1)
trial_mask = (auto_water == 0) & (free_water == 0) & recording_valid
```
```python
if performance <= MIN_PERFORMANCE: return None
if control_hits_left < 50 or control_hits_right < 50: return None
```

iii. The agent borrowed session criteria from the methods/reference analysis, while deliberately retaining early-lick, ignore, and photostimulation trials because they are required decoder targets/inputs. After discovering all-zero late trials, it added the observation-window test. It excluded water-delivery trials as nonstandard behavior.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from `units/spike_times` and its ragged index, selected by `units/classification`, aligned using `BehavioralEvents/go_start_times`; `anno_name` is additionally used to retain/map regions.

ii.
```python
spike_times_data = f['units/spike_times'][:]
spike_times_index = f['units/spike_times_index'][:]
classification = np.array([... for c in f['units/classification'][:]])
```

iii. The agent identified spike times as the neural source and chose the classifier verdict described by the spike-sorting QC paper.

## 2-b. How is the `neural` data processed?

i. For every retained trial and unit, spikes are shifted by the go cue, restricted to the window, histogrammed into 80 bins, and divided by 0.05 s to obtain unsmoothed firing rates in Hz.

ii.
```python
rel_times = spike_times - go_cue_time
mask = (rel_times >= bin_edges[0]) & (rel_times < bin_edges[-1])
counts, _ = np.histogram(rel_times[mask], bins=bin_edges)
return counts.astype(np.float64) / bin_width
```

iii. The trajectory states that 50-ms spike counts divided by bin width provide firing rates and that no further normalization was needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units classified exactly `good` and whose `anno_name` maps to one of the manual 14 region categories are retained. Sessions with no such units are dropped.

ii.
```python
good_mask = classification == 'good'
for i in range(len(classification)):
    if not good_mask[i]: continue
    region = map_anno_name_to_region(anno_names[i])
    if region is not None:
        unit_indices.append(i)
```

iii. The classifier was selected as the modern QC verdict. The agent manually constructed a 14-region mapping to mimic reference region groupings and chose to exclude unrecognized/blank annotations.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike times are shifted by that trial's absolute go-cue timestamp, then histogrammed on the common relative grid from -2.5 to +1.5 s.

ii.
```python
gc = go_cue_times[ti]
fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The agent recognized that spikes and cues share an absolute NWB clock and that subtracting the go cue gives the requested alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 nonoverlapping 50-ms bins. Raw spike times are binned directly; no later rebinning, smoothing, or interpolation is applied.

ii.
```python
BIN_WIDTH = 0.05
T_START, T_END = -2.5, 1.5
N_BINS = int((T_END - T_START) / BIN_WIDTH)
BIN_EDGES = np.linspace(T_START, T_END, N_BINS + 1)
```

iii. These values directly implement the decoder instructions and were explicitly noted in the trajectory.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses session `sample_start_times`, the trial's go cue, and the common bin centers; the last sample start before the go cue is selected.

ii.
```python
before = sample_start_times[sample_start_times < go_cue_time]
return before[-1] - go_cue_time
```

iii. The agent observed that early licking can replay the sample epoch, so the last preceding sample start is the relevant tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Tone onset is expressed relative to go, and each go-relative bin center subtracts that value. If no earlier sample event exists, the script imputes the typical -1.85 s offset.

ii.
```python
if tone_onset_rel is None:
    tone_onset_rel = -1.85
time_from_tone = BIN_CENTERS - tone_onset_rel
```

iii. The trajectory derived 1.85 s from the 0.65-s sample plus 1.2-s delay and used it as a defensive fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at exactly the same 80 go-cue-relative bin centers as neural activity and stacked as the first input row.

ii.
```python
time_from_tone = BIN_CENTERS - tone_onset_rel
input_trial = np.stack([time_from_tone, photostim_binary], axis=0)
```

iii. The shared go-relative centers were chosen to make every input timepoint correspond to the same neural bin.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It derives from trial `photostim_onset`, `photostim_duration`, and `start_time`, plus the trial go-cue timestamp.

ii.
```python
ps_onset_str = ... f['intervals/trials/photostim_onset'][:]
ps_dur_str = ... f['intervals/trials/photostim_duration'][:]
trial_start_times = f['intervals/trials/start_time'][:]
```

iii. During testing the agent discovered onset is relative to trial start, corrected an earlier absolute-time interpretation, and verified both 0 and 1 occur.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For non-`N/A` trials it parses onset/duration, converts onset to absolute and then go-relative time, and marks bin centers in the half-open stimulation interval as 1. Parsing failures silently leave zeros.

ii.
```python
ps_abs_onset = trial_start_times[ti] + ps_onset_val
ps_start_rel = ps_abs_onset - gc
ps_end_rel = ps_start_rel + ps_dur
photostim_binary = ((BIN_CENTERS >= ps_start_rel) &
                    (BIN_CENTERS < ps_end_rel)).astype(np.float64)
```

iii. The binary time series follows the requested on/off representation. The trajectory documents the trial-start correction and sample verification.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation times are converted to go-relative seconds and evaluated at the same bin centers used for neural trials.

ii.
```python
ps_start_rel = ps_abs_onset - gc
photostim_binary = (BIN_CENTERS >= ps_start_rel) & (BIN_CENTERS < ps_end_rel)
```

iii. Both streams are thereby represented on the identical go-cue-relative 50-ms grid.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The script uses only `trial_instruction`, not actual response/outcome information: instructed left becomes choice 0 and everything else becomes choice 1.

ii.
```python
if instructions[ti] == 'left':
    choice = 0
else:
    choice = 1
```

iii. The trajectory described choice as left/right but did not resolve that misses imply the opposite lick and ignores imply no lick; it effectively equated instructed side with choice.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The two-valued integer is repeated across all 80 time bins. Only `['left', 'right']` are declared, with no no-lick category.

ii.
```python
np.full(N_BINS, output_choice[i], dtype=np.int64)
```
```python
['left', 'right']
```

iii. The agent chose a per-trial categorical output and broadcast it so all outputs share `(4, 80)` shape. Its decoder check treated this as a two-class task.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from `intervals/trials/outcome`.

ii.
```python
outcomes = np.array([... for o in f['intervals/trials/outcome'][:]])
```

iii. The NWB column already contains exactly the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2, and the value is broadcast across 80 bins.

ii.
```python
if outcomes[ti] == 'ignore': outcome = 0
elif outcomes[ti] == 'miss': outcome = 1
else: outcome = 2
```

iii. This matches the requested categorical order; broadcasting provides a common time-shaped output array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early_lick = np.array([... for e in f['intervals/trials/early_lick'][:]])
```

iii. The task requires this flag as an output, which is why the agent deliberately retained early-lick trials.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'early'` maps to 1 and every other value to 0, then the value is repeated across 80 bins.

ii.
```python
el = 1 if early_lick[ti] == 'early' else 0
np.full(N_BINS, output_early_lick[i], dtype=np.int64)
```

iii. The agent encoded no/yes as 0/1 and broadcast this per-trial label for consistent output shape.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses timestamps and column 1 of `Camera0_side_TongueTracking/data`. Although column 2 contains likelihood, the final code ignores it.

ii.
```python
tongue_ts = f['.../Camera0_side_TongueTracking/timestamps'][:]
tongue_data = f['.../Camera0_side_TongueTracking/data'][:]
tongue_y = tongue_data[:, 1]
```

iii. The agent initially considered confidence filtering, but decided to use all tracking values to obtain an apparent 40/20/40 category distribution.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, camera y values are selected in the go-relative window and averaged within each 50-ms bin. Percentiles are then computed over all non-NaN trial-bin means in that retained session. Empty bins are defaulted to the middle category.

ii.
```python
tongue_y_binned[b] = np.mean(y_window[bin_mask])
valid_tongue_y = np.concatenate(tongue_y_all_trials)
valid_tongue_y = valid_tongue_y[~np.isnan(valid_tongue_y)]
p40 = np.percentile(valid_tongue_y, 40)
p60 = np.percentile(valid_tongue_y, 60)
```

iii. The trajectory rejected confidence filtering because most frames were low-confidence and opted to interpret all coordinates, reporting a balanced distribution after this change.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Values below p40 are 0, values from p40 through p60 are 1, and values above p60 are 2. NaN bins and sessions with no valid data are also assigned 1. No category 3 (`not visible`) is created.

ii.
```python
ty_disc = np.ones(N_BINS, dtype=np.int64)
ty_disc[valid_mask & (ty < p40)] = 0
ty_disc[valid_mask & (ty >= p40) & (ty <= p60)] = 1
ty_disc[valid_mask & (ty > p60)] = 2
```

iii. The agent believed the specification allowed only three categories and explicitly chose middle-class imputation for missing bins, despite the instruction listing a fourth not-visible class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go-2.5, go+1.5)` are shifted relative to go and averaged into the same 80 intervals defined by `BIN_EDGES`.

ii.
```python
idx = np.searchsorted(tongue_timestamps, [go_cue_time + bin_edges[0],
                                          go_cue_time + bin_edges[-1]])
ts_rel = ts_window - go_cue_time
bin_mask = (ts_rel >= bin_edges[b]) & (ts_rel < bin_edges[b + 1])
```

iii. The agent intentionally used the identical window and bin grid so tongue category at each position corresponds to the neural bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Files are closed with `finally`; sessions without usable units/control trials or enough retained trials are skipped. Missing tone onset is imputed as -1.85 s, malformed photostimulation silently becomes all-off, missing tongue bins become middle class, and unrecognized region annotations cause units to be dropped. Assertions catch trial/go-count mismatch.

ii.
```python
if tone_onset_rel is None: tone_onset_rel = -1.85
except (ValueError, TypeError): pass
ty_disc = np.ones(N_BINS, dtype=np.int64)
finally: f.close()
```

iii. The trajectory added pragmatic fallbacks to keep conversion running and used validation warnings to diagnose observation gaps. It did not preserve missing tongue visibility as its own class.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant work is the nested trial-by-unit spike histogramming, followed by per-trial/per-bin tongue masking; reading many large NWB arrays and serializing the 19.7-GB float64 result are also expensive.

ii.
```python
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times):
        fr_matrix[j, :] = compute_firing_rates(st, gc, BIN_EDGES)
```

iii. The trajectory shows the full conversion running for an extended period and producing a 19.71-GB file, but gives no formal profiling. This assessment follows the loop structure and data volume.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-by-unit firing-rate computation could vectorize all trials for each unit using `searchsorted`; tongue binning could use digitized bin indices and grouped sums; repeated `list.index` mappings could use dictionaries.

ii.
```python
for ti in trial_indices:
    for j, st in enumerate(unit_spike_times): ...
for b in range(n_bins):
    bin_mask = ...
```

iii. The agent did not discuss these optimizations. Its implementation prioritizes direct, readable loops and validates successfully, but scales worse than the reference.

## 10-c. What processing does the code repeat multiple times?

i. For every trial and unit it repeatedly subtracts the go cue from that unit's full spike vector, masks it, and calls `np.histogram`. For each trial/bin it repeatedly scans camera-window timestamps. It also performs linear `list.index` lookups while assembling subject and region indices.

ii.
```python
rel_times = spike_times - go_cue_time
counts, _ = np.histogram(rel_times[mask], bins=bin_edges)
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The trajectory does not justify this repetition; it arose from the straightforward per-trial implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Session-level control masks/counts/performance are computed only to enforce the agent's extra session filter, and verbose distribution/summary statistics are computed only for logging. Otherwise most loaded/derived arrays contribute to filtering or output.

ii.
```python
is_control = ((early_lick == 'no early') & ...)
performance = np.sum(is_control & (outcomes == 'hit')) / n_control_responding
region_counts = {}
```

iii. The performance work was intentional because the agent believed the methods' session criteria should be applied; the summaries supported its sanity checks but are not stored for decoder use.
