# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `sub-*` directory under the relative `data` directory, sorts each subject's NWB files, and processes each file with `h5py`. It reads trials, events, units, electrodes, and tongue tracking directly from NWB HDF5 groups.

ii.
```python
subjects = sorted([d for d in os.listdir(DATA_DIR) if d.startswith('sub-')])
for subj in subjects:
    nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
```
```python
f = h5py.File(nwb_path, 'r')
trials = f['intervals']['trials']
spike_times_flat = f['units']['spike_times'][:]
```

iii. The notes say there are 28 subject directories and 174 NWB files and explain that the reference used DataJoint `.mat` exports, so the agent mapped equivalent NWB fields and used `h5py` for speed.

## 1-b. How are the data split into subjects?

i. Subject boundaries come from `sub-*` directory names. The first accepted session for a subject appends that folder name to `all_subjects`; each session receives its index in that encounter-ordered list.

ii.
```python
all_files.append({'subject': subj, 'path': os.path.join(subj_dir, nwb_file)})
```
```python
if subj not in all_subjects:
    all_subjects.append(subj)
all_subject_idx.append(all_subjects.index(subj))
```

iii. The agent treated the DANDI directory layout as the subject organization. The notes report 28 subjects and use identifiers such as `sub-440956`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Accepted files append one element to each session-level output list. The agent additionally excludes sessions failing trial availability, performance, per-side hit-count, or good-unit criteria.

ii.
```python
for i, nwb_info in enumerate(nwb_files):
    result = process_session(nwb_info['path'], nwb_info['subject'], ...)
    if result is None:
        skipped += 1
        continue
    all_neural.append(result['neural'])
```

iii. The notes identify one NWB file with one behavioral/ephys session, but justify retaining only sessions with at least 65% correct and at least 50 correct trials per side based on the methods paper. This leaves 143 of 174 files.

## 1-d. How are the data split into trials?

i. Trial rows come from `/intervals/trials`; their integer positions index trial columns and the corresponding go-cue array. Each selected index produces one neural, input, and output array.

ii.
```python
n_trials_total = len(trials['id'])
go_times = f['acquisition']['BehavioralEvents']['go_start_times']['timestamps'][:]
trial_indices = np.where(trial_mask)[0]
for trial_idx in trial_indices:
    go_time = go_cue_times[trial_idx]
```

iii. The notes describe NWB's trials table and behavioral event series as equivalent to the reference data fields. The code assumes their rows/events correspond by position.

## 1-e. How are trials filtered based on quality controls?

i. The agent excludes `auto_water`, `free_water`, and trials it deems outside neural coverage. Coverage is approximated by limiting indices to `is_good_trials.shape[1]` and requiring the complete go-aligned window to lie within the global minimum/maximum spike times, with a one-second tolerance. It retains early-lick, ignore, and photostimulation trials, requires two trials, then filters whole sessions by control/non-early performance (65% correct and 50 hits per side).

ii.
```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
for t_idx in range(min(n_trials_total, n_recorded_trials)):
    if (go + WINDOW_START >= min_spike_time - 1.0 and
        go + WINDOW_END <= max_spike_time + 1.0):
        neural_valid[t_idx] = True
trial_mask = (auto_water == 0) & (free_water == 0) & neural_valid
```
```python
if correct_rate < MIN_CORRECT_RATE: return None
if correct_left < 50 or correct_right < 50: return None
```

iii. The notes say early-lick, ignore, and photostim trials must remain because they define requested variables, while water trials are excluded. They attribute session thresholds to the methods and describe the coverage heuristic as handling partial recordings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, selected by `units/classification == b'good'`, and is aligned using `BehavioralEvents/go_start_times`.

ii.
```python
spike_times_flat = f['units']['spike_times'][:]
spike_times_index = f['units']['spike_times_index'][:]
classification = f['units']['classification'][:]
good_indices = np.where(classification == b'good')[0]
```

iii. The notes identify absolute NWB spike timestamps and classifier-based good-unit labels as the appropriate source and QC verdict.

## 2-b. How is the `neural` data processed?

i. For each selected trial and each good unit, spikes in the four-second window are selected with `searchsorted`, shifted relative to the go cue, histogrammed into 80 non-overlapping bins, and divided by 0.05 seconds to produce Hz. There is no smoothing or normalization.

ii.
```python
idx_lo = np.searchsorted(unit_spikes, abs_start)
idx_hi = np.searchsorted(unit_spikes, abs_end)
aligned = unit_spikes[idx_lo:idx_hi] - go_time
counts = np.histogram(aligned, bins=bin_edges)[0]
fr_trial[i, :] = counts / bin_width
```

iii. The notes say this implements the requested 50-ms firing rates while replacing the paper's 40-ms sliding histogram because the decoder instructions take precedence.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose classifier label is exactly `good` are retained; sessions with zero such units are skipped. No individual metric thresholds are applied.

ii.
```python
classification = f['units']['classification'][:]
good_mask = classification == b'good'
good_indices = np.where(good_mask)[0]
if n_good == 0:
    return None
```

iii. The agent connects this label to the five region-specific logistic-regression QC classifiers in the QC paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's go-cue timestamp is treated as zero; absolute bounds are `go-2.5` and `go+1.5`, and selected spikes are shifted by the go time before histogramming.

ii.
```python
go_time = go_cue_times[trial_idx]
abs_start = go_time + window_start
abs_end = go_time + window_end
aligned = unit_spikes[idx_lo:idx_hi] - go_time
```

iii. The notes state that all NWB timestamps are absolute and that the decoder task explicitly requires go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 50-ms, non-overlapping bins over `[-2.5, 1.5]`, yielding 80 samples. Raw spike events are temporally binned once; no further rebinning occurs.

ii.
```python
BIN_WIDTH = 0.05
N_TIMEBINS = int(round((WINDOW_END - WINDOW_START) / BIN_WIDTH))
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
```

iii. The agent explicitly chose the task specification over the reference analysis's 40-ms sliding window and 3.4-ms stride.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times` and each trial's `go_start_times`; the last sample onset at or before the go cue is selected.

ii.
```python
sample_starts_all = f['acquisition']['BehavioralEvents']['sample_start_times']['timestamps'][:]
ss_idx = np.searchsorted(sample_starts_all, go_time, side='right') - 1
```

iii. The notes map sample-start events to tone onset. Selecting the last preceding event accommodates repeated sample epochs after early licking.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue, then subtracted from every go-relative bin center. If no preceding sample exists, the agent substitutes `-1.85` seconds.

ii.
```python
tone_onset_rel = sample_starts_all[ss_idx] - go_time
time_from_tone = bin_centers - tone_onset_rel
```
```python
else:
    tone_onset_rel = -1.85
```

iii. The agent describes this as continuous elapsed time from tone onset. The fallback is an undocumented defensive default rather than a data-derived value.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 bin centers relative to the same trial go cue as the neural histogram.

ii.
```python
bin_centers = np.linspace(WINDOW_START + BIN_WIDTH/2,
                          WINDOW_END - BIN_WIDTH/2, n_bins)
time_from_tone = bin_centers - tone_onset_rel
```

iii. The notes say inputs and neural activity share the go-cue-aligned bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, and `start_time`, plus the go timestamp.

ii.
```python
photostim_onset_trial = trials['photostim_onset'][:]
photostim_duration_trial = trials['photostim_duration'][:]
trial_start_times = trials['start_time'][:]
```

iii. The notes map those NWB fields directly to a time-varying binary input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. `N/A` trials remain all zero. Otherwise onset strings and durations are converted to floats, onset is made absolute from trial start and then relative to go, and bin centers in the half-open stimulation interval are set to one.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
ps_rel_end = ps_rel_start + ps_dur_float
photostim_binary = ((bin_centers >= ps_rel_start) &
                    (bin_centers < ps_rel_end)).astype(np.float32)
```

iii. The agent says the decoder requires the light state at every time point, rather than a trial-level stimulation flag.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Stimulation bounds are converted to go-relative seconds and tested at the exact neural bin centers.

ii.
```python
ps_rel_start = (trial_start + ps_onset_float) - go_time
photostim_binary = ((bin_centers >= ps_rel_start) & ...)
```

iii. The shared bin centers and go cue provide alignment without interpolation.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent derives choice solely from `trial_instruction`, coding instructed-left as 0 and everything else as 1. It does not use outcome to reverse the side on misses or add no-lick for ignores.

ii.
```python
choice = 0 if trial_instruction[trial_idx] == b'left' else 1
```

iii. The notes explicitly map `trial_instruction` to choice and call it left/right. No justification addresses the distinction between instructed direction and the animal's actual response.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The binary value is repeated across all 80 bins, with output labels only `left` and `right`.

ii.
```python
output_trial[0, :] = choice
```
```python
['left', 'right'],
```

iii. The agent treats choice as a per-trial categorical output and repeats it to share the time-varying output tensor shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` byte strings.

ii.
```python
outcome = trials['outcome'][:]
out = outcome[trial_idx]
```

iii. The requested categories already exist in NWB, so the notes describe a direct mapping.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` map to 0, 1, and 2 and are repeated over time. Unexpected values silently map to 0.

ii.
```python
if out == b'ignore': outcome_val = 0
elif out == b'miss': outcome_val = 1
elif out == b'hit': outcome_val = 2
else: outcome_val = 0
output_trial[1, :] = outcome_val
```

iii. The mapping and repetition follow the requested categorical order and per-trial nature.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read from the trial-table `early_lick` field.

ii.
```python
early_lick = trials['early_lick'][:]
```

iii. The notes identify this as an explicit NWB field and retain these trials because early lick is a requested output.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `early` maps to 1 and every other value maps to 0; the value is repeated across bins.

ii.
```python
early = 1 if early_lick[trial_idx] == b'early' else 0
output_trial[2, :] = early
```

iii. The notes map no/yes to 0/1 and treat it as a per-trial variable.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses the side-camera tongue tracking series: column 1 for y position, column 2 for tracking likelihood, and the series timestamps.

ii.
```python
tongue_data = f['acquisition']['BehavioralTimeSeries']['Camera0_side_TongueTracking']['data'][:]
tongue_timestamps = ...['timestamps'][:]
y_slice = tongue_data[idx_start:idx_end, 1]
lk_slice = tongue_data[idx_start:idx_end, 2]
```

iii. The notes identify `Camera0_side_TongueTracking` as the approximately 300-Hz source for x, y, and likelihood.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For each retained trial, frames with likelihood greater than 0.1 are averaged within each 50-ms bin. The 40th/60th percentiles are then computed per session from all non-NaN bin means across retained trials. Missing series produces all NaNs and thresholds `(0, 0)`.

ii.
```python
high_conf = lk_slice > 0.1
bin_mask = ... & high_conf
tongue_y_binned[b] = np.mean(y_slice[bin_mask])
```
```python
p40 = np.percentile(np.array(all_tongue_y), 40)
p60 = np.percentile(np.array(all_tongue_y), 60)
```

iii. The notes justify likelihood filtering and per-session percentiles, but state only that valid tracking uses likelihood above 0.1. They do not justify limiting the percentile population to retained trial windows rather than the whole session.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Valid values below p40 become 0, values from p40 through p60 become 1, and values above p60 become 2. The array is initialized to 1, so bins with no visible tongue are incorrectly left in the middle category; no fourth class is declared.

ii.
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
valid_mask = ~np.isnan(tongue_y)
tongue_y_disc[valid_mask & (tongue_y < p40)] = 0
tongue_y_disc[valid_mask & (tongue_y >= p40) & (tongue_y <= p60)] = 1
tongue_y_disc[valid_mask & (tongue_y > p60)] = 2
```

iii. The notes claim the output follows per-session percentile discretization but overlook the instruction that category 3 represents not visible.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are selected around each absolute go cue, converted to go-relative time, and averaged using the same 80 edges as neural activity. The initial search range is padded by one bin, but exact masks restrict contributions to the intended edges.

ii.
```python
ts_slice = tongue_timestamps[idx_start:idx_end] - go_time
bin_mask = (ts_slice >= bin_edges[b]) & (ts_slice < bin_edges[b+1]) & high_conf
```

iii. The notes say `searchsorted` accelerates camera alignment and that the shared go-relative 50-ms grid synchronizes behavior and neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Empty spike arrays drop the session; sessions with no good units or fewer than two valid trials are dropped. Missing tongue series and empty tongue bins become NaN internally, but are ultimately encoded as tongue class 1. A missing preceding tone gets a hard-coded `-1.85`-second fallback. Unexpected outcome values map to ignore. Files are explicitly closed on handled early returns.

ii.
```python
if len(spike_times_flat) == 0:
    f.close(); return None
```
```python
else:
    tongue_y_trials = [np.full(n_bins, np.nan) for _ in trial_indices]
```
```python
tongue_y_disc = np.ones(n_bins, dtype=np.int64)
```

iii. The notes frame trial/session dropping as protection against absent neural coverage. They mention validation but do not identify the loss of the explicit not-visible tongue category or justify the tone/outcome fallbacks.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large NWB arrays and computing firing rates dominate; tongue tracking is another substantial pass. The code times loading, firing rates, inputs, tongue tracking, outputs, and whole sessions separately.

ii.
```python
print(f"    Data loading: {t1-t0:.1f}s")
print(f"    Firing rate computation: {t2-t1:.1f}s")
print(f"    Tongue tracking: {t4-t3:.1f}s")
```

iii. The notes report about seven seconds per session after optimizing spike searches and pre-extraction, identifying spike processing and I/O as primary costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Neural processing has nested trial-by-unit loops and calls `np.histogram` per populated unit/trial. Tongue processing loops over trials and then all 80 bins. Input/output construction also loops over trials. Brain-region conversion repeatedly performs linear list lookup.

ii.
```python
for trial_idx in trial_indices:
    for i, unit_spikes in enumerate(good_spike_times):
        counts = np.histogram(aligned, bins=bin_edges)[0]
```
```python
for b in range(n_bins):
    bin_mask = ...
```

iii. The agent says `searchsorted` and pre-extracting unit spikes were optimizations, but does not discuss flattening all trial edges as the reference does or vectorizing tongue bin aggregation.

## 10-c. What processing does the code repeat multiple times?

i. It searches each unit's spikes separately for every trial, constructs the same bin grid in both neural and tongue functions, loops over selected trials separately for neural, inputs, tongue, and outputs, and repeatedly uses `brain_regions.index` while assembling indices.

ii.
```python
bin_edges = np.linspace(window_start, window_end, n_bins + 1)
```
appears in both `compute_firing_rates_fast` and `get_tongue_y_for_trials`.

iii. The notes emphasize pre-extracting spikes and faster searches but do not acknowledge these repeated passes and computations.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `is_good_trials` only to obtain its second dimension, reads `max_spike_time`/`min_spike_time` for a heuristic, computes session performance and per-side counts used only to discard sessions, calculates/stores `correct_rate` although assembly does not save it, and builds all-unit brain-region labels before retaining good-unit labels. Optional plotting also computes summaries solely for diagnostics.

ii.
```python
n_recorded_trials = f['units']['is_good_trials'].shape[1]
correct_rate = hits / (hits + misses)
return {..., 'correct_rate': correct_rate}
```
```python
for i in range(n_total):
    ...
    unit_regions.append(region)
good_regions = [unit_regions[i] for i in range(n_total) if good_mask[i]]
```

iii. The agent presents performance calculations as required session curation and plotting as an optional sanity check. Relative to the reference conversion, those session filters are not only discarded computations but also lead to an unjustified loss of data.
