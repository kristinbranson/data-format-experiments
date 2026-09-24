# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every `sub-*` directory under `/app/data`, then every `.nwb` file in each directory. It opens each file with `NWBHDF5IO`; one file is treated as one session.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
nwb = NWBHDF5IO(nwb_path, 'r').read()
```

iii. The notes identify 28 subject directories and 174 NWB session files and explain that NWB is the available representation of the published data.

## 1-b. How are the data split into subjects?

i. The containing directory name (for example, `sub-440956`) is used as the subject ID. A first-seen lookup assigns each retained session a subject index.

ii.
```python
for i, (subject_id, nwb_path) in enumerate(nwb_files):
    if subject_id not in subjects_seen:
        subjects_seen[subject_id] = len(subjects_seen)
        subjects_list.append(subject_id)
```

iii. The notes describe the directory layout as organized by subject and report 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; retained session results are appended in sorted subject/file order. Sessions with no retained neurons or fewer than two retained trials are skipped.

ii.
```python
result = process_session(nwb_path, subject_id)
if result is None:
    continue
all_sessions.append(result)
```

iii. The notes explicitly state that each NWB file is a session and use the expected 174 files minus one un-QC'd file to justify 173 retained sessions.

## 1-d. How are the data split into trials?

i. Trial-table row indices define trials. The same index is used for go cues and trial labels, and every retained index produces one neural/input/output item.

ii.
```python
trial_indices = np.where(trial_mask)[0]
for ti in trial_indices:
    go_cue = go_start_times[ti]
```

iii. The AI found one behavioral trial table and aligned event arrays in each NWB file; it implicitly assumes one go cue per trial.

## 1-e. How are trials filtered based on quality controls?

i. It intersects `auto_water == 0`, `free_water == 0`, and recording coverage inferred from every distinct-length `obs_intervals` array among retained units. Coverage means a trial start is within one second of an observation start. Early-lick, photostimulation, and ignore trials are retained; sessions with fewer than two trials are dropped.

ii.
```python
covered = np.abs(trial_starts[:, None] - obs_starts[None, :]).min(axis=1) < 1.0
valid_trials &= covered
trial_mask = (auto_water == 0) & (free_water == 0) & recording_mask
```

iii. The notes say early lick, photostimulation, and ignore must remain because they are requested variables, while auto/free-water trials are behavioral confounds. `obs_intervals` was added after sample validation exposed trials without neural recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from `nwb.units['spike_times']` for units whose `classification` is `good` and whose anatomical annotation maps to one of 14 coarse regions. Trial go-cue timestamps provide alignment.

ii.
```python
classification = nwb.units['classification'][:]
good_mask = np.array([c == 'good' for c in classification])
st = nwb.units['spike_times'][idx]
```

iii. The notes identify spike times as absolute NWB times and `classification == 'good'` as the classifier-based QC verdict.

## 2-b. How is the `neural` data processed?

i. For every retained trial and neuron, spikes in the four-second window are assigned to non-overlapping bins, counted, and divided by 0.05 seconds to produce Hz. There is no smoothing or normalization.

ii.
```python
mask = (st >= abs_start) & (st < abs_end)
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
np.add.at(fr[i], bin_indices, 1.0)
fr /= bin_size
```

iii. The notes explain that the task-required 50-ms non-overlapping bins override the paper's 40-ms sliding-window analysis.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == 'good'` and a fine annotation recognized by the hand-written coarse-region mapper. No firing-rate threshold is applied; sessions with no surviving neurons are removed.

ii.
```python
good_unit_indices = np.where(good_mask)[0]
good_unit_indices = good_unit_indices[valid_neuron_mask]
if n_neurons == 0:
    return None
```

iii. The AI ties `good` to the QC white paper, says histology is required, and regards the paper's 2-Hz filter as analysis-specific. It reports that all 69,453 good units ultimately survived mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The go cue is added to -2.5 and +1.5 seconds to form absolute window bounds; absolute spike times are binned relative to the lower bound.

ii.
```python
abs_start = go_cue_time + t_start
abs_end = go_cue_time + t_end
bin_indices = ((spikes_in_window - abs_start) / bin_size).astype(int)
```

iii. The notes establish that spikes and behavioral events share an absolute NWB clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output contains 80 non-overlapping 50-ms bins from -2.5 to +1.5 seconds. Raw spike times are binned directly; no later rebinning occurs.

ii.
```python
BIN_SIZE_S = 0.05
N_TIMEBINS = int((T_END - T_START) / BIN_SIZE_S)
```

iii. This exactly follows the decoder task, deliberately replacing the reference analysis's denser sliding histogram.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `BehavioralEvents/sample_start_times`, the trial's `go_start_times` entry, and the common bin centers.

ii.
```python
sample_start_times = events.time_series['sample_start_times'].timestamps[:]
valid = sample_start_times[sample_start_times < go_cue_time]
return valid[-1]
```

iii. The last sample before go is selected because an early lick may replay the sample epoch.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The last preceding sample time is subtracted from every absolute neural-bin center. If no preceding sample exists, a hard-coded onset 1.85 seconds before go is substituted.

ii.
```python
tone_rel = tone_onset_time - go_cue_time
time_from_tone = bin_centers - tone_rel
```

iii. The notes justify selecting the last sample replay; the fallback is defensive code and is not separately justified.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the exact same 80 go-relative bin centers used for neural activity.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
```

iii. Both streams use the shared go cue and bin constants.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute timestamps from `BehavioralEvents/photostim_start_times` and `photostim_stop_times` rather than the trial-table onset/duration fields.

ii.
```python
photostim_start_abs = events.time_series['photostim_start_times'].timestamps[:]
photostim_stop_abs = events.time_series['photostim_stop_times'].timestamps[:]
```

iii. The notes identify these event streams as the absolute photostimulation intervals and plan a binary time-varying input.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. All stimulation intervals overlapping a trial window are selected. A bin is one when its center is between an interval's start and stop (both endpoints inclusive), otherwise zero.

ii.
```python
mask = (abs_centers >= start) & (abs_centers <= stop)
stim[mask] = 1.0
```

iii. The AI states that the task requires on/off status at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Go-relative bin centers are converted to absolute times and compared with absolute stimulation timestamps.

ii.
```python
abs_centers = bin_centers + go_cue_time
```

iii. The common absolute NWB clock is the alignment mechanism.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI uses only `trials['trial_instruction']`; it does not use outcome or lick timestamps, so this is instructed direction rather than actual lick choice.

ii.
```python
choice = 0 if instructions[ti] == 'left' else 1
```

iii. The notes' mapping table explicitly equates `trial_instruction` with choice and describes left/right coding; no justification addresses misses or ignored trials.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Instruction strings are encoded left=0/right=1 and repeated through all 80 bins. There is no no-lick category.

ii.
```python
full_output[0, :] = out_dict['choice']
'output_values': [['left', 'right'], ...]
```

iii. The AI treats choice as a per-trial categorical output and repeats all per-trial outputs to obtain a common time-varying shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trials table's `outcome` column.

ii.
```python
outcome_str = outcomes[ti]
```

iii. The raw categories exactly match those requested.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2 (unknown strings default to ignore), then the value is repeated across time.

ii.
```python
outcome_val = {'ignore': 0, 'miss': 1, 'hit': 2}.get(outcome_str, 0)
full_output[1, :] = out_dict['outcome']
```

iii. The notes document the same three-code mapping.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes from `trials['early_lick']`.

ii.
```python
early_lick = trials['early_lick'][:]
```

iii. The field explicitly records the requested per-trial state.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `'no early'` becomes 0 and every other value becomes 1; the result is repeated across all bins.

ii.
```python
early_val = 0 if early_lick[ti] == 'no early' else 1
full_output[2, :] = out_dict['early_lick']
```

iii. The notes specify no=0/yes=1 and retain early-lick trials because this is a decoder target.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 (y), column 2 (tracking likelihood), and timestamps from `Camera0_side_TongueTracking`.

ii.
```python
tongue_y_all = tongue_data_all[:, 1].astype(np.float32)
tongue_likelihood = tongue_data_all[:, 2].astype(np.float32)
```

iii. The notes identify the camera array as x, y, likelihood sampled near 300 Hz.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames below likelihood 0.9 become NaN. Visible frames are averaged within each 50-ms trial bin. Percentiles are then computed from visible bin means pooled only across retained trial windows in that session.

ii.
```python
y_in_window[like_in_window < 0.9] = np.nan
tongue_y_binned[has_data] = sums[has_data] / counts[has_data]
p40 = np.percentile(tongue_y_arr, 40)
```

iii. The notes cite DeepLabCut likelihood filtering and per-session discretization, though their planning text says “all valid time points” and does not explain the eventual 0.9 threshold.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible values below p40 are 0, p40 through p60 inclusive are 1, and above p60 are 2. Missing bins remain the array's initialized value 0; no required class 3 (“not visible”) exists.

ii.
```python
tongue_y_disc = np.zeros(N_TIMEBINS, dtype=np.float32)
tongue_y_disc[valid & (tongue_y_raw >= p40) & (tongue_y_raw <= p60)] = 1
tongue_y_disc[valid & (tongue_y_raw > p60)] = 2
```

iii. The notes aim for the requested 40/60 split but omit the fourth category throughout and even report output range `[0,2]` as correct.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames in `[go-2.5, go+1.5)` are assigned to the same 50-ms absolute-time intervals as spikes and averaged per bin.

ii.
```python
bin_edges = np.arange(n_bins + 1) * bin_size + t_start + go_cue_time
bin_indices = np.digitize(t_in_window, bin_edges) - 1
```

iii. The notes rely on the common absolute NWB timestamps and the same go-aligned window.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions without good/mapped units and trials without recording coverage are removed. A missing tone gets a nominal -1.85-s fallback. Low-confidence/missing tongue samples become NaN during averaging, but empty bins are ultimately mislabeled class 0. Unknown outcome strings silently become ignore, and unmapped anatomical labels cause neuron removal.

ii.
```python
if tone_onset is None:
    tone_onset = go_cue - 1.85
outcome_val = {...}.get(outcome_str, 0)
```

iii. The notes discuss the un-QC'd session, annotation mapping, recording gaps, and two all-zero trials, but do not recognize the missing-tongue category error.

## 10-a. What are the most time-consuming steps of the code?

i. Neural binning dominates because it loops over every retained trial and, inside that, every neuron while repeatedly scanning/filtering its spike vector. NWB/video reads and writing the 11.3-GB pickle are also substantial. The reported full conversion took about 2,301 seconds.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(spike_times_per_unit, go_cue, ...)
# inside compute_firing_rates:
for i, st in enumerate(spike_times_list):
```

iii. The notes estimate roughly 6.5 seconds per sample session and report about 38 minutes for the full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The outer trial-by-trial neural computation could bin all trial edges for each unit in one `searchsorted` call, as the human solution does. Tone lookup could also use one vectorized `searchsorted`; photostim and tongue processing retain per-trial loops.

ii.
```python
for ti in trial_indices:
    fr = compute_firing_rates(...)
    tone_onset = find_last_sample_before_go(...)
```

iii. The AI labels tongue bin aggregation “vectorized” within a trial but does not discuss vectorizing across trials or the expensive nested neural loop.

## 10-c. What processing does the code repeat multiple times?

i. For every trial it recomputes bin centers/edges, scans every neuron's spikes with a Boolean mask, filters the full sample-start array, tests all photostim intervals for overlap, and scans the full camera timestamp array with a Boolean mask.

ii.
```python
bin_centers = np.arange(n_bins) * bin_size + t_start + bin_size / 2
mask = (tongue_timestamps >= abs_start) & (tongue_timestamps < abs_end)
```

iii. No explicit justification is given; the code favors simple per-trial helper functions. This repetition explains much of the runtime difference from the reference implementation.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion, little computed data is wholly discarded, but it reads/stores labels for all trials before filtering, constructs dictionaries for delayed output assembly, and calculates/prints extensive summaries. Optional `--show-processing` additionally renders plots that are not part of the converted dataset. It also computes recording coverage across multiple unit interval patterns where the dataset's shared observation intervals suffice.

ii.
```python
output_trials_raw.append({'choice': choice, ..., 'tongue_y_raw': tongue_y_trial})
if args.show_processing and i < 2:
    plot_processing(...)
```

iii. Plotting is explicitly described as a validation aid; the summaries and intermediate dictionaries support sanity checks and assembly rather than downstream decoder fields.
