# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted `sub-*` directories, then sorted `*.nwb` files within each directory. Each NWB file is opened with `NWBHDF5IO`; behavior, fluorescence, neuropil, deconvolved traces, ROI metadata, timestamps, and session metadata are loaded. Every successfully processed file becomes one session, although sessions can be skipped by later quality gates.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
io = NWBHDF5IO(filepath, 'r')
nwb = io.read()
```

iii. The trajectory says the data are NWB behavior+ophys files for 11 subjects and that all available sessions should be traversed. It also says NWB contains the required neural and behavioral streams. The final documentation reports 152 sessions and 12,216 trials, consistent with the discovered dataset.

## 1-b. How are the data split into subjects?

i. A subject is defined by a `sub-*` directory. The stored subject name is the directory name without `sub-`; sessions receive its index in the accumulating `all_subjects` list. The NWB `subject_id` is also read and returned by session processing.

ii.
```python
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
subj_idx = all_subjects.index(subject_name)
...
subject_idx.append(subj_idx)
```

iii. The agent identified the 11 subject directories (`m3`, `m4`, `m7`, and `m11`–`m19` as available) and treated directory organization as the subject boundary.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Files are processed independently and their trial lists are appended as one entry in `neural`, `input`, and `output`. A session is discarded if it has fewer than five curated cells, fewer than three initially identified trials, or fewer than two retained trials.

ii.
```python
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
    if result is None:
        continue
    all_neural.append(result['neural'])
```

iii. The trajectory recognized one behavior+ophys NWB file per session. It justified pooling planes within a session from the paper’s statement that planes were pooled for most analyses.

## 1-d. How are the data split into trials?

i. Trial starts are all positive samples of `trial_start`. For each start, the end is the first later sample for which `teleport > 0`. The trial is retained at this stage only if at least one `scanning == 1` sample occurs between start and end. Per-trial arrays use the half-open slice `[start:end)`.

ii.
```python
start_indices = np.where(trial_start_signal > 0)[0]
end_indices = np.where(teleport_signal > 0)[0]
for s in start_indices:
    next_ends = end_indices[end_indices > s]
    if len(next_ends) > 0:
        e = next_ends[0]
        if np.any(scanning[s:e] == 1):
            trial_starts.append(s)
            trial_ends.append(e)
```

iii. The agent explored position ranges and trial statistics and concluded that trial start to teleport represents one traversal. Its later notes inconsistently describe trials as extending to the next `trial_start`, but the implemented decision is start-to-teleport.

## 1-e. How are trials filtered based on quality controls?

i. Boundaries are kept only if scanning occurs during the interval. Later, trials shorter than five samples are skipped. At session level, files with fewer than three boundary pairs are skipped, and processed sessions with fewer than two valid trials are skipped. This differs from the reference’s 50-sample trial cutoff.

ii.
```python
if np.any(scanning[s:e] == 1):
    trial_starts.append(s)
...
if len(trial_starts) < 3:
    return None
...
if n_timepoints < 5:
    continue
```

iii. The trajectory does not give an evidence-based reason for the five-sample cutoff or scanning gate. It emphasizes matching roughly 80 trials/session and minimum decoder viability; the code’s final `<2` check follows the target format’s requirement of at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural events are derived from NWB `ophys/Fluorescence` and `ophys/Neuropil`, pooled across planes. The NWB `Deconvolved` stream is loaded but is only used when `compute_own_dff=False`, which the full conversion does not request. ROI `iscell` and behavioral speed are used for cell filtering.

ii.
```python
fluor_planes = [np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys]
neuro_planes = [np.array(ophys['Neuropil'][k].data[:]) for k in plane_keys]
...
result = process_session(nwb_file, compute_own_dff=True)
```

iii. The agent explicitly investigated whether NWB `Deconvolved` represented the paper’s signal and ultimately chose to recompute dF/F and OASIS events to follow the paper rather than rely on Suite2p’s stored deconvolution.

## 2-b. How is the `neural` data processed?

i. Fluorescence is transposed neuron-by-time, masked to trial intervals, corrected by subtracting `0.7*Fneu`, and has `0.7` times each trial’s mean neuropil added back. Each trial is smoothed at sigma 15, subjected to 300-sample minimum then maximum filtering, converted to `(F-baseline)/abs(baseline)`, and smoothed at sigma 2. Curated/non-interneuron cells are then deconvolved with OASIS independently for every trial (`tau=.7`, fixed 15.5078125 Hz); failures fall back to positive dF/F. Multi-plane ROIs are concatenated.

ii.
```python
f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
f_[:, start:end] += NEUROPIL_COEF * np.nanmean(f_neu_[:, start:end], axis=1, keepdims=True)
f_smooth = nansmooth(f_[:, start:end], 15, axis=1)
baseline = ndimage.minimum_filter1d(f_smooth, 300, axis=-1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=-1)
dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
...
events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
```

iii. The trajectory repeatedly compared the implementation with the paper’s neuropil correction, maximin baseline, Gaussian smoothing, and OASIS pipeline. It fixed the window at 300 samples after noticing that exact value in the source. It accepted an interneuron-rate discrepancy as a pragmatic approximation. It did not account for the paper code’s session-specific `keep_teleports` behavior and chose per-trial deconvolution.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with `iscell[:,0]` true are candidates. For each such ROI, Pearson correlation is calculated between dF/F and speed at valid within-trial samples; cells with correlation greater than 0.5 are removed. Sessions with fewer than five curated cells are dropped. NaNs in retained trial events are changed to zero.

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
corr = np.corrcoef(cell_data, speed_data)[0, 1]
if corr > INTERNEURON_SPEED_CORR_THR:
    mask[idx] = False
```

iii. The paper describes manual Suite2p curation and removal of putative interneurons at `r > .5`. The agent observed about 3.8% removal versus the paper’s much lower reported mean, investigated alternatives, then retained the stated paper threshold despite the discrepancy.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural timepoints are sliced from the behavioral `trial_start` index through (but excluding) the first positive teleport index. Thus column zero is the trial-start-aligned neural sample; there is no interpolation or temporal shift.

ii.
```python
s, e = trial_starts[t], trial_ends[t]
neural_trial = events[:, s:e].copy()
```

iii. The agent states that neural and behavior arrays are already aligned at the imaging-frame rate, so common indices suffice for trial-start alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Metadata declares `1000/15.5078125 = 64.48 ms`. No rebinning or resampling is applied. A fixed effective rate of 15.5078125 Hz is used even for pooled multi-plane sessions.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. The agent found the effective imaging rate to be about 15.5 Hz and treated the NWB arrays as already sampled per plane at that rate. Its notes cite the expected ~64.5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from timestamps attached to the raw `position` behavioral time series, plus the detected trial start index.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The agent determined behavioral streams and neural samples share the imaging-rate time base and selected position timestamps as that time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp at the first sample of the trial is subtracted from every timestamp in the trial, producing seconds beginning at zero. No interpolation or rebinning occurs.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
input_trial[0, :] = time_from_start
```

iii. This directly implements the requested continuous time from trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Exactly the same `[s:e)` indices define timestamps and neural columns, and the resulting vector has `n_timepoints=e-s`. If total neural and behavioral lengths differ, both are truncated to the smaller length before trial extraction, though boundaries were computed before truncation.

ii.
```python
neural_trial = events[:, s:e].copy()
time_from_start = timestamps[s:e] - timestamps[s]
```

iii. The agent’s stated rationale is that behavior and imaging are already frame-aligned, with truncation handling observed off-by-one mismatches.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the raw behavioral `environment` time series, using its value at the detected trial start.

ii.
```python
environment = nwb_data['environment']
env_type = float(environment[s])
```

iii. Exploration showed environment values correspond to binary ENV1/ENV2 and are constant within trials.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The start value is converted to float and broadcast over the trial. Any negative/unknown value is silently changed to ENV1 (`0`).

ii.
```python
if env_type < 0:
    env_type = 0.0
input_trial[1, :] = env_type
```

iii. The agent justified ordinary values as direct ENV1/ENV2 coding, but supplied no data-driven justification for defaulting an unknown environment to ENV1.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the zero-based loop index over detected trials, not from the raw NWB `trial number` values (although that raw array is loaded and passed unused to boundary detection).

ii.
```python
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
```

iii. The agent discovered inconsistencies in raw trial-number/trial-start representations and treated the detected sequential trial order as the reliable within-session number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float and broadcast to every timepoint. If a short trial is later skipped, subsequent retained trials preserve their original indices, so gaps are possible.

ii.
```python
trial_num = float(t)
input_trial[2, :] = trial_num
```

iii. The agent describes this as zero-indexed trial number within session; no additional transformation was considered necessary.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from raw `Reward` timestamps, position timestamps that delimit each trial in real time, and additionally whether raw `reward_zone` was positive during that previous trial.

ii.
```python
reward_in_trial = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
entered_rz = np.any(rz_signal[trial_start:trial_end] > 0)
return int(reward_in_trial and entered_rz)
```

iii. The trajectory identifies `Reward` timestamps as reward delivery. The additional reward-zone-entry requirement was described as ensuring the mouse both received reward and entered the zone, though the reference uses reward delivery alone.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial outcome is precomputed as the conjunction above. For trial `t>0`, the value from `t-1` is broadcast across the trial. For the first trial, the agent hard-codes `1` (“assume rewarded before session”), contrary to the requested omitted=0 convention and reference choice of zero.

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
```

iii. No trajectory evidence supports the pre-session rewarded assumption. The code comment is the only justification.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from raw `position`, raw `reward_zone`, detected trial boundaries, and fixed zone ranges A `[80,130]`, B `[200,250]`, C `[320,370]`. Active-zone positions identify a label; missing labels are inferred from neighboring trials.

ii.
```python
rz_positions = pos_trial[rz_trial > 0]
mean_rz_pos = np.mean(rz_positions)
...
dist = abs(mean_rz_pos - zone_center)
```

iii. The agent examined active reward-zone samples and mapped their positions to the paper’s three known ranges. For omissions, it reasoned the zone could be inherited from nearby trials in the same session.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The mean position where `reward_zone>0` is assigned to the nearest zone center. Missing trial labels take the next observed label, otherwise the preceding label, and ultimately default to A. At each timepoint, signed distance is position minus the near boundary before the zone, zero inside, or position minus the far boundary after the zone.

ii.
```python
if position < rz_start:
    return position - rz_start
elif position > rz_end:
    return position - rz_end
else:
    return 0.0
```

iii. The signed-distance definition follows the requested reward-relative coordinate. The trajectory notes omission trials may lack an active reward-zone signal; it chose neighbor filling, rather than the reference’s probabilistic Viterbi sequence inference.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. A scalar loop assigns seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii.
```python
if d < -50: result[i] = 0
elif d < -10: result[i] = 1
elif d < 0: result[i] = 2
elif d == 0: result[i] = 3
elif d <= 10: result[i] = 4
elif d <= 50: result[i] = 5
else: result[i] = 6
```

iii. The agent directly transcribed the decoder-task thresholds.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural events use the same trial slice `[s:e)`, yielding one category per neural column with no interpolation.

ii.
```python
neural_trial = events[:, s:e].copy()
pos_trial = position[s:e]
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
```

iii. The agent considered behavioral and imaging arrays already aligned and handled only total-length truncation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw behavioral `position` time series sliced by trial boundaries.

ii.
```python
position = nwb_data['position']
pos_trial = position[s:e]
```

iii. The agent identified the position stream as centimeters along the 450-cm corridor.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0,449.999]`, divided by 90, floored to an integer, and clipped to classes 0–4.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. The agent used five equal 90-cm bins over a 450-cm track. Clipping was intended to absorb teleport/out-of-range values into endpoint bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Categories are `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, and `[360,450]` after clipping. Exact internal boundaries enter the upper bin.

ii.
```python
bin_size = TRACK_LENGTH / n_bins
result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. This follows the instruction to split the 450-cm track into five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural activity are sliced with identical `[s:e)` indices, with no resampling.

ii.
```python
neural_trial = events[:, s:e].copy()
pos_trial = position[s:e]
```

iii. The agent relied on the shared frame-level time base.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw behavioral `lick` time series.

ii.
```python
lick_cumul = nwb_data['lick']
lick_trial = lick_cumul[s:e].copy()
```

iii. During exploration the agent described this signal as cumulative/possibly count-valued and concluded positive values represent lick occurrence.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Every raw value greater than zero is assigned 1; other values are 0. Despite documentation mentioning a difference operation, the code does not differentiate the series.

ii.
```python
lick_binary = (lick_trial > 0).astype(float)
output_trial[3, :] = lick_binary.astype(int)
```

iii. The stated goal was to convert a count-like lick stream to the required binary no/yes output. The implementation’s positive-value threshold matches the reference, though the agent’s later prose incorrectly says it computes a diff.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Raw lick samples and neural columns use the same `[s:e)` slice.

ii.
```python
neural_trial = events[:, s:e].copy()
lick_trial = lick_cumul[s:e].copy()
```

iii. Shared sampling and indices were assumed to provide alignment.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from raw position at samples where raw `reward_zone>0`, plus fixed zone ranges and neighboring trials when direct observations are absent.

ii.
```python
rz_positions = pos_trial[rz_trial > 0]
mean_rz_pos = np.mean(rz_positions)
...
labels[i] = labels[j]
```

iii. The agent found that active reward-zone samples occur near the known A/B/C locations and used this as the observable label; omissions motivated sequence-based filling from neighbors.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The mean active-zone position is assigned to the closest fixed zone center. Missing labels search forward first, then backward; unresolved labels default to A. Labels are encoded A=0, B=1, C=2 and broadcast across time.

ii.
```python
for zone, (start, end) in REWARD_ZONES.items():
    dist = abs(mean_rz_pos - (start + end) / 2)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The agent justified nearest-zone assignment from known paper locations. Its neighbor heuristic was a pragmatic omission-trial solution; it did not reproduce the reference Viterbi model that balances emissions with rare zone switches.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the raw `Reward` timestamps and position timestamps defining the trial interval, with raw `reward_zone` used as an additional gate.

ii.
```python
reward_in_trial = np.any((reward_timestamps >= t_start) & (reward_timestamps <= t_end))
entered_rz = np.any(rz_signal[trial_start:trial_end] > 0)
```

iii. The agent identified reward timestamps as the delivery record, but chose to require zone entry as corroboration.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 only when a reward timestamp lies inclusively between its start and end timestamps and `reward_zone` is positive somewhere in the sample slice; otherwise it is 0. This scalar is broadcast over the trial.

ii.
```python
return int(reward_in_trial and entered_rz)
...
output_trial[5, :] = rew_outcome
```

iii. The agent’s rationale was that a rewarded trial should contain both delivery and reward-zone entry. The reference decision uses presence of a Reward event alone and validates its timestamp-to-frame alignment.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length mismatches are truncated to the shorter length; event NaNs are replaced with zero; OASIS exceptions fall back to nonnegative dF/F; missing zone labels are filled from the next then previous observed trial and finally A; negative environments become ENV1. Invalid/too-small sessions and very short trials are skipped. There are few assertions, and trial boundaries are computed before length truncation.

ii.
```python
min_len = min(n_behav, n_neural)
F = F[:, :min_len]
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
...
rz_label = 'A'  # fallback
```

iii. The agent explicitly observed off-by-one multi-plane length differences and chose truncation. Neighbor fill was intended for omission trials. Other defaults/fallbacks are defensive but were not supported by reference behavior or explicit trajectory evidence.

## 13-a. What are the most time-consuming steps of the code?

i. Loading every large NWB stream, repeated per-trial maximin filtering, per-trial OASIS deconvolution, retaining the full dataset in memory, and serializing the roughly 19.5-GB pickle dominate. Decoder training was also lengthy but is outside conversion itself.

ii.
```python
fluor_planes = [np.array(ophys['Fluorescence'][k].data[:]) for k in plane_keys]
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    events_trial = deconvolve_oasis(...)
...
pickle.dump(data, f)
```

iii. The trajectory shows full conversion running as a long background job and avoids rerunning sample conversion because it would process everything again. It reports a ~19.5-GB full output, supporting I/O, neural preprocessing, and serialization as the expensive work.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Scalar loops in distance binning, speed binning, signed-distance computation, interneuron correlations, reward-outcome detection, reward-zone labeling/filling, and repeated scans for the next teleport or neighbor label could be vectorized. Trial loops are partly inherent because lengths differ, but whole-session masks/digitization and correlation matrices could reduce Python overhead.

ii.
```python
for i, d in enumerate(distances): ...
for i, s in enumerate(speeds): ...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end) for p in pos_trial])
for idx in cell_indices:
    corr = np.corrcoef(...)
```

iii. The agent did not explicitly analyze vectorization in the trajectory. Its code favors clear scalar/per-trial loops; these opportunities are inferred from the implementation.

## 13-c. What processing does the code repeat multiple times?

i. Trial intervals are traversed repeatedly to mask F/Fneu, compute baselines, smooth dF/F, deconvolve, determine zone labels, determine reward outcomes, and build arrays. Reward-zone signals and positions are sliced more than once. It also loads NWB `Deconvolved`, reward amounts, autoreward, and several metadata arrays even though the selected full path does not use all of them.

ii.
```python
for start, end in zip(trial_starts, trial_ends):  # mask
...
for start, end in zip(trial_starts, trial_ends):  # baseline
...
for start, end in zip(trial_starts, trial_ends):  # smoothing
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):  # OASIS
```

iii. The trajectory focused on correctness and full-run validation rather than consolidating passes. It also notes `--sample-only` actually invokes the same complete conversion, an explicit source of repeated work if used.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The full conversion loads and concatenates NWB `Deconvolved` traces but then recomputes events and discards the stored traces. It loads `reward_data`, `autoreward`, `trial_number`, `plane_idx`, and some session metadata that do not affect decoder arrays (plane indices are returned but all neurons are labeled CA1). The sample dataset and extensive sanity checks are extra to the requested full pickle. `--sample-only` still performs and saves the full conversion.

ii.
```python
deconv_planes = [np.array(ophys['Deconvolved'][k].data[:]) for k in plane_keys]
...
result = process_session(nwb_file, compute_own_dff=True)
...
'reward_data': np.array(bts.time_series['Reward'].data[:]),
```

iii. The agent initially compared stored deconvolution with recomputed activity, which explains loading it during development, but left that cost in the final path. The trajectory itself recognizes that sample-only mode does not limit processing and avoids rerunning it.
