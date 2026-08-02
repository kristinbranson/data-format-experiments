# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI iterates over every `sub-*` directory under `data/`, then over every `.nwb` file in each subject directory. Each NWB file is opened with `pynwb`, and the script loads many behavioral series plus fluorescence, neuropil, deconvolved activity, and ROI metadata into memory before later splitting them into trials.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
nwb_files = sorted(glob(os.path.join(subj_path, '*.nwb')))
...
io = NWBHDF5IO(filepath, 'r')
nwb = io.read()
...
data = {
    'position': np.array(bts.time_series['position'].data[:]),
    'speed': np.array(bts.time_series['speed'].data[:]),
    ...
    'fluorescence': fluorescence,
    'neuropil': neuropil_data,
    'deconvolved': deconvolved,
}
```

iii. The module docstring says the pipeline should "Load NWB files for each session" and `CONVERSION_NOTES.md` says the data are NWB files from 11 mice and 152 sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `sub-*` directory names, with the `sub-` prefix stripped. The subject id stored inside each NWB file is also read, but the output `subjects` list is driven by directory names.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)
                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])
...
subject_name = subj_dir.replace('sub-', '')
if subject_name not in all_subjects:
    all_subjects.append(subject_name)
...
'subject_id': nwb.subject.subject_id if nwb.subject else None,
```

iii. `CONVERSION_NOTES.md` explicitly lists the 11 mouse ids and describes them as the source dataset.

## 1-c. How are the data split into sessions?

i. Each `.nwb` file is treated as one session. Successful `process_session(...)` calls become one entry in `data['neural']`, `data['input']`, and `data['output']`.

ii.
```python
for nwb_file in nwb_files:
    result = process_session(nwb_file, compute_own_dff=True)
    if result is None:
        continue
    all_neural.append(result['neural'])
    all_input.append(result['input'])
    all_output.append(result['output'])
```

iii. The script docstring says "Load NWB files for each session," and the notes report 152 total sessions.

## 1-d. How are the data split into trials?

i. Implemented behavior: trial starts are indices where `trial_start > 0`; trial ends are the first later index where `teleport > 0`; trials are only kept if `scanning` is 1 somewhere between the chosen start and end. This means the code does not use teleport rising edges. `CONVERSION_NOTES.md` conflicts with the code here: it says trial boundaries are from one `trial_start` to the next or recording end.

ii.
```python
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_numbers, scanning):
    start_indices = np.where(trial_start_signal > 0)[0]
    end_indices = np.where(teleport_signal > 0)[0]
    ...
    for s in start_indices:
        next_ends = end_indices[end_indices > s]
        if len(next_ends) > 0:
            e = next_ends[0]
            if np.any(scanning[s:e] == 1):
                trial_starts.append(s)
                trial_ends.append(e)
```

iii. The notes justify trial splitting by saying trials are defined by the `trial_start` signal. The trajectory shows the agent spent time inspecting `reward_zone`, `trial_start`, and trial structure, but the final code uses the first later positive `teleport` sample and a `scanning` gate.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than 5 samples are dropped. Entire sessions are also dropped if they have fewer than 3 detected trials or fewer than 2 valid trials after filtering. Trials whose span has no `scanning == 1` are excluded earlier by the boundary finder.

ii.
```python
if len(trial_starts) < 3:
    print(f"    Skipping: only {len(trial_starts)} trials")
    return None
...
if n_timepoints < 5:
    continue
...
if valid_trial_count < 2:
    print(f"    Skipping: only {valid_trial_count} valid trials")
    return None
```

iii. The justification is implicit in code comments and decoder-format constraints. There is no matching explanation in `CONVERSION_NOTES.md` beyond reporting successful verification.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final `neural` matrices are derived from raw `Fluorescence` and `Neuropil` NWB series, then deconvolved by the script. The code also loads NWB `Deconvolved`, but with `compute_own_dff=True` that precomputed array is not used for the final output.

ii.
```python
fluorescence = np.array(ophys['Fluorescence']['plane0'].data[:])
neuropil_data = np.array(ophys['Neuropil']['plane0'].data[:])
deconvolved = np.array(ophys['Deconvolved']['plane0'].data[:])
...
F = nwb_data['fluorescence'].T
Fneu = nwb_data['neuropil'].T
deconv_nwb = nwb_data['deconvolved'].T
```

iii. The script docstring and notes say the intent was to "Compute dF/F from raw fluorescence" and then "Deconvolve with OASIS."

## 2-b. How is the `neural` data processed?

i. The AI recomputes neural activity instead of using the stored deconvolved traces. It masks fluorescence to trial periods, subtracts `0.7 * neuropil`, adds back the per-trial neuropil mean, computes a maximin baseline, converts to dF/F, smooths with a Gaussian kernel, filters cells, and deconvolves each trial with OASIS (or a nonnegative fallback).

ii.
```python
f_[:, nanmask] = f_[:, nanmask] - NEUROPIL_COEF * f_neu_[:, nanmask]
...
baseline = ndimage.minimum_filter1d(f_smooth, window_size, axis=-1)
baseline = ndimage.maximum_filter1d(baseline, window_size, axis=-1)
dff[:, start:end] = (f_[:, start:end] - baseline) / np.abs(baseline)
...
dff[:, start:end] = nansmooth(dff[:, start:end], 2, axis=1)
...
events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
```

iii. `CONVERSION_NOTES.md` describes this as matching `preprocessing.py`. The trajectory shows the agent explicitly choosing to keep its own dF/F and deconvolution pipeline after worrying about interneuron filtering rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells are first restricted to curated ROIs where `iscell[:, 0]` is true, then a second filter removes putative interneurons whose activity correlates with running speed above 0.5.

ii.
```python
iscell = nwb_data['iscell'][:, 0].astype(bool)
...
cell_mask = filter_interneurons(dff, speed, iscell)
...
if corr > INTERNEURON_SPEED_CORR_THR:
    mask[idx] = False
```

iii. The notes state this was meant to match the paper’s interneuron exclusion criterion, and the trajectory repeatedly discusses the unexpectedly high fraction of removed cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to trial start by slicing the recomputed event matrix between each detected trial start and trial end. No further time shift is applied within each trial.

ii.
```python
for t in range(len(trial_starts)):
    s, e = trial_starts[t], trial_ends[t]
    ...
    neural_trial = events[:, s:e].copy()
```

iii. The script docstring says "Align all data to trial start," and the metadata says the temporal alignment event is "start of trial."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The script uses a fixed frame rate of 15.5078125 Hz and therefore a fixed bin size of `1000 / 15.5078125` ms, about 64.5 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` reports 15.5078125 Hz imaging and 64.48 ms bins. The trajectory also mentions multi-plane frame-rate handling, although the final code keeps a single fixed rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the `position` timestamps loaded as `nwb_data['timestamps']`.

ii.
```python
'timestamps': np.array(bts.time_series['position'].timestamps[:]),
...
time_from_start = (timestamps[s:e] - timestamps[s])
```

iii. The notes justify this generically by saying data were aligned to trial start; they do not discuss why `position` timestamps were chosen over other behavioral timestamps.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp in that slice is subtracted from all timestamps in the same slice.

ii.
```python
time_from_start = (timestamps[s:e] - timestamps[s])
...
input_trial[0, :] = time_from_start
```

iii. This follows directly from the requirement to align by trial start. No separate justification is given.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The same `[s:e]` indices are used for timestamps and neural events. If behavioral and neural arrays differ in overall length, both are truncated to the minimum length first.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    F = F[:, :min_len]
    ...
    nwb_data['timestamps'] = nwb_data['timestamps'][:min_len]
...
time_from_start = (timestamps[s:e] - timestamps[s])
neural_trial = events[:, s:e].copy()
```

iii. `CONVERSION_NOTES.md` explicitly calls out off-by-one truncation for some sessions.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the NWB behavioral `environment` time series.

ii.
```python
'environment': np.array(bts.time_series['environment'].data[:]),
...
env_type = float(environment[s])
```

iii. The notes describe environment as a 0/1 timeseries corresponding to ENV1 and ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The script takes the first environment value in the trial, treats it as the per-trial environment label, and broadcasts it across all time bins of that trial. If the first value is negative, it is replaced with 0.

ii.
```python
env_type = float(environment[s])
if env_type < 0:
    env_type = 0.0
...
input_trial[1, :] = env_type
```

iii. The notes say environment is per trial and binary. The default-to-0 fallback is only justified implicitly as defensive handling.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is not read from the stored `trial number` series. It is derived from the loop index `t` after trial segmentation.

ii.
```python
for t in range(len(trial_starts)):
    ...
    trial_num = float(t)
```

iii. The notes say trial number is "0-indexed trial number within session."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is cast to float and broadcast across the full trial duration.

ii.
```python
trial_num = float(t)
...
input_trial[2, :] = trial_num
```

iii. No separate justification is given beyond the decoder input definition.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Previous trial outcome is derived from the list `is_rewarded`, and `is_rewarded` itself is computed from `Reward` timestamps, behavioral timestamps, trial boundaries, and whether `reward_zone > 0` occurred within the trial.

ii.
```python
reward_timestamps = nwb_data['reward_timestamps']
...
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
rz_trial = rz_signal[trial_start:trial_end]
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
...
prev_outcome = float(is_rewarded[t - 1])
```

iii. `CONVERSION_NOTES.md` says reward outcome was detected from `Reward` timestamps relative to trial boundaries. The extra reward-zone gate is visible only in code.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For each trial after the first, the script copies the previous element of `is_rewarded` and broadcasts it across the trial. For the first trial, it hard-codes `prev_outcome = 1.0`, assuming the previous trial was rewarded.

ii.
```python
if t == 0:
    prev_outcome = 1.0  # assume rewarded before session
else:
    prev_outcome = float(is_rewarded[t - 1])
...
input_trial[3, :] = prev_outcome
```

iii. No explicit justification appears in the notes. This seems to be an undocumented assumption in the implementation.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from per-trial `position` values and a per-trial reward-zone label inferred from the `reward_zone` signal and position. The label is chosen from the mean position where `reward_zone > 0`, then missing labels are filled from neighboring trials.

ii.
```python
rz_positions = pos_trial[rz_trial > 0]
mean_rz_pos = np.mean(rz_positions)
...
for zone, (start, end) in REWARD_ZONES.items():
    zone_center = (start + end) / 2
    dist = abs(mean_rz_pos - zone_center)
...
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
```

iii. `CONVERSION_NOTES.md` says reward zone is determined from positions where the `reward_zone` signal is active, with omission trials inferred from neighbors. The trajectory shows the agent inspecting this variable carefully.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each timepoint in the trial, the script computes signed distance from the current position to the reward-zone interval: negative before the zone, zero inside it, positive after it.

ii.
```python
def distance_to_reward_zone(position, rz_start, rz_end):
    if position < rz_start:
        return position - rz_start
    elif position > rz_end:
        return position - rz_end
    else:
        return 0.0
...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. The notes explicitly describe this signed-distance convention.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The AI hand-coded the seven bins with chained inequalities. Exact `0` is its own class; `0 < d <= 10` maps to class 4; `10 < d <= 50` maps to class 5; values above 50 map to class 6.

ii.
```python
if d < -50:
    result[i] = 0
elif d < -10:
    result[i] = 1
elif d < 0:
    result[i] = 2
elif d == 0:
    result[i] = 3
elif d <= 10:
    result[i] = 4
elif d <= 50:
    result[i] = 5
else:
    result[i] = 6
```

iii. The docstring says these bins follow the instruction categories. No separate discussion of boundary conventions appears.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. It is computed from the same per-trial `[s:e]` slices used for neural data, then stored as a time-varying row in the per-trial output matrix.

ii.
```python
neural_trial = events[:, s:e].copy()
pos_trial = position[s:e]
...
output_trial[0, :] = dist_binned
```

iii. The alignment follows the script’s trial slicing strategy and the notes’ statement that all data are aligned to trial start.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived from the `position` behavioral time series.

ii.
```python
'position': np.array(bts.time_series['position'].data[:]),
...
pos_trial = position[s:e]
```

iii. The notes describe this as the mouse’s position on the 0-450 cm linear track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position values are clipped into `[0, 450)` and then binned.

ii.
```python
pos_clipped = np.clip(pos_trial, 0, TRACK_LENGTH - 0.001)
pos_binned = discretize_position(pos_clipped, n_bins=5)
```

iii. The trajectory mentions concern about negative teleport-zone positions and a decision to clip into the corridor range.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The script uses 5 equal-width bins over 0-450 cm, i.e. 90 cm bins: `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360,450)`.

ii.
```python
def discretize_position(positions, n_bins=5):
    bin_size = TRACK_LENGTH / n_bins  # 90 cm bins
    result = np.clip(np.floor(positions / bin_size).astype(int), 0, n_bins - 1)
```

iii. `CONVERSION_NOTES.md` explicitly documents these 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It uses the same trial slice as the neural data and is written into the second row of the per-trial output array.

ii.
```python
pos_trial = position[s:e]
...
output_trial[1, :] = pos_binned
```

iii. This is a direct consequence of the per-trial slicing scheme.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral series, named `lick_cumul` in code.

ii.
```python
'lick': np.array(bts.time_series['lick'].data[:]),
...
lick_cumul = nwb_data['lick']
lick_trial = lick_cumul[s:e].copy()
```

iii. `CONVERSION_NOTES.md` says the NWB lick series is cumulative.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Implemented behavior: the code thresholds each sliced lick value at `> 0` and stores the result as binary. This conflicts with `CONVERSION_NOTES.md`, which says the cumulative series was differenced first.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
lick_binary = (lick_trial > 0).astype(float)
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. The notes claim "Converted to binary per-frame by computing the diff," but the final implementation never calls `np.diff`; it only uses a threshold.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. It is sliced with the same `[s:e]` trial indices as the neural data and stored as a time-varying output row.

ii.
```python
lick_trial = lick_cumul[s:e].copy()
...
output_trial[3, :] = lick_binary.astype(int)
```

iii. This follows the script’s single indexing scheme for all trial-aligned variables.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from the `reward_zone` signal plus `position`, using the same zone-inference procedure described for 7-a.

ii.
```python
rz_labels = determine_reward_zone_for_all_trials(position, rz_signal, trial_starts, trial_ends)
...
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The notes justify this by saying the active reward-zone positions identify zone A/B/C and omission trials are filled from neighbors.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The script chooses the closest reward-zone center to the mean position where `reward_zone > 0` within the trial, fills missing labels from neighboring trials, falls back to `'A'` if still missing, and then maps `A/B/C` to `0/1/2`.

ii.
```python
mean_rz_pos = np.mean(rz_positions)
...
if labels[i] is None:
    for j in range(i + 1, n_trials):
        if labels[j] is not None:
            labels[i] = labels[j]
            break
...
if rz_label is None:
    rz_label = 'A'  # fallback
rz_loc = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. `CONVERSION_NOTES.md` documents the neighbor-based inference for omission trials.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` timestamps, behavioral timestamps, trial boundaries, and the `reward_zone` signal. A trial is only marked rewarded if a reward timestamp falls inside the trial and the mouse entered the reward zone.

ii.
```python
reward_in_trial = np.any(
    (reward_timestamps >= t_start) & (reward_timestamps <= t_end)
)
rz_trial = rz_signal[trial_start:trial_end]
entered_rz = np.any(rz_trial > 0)
return int(reward_in_trial and entered_rz)
```

iii. The notes only mention `Reward` timestamps relative to trial boundaries; the additional reward-zone requirement is undocumented outside the code.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The script computes `is_rewarded` once per trial using the rule above, then broadcasts that per-trial 0/1 label across every time bin of the trial.

ii.
```python
is_rewarded = []
for i in range(len(trial_starts)):
    rew = determine_trial_rewarded(...)
    is_rewarded.append(rew)
...
rew_outcome = is_rewarded[t]
...
output_trial[5, :] = rew_outcome
```

iii. The notes describe reward outcome as per trial and binary.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles several edge cases: truncates neural/behavior arrays to the shorter length; drops sessions with too few curated cells or too few trials; drops trials shorter than 5 samples; replaces NaNs in neural data with 0; replaces negative environment labels with 0; fills missing reward-zone labels from neighboring trials and finally defaults to zone A.

ii.
```python
if n_behav != n_neural:
    min_len = min(n_behav, n_neural)
    ...
if n_total_cells < 5:
    return None
...
if n_timepoints < 5:
    continue
...
neural_trial = np.nan_to_num(neural_trial, nan=0.0)
...
if env_type < 0:
    env_type = 0.0
...
if rz_label is None:
    rz_label = 'A'
```

iii. `CONVERSION_NOTES.md` explicitly mentions the shape-mismatch truncation and omission-trial reward-zone inference. Other fallbacks are only implicit in the code.

## 13-a. What are the most time-consuming steps of the code?

i. The most time-consuming work is repeatedly loading all NWB files, recomputing per-trial dF/F, running per-cell speed-correlation filtering, and deconvolving every trial with OASIS. These dominate the cost much more than packaging outputs.

ii.
```python
nwb = io.read()
...
dff = compute_dff_per_trial(F, Fneu, trial_starts, trial_ends, ...)
cell_mask = filter_interneurons(dff, speed, iscell)
...
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)):
    events_trial = deconvolve_oasis(trial_dff_clean, frame_rate=effective_frame_rate)
```

iii. The trajectory shows the agent spending most of its effort reasoning about dF/F, deconvolution, and interneuron filtering, including several attempts to reconcile filtering rates.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-cell correlation loop in `filter_interneurons`, the per-sample loops in `discretize_distance` and `discretize_speed`, the list-comprehension distance calculation over positions, and some of the repeated per-trial passes in dF/F computation.

ii.
```python
for idx in cell_indices:
    cell_data = dff[idx, valid]
    ...
for i, d in enumerate(distances):
    ...
for i, s in enumerate(speeds):
    ...
dist_to_rz = np.array([distance_to_reward_zone(p, rz_start, rz_end)
                       for p in pos_trial])
```

iii. The code comments do not discuss efficiency; this is implied by the explicit Python loops.

## 13-c. What processing does the code repeat multiple times?

i. It repeats passes over the same trial boundaries several times: once to mask fluorescence, again to compute baselines, again to smooth dF/F, again to deconvolve, and again to build per-trial outputs. It also loads NWB `Deconvolved` traces even though the default path recomputes events and does not use them.

ii.
```python
for start, end in zip(trial_starts, trial_ends):
    f_[:, start:end] = F[:, start:end]
...
for start, end in zip(trial_starts, trial_ends):
    baseline = ...
...
for start, end in zip(trial_starts, trial_ends):
    dff[:, start:end] = nansmooth(...)
...
deconv_nwb = nwb_data['deconvolved'].T
...
result = process_session(nwb_file, compute_own_dff=True)
```

iii. The notes defend the recomputed pipeline as matching the paper, but they do not address the repeated passes or the unused precomputed deconvolved data.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main unnecessary work is loading precomputed NWB `Deconvolved` data and then ignoring it in the default path, plus fully recomputing dF/F and OASIS events even though only the final event matrices are kept. The script also preserves `plane_idx` internally but ultimately writes only all-zero CA1 region labels.

ii.
```python
deconv_nwb = nwb_data['deconvolved'].T  # Pre-computed deconvolved
...
result = process_session(nwb_file, compute_own_dff=True)
...
planes = nwb_data['plane_idx'][cell_indices]
...
brain_region_idx.append(np.zeros(result['n_neurons'], dtype=int))
```

iii. The trajectory shows the agent knowingly preferring a heavier custom preprocessing route because it believed that better matched the paper, even after noticing discrepancies with the stored NWB deconvolved signal.
