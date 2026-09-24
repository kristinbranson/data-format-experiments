# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent enumerates every `sub-*` directory, sorts every `.nwb` file within it, and processes each file as one session using `h5py`. Unless `--sample` is supplied, every discovered file is processed.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir) if d.startswith('sub-')])
for subj in subjects:
    files = sorted(glob(os.path.join(data_dir, subj, '*.nwb')))
    for fpath in files:
        sessions.append({'subject': subj.replace('sub-', ''), 'filepath': fpath, ...})
...
with h5py.File(filepath, 'r') as f:
```

iii. The notes report 11 subjects and 152 sessions, matching the paper, and state that all 152 sessions were successfully processed. The agent chose direct HDF5 access rather than `pynwb`.

## 1-b. How are the data split into subjects?

i. Subject directories define subjects; the stored NWB `subject_id` is attached to each processed session, and the final sorted unique IDs define `subjects` and `subject_idx`.

ii.
```python
subject_id = f['general/subject/subject_id'][()].decode()
...
unique_subjects = sorted(set(all_subject_ids))
subject_idx = np.array([unique_subjects.index(s) for s in all_subject_ids])
```

iii. The notes say the 11 `sub-{id}` directories match the expected 11 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; its trial lists are appended as one element of each top-level session list.

ii.
```python
for i, sess_info in enumerate(sessions_info):
    result = process_session(sess_info['filepath'], ...)
    ...
    all_neural.append(result['neural'])
```

iii. The agent reports 152 NWB files and 152 converted sessions, consistent with the dataset and paper.

## 1-d. How are the data split into trials?

i. Trial starts are every positive `trial_start` sample and trial ends are every positive `teleport` sample. If counts disagree, the first positive teleport after each start is selected. Each trial is sliced `[start:end)`.

ii.
```python
trial_start_inds = np.where(trial_start > 0)[0]
teleport_inds = np.where(teleport_sig > 0)[0]
...
tp_after = teleport_inds[teleport_inds > ts]
matched_teleports.append(tp_after[0])
...
trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes identify `trial_start` and `teleport` as trial boundaries and report about 80.4 trials/session, matching the paper. They do not justify using all positive teleport samples rather than rising edges.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than two detected trials are skipped. Within a session, only trials shorter than two samples are dropped; no 50-frame minimum is used.

ii.
```python
if n_trials < 2:
    return None
...
if n_tp < 2:
    continue
```

iii. The agent only explains the required two-trial session minimum. Its notes do not justify the two-timepoint threshold.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural data come directly from each plane of `processing/ophys/Deconvolved`. `Fluorescence` and `Neuropil` are also loaded, but only to construct a simplified dF/F signal for interneuron filtering.

ii.
```python
d = f[f'processing/ophys/Deconvolved/{plane}/data'][:]
fl = f[f'processing/ophys/Fluorescence/{plane}/data'][:]
ne = f[f'processing/ophys/Neuropil/{plane}/data'][:]
...
neural_all = deconv[:, final_cell_mask].T
```

iii. The notes assert that the NWB `Deconvolved` arrays are already the fully processed events. This assertion is the basis for not recomputing the paper's event signal.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated, filtered by cell masks, transposed to neuron-by-time, cast to `float32`, trial-sliced, and NaNs replaced by zero. The paper's maximin dF/F and OASIS steps are not applied to the final signal.

ii.
```python
deconv = np.concatenate(deconv_list, axis=1)
neural_all = deconv[:, final_cell_mask].T.astype(np.float32)
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
```

iii. The notes describe the correct raw-F processing pipeline but say it is unnecessary because `Deconvolved` is already processed. They cite convenience and observed decoder performance as validation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must have `iscell[:,0] == 1`. Putative interneurons are then removed when a simplified median-baseline dF/F has correlation greater than 0.5 with speed on samples satisfying speed > 0 and position >= 0.

ii.
```python
cell_mask_concat = np.array([iscell[concat_to_iscell[i], 0] == 1 ...])
f_corrected = fluorescence - 0.7 * neuropil_data
dff_simple = (f_corrected - f_median) / np.abs(f_median)
valid_mask = (speed > 0) & (position >= 0) & ~np.isnan(speed)
...
final_cell_mask = cell_mask_concat & ~interneuron_mask
```

iii. The agent cites the paper's `iscell` curation and speed–dF/F correlation threshold. It calls its correlation computation vectorized, but does not justify replacing the paper's trial-wise maximin dF/F with a global median-baseline approximation or restricting correlation samples differently.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples are sliced starting exactly at each detected trial-start index, so column zero is aligned to trial start.

ii.
```python
si = trial_start_inds[t]
ei = teleport_inds[t]
trial_neural = neural_all[:, si:ei].copy()
```

iii. The notes state that behavior and neural data are already sampled on the same imaging-frame grid.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The nominal rate is 15.5078125 Hz, corresponding to 64.48 ms per sample; metadata uses this constant.

ii.
```python
IMAGING_RATE_NOMINAL = 15.5078125
time_bin_ms = 1000.0 / IMAGING_RATE_NOMINAL
```

iii. The notes report behavior and neural streams at approximately 15.5 Hz and a final 64.48 ms bin.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the within-trial frame index and the NWB imaging-plane rate, not directly from behavior timestamps.

ii.
```python
imaging_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
frame_time = 1.0 / imaging_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The agent states that behavioral streams are aligned to the imaging rate, so frame count times frame duration represents elapsed time.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based frame sequence is multiplied by seconds per frame.

ii.
```python
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The notes describe this mapping explicitly and report a range beginning at 0 seconds.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It has exactly `n_tp = end-start` entries and is stacked with the same trial slice as neural data.

ii.
```python
n_tp = ei - si
trial_neural = neural_all[:, si:ei].copy()
time_from_start = np.arange(n_tp, dtype=np.float32) * frame_time
```

iii. The agent relies on common frame indexing and truncates stream-length mismatches before trial segmentation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although the raw `environment` time series is loaded, the final value is inferred from the NWB `identifier` scene string.

ii.
```python
identifier = f['identifier'][()].decode()
scene = parse_scene(identifier)
...
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
trial_env = env_per_trial.copy()
```

iii. The notes claim scene parsing is more reliable for cross-environment switches and supports all observed scene-name formats.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. `Env1` and `Env2` tokens are mapped to 0 and 1. Switch scenes are divided at a hard-coded `change_trial=30`, and the scalar is repeated over each trial's samples.

ii.
```python
env = env_num - 1
...
env_per_trial[:change_trial] = from_env
env_per_trial[change_trial:] = to_env
...
np.full((1, n_tp), env_type, dtype=np.float32)
```

iii. The agent says scene parsing handles cross-environment switches, but gives no data-driven validation of the fixed switch index and the advertised fallback to the NWB environment field is not implemented.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the zero-based loop index over detected trial starts, not the NWB `trial number` series.

ii.
```python
for t in range(n_trials):
    ...
    trial_num = np.float32(t)
```

iii. The mapping plan describes an integer per trial; no further raw-data justification is supplied.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The index is cast to `float32` and repeated at all timepoints in the trial.

ii.
```python
np.full((1, n_tp), trial_num, dtype=np.float32)
```

iii. The agent notes that per-trial variables are expanded to meet the uniform `(n_input, n_timepoints)` format.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It comes from sparse `Reward/timestamps`, aligned to the nearest position timestamp and summarized into a binary reward flag for each preceding trial.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
frame_idx = np.argmin(np.abs(behav_timestamps - rt))
reward_frames[frame_idx] = 1.0
```

iii. The notes correctly recognize that Reward is event-based rather than frame-aligned and must be mapped to behavior frames.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded if any mapped reward frame lies in `[start,end)`. The resulting vector is shifted by one trial; trial zero is set to omission (0), then repeated over time.

ii.
```python
trial_rewarded[t] = int(np.any(reward_frames[si:ei] > 0))
...
for t in range(1, n_trials):
    prev_outcome[t] = trial_rewarded[t - 1]
```

iii. The agent follows the requested binary lagged outcome definition and explicitly documents the first-trial convention.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone coordinates inferred from the NWB identifier/scene. It does not use the raw `reward_zone` time series to infer the actual per-trial zone.

ii.
```python
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
trial_pos = position[si:ei]
signed_dist = compute_distance_to_reward_zone(trial_pos, rz_start, rz_end)
```

iii. The agent cites the paper/code coordinates A=[80,130], B=[200,250], C=[320,370] and assumes scene names plus a trial-30 switch identify zones reliably.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the zone start is used before the zone, zero inside it, and position minus the zone end after it.

ii.
```python
dist[before] = position[before] - rz_start
dist[after] = position[after] - rz_end
dist[inside] = 0.0
```

iii. This implements the requested signed distance to any point in the reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks assign seven categories. In particular, +10 remains class 4 and +50 remains class 5.

ii.
```python
bins[(distance > 0) & (distance <= 10)] = 4
bins[(distance > 10) & (distance <= 50)] = 5
bins[distance > 50] = 6
```

iii. The agent says these thresholds implement the instruction labels. Its exact inclusion of +10/+50 differs from the reference's `np.digitize` edge behavior but is a reasonable reading of the written intervals.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical `[si:ei)` frame slice.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
```

iii. The agent relies on the shared imaging-frame grid and prior length truncation.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `processing/behavior/BehavioralTimeSeries/position/data`.

ii.
```python
position = f['processing/behavior/BehavioralTimeSeries/position/data'][:]
trial_pos = position[si:ei]
```

iii. The notes identify this as the 450 cm-track position variable.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm, floored to an integer, and clipped to 0–4.

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. Five 90 cm bins implement five equal divisions of the 450 cm track; clipping accommodates slight out-of-range samples.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Boundaries are 90, 180, 270, and 360 cm, with boundary values assigned to the higher bin and all extremes clipped into end bins.

ii.
```python
np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The choice matches the requested five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Both use the same trial start and end frame indices.

ii.
```python
trial_neural = neural_all[:, si:ei].copy()
trial_pos = position[si:ei]
```

iii. Common frame sampling is the agent's stated alignment rationale.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw `lick` behavioral time series.

ii.
```python
lick = f['processing/behavior/BehavioralTimeSeries/lick/data'][:]
```

iii. The notes identify lick as a frame-aligned behavior stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Positive values are binarized to 1. Additionally, if more than 30% of a trial's raw samples exceed 2, the entire trial is forced to zero rather than marked missing.

ii.
```python
lick_binary[lick_binary > 0] = 1
...
if (np.sum(trial_lick > 2) / len(trial_lick)) > 0.30:
    lick_binary[si:ei] = 0
```

iii. The agent invokes the reference lick-sensor-error rule but deliberately uses zero “for cleaner output” instead of the reference's NaN/missing-data treatment.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The binarized lick stream is sliced with the same `[si:ei)` interval.

ii.
```python
trial_lick = lick_binary[si:ei]
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. Alignment relies on the common frame grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived solely from the identifier's scene name and trial index, using fixed A/B/C coordinates; raw `reward_zone` is not loaded.

ii.
```python
scene = parse_scene(identifier)
rz_labels, rz_coords, env_per_trial = get_reward_zone_labels(scene, n_trials)
```

iii. The notes say scene parsing handles 26 observed naming formats and is used to identify within- and cross-environment switches.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene tokens are parsed into A/B/C; switch sessions change labels at trial 30. Labels are mapped A=0, B=1, C=2 and repeated over time.

ii.
```python
labels[:change_trial] = from_zone
labels[change_trial:] = to_zone
...
zone_map = {'A': 0, 'B': 1, 'C': 2}
rz_loc = zone_map.get(rz_labels[t], 0)
```

iii. Balanced class fractions and plausible decoder performance are offered as sanity checks; no check against the actual `reward_zone` samples is documented.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from sparse Reward event timestamps and position timestamps.

ii.
```python
reward_timestamps = f['processing/behavior/BehavioralTimeSeries/Reward/timestamps'][:]
frame_idx = np.argmin(np.abs(behav_timestamps - rt))
```

iii. The notes state that sparse events must be converted to a frame-aligned signal.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each event is assigned to its nearest behavior frame; a trial is 1 if any event occurs in its frame interval, otherwise 0. The scalar is repeated across the trial.

ii.
```python
if np.any(reward_frames[si:ei] > 0):
    trial_rewarded[t] = 1
...
np.full((1, n_tp), rew_outcome, dtype=np.int64)
```

iii. The observed 15.7% omission rate is reported as matching the paper. Unlike the reference, the code does not assert a maximum timestamp alignment error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior length mismatches are truncated to the shorter length. Trial/teleport count mismatches are greedily paired. Neural NaNs are replaced by zero. Suspected broken lick trials are set to no-lick. Trials under two frames and sessions under two trials are skipped.

ii.
```python
min_len = min(n_timepoints, n_behav)
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
...
lick_binary[si:ei] = 0
```

iii. The notes document ten off-by-one multi-plane length mismatches and justify truncation. The zero replacements are described as producing cleaner output, but the risk of converting missing observations into biological zeros is not discussed.

## 13-a. What are the most time-consuming steps of the code?

i. The code times each session and the full run. Reading multiple large NWB arrays (Deconvolved, Fluorescence, Neuropil), processing all 152 sessions, holding the large nested output, and pickling the 9.4 GB dataset are the evident dominant costs.

ii.
```python
t0 = time.time()
with h5py.File(filepath, 'r') as f:
    ...
pickle.dump(data, f, protocol=4)
```

iii. The notes report 762.5 seconds for full conversion and a 9.4 GB output, but do not provide a measured stage-by-stage profile.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Reward-to-frame mapping repeatedly scans all behavior timestamps with `argmin`; trial reward and lick-QC loops could use indexed reductions; the final per-cell threshold assignment loop is unnecessary because the correlation result is already vectorized. Per-trial construction is less readily vectorized because trials have variable lengths.

ii.
```python
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behav_timestamps - rt))
...
for i, col_idx in enumerate(accepted_cols):
    if corrs[i] > 0.5:
        interneuron_mask[col_idx] = True
```

iii. The notes specifically celebrate vectorized z-score/dot-product correlation, but do not discuss the remaining avoidable loops.

## 13-c. What processing does the code repeat multiple times?

i. For every session it loads all three neural arrays even though only Deconvolved is saved and Fluorescence/Neuropil serve only QC. Trial ranges are traversed separately for reward outcome, lick correction, previous outcome, and final trial construction. Positive licks are binarized twice.

ii.
```python
lick_binary[lick_binary > 0] = 1
...
lick_out = (trial_lick > 0).astype(np.int64)
```

iii. The agent does not document these repetitions; its notes focus on correctness and overall run time.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `environment` is loaded and truncated but never used; `trial_input_pt`, `plane_cell_offset`, and `plane_mask` are unused; fluorescence/neuropil-derived dF/F is discarded after filtering; reward-frame timing is discarded after per-trial reduction; plotting receives `filepath` without using it.

ii.
```python
environment = f['processing/behavior/BehavioralTimeSeries/environment/data'][:]
plane_cell_offset = 0
plane_mask = planeIdx == plane_num
trial_input_pt = np.array([env_type, trial_num, prev_out], dtype=np.float32)
```

iii. The agent provides no justification for these discarded intermediates. Some are remnants of an intended fallback or earlier implementation.
