# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes the 11 mice and their expected experimental days/scenes in `SESSIONS_DICT`, lists every NWB file in each corresponding subject directory, and reads each matched file directly with `h5py`. Files without a matching scene and sessions with fewer than two retained trials are skipped.

ii. ```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))
nwb_files = sorted([f for f in os.listdir(subj_dir) if f.endswith('.nwb')])
f = h5py.File(nwb_path, 'r')
```

iii. The trajectory says the 11 directories matched the paper's 11 switch-condition mice and treats each NWB file as one mouse-day. It chose direct HDF5 access after inspecting the NWB hierarchy.

## 1-b. How are the data split into subjects?

i. Subjects are the mouse keys in the hard-coded session dictionary, numerically sorted; sessions receive the corresponding index in that list.

ii. ```python
subjects = sorted(SESSIONS_DICT.keys(), key=lambda x: int(x[1:]))
all_subject_idx.append(subjects.index(subj))
```

iii. The agent counted 11 available switch-task mice, matching the paper, and noted that fixed-condition mice were absent.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. The day is parsed from `_ses-XX_`, then matched to a scene entry for that mouse.

ii. ```python
ses_str = nwb_file.split('_ses-')[1].split('_')[0]
exp_day = int(ses_str)
for entry in SESSIONS_DICT[subj]:
    if entry['exp_day'] == exp_day:
        scene = entry['scene']
```

iii. The trajectory identifies each file as one mouse on one experimental day and uses the repository's `sessions_dict.py` metadata to interpret it.

## 1-d. How are the data split into trials?

i. Trial starts are every positive sample in `trial_start`; ends are every positive sample in `teleport`. The two lists are truncated to the shorter count, and slices are `[start:end)`.

ii. ```python
trial_start_idx = np.where(trial_start_signal > 0)[0]
teleport_idx = np.where(teleport_signal > 0)[0]
n_trials = min(len(trial_start_idx), len(teleport_idx))
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The agent states that trials run from `trial_start` to `teleport`, based on inspection of the signals and the paper's task structure.

## 1-e. How are trials filtered based on quality controls?

i. Trials with an end not after the start or fewer than two samples are skipped. At the session level, fewer than two retained trials causes the session to be skipped.

ii. ```python
if t_end <= t_start:
    continue
if n_tp < 2:
    continue
if neural is None or len(neural) < 2:
    continue
```

iii. No substantive trajectory justification was given for the two-sample cutoff; the session cutoff follows the decoder format requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final neural data comes from the NWB `processing/ophys/Deconvolved` plane arrays. `Fluorescence` and `Neuropil` are used only to compute dF/F for interneuron filtering.

ii. ```python
dec_group = f['processing/ophys/Deconvolved']
dec_planes = [dec_group[p]['data'][:] for p in planes]
deconvolved = np.concatenate(dec_planes, axis=1)
```

iii. The agent reasoned that `Deconvolved` was the stored result of the paper's full processing pipeline and chose to trust it instead of recomputing events, despite explicitly considering both options.

## 2-b. How is the `neural` data processed?

i. Precomputed deconvolved values are pooled across planes, cropped to the common neural/behavior length, filtered by cell indices, sliced into trials, transposed to neuron-by-time, and cast to float32. The neural signal itself is not recomputed.

ii. ```python
deconvolved = np.concatenate(dec_planes, axis=1)
deconv_filtered = deconvolved[:, final_cell_indices]
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The agent believed this field represented neuropil correction, maximin dF/F, smoothing, and OASIS already. It pooled planes because the paper says planes were pooled for analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must pass Suite2P `iscell`; among these, cells whose recomputed dF/F has Pearson correlation with speed greater than 0.5 are removed as putative interneurons.

ii. ```python
cell_mask = iscell[:, 0].astype(bool)
is_int = detect_interneurons(dff, speed, threshold=0.5)
final_cell_indices = cell_indices[~is_int]
```

iii. The agent cites the paper's manual curation and >0.5 speed-correlation criterion. It recreated dF/F from F and Fneu specifically because the stored fluorescence is raw and this filter needs dF/F.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials begin at the detected trial-start sample, making the first column time zero; no interpolation or shifting is performed.

ii. ```python
t_start = trial_start_idx[t]
neural = deconv_filtered[t_start:t_end, :].T.astype(np.float32)
```

iii. The agent concluded neural and behavior share frame indices, so trial slicing supplies start-of-trial alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The script uses the NWB imaging rate and reports the reciprocal of the median session rate in milliseconds.

ii. ```python
frame_rate = f['general/optophysiology/ImagingPlane/imaging_rate'][()]
time_bin_ms = 1000.0 / np.median(frame_rates)
```

iii. The agent intended to preserve the native approximately 15.5 Hz sampling (~64.5 ms) and regarded the recordings as consistent.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of retained frames and the ImagingPlane `imaging_rate`; loaded behavioral timestamps are not used in this calculation.

ii. ```python
dt = 1.0 / frame_rate
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. The trajectory says native frame rate supplies the time bins and that neural and behavioral sampling are aligned.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based frame index is multiplied by the reciprocal imaging rate.

ii. ```python
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. This was justified as preserving native sampling and setting the first trial frame to zero.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is generated with exactly `neural.shape[1]` samples, so entry `j` corresponds to neural column `j`.

ii. ```python
n_tp = neural.shape[1]
time_from_start = np.arange(n_tp, dtype=np.float32) * dt
```

iii. The agent states the streams share frame indexing and therefore need no interpolation.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although the NWB `environment` signal is loaded, the saved input is derived from the hard-coded scene string and trial index.

ii. ```python
env_signal = beh['environment/data'][:]
env = get_env_for_trial(scene, t)
env_binary = 0 if env == 1 else 1
```

iii. The agent used `sessions_dict.py` scene names and assumed switches occur at trial index 30; it reported that observed environment transitions appeared correct.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene text is parsed into before/after environments, trial 30 selects the after value on switch days, and ENV1/ENV2 are mapped to 0/1 and broadcast across the trial.

ii. ```python
if zone_after is not None and trial_idx >= SWITCH_TRIAL:
    return env_after if env_after is not None else env_before
np.full(n_tp, env_binary, dtype=np.float32)
```

iii. The agent believed task switches occurred at trial 30 and used repository session metadata to encode them.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the zero-based loop index over detected trial boundaries, not the NWB `trial number` series.

ii. ```python
for t in range(n_trials):
    np.full(n_tp, t, dtype=np.float32)
```

iii. The agent treated detected starts/teleports as authoritative and used a sequential within-session index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is broadcast over every time point of the trial.

ii. ```python
np.full(n_tp, t, dtype=np.float32)
```

iii. No additional processing was considered necessary for a per-trial continuous input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from NWB `Reward/timestamps` and the position timestamps delimiting the preceding processed trial.

ii. ```python
reward_ts = beh['Reward/timestamps'][:]
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
```

iii. The agent identified reward delivery timestamps as the direct record of whether a trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. The first trial is assigned 0. After each retained trial, its binary reward value becomes the next retained trial's previous-outcome input and is broadcast across time.

ii. ```python
prev_rewarded = 0
np.full(n_tp, prev_rewarded, dtype=np.float32)
prev_rewarded = rewarded
```

iii. This implements the requested omitted=0/rewarded=1 convention; no special discussion of skipped trials was provided.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone A/B/C inferred from the hard-coded scene metadata and trial index. It does not use the raw `reward_zone` series.

ii. ```python
pos_trial = position[t_start:t_end]
zone_letter = get_reward_zone_for_trial(scene, t)
zone_start, zone_end = REWARD_ZONES[zone_letter]
```

iii. After finding the raw reward-zone signal difficult to interpret, the agent chose authoritative-looking session scene names and fixed published zone boundaries, with a trial-30 switch.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position is first clipped to 0–450 cm. Distance is negative before the zone, zero inside its inclusive bounds, and positive after its far edge, then discretized.

ii. ```python
pos_trial = np.clip(pos_trial, 0, 450)
dist[before] = position[before] - zone_start
dist[inside] = 0.0
dist[after] = position[after] - zone_end
```

iii. The agent says this matches the paper's reward-relative coordinate and fixed 50 cm zones.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement seven categories: `<-50`, `[-50,-10)`, `[-10,0)`, exactly zero, `(0,10]`, `(10,50]`, and `>50`.

ii. ```python
out[(dist >= -10) & (dist < 0)] = 2
out[dist == 0] = 3
out[(dist > 0) & (dist <= 10)] = 4
```

iii. The agent states these boundaries come directly from the decoder instructions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural activity are sliced using the same `[t_start:t_end)` frame interval; no resampling is done.

ii. ```python
neural = deconv_filtered[t_start:t_end, :].T
pos_trial = position[t_start:t_end]
```

iii. Shared frame indexing was taken to guarantee alignment.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes from the NWB behavioral `position/data` series.

ii. ```python
position = beh['position/data'][:]
pos_trial = position[t_start:t_end]
```

iii. The agent recognized this as corridor position in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Per-trial position is clipped to [0, 450] before categorization.

ii. ```python
pos_trial = np.clip(pos_trial, 0, 450)
pos_disc = discretize_position(pos_trial)
```

iii. The agent gave no explicit trajectory justification for clipping, beyond treating the track as 450 cm long.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five explicit masks split at 90, 180, 270, and 360 cm; 360 and above is category 4.

ii. ```python
out[(position >= 270) & (position < 360)] = 3
out[position >= 360] = 4
```

iii. These are the requested five equal 90 cm bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position uses the same trial slice and length as neural data.

ii. ```python
neural = deconv_filtered[t_start:t_end, :].T
pos_trial = position[t_start:t_end]
```

iii. The agent relied on common frame indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from NWB behavioral `lick/data`.

ii. ```python
lick = beh['lick/data'][:]
lick_trial = lick[t_start:t_end]
```

iii. The trajectory identifies this as the time-varying lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Values greater than zero become 1 and all others become 0.

ii. ```python
lick_binary = (lick_trial > 0).astype(np.int64)
```

iii. Binarization implements the requested no/yes output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data are sliced with identical trial bounds.

ii. ```python
lick_trial = lick[t_start:t_end]
neural = deconv_filtered[t_start:t_end, :].T
```

iii. Shared frame indexing was considered sufficient.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from hard-coded session scene metadata and trial number rather than the NWB `reward_zone` and `position` series.

ii. ```python
zone_letter = get_reward_zone_for_trial(scene, t)
zone_idx = {'A': 0, 'B': 1, 'C': 2}[zone_letter]
```

iii. The agent found the recorded signal ambiguous and instead trusted `sessions_dict.py`, published boundaries, and a trial-30 switch.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene strings are parsed for before/after zone letters; trial indices 30 onward use the after-zone on switch days. A/B/C map to 0/1/2 and are broadcast over time.

ii. ```python
if zone_after is not None and trial_idx >= SWITCH_TRIAL:
    return zone_after
np.full(n_tp, zone_idx, dtype=np.int64)
```

iii. The agent justified this from repository session metadata and its understanding of switch-day structure.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It comes from `Reward/timestamps`, compared against position timestamps at the first and last samples of each trial.

ii. ```python
reward_ts = beh['Reward/timestamps'][:]
t_start_time = timestamps[t_start]
t_end_time = timestamps[t_end - 1]
```

iii. The agent regarded timestamped reward delivery as the direct outcome record.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies within its inclusive time window, otherwise 0; the result is broadcast across all trial samples.

ii. ```python
rewarded = int(np.any((reward_ts >= t_start_time) & (reward_ts <= t_end_time)))
np.full(n_tp, rewarded, dtype=np.int64)
```

iii. This directly implements the binary per-trial output requested.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavioral streams are silently cropped to their common minimum length; start/end arrays are truncated to equal counts; invalid or <2-sample trials, zero-neuron sessions, missing directories/scenes, and sessions with <2 trials are skipped. NaN cell-speed correlations are retained as non-interneurons. There is no interpolation or explicit missing-value repair.

ii. ```python
n_frames = min(n_frames_neural, n_frames_behav)
n_trials = min(len(trial_start_idx), len(teleport_idx))
if np.isnan(r):
    continue
```

iii. The cropping was added after a multi-plane length mismatch caused an error. Other guards are defensive; the trajectory reports successful format validation.

## 13-a. What are the most time-consuming steps of the code?

i. Reading all large NWB neural arrays, per-trial maximin dF/F computation, and especially nested per-cell Gaussian filtering/correlation across every session dominate conversion. Serial processing of 152 sessions and pickling the full dataset also cost time.

ii. ```python
dec_planes = [dec_group[p]['data'][:] for p in planes]
for start, stop in zip(trial_starts, trial_ends):
    for c in range(n_cells):
        bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
```

iii. The trajectory's repeated long conversion runs support this assessment, though the agent did not explicitly profile the code.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-cell Gaussian smoothing loops in dF/F computation can use `gaussian_filter1d(..., axis=1)` on the full cell matrix. Per-cell correlations can be computed in array form. Some per-trial classification/broadcast work could also be vectorized over the session, though variable trial lengths still require splitting.

ii. ```python
for c in range(n_cells):
    bl[c] = ndi.gaussian_filter1d(bl[c], sigma=15)
for c in range(n_cells):
    r = np.corrcoef(dff[c, nanmask], speed[nanmask])[0, 1]
```

iii. The agent provided no explicit efficiency justification; these loops mirror straightforward reference-style processing.

## 13-c. What processing does the code repeat multiple times?

i. Every session repeats full-array loading, trial-boundary extraction, per-trial dF/F baseline/smoothing, and trial construction. Trial slices and constant arrays are repeatedly created, and scene parsing is repeated for environment and reward zone on every trial.

ii. ```python
zone_letter = get_reward_zone_for_trial(scene, t)
env = get_env_for_trial(scene, t)
np.full(n_tp, zone_idx, dtype=np.int64)
```

iii. No explicit rationale was given; the session/trial organization was chosen for clarity and direct correspondence to the output structure.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and crops `env_signal` but never uses it. It computes full-session dF/F only to derive a cell mask, then discards dF/F and saves a different precomputed signal. It also parses `env_before`, `zone_before`, `env_after`, and `zone_after` once in `process_session` without using those local values.

ii. ```python
env_signal = beh['environment/data'][:]
dff = compute_dff_for_interneuron_detection(...)
env_before, zone_before, env_after, zone_after = parse_scene(scene)
```

iii. The dF/F computation was intentionally retained for the paper's interneuron criterion; the unused environment data and parse result appear accidental.
