# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent uses a hard-coded `SESSIONS_META` table for 11 mice and 152 expected sessions, constructs each NWB path, and reads datasets directly with `h5py`. Missing directories or files are warned about and skipped.

ii.
```python
subjects = sorted(SESSIONS_META.keys())
for sub_idx, subject_id in enumerate(subjects):
    for ses_num, scene, exp_day in SESSIONS_META[subject_id]:
        nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
        result = process_session(os.path.join(sub_dir, nwb_filename), subject_id, scene, exp_day)
```

iii. The trajectory says the table was derived from `sessions_dict.py`, maps GCAMP identifiers to `sub-m*`, and was checked until all 152 sessions were included. Direct HDF5 loading was chosen after inspecting NWB structure.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted keys of the hard-coded session metadata; each processed session receives that key and its enumerated subject index.

ii.
```python
subjects = sorted(SESSIONS_META.keys())
for sub_idx, subject_id in enumerate(subjects):
    ...
    all_subject_idx.append(sub_idx)
```

iii. The agent justified the mapping by cross-referencing subject folders, NWB subject IDs, and the paper's 11 switch-task mice.

## 1-c. How are the data split into sessions?

i. One expected NWB file is one session. Its filename uses the experimental day directly as `ses-XX`.

ii.
```python
nwb_filename = f'sub-{subject_id}_ses-{exp_day:02d}_behavior+ophys.nwb'
result = process_session(nwb_path, subject_id, scene, exp_day)
all_neural.append(result['neural'])
```

iii. The trajectory records a correction for m11: session numbers equal experiment days, rather than being renumbered from the first imaging day.

## 1-d. How are the data split into trials?

i. Trial starts are every positive sample of `trial_start`; trial ends are every positive sample of `teleport`. The two lists are truncated to equal length, pairs with start after end are discarded, and slices are `[start:end)`.

ii.
```python
tstart_inds = np.where(trial_start_signal > 0)[0]
teleport_inds = np.where(teleport_signal > 0)[0]
n_trials = min(len(tstart_inds), len(teleport_inds))
valid = [i for i in range(n_trials) if tstart_inds[i] < teleport_inds[i]]
```

iii. The agent stated that trials should use `trial_start` and teleport markers and exclude teleport periods. It did not explain why all positive teleport samples, rather than rising edges, are safe.

## 1-e. How are trials filtered based on quality controls?

i. Sessions with fewer than two boundary-valid trials are skipped. During construction, only trials shorter than two samples are removed; a session is again skipped if fewer than two trials remain.

ii.
```python
if n_trials < 2: return None
...
if n_timepoints < 2:
    continue
...
if len(neural_trials) < 2: return None
```

iii. The trajectory mentions valid trial boundaries and the decoder's two-trial requirement, but gives no evidence for the two-sample cutoff. It did not implement the reference's 50-timepoint filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural output is recomputed from the NWB `Fluorescence` (F) and `Neuropil` (Fneu) series, pooled across planes, rather than using the stored `Deconvolved` series.

ii.
```python
F_list.append(fluor_group[plane_key]['data'][:].T)
Fneu_list.append(neuro_group[plane_key]['data'][:].T)
F = np.concatenate(F_list, axis=0)
```

iii. The agent explicitly reasoned that the NWB deconvolution was Suite2p output and did not match the paper's custom F/Fneu-to-OASIS pipeline.

## 2-b. How is the `neural` data processed?

i. Within-trial F and Fneu are retained; Fneu is subtracted at 0.7, its trial mean is added back, a sigma-15 Gaussian plus 300-frame minimum/maximum maximin baseline is computed, dF/F is formed, smoothed at sigma 2, and deconvolved with OASIS (`tau=0.7`, rate `imaging_rate/n_planes`). Unlike the reference, all sessions exclude teleport intervals from baseline calculation.

ii.
```python
f_[:, nanmask] = f_[:, nanmask] - NEU_COEF * f_neu_[:, nanmask]
flow[:, start:stop] = nansmooth(f_[:, start:stop], BASELINE_SMOOTH_SIGMA, axis=1)
flow[:, start:stop] = minimum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
flow[:, start:stop] = maximum_filter1d(flow[:, start:stop], BASELINE_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - flow[:, nanmask]) / np.abs(flow[:, nanmask])
spks[:, start:stop] = dcnv.oasis(dff[:, start:stop], 2000, TAU, frame_rate / n_planes)
```

iii. The trajectory ties each parameter to the paper and says recomputation is needed for an exact match. It does not discuss or implement the per-session `keep_teleports` metadata used by the reference.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs are restricted to Suite2p `iscell`; then neurons whose within-trial dF/F has Pearson correlation with speed above 0.5 are removed.

ii.
```python
cell_mask = iscell[:, 0] == 1
...
r, _ = pearsonr(dff_valid, speed_valid)
if r > INTERNEURON_SPEED_CORR_THR:
    keep_neuron[n_idx] = False
```

iii. The agent identifies these as the paper's manual ROI curation and putative-interneuron exclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural activity is sliced from the detected trial-start index to the teleport index, so column zero is trial start.

ii.
```python
t0, t1 = tstart_inds[i], teleport_inds[i]
trial_neural = spks[:, t0:t1]
```

iii. The trajectory explicitly selected start-of-trial alignment and said no resampling was needed because streams share frame indices.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Native per-plane sampling is used (about 15.5 Hz, roughly 64.5 ms); OASIS receives scanner rate divided by plane count. Metadata, however, is computed as the median `1000/imaging_rate` without plane-count correction.

ii.
```python
spks[:, start:stop] = dcnv.oasis(..., frame_rate / n_planes)
...
time_bin_sizes.append(1000.0 / info['imaging_rate'])
```

iii. The trajectory says to retain the native ~15.5 Hz rate. It recognized multi-plane rate handling for deconvolution but did not carry `n_planes` into metadata.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the timestamps attached to the position series.

ii.
```python
timestamps = bts['position']['timestamps'][:]
time_from_start = timestamps[t0:t1] - timestamps[t0]
```

iii. The trajectory treats behavioral timestamps as frame-aligned with neural samples.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The first timestamp in each trial is subtracted from every timestamp in that trial.

ii.
```python
time_from_start = (timestamps[t0:t1] - timestamps[t0])
```

iii. This implements time zero at the alignment event; no further justification was recorded.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The timestamp and neural arrays are first trimmed to their common session length, then sliced with identical trial boundaries.

ii.
```python
n_frames = min(len(position), F.shape[1])
timestamps = timestamps[:n_frames]
F = F[:, :n_frames]
```

iii. The trajectory encountered small neural/behavior length mismatches in multi-plane recordings and chose common-length trimming.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is not read from the raw `environment` time series. It is inferred from the hard-coded session `scene` string and trial index.

ii.
```python
env_type = get_environment_per_trial(scene, i)
```

iii. The agent used `sessions_dict.py` metadata to map sessions to environments, including cross-environment switches.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene names beginning with Env2 map to 1 and others to 0; cross-environment scenes switch at hard-coded trial 30.

ii.
```python
if '_to_Env' in scene:
    if trial_idx < change_trial:
        return 0 if 'Env1' in parts[0] else 1
    return 0 if parts[1].startswith('1') else 1
```

iii. The agent assumed the experimental switch occurs at trial 30. It did not justify preferring inferred metadata over the recorded environment values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Trial number is the zero-based loop index over detected trial boundaries.

ii.
```python
for i in range(n_trials):
    trial_number = float(i)
```

iii. The trajectory defined it as a within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is broadcast across all samples of the trial.

ii.
```python
np.full(n_timepoints, trial_number, dtype=float)
```

iii. No additional processing rationale was recorded.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps after labeling each trial rewarded if an event timestamp lies between its start and teleport timestamps.

ii.
```python
reward_timestamps = bts['Reward']['timestamps'][:]
reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
trial_rewarded[i] = int(np.any(reward_in_trial))
```

iii. The trajectory planned to use reward events for binary rewarded/omitted outcomes.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Trials after the first copy `trial_rewarded[i-1]`. The first trial is set to rewarded (1), unlike the reference's omitted/no-previous default of 0.

ii.
```python
if i == 0:
    prev_outcome = 1.0
else:
    prev_outcome = float(trial_rewarded[i - 1])
```

iii. The code comment calls 1 the first-trial default, but the trajectory provides no justification for this choice.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus reward-zone boundaries inferred from hard-coded scene metadata and a trial-30 switch, rather than deriving zone identity from raw `reward_zone` and position signals.

ii.
```python
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
rz_start, rz_end = REWARD_ZONES[rz_label]
pos_trial = position[t0:t1]
```

iii. The trajectory explored the NWB `reward_zone` signal, concluded it was a sparse VR signal rather than the requested distance categories, and chose known zone ranges plus session metadata.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Position minus the near edge is used before the zone, zero inside, and position minus the far edge after the zone; the result is then categorized.

ii.
```python
dist = np.where(position < rz_start, position - rz_start,
                np.where(position > rz_end, position - rz_end, 0.0))
```

iii. The agent reasoned that this directly implements signed distance to the current reward zone.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven explicit Boolean masks implement the requested cutoffs: below -50, [-50,-10), [-10,0), exactly zero, (0,10], (10,50], and above 50.

ii.
```python
bins[dist < -50] = 0
bins[(dist >= -50) & (dist < -10)] = 1
bins[(dist >= -10) & (dist < 0)] = 2
bins[dist == 0] = 3
bins[(dist > 0) & (dist <= 10)] = 4
bins[(dist > 10) & (dist <= 50)] = 5
bins[dist > 50] = 6
```

iii. The trajectory notes that these are the decoder specification's bins.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and spikes use the identical `[t0:t1]` frame slice.

ii.
```python
trial_neural = spks[:, t0:t1]
pos_trial = position[t0:t1]
```

iii. The agent assumed the NWB behavioral and imaging samples were already aligned, after common-length trimming.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the raw behavior `position` data.

ii.
```python
position = bts['position']['data'][:]
pos_trial = position[t0:t1]
```

iii. The agent treats position as the VR corridor coordinate in centimeters.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90, converted to integer, and clipped to categories 0 through 4.

ii.
```python
bins = np.clip((position / 90).astype(int), 0, 4)
```

iii. The agent used five equal 90-cm sections of the 450-cm track.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The effective boundaries are 90, 180, 270, and 360 cm, with clipping of out-of-range values to endpoint categories.

ii.
```python
np.clip((position / 90).astype(int), 0, 4)
```

iii. This was justified by the requested five equal-sized bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Position and neural data are sliced with the same trial indices.

ii.
```python
trial_neural = spks[:, t0:t1]
pos_trial = position[t0:t1]
```

iii. The agent relied on shared frame indexing.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the raw behavior `lick` time series.

ii.
```python
lick = bts['lick']['data'][:]
lick_trial = lick_corrected[t0:t1]
```

iii. The trajectory identifies lick as an available behavior signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. If more than 30% of a trial's samples have raw lick values above 2, the whole trial's lick signal becomes NaN. Thereafter NaNs become 0 and all positive values become 1.

ii.
```python
if np.sum(trial_licks > 2) / n_frames_trial > 0.3:
    lick_corrected[t0:t1] = np.nan
lick_binary = np.where(np.isnan(lick_trial), 0, (lick_trial > 0).astype(int))
```

iii. The agent's comment attributes the 30% rule to the paper as lick-sensor error correction. The trajectory does not explain converting an entire corrupted trial to “no lick.”

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The corrected lick array and neural array use identical trial slices.

ii.
```python
trial_neural = spks[:, t0:t1]
lick_trial = lick_corrected[t0:t1]
```

iii. Shared frame indexing is the stated alignment strategy.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred solely from the hard-coded scene string and zero-based trial index, not raw `reward_zone` data.

ii.
```python
rz_label = get_reward_zone_for_trial(scene, i, n_trials)
```

iii. The agent decided the stored reward-zone signal was not itself a zone label and used the known experimental schedule.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Single-zone scene names map directly to A/B/C. Switch scene names are parsed into before/after labels and change at trial 30; A/B/C then map to 0/1/2 and are broadcast over the trial.

ii.
```python
if trial_idx < change_trial:
    return before_zone
return after_zone
...
rz_location = {'A': 0, 'B': 1, 'C': 2}[rz_label]
```

iii. The trajectory says session metadata describes each reward-zone transition. It did not validate the hard-coded change point against each session's recorded signal.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the timestamps of raw `Reward` events and position timestamps defining trial time windows; reward amounts are loaded but unused.

ii.
```python
reward_data = bts['Reward']['data'][:]
reward_timestamps = bts['Reward']['timestamps'][:]
```

iii. The agent treats the presence of any reward event as the outcome, consistent with binary rewarded/omitted labels.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. A trial is 1 if any reward timestamp lies inclusively between start and teleport times, otherwise 0; the value is broadcast over the trial.

ii.
```python
reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
trial_rewarded[i] = int(np.any(reward_in_trial))
```

iii. The trajectory planned this event-within-trial test and validated decoder performance, but recorded no explicit timestamp-error check.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural and behavior arrays are silently trimmed to their shortest length; start/end lists are truncated and invalid pairs removed; missing files, sessions with fewer than two trials, and sessions without neurons are skipped. Neural NaNs become zero. Lick sensor-error trials become all zero after NaN replacement.

ii.
```python
n_frames = min(len(position), F.shape[1])
n_trials = min(len(tstart_inds), len(teleport_inds))
trial_neural = np.nan_to_num(trial_neural, nan=0.0)
if n_neurons < 1: return None
```

iii. The trajectory specifically encountered small multi-plane length mismatches and chose minimum-length trimming. Other defensive behavior is present in code but was not substantively justified.

## 13-a. What are the most time-consuming steps of the code?

i. Reading every large NWB recording, repeated per-trial maximin filtering and OASIS deconvolution, per-neuron correlations, and serial processing of all 152 sessions are the expensive steps.

ii.
```python
for ses_num, scene, exp_day in sessions:
    result = process_session(...)
...
for start, stop in zip(trial_starts, trial_ends):
    spks[:, start:stop] = dcnv.oasis(...)
```

iii. The trajectory shows full conversion runs as long-running background tasks and identifies custom deconvolution as necessary; it gives no formal performance analysis.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron-by-neuron Pearson loop could be replaced by a vectorized correlation calculation. Reward outcome can be assigned using mapped event indices rather than scanning all timestamps for every trial. Some trial masks/copies could be constructed together, although variable-length deconvolution naturally remains per trial.

ii.
```python
for n_idx in range(dff.shape[0]):
    r, _ = pearsonr(dff[n_idx, valid_mask], speed_valid)
...
for i in range(n_trials):
    reward_in_trial = (reward_timestamps >= t_start_time) & (reward_timestamps <= t_end_time)
```

iii. The agent did not discuss vectorization; these opportunities are inferred from its implementation.

## 13-c. What processing does the code repeat multiple times?

i. Trial ranges are looped over to copy F, copy Fneu, add neuropil/compute baseline, smooth/deconvolve, label reward, correct licks, and construct outputs. Reward timestamps are compared in full once per trial.

ii.
```python
for start, stop in zip(trial_starts, trial_ends): ...  # repeated in compute_dff_and_deconvolve
for i in range(n_trials): ...                          # repeated in process_session
```

iii. The trajectory does not justify the repetition; it arose from a straightforward staged implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads `reward_data` but never uses it, computes `frame_time` but only returns it without saving it, retains filtered `dff` after filtering although only `spks` is output, and accepts unused `n_trials` in `get_reward_zone_for_trial`. The full dF/F array is nevertheless needed temporarily for interneuron filtering.

ii.
```python
reward_data = bts['Reward']['data'][:]
frame_time = np.median(np.diff(timestamps))
dff = dff[keep_neuron, :]
```

iii. The trajectory does not identify these as deliberate; they appear to be implementation leftovers.
