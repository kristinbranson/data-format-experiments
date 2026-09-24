# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. It enumerates sorted `sub-*` directories and every `.nwb` file, then opens each file directly with `h5py` and reads behavior, metadata, segmentation, and ophys arrays.

ii.
```python
subjects = sorted([d for d in os.listdir(data_dir)\n                   if d.startswith('sub-') and os.path.isdir(os.path.join(data_dir, d))])\n...\nwith h5py.File(filepath, 'r') as f:\n    behav = f['processing']['behavior']['BehavioralTimeSeries']\n    ophys = f['processing']['ophys']
```

iii. The notes say direct NWB paths were inspected and that full mode processes all 152 files; direct HDF5 was chosen instead of `pynwb`.

## 1-b. How are the data split into subjects?

i. Subject directories and the NWB `general/subject/subject_id` identify mice. While building sessions, unique subject IDs are appended and each session gets an index into that list.

ii.
```python
subject_id = f['general']['subject']['subject_id'][()]\n...\nif subj not in subjects_list:\n    subjects_list.append(subj)\nsubject_idx_list.append(subjects_list.index(subj))
```

iii. The notes treat each `sub-*` directory/NWB subject ID as a mouse and report 11 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one output session. Files are processed individually and their trial lists are appended; sessions with fewer than two emitted trials are skipped.

ii.
```python
for i, file_info in enumerate(nwb_files):\n    neural_trials, input_trials, output_trials, session_info = process_session(...)\n    if len(neural_trials) < 2:\n        continue\n    all_neural.append(neural_trials)
```

iii. The notes explicitly state “Each NWB file = 1 session,” consistent with filenames and target nesting.

## 1-d. How are the data split into trials?

i. Trial starts are every sample with `trial_start > 0`; ends are every sample with `teleport > 0`. The arrays are truncated to equal count, invalid start/end pairs are removed, and slices are `[start:end)`.

ii.
```python
trial_starts = np.where(trial_start_signal > 0)[0]\nteleports = np.where(teleport_signal > 0)[0]\nn_trials = min(len(trial_starts), len(teleports))\n...\ntrial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes say trials follow `trial_start` through teleport and include all intervening frames.

## 1-e. How are trials filtered based on quality controls?

i. Pairs with teleport not after start are removed; trials shorter than 5 frames are skipped; sessions with fewer than two surviving trials are skipped. Lick-error trials are retained.

ii.
```python
valid = teleports > trial_starts\n...\nif n_timepoints < 5:\n    continue\n...\nif len(neural_trials) < 2:\n    continue
```

iii. The agent described keeping valid frames and defensive short-trial/session checks. Its 5-frame cutoff differs from the reference’s documented 50-frame rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Final `neural` is taken from NWB `processing/ophys/Deconvolved` for all planes. `Fluorescence` and `Neuropil` are loaded only to compute dF/F for interneuron screening.

ii.
```python
deconv_data = ophys['Deconvolved']['plane0']['data'][:]\nfluor_data = ophys['Fluorescence']['plane0']['data'][:]\nneuropil_data = ophys['Neuropil']['plane0']['data'][:]\n...\nneural_all = deconv_data[:, final_neuron_mask]
```

iii. The notes call the stored Deconvolved array precomputed events matching the paper and therefore avoid recomputing final OASIS events.

## 2-b. How is the `neural` data processed?

i. Stored deconvolved arrays are concatenated across planes, filtered by ROI mask, sliced by trial, transposed to neuron × time, NaNs replaced by zero, and cast to float32. A separate approximate dF/F is computed per trial solely for QC; no OASIS is run.

ii.
```python
neural_all = deconv_data[:, final_neuron_mask]\ntrial_neural = neural_all[start:end, :].T.copy()\ntrial_neural[np.isnan(trial_neural)] = 0\nneural_trials.append(trial_neural.astype(np.float32))
```

iii. The agent believed NWB Deconvolved already represented the paper’s desired OASIS output. Notes describe neuropil subtraction/maximin/smoothing only for interneuron detection.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must pass `iscell` and not have dF/F–speed correlation above 0.5. Correlations are vectorized over globally valid samples.

ii.
```python
iscell = seg['iscell'][:, 0].astype(bool)\nis_interneuron = identify_interneurons(dff, speed, iscell)\n...\nfinal_neuron_mask[iscell_indices[~is_interneuron]] = True
```

iii. This follows the paper’s manual cell curation and putative-interneuron exclusion, according to the notes.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is implicit: each trial begins at the detected `trial_start` frame and neural samples are sliced from that same index; no resampling or offset is applied.

ii.
```python
start = trial_starts[t]\nend = teleports[t]\ntrial_neural = neural_all[start:end, :].T.copy()
```

iii. The notes identify the temporal alignment event as the first trial frame.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. Metadata uses the hard-coded 15.5078125 Hz rate, i.e. about 64.484 ms per frame, for every session.

ii.
```python
IMAGING_RATE = 15.5078125\nFRAME_PERIOD = 1.0 / IMAGING_RATE\n...\n'time_bin_size': 1000.0 / IMAGING_RATE
```

iii. The notes say one native imaging frame is retained and report approximately 64.5 ms.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the number of samples after the detected trial start and the hard-coded frame period, not from raw timestamps (although position timestamps are loaded).

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD
```

iii. The notes say frame timestamps relative to trial start, but implementation uses an ideal uniform sampling grid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. It creates `0, 1, …, n-1`, multiplies by `1/15.5078125`, and stores it as float32.

ii.
```python
time_from_start = np.arange(n_timepoints) * FRAME_PERIOD\ntrial_input[0, :] = time_from_start
```

iii. The stated goal is elapsed seconds from the aligned first frame at native frame rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. Its length equals the neural slice length and index zero corresponds to the same trial-start frame.

ii.
```python
n_timepoints = end - start\ntrial_neural = neural_all[start:end, :].T.copy()\ntrial_input = np.zeros((4, n_timepoints), dtype=np.float32)
```

iii. The agent relied on common sample indices and native rate.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It uses the NWB behavior `environment` array.

ii.
```python
environment = behav['environment']['data'][:]
```

iii. The notes identify 0/1 as ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial it removes negative values, takes the median, converts to int, defaults to 0 if none remain, and broadcasts the result across time.

ii.
```python
valid_env = env_vals[env_vals >= 0]\ntrial_env[t] = int(np.median(valid_env)) if len(valid_env) > 0 else 0\n...\ntrial_input[1, :] = env_val
```

iii. The agent treats environment as per-trial and robustly collapses what should be constant values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. Although raw `trial number` is loaded, the output is derived from the zero-based loop index over detected trials.

ii.
```python
trial_number = behav['trial number']['data'][:]\n...\ntrial_num = float(t)
```

iii. The notes describe trial number as zero-indexed within session; the code avoids disagreements in the stored signal.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to float and broadcast across every timepoint of that trial.

ii.
```python
trial_num = float(t)\ntrial_input[2, :] = trial_num
```

iii. The value is intended as a constant per-trial decoder input.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It comes from event timestamps in behavioral `Reward`, mapped to position timestamps; `Reward/data` is loaded but not used.

ii.
```python
reward_data = behav['Reward']['data'][:]\nreward_timestamps = behav['Reward']['timestamps'][:]\nbehav_timestamps = behav['position']['timestamps'][:]
```

iii. The notes say reward event timestamps determine whether a trial was rewarded.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Reward timestamps are placed with `searchsorted`; a trial is rewarded if any mapped index lies in `[start,end)`. Trial 0 gets 0, otherwise the preceding trial’s boolean is broadcast.

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)\n...\ntrial_rewarded[t] = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))\nprev_trial_outcome[1:] = trial_rewarded[:-1]
```

iii. This implements omitted=0/rewarded=1 and defines no previous outcome as 0.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavior `position`, behavior `reward_zone`, and fixed A/B/C boundaries. The current zone is inferred from mean position where `reward_zone > 0`.

ii.
```python
position = behav['position']['data'][:]\nreward_zone_signal = behav['reward_zone']['data'][:]\n...\nrz_pos = pos_trial[rz_trial > 0]\nmean_rz_pos = np.mean(rz_pos)
```

iii. The notes say NWB lacks scene labels, so zone identity must be inferred from reward-zone-active positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The inferred zone is nearest by center. Missing zones are forward-filled, leading gaps backfilled, and an all-missing fallback uses B. Signed distance is position minus the near edge before the zone, zero inside, and position minus far edge after.

ii.
```python
distance[before] = position[before] - rz_start\ndistance[inside] = 0.0\ndistance[after] = position[after] - rz_end
```

iii. The agent chose signed distance to the nearest zone edge, matching the task semantics; persistence filling was meant to handle omissions.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit boolean masks create seven categories at -50, -10, 0, 10, and 50 cm; zero is its own class.

ii.
```python
bins[(distance >= -50) & (distance < -10)] = 1\nbins[(distance >= -10) & (distance < 0)] = 2\nbins[distance == 0] = 3\nbins[(distance > 0) & (distance <= 10)] = 4
```

iii. The notes state these are the requested thresholds.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural arrays use the same `[start:end)` indices, so the output has one category per neural frame.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()\ntrial_pos = position[start:end]\ndist_bins = discretize_distance(...)
```

iii. The agent assumes behavior and imaging arrays are already frame-aligned.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from behavior `position`.

ii.
```python
position = behav['position']['data'][:]\n...\ntrial_pos = position[start:end]
```

iii. The notes identify position as centimeters on the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is divided by 90 cm, floored to integer, and clipped to classes 0–4.

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. This implements five equal-width bins and absorbs out-of-range samples into end bins.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are effectively 90, 180, 270, and 360 cm, with clipping below 0 and at/above 450.

ii.
```python
bins = np.clip(np.floor(position / 90.0).astype(np.int64), 0, 4)
```

iii. The agent follows five 90 cm bins over a 450 cm corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The position slice uses exactly the neural trial’s start and end indices.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()\ntrial_pos = position[start:end]
```

iii. Common frame indexing is the asserted alignment strategy.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It uses the NWB behavior `lick` array.

ii.
```python
lick = behav['lick']['data'][:]
```

iii. The notes identify it as a cumulative/count-like lick signal.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The full signal is clipped to [0,1]. Trials with more than 35% raw samples above 2 are set to NaN, but those NaNs are then converted to 0 in emitted output.

ii.
```python
lick_binary = np.clip(lick, 0, 1).astype(np.float64)\n...\nif frac_bad > LICK_ERROR_FRACTION:\n    lick_binary[start:end] = np.nan\n...\nlick_vals = np.where(np.isnan(trial_lick), 0, trial_lick).astype(np.int64)
```

iii. The notes cite reference lick-error correction and binary licks, but elsewhere claim `diff(cumulative_lick)>0`; that is not what code implements.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick output is sliced with the same `[start:end)` frame indices as neural.

ii.
```python
trial_neural = neural_all[start:end, :].T.copy()\ntrial_lick = lick_binary[start:end]
```

iii. The agent assumes synchronized behavior and neural frame arrays.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses behavior `reward_zone` and `position`, plus fixed zone ranges A 80–130, B 200–250, C 320–370 cm.

ii.
```python
REWARD_ZONES = {'A': (80, 130), 'B': (200, 250), 'C': (320, 370)}\n...\nzone = identify_reward_zone(position, reward_zone_signal, ...)
```

iii. The notes say scene information is absent and zones must be inferred spatially.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Mean active-zone position is assigned to the nearest fixed zone center. Missing values are forward/backfilled (or B internally), labels map A/B/C to 0/1/2, and the class is broadcast per trial.

ii.
```python
trial_rz_idx = np.array([{'A':0,'B':1,'C':2}.get(lbl, 0) for lbl in trial_rz_label])\n...\ntrial_output[4, :] = rz_loc
```

iii. The agent uses persistence across trials to cover omissions; unlike the reference, it does not use Viterbi smoothing.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from behavioral `Reward` timestamps and behavior position timestamps.

ii.
```python
reward_timestamps = behav['Reward']['timestamps'][:]\nbehav_timestamps = behav['position']['timestamps'][:]
```

iii. The notes treat any reward event during a trial as rewarded.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. `searchsorted` maps each reward time to a frame (then clips it). A trial is 1 if any mapped event index falls within its bounds, otherwise 0; the value is broadcast.

ii.
```python
reward_frame_indices = np.searchsorted(behav_timestamps, reward_timestamps)\nreward_frame_indices = np.clip(reward_frame_indices, 0, n_samples - 1)\ntrial_rewarded[t] = np.any((reward_frame_indices >= start) & (reward_frame_indices < end))
```

iii. This supplies the requested binary per-trial outcome, though no half-bin alignment check is made.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior/neural length mismatches are truncated to the minimum; unmatched boundary counts are truncated; invalid pairs and very short trials are removed; missing zones are filled; invalid environment defaults to 0; neural NaNs and lick-error NaNs become 0.

ii.
```python
n_samples = min(n_behav_samples, n_neural_samples)\n...\nn_trials = min(len(trial_starts), len(teleports))\n...\ntrial_neural[np.isnan(trial_neural)] = 0
```

iii. The notes specifically mention one-frame multi-plane mismatches and defensive filling/filtering. Some choices silently relabel missing values rather than preserving them.

## 13-a. What are the most time-consuming steps of the code?

i. The code times file loading, dF/F calculation, interneuron correlation, and total session work. Notes report the full 152-session conversion took ~879 seconds; dF/F and large-array I/O dominate.

ii.
```python
t_load = time.time() - t0\n...\nt_dff = time.time() - t1\n...\nt_int = time.time() - t1
```

iii. The notes discuss optimizing correlation from 4.3 s to 0.4 s/session and identify full conversion as about 14.6 minutes.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining loops include per-trial reward tests, zone inference/filling, lick QC, environment aggregation, previous-outcome assignment, output assembly, and per-trial dF/F filtering. Several session-wide boolean/reduction operations could be vectorized, though variable trial lengths make assembly natural.

ii.
```python
for i, (start, stop) in enumerate(zip(trial_starts, teleports)):\n...\nfor t in range(n_trials):\n    trial_rewarded[t] = ...
```

iii. The agent explicitly vectorized neuron–speed correlations; notes present that as the main performance improvement.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly loops over all trials for rewards, zones, coordinates, licks, environment, previous outcome, and final construction. It also reads stored Deconvolved plus raw F/Fneu and computes dF/F even though final neural comes from Deconvolved.

ii.
```python
for t in range(n_trials):  # repeated in several blocks\n...\ndff = compute_dff(...)\n...\nneural_all = deconv_data[:, final_neuron_mask]
```

iii. The repetitions support separate QC/feature stages; dF/F is justified only for interneuron screening.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads unused `reward_data`, `trial_number`, `scanning`, and `plane_idx`; computes dF/F only to derive a mask and discards it; derives timing/diagnostic metadata; and optional plots recompute continuous distance and placeholders. Stored Deconvolved is loaded before only its filtered subset is retained.

ii.
```python
reward_data = behav['Reward']['data'][:]\ntrial_number = behav['trial number']['data'][:]\nscanning = behav['scanning']['data'][:]\nplane_idx = seg['planeIdx'][:]
```

iii. The notes emphasize diagnostics and validation; they do not explicitly acknowledge most unused arrays.


