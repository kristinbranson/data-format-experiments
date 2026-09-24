# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every `.nwb` beneath sorted `data/sub-*` directories and loads each file directly with `h5py`; `--sample` deliberately restricts this to two files.

ii. `for sub_dir in sorted(os.listdir(data_dir)):` / `if fname.endswith('.nwb'):` / `f = h5py.File(nwb_path, 'r')`

iii. The notes say this covers 11 subjects, 152 sessions, and 12,216 NWB trials, and that direct HDF5 paths map to the reference session structure.

## 1-b. How are the data split into subjects?

i. Subject IDs are parsed from directories such as `sub-m11`; a first-seen map produces `subjects` and per-session `subject_idx`.

ii. `subject_id = sub_dir.replace('sub-m', '')` and `subject_map[sub_name] = len(subjects)`.

iii. The notes identify the 11 expected switch-task mice and the GCAMP-to-mouse naming correspondence.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session; its session number is parsed from `_ses-YY_` and each successful `process_session` result is appended once.

ii. `ses_num = int(fname.split('_ses-')[1].split('_')[0])` and `neural_list.append(result['neural'])`.

iii. The notes report all 152 files/sessions and 12–14 sessions per subject.

## 1-d. How are the data split into trials?

i. Starts are every positive `trial_start` sample. For each start, the first later positive `teleport` sample is the exclusive stop.

ii. `trial_start_inds = np.where(trial_start_signal > 0)[0]`; `future_teleports = teleport_inds[teleport_inds > start]`; `trial_neural = neural_all[start:stop, :].T`.

iii. The notes state that trial-start-to-teleport matches the reference session convention.

## 1-e. How are trials filtered based on quality controls?

i. Trials shorter than three frames are skipped; sessions are skipped if fewer than two trials remain. No other trial exclusion is applied (lick-error trials are retained with altered licks).

ii. `if trial_len < 3: continue` and `if len(neural_trials) < 2: return None`.

iii. The agent describes this as short-trial/session format protection; it separately treats lick sensor errors as a correction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is taken from each plane of the NWB `processing/ophys/Deconvolved` group, then concatenated across planes.

ii. `planes = sorted(ophys['Deconvolved'].keys())`; `deconv_parts.append(ophys['Deconvolved'][plane]['data'][()])`.

iii. The agent asserted that NWB deconvolved events match the reference `sess.timeseries['events']` and chose them instead of recomputing from Fluorescence/Neuropil.

## 2-b. How is the `neural` data processed?

i. Plane matrices are concatenated along ROIs, filtered, sliced by trial, transposed to neuron × time, and cast to float32. No dF/F, smoothing, baseline calculation, or deconvolution is performed by this script.

ii. `deconv = np.concatenate(deconv_parts, axis=1)` and `trial_neural = neural_all[start:stop, :].T.astype(np.float32)`.

iii. The notes justify this by treating stored Deconvolved as already-OASIS-processed events and cite exact spot checks against NWB values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only ROIs with Suite2p `iscell[:,0] == 1` are kept; sessions with fewer than five such cells are skipped. Putative interneurons are not removed.

ii. `cell_mask = iscell[:, 0] == 1`; `if n_cells < 5: return None`.

iii. The notes identify manual `iscell` curation, acknowledge the paper’s interneuron exclusion and the elevated maximum cell count, but do not implement it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples begin at the `trial_start` index and end immediately before teleport, so time zero is the first selected sample.

ii. `start = trial_start_inds[i]`; `trial_neural = neural_all[start:stop, :].T`.

iii. The notes call this temporal alignment to trial start and report exact neural spot checks.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Native resolution is retained, approximately 64.5 ms (`1000/imaging_rate`); no rebinning/resampling occurs.

ii. `dt = 1.0 / imaging_rate` and `'time_bin_size': dt_ms`.

iii. The notes report ~15.5 Hz and explicitly choose the native imaging rate.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from frame index and the NWB ImagingPlane `imaging_rate`, not behavior timestamps.

ii. `imaging_rate = f['general']['optophysiology']['ImagingPlane']['imaging_rate'][()]`; `time_from_start = np.arange(trial_len) * dt`.

iii. The notes describe this as native-rate elapsed time and validate a spot trial.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based integer frame sequence is multiplied by seconds per frame.

ii. `dt = 1.0 / imaging_rate`; `time_from_start = np.arange(trial_len) * dt`.

iii. This was selected because the sampling is treated as uniform at the imaging rate.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created with exactly `trial_len`, the same start/stop slice as neural, and is assigned across the corresponding columns.

ii. `trial_input = np.zeros((4, trial_len))`; `trial_input[0, :] = time_from_start`.

iii. The agent reports exact trial-level time spot checks.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. Although raw `environment` is loaded, the output is derived from the hard-coded per-subject/session scene table.

ii. `scene = get_scene_for_session(subject_id, session_num)`; `env_per_trial = get_environment_from_scene(scene, n_trials)`.

iii. The notes say the scene table is from reference code and maps Env1=0, Env2=1.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Scene strings are parsed; fixed scenes fill one value and cross-environment scenes switch at trial index 30. The result is broadcast across time.

ii. `env[:change_trial] = before_env`; `env[change_trial:] = after_env`; `trial_input[1, :] = float(env_type)`.

iii. The agent relies on the paper’s “switch after 30 trials” and its corrected sessions dictionary.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It comes from the zero-based loop index over matched trial boundaries; the loaded raw `trial number` array is unused.

ii. `for i in range(n_trials):` and `trial_number = i`.

iii. The mapping plan specifies a 0-indexed per-trial number.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The loop index is converted to float and broadcast over all timepoints.

ii. `trial_input[2, :] = float(trial_number)`.

iii. No further processing was considered necessary.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward.timestamps`, position timestamps, and whether raw `reward_zone` is active within the prior trial.

ii. `trial_rewards = np.sum((reward_ts >= timestamps[start]) & (reward_ts <= timestamps[stop]))`; `prev_outcome[1:] = isreward[:-1]`.

iii. The notes define 0 as omission and 1 as rewarded and report a direct previous/current consistency check.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current outcome is 1 only when at least one reward timestamp and active reward-zone sample occur; this vector is shifted by one, with first trial 0, then broadcast.

ii. `isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0`; `trial_input[3, :] = float(prev_out)`.

iii. The extra reward-zone condition was intended to distinguish true omissions.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` plus zone coordinates inferred from the hard-coded scene/session table and a fixed switch at trial 30.

ii. `rz_coords, rz_labels = get_reward_zones_from_scene(scene, n_trials)`; `trial_pos = position[start:stop]`.

iii. The notes say the scene mapping comes from reference code and was checked against observed reward positions.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is position minus the near edge before a zone, zero inside its inclusive bounds, and position minus the far edge after it.

ii. `distance[before] = position[before] - rz_start`; `distance[after] = position[after] - rz_end`.

iii. This represents reward-relative position on the three documented 50-cm zones.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit masks create the seven requested classes with boundaries -50, -10, 0, 10, and 50 cm.

ii. `bins[(distances >= -10) & (distances < 0)] = 2`; `bins[distances == 0] = 3`; `bins[distances > 50] = 6`.

iii. The comments reproduce the requested category definitions.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use the identical `[start:stop]` indices, yielding the same number of columns.

ii. `trial_pos = position[start:stop]` and `trial_neural = neural_all[start:stop, :].T`.

iii. The agent states that NWB behavior and neural arrays are aligned and validated sample output bins.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from `behavior/position/data`.

ii. `position = behav['position']['data'][()]`; `trial_pos = position[start:stop]`.

iii. The notes identify position as the 450-cm corridor coordinate.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Values are clipped to [0,450] before discretization.

ii. `pos_clipped = np.clip(trial_pos, 0, TRACK_LENGTH)`.

iii. The agent intended clipping to keep small out-of-track values in valid endpoint classes.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. `linspace(0,450,6)` supplies five 90-cm bins; digitization against upper edges is clipped to labels 0–4.

ii. `bin_edges = np.linspace(0, TRACK_LENGTH, n_bins + 1)`; `bins = np.digitize(positions, bin_edges[1:])`.

iii. This directly follows the requested five equal track bins.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same per-trial start/stop indices are applied to position and neural arrays.

ii. `trial_pos = position[start:stop]`; `trial_output[1, :] = pos_bins`.

iii. The notes report an exact position-bin spot check.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It comes from the raw behavior `lick` series.

ii. `lick_raw = behav['lick']['data'][()]`; `trial_lick = lick_raw[start:stop].copy()`.

iii. The notes identify this as the behavioral lick stream.

## 9-b. What processing is involved in computing `output` *Lick*?

i. If over 35% of trial samples exceed 2, the whole trial is set to zero; otherwise values above 1 become 1, followed by integer casting.

ii. `if np.sum(trial_lick > 2) / len(trial_lick) > 0.35: trial_lick[:] = 0` else `trial_lick[trial_lick > 1] = 1`.

iii. The notes attribute 0.35 to reference code’s lick-sensor-error correction and say decoder continuity motivated retaining trials.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural arrays are sliced with the same trial indices.

ii. `trial_lick = lick_raw[start:stop].copy()`; `trial_output[3, :] = trial_lick`.

iii. The agent reports an exact lick-processing spot check.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from subject/session and the hard-coded scene dictionary, not directly from per-trial raw reward-zone/position observations.

ii. `scene = get_scene_for_session(...)`; `rz_labels = get_reward_zones_from_scene(...)[1]`.

iii. The notes say the corrected dictionary was imported from reference code and verified against actual reward positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Scene location letters determine A/B/C; transition sessions change at trial 30. Letters map to 0/1/2 and are broadcast.

ii. `rz_coords[:change_trial] = ...`; `rz_label_idx[i] = 0` (or 1/2); `trial_output[4, :] = rz_loc`.

iii. The fixed change point follows the paper’s stated experimental design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses reward event timestamps, position timestamps defining trial time bounds, and the raw reward-zone activity series.

ii. `reward_ts = behav['Reward']['timestamps'][()]`; `rzone_active = np.any(rzone_data[start:stop+1] > 0)`.

iii. The agent intended reward delivery plus zone activity to identify genuine rewarded versus omission trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. It counts reward events between inclusive trial timestamp bounds, requires active reward-zone data, converts to binary, and broadcasts across the trial.

ii. `isreward[i] = 1 if (trial_rewards > 0 and rzone_active) else 0`; `trial_output[5, :] = rew_out`.

iii. Reward frequency (~84.3%) matched the paper’s expected ~85%, which the agent used as validation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Very short trials, sessions with <2 valid trials or <5 cells, and any session raising an exception are skipped. Lick-error trials are zeroed. There is no explicit neural/behavior length reconciliation, missing-value handling, or timestamp-alignment assertion.

ii. `except Exception as e: ... continue`; `if n_cells < 5: return None`; `if trial_len < 3: continue`.

iii. The notes emphasize multi-plane fixes, short-trial checks, sensor-error handling, and successful validator/spot checks.

## 13-a. What are the most time-consuming steps of the code?

i. Full eager HDF5 loading of large neural/behavior arrays, concatenation and trial materialization, pickle serialization, and optional plotting dominate. The notes estimate ~90 seconds for 152 sessions; the resulting pickle is ~9.8 GB.

ii. `deconv_parts.append(...['data'][()])`; `pickle.dump(data, f)`.

iii. Runtime and output size are documented from the sample and full runs.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Start-to-next-teleport matching repeatedly filters all teleports (quadratic-like), reward outcome loops scan all reward timestamps, and A/B/C label assignment is a loop. Trial construction is less readily vectorized because lengths vary.

ii. `for i in range(len(trial_start_inds)): future_teleports = teleport_inds[teleport_inds > start]`; `for i in range(n_trials): trial_rewards = np.sum(...)`.

iii. The agent did not discuss these optimization opportunities; its notes focus on correctness and measured runtime.

## 13-c. What processing does the code repeat multiple times?

i. Each trial separately performs reward-time comparisons and slices/copies aligned streams; full output-distribution reporting later concatenates every output again. A dead preliminary `trial_input` allocation is immediately overwritten.

ii. `trial_input = np.array(...)` followed by `trial_input = np.zeros((4, trial_len), ...)`; later `vals.append(trial[out_idx, :])` for every output.

iii. No justification is given for the dead allocation or repeated summary traversal.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several arrays are eagerly loaded but unused (`trial_num`, `env_data`, `scanning`, `reward_data`, `autoreward`); the first `trial_input` array is discarded; optional figures and aggregate printed distributions do not affect the pickle.

ii. `trial_num = ...`; `env_data = ...`; `scanning = ...`; `reward_data = ...`; `autoreward = ...`.

iii. These appear to be exploration/reference remnants; the agent gives no downstream need for them.
