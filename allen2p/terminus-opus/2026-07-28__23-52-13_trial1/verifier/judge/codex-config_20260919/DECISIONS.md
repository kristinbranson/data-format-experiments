# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Local NWB IDs are intersected with the experiment CSV; only active-behavior, non-Multiscope experiments are loaded directly with `BehaviorOphysExperiment.from_nwb_path`.

ii. `exp_table = exp_table[exp_table['behavior_type'] == 'active_behavior']; exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']`; `return BehaviorOphysExperiment.from_nwb_path(nwb_path)`

iii. The notes justify active behavior as the requested task and exclude 11 Hz Multiscope recordings to enforce one nominal bin size.

## 1-b. How are the data split into subjects?

i. Unique retained `mouse_id` strings are assigned indices in encounter order.

ii. `mouse_id = str(meta['mouse_id']); subject_set[mouse_id] = len(subject_set)`

iii. The SDK mouse ID is treated as the animal identifier and counts are checked against the local subset.

## 1-c. How are the data split into sessions?

i. Each `ophys_experiment_id`—one imaging plane—is one output session; experiments sharing `ophys_session_id` are not combined.

ii. `for i, exp_id in enumerate(exp_ids): ... all_neural.append(neural_trials)`

iii. The notes explicitly decide “Each experiment = one session” so planes are independent.

## 1-d. How are the data split into trials?

i. SDK trial rows are sliced over ophys frames in `[start_time, stop_time)`; trials under five frames are skipped.

ii. `frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time); frame_indices = np.where(frame_mask)[0]`

iii. The agent uses canonical SDK boundaries and full variable-length trials to preserve time-varying outputs.

## 1-e. How are trials filtered based on quality controls?

i. Only go/catch, non-aborted, non-auto-rewarded trials remain. It also rejects trials under five frames and experiments with fewer than two trials or neurons.

ii. `valid_trials = trials[(trials['go'] | trials['catch']) & ~trials['aborted'] & ~trials['auto_rewarded']]`; `if n_neurons < 2: return None`

iii. Go/catch filtering follows the prompt; minimum counts support decoder evaluation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It uses deconvolved `dataset.events['events']`, not dF/F.

ii. `events_array = np.vstack(dataset.events['events'].values).astype(np.float32)`

iii. The notes cite the methods statement that neural analyses used detected calcium events (FastLZeroSpikeInference).

## 2-b. How is the `neural` data processed?

i. Event vectors are stacked, cast to float32, and trial-sliced; there is no smoothing, normalization, rebinning, or plane merging.

ii. `neural_trial = events_array[:, frame_indices]`

iii. Events are considered already processed by Allen, and each plane is deliberately independent.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Allen’s ROI curation is accepted without cell-level filtering, but experiments with fewer than two neurons are removed.

ii. `if n_neurons < 2: return None`

iii. The notes say classifier-based ROI filtering is already present.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Frames from trial start through trial stop are selected by ophys timestamps; metadata names trial start as alignment event.

ii. `frame_mask = (ophys_ts >= start_time) & (ophys_ts < stop_time)`; `'temporal_alignment_event': 'Trial start time'`

iii. The agent places every modality on the ophys clock and retains full trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning occurs. Multiscope data are excluded and metadata hard-codes `1000/31` ms (~32.3 ms), while samples remain at native ophys frames.

ii. `exp_table = exp_table[exp_table['equipment_name'] != 'MESO.1']`; `'time_bin_size': 1000.0 / 31.0`

iii. The notes say this gives consistent bins across Scientifica recordings.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It comes from `stimulus_presentations.image_name`, start/end times, restricted to `change_detection_behavior`.

ii. `sp_active = sp[sp['stimulus_block_name'] == 'change_detection_behavior']`

iii. The mapping plan treats presentations as the time-resolved image source.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Globally sorted non-omitted names become integers. Presentation frames receive an ID, grey periods are forward-filled, and unresolved trial-leading frames use the first valid ID or 0.

ii. `image_idx[mask] = image_to_idx[img_name]`; `elif last_img >= 0: image_idx[i] = last_img`; `img_trial[neg_mask] = first_valid`

iii. Forward fill represents the preceding non-grey image; notes document repairing four unresolved initial frames.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Labels are rasterized on ophys timestamps and sliced with the identical `frame_indices` used for neural events.

ii. `img_trial = image_idx_full[frame_indices]; neural_trial = events_array[:, frame_indices]`

iii. The agent reports matching identities to source presentations.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses active-block `stimulus_presentations.is_change`, `start_time`, and `end_time`.

ii. `change_sp = sp_active[sp_active['is_change'] == True]`

iii. The notes regard this as the authoritative actual-change marker.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is set to one only during each change presentation interval.

ii. `mask = (ophys_ts >= row['start_time']) & (ophys_ts < row['end_time']); change[mask] = 1`

iii. No further transform is claimed.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already boolean: no change is 0 and a true change presentation is 1.

ii. `change = np.zeros(n_tp, dtype=np.int32); change[mask] = 1`

iii. Output values are `['no_change', 'change']`.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change intervals are evaluated on ophys timestamps and sliced by neural trial indices.

ii. `change_trial = change_full[frame_indices]`

iii. The notes say all time-varying outputs share the ophys clock.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses `dataset.running_speed.timestamps` and `.speed`.

ii. `resample_to_ophys(running['timestamps'].values, running['speed'].values, ophys_ts)`

iii. The notes identify the SDK wheel signal at roughly 60 Hz.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation targets ophys timestamps. Five global percentile bins are computed from all full-session samples, not just retained trials.

ii. `np.interp(ophys_timestamps, signal_timestamps, signal_values)`; `running_edges = np.percentile(all_running_concat, np.linspace(0, 100, 6))`

iii. Global percentiles are intended to yield consistent, balanced classes.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Global 20/40/60/80th percentile cutoffs produce integer bins 0–4.

ii. `np.digitize(running_full, running_edges[1:-1])`

iii. This follows the requested five equal-percentile categories.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated to all ophys timestamps, then sliced using common trial indices.

ii. `running_trial = running_binned[frame_indices]`

iii. The notes report independent alignment/bin checks.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `eye_tracking.pupil_width` and timestamps, filters NaN width, but does not use `likely_blink`.

ii. `pupil_raw = eye['pupil_width'].values; valid_mask = ~np.isnan(pupil_raw)`

iii. The notes identify pupil width as diameter and do not claim explicit blink removal.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. More than ten valid samples are linearly interpolated to ophys time; otherwise all values are NaN. Five global full-session percentile bins are then used.

ii. `pupil_full = resample_to_ophys(pupil_ts[valid_mask], pupil_raw[valid_mask], ophys_ts)`; `pupil_edges = np.percentile(all_pupil_concat, np.linspace(0, 100, 6))`

iii. Global bins keep class meanings common across experiments.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global 20/40/60/80th percentiles create bins 0–4; missing values remain 0.

ii. `pupil_binned[valid_pupil_mask] = np.clip(np.digitize(pupil_full[valid_pupil_mask], pupil_edges[1:-1]), 0, 4)`

iii. This implements five percentile categories and stores their edges.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Width is interpolated to ophys timestamps and pupil classes use the same trial indices as neural data.

ii. `pupil_trial = pupil_binned[frame_indices]`

iii. The notes report a recomputation check.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It checks trial booleans `hit`, `miss`, `false_alarm`, and `correct_reject` in that order.

ii. `if trial_row['hit']: return 0 ... if trial_row['correct_reject']: return 3`

iii. These are documented as the four canonical outcomes.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Outcomes map to 0–3 and repeat across every frame; no match maps to -1.

ii. `outcome_trial = np.full(len(frame_indices), outcome, dtype=np.int32)`

iii. Repetition gives all output rows the same time dimension while retaining a static trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Load failures are skipped; missing/insufficient pupil becomes NaN then bin 0; unresolved image IDs are filled; short trials and low-count experiments are skipped. Extraction errors outside loading are not broadly caught, and `np.interp` endpoint-extrapolates.

ii. `except: pupil_full = np.full(..., np.nan)`; `img_trial[neg_mask] = first_valid`; `except Exception as e: all_session_data.append(None)`

iii. Notes document image repair, missing pupil fallback, and accept all-zero neural trials as genuine sparse events.

## 9-a. What are the most time-consuming steps of the code?

i. NWB loading/full-array retention dominates; presentation-by-presentation rasterization also repeatedly scans timestamps. The output is over 8 GB.

ii. `for i, exp_id in enumerate(exp_ids): dataset = load_experiment(exp_id); session_data = extract_session_data(...)`

iii. Notes report about 24 minutes for conversion and an 8239.5 MB pickle.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-presentation mask construction, frame-by-frame forward filling, trial iteration, and image-name collection could be vectorized or use search/slicing.

ii. `for j in range(len(starts)):`; `for i in range(n_tp):`; `for _, trial_row in valid_trials.iterrows():`

iii. The code calls presentation handling vectorized, but only each individual mask is vectorized.

## 9-c. What processing does the code repeat multiple times?

i. Image identities are rasterized in pass 1 using a growing incomplete map, then overwritten in pass 2 using the final map. Optional plots also recompute bins.

ii. `session_data = extract_session_data(dataset, temp_image_to_idx)`; `session_data['image_idx_full'] = get_image_at_ophys(..., image_to_idx)`

iii. The two-pass design is justified by global mappings/percentiles, although provisional image rasterization is unnecessary.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Provisional image arrays are overwritten; continuous behavior over non-trial frames and substantial full-session state are retained though not saved; optional plotting recomputes arrays only for visualization.

ii. `temp_image_to_idx = {...}; session_data = extract_session_data(...); session_data['image_idx_full'] = get_image_at_ophys(...)`

iii. Notes emphasize validation and the two-pass workflow but do not address discarded labels or memory cost.
