# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes 12 `SessionSpec` records from the paper’s Figure 8 context-analysis roster, all under `/app/data/Ephys_Behavior`, and loads each session’s `data_structure_*.mat` with `mat73` plus its `motionEnergy_*.mat` with SciPy. It does not load the randomized-delay folder or the reference’s full 44-session roster.

ii. `SESSION_SPECS = [SessionSpec("JEB6", "2021-04-18", 2), ...]` and `obj = mat73.loadmat(spec.data_path)["obj"]`.

iii. The trajectory says the roster follows `Figure8a_thru_c.m` and the loader files; its notes explicitly frame this as the paper’s “two-context ALM ephys cohort.”

## 1-b. How are the data split into subjects?

i. Each `SessionSpec.animal` is treated as a subject. Subjects are added in first-seen order and every session gets the corresponding integer index, yielding seven IDs.

ii. `subject_idx = subject_to_idx.get(spec.animal)`; `subjects.append(spec.animal)`; `full_data["subject_idx"].append(subject_idx)`.

iii. The trajectory notes that the released Figure 8 roster contains seven distinct animal IDs, despite the paper text saying six mice, and chooses the explicit released roster.

## 1-c. How are the data split into sessions?

i. Each hard-coded animal/date pair is one session and one element of `neural`, `input`, and `output`; a single specified probe is used per session.

ii. `for spec in SESSION_SPECS:` followed by `full_data["neural"].append(neural_trials)` (and analogous appends).

iii. The trajectory justifies this by treating the Figure 8 loader’s 12 entries as the task-specific cohort.

## 1-d. How are the data split into trials?

i. Raw arrays are indexed by Bpod trial number. Kept trial indices are obtained from a Boolean mask, and each index becomes one neural/input/output record.

ii. `kept_trial_indices = np.flatnonzero(keep_trials)` and `for out_pos, trial_idx in enumerate(kept_trial_indices):`.

iii. The trajectory treats `bp.Ntrials` and the aligned per-trial fields as the authoritative trial partition.

## 1-e. How are trials filtered based on quality controls?

i. Only hit or miss trials are retained; early-lick, `no`/ignore, and stimulation-enabled trials are removed. Sessions must retain at least two trials.

ii. `return (hit | miss) & ~early & ~no & ~stim`.

iii. The trajectory says early, no-response, and stimulation trials are excluded to follow the paper’s analysis, even though the conversion prompt defines ignore as an output class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the selected `obj.clu` probe’s per-unit `trial`, `trialtm`, and `quality` fields, plus `bp.ev.goCue` for alignment.

ii. `spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1`; `spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()`.

iii. The agent says this follows the shared MATLAB spike-alignment pipeline and the probe selection in the metadata loaders.

## 2-b. How is the `neural` data processed?

i. Spikes are binned, converted to Hz, and smoothed with the ported `my_smooth` causal half-Gaussian (`SMOOTH=15`, reflect pre-padding). No z-scoring or baseline normalization is applied.

ii. `rates = counts.T / DT`; `trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect")`.

iii. The trajectory claims this matches `mySmooth.m`, including causal kernel construction and boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters labelled garbage, gabrga, noisy, or real? are removed. A seven-condition PSTH average is then used to remove units with mean firing rate at or below 1 Hz; sessions with fewer than ten surviving units raise an error.

ii. `bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])`; `keep = mean_frs > LOW_FR`.

iii. The trajectory says these filters reproduce the released code path and paper’s >1 Hz rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike’s within-trial time has that trial’s go-cue time subtracted before bin assignment.

ii. `aligned = spike_times - go_cue[spike_trials]`.

iii. The agent identifies `goCue` as the shared response-cue/water-drop alignment field used by the source code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output uses 10 ms bins from −3.0 to +2.5 s (550 bins). Raw spikes are rebinned directly to this grid; video streams are interpolated to the same bin centers.

ii. `TMIN = -3.0`; `TMAX = 2.5`; `DT = 1 / 100`; `time_bin_size = float(DT * 1000.0)`.

iii. The trajectory says these values match the selected shared MATLAB context-analysis pipeline.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw field; it is the fixed bin-center axis implied by `TMIN`, `TMAX`, and `DT`.

ii. `TIME = edges[:-1] + DT / 2` and `input_trials.append(TIME[np.newaxis, :])`.

iii. The agent treats the common aligned grid as the direct representation of elapsed time from go cue.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. Uniform edges are generated and shifted by half a bin to obtain centers; the same row is stored for every trial.

ii. `edges = np.arange(TMIN, TMAX + DT * 0.5, DT)`; `time = edges[:-1] + DT / 2`.

iii. This was chosen to make input samples correspond to the centers of neural bins.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. Neural spike bins use the same `TMIN` and `DT`, and the input contains their centers, so index positions align one-to-one.

ii. `bins = np.floor((aligned - TMIN) / DT).astype(np.int64)` and `input_trials.append(TIME[np.newaxis, :])`.

iii. The trajectory reports verifying that all time axes match.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is copied from `bp.R` for retained hit/miss trials.

ii. `lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]`.

iii. The agent’s notes describe `bp.R` as left=0/right=1 for the retained response trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No outcome-dependent inversion is performed; the Boolean/int `R` value is repeated across all 550 time bins. Ignore/no-lick trials are absent.

ii. `np.full((1, NT), lick_direction[out_pos], dtype=np.int16)`.

iii. The trajectory assumes `bp.R` directly represents actual lick direction after no-response trials are excluded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp.autowater`.

ii. `context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)`.

iii. The trajectory identifies autowater trials as WC and other retained trials as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The flag is inverted so WC=0 and DR=1, then repeated over time.

ii. `np.full((1, NT), context[out_pos], dtype=np.int16)`.

iii. This matches the requested label order in the agent’s notes.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is copied from `bp.hit` on the retained hit/miss trials.

ii. `outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]`.

iii. Because only hit/miss trials survive, the agent treats hit as a sufficient binary outcome indicator.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit becomes correct=1 and miss becomes incorrect=0; the value is repeated over time. No ignore=2 category is emitted.

ii. `np.full((1, NT), outcome[out_pos], dtype=np.int16)`.

iii. The trajectory follows the paper’s omission of no-response trials, despite the prompt requesting an ignore category.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It uses `obj.traj` x/y tracks for seven tongue landmarks across side and bottom views, their `frameTimes`, and session video/behavior synchronization fields.

ii. `TONGUE_FEATURES = [(0, "tongue"), ... (1, "bottomleft_tongue")]` and `ts = get_trial_ts(cam, trix)[:, :2, feat_idx]`.

iii. The trajectory says averaging all available tongue landmarks is a decoder-specific scalar construction after porting the paper’s kinematic preprocessing.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Each x/y coordinate is interpolated to the 10 ms grid, gradients are taken, speed magnitude is computed, and speeds are averaged across seven landmarks. NaNs are converted to zero. DLC likelihood is not used.

ii. `speeds.append(np.sqrt(xvel**2 + yvel**2))`; `out = np.nanmean(stacked, axis=0)`; `np.nan_to_num(out, nan=0.0)`.

iii. The trajectory claims this preserves the MATLAB tongue handling; it explicitly interprets zeros as non-visible placeholders.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session 50th percentile is calculated only from strictly positive values over retained trials. All bins are classified below=0 or at/above=1; there is no not-visible class.

ii. `threshold_source = kept[kept > 0]`; `binary = (kept >= threshold).astype(np.int64)`.

iii. The trajectory calls excluding zeros a deliberate decoder-specific adaptation because source MATLAB uses zero for invisible tongue samples.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by a session video offset, shifted by each trial’s go cue, and linearly interpolated directly onto `TIME`; a 400 Hz synthetic fallback is used if frame times are absent.

ii. `old_t = frame_times - vidshift - align_times[trix]`; `interp_with_nan(old_t, ts[:, 0], taxis)`.

iii. The agent says `findVideoOffset` logic matches the paper and puts camera variables on the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It uses the bottom-camera `top_paw` and `bottom_paw` x/y trajectories, frame times, and synchronization fields.

ii. `PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]`.

iii. The trajectory says mean speed across both paw landmarks defines the requested scalar.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Positions are interpolated, nearest-filled, differentiated, baseline derivative is subtracted (including the source-code y-term quirk), speed magnitudes are computed, and the two landmarks are averaged; remaining invalid values become zero.

ii. `yvel[:, trix] = yvel[:, trix] - basederiv[0]`; `out = np.nanmean(stacked, axis=0)`.

iii. The agent says it intentionally preserves the MATLAB baseline-subtraction behavior.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is split at the per-session median over retained trials and all time bins into 0/1 only; no not-visible category is retained.

ii. `paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)`.

iii. The trajectory cites the prompt’s per-session 50th-percentile requirement.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-camera frame times receive the same video-offset and go-cue correction and are interpolated onto `TIME`.

ii. `old_t = frame_times - vidshift - align_times[trix]`.

iii. The trajectory says all movement streams share the neural grid after clock correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It comes from the companion `motionEnergy_<animal>_<date>.mat` `me.data` trial traces, plus side-camera frame times and synchronization fields.

ii. `raw_data = me.data`; `raw_trials = list(np.asarray(raw_data, dtype=object).ravel())`.

iii. The trajectory says this follows `loadMotionEnergy.m` and handles its nested wrapper.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Each existing per-frame scalar trace is linearly interpolated to the 10 ms grid and missing edges/interior values are nearest-filled (or zero if wholly absent). The file’s `moveThresh` is loaded but not used for classification.

ii. `resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)`; `fill_nearest_1d(..., fill_value=0.0)`.

iii. The agent says upstream data already contains the spatially reduced signal, so only timing conversion is needed.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It is split at the median over retained trials/time bins into classes 0 and 1. No explicit no-video class is emitted.

ii. `motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)`.

iii. The trajectory cites the requested session median, while relying on filling to eliminate missing-video values.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by video offset and trial go cue, then interpolated onto the neural bin centers; missing frame times use a 400 Hz, −0.5 s fallback convention.

ii. `old_t = frame_times - vidshift - align_times[trix]`; fallback: `frame_times = np.arange(1, trial_me.size + 1) / 400.0`.

iii. The agent says this matches the source’s video alignment and ensures identical output lengths.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Non-finite spike records are discarded. Position interpolation produces NaNs outside coverage; non-tongue positions/velocities and motion energy are nearest-filled, while tongue NaNs become zero. Missing frame times use a synthetic 400 Hz clock. Trial-count mismatches raise errors.

ii. `finite = ... & np.isfinite(spike_times)`; `fill_nearest_1d(...)`; `if len(raw_trials) != ntrials: raise ValueError(...)`.

iii. The trajectory emphasizes preserving rectangular, NaN-free decoder arrays and reports validator checks as justification.

## 11-a. What are the most time-consuming steps of the code?

i. The trajectory identifies full conversion/loading and especially full decoder training as the long-running steps; within conversion, repeated trajectory interpolation and smoothing across features, trials, and units are the obvious compute-heavy operations.

ii. `for spec in SESSION_SPECS:`; `for view_index, feat_name in feature_specs:`; `for unit_pos, unit_idx in enumerate(keep_units):`.

iii. It monitored long conversion/training processes and described full training as “the long pole.”

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-feature/per-trial position interpolation, per-trial velocity calculation, per-trial motion-energy interpolation, per-unit spike binning/smoothing, smoothing columns, and final per-trial packaging are explicit loops. Some fixed-shape gradient/smoothing and packaging loops could be vectorized; ragged raw frame/spike arrays make others less straightforward.

ii. `for trix in range(ntrials):`, `for col in range(arr_filt.shape[1]):`, and `for out_pos, trial_idx in enumerate(kept_trial_indices):`.

iii. The trajectory focused on source fidelity and validation, not loop optimization.

## 11-c. What processing does the code repeat multiple times?

i. Feature names are rediscovered and every landmark independently reloads/interpolates its camera trial data. Video offset is also recomputed in `load_motion_energy` after being found for the session. The identical `TIME` input is appended once per trial.

ii. `feat_idx = find_dlc_feat_index(obj, view_index, feat_name)` inside each feature call; `vidshift = find_video_offset(obj)`; `input_trials.append(TIME[np.newaxis, :])`.

iii. No explicit justification for these repetitions appears; the trajectory prioritizes a direct MATLAB-style implementation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds seven-condition PSTHs only to reduce them to one mean firing-rate value per unit; loads and returns motion energy `moveThresh` without using it; creates a deep-copied sample dataset and extensive summaries even though the requested deliverable is the full pickle; and computes `ADVANCE_MOVEMENT` despite it being zero.

ii. `psth = np.zeros(...)`; `"moveThresh": float(...)`; `sample_data = build_sample_dataset(full_data)`.

iii. The trajectory added sample generation, statistics, and repeated validation to support debugging and documentation, not the requested downstream dataset itself.
