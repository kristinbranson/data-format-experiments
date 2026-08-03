# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent globbed every `data/sub-*/*.nwb` file, sorted the paths, and processed each file as one session. Within each file it opened the NWB, read one deconvolved ROI response series (`plane0`) and a fixed set of behavioral time series, then built per-trial arrays in a Python loop. It did not follow the reference survey-then-convert workflow, and it did not load all deconvolved planes.

ii. ```python
 def get_files(sample=False):
     files = sorted(Path('data').glob('sub-*/*.nwb'))
     return files[:2] if sample else files
 
 def process_file(fpath):
     with NWBHDF5IO(str(fpath), 'r', load_namespaces=True) as io:
         nwb = io.read()
         bts = nwb.processing['behavior'].data_interfaces['BehavioralTimeSeries'].time_series
         deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
 ```

iii. In `CONVERSION_NOTES.md`, the agent justified this by saying the released data are NWB files under `data/sub-*`, that trial variables live in `BehavioralTimeSeries`, and that deconvolved activity should be the primary neural signal. The notes also describe a simpler direct NWB-to-decoder conversion rather than reusing the reference multi-step session-building pipeline.

## 1-b. How are the data split into subjects?

i. Subjects are taken from `nwb.subject.subject_id` when available, otherwise from the parent directory name. After processing all sessions, unique subject IDs are sorted and stored in `subjects`, and `subject_idx` maps each session to its subject.

ii. ```python
 subj = getattr(nwb.subject, 'subject_id', fpath.parent.name)
 ...
 subjects = sorted({s['subject'] for s in sessions})
 subject_to_idx = {s: i for i, s in enumerate(subjects)}
 'subject_idx': np.array([subject_to_idx[s['subject']] for s in sessions], dtype=np.int64),
 ```

iii. The notes say the data are organized under per-subject directories `data/sub-*`, and the goal was to preserve those mice as the subject axis in the output format.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The script records `nwb.session_id` but uses file iteration order to define session order in the output lists.

ii. ```python
 files = sorted(Path('data').glob('sub-*/*.nwb'))
 ...
 sess_id = nwb.session_id
 ...
 for i, f in enumerate(files, 1):
     sess = process_file(f)
     if len(sess['neural']) >= 2:
         sessions.append(sess)
 ```

iii. In the notes, the agent states that there are 152 session files total and that each session is a single `*_behavior+ophys.nwb` file containing both behavior and ophys.

## 1-d. How are the data split into trials?

i. Trials are defined from the `trial_start` behavioral time series. The script finds all positive `trial_start` samples and uses the next `trial_start` index, or the end of the recording for the last trial, as the trial end. It does not use `teleport` to terminate trials.

ii. ```python
 def trial_bounds_from_trial_start(trial_start):
     starts = np.flatnonzero(trial_start > 0)
     bounds = []
     for i, s in enumerate(starts):
         e = starts[i + 1] if i + 1 < len(starts) else len(trial_start)
         if e > s:
             bounds.append((s, e))
     return bounds
 ```

iii. The notes justify trial alignment around `trial_start` and say trial boundaries should use `trial_start` as onset and the next `trial_start` or `teleport` as the end. The code only implements the `next trial_start` part.

## 1-e. How are trials filtered based on quality controls?

i. Trial filtering is minimal. The code skips windows shorter than 2 samples, skips windows whose `trial number` is entirely negative, and skips windows where all positions are `<= -100`.

ii. ```python
 for ti, (s, e) in enumerate(bounds):
     if e - s < 2:
         continue
     if np.nanmax(trial_num[s:e]) < 0:
         continue
     if np.all(position[s:e] <= -100):
         continue
 ```

iii. The notes describe this as baseline exclusion: invalid baseline periods have `trial number`/`environment` of `-1` or sentinel positions and should be excluded. The agent did not document any stronger trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is taken directly from `processing['ophys']['Deconvolved'].roi_response_series['plane0']`.

ii. ```python
 deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
 neural = np.asarray(deconv.data[:], dtype=np.float32)
 ```

iii. The notes explicitly say to use deconvolved activity because the paper/methods describe deconvolved activity matrices as the neural signal for the downstream analyses.

## 2-b. How is the `neural` data processed?

i. The script reads the deconvolved matrix as `(time, neurons)`, derives timestamps from `deconv.timestamps` or `deconv.rate`, slices it by trial bounds, transposes each trial to `(neurons, time)`, and casts to `float32`. No resampling, normalization, concatenation across planes, or reference-style interpolation is applied.

ii. ```python
 neural = np.asarray(deconv.data[:], dtype=np.float32)
 timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
 ...
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say to use the native shared sampling grid and to make per-trial `(neurons, time)` matrices aligned to trial start.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not apply neuron-level QC to the neural matrix. It does not use `iscell`, does not drop ROIs based on metadata, and does not crop neural samples to match behavior except indirectly through trial slicing.

ii. ```python
 deconv = nwb.processing['ophys'].data_interfaces['Deconvolved'].roi_response_series['plane0']
 neural = np.asarray(deconv.data[:], dtype=np.float32)
 ...
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes focus on choosing deconvolved activity and excluding baseline/trial artifacts; they do not document any neuron-quality curation step in the implemented converter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Within each trial window, neural time zero is the first sample at `trial_start`, and the neural slice is taken over the same `[s:e]` window used for the other variables.

ii. ```python
 rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say temporal alignment should be `trial_start`, matching the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output keeps the native deconvolved sample grid. The metadata time bin is computed from the deconvolution rate as `1000 / rate` ms, and no rebinning is applied.

ii. ```python
 rate = float(deconv.rate) if getattr(deconv, 'rate', None) is not None else float(1.0 / np.median(np.diff(timestamps)))
 ...
 'time_bin_size': float(1000.0 / sessions[0]['rate']) if sessions else None,
 ```

iii. The notes say to use the native shared sampling grid at about 15.5 Hz because behavior and deconvolved traces appeared synchronized.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from the deconvolved timestamps for the selected trial window.

ii. ```python
 timestamps = np.asarray(deconv.timestamps[:]) if deconv.timestamps is not None else np.arange(neural.shape[0]) / float(deconv.rate)
 ...
 rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
 ```

iii. The notes justify this as using the native shared time base because the agent believed neural and behavior streams were already synchronized.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. For each trial, the first timestamp of the trial slice is subtracted from every timestamp in that slice, producing a continuous time-from-trial-start vector in seconds.

ii. ```python
 rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
 ```

iii. The notes say trials are aligned to `trial_start`, so this variable should begin at 0 for each trial.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is aligned sample-for-sample by using the same trial indices `[s:e]` used for the neural slice.

ii. ```python
 rel_time = (timestamps[s:e] - timestamps[s]).astype(np.float32)
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say all trial variables share a common time base and should be segmented on the same trial windows.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the `environment` behavioral time series.

ii. ```python
 env = np.asarray(behavior['environment'][0]).ravel()
 ```

iii. The notes say valid task samples use environment codes `0/1` directly and baseline samples use `-1` and should be excluded.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The code takes the nonnegative environment values within a trial, computes their median, rounds to an integer label, and broadcasts that single label across all timepoints in the trial. If there are no valid values, it falls back to 0.

ii. ```python
 env_valid = env[s:e][env[s:e] >= 0]
 env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
 ...
 np.full(e - s, env_label, dtype=np.float32)
 ```

iii. The notes describe environment as a per-trial binary input that should be broadcast across timepoints after excluding baseline `-1` samples.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is derived from the `trial number` behavioral time series.

ii. ```python
 trial_num = np.asarray(behavior['trial number'][0]).ravel()
 ```

iii. The notes say to use the within-session trial index as the trial-number input.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The code keeps only nonnegative trial-number samples within a trial, takes their median, and broadcasts that scalar across the whole trial. If none are valid, it falls back to the loop index `ti`.

ii. ```python
 tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
 tr_label = float(np.median(tr_valid)) if len(tr_valid) else float(ti)
 ...
 np.full(e - s, tr_label, dtype=np.float32)
 ```

iii. The notes say trial number is a per-trial continuous label that should be broadcast across timepoints.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps, converted to a per-trial reward flag and then shifted by one trial.

ii. ```python
 reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
 rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
 reward_outcomes.append(rew)
 ```

iii. The notes justify this by pointing out that the reference analyses distinguish rewarded and omission trials, so previous reward outcome should come from whether a reward event occurred in the previous trial window.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each trial gets a reward flag `rew`. The previous-trial input is `reward_outcomes[-2]` once at least two trials have been seen; otherwise it is 0 on the first trial. That scalar is broadcast across the whole trial.

ii. ```python
 rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
 reward_outcomes.append(rew)
 ...
 prev_rew = reward_outcomes[-2] if len(reward_outcomes) >= 2 else 0
 ...
 np.full(e - s, prev_rew, dtype=np.float32)
 ```

iii. The notes say the first trial should use a safe default such as 0 and later trials should carry the previous trial’s rewarded/omitted status.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus an inferred trial reward-zone identity, which itself comes from the `reward_zone` time series when available and falls back to reward-event position or trial median position.

ii. ```python
 position = np.asarray(behavior['position'][0]).ravel()
 reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
 reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
 ```

iii. The notes say raw reward-zone codes `1-6` should be collapsed to three physical locations A/B/C by using the physical position of active reward-zone samples or reward delivery rather than the raw code itself.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. For each trial, the agent infers one zone label. If there are nonzero `reward_zone` samples, it takes the most common raw code and maps that code to a median position; if not, it falls back to the reward position or the trial’s median position. That inferred zone is collapsed to A/B/C, mapped to a fixed center (`90`, `205`, or `325` cm), and distance is computed as `position - zone_center`.

ii. ```python
 zone_centers = reward_zone_centers(position, reward_zone)
 ...
 rz_vals = reward_zone[s:e]
 rz_nz = rz_vals[rz_vals > 0]
 if len(rz_nz):
     code = int(np.bincount(rz_nz.astype(int)).argmax())
     center = zone_centers.get(code, np.nan)
 elif rew:
     ridx = np.searchsorted(timestamps, reward_ts[(reward_ts >= t0) & (reward_ts <= t1 + 1e-9)][0])
     ridx = min(ridx, len(position) - 1)
     center = float(position[ridx])
 else:
     center = np.nanmedian(position[s:e])
 zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
 zone_center_lookup = {0: 90.0, 1: 205.0, 2: 325.0}
 d = position[s:e] - zone_center_lookup[zone_label]
 ```

iii. The notes justify this as a practical collapse from six sparse NWB reward-zone codes to three physical reward locations along the corridor.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. The script bins the signed distance with a custom scalar function: `< -50`, `[-50,-10)`, `[-10,0)`, exactly `0`, `(0,10]`, `(10,50]`, and `> 50`.

ii. ```python
 def distance_bin(d):
     if d < -50:
         return 0
     if d < -10:
         return 1
     if d < 0:
         return 2
     if d == 0:
         return 3
     if d <= 10:
         return 4
     if d <= 50:
         return 5
     return 6
 ...
 out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
 ```

iii. The notes say the discretization was chosen to match the decoder task bins, though the sample-validation notes explicitly observed that the exact `0` bin was absent and might need refinement.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. The distance vector is computed from `position[s:e]` using the same trial slice as the neural data.

ii. ```python
 d = position[s:e] - zc
 out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
 ...
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say all outputs should be built on the same trial windows as the neural activity after aligning trials to `trial_start`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavioral time series.

ii. ```python
 position = np.asarray(behavior['position'][0]).ravel()
 ```

iii. The notes map `position` directly to the absolute-position decoder output.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The trial slice `position[s:e]` is passed directly into a binning helper. There is no smoothing or interpolation.

ii. ```python
 out_pos = pos_bins(position[s:e])
 ```

iii. The notes say this output should be a time-varying discretization of position over the corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. The code makes 5 equal-width bins over `[0, 450)` by using `np.linspace(lo, hi, 6)`, clipping values to that range, and digitizing against the 4 interior edges.

ii. ```python
 def pos_bins(pos, lo=0.0, hi=450.0):
     edges = np.linspace(lo, hi, 6)
     out = np.digitize(np.clip(pos, lo, hi - 1e-6), edges[1:-1], right=False)
     return out.astype(np.int64)
 ```

iii. The notes justify this as matching the task instruction to use five equal-sized bins across the corridor.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The absolute-position vector uses `position[s:e]`, the same trial slice used for the neural matrix.

ii. ```python
 out_pos = pos_bins(position[s:e])
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say outputs and inputs should share the trial segmentation used for the neural data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavioral time series.

ii. ```python
 lick = np.asarray(behavior['lick'][0]).ravel()
 ```

iii. The notes say lick values in NWB are counts/amplitudes and should be binarized for the decoder.

## 9-b. What processing is involved in computing `output` *Lick*?

i. The lick signal is thresholded at `> 0` and cast to `int64` to create a binary no/yes vector.

ii. ```python
 out_lick = (lick[s:e] > 0).astype(np.int64)
 ```

iii. The notes explicitly say to convert nonzero lick values to 1 for the decoder output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. The lick slice uses the same `[s:e]` trial window as the neural data.

ii. ```python
 out_lick = (lick[s:e] > 0).astype(np.int64)
 trial_neural = neural[s:e, :].T.astype(np.float32)
 ```

iii. The notes say all time-varying outputs should be sliced on the same synchronized trial grid.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` plus fallback information from `Reward` event times and `position`.

ii. ```python
 reward_zone = np.asarray(behavior['reward_zone'][0]).ravel()
 reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
 position = np.asarray(behavior['position'][0]).ravel()
 ```

iii. The notes say reward-zone identity A/B/C should be inferred from physical reward-zone location rather than taken directly from the sparse raw codes `1-6`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The agent infers one zone center per trial, collapses that position into A/B/C by thresholds `<150`, `<260`, or higher, and then broadcasts the resulting label across all timepoints in the trial.

ii. ```python
 def collapse_zone_position_to_abc(pos):
     if pos < 150:
         return 0
     if pos < 260:
         return 1
     return 2
 ...
 zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
 out_rz = np.full(e - s, zone_label, dtype=np.int64)
 ```

iii. The notes justify this as collapsing six NWB reward-zone codes onto three physical locations near `~80-100`, `~200`, and `~320-330` cm.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from `Reward` event timestamps.

ii. ```python
 reward_ts = np.asarray(behavior['Reward'][1]).ravel() if behavior['Reward'][1] is not None else np.array([])
 ```

iii. The notes say the paper distinguishes rewarded and omission trials, so per-trial reward outcome should come from whether a `Reward` event occurred during the trial.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The script marks a trial as rewarded if any reward timestamp falls within the trial window `[t0, t1]`, converts that to `0/1`, and broadcasts it across the trial.

ii. ```python
 rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
 out_rew = np.full(e - s, rew, dtype=np.int64)
 ```

iii. The notes justify this exactly: define reward outcome by whether a `Reward` event occurs within the trial.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is ad hoc. The code skips obviously invalid trials, ignores negative baseline codes, falls back to reward-event position or trial median position when `reward_zone` is missing, defaults environment to 0 and previous reward to 0 when necessary, uses `225.0` as a default reward-zone position when the inferred center is not finite, and catches failures in region inference by returning `'unknown'`.

ii. ```python
 if np.nanmax(trial_num[s:e]) < 0:
     continue
 if np.all(position[s:e] <= -100):
     continue
 ...
 center = zone_centers.get(code, np.nan)
 ...
 else:
     center = np.nanmedian(position[s:e])
 zone_label = collapse_zone_position_to_abc(center if np.isfinite(center) else 225.0)
 ...
 env_label = int(np.round(np.median(env_valid))) if len(env_valid) else 0
 ...
 except Exception:
     return 'unknown'
 ```

iii. The notes describe these mostly as baseline exclusion and sensible defaults, but they do not document a systematic missing-data strategy beyond those fallbacks.

## 13-a. What are the most time-consuming steps of the code?

i. The heavy steps are repeated NWB file reads and materializing large deconvolved/behavior arrays for every session, followed by the per-trial loop that builds outputs. The agent also ran full conversion on all 152 sessions.

ii. ```python
 for i, f in enumerate(files, 1):
     st = time.time()
     sess = process_file(f)
     ...
     print(f'processed {i}/{len(files)} {f.name}: trials={len(sess["neural"])} neurons={(sess["brain_region_idx"].shape[0])} time={time.time()-st:.2f}s', flush=True)
 ```

iii. In Step 6 of `CONVERSION_NOTES.md`, the agent explicitly noted that repeated full NWB reads could be slow for all 152 sessions and that conversion time should be monitored.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several Python-level loops that could be vectorized: the per-trial loop over all trial bounds, the list-comprehension binning for distance and speed, and the reward-zone center construction loop over raw reward-zone codes.

ii. ```python
 for ti, (s, e) in enumerate(bounds):
     ...
     out_dist = np.array([distance_bin(float(x)) for x in d], dtype=np.int64)
     out_speed = np.array([speed_bin(float(max(0.0, x))) for x in speed[s:e]], dtype=np.int64)
 
 def reward_zone_centers(position, reward_zone):
     centers = {}
     for code in sorted(c for c in np.unique(reward_zone) if c > 0):
         ...
 ```

iii. The notes claim the script uses vectorized slicing and a single pass per session, but they only acknowledge slowness at the file-read level and do not discuss these remaining Python loops.

## 13-c. What processing does the code repeat multiple times?

i. The code repeatedly slices the same trial windows out of different arrays, repeatedly recomputes reward-range masks inside each trial, and repeatedly constructs full-length broadcast arrays for per-trial scalar inputs and outputs.

ii. ```python
 rew = int(np.any((reward_ts >= t0) & (reward_ts <= t1 + 1e-9)))
 ...
 env_valid = env[s:e][env[s:e] >= 0]
 tr_valid = trial_num[s:e][trial_num[s:e] >= 0]
 ...
 np.full(e - s, env_label, dtype=np.float32)
 np.full(e - s, tr_label, dtype=np.float32)
 np.full(e - s, prev_rew, dtype=np.float32)
 np.full(e - s, zone_label, dtype=np.int64)
 np.full(e - s, rew, dtype=np.int64)
 ```

iii. The notes say the script was intentionally simple and single-pass. They do not justify the repeated per-trial recomputation beyond ease of implementation.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script contains an unused helper (`contiguous_segments`), accumulates `trial_zone_labels` but never uses it afterward, infers region strings only to collapse every neuron in a session to one region index later, and exposes a `--show-processing` flag that is only a placeholder and never produces the requested plots.

ii. ```python
 def contiguous_segments(mask):
     ...
 
 sess_neural, sess_input, sess_output = [], [], []
 reward_outcomes = []
 trial_zone_labels = []
 ...
 trial_zone_labels.append(zone_label)
 ...
 ap.add_argument('--show-processing', action='store_true', help='Placeholder; save processing plots for up to 2 sessions')
 ```

iii. The notes acknowledge that `--show-processing` is still a placeholder and that visualizations were not implemented, and they never identify a downstream use for `trial_zone_labels` or `contiguous_segments`.
