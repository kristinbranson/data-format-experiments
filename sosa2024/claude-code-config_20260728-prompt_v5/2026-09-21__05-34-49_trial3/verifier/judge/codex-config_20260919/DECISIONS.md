# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively enumerates sorted `sub-*` directories and every `.nwb` file in each, then opens each file with `h5py` and reads all required ophys, behavior, event, subject, and session arrays. The default is full conversion; `--sample` selects two named sessions. Exceptions are caught per session, so a failing file is logged and omitted rather than stopping the run.

ii.
```python
for subj_dir in sorted(os.listdir(data_dir)):
    if not subj_dir.startswith('sub-'):
        continue
    ...
    if nwb_file.endswith('.nwb'):
        files.append(os.path.join(subj_path, nwb_file))
...
with h5py.File(filepath, 'r') as f:
    ophys = f['processing']['ophys']
    bts = f['processing']['behavior']['BehavioralTimeSeries']
```

iii. The notes say the data contain 11 switch-task mice and 152 sessions and report that all 152 were processed. The agent chose direct HDF5 access after inspecting the NWB hierarchy.

## 1-b. How are the data split into subjects?

i. Files are grouped initially by `sub-*` directory. The definitive ID is read from each NWB's `general/subject/subject_id`; unique IDs are sorted, and each retained session receives an index into that list.

ii.
```python
data['subject_id'] = f['general']['subject']['subject_id'][()].decode()
subjects = sorted(set(r['subject_id'] for r in results))
subject_idx.append(subject_to_idx[r['subject_id']])
```

iii. The agent justified this with the observed directory structure and the expected 11 switch-task subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is one session. Each successfully processed file contributes one element to the session-level `neural`, `input`, and `output` lists; its NWB `session_id` is retained in metadata.

ii.
```python
for i, filepath in enumerate(nwb_files):
    result = process_session(filepath, ...)
    if result is not None:
        all_results.append(result)
...
neural.append(r['neural'])
```

iii. The filenames and NWB metadata encode sessions, and the reported result has the expected 152 sessions.

## 1-d. How are the data split into trials?

i. Trial starts are frames where `trial_start == 1`. For every start, the first later frame where `teleport == 1` is the exclusive end. The stored `trial number` signal is passed to the helper but not used to define boundaries.

ii.
```python
starts = np.where(trial_start_signal == 1)[0]
teleports = np.where(teleport_signal == 1)[0]
for s in starts:
    next_teleports = teleports[teleports > s]
    if len(next_teleports) > 0:
        trial_starts.append(s)
        trial_ends.append(next_teleports[0])
```

iii. The notes define a trial as trial-start through (but excluding) teleport, matching the active-track interval.

## 1-e. How are trials filtered based on quality controls?

i. Trials are removed when more than 30% of frames have `lick > 2`, when neural values are all NaN, or when fewer than two samples remain (including after dual-plane averaging). Sessions with fewer than two raw or retained trials are removed. No 50-timepoint minimum is used.

ii.
```python
frac_high = np.sum(trial_licks > 2) / n_frames
if frac_high > threshold:
    error_trials.append(i)
...
if i in lick_error_trials: continue
if np.all(np.isnan(trial_neural)): continue
if T < 2: continue
```

iii. The agent cites the paper's stuck-lick-sensor criterion and notes that its 0.3 threshold removes exactly the reported 81 trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. It is derived from raw `Fluorescence` and `Neuropil`, restricted using `iscell` and plane membership. The stored `Deconvolved` series is not used.

ii.
```python
fl = fluor_grp[pk]['data'][:].T
neu = ophys['Neuropil'][pk]['data'][:].T
fluor_list.append(fl[plane_iscell])
neuropil_list.append(neu[plane_iscell])
```

iii. The notes correctly distinguish the NWB Suite2p deconvolution from the paper's own preprocessing and say raw F/Fneu should be reprocessed.

## 2-b. How is the `neural` data processed?

i. For each trial it subtracts `0.7*Fneu`, adds back the trial mean neuropil term, computes a Gaussian-smoothed (sigma 15) 300-frame min-then-max baseline, calculates dF/F, and smooths dF/F with sigma 2. It saves this smoothed dF/F directly; it never performs the paper's OASIS deconvolution. Dual-plane sessions are then averaged in adjacent pairs.

ii.
```python
f_ = f_ - NEUROPIL_COEF * f_neu_
trial_data = trial_data + NEUROPIL_COEF * neuropil_mean
smoothed = nansmooth(trial_data, BASELINE_SMOOTH_SIGMA, axis=1)
bl = ndi.minimum_filter1d(smoothed, BASELINE_MINMAX_WINDOW, axis=-1)
bl = ndi.maximum_filter1d(bl, BASELINE_MINMAX_WINDOW, axis=-1)
dff[:, nanmask] = (f_[:, nanmask] - baseline[:, nanmask]) / np.abs(baseline[:, nanmask])
```

iii. The agent claimed this matched the reference dF/F pipeline and explicitly described dF/F as more informative for its neural-network decoder. Although its notes identified optional OASIS deconvolution, it chose not to apply it.

## 2-c. How is the `neural` data filtered based on quality controls?

i. ROIs must pass Suite2p `iscell`. Cells with Pearson correlation between dF/F and speed greater than 0.5 are removed. A session is dropped if no cells remain; trial NaNs are replaced with zero.

ii.
```python
plane_iscell = iscell[plane_mask, 0].astype(bool)
...
r = np.corrcoef(dff[c, valid], speed[valid])[0, 1]
if r > INTERNEURON_R_THRESH:
    is_interneuron[c] = True
```

iii. The notes cite manual Suite2p curation and the Methods' r > 0.5 putative-interneuron exclusion, while deliberately retaining non-place cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural arrays are sliced at the detected trial-start frame, so column zero is aligned to trial start. The slice ends immediately before teleport. No interpolation or explicit timestamp offset is applied.

ii.
```python
trial_neural = dff[:, start:end]
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The notes state that trial start is the alignment event and that active trial frames are extracted directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The target is 15.5078125 Hz (64.48 ms). Single-plane data are retained. For dual-plane files, adjacent samples of neural, position, and speed are averaged and lick is max-pooled by two; hence explicit temporal rebinning is applied.

ii.
```python
downsample = 2 if raw['is_dual_plane'] else 1
effective_rate = raw['rate'] / downsample
trial_neural = trial_neural[:, :T_new * downsample].reshape(n_cells, T_new, downsample).mean(axis=2)
...
'time_bin_size': 1000.0 / TARGET_RATE
```

iii. The agent reasoned that the two-plane scanner rate (~31 Hz) should be downsampled by two to make every session ~15.5 Hz.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from sample indices and the effective imaging rate, not from the loaded behavior timestamps.

ii.
```python
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The notes describe this as frame index relative to trial start divided by rate.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. A zero-based sequence is divided by the effective sampling rate; dual-plane sequences use the halved rate after pairwise averaging.

ii.
```python
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
```

iii. The agent intended a continuous seconds-from-start input with zero at the aligning frame.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is created after neural slicing/rebinning with exactly the neural trial's `T`, so its columns align by construction.

ii.
```python
T = trial_neural.shape[1]
time_from_start = np.arange(T, dtype=np.float32) / effective_rate
inputs = np.stack([time_from_start, env, trial_num, prev_outcome], axis=0)
```

iii. The agent relies on common frame indexing and checks format/dimensions in decoder verification.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment` series.

ii.
```python
data['environment'] = bts['environment']['data'][:].astype(np.float64)
```

iii. The notes identify values 0/1 as the two environments and negative values as ITI markers.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. For each trial, negative values are discarded and the first remaining environment value is rounded to an integer; missing trials default to 0. That scalar is broadcast over time.

ii.
```python
active_env = env_vals[env_vals >= 0]
environments.append(int(round(active_env[0])) if len(active_env) > 0 else 0)
env = np.full(T, environments[i], dtype=np.float32)
```

iii. The agent treated environment as a constant per-trial binary decoder input.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is the zero-based loop index over detected trial boundaries, not the raw `trial number` values.

ii.
```python
for i in range(n_trials):
    trial_num = np.full(T, i, dtype=np.float32)
```

iii. The agent's mapping plan calls this the integer within-session trial index.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. No transform beyond broadcasting the loop index to all timepoints. Filtering does not renumber later trials.

ii.
```python
trial_num = np.full(T, i, dtype=np.float32)
```

iii. This implements the intended per-trial continuous covariate.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from `Reward` event timestamps, behavior position timestamps, and the raw `trial number` signal via the intermediate `rewarded` array.

ii.
```python
rewarded = determine_reward_per_trial(raw['reward_timestamps'], raw['behavior_timestamps'],
                                      raw['trial_number'], trial_starts, trial_ends)
```

iii. The notes say previous outcome should reflect reward events in the preceding original trial, with the first trial set to zero.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Each reward timestamp is mapped to the globally nearest behavior frame; the integer raw trial number at that frame indexes `rewarded`. Trial i receives `rewarded[i-1]`, broadcast through time, or 0 for i=0.

ii.
```python
frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
trial_num = int(trial_number_signal[frame_idx])
rewarded[trial_num] = 1
...
prev_outcome = np.full(T, rewarded[i - 1] if i > 0 else 0, dtype=np.float32)
```

iii. The agent reports spot-checking this against actual rewards in preceding trials.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses raw `position` and a per-trial zone inferred from positions at which raw `reward_zone > 0`.

ii.
```python
zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'], trial_starts, trial_ends)
distance = compute_distance_to_reward_zone(trial_pos, zone_start, zone_end)
```

iii. The notes resolve the raw state signal into paper zones A=[80,130], B=[200,250], and C=[320,370].

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Mean position over frames with `reward_zone > 0` is assigned to the nearest zone center. Missing labels inherit the first available nearest neighbor (backward preferred at equal distance). Signed distance is position minus the near boundary before a zone, zero inside, and position minus the far boundary after it. Dual-plane positions are first averaged in pairs.

ii.
```python
mean_pos = pos[mask].mean()
dist = abs(mean_pos - center)
...
distance[before] = position[before] - zone_start
distance[inside] = 0
distance[after] = position[after] - zone_end
```

iii. The agent justified center matching using observed reward-zone-position clusters and nearest-neighbor filling for rare missing signals.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit Boolean masks implement the seven requested categories: <-50, [-50,-10), [-10,0), exactly 0, (0,10], (10,50], and >50 cm.

ii.
```python
bins[distance < -50] = 0
bins[(distance >= -50) & (distance < -10)] = 1
...
bins[distance > 50] = 6
```

iii. The agent says these boundaries directly follow the task specification.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data use identical trial slices. In dual-plane sessions both are pair-averaged with the same grouping before distance is computed.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = raw['position'][start:end]
...
trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
```

iii. The agent relies on common indices and reports spot-checking raw position and distance outputs.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It comes directly from the behavior `position` series.

ii.
```python
data['position'] = bts['position']['data'][:].astype(np.float64)
trial_pos = raw['position'][start:end]
```

iii. The notes identify this series as centimeters along the 450 cm track.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Single-plane positions are sliced unchanged before categorization. Dual-plane positions are averaged in consecutive pairs.

ii.
```python
trial_pos = trial_pos[:T_new * downsample].reshape(T_new, downsample).mean(axis=1)
pos_bins = discretize_position(trial_pos)
```

iii. The agent's goal was a common ~15.5 Hz grid across sessions.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It uses five classes with boundaries 90, 180, 270, and 360 cm; values below 90 are class 0 and values at or above 360 are class 4.

ii.
```python
bins[position < 90] = 0
bins[(position >= 90) & (position < 180)] = 1
...
bins[position >= 360] = 4
```

iii. Five 90 cm intervals exactly span the nominal 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. It is sliced on the same start/end frames and, for dual-plane sessions, rebinned with the same pairs as neural data.

ii.
```python
trial_neural = dff[:, start:end]
trial_pos = raw['position'][start:end]
```

iii. Common indexing and matching lengths provide alignment.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` series.

ii.
```python
data['lick'] = bts['lick']['data'][:].astype(np.float64)
trial_lick = raw['lick'][start:end]
```

iii. The notes identify the raw values as per-frame lick counts/states.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Dual-plane samples are max-pooled in pairs, then any value greater than zero becomes 1 and all others become 0.

ii.
```python
trial_lick = trial_lick[:T_new * downsample].reshape(T_new, downsample).max(axis=1)
lick_binary = (trial_lick > 0).astype(int)
```

iii. Max pooling preserves an event during downsampling, and thresholding implements the requested binary output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural data use the same trial boundaries and pair groups in dual-plane data.

ii.
```python
trial_neural = dff[:, start:end]
trial_lick = raw['lick'][start:end]
```

iii. The agent reports a 266-frame raw lick spot-check with exact agreement.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is inferred jointly from behavior `reward_zone` and `position` within each trial.

ii.
```python
zone_labels = determine_reward_zone(raw['position'], raw['reward_zone'], trial_starts, trial_ends)
```

iii. The raw reward-zone signal contains interaction states rather than the desired A/B/C label, motivating inference from its active positions.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Mean active-zone position is mapped to the closest fixed zone center; missing trials inherit a nearest labeled trial. A/B/C become 0/1/2 and are broadcast through the trial; any unresolved label defaults to A.

ii.
```python
zone_idx = ZONE_LABELS.index(zone_label) if zone_label in ZONE_LABELS else 0
rz_loc = np.full(T, zone_idx, dtype=int)
```

iii. The agent reports approximately equal A/B/C fractions and treats this as a sanity check.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward` timestamps, behavior position timestamps, and raw `trial number` values.

ii.
```python
rewarded = determine_reward_per_trial(raw['reward_timestamps'], raw['behavior_timestamps'],
                                      raw['trial_number'], trial_starts, trial_ends)
```

iii. Reward is an event series with its own timestamps, so the agent maps events onto behavior frames/trials.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Each event is assigned to the globally closest behavior timestamp. The raw trial number at that frame sets a binary array element, which is broadcast across that trial.

ii.
```python
frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
trial_num = int(trial_number_signal[frame_idx])
rewarded[trial_num] = 1
...
reward_out = np.full(T, rewarded[i], dtype=int)
```

iii. The agent says this produced the expected ~84–85% reward rate and passed manual checks.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Neural/behavior streams are truncated to their common minimum length. Missing reward-zone labels inherit a neighboring label, falling back to A if still unresolved. Neural NaNs are converted to zero; all-NaN trials are skipped. Files that raise exceptions are omitted, and sessions/trials failing minimum viability checks are skipped.

ii.
```python
n_min = min(n_neural, n_behav)
...
trial_neural = np.nan_to_num(trial_neural, nan=0.0).astype(np.float32)
...
except Exception as e:
    print(f"  ERROR processing {label}: {e}")
```

iii. The notes characterize truncation as a fix for small frame mismatches and report edge-case checks. They do not document the risk that broad exception handling silently yields an incomplete dataset.

## 13-a. What are the most time-consuming steps of the code?

i. The agent identifies loading large NWB arrays and the per-session dF/F maximin filtering as the major conversion costs; full decoder training is separately expensive. The script times file loading, dF/F, and total session processing.

ii.
```python
t_load = time.time() - t0
...
dff = compute_dff(...)
t_dff = time.time() - t1
```

iii. Sample timing was used to extrapolate full conversion runtime; full output size was ~8.4 GB.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Cell-by-cell interneuron correlations, trial-by-trial baseline/smoothing, reward-event nearest-frame searches, environment extraction, missing-zone filling, and trial assembly are Python loops. Some can be vectorized or replaced by `searchsorted`; variable trial lengths make complete trial vectorization less direct.

ii.
```python
for c in range(n_cells):
    r = np.corrcoef(...)[0, 1]
for rt in reward_timestamps:
    frame_idx = np.argmin(np.abs(behavior_timestamps - rt))
```

iii. The agent did not explicitly discuss these optimization opportunities in its notes; this summary is inferred from its implementation.

## 13-c. What processing does the code repeat multiple times?

i. It repeatedly slices every trial; `compute_dff` loops over trials once to copy data, again for baseline, and again to smooth. Trial boundaries are also traversed for lick errors, reward-zone inference, environments, and final output construction. Membership in the lick-error list is checked linearly for every trial.

ii.
```python
for start, end in zip(trial_starts, trial_ends): ...  # repeated in compute_dff
for i, (start, end) in enumerate(zip(trial_starts, trial_ends)): ...
for i in range(n_trials):
```

iii. The agent did not document repeated work; it prioritized faithful, readable staged processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/stores unused values (`cell_mask_list`, `offset`, reward data values, and the unused trial-number argument to `get_trial_boundaries`). It computes/retains metadata diagnostics not consumed by training. Optional plotting rereads raw slices solely for figures. Most importantly, substantial dF/F work is not discarded because dF/F itself is saved, though the reference would use it only for interneuron filtering and save deconvolved events.

ii.
```python
cell_mask_list.append(plane_iscell)
offset += n_rois
...
def get_trial_boundaries(trial_start_signal, teleport_signal, trial_number_signal):
```

iii. The notes present plotting and diagnostics as validation aids; they do not identify unnecessary discarded computations.
