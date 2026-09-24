# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively discovers every `.nwb` file below `data`, sorts the paths, opens each with `h5py`, and reads behavior and the first available `DfOverF` (preferred) or `Fluorescence` series. It retains only converted sessions having at least two trials. In sample mode it instead selects two sessions.

ii.
```python
files = sorted(Path('data').rglob('*.nwb'))
for fp in files:
    sess = convert_session(fp, show_processing=args.show_processing)
    if sess['n_trials'] >= 2:
        sessions.append(sess)
```

iii. The notes identify one NWB file per subject/session and call NWB behavior the canonical synchronized source. They report 11 subjects and 152 files/sessions and state that the full release, rather than only the paper's 77-session subset, should be converted.

## 1-b. How are the data split into subjects?

i. Subject IDs are read from each NWB file's `general/subject/subject_id`. After conversion, unique IDs are sorted and each session receives the corresponding integer `subject_idx`.

ii.
```python
subj = decode_scalar(f['general/subject/subject_id'][()])
subjects = sorted(set(s['subject'] for s in sessions))
subject_idx = np.asarray([subj_to_idx[s['subject']] for s in sessions])
```

iii. The notes say the directory/file organization contains 11 subjects and that each NWB is a subject/session, consistent with the NWB metadata.

## 1-c. How are the data split into sessions?

i. Every NWB file is treated as one session. Its session ID is read from `general/session_id`; output session order is sorted file-path order. Sessions with fewer than two retained trials are dropped.

ii.
```python
sess = decode_scalar(f['general/session_id'][()])
sess = convert_session(fp, show_processing=args.show_processing)
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The notes explicitly state that each NWB corresponds to one subject/session and report all 152 NWB sessions as retained.

## 1-d. How are the data split into trials?

i. Trial starts are rising edges of `trial_start` restricted to samples whose native `trial number` is nonnegative. If none exist, positive changes in trial number are a fallback. A trial ends at the next start, while the final trial extends to the end of the behavioral array. Trial slices are then intersected with neural timestamps.

ii.
```python
starts = rising_edges(trial_start, 0.5)
starts = starts[valid[starts]]
if len(starts) == 0:
    starts = changes[valid[changes]]
e = starts[i + 1] if i + 1 < len(starts) else len(trial_num)
```

iii. The agent notes that no standard NWB trials table exists and therefore trials must be reconstructed. It planned to check starts against trial-number changes, but chose consecutive starts rather than the teleport-defined endpoint used by the reference.

## 1-e. How are trials filtered based on quality controls?

i. Starts outside native valid trials are excluded. A trial is skipped if it contains fewer than two behavior samples or fewer than two overlapping neural samples; sessions with fewer than two resulting trials are skipped. There is no reference-style 50-timepoint trial threshold.

ii.
```python
valid = trial_num >= 0
if np.sum(nmask) < 2 or (e - s) < 2:
    continue
if sess['n_trials'] >= 2:
    sessions.append(sess)
```

iii. The notes justify excluding pre-task/teleport periods and satisfying the decoder's two-trial minimum. They do not document a scientific minimum-duration criterion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The agent uses the first data array under `processing/ophys/DfOverF`, falling back to `processing/ophys/Fluorescence`; it does not use raw fluorescence plus neuropil together.

ii.
```python
for base in ['processing/ophys/DfOverF', 'processing/ophys/Fluorescence']:
    if base in f:
        ...
        return data, ts, base, k
```

iii. The notes say processed imaging should be preferred because the reference includes dF/F processing, and describe NWB `DfOverF` as the closest available processed activity.

## 2-b. How is the `neural` data processed?

i. The selected stored trace is only oriented as neurons × time, sliced by trial timestamps, and cast to `float32`. No neuropil subtraction, within-trial maximin baseline, smoothing, OASIS deconvolution, or plane pooling logic is applied.

ii.
```python
neural_nt = neural if neural.shape[0] == n_rois else neural.T
neural_trial = neural_nt[:, nmask].astype(np.float32)
```

iii. The agent believed the released `DfOverF` could stand in for the paper's processed activity and said it matched reference preprocessing “as closely as possible.” This overlooks the paper code's derived deconvolved event signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROI quality filtering is performed. All columns/rows in the chosen response series are retained; segmentation metadata are used only to infer orientation. Every neuron is assigned CA1.

ii.
```python
n_rois, regions = load_n_rois_and_regions(f)
brain_region_idx = np.zeros(neural_nt.shape[0], dtype=np.int64)
```

iii. The notes recognize `is_putative_interneuron` and other curation code, but decide to “start from all valid neural ROIs unless” exclusion proves required. The implementation never checks `iscell` or speed correlation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Behavior trial start time `t0` and the last pre-next-start behavior time `t1` define a mask on neural timestamps. Thus each neural trial begins at the first neural timestamp at or after trial start, without interpolation or an explicit zero-time sample.

ii.
```python
t0 = bt[s]
t1 = bt[e - 1]
nmask = (neural_t >= t0) & (neural_t <= t1)
neural_trial = neural_nt[:, nmask]
```

iii. The notes state that decoder trials must be time-aligned to trial start and treat NWB timestamps as the synchronization authority.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied: stored neural samples/timestamps are retained. If neural timestamps are absent, evenly spaced timestamps spanning the behavior recording are synthesized. Critically, metadata records the bin size as NaN instead of the approximately 64.5 ms native per-plane interval.

ii.
```python
neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
...
'time_bin_size': float(np.nan),
```

iii. The agent describes imaging as about 15.5 Hz and says no resampling is needed, but never computes or saves that resolution.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from neural timestamps plus the behavior `position` timestamps used as the behavior clock and the trial-start sample.

ii.
```python
bt = b.get('position__timestamps', None)
nt = neural_t[nmask]
time_from_start = (nt - t0).astype(np.float32)
```

iii. The notes designate synchronized NWB behavior timestamps as canonical and require a continuous seconds-since-start input at neural resolution.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The behavior timestamp at the detected start is subtracted from every retained neural timestamp. There is no rounding or resampling.

ii.
```python
t0 = bt[s]
time_from_start = (nt - t0).astype(np.float32)
```

iii. The justification is the explicit decoder requirement to align time to trial start.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. It is computed directly at the selected neural timestamps, so its length and columns exactly match `neural_trial`.

ii.
```python
nt = neural_t[nmask]
neural_trial = neural_nt[:, nmask]
time_from_start = nt - t0
```

iii. The agent's sanity checks reported converted neural and inputs matched direct converter output and relied on NWB synchronization.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It comes from the behavior `environment` series.

ii.
```python
env = map_environment(np.asarray(b['environment']))
```

iii. The notes identify the raw environment stream and report that its observed values are 0/1, corresponding to ENV1/ENV2.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Session-wide values are mapped to binary by treating the largest non--1 value as 1 and other values as 0 (or simply `x > 0` if fewer than two values). The rounded within-trial median is broadcast to neural timepoints.

ii.
```python
return np.where(x == hi, 1, 0)
env_trial = np.full(nt.shape, int(np.round(np.nanmedian(env[sl]))))
```

iii. The agent wanted a consistent binary encoding even if native values were ±1, although the notes ultimately report native 0/1 values.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It uses the native behavior `trial number` evaluated at each detected trial start (`tid`).

ii.
```python
trial_ids = trial_num[starts].astype(int)
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The mapping plan explicitly chooses native trial numbering and broadcasts the per-trial scalar.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The value is cast to integer while finding bounds, then broadcast as a float row over every neural timepoint; it is not renumbered after skipped trials.

ii.
```python
bounds.append((int(trial_ids[i]), int(s), int(e)))
trialnum_trial = np.full(nt.shape, tid, dtype=np.float32)
```

iii. The agent considered native numbering the appropriate trial identity and used broadcasting for decoder format compatibility.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from timestamps in the behavior `Reward` series, behavior timestamps, and the reconstructed trial intervals.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The notes document correcting an earlier lick-based inference and using actual NWB reward-event timestamps to distinguish rewarded and omission trials.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. A trial is rewarded when any reward timestamp lies inclusively between its first and last behavior timestamps. A running `prev_out`, initialized to zero, is broadcast into the next converted trial and updated after each retained trial.

ii.
```python
rewarded = int(np.any((reward_event_timestamps >= t0) &
                      (reward_event_timestamps <= t1)))
prev_out_trial = np.full(nt.shape, prev_out)
prev_out = int(reward_outcomes[i])
```

iii. This implements omitted=0/rewarded=1 and the notes report a sanity check that previous outcome matched the prior output. The state update follows retained trials rather than necessarily the immediately preceding raw trial if one is skipped.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It uses behavior `position` and `reward_zone`. For each trial, the median position among samples where `reward_zone > 0` is inferred as a reward center; behavior positions nearest to neural timestamps are then compared with that center.

ii.
```python
nz = rz_signal[sl] > 0
trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz]))
reward_center = float(trial_reward_pos.get(tid, np.nan))
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The notes say raw reward-zone codes were initially misleading, so the agent inferred zones from positions of nonzero reward-zone samples and clustered them into three locations.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. The code subtracts the inferred trial reward-zone median position from current position. This is distance to a center point, not signed distance to the nearest boundary of the known 50-cm-wide reward zone. Missing centers yield NaN and are forced to class 0.

ii.
```python
dist = pos - reward_center
...
out[np.isnan(dist)] = 0
```

iii. The agent characterizes this as signed distance to the reward-zone “location” and considers inference from occupancy robust, but does not justify replacing distance-to-any-location-in-zone with center distance.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Explicit comparisons create classes 0–6 at -50, -10, 0, 10, and 50 cm, with `np.isclose(dist, 0)` assigned class 3. Exactly -50 goes to class 1 and exactly 50 to class 5.

ii.
```python
out[(dist >= -50) & (dist < -10)] = 1
out[(dist >= -10) & (dist < 0)] = 2
out[np.isclose(dist, 0)] = 3
out[(dist > 0) & (dist <= 10)] = 4
```

iii. The agent intended to implement the seven instruction-specified bins directly.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Each neural timestamp is mapped with `searchsorted(..., side='left')` to the first behavior sample at or after it; the corresponding position is used. This is a ceiling lookup, not interpolation or guaranteed nearest-neighbor matching.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
bidx = np.clip(bidx, 0, len(bt) - 1)
dist_bins, _ = discretize_dist_to_reward(pos[bidx], reward_center)
```

iii. The agent relies on synchronized NWB clocks and describes behavior as aligned to imaging timestamps.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the behavior `position` series.

ii.
```python
pos = np.asarray(b['position'])
abs_pos_bins, pos_edges = discretize_position(pos, valid_mask, n_bins=5)
```

iii. The mapping plan identifies raw corridor position as the source.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The minimum and maximum position among samples with nonnegative trial number and position above -400 are computed per session. Five equal-width session-data-range bins are then formed; all positions are digitized and clipped to classes 0–4.

ii.
```python
lo, hi = np.nanmin(valid_pos), np.nanmax(valid_pos)
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1]), 0, n_bins - 1)
```

iii. The agent interpreted “five equal bins” as spanning the observed valid range, rather than the instructed fixed 450-cm corridor.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Thresholds are the four interior points of the per-session linear spacing from observed minimum to maximum, not fixed 90/180/270/360 cm thresholds.

ii.
```python
edges = np.linspace(lo, hi, n_bins + 1)
idx = np.clip(np.digitize(pos, edges[1:-1], right=False), 0, n_bins - 1)
```

iii. The decision was meant to produce five equal-size bins but the notes do not address cross-session inconsistency or the explicit track limits.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Precomputed behavior-sample bins are indexed using the same ceiling `searchsorted` mapping from neural timestamps.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
out = np.vstack([..., abs_pos_bins[bidx], ...])
```

iii. The agent relies on shared NWB timing and reports spot checks against raw NWB behavior.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the behavior `lick` series.

ii.
```python
lick = np.asarray(b['lick'])
```

iii. The notes map the lick behavior stream directly to the requested binary output.

## 9-b. What processing is involved in computing `output` *Lick*?

i. At behavior indices selected for neural timestamps, values strictly greater than 0.5 become 1 and all others become 0. No paper lick-sensor correction is used.

ii.
```python
(lick[bidx] > 0.5).astype(np.int64)
```

iii. The notes identify reference lick-correction functions but choose simple binary conversion for the decoder output.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick samples are selected with the same first-at-or-after behavior index for each neural timestamp.

ii.
```python
bidx = np.searchsorted(bt, nt, side='left')
(lick[bidx] > 0.5).astype(np.int64)
```

iii. Alignment is justified by synchronized NWB streams and raw/converted spot checks.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It is derived from `reward_zone` and `position`: the median position where reward-zone signal is positive is calculated for every trial.

ii.
```python
nz = rz_signal[sl] > 0
trial_reward_pos[tid] = float(np.nanmedian(pos[sl][nz]))
```

iii. The agent found the raw reward-zone code unsuitable as a direct A/B/C label and therefore used its position-dependent occupancy.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Across each session, finite per-trial medians are KMeans-clustered into up to three clusters; sorted cluster centers define 0/1/2. Each trial is assigned its nearest center and the label is broadcast. Missing trials default to label 0.

ii.
```python
km = KMeans(n_clusters=k, random_state=0, n_init=20).fit(vals.reshape(-1, 1))
centers = np.sort(km.cluster_centers_.ravel())
labels[tid] = int(np.argmin(np.abs(np.asarray(finite_centers) - rp)))
```

iii. The notes say clustering recovered three canonical, nondegenerate A/B/C locations after direct code mapping failed.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It uses `Reward__timestamps`, the common behavior timestamp vector, and reconstructed trial bounds.

ii.
```python
reward_event_timestamps = b.get('Reward__timestamps', np.asarray([]))
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
```

iii. The notes document this as a correction over an initial lick heuristic and the appropriate representation of actual reward delivery.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Outcome is 1 if any reward timestamp is between the first and last behavior timestamp of the reconstructed trial, inclusive, otherwise 0. It is broadcast over all neural samples.

ii.
```python
rewarded = int(np.any((reward_event_timestamps >= t0) &
                      (reward_event_timestamps <= t1)))
np.full(nt.shape, int(reward_outcomes[i]), dtype=np.int64)
```

iii. The agent uses event presence to encode rewarded versus omission trials and reports a nondegenerate distribution.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Scalar bytes are decoded; missing series timestamps fall back to another behavior timestamp array; absent neural timestamps are synthesized; orientation is heuristically inferred; empty reward events produce omissions; missing reward positions default to zone class 0 and distance class 0; indices are clipped; too-short trials/sessions are skipped. Most cases silently coerce rather than warn or assert.

ii.
```python
bt = any_ts[0]
neural_t = np.linspace(bt[0], bt[-1], neural_nt.shape[1])
out[np.isnan(dist)] = 0
bidx = np.clip(bidx, 0, len(bt) - 1)
```

iii. The notes emphasize robust full-dataset completion and successful validator checks. They do not document reference-style timestamp/length assertions and treat fallbacks as pragmatic safeguards.

## 13-a. What are the most time-consuming steps of the code?

i. The likely dominant conversion costs are reading large neural arrays from all 152 NWB files and KMeans once per session; optional plotting adds work. The agent did not profile conversion. Full decoder training, although outside conversion proper, was by far the documented runtime bottleneck.

ii.
```python
data = np.asarray(g['data'][:])
km = KMeans(...).fit(vals.reshape(-1, 1))
```

iii. The notes call sample conversion fast and full conversion manageable in minutes, leave runtime tables blank, and devote most trajectory time to 200-epoch decoder training.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Per-trial reward-position and reward-outcome loops, reward-label assignment, and the main per-trial conversion loop could partly be vectorized or precomputed. File/session iteration is naturally sequential in the implementation.

ii.
```python
for tid, s, e in trials:
    ...
for i, (tid, s, e) in enumerate(trials):
    ...
```

iii. The notes' “Code inefficiencies” and “speedups” fields are blank, so the agent provides no explicit optimization justification.

## 13-c. What processing does the code repeat multiple times?

i. Trial slices and time bounds are revisited when inferring reward positions, inferring reward outcomes, and constructing output trials. Full position is digitized before only trial-indexed values are used. In sample selection, files are opened to inspect environment and then selected files are reopened for conversion.

ii.
```python
trial_reward_pos = infer_trial_reward_positions(pos, rz_raw, trials)
reward_outcomes = infer_reward_outcomes_from_events(trials, bt, reward_event_timestamps)
for i, (tid, s, e) in enumerate(trials):
```

iii. The notes do not identify repeated conversion work; they focus instead on validation and decoder training.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads ROI `location` values but never uses them, loads `teleport` without using it, computes and returns `pos_edges` but discards them, stores session ID/zone centers/neural source only in temporary session dictionaries, and optional plots do not affect conversion. It also digitizes the entire position recording although only retained trial samples are output.

ii.
```python
teleport = np.asarray(b['teleport'])
abs_pos_bins, pos_edges = discretize_position(...)
n_rois, regions = load_n_rois_and_regions(f)
```

iii. The agent gives no justification; its efficiency documentation fields are blank.
