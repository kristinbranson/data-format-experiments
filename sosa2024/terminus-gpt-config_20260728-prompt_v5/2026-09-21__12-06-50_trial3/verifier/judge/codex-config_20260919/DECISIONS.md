# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent recursively finds every `.nwb` file under `/app/data`, sorts the paths, and processes each file as one session with `h5py`. In full/default mode all 152 files are used; sample mode keeps the first two.

ii.
```python
def find_nwb_files():
    return sorted(Path('/app/data').rglob('*.nwb'))

files = find_nwb_files()
if args.sample:
    files = files[:2]
...
with h5py.File(fpath, 'r') as f:
```

iii. The notes say the data are NWB files, report 11 subjects and 152 sessions, and state that the full conversion produced 152 sessions. The agent chose direct HDF5 access for speed and loaded complete session arrays into memory.

## 1-b. How are the data split into subjects?

i. Subject identity is read from each NWB file's `general/subject/subject_id`; a dictionary assigns the first encountered ID an index. However, the implementation accidentally appends each new subject twice to `subjects`.

ii.
```python
subj = read_scalar(f, 'general/subject/subject_id')
...
if subj not in subject_to_idx:
    subject_to_idx[subj] = len(subjects)
    subjects.append(subj)
    subjects.append(subj)
```

iii. The notes identify standard NWB subject metadata as the source and claim the converted data contain 11 subjects, but they do not document the duplicate-append bug.

## 1-c. How are the data split into sessions?

i. Every NWB file is treated as a session. Its session ID is read from `general/session_id`, with the filename stem as a fallback; sessions with fewer than two retained trials are skipped.

ii.
```python
sess_id = read_scalar(f, 'general/session_id', fpath.stem)
...
for i, fpath in enumerate(files):
    subj, sess_id, n_neurons, n_list, in_list, out_list = convert_session(fpath)
    if len(n_list) < 2:
        continue
```

iii. The agent observed one recording/session per NWB file and used the target-format requirement that a session contain at least two trials.

## 1-d. How are the data split into trials?

i. Trial starts are rising edges of `trial_start`; ends are the first later rising edge of `teleport`. A candidate is retained only if its interval contains a nonnegative stored trial number. The end sample is excluded.

ii.
```python
starts = np.where(np.diff(trial_start.astype(int), prepend=0) > 0)[0]
ends = np.where(np.diff(teleport.astype(int), prepend=0) > 0)[0]
for s in starts:
    e_candidates = ends[ends > s]
    ...
    e = int(e_candidates[0])
    if np.any(trial_number[s:e] >= 0):
        bounds.append((int(s), int(e)))
```

iii. The notes say the reference code uses trial-start and teleport indices, while `trial number == -1` identifies off-trial samples. This motivated start-to-teleport intervals plus the validity check.

## 1-e. How are trials filtered based on quality controls?

i. Within every start-to-teleport interval, the agent keeps only samples with nonnegative `trial number` and positive `scanning`. It drops a trial only when fewer than two such samples remain, and drops a session if fewer than two trials remain.

ii.
```python
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
if valid.sum() < 2:
    continue
...
if len(n_list) < 2:
    continue
```

iii. The notes characterize `trial number = -1` as off-trial and planned to exclude it. They also state scanning appeared valid throughout. No paper-based justification is given for the two-sample threshold; the reference's 50-timepoint rule was not adopted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes directly from `processing/ophys/Deconvolved/plane0/data`. `Fluorescence/plane0/data` is also loaded, but only to assert equal shape and is then discarded.

ii.
```python
neural = np.asarray(f['processing/ophys/Deconvolved/plane0/data'], dtype=np.float32)
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
...
assert neural.shape == fluor.shape
```

iii. The agent's notes say the Methods use deconvolved activity and therefore prefer the NWB `Deconvolved` field, with fluorescence merely a fallback. This overlooks that the paper code recomputes its analyzed events from fluorescence and neuropil.

## 2-b. How is the `neural` data processed?

i. The stored plane-0 deconvolved matrix is trimmed to a common session length, sliced at retained trial indices, transposed from time-by-ROI to ROI-by-time, and cast to `float32`. There is no neuropil correction, dF/F calculation, smoothing, OASIS recomputation, or multi-plane pooling.

ii.
```python
common_len = min(lengths)
neural = neural[:common_len]
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes justify this as using the Methods' “deconvolved activity” and the directly aligned NWB time base. They do not establish that the stored suite2p deconvolution is the paper's final analyzed event signal.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No ROI quality filtering is applied. All columns in plane 0 are retained; neither the NWB `iscell` curation nor the paper's speed-correlation putative-interneuron exclusion is used.

ii.
```python
n_time, n_neurons = neural.shape
...
neu = neural[idx].T.astype(np.float32)
brain_region_idx.append(np.zeros(n_neurons, dtype=np.int64))
```

iii. The notes noticed `is_putative_interneuron` in the reference repository but left the inclusion/exclusion criteria unresolved. Their reported raw ROI counts versus converted counts were accepted without completing the reference curation comparison.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial start is the alignment event. Neural samples are selected with the same retained indices as behavior, beginning at the first valid/scanning sample in the start-to-teleport interval; no fixed window or interpolation is used.

ii.
```python
idx = np.where(valid)[0] + s
...
neu = neural[idx].T.astype(np.float32)
...
'temporal_alignment_event': 'trial start',
'off_start': 0.0,
```

iii. The agent states that neural and behavior arrays have matching lengths and uses the NWB time base directly for trial-start-aligned temporal decoding.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning or resampling is applied; samples remain at the stored cadence (observed in notes as about 0.0645 s). Nevertheless, metadata incorrectly records `time_bin_size` as `None`.

ii.
```python
ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
...
neu = neural[idx].T.astype(np.float32)
...
'time_bin_size': None,
```

iii. The notes explicitly choose the directly aligned NWB time base and report approximately 0.06448 s spacing. They do not justify leaving the required millisecond metadata unset.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. It is derived from `processing/behavior/BehavioralTimeSeries/position/timestamps`.

ii.
```python
ts = np.asarray(f['processing/behavior/BehavioralTimeSeries/position/timestamps'], dtype=np.float64)
```

iii. The agent found behavior and neural streams sample-aligned and chose the position timestamps as the common time base.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The timestamp of the first retained trial sample is subtracted from every retained timestamp, then values are cast to `float32`.

ii.
```python
t0 = ts[idx[0]]
t_rel = (ts[idx] - t0).astype(np.float32)
```

iii. This directly implements seconds elapsed from trial start; the notes report an observed sample range starting at 0.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. The identical `idx` array slices timestamps and neural data, so columns correspond one-to-one.

ii.
```python
t_rel = (ts[idx] - t0).astype(np.float32)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The notes say examined neural and behavioral arrays have matched sample counts and that raw-versus-converted spot checks passed.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. It is derived from the in-trial `environment` behavior series. The first valid value is recorded per trial.

ii.
```python
env_vals = beh['environment'][s:e][valid]
env_per_trial.append(int(first_valid(env_vals)) if env_vals.size else -1)
```

iii. The notes identify `environment` as the direct ENV1/ENV2 source and `-1` as off-trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The first finite in-trial environment value is used and broadcast across time. Although a binary mapping is computed, it is never applied; missing/negative values default to zero.

ii.
```python
env_map = compute_env_mapping(env_per_trial)
...
np.full_like(t_rel, float(env_per_trial[i] if env_per_trial[i] >= 0 else 0), dtype=np.float32),
```

iii. The notes planned to map valid codes to binary after checking sessions, but the final code assumes the raw nonnegative codes already have the desired encoding.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. It is taken from the first valid value of the raw `trial number` behavior series, with the loop count as a fallback.

ii.
```python
tn_vals = beh['trial number'][s:e][valid]
trial_nums.append(float(first_valid(tn_vals)) if tn_vals.size else len(trial_nums))
```

iii. The notes say they would use the stored observed trial number. They did not resolve the reference finding that this series disagrees with trial-start boundaries.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. The chosen raw/fallback scalar is converted to float and broadcast across all trial timepoints. It is not renumbered after filtering.

ii.
```python
np.full_like(t_rel, trial_nums[i], dtype=np.float32),
```

iii. The agent viewed it as a per-trial variable and therefore made it constant over the trial.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. It is derived from the capitalized `Reward` behavior series and its timestamps, when present.

ii.
```python
reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None
```

iii. The notes document fixing an initial all-zero result caused by the capitalized field name and separate reward time base.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Current-trial reward is `any(Reward > 0)` within the trial's timestamp interval (or aligned slice). The preceding entry is broadcast for trial `i`; the first trial is assigned zero.

ii.
```python
m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
...
np.full_like(t_rel, reward_outcomes[i-1] if i > 0 else 0, dtype=np.float32),
```

iii. The notes explicitly adopt the reference-code rule “any reward > 0” and shift it by one trial, with omitted/first trial encoded zero.

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. It is derived from `position` plus an inferred trial reward-zone center. The center is the median position among samples where raw `reward_zone > 0`; if unavailable, median trial position is substituted.

ii.
```python
pos_vals = beh['position'][s:e][valid]
rz_vals = beh['reward_zone'][s:e][valid]
m = rz_vals > 0
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
...
if rz_center is None or not np.isfinite(rz_center):
    rz_center = float(np.nanmedian(pos))
```

iii. The notes call this mapping provisional. They say the raw reward-zone stream is not a direct A/B/C label and infer centers from positions where it is active.

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance is calculated as `position - inferred center`. Thus it is zero only at the center, rather than zero everywhere inside the reward-zone interval.

ii.
```python
dist = pos - float(rz_center)
...
discretize_distance(dist),
```

iii. The agent intended “distance to reward-zone location,” but did not use the known A/B/C boundaries or the instruction's distance to any location in the zone. The notes retain this as provisional even after full conversion.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. A vectorized function assigns seven integer classes. Most bounds match the requested values, but exactly `-10` is put in class 1 rather than class 2.

ii.
```python
out[d < -50] = 0
out[(d >= -50) & (d <= -10)] = 1
out[(d > -10) & (d < 0)] = 2
out[np.isclose(d, 0)] = 3
out[(d > 0) & (d <= 10)] = 4
out[(d > 10) & (d <= 50)] = 5
out[d > 50] = 6
```

iii. The output labels mirror the requested seven bins. No justification is given for the `-10` endpoint deviation; `np.isclose` was used to isolate zero.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position and neural data are sliced by the same `idx`; distance is computed for exactly those positions.

ii.
```python
pos = beh['position'][idx].astype(np.float32)
dist = pos - float(rz_center)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The agent relies on the matched NWB sampling grid and reports raw-versus-converted alignment spot checks.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. It is derived directly from the `position` behavior series.

ii.
```python
pos = beh['position'][idx].astype(np.float32)
```

iii. The notes identify the NWB position stream and the 450 cm track as the direct source.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Position is clipped to `[0, 450]` and passed to the categorical discretizer. The clipping does not change the resulting end-bin class relative to open-ended bins, but discards the original out-of-range magnitude.

ii.
```python
discretize_position(np.clip(pos, 0, TRACK_LEN)),
```

iii. The agent chose the main 450 cm corridor and excluded teleport periods; the notes do not separately justify clipping.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. It is assigned to five bins at 90, 180, 270, and 360 cm.

ii.
```python
out[pos < 90] = 0
out[(pos >= 90) & (pos < 180)] = 1
out[(pos >= 180) & (pos < 270)] = 2
out[(pos >= 270) & (pos < 360)] = 3
out[pos >= 360] = 4
```

iii. This follows the requested five equal-width bins over a 450 cm track.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. The same retained sample indices slice position and neural activity.

ii.
```python
pos = beh['position'][idx].astype(np.float32)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The agent relies on matched sample lengths/timing and reports spot-checking converted slices against raw data.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. It is derived from the `lick` behavior series.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
```

iii. The notes observed raw lick values greater than one and identified them as lick counts/events.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Any positive value is mapped to 1; all other values are 0.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
```

iii. This implements the decoder's binary no/yes requirement; specialized lick corrections in the paper were considered analysis-specific.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Lick and neural activity use the identical retained sample indices.

ii.
```python
lick = (beh['lick'][idx] > 0).astype(np.int64)
...
neu = neural[idx].T.astype(np.float32)
```

iii. The agent states all behavior and neural arrays are natively aligned and reports a lick slice `np.allclose` spot check.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. It uses both `reward_zone` and `position`: per-trial center is the median position where `reward_zone > 0`.

ii.
```python
pos_vals = beh['position'][s:e][valid]
rz_vals = beh['reward_zone'][s:e][valid]
m = rz_vals > 0
rz_centers.append(float(np.nanmedian(pos_vals[m])) if np.any(m) else None)
```

iii. The notes say the raw reward-zone values are not direct A/B/C labels, so location must be inferred spatially.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Unique rounded centers in each session are sorted, centers within 30 cm are merged, and the first three clusters are mapped in order to A/B/C. Each trial is assigned its nearest discovered center; absent mappings default to A. This is session-local and does not use the known zone ranges or cross-session continuity.

ii.
```python
rz_map = compute_rz_mapping_from_centers(rz_centers)
...
nearest = min(rz_map.keys(), key=lambda x: abs(x - rz_center))
rz_cat = rz_map[nearest]
...
np.full(t_rel.shape, rz_cat, dtype=np.int64),
```

iii. The notes describe this as a provisional fix after the initial result was constant/incorrect. Strong decoder accuracy was treated as supporting evidence, despite no reference-equivalent validation.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. It is derived from the `Reward` behavior series and, when available, its separate timestamps.

ii.
```python
reward = np.asarray(f[f'{base}/Reward/data']) if f'{base}/Reward/data' in f else None
reward_ts = np.asarray(f[f'{base}/Reward/timestamps']) if f'{base}/Reward/timestamps' in f else None
```

iii. The notes document discovering the capitalized field and separate timestamp base while fixing an initially all-zero output.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The trial is 1 if any positive reward falls between its start and last timestamp (inclusive), otherwise 0; the value is broadcast across the trial. Missing reward data defaults to 0.

ii.
```python
m_rew = (reward_ts >= t_start) & (reward_ts <= t_end)
reward_outcomes.append(int(np.any(reward[m_rew] > 0)))
...
np.full(t_rel.shape, reward_outcomes[i], dtype=np.int64),
```

iii. The agent says this matches the reference `isreward` concept. It notes the resulting class balance and near-chance decoder accuracy but retains the representation.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. All neural, timestamp, and behavior arrays are trimmed to their common minimum length on mismatch. Trials without a later teleport are skipped; invalid/off-scan samples are removed; trials and sessions below minimum counts are skipped. Missing environment/reward/zone information is replaced with zero or median trial position rather than rejected. There are few assertions beyond neural/fluorescence shape equality.

ii.
```python
common_len = min(lengths)
...
if e_candidates.size == 0:
    continue
...
if rz_center is None or not np.isfinite(rz_center):
    rz_center = float(np.nanmedian(pos))
...
reward_outcomes.append(0)
```

iii. The notes document trimming small length mismatches as the main fix and describe default/fallback mappings as pragmatic. They do not validate timestamp error bounds or preserve uncertainty for missing zone information.

## 13-a. What are the most time-consuming steps of the code?

i. Reading full session-level neural, fluorescence, timestamps, and behavior arrays and writing the roughly 18 GB pickle dominate. The conversion prints per-session and total timings; within-session transforms are vectorized and relatively cheap.

ii.
```python
with h5py.File(fpath, 'r') as f:
    neural = np.asarray(...)
    fluor = np.asarray(...)
...
with outpath.open('wb') as f:
    pickle.dump(data, f)
```

iii. The notes explicitly say full arrays are loaded into memory and that full conversion is tractable at seconds per session; the recorded output size makes serialization another major cost.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. The file/session loop is necessarily I/O-oriented. The three separate loops over trial bounds (collect metadata, infer centers, construct outputs) could be consolidated; reward interval queries could be vectorized/search-sorted. The center-merging loop is tiny. Per-neuron processing was avoided.

ii.
```python
for s, e in bounds:                 # metadata and outcomes
...
for i, (s, e) in enumerate(bounds): # zone centers
...
for i, (s, e) in enumerate(bounds): # final trials
```

iii. The notes highlight vectorized slicing and avoidance of per-neuron loops, while acknowledging per-trial processing and full-session loads.

## 13-c. What processing does the code repeat multiple times?

i. Each trial interval is sliced in three passes. Validity masks, reward-zone arrays, and positions are recomputed in the metadata/center/output passes. Full conversion itself does not repeat a separate survey run inside the script.

ii.
```python
valid = beh['trial number'][s:e] >= 0
...
valid = beh['trial number'][s:e] >= 0
...
valid = (beh['trial number'][s:e] >= 0) & (beh['scanning'][s:e] > 0)
```

iii. The agent's notes do not call out these repeated trial passes; they focus instead on avoiding unnecessary I/O and neuron loops.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and trims the full fluorescence array solely for a shape assertion; computes `env_map` but never uses it; defines an unused reward-zone identity helper; and parses `--show-processing` without generating plots. It also stores time-varying broadcasts for all per-trial variables, increasing the very large pickle.

ii.
```python
fluor = np.asarray(f['processing/ophys/Fluorescence/plane0/data'], dtype=np.float32)
assert neural.shape == fluor.shape
...
env_map = compute_env_mapping(env_per_trial)
...
def reward_zone_label_from_position(rz_val):
    return rz_val
```

iii. The notes mention fluorescence as a fallback, but it is not used as one. They also say `--show-processing` was not implemented and recognize the script's large in-memory/full-array strategy.
