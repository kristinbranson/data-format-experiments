# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy` as the master index, eagerly merges every available `Beh_<experiment>.npy` into `all_beh`, and loads each session's plane-wise spike file and retinotopy file during processing. It deduplicates repeated experiment-index entries by the mouse/date/block session key; the full run produced 89 sessions from 19 mice.

ii. ```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_beh = load_all_behavior(exp_info)
sessions = build_session_list(exp_info)
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in data['spks']], 0)
```

iii. The notes say the index contains 142 entries but only 89 unique recordings, behavior is repeated across experiment types, spikes are stored per session and plane, and retinotopy supplies neuron areas. The eager behavior merge was chosen to make one base-key lookup per session.

## 1-b. How are the data split into subjects?

i. Subject identity is `mname`. Subjects are the sorted unique mouse names, and every session receives the index of its mouse in that list.

ii. ```python
subject_list = sorted(set(s['mname'] for s in sessions))
subject_idx_list.append(subject_list.index(sess['mname']))
```

iii. The notes identify 19 mice and treat the explicit `mname` field as authoritative.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Repeated index records are combined in a dictionary, their experiment types are retained as metadata, and sessions are sorted by mouse and date.

ii. ```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_dict:
    session_dict[key] = {'key': key, 'mname': ndb['mname'],
                         'datexp': ndb['datexp'], 'blk': ndb['blk'], 'exp_types': []}
session_dict[key]['exp_types'].append(exp_type)
```

iii. This was justified by the paper's 89 recordings and the observation that 33 sessions occur under multiple experiment types.

## 1-d. How are the data split into trials?

i. For each raw trial index `n`, the agent selects imaged frames whose `ft_trInd == n` and whose `ft_CorrSpc` flag is true. Thus a trial spans corridor-entry through the textured 4 m section and remains variable length. Trials with fewer than two selected frames become `None`.

ii. ```python
for n in range(ntrials):
    frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
    if len(frames) >= 2:
        trials.append(frames)
    else:
        trials.append(None)
```

iii. The notes say this preserves continuous frame-rate time series and defines a trial as corridor entry to gray-space entry, where stimuli, cues, and rewards occur.

## 1-e. How are trials filtered based on quality controls?

i. Only trials with fewer than two surviving corridor frames are removed. No upper-length/outlier filter is applied, so the 5,607-frame stopped trial and other extreme traversals are retained.

ii. ```python
if len(frames) >= 2:
    trials.append(frames)
else:
    trials.append(None)
```

iii. The notes state that reference code had no explicit trial filter and later explicitly accept the 5,607-frame case as an animal stopping in the corridor, so it was retained to follow that perceived convention.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the `spks` list in each `<mouse>_<date>_<block>_neural_data.npy`; neuron regions/filtering come from `iarea` in `<mouse>_<date>_trans.npz`.

ii. ```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in data['spks']], 0)
trans = np.load(path, allow_pickle=True)
return trans['iarea']
```

iii. The agent identified `spks` as already deconvolved Suite2p calcium traces and `iarea` as the retinotopic assignment.

## 2-b. How is the `neural` data processed?

i. Plane arrays are concatenated by neuron, area-filtered, sliced to each trial's corridor frames, and copied as float32 matrices of shape neurons by time. No dF/F calculation, deconvolution, normalization, padding, or position interpolation is performed.

ii. ```python
spk = np.concatenate([nspk for nspk in data['spks']], 0)
spk_filtered = spk[neuron_mask]
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The notes correctly say the files already contain deconvolved traces. They reject the paper's position interpolation because the requested decoder needs time-based streams.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea` -1 or 7 are excluded; all remaining codes are mapped to V1, mHV, lHV, or aHV. There is no additional cell-quality filter.

ii. ```python
EXCLUDED_AREAS = {-1, 7}
mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
```

iii. The notes connect this to the reference code's exclusion of unmapped/boundary areas and state Suite2p had already classified cells.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each matrix starts at the first frame satisfying the trial index and corridor-space flag, interpreted as corridor entry/trial start, and ends at gray-space entry. No padding or common-duration crop is used.

ii. ```python
frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The agent says variable-length trials preserve the continuous traversal and metadata records corridor entry as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied: one raw imaging frame is one bin. Per-session median differences in `ft` are computed, and the median over sessions is stored as about 314.7 ms (nominally 3.17 Hz).

ii. ```python
dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
median_dt = np.median(dt_secs)
'time_bin_size': median_dt * 1000
```

iii. The notes say behavior already shares the imaging-frame grid, making resampling unnecessary.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the selected global frame indices, trial-level `SoundFr`, and a session-wide median frame interval calculated from `ft`.

ii. ```python
SoundFr = beh['SoundFr']
dt_sec = np.median(np.diff(ft)) * 24 * 3600
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. The mapping notes describe it as `SoundFr - frame_idx`, but the implementation and its comment instead use `frame_idx - SoundFr`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame-number difference is multiplied by the median seconds per frame. The implemented sign is negative before the cue and positive after it—time relative to cue, not time remaining until cue.

ii. ```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. The notes explicitly justify negative-before/positive-after values, although that contradicts the field name and the notes' own source-to-target formula.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at the exact `frames` used to slice neural columns, giving one value per neural bin.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. The agent's general justification is that all streams use imaging-frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and the chronological ordering of that mouse's deduplicated sessions by `datexp`.

ii. ```python
sess_list.sort(key=lambda s: s['datexp'])
for i, s in enumerate(sess_list):
    s['day_index'] = i
```

iii. The notes treat recorded-session order as the natural measure of training progression because actual consecutive calendar days are not guaranteed.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Each mouse's first recording is assigned 0 and subsequent sessions increment by one; the session value is broadcast across every bin in every trial.

ii. ```python
s['day_index'] = i
input_arr[1, :] = float(sess['day_index'])
```

iii. The notes report a 0–7 range matching one to eight recorded sessions per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from each selected corridor frame, the first selected frame of that trial, and the median session frame interval from `ft`; it does not directly use raw `StartFr`.

ii. ```python
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The agent notes that fractional `StartFr` is awkward and elected to identify trial start via `ft_trInd`/`ft_CorrSpc`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first selected frame is subtracted from every selected frame and the result is converted to seconds, producing exactly zero at the first stored bin.

ii. ```python
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The notes intended a nonnegative time axis starting at corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The same `frames` array defines both neural columns and elapsed-time samples.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. The agent relies on common imaging-frame indices for alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from trial-level `isRew`.

ii. ```python
isRew = beh['isRew']
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. The notes identify this as the raw flag for whether the trial's corridor is rewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The value is converted to 0.0/1.0 and broadcast over all bins in the trial.

ii. ```python
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. No further processing was considered necessary; the notes confirm reward is absent in many naive/unsupervised sessions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It comes from the trial-level `WallName` string.

ii. ```python
WallName = beh['WallName']
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. The notes prefer `WallName` because it remains informative for swap sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Fifteen exact wall variants (such as `leaf1_swap1`) are sorted and integer-encoded, then the code is broadcast across the trial. They are not collapsed to the requested four semantic categories circle, leaf, rock, and wood.

ii. ```python
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', ...])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. The agent explicitly chose 15 categories to “preserve maximum information,” despite describing the task elsewhere as four-category texture discrimination.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`; `LickTrind` is read but not actually used in constructing the indicator.

ii. ```python
lick_fr = beh['LickFr'].astype(int)
lick_trind = beh['LickTrind'].astype(int)
```

iii. The notes identify `LickFr` as the imaging-frame location of each lick and planned to sanity-check it against behavior.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frames are truncated to integers, out-of-range values are discarded, and corresponding frame indicators are set to one. Multiple licks in one frame remain binary.

ii. ```python
lick_indicator = np.zeros(nfr, dtype=np.int64)
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
lick_indicator[lick_fr[valid_mask]] = 1
```

iii. The agent intended the requested binary per-frame output and documented lick-free pretraining sessions as expected.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The global frame indicator is indexed by the identical corridor `frames` used for neural data.

ii. ```python
lick_arrays.append(lick_indicator[frames].copy())
output_arr[1, :] = lick_arrays[n]
```

iii. The frame-number representation was considered already aligned to imaging.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos`, truncated to the number of neural frames.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr]
```

iii. The notes establish that position is in decimeters and the textured corridor occupies 0–40 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Values at selected frames are floor-divided by 10 dm and cast to integer categories; clipping protects the allowed 0–3 range.

ii. ```python
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. This directly implements four equal 1 m bins over the 4 m texture region.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40)` dm map to labels 0–3; values outside are clipped to the nearest endpoint category.

ii. ```python
np.clip(ft_Pos[frames] // 10, 0, 3)
```

iii. The thresholds follow the instruction's four equal-length, one-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed with the same selected imaging-frame indices as neural activity.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3)
```

iii. The notes say the behavior stream is already on the neural frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from frame-level `ft_RunSpeed` across frames marked `ft_CorrSpc`, truncated by the neural frame count.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
all_speeds.append(ft_RunSpeed[corr_mask])
```

iii. The agent chose the directly available speed stream.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. In a first pass the code pools all corridor-frame speeds across all selected sessions, computes global 25th/50th/75th value percentiles, and in the second pass applies those thresholds with `np.digitize`.

ii. ```python
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The agent wanted globally comparable quartile categories. It later observed that ties at zero make Q1 only 9.8% of the full data but called this “not a bug.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` assigns bins 0–3 around the three global percentile values. Because repeated zero values cannot be split by value thresholds, the resulting categories do not each contain 25% of full-data frames.

ii. ```python
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
np.digitize(ft_RunSpeed[frames], speed_quantiles)
```

iii. The intended justification was four global quartiles, but the agent knowingly accepted materially unequal occupancy after full validation.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is digitized at exactly the global frame indices used for neural columns.

ii. ```python
neural = spk_filtered[:, frames].astype(np.float32)
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles)
```

iii. Common imaging-frame indexing provides the alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior streams are truncated to neural frame count; invalid lick indices are discarded; trials with fewer than two corridor frames are skipped; absent behavior files are skipped while loading; duplicate/swap-suffixed behavior keys are normalized to a base key. There is no per-session exception recovery, explicit missing-key validation, or length-outlier handling.

ii. ```python
ft = beh['ft'][:nfr]
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
if len(frames) >= 2: trials.append(frames)
if not os.path.exists(beh_path): continue
```

iii. The notes specifically identify the one-frame neural/behavior mismatch and say truncation matches reference behavior. They regard the dataset as clean and swap behavior as identical.

## 12-a. What are the most time-consuming steps of the code?

i. Loading very large spike files, concatenating/copying plane arrays, extracting float32 neural data for tens of thousands of neurons across all trials, and writing the 241.67 GB pickle dominate. The second pass took about 1,319 seconds; the complete run took about 1,894 seconds.

ii. ```python
data = np.load(path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in data['spks']], 0)
neural = spk_filtered[:, frames].astype(np.float32)
pickle.dump(data, f, protocol=4)
```

iii. The script prints pass/session timings, while the notes acknowledge large neural arrays and an unexpectedly long full conversion.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-frame extraction scans the full frame arrays once per trial; neuron masks/region indices use Python comprehensions; subject indices use repeated linear `list.index`; and per-trial lick slicing/copying is prebuilt then revisited. The unavoidable per-session/per-trial assembly still requires separate variable-length arrays.

ii. ```python
for n in range(ntrials):
    frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
subject_idx_list.append(subject_list.index(sess['mname']))
```

iii. The notes call the lick construction “vectorized,” but do not document these remaining vectorization opportunities.

## 12-c. What processing does the code repeat multiple times?

i. Every spike file is loaded once in pass 1 merely to obtain shapes/counts and again in pass 2 for conversion. Area files and neuron inclusion counts are likewise loaded/computed in both passes. Trial membership is scanned/count-checked in pass 1 and reconstructed in pass 2. Optional plotting loads spikes and repeats filtering/trial/behavior processing a third time.

ii. ```python
# pass 1
data = np.load(path, allow_pickle=True).item()
iarea = load_area_ids(mname, datexp)
# pass 2
spk = load_spk(mname, datexp, blk)
iarea = load_area_ids(mname, datexp)
```

iii. The two-pass design was justified as collecting global speed thresholds before building the dataset, but the redundant neural and area I/O was not justified; comments incorrectly describe loading the object file “just to get shape” as fast.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Pass 1 computes `nneu_total`, `n_included`, `ntrials`, `ntrials_valid`, and `nfr` for `session_info`, but downstream assembly uses only `dt_sec`. `LickTrind` is read and unused. `exp_types`, several result counters, and optional plotting values are not part of decoder data. Most importantly, float32 neural copies double the storage needed relative to the reference's float16 without adding meaningful source precision.

ii. ```python
lick_trind = beh['LickTrind'].astype(int)
session_info.append({'n_included': n_included, 'nneu_total': nneu_total,
                     'ntrials': ntrials, 'ntrials_valid': n_valid,
                     'nfr': nfr, 'dt_sec': dt_sec})
dt_secs = [si['dt_sec'] for si in session_info]
```

iii. The agent used these values for console diagnostics and sanity checks, but did not distinguish diagnostic work from conversion necessities or address the large discarded/redundant computations.
