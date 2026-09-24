# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent scans every behavior `.npy`, keeps dictionary entries whose keys match a session-ID regex, and stores them in one dictionary. It then retains sorted behavior IDs having a matching spike file. Each spike file's three `spks` matrices is loaded and concatenated. It does not use `Imaging_Exp_info.npy` or retinotopy files.

ii.
```python
for f in sorted(BEH_DIR.glob('*.npy')):
    obj = np.load(f, allow_pickle=True).item()
    for k, v in obj.items():
        if is_session_key(k):
            beh[k] = {'file': f.name, 'data': v}
common_sessions = [sid for sid in sorted(beh.keys())
                   if (SPK_DIR / f'{sid}_neural_data.npy').exists()]
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The notes say true session-like keys must be separated from metadata keys, the three neural blocks share a time axis, and all neurons should be preserved. This yielded 75 sessions, rather than the reference index's 89.

## 1-b. How are the data split into subjects?

i. Subject is the prefix before the first underscore in a session ID. Unique prefixes are sorted and sessions index that list.

ii.
```python
def get_subject(session_id):
    return session_id.split('_')[0]
subjects = sorted({get_subject(sid) for sid in common_sessions})
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The notes report 19 subjects confirmed from neural filenames and reject non-session behavior keys that had produced spurious labels.

## 1-c. How are the data split into sessions?

i. A regex recognizes keys of the form mouse/date/block, optionally suffixed by `_swapN`; a session is included if an identically named spike file exists. Duplicate IDs encountered in later behavior files overwrite earlier entries.

ii.
```python
SESSION_RE = re.compile(r'^[A-Z0-9]+_\d{4}_\d{2}_\d{2}_[0-9]+(?:_swap[0-9]+)?$')
beh[k] = {'file': f.name, 'data': v}
```

iii. The rationale was to exclude cohort/metadata keys and preserve actual session-specific entries. The notes acknowledge only 75 converted sessions and one further skipped session.

## 1-d. How are the data split into trials?

i. Unique finite integer values of `ft_trInd` define trials. Every frame bearing that trial ID is retained; trials with fewer than two frames are dropped. The code does not restrict frames with `ft_CorrSpc`.

ii.
```python
valid = ~np.isnan(ft_tr)
tr_ids = np.unique(ft_tr[valid].astype(int))
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
    if idx.size < 2:
        continue
```

iii. The notes say the converter segments trials via `ft_trInd` and aligns them to trial start/corridor entry. No justification is given for retaining all trial-labelled frames rather than only corridor frames.

## 1-e. How are trials filtered based on quality controls?

i. Trials are retained when they have at least two indexed frames. Sample mode additionally takes only the first 20 trial IDs. Sessions with fewer than two surviving trials are discarded. There is no long/stationary-trial exclusion.

ii.
```python
if max_trials is not None:
    tr_ids = tr_ids[:max_trials]
if idx.size < 2:
    continue
if len(neural_trials) < 2:
    return None
```

iii. The notes identify `max_trials=20` solely as a sample-mode speedup and mention the one session skipped by `convert_session`; they document no trial-quality analysis.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the `spks` list in each `<session_id>_neural_data.npy` file.

ii.
```python
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)
```

iii. The methods and notes identify `spks` as Suite2p-derived deconvolved fluorescence, already sharing a common time axis across three blocks.

## 2-b. How is the `neural` data processed?

i. The three neuron-by-time blocks are concatenated on the neuron axis, truncated to the common behavior/neural length, sliced by trial frame indices, and stored as float32. There is no dF/F calculation, deconvolution, padding, or temporal resampling.

ii.
```python
neural = neural[:, :T]
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes correctly state that paper analyses use the already deconvolved traces and that the blocks should be concatenated.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. Retinotopy is not loaded. Instead, every neuron is assigned to one of three artificial, approximately equal contiguous blocks, with a hard-coded special split for session `2022_07_12_1`.

ii.
```python
'brain_regions': ['unknown_block0', 'unknown_block1', 'unknown_block2'],
np.full(19408 if '2022_07_12_1' in sid else nneu // 3, 0, dtype=int)
```

iii. The notes say Suite2p classification may already curate cells and planned to preserve all neurons. They never resolved the required retinotopy-area mapping and describe the fake labels as a known limitation.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. A trial begins at its first frame carrying the relevant `ft_trInd`; its complete set of such frames is sliced from neural data. The metadata calls this “trial start / corridor entry,” but the code does not use `StartFr` or `ft_CorrSpc` to enforce corridor entry.

ii.
```python
idx = np.where(ft_tr.astype(float) == float(tr))[0]
t0 = ft[idx[0]]
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The agent asserted that segmentation by `ft_trInd` aligns trials to corridor entry. The converted position distribution (92.4% in the last clipped bin) contradicts that interpretation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One original imaging frame is one output bin; no rebinning is applied. Metadata estimates milliseconds from the median adjacent `ft` difference in the first session.

ii.
```python
'time_bin_size': float(np.median(np.diff(np.asarray(
    beh[common_sessions[0]]['data']['ft'])))) * 24 * 3600 * 1000
```

iii. The notes treat frame-level neural and behavioral arrays as already aligned. This is approximately the 3.17 Hz/315 ms reference resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundTime` and `Trial_start_time`, plus frame timestamp `ft`.

ii.
```python
sound_times = np.asarray(b['SoundTime'])
trial_start_times = np.asarray(b['Trial_start_time'])
```

iii. The notes identify cue timing relative to trial start as central, but do not justify choosing timestamps over the reference's `SoundFr` frame index.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The cue delay is `(SoundTime - Trial_start_time)` converted from days to seconds; elapsed time from the first trial-labelled frame is subtracted. Missing indexed cue data produces an all-NaN trial vector.

ii.
```python
times = (ft[idx] - ft[idx[0]]) * 24 * 3600
cue_rel = float((sound_times[tr] - trial_start_times[tr]) * 24 * 3600)
time_to_cue = (cue_rel - times).astype(np.float32)
```

iii. The notes record fixing an initial days-to-seconds bug and define the desired quantity as cue time minus current time.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It has one value for every `idx` frame used to slice neural activity. Its zero point, however, is based on `Trial_start_time` while elapsed frame time is based on the first trial-labelled frame.

ii.
```python
times = (ft[idx] - t0) * 24 * 3600
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The agent says frame-level behavior and neural streams share the same grid. It did not check whether the first indexed frame equals `Trial_start_time`/corridor entry.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is inferred from the behavior filename, not from session dates or chronological within-mouse session order.

ii.
```python
name = beh_file.lower()
if 'before_learning' in name or 'before_grating' in name: return 0.0
if 'after_learning' in name or 'after_grating' in name: return 1.0
m = re.search(r'train(\d+)', name)
```

iii. The planning notes proposed session ordering from filenames/conditions but never reconciled these heterogeneous labels into a real within-subject training-day index.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Before conditions map to 0, after conditions to 1, `trainN`/`testN` to N, and unmatched names to 0. The value is recomputed inside every trial loop and broadcast across time.

ii.
```python
day_val = np.full(idx.size,
    get_day_value(session_id, beh_entry['file']), dtype=np.float32)
```

iii. The notes describe this as a per-trial continuous value, but provide no validation that the filename-derived values mean “day of training” consistently across mice.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived solely from `ft` timestamps at frames selected by `ft_trInd`; `Trial_start_time` is loaded but is not used for this input.

ii.
```python
t0 = ft[idx[0]]
times = (ft[idx] - t0) * 24 * 3600
```

iii. The notes call the first selected frame trial start/corridor entry without demonstrating equivalence to `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first selected frame timestamp is subtracted from each selected timestamp and MATLAB-day differences are converted to seconds.

ii.
```python
time_since = times.astype(np.float32)
```

iii. The agent fixed the seconds conversion and sanity-checked that the first trial begins at zero with roughly frame-spaced increments.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is calculated on exactly the same frame indices and therefore has the same length as each neural trial, though alignment is to the first trial-labelled frame rather than explicitly to corridor entry.

ii.
```python
inp = np.vstack([time_to_cue, day_val, time_since, rew_avail])
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes' shape checks confirmed neural, input, and output trial lengths match.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from per-trial `isRew`.

ii.
```python
is_rew = np.asarray(b['isRew']).astype(int)
```

iii. The mapping plan interprets it as one for a rewarded corridor and zero otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The integer flag is broadcast across all frames of a trial. An out-of-range trial ID falls back to zero.

ii.
```python
rew_avail = np.full(idx.size,
    int(is_rew[tr]) if tr < len(is_rew) else 0, dtype=np.float32)
```

iii. The agent considered this a per-trial binary contextual input; no additional processing was thought necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It normally comes from per-trial `WallName`, with the first frame's `ft_WallID` as a fallback.

ii.
```python
wall_names = np.asarray(b['WallName']).astype(str)
trial_wall = wall_names[tr] if tr < len(wall_names) else ft_wall[idx[0]]
```

iii. The notes prefer `WallName`/`stim_id` and say actual labels should be preserved rather than imposing a fixed subset.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique literal wall names across included sessions are alphabetically mapped to integer categories and broadcast per trial. Crops and swapped variants are not collapsed to the four base textures.

ii.
```python
cats = sorted(str(x) for x in np.unique(all_wall_names))
return {c: i for i, c in enumerate(cats)}, cats
stim_cat = np.full(idx.size, global_cat_map.get(str(trial_wall), 0))
```

iii. The agent explicitly chose to preserve session-specific categories. Full output had 13 categories, producing an incorrect chance baseline relative to the requested broad visual categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session-level `LickFr` frame indices.

ii.
```python
lick_fr = np.asarray(b.get('LickFr', []))
```

iii. Although planning mentioned `LickTime`/`LickTrind`, implementation used the directly frame-aligned `LickFr` representation.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A zero vector over the common frame range is created. Valid integer lick-frame indices are set to one; multiple licks in a frame remain binary.

ii.
```python
lick_fr = lick_fr.astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
lick[lick_fr] = 1
```

iii. The notes intended a binary time series with one for a lick bin and zero otherwise.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary frame vector is sliced by the same trial indices used for neural activity.

ii.
```python
lick_out = lick[idx].astype(np.int64)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The rationale is that both streams are already indexed on imaging frames; matching trial shapes were verified.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos`.

ii.
```python
ft_pos = np.asarray(b['ft_Pos'])
```

iii. The mapping plan identified frame-level corridor position and intended four equal 1 m bins.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code floor-divides `ft_Pos` by 1.0, casts to integer, and clips to 0–3. It fails to convert the raw decimeter units to meters.

ii.
```python
pos_bin = np.clip((ft_pos[idx] // 1.0).astype(int), 0, 3).astype(np.int64)
```

iii. The notes claim the boundaries match 1 m bins, but their own validation found values beginning `[0, 2, 2, 2, 2]` and did not recognize the unit error.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Nominal category labels are `bin1` through `bin4`, but thresholds in raw units are [0,1), [1,2), [2,3), and ≥3 due to clipping, instead of decimeter thresholds 0,10,20,30,40.

ii.
```python
'output_values': [..., ['bin1', 'bin2', 'bin3', 'bin4'], ...]
```

iii. The agent stated these were four equal-length 1 m bins. Full verification showed 92.4% of frames in category 3, exposing the mistake.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is truncated to the common length and indexed with the identical trial-frame array used for neural activity.

ii.
```python
ft_pos = ft_pos[:T]
pos_bin = ... ft_pos[idx] ...
```

iii. The frame alignment itself follows the notes' plan, although the selected windows include frames outside the reference corridor window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level `ft_RunSpeed` values.

ii.
```python
ft_speed = np.asarray(b['ft_RunSpeed'])
```

iii. The mapping plan directly identifies running speed as the source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global numerical quantiles are computed from all finite speed samples in all selected sessions, including frames not ultimately retained. Each trial value is digitized against those thresholds.

ii.
```python
qs = np.quantile(all_speeds, [0.25, 0.5, 0.75])
return np.digitize(x, qs, right=False).astype(np.int64)
```

iii. The agent intended four global quartile bins. It did not account for ties (especially zero speed) or reference per-session retained-frame rank bins.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` produces q1–q4 according to the three global 25th/50th/75th percentile thresholds. Equal values all go to the same side, so bins are not equal sized (full fractions were about 9%, 41%, 25%, 25%).

ii.
```python
'output_values': [..., ['q1', 'q2', 'q3', 'q4']]
```

iii. The notes state each bin should correspond to 25% of valid samples, but validation results were not used to revise the tied-value behavior.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The speed array is truncated to common length and the same trial frame indices are digitized and stored beside neural frames.

ii.
```python
ft_speed = ft_speed[:T]
speed_bin = apply_speed_bins(ft_speed[idx], speed_qs)
```

iii. The agent relies on the common imaging-frame grid; trial dimensions were shape-validated.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Streams are truncated to their minimum common length; absent spike files and sessions with fewer than two trials are skipped; lick indices outside bounds are dropped; missing cue indices yield NaNs; missing reward/stimulus indices fall back to zero or `ft_WallID`. A broad regex filters non-session keys.

ii.
```python
T = min(neural.shape[1], len(ft), len(ft_tr), len(ft_pos), len(ft_speed), len(ft_wall))
if neural is None: return None
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < T)]
time_to_cue = np.full(idx.size, np.nan, dtype=np.float32)
```

iii. The notes document fixes for time units and missing optional cue-window fields, acknowledge one skipped session, and emphasize that the produced pickle passed structural verification.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating the enormous spike matrices, then copying a large neuron-by-frame slice for every trial, dominate. The converter also scans all behavior speeds to obtain global quantiles.

ii.
```python
mats = np.load(f, allow_pickle=True).item()['spks']
return np.concatenate(mats, axis=0).astype(np.float32)
neural_trials.append(neural[:, idx].astype(np.float32))
```

iii. The notes explicitly identify full-session neural slicing as expensive and introduced sample limits because initial sample conversion was too slow.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly scans all `ft_tr` frames with `np.where`; trial indices could be grouped once. Stimulus/reward/day broadcasting and output construction also remain per-trial, although large neural slicing inherently remains trial-wise.

ii.
```python
for tr in tr_ids:
    idx = np.where(ft_tr.astype(float) == float(tr))[0]
```

iii. The agent documented expensive neural slicing but did not identify or optimize the repeated full-vector trial search.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded once to build the index, then behavior arrays are scanned for global walls/speeds, then again during conversion. Within every trial, `ft_tr.astype(float)` and filename-based day inference are recomputed.

ii.
```python
for sid in common_sessions:  # global category/quantile pass
    ...
for sid in common_sessions:  # conversion pass
    converted = convert_session(...)
```

iii. No rationale or explicit discussion of repeated processing appears in the notes.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/truncates `ft_WallID` for a fallback rarely needed, globally collects all raw speeds/wall labels, repeatedly casts trial indices, and fabricates three brain-region arrays that do not convey real anatomy. It also loads `session_id` into `get_day_value` without using it.

ii.
```python
ft_wall = np.asarray(b['ft_WallID']).astype(str)
def get_day_value(session_id, beh_file):
    name = beh_file.lower()
```

iii. The notes do not discuss discarded processing. Their focus was reducing sample trial count and passing downstream validation.
