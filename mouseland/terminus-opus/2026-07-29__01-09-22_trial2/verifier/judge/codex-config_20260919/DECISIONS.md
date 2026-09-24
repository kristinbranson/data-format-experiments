# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `beh/Imaging_Exp_info.npy` as the master index, deduplicates recordings by mouse/date/block, then loads each session's plane-wise spike file, retinotopy file, and behavior dictionary. Unlike the reference, behavior files are reopened per session; they are also loaded in separate passes for stimulus names and global speed thresholds.

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
beh = beh_all[beh_key]
```

iii. The notes say all 89 unique recordings should be included and that spike loading should match the authors' `load_spk` by concatenating imaging planes. They report 89 sessions and 19 subjects, matching the paper.

## 1-b. How are the data split into subjects?

i. `mname` defines the subject. Subjects are accumulated in first-session order, and each retained session receives the index of its mouse in that list.

ii.
```python
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
```

iii. The notes identify 19 unique mice and treat the mouse name supplied by the experiment index as the subject identity.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` recording. Duplicate index entries are collapsed by `session_id`, keeping the first experiment-type entry.

ii.
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
if session_id not in session_map:
    session_map[session_id] = (exp_type, entry, beh_key)
```

iii. The notes state that each unique neural recording is one session and that sessions with `stimtype` are handled using the first occurrence.

## 1-d. How are the data split into trials?

i. Trials use the raw `ft_trInd` label, but retain only frames where `ft_move > 0`. Thus a trial is a potentially noncontiguous collection of moving frames, not the full textured-corridor interval used by the reference.

ii.
```python
vr_move = ft_move > 0
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

iii. The notes justify this as matching the paper/reference statement that analyses considered only running or VR-moving timepoints.

## 1-e. How are trials filtered based on quality controls?

i. A trial is dropped only when fewer than two moving frames remain. A whole session is dropped if fewer than two trials remain. There is no long-trial/outlier filter.

ii.
```python
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. The notes explicitly say all trials are included with frames filtered for VR movement; the two-frame rule satisfies the decoder's minimum usable size.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the `spks` arrays in each session's neural file, concatenated across imaging planes. `iarea` from the corresponding retinotopy file supplies neuron regions and filtering.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
)
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. The notes say the raw `spks` are already Suite2p deconvolved fluorescence traces, the representation used by the paper.

## 2-b. How is the `neural` data processed?

i. After neuron and frame selection, the deconvolved traces are copied per trial and cast to float32. No dF/F calculation, smoothing, interpolation, normalization, or padding is performed.

ii.
```python
spk = spk[neuron_mask]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes justify direct use because the supplied arrays are already deconvolved; they describe frame selection and garbage collection as the principal processing/memory choices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea == -1` or `iarea == 7` are excluded. Remaining codes are mapped to V1, mHV, lHV, or aHV.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
iarea_filtered = iarea[neuron_mask]
```

iii. The notes say this exactly follows the authors' visual-cortex curation and `neu_area_ID` mapping.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent labels alignment as corridor entry, but operationally selects every `ft_move > 0` frame carrying the trial label. It neither uses `StartFr` nor `ft_CorrSpc`, and resets elapsed time at the first retained moving frame.

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes state “Trials aligned to corridor entry” and treat `ft_move > 0` as reference-consistent, but do not explain the mismatch between that label and the implemented first-moving-frame origin.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Each retained imaging frame is one bin at 3.17 Hz, reported as about 315.46 ms. There is no temporal rebinning.

ii.
```python
FRAME_RATE = 3.17
TIME_BIN_MS = 1000.0 / FRAME_RATE
```

iii. The notes report 3.17 Hz from the reference notebook and confirm 315.5 ms bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the imaging-frame timestamps `ft`.

ii.
```python
ft = beh['ft'][:n_frames_neural]
sound_fr = beh['SoundFr']
s_fr = sound_fr[trial_idx]
```

iii. The mapping table in the notes specifies `SoundFr, ft` as the two sources and says actual MATLAB-datenum frame times are used.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Fractional `SoundFr` is linearly interpolated between `ft` samples, with boundary clamping and zeros for NaN. The code then computes frame time minus sound time in seconds, so its sign is actually “time since sound,” opposite to time-to-cue in the reference.

ii.
```python
sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes describe this as “Time from frame to sound cue,” but do not acknowledge or justify the implemented sign.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `valid_frame_indices` used for the neural columns, so array samples align one-to-one despite the sign error.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
```

iii. The notes say computations use actual frame timestamps; no separate resampling is applied.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's mouse name and `datexp` calendar date in the master index.

ii.
```python
mouse_sessions[s['mname']].append((i, s))
datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))
```

iii. The notes map `datexp` to day of training and describe it as days since the mouse's first session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are sorted by date per mouse, and the calendar-day difference from the first recording is broadcast over every frame of the trial.

ii.
```python
first_date = dates[0][1]
day_of_training[idx] = (date - first_date).days
...
input_trial[1] = np.float32(day_val)
```

iii. The agent intentionally interpreted “day” as elapsed calendar days. The reference instead counts recorded training sessions ordinally because recordings are not necessarily on consecutive dates.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived only from `ft` at retained trial frames; the raw `StartFr` event is not used.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes say this is elapsed time from corridor entry, but provide no justification for substituting the first moving frame for `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The timestamp of the first retained moving frame is subtracted from every retained frame timestamp and converted from days to seconds, forcing the first value to zero.

ii.
```python
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The mapping plan calls it elapsed time from trial start, but does not discuss fractional start-frame interpolation used by the reference.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It has one value for every selected neural frame and uses the same indices, although its zero event is the first moving frame rather than corridor entry.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
ft_trial = ft[valid_frame_indices]
input_trial[2] = time_since_start
```

iii. The notes claim corridor-entry alignment; the implementation guarantees sample-wise alignment but not the requested event alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the per-trial `isRew` flag.

ii.
```python
is_rew = beh['isRew']
```

iii. The notes map `isRew` to 1 for rewarded and 0 otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The truth value is converted to 1.0 or 0.0 and broadcast across every retained frame of the trial.

ii.
```python
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. No additional processing is documented or needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Per-trial labels come from `WallName`; `UniqWalls` across all sessions is separately used to construct the vocabulary.

ii.
```python
for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(wn))
...
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. The notes map `WallName` directly to a category index.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent sorts all 15 raw wall-pattern names, assigns each a separate integer, and broadcasts that integer across the trial. It does not collapse crop/swap variants to the four base textures.

ii.
```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
output_trial[0] = stim_idx
```

iii. The notes celebrate performance over 15 categories and therefore show this was deliberate, though the requested examples and reference use four broad visual categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from the session's `LickFr` frame numbers, defaulting to an empty array if the key is absent.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```

iii. The notes identify `LickFr` as the direct source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are rounded to nearest integers, invalid indices are discarded, and valid frames are marked 1 in an otherwise-zero binary vector. Multiple licks in one frame remain 1.

ii.
```python
idx = np.round(lick_fr).astype(int)
idx = idx[(idx >= 0) & (idx < n_frames)]
lick_binary[idx] = 1
```

iii. The notes describe binary lick detection per frame but do not justify rounding rather than the reference's truncation.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The per-frame lick vector is indexed with the exact same frame indices as the trial's neural data.

ii.
```python
lick_trial = lick_binary[valid_frame_indices]
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The documentation treats all time-varying streams as being on the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the per-imaging-frame `ft_Pos` values.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames_neural]
```

iii. The notes identify `ft_Pos` and interpret it in decimeters over a 6 m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 15 decimeters and floored, producing four 1.5 m bins across the full 6 m corridor, then clipped to 0–3.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. The notes explicitly choose four equal bins over the full 6 m (4 m texture plus 2 m gray), overlooking the instruction's four 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 15, 30, 45, and 60 dm, labeled `0-1.5m`, `1.5-3m`, `3-4.5m`, and `4.5-6m`; clipping folds any out-of-range value into an endpoint class.

ii.
```python
output_values = [all_stim_names, ['no_lick', 'lick'],
                 ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ...]
```

iii. The stated rationale is four equal bins over the complete physical corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sliced at the same selected frame indices as the neural matrix.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. Both streams already share the imaging-frame grid, so no interpolation is documented.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`; `ft_move` determines which speed samples are eligible when global thresholds are estimated.

ii.
```python
vr_move = beh['ft_move'] > 0
speeds = beh['ft_RunSpeed'][vr_move]
```

iii. The notes identify `ft_RunSpeed` as the output source and use running-only frames in keeping with their paper interpretation.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Up to 5,000 moving-frame speeds are deterministically sampled per session, pooled over all sessions, and global 25th/50th/75th percentiles are computed. Each selected trial speed is digitized using those thresholds.

ii.
```python
if len(speeds) > 5000:
    speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
quartiles = np.percentile(np.concatenate(all_speeds), [25, 50, 75])
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3)
```

iii. The notes justify global quartile bins and sampling as a way to compute them without loading neural data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric percentile boundaries define Q1–Q4 via `np.digitize`. Because thresholds are global, sessions are capped at equal sample weight, ties are not split, and frames later excluded from trials influence the boundaries, the converted bins need not each contain 25% of retained data.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles)
```

iii. The agent interpreted “each corresponding to 25% of the data” as global percentile thresholds rather than the reference's per-session rank split over retained frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The raw speed vector is indexed at exactly the neural trial's selected frame indices before categorization.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3)
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The notes assume the behavior streams and neural traces share the frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior vectors are sliced to the neural frame count. Missing `LickFr` becomes no licks; negative/out-of-range lick frames are discarded. NaN sound frames become an all-zero time-to-sound vector and out-of-range sound frames are clamped. Trials/sessions with too little surviving data are skipped. There is no general per-session exception handler.

ii.
```python
ft = beh['ft'][:n_frames_neural]
beh.get('LickFr', np.array([]))
idx = idx[(idx >= 0) & (idx < n_frames)]
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
```

iii. The notes say edge cases were checked and no critical issues were found, but do not substantively justify zero-filling missing sound times.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating enormous plane-wise spike arrays dominates; copying selected neural data to float32, writing every session to a temporary pickle, and rereading all temporary pickles also impose substantial I/O and memory costs. The full run reportedly took 73.1 minutes and produced 211 GB.

ii.
```python
spk = np.concatenate([...], 0)
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
pickle.dump({...}, f, protocol=4)
...
sd = pickle.load(f)
```

iii. The notes identify temp files as a memory optimization and report the observed full conversion runtime; they do not explicitly profile the stages.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly compares all session frames against one trial index, and then separately slices/copies all streams. Stimulus collection and session/day grouping also use Python loops, but are negligible beside neural I/O.

ii.
```python
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

iii. The agent supplied no explicit vectorization analysis; the notes focus on caching, temporary files, and garbage collection instead.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded during global stimulus discovery, global speed-threshold estimation, and again once per session during conversion. Trial masks scan the same session-length arrays once per trial. Neural data is serialized to session cache files and then immediately deserialized for final assembly.

ii.
```python
beh_cache[exp_type] = np.load(...).item()  # in get_all_stim_names
beh_cache[exp_type] = np.load(...).item()  # in compute_global_speed_quartiles
beh_all = np.load(...).item()              # in every process_session
```

iii. The notes call speed computation behavior-only and temp files memory optimizations, but do not discuss the repeated behavior I/O or serialization pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and stores extra metadata (`entry`, experiment type, stimulus map, global boundaries and descriptive constants), builds `session_meta` but never uses it, repeatedly calls explicit garbage collection, and optionally creates extensive diagnostic plots. More importantly, temporary session serialization is discarded after final assembly.

ii.
```python
session_meta = []
gc.collect()
...
for tf in temp_files:
    os.remove(tf)
```

iii. The notes present temporary files, garbage collection, and plots as validation/memory aids. They were useful operationally, but the intermediate cache and diagnostics are not consumed by decoder training.
