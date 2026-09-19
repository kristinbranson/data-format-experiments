# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories: `beh/` for behavior, `spk/` for neural data, and `retinotopy/` for brain area assignments. `Imaging_Exp_info.npy` serves as the master session index, grouped by experiment type. Each unique session (identified by mname, datexp, blk) is processed once. Behavior files are loaded per experiment type, neural data per session, and retinotopy per session.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# Neural data:
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
# Retinotopy:
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
# Behavior:
beh = np.load(fn, allow_pickle=True).item()
```

iii. The AI documents in CONVERSION_NOTES.md that data is loaded from three subdirectories, each physical session is included once, and the same loading pattern as the reference code's `load_spk()`, `load_retino()`, and `load_exp_beh()` functions is used.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `mname` from the experiment info entries. Unique mouse names are sorted and an index mapping is created.

ii.
```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI notes 19 unique mice from 89 sessions, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is a unique combination of (mname, datexp, blk). The AI deduplicates across experiment types, keeping only the first occurrence of each physical session.

ii.
```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
```

iii. The AI documents that the same physical session can appear in multiple experiment types (142 session-experiment pairs for 89 unique sessions) and decides to include each physical session once.

## 1-d. How are the data split into trials?

i. The AI defines trials using `StartFr` and `EndFr` from the behavior data. Each trial spans from corridor entry (StartFr) to corridor exit (EndFr), which includes both the textured corridor and the gray space. This differs from the reference which uses `ft_trInd` and `ft_CorrSpc` to select only corridor-texture frames.

ii.
```python
start = StartFr[t]
end = EndFr[t]
# Skip if frames out of bounds
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1; continue
end = min(end, len(ft_Pos), len(ft_RunSpeed))
n_tp = end - start
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI documents using StartFr to EndFr as the trial window and notes that the last position bin (3m+) includes the gray space. The CONVERSION_NOTES state "Gray space included: Reference excludes gray space for texture analyses. Our position bin 3 (3m+) includes gray space, which is appropriate for the decoder."

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on: (1) StartFr/EndFr bounds validity (start >= 0, end <= n_total_frames, end > start), (2) clipping end to available behavioral data length, and (3) requiring at least 2 timepoints (n_tp >= 2). Sessions with fewer than 2 surviving trials are skipped. There is no percentile-based length filter for abnormally long trials.

ii.
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1; continue
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1; continue
n_tp = end - start
if n_tp < 2:
    skipped += 1; continue
# ...
if len(neural_trials) < 2:
    print(f"    WARNING: Only {len(neural_trials)} trials, skipping session")
    continue
```

iii. The AI does not document filtering for excessively long trials. The CONVERSION_NOTES do not mention removing outlier trials based on trial length.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`{mouse}_{date}_{blk}_neural_data.npy`), which contains a list of per-plane arrays concatenated into a single (n_neurons, n_frames) matrix. Brain region assignment comes from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = load_retino(session['db_entry'])
```

iii. The AI correctly identifies these as Suite2p deconvolved calcium traces.

## 2-b. How is the `neural` data processed?

i. The raw deconvolved traces are sliced by trial (StartFr to EndFr) and cast to float16. No additional processing (normalization, smoothing, etc.) is applied.

ii.
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI notes in CONVERSION_NOTES that data is already deconvolved by Suite2p and no further processing is needed.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI includes ALL neurons, including those outside visual cortex areas (iarea == -1, 7). The brain_regions list includes an 'other' category for these neurons. This differs from the reference which drops neurons outside V1/mHV/lHV/aHV.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
for area_val, region_name in AREA_MAP.items():
    mask = iarea == area_val
    idx[mask] = region_to_idx[region_name]
```

iii. The AI explicitly decided to include all neurons: "For our decoder: include ALL neurons (no filtering by area), as the decoder should learn to use relevant neurons" (CONVERSION_NOTES Step 3). In Step 10, the AI notes this as a "Deliberate Difference from Reference."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry (StartFr). Each trial starts at StartFr and ends at EndFr. Variable-length trials are stored as-is with no padding.

ii.
```python
start = StartFr[t]
end = EndFr[t]
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI uses StartFr as the alignment event (corridor entry/trial start), consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The frame period (~315 ms at ~3.17 Hz) is used as the time bin. The frame period is computed empirically from the frame timestamps of the first session.

ii.
```python
ft = sample_beh['ft']
dt = np.diff(ft) * 24 * 3600  # datenum to seconds
frame_period = float(np.nanmedian(dt))
# ...
'time_bin_size': frame_period * 1000,  # in ms
```

iii. The AI notes the imaging rate is ~3.17 Hz and uses the native frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (per-trial sound cue frame number) and frame indices computed as `np.arange(start, end)`.

ii.
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The AI uses the frame index and frame period directly, rather than interpolating onto the frame time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The difference `(SoundFr - frame_index) * frame_period` is computed in seconds. Positive values mean the cue hasn't happened yet, negative values mean it has passed. The AI multiplies the frame difference by a single empirically-estimated frame period, rather than using the actual frame timestamps and interpolation.

ii.
```python
time_to_sound = (sound_fr - frame_indices) * frame_period  # positive before, negative after
```

iii. The AI notes this represents "positive = time until cue, negative = time since cue."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same frame range (start:end) as the neural data, ensuring alignment.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. Same frame indices used for all data streams within a trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the chronological ordering of session dates (`datexp`) within each mouse.

ii.
```python
def compute_training_days(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((s['datexp'], i))
    days = np.zeros(len(sessions), dtype=np.float32)
    for mname, sess_list in mouse_sessions.items():
        sess_list.sort(key=lambda x: x[0])
        for day_idx, (datexp, global_idx) in enumerate(sess_list):
            days[global_idx] = float(day_idx)
    return days
```

iii. The AI describes this as "ordinal session index (by date) within each mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are sorted by date within each mouse and assigned sequential indices starting from 0. This is broadcast as a constant across all timepoints in a trial.

ii.
```python
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. Same approach as reference: counting recorded sessions chronologically per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (trial start frame) and frame indices computed as `np.arange(start, end)`.

ii.
```python
time_since_start = (frame_indices - start) * frame_period
```

iii. Uses frame index difference multiplied by frame period.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The difference `(frame_index - StartFr) * frame_period` is computed in seconds. Starts at 0 and increases. Uses the same empirically-estimated frame period as the sound cue computation.

ii.
```python
time_since_start = (frame_indices - start) * frame_period
```

iii. Simple subtraction from start frame, converted to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame range (start:end) as neural data.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. Same frame indices used for all data streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating whether the corridor is rewarded.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. Direct use of the `isRew` field.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` value is cast to float (0.0 or 1.0) and broadcast across all timepoints.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The AI uses `WallName` to get the stimulus identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each of the 15 unique wall names (circle1, circle2, ..., wood5) to a separate category index (0-14), treating each variant as a distinct stimulus. This differs significantly from the reference which groups the 15 names into 4 base categories (circle, leaf, rock, wood).

ii.
```python
all_stimuli = get_all_stimuli(sessions, beh_cache)  # Returns 15 unique names
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
# output_values: all_stimuli = ['circle1', 'circle2', 'circle3', 'leaf1', ...]
```

iii. The AI lists all 15 stimuli as separate categories. The CONVERSION_NOTES show "Stimuli (15): ['circle1', 'circle2', 'circle3', 'leaf1', ...]" and this is not flagged as a potential issue.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, which contains the fractional frame indices of lick events in the session.

ii.
```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. The AI correctly identifies LickFr as the source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to the nearest integer (using `np.round`), then a binary vector is created for the trial window [StartFr, EndFr). A frame is 1 if at least one lick falls in it. The reference uses `astype(int)` (truncation) rather than rounding.

ii.
```python
def make_lick_vector(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_vec = np.zeros(n_frames, dtype=np.int64)
    lick_frs = beh['LickFr']
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        trial_lick_frs = lick_frs_int[mask] - start_fr
        trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
        lick_vec[trial_lick_frs] = 1
    return lick_vec
```

iii. The AI processes licking per-trial window, creating the binary vector from LickFr values within the trial's frame range.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is created for the same frame range [StartFr, EndFr) as the neural data, ensuring alignment.

ii.
```python
lick = make_lick_vector(beh, start, end)
```

iii. Same start/end frame indices used for all data streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. Direct use of ft_Pos.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized using `np.digitize` with edges at [10, 20, 30] dm, giving 4 bins. Because the AI's trial window includes gray space (position 40-60 dm), the last bin (index 3) captures both 30-40 dm of texture and 40-60 dm of gray space.

ii.
```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)
```

iii. The AI labels the bins as `['0-1m', '1-2m', '2-3m', '3m+']` (note the "3m+" indicating it includes gray space).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four bins: [0,10) dm -> bin 0, [10,20) dm -> bin 1, [20,30) dm -> bin 2, [30,+inf) dm -> bin 3. The reference clips to [0,3] with `ft_Pos // 10`, giving bins [0,10), [10,20), [20,30), [30,40) dm, because only corridor texture frames are included.

ii.
```python
bins = np.digitize(pos, [10, 20, 30])
```

iii. The AI's binning includes gray space in the last bin due to the trial window definition.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced with the same [start:end] frame range as neural data.

ii.
```python
pos = ft_Pos[start:end]
```

iii. Same frame range as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. Direct use of ft_RunSpeed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartiles across ALL sessions (percentile-based using `np.percentile` at [25, 50, 75]) and applies them with `np.digitize`. This differs from the reference in two ways: (1) quartiles are global rather than per-session, and (2) percentile thresholds are used instead of rank-based binning, which doesn't handle ties well (many frames have speed=0).

ii.
```python
def compute_speed_quartiles(sessions, beh_cache):
    all_speeds = []
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

def digitize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The AI acknowledges the uneven distribution: "Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9% — uneven due to 21% of frames at speed=0 exactly." The conversion notes flag this but do not correct it.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is digitized using the 25th, 50th, and 75th percentiles of the global speed distribution as bin edges. Due to ties at speed=0 (the 25th percentile), the bins are highly uneven (Q1=9.2%, Q2=40.9% vs the target 25% each).

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
bins = np.digitize(speed, quartiles)
```

iii. The AI notes the unevenness but accepts it.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is sliced with the same [start:end] frame range as neural data.

ii.
```python
speed = ft_RunSpeed[start:end]
```

iii. Same frame range as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips `end` to the available behavioral data length (`min(end, len(ft_Pos), len(ft_RunSpeed))`), skips trials where start >= end or n_tp < 2, and clips lick frames to valid range. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1; continue
# Licking:
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
# Session level:
if len(neural_trials) < 2:
    continue
```

iii. The AI handles edge cases at trial boundaries but does not explicitly handle the behavior-extends-past-imaging case the way the reference does (by cutting to the number of imaged frames).

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (89 files totaling hundreds of GB). The full conversion takes ~33 minutes, dominated by I/O.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
```

iii. The AI reports timing per session (typically 10-40s each) with total time ~33 minutes.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop that creates lick vectors could be vectorized by creating a single session-wide lick array (as the reference does). Also, the stimulus index lookup loops through all trials individually.

ii.
```python
for t in range(ntrials):
    # ... per-trial lick vector creation
    lick = make_lick_vector(beh, start, end)
    # ... per-trial stimulus lookup
    stim_name = str(WallName[t])
    stim_idx = stim_to_idx[stim_name]
```

iii. The per-trial lick vector creation makes a function call per trial, while the reference creates one session-wide binary array.

## 12-c. What processing does the code repeat multiple times?

i. The global speed quartile computation reads all behavior data from all sessions, then the main processing loop reads them again. The behavior data is loaded and cached, so this is reading from cache, but the speed array concatenation is redundant work.

ii.
```python
# First pass: compute speed quartiles
def compute_speed_quartiles(sessions, beh_cache):
    all_speeds = []
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
# Second pass: process sessions
for i, session in enumerate(sessions):
    beh = beh_cache[session['exp_type']][session['beh_key']]
```

iii. The behavior data is cached so the I/O isn't repeated, but the speed array is iterated over twice.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes neurons from areas outside visual cortex (iarea=-1, 7 mapped to 'other'), which would typically be excluded. The decoder may ignore or be harmed by these non-visual-cortex neurons. Also, gray space frames are included which add position data outside the corridor.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
AREA_MAP = {-1: 'other', 7: 'other'}
```

iii. The AI explicitly chose to include all neurons and gray space as design decisions.
