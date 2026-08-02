# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three sources: (1) `Imaging_Exp_info.npy` to enumerate all sessions across experiment types, (2) `Beh_{exp_type}.npy` files for behavioral data, (3) `{mouse}_{date}_{blk}_neural_data.npy` files for neural data, and (4) `{mouse}_{date}_trans.npz` files for retinotopy. Sessions are deduplicated by unique `(mname, datexp, blk)` tuples across experiment types, resulting in 89 unique sessions.

ii.
```python
def build_session_list():
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
    ).item()
    seen = set()
    sessions = []
    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = (db['mname'], db['datexp'], db['blk'])
            if key in seen:
                continue
            seen.add(key)
            beh_key = get_beh_key(db)
            sessions.append({...})
    sessions.sort(key=lambda s: (s['mname'], s['datexp']))
    return sessions
```

iii. The AI documented that the same physical session appears in multiple experiment types (142 session-experiment pairs for 89 sessions). It decided to use each physical session once, picking the first experiment type encountered. This was documented in CONVERSION_NOTES.md Step 4.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by unique mouse names (`mname` field). The AI collects all unique `mname` values from the session list, sorts them alphabetically, and creates a subject-to-index mapping. 19 unique subjects were found.

ii.
```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```

iii. The AI verified 19 mice match the paper's report of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `(mname, datexp, blk)` tuples. The same physical recording session can appear in multiple experiment types; the AI deduplicates to get 89 unique sessions. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
# ...
if len(neural_trials) < 2:
    print(f"    WARNING: Only {len(neural_trials)} trials, skipping session")
    continue
```

iii. The AI documented this in Step 4 as "Key Decision: Session-Experiment Mapping" — same physical recording in multiple experiment types should be included once.

## 1-d. How are the data split into trials?

i. Trials are defined by `StartFr` (corridor entry frame) and `EndFr` (corridor exit frame) from the behavior data. Each trial corresponds to one corridor traversal. The number of trials per session comes from `beh['ntrials']`.

ii.
```python
ntrials = beh['ntrials']
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
    # extract neural[:, start:end], etc.
```

iii. The AI used the standard trial boundaries from the behavior data, matching the reference code's approach.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: trials are skipped if (a) start < 0, (b) end > n_total_frames, (c) end <= start, (d) end exceeds behavioral array length, or (e) trial has fewer than 2 timepoints. No filtering based on ft_move (running status) is applied.

ii.
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1
    continue
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
n_tp = end - start
if n_tp < 2:
    skipped += 1
    continue
```

iii. The AI documented that the reference code uses all trials (CONVERSION_NOTES Step 3), and added only boundary checks for data integrity. The full conversion reported no skipped trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in `{mouse}_{date}_{blk}_neural_data.npy` files. These contain Suite2p deconvolved calcium traces (spike deconvolution with 0.75s decay timescale) from multiple imaging planes.

ii.
```python
def load_spk(db):
    fn = '%s_%s_%s_neural_data.npy' % (db['mname'], db['datexp'], db['blk'])
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The AI correctly identified these as Suite2p deconvolved traces and noted that deltaF/F computation is NOT needed.

## 2-b. How is the `neural` data processed?

i. Neural data is loaded by concatenating imaging planes, then per-trial segments are extracted using StartFr:EndFr frame indices. The data is stored as float16 to reduce memory. No normalization, smoothing, or additional processing is applied. No ft_move filtering is applied (stationary frames are included).

ii.
```python
spk = load_spk(session['db_entry'])
# ...
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI decided not to apply ft_move filtering, documented as "Deliberate Differences from Reference" in Step 10: "Reference code excludes stationary frames (ft_move==0) for d-prime analyses. Our decoder includes all frames to maintain temporal continuity."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality filtering is applied. All neurons are included regardless of brain area (including iarea=-1 and iarea=7, mapped to "other"). The reference code excludes these in its analyses. No corridor-activity filtering or d-prime-based filtering is applied.

ii.
```python
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
```

iii. Documented in CONVERSION_NOTES Step 3: "For our decoder: include ALL neurons (no filtering by area), as the decoder should learn to use relevant neurons." Step 10 reaffirms: "Reference sometimes filters by area or d-prime. Decoder includes all neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) using `StartFr`. Each trial's neural data starts at frame `StartFr[t]` and ends at frame `EndFr[t]`. The offset from alignment event to trial start is 0.0 (they are the same event).

ii.
```python
neural = spk[:, start:end].astype(np.float16)
# ...
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)" which corresponds to StartFr.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the raw imaging frame rate (~314.7 ms, ~3.18 Hz). No temporal rebinning is applied. The frame period is computed from the behavior timestamp array `ft` (MATLAB datenum format).

ii.
```python
ft = sample_beh['ft']
dt = np.diff(ft) * 24 * 3600  # datenum to seconds
frame_period = float(np.nanmedian(dt))
# ...
'time_bin_size': frame_period * 1000,  # in ms
```

iii. The AI documented the frame rate as ~3.17 Hz, consistent with the reference code's fs=3.17Hz.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (per-trial sound cue frame index) and the frame indices within each trial.

ii.
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The AI correctly identified SoundFr as the per-trial sound cue frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the time to sound cue is computed as `(SoundFr - frame_index) * frame_period`. This gives positive values before the sound cue (time remaining until cue) and negative values after (time elapsed since cue), in seconds.

ii.
```python
time_to_sound = (sound_fr - frame_indices) * frame_period  # positive before, negative after
```

iii. This creates a continuous, time-varying input as specified in the instructions.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed at each neural frame index, so it is inherently aligned with the neural data (same frame indices, same length).

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. Direct frame-level alignment ensures no temporal mismatch.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` (date string) field in the session database entries.

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

iii. The AI computed ordinal session index within each mouse sorted by date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by date (`datexp`), and assigned ordinal indices (0, 1, 2, ...). This ordinal index is used as the "day of training" value, broadcast to all frames within a trial.

ii.
```python
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. The maximum day value is 7 (from verification output: day_of_training range [0.0, 7.0]), matching that the maximum sessions per subject is 8 (TX119 and TX123 have 8 sessions each).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The instructions specify "Reward availability" (not "Environment type" explicitly). The AI derives this from `isRew` in the behavior data, which indicates whether the current trial's corridor is rewarded (1) or not (0).

ii.
```python
isRew = beh['isRew']
# ...
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. `isRew` directly encodes the reward condition of each corridor/trial.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. The `isRew` value for each trial is converted to a float and broadcast to all frames within the trial. No additional processing is performed.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The instruction specifies "Reward availability: 1 if in rewarded corridor, 0 if not, discrete, per-trial" which matches this implementation.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the frame indices within each trial relative to `StartFr`.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. StartFr marks corridor entry, which is the trial start event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, the time since trial start is `(frame_index - StartFr) * frame_period`, in seconds. This starts at 0 and increases monotonically within each trial.

ii.
```python
time_since_start = (frame_indices - start) * frame_period
```

iii. Creates a continuous, time-varying ramp from 0.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at each neural frame index, inherently aligned with neural data.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
```

iii. Same frame-level alignment as other time-varying inputs.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a per-trial boolean/integer indicating whether the corridor is rewarded.

ii.
```python
isRew = beh['isRew']
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. isRew is loaded directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The isRew value for each trial is cast to float and broadcast to all timepoints within the trial. No additional logic is applied.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. Matches the instruction: "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']` (per-trial wall texture name) and `beh['UniqWalls']` (list of unique wall names per session).

ii.
```python
WallName = beh['WallName']
# ...
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The AI collected all unique stimulus names across all sessions (15 total) and created a sorted global mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's WallName is mapped to a global index (0-14) from a sorted list of all 15 unique stimuli. The index is broadcast to all timepoints within the trial.

ii.
```python
all_stimuli = get_all_stimuli(sessions, beh_cache)  # sorted unique names
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
# per trial:
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The 15 stimuli include: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']`, which contains fractional frame indices of lick events.

ii.
```python
lick_frs = beh['LickFr']
```

iii. LickFr provides the timing of each lick in neural frame coordinates.

## 8-b. What processing is involved in computing `output` *Licking*?

i. LickFr values are rounded to the nearest integer frame, then filtered to the trial's frame range [StartFr, EndFr). A binary vector is created where 1 = lick occurred at that frame, 0 = no lick.

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

iii. The result is a binary time series as specified: "0 = not licking, 1 = licking."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames (LickFr) are in the same frame coordinate system as the neural data. By filtering to [StartFr, EndFr) and subtracting StartFr, the lick vector is aligned to the neural trial data.

ii.
```python
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
trial_lick_frs = lick_frs_int[mask] - start_fr
```

iii. Direct frame-level alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, which contains the position in the virtual corridor (in decimeters) at each neural frame.

ii.
```python
ft_Pos = beh['ft_Pos']
pos = ft_Pos[start:end]
```

iii. ft_Pos is a per-frame position variable ranging from 0 to ~60 dm (0-6m corridor).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Per-trial position is extracted from ft_Pos for frames [StartFr, EndFr), then discretized.

ii.
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. Position is extracted at neural frame resolution.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is digitized using bin edges at 10, 20, 30 dm, creating 4 bins: [0,10dm)=[0-1m), [10,20dm)=[1-2m), [20,30dm)=[2-3m), [30dm+)=[3m+). The last bin includes both the final meter of texture (3-4m) AND the 2m gray space (4-6m).

ii.
```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)

POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
```

iii. The instruction says "4 equal-length, 1-m-long spatial bins" over the corridor. The first 3 bins are 1m each, but the last bin covers 3m (including gray space), resulting in an unequal distribution: {0-1m: 19.4%, 1-2m: 15.9%, 2-3m: 16.1%, 3m+: 48.6%}.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. ft_Pos is indexed at the same frames as the neural data (StartFr:EndFr), so position is inherently aligned.

ii.
```python
pos = ft_Pos[start:end]
```

iii. Same frame indices as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, the running speed at each neural frame.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed']
speed = ft_RunSpeed[start:end]
```

iii. ft_RunSpeed is a per-frame continuous variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Global speed quartiles (25th, 50th, 75th percentiles) are computed across all frames in all sessions, then speed values are digitized into 4 bins.

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
```

iii. The instruction says "4 bins, each corresponding to 25% of the data." Global quartiles achieve this.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is digitized using the global quartile values as bin edges.

ii.
```python
def digitize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The actual distribution is uneven: {Q1_slow: 9.2%, Q2: 40.9%, Q3: 25.0%, Q4_fast: 24.9%} because many frames have speed=0, which falls exactly at the 25th percentile boundary. np.digitize assigns values equal to an edge to the upper bin.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. ft_RunSpeed is indexed at the same frames as neural data.

ii.
```python
speed = ft_RunSpeed[start:end]
```

iii. Same frame-level alignment as other variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (a) trials with invalid frame boundaries are skipped, (b) EndFr is clipped to available behavioral array length, (c) lick frame indices are clipped to valid range, (d) sessions with fewer than 2 valid trials are skipped. No handling for NaN values in behavioral arrays or extremely long trials (max 5621 frames = ~30 minutes).

ii.
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1; continue
end = min(end, len(ft_Pos), len(ft_RunSpeed))
# ...
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
```

iii. The AI documented that no trials were skipped in the full conversion.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural .npy files (up to 90K neurons per session), extracting per-trial segments, and saving the 201GB pickle file. Total conversion time was 33.4 minutes for 89 sessions.

ii.
```python
spk = load_spk(session['db_entry'])  # loads tens of thousands of neurons
# ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=5)  # 201 GB file
```

iii. The AI documented processing time as ~15s/session in Step 7.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` (iterating over `ntrials` trials) could partially be vectorized: computing time_to_sound, time_since_start, and position/speed binning for all frames at once using array operations on the full session data, then splitting into trials.

ii.
```python
for t in range(ntrials):
    # Per-trial: time_to_sound, position binning, speed binning, lick vector
    # Each of these could operate on full-session arrays then be split
```

iii. The AI noted efficiency concerns in CONVERSION_NOTES but did not vectorize the trial loop.

## 12-c. What processing does the code repeat multiple times?

i. The code iterates over all sessions three times: once in `compute_speed_quartiles()`, once in `get_all_stimuli()`, and once in the main processing loop. The first two could be merged into the main processing loop or combined into a single pass.

ii.
```python
speed_quartiles = compute_speed_quartiles(sessions, beh_cache)  # iterates all sessions
all_stimuli = get_all_stimuli(sessions, beh_cache)              # iterates all sessions
for i, session in enumerate(sessions):                          # main processing loop
```

iii. This is a minor inefficiency since the behavior data is cached.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) Including neurons with iarea=-1 and iarea=7 ("other" region, ~585K neurons total) that the reference code excludes from analyses. (b) Including stationary frames (ft_move==0) that the reference paper explicitly excludes. (c) Including extremely long trials (up to 5621 frames / ~30 minutes) caused by extended stationary periods. (d) Storing neural data as float16 which the decoder converts back to float32.

ii.
```python
-1: 'other', 7: 'other',  # included but reference code excludes these
# No ft_move filtering applied
neural = spk[:, start:end].astype(np.float16)  # converted back to float32 by decoder
```

iii. The AI documented these as "Deliberate Differences from Reference" but they add significant data volume and noise.
