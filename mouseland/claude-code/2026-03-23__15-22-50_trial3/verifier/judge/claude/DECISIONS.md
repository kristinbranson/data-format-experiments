# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories under `data/`: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. `Imaging_Exp_info.npy` is read first to enumerate all sessions grouped by experiment type. The behavior files are loaded per experiment type via `load_beh()`, spike data via `load_spk()`, and retinotopy via `load_retino()`. Behavior data is cached by experiment type to avoid redundant loads.

ii. Loading functions:
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()

# Neural data
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)

# Behavior
fn = os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type)
return np.load(fn, allow_pickle=True).item()

# Retinotopy
fn = '%s_%s_trans.npz' % (db['mname'], db['datexp'])
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The AI documented this approach in CONVERSION_NOTES.md Steps 1-2, noting the three data sources and their organization. The approach mirrors the reference code's `load_spk()` and `load_retino()` utilities.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mname` field from `Imaging_Exp_info.npy`. The sorted unique mouse names form the subjects list, and each session is mapped to its subject index.

ii.
```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
subject_idx_list.append(subject_to_idx[session['mname']])
```

iii. The AI noted 19 unique mice across 89 sessions, consistent with the paper.

## 1-c. How are the data split into sessions?

i. A session is identified by the tuple `(mname, datexp, blk)`. The AI deduplicates sessions that appear under multiple experiment types by keeping only the first occurrence, yielding 89 unique sessions.

ii.
```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
```

iii. The AI documented this in CONVERSION_NOTES.md Step 4, noting that the same physical recording can appear under multiple experiment types and should be included only once.

## 1-d. How are the data split into trials?

i. The AI defines trials using `StartFr` and `EndFr` from the behavior data, extracting neural and behavioral data from `StartFr` to `EndFr` for each trial. This includes both the textured corridor and the gray space. Trials are skipped if frames are out of bounds, if `end <= start`, or if fewer than 2 frames remain.

ii.
```python
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)

for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
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
    neural = spk[:, start:end].astype(np.float16)
```

iii. The AI decided to use `StartFr:EndFr` as the trial window, which includes gray space after the textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. The AI skips trials where `start < 0`, `end > n_total_frames`, `end <= start`, or the trial has fewer than 2 frames. Sessions with fewer than 2 surviving trials are also skipped.

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

iii. No explicit quality filtering beyond boundary checks is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains a list of arrays (one per imaging plane) concatenated into a single `(n_neurons, n_frames)` array. The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
```

iii. Documented in CONVERSION_NOTES.md Step 1, matching the reference `load_spk()` function.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are extracted per trial from `StartFr` to `EndFr` and cast to float16. No further processing (no dF/F, no deconvolution) is applied. Trials have variable length — no fixed window or padding is used.

ii.
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI noted the data is already Suite2p deconvolved traces, so no additional processing is needed. Variable-length trials are kept as-is.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI includes ALL neurons, including those outside the four visual cortex areas (iarea = -1 and 7, mapped to 'other'). No neurons are filtered out.

ii.
```python
AREA_MAP = {
    8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']

def get_brain_region_idx(iarea):
    idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
    for area_val, region_name in AREA_MAP.items():
        mask = iarea == area_val
        idx[mask] = region_to_idx[region_name]
    return idx
```

iii. The AI justified this by stating "include ALL neurons (no filtering by area), as the decoder should learn to use relevant neurons" in CONVERSION_NOTES.md.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data for each trial is extracted from `StartFr` (corridor entry) to `EndFr` (corridor exit including gray space). The alignment event is corridor entry (`StartFr`), consistent with the instructions. However, the trial window extends through the gray space rather than stopping at the end of the textured corridor.

ii.
```python
start = StartFr[t]
end = EndFr[t]
neural = spk[:, start:end].astype(np.float16)
```

iii. The AI set `temporal_alignment_event` to "Trial start (corridor entry)" and `off_start` to 0.0.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No temporal rebinning is applied. The native imaging frame rate (~3.17 Hz, ~315 ms per frame) is preserved. The time bin size is computed from the median inter-frame interval of one session.

ii.
```python
dt = np.diff(ft) * 24 * 3600  # datenum to seconds
frame_period = float(np.nanmedian(dt))
# ...
'time_bin_size': frame_period * 1000,  # in ms
```

iii. The AI computed the frame period from the data and reported ~314.7 ms, consistent with the ~3.17 Hz rate from the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame indices within the trial.

ii.
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The AI uses SoundFr as a per-trial scalar frame index.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(SoundFr - frame_index) * frame_period` for each frame in the trial. This gives positive values before the sound cue and negative values after, which matches the "time to sound cue" semantics. The conversion from frames to seconds uses a constant `frame_period` derived from the median inter-frame interval.

ii.
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period  # positive before, negative after
```

iii. The AI assumes a constant frame period across all frames, rather than using the actual frame timestamps (`ft`) for interpolation as the reference does.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`start:end`) used for the neural data extraction, so it is inherently aligned.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. All data streams use the same frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session dates (`datexp`) for each mouse, sorted chronologically.

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

iii. The AI counts the ordinal session index within each mouse, starting from 0.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions for each mouse are sorted by date, and the ordinal index (0, 1, 2, ...) is assigned as the training day. The value is broadcast as a constant across all timepoints in a trial.

ii.
```python
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. The AI noted this gives the chronological session index within each mouse. Note: if `--sample` mode is used, `compute_training_days` is called on the filtered session list, which could give different day counts than the full dataset. However, the main code computes this correctly.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the frame number at corridor entry) and the frame indices within the trial.

ii.
```python
time_since_start = (frame_indices - start) * frame_period
```

iii. Uses the same frame-based approach as time_to_sound_cue.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - StartFr) * frame_period`, giving seconds since corridor entry. The first value is 0.0 and increases by `frame_period` per frame.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. Uses a constant frame period rather than actual frame timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices used for neural data extraction.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
```

iii. Same alignment approach as all other variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks trials run in the rewarded corridor.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. Direct from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float and broadcast across all timepoints in the trial. 1.0 for rewarded, 0.0 for unrewarded.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the texture on the corridor walls for each trial.

ii.
```python
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. Uses WallName directly.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI treats each unique WallName as a separate stimulus category, yielding 15 unique categories (circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, rock1, rock2, wood1, wood2, wood5, wood1_swap1, wood1_swap2). The sorted unique names become the output values.

ii.
```python
all_stimuli = get_all_stimuli(sessions, beh_cache)  # sorted unique wall names
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}

# In get_all_stimuli:
for wn in beh['UniqWalls']:
    all_stim.add(str(wn))
return sorted(all_stim)
```

iii. The AI chose to keep individual stimulus names rather than grouping them into 4 base texture categories (circle, leaf, rock, wood). The instructions say "Visual stimulus category. e.g. circle1, leaf2, etc." which could support either interpretation, but the reference groups into 4 categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the fractional frame indices of each lick in the session.

ii.
```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. Uses LickFr directly.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame indices are rounded to the nearest integer, then a binary vector is created for each trial's frame range (`start` to `end`). A frame is 1 if at least one lick falls in it.

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

iii. The AI uses `np.round()` to convert fractional lick frames to integer indices, while the reference uses `.astype(int)` (truncation).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is computed for the same frame range (`start_fr` to `end_fr`) as the neural data, so it is inherently aligned.

ii.
```python
lick = make_lick_vector(beh, start, end)
```

iii. Same frame range as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. Direct from behavior data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 4 bins using `np.digitize` with edges at [10, 20, 30] dm, giving bins [0,10), [10,20), [20,30), and [30,inf). The labels are '0-1m', '1-2m', '2-3m', '3m+'.

ii.
```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)

POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
```

iii. The AI includes gray space positions (40-60 dm) in the last bin ('3m+').

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `np.digitize(pos, [10, 20, 30])` returns 0 for [0,10), 1 for [10,20), 2 for [20,30), and 3 for [30,inf). This effectively creates 4 bins, with the 4th bin encompassing both the last meter of texture (30-40 dm) and the 2m gray space (40-60 dm).

ii.
```python
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]  # last bin includes gray space
bins = np.digitize(pos, [10, 20, 30])
```

iii. The AI explicitly documented that the last bin includes gray space, which differs from the reference's 4 equal 1m bins within the texture only.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted from `ft_Pos[start:end]` using the same frame range as neural data.

ii.
```python
pos = ft_Pos[start:end]
```

iii. Same frame indices as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. Direct from behavior data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartiles across ALL frames from ALL sessions using `np.percentile([25, 50, 75])`, then uses `np.digitize` to bin each frame's speed into one of 4 quartile bins.

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

iii. The AI computes speed quartiles globally using `np.percentile`, and the quartile thresholds are computed across all frames (including gray space and non-corridor frames), not just the frames kept for trials.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is categorized using `np.digitize` with the three quartile values as bin edges. This gives 4 bins labeled 'Q1_slow', 'Q2', 'Q3', 'Q4_fast'.

ii.
```python
speed_labels = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']
bins = np.digitize(speed, quartiles)
```

iii. Because `np.percentile` is used rather than rank-based splitting, when many frames have the same speed value (e.g., speed=0), the bins will be unequal in size.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted from `ft_RunSpeed[start:end]` using the same frame range as neural data.

ii.
```python
speed = ft_RunSpeed[start:end]
```

iii. Same frame indices as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips `end` to the minimum of `EndFr`, `len(ft_Pos)`, and `len(ft_RunSpeed)` to handle behavior arrays shorter than the neural data. Trials with invalid frame ranges are skipped. Lick frames outside the trial window are filtered out and remaining ones are clipped to valid range. Failed sessions are caught with a try/except and skipped.

ii.
```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue

# Lick handling:
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)

# Session error handling:
try:
    res = process_session(...)
except Exception as error:
    print('failed %s: %s: %s' % (...))
    continue
```

iii. The AI handles edge cases with bounds checking and clipping.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files, which are large numpy arrays (total ~405 GB across all sessions). Each session requires loading and concatenating imaging plane data.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
```

iii. The AI reported ~33 minutes for full conversion, dominated by I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session` iterates over each trial individually, creating numpy arrays and stacking inputs/outputs. The lick vector creation could potentially be vectorized across trials. The stimulus name lookup iterates through all stimuli per trial.

ii.
```python
for t in range(ntrials):
    # ... per-trial processing
    neural = spk[:, start:end].astype(np.float16)
    lick = make_lick_vector(beh, start, end)
    pos_bin = digitize_position(pos)
    speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The loop is straightforward and not a major bottleneck compared to I/O.

## 12-c. What processing does the code repeat multiple times?

i. The `compute_speed_quartiles` function iterates over all sessions to concatenate speeds, which reads from the behavior cache. This is separate from the main processing loop. The frame period is computed once from a single session.

ii.
```python
def compute_speed_quartiles(sessions, beh_cache):
    all_speeds = []
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
```

iii. The speed quartile computation is a separate pass over the data but uses cached behavior, so it's not I/O bound.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes neurons from the 'other' brain region (iarea = -1, 7) that are outside the visual cortex. These neurons are not used in the reference analyses and add unnecessary data to the output. The AI also includes gray space frames in each trial, which are beyond the 4m textured corridor relevant to the task.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
AREA_MAP = { -1: 'other', 7: 'other', ... }
```

iii. The AI justified including all neurons by saying "the decoder should learn to use relevant neurons," but this adds ~12% more neurons that are outside the regions of interest.
