# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as a master index, iterates over experiment types and their sessions, deduplicates by `mname_datexp_blk` key, and for each unique session loads the behavior file (`Beh_<exp_type>.npy`), neural data (`<session_id>_neural_data.npy`), and retinotopy (`<mname>_<datexp>_trans.npz`). Behavior files are cached per experiment type.

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# Deduplication
for exp_type, sessions in info.items():
    for db in sessions:
        key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
        if key not in seen_keys:
            seen_keys.add(key)
            all_sessions.append((exp_type, db, key))
# Loading
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (db['mname'], db['datexp'])), allow_pickle=True)
```

iii. The AI's CONVERSION_NOTES.md documents using `Imaging_Exp_info.npy` as the master index and deduplicating sessions. The approach follows the reference code's `load_spk` and `load_retino` patterns.

## 1-b. How are the data split into subjects?

i. Subjects are identified by `db['mname']`. As sessions are processed, unique mouse names are collected into a dict preserving insertion order. `subject_idx` maps each session to its subject's index.

ii.
```python
mname = db['mname']
if mname not in subjects_set:
    subjects_set[mname] = len(subjects_set)
subject_idx_list.append(subjects_set[mname])
subjects = list(subjects_set.keys())
```

iii. The AI noted 19 subjects matching the paper. Subjects are ordered by first appearance rather than sorted alphabetically.

## 1-c. How are the data split into sessions?

i. A session is identified by `mname_datexp_blk`. Sessions appearing under multiple experiment types are deduplicated via a `seen_keys` set. Each unique key becomes one session. The first experiment type encountered for a session determines which behavior file is used.

ii.
```python
key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if key not in seen_keys:
    seen_keys.add(key)
    all_sessions.append((exp_type, db, key))
```

iii. 89 unique sessions found, matching the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by iterating from 0 to `beh['ntrials']`. Each trial spans from `StartFr` (floored to int) to `EndFr` (floored to int), which includes both the textured corridor AND the gray space. This differs from the reference which uses `ft_trInd` and `ft_CorrSpc` to restrict to corridor space only.

ii.
```python
start_fr_int = np.floor(start_fr).astype(int)
end_fr_int = np.floor(end_fr).astype(int)
start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
end_fr_int = np.clip(end_fr_int, 0, nfr)
# ...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The AI's CONVERSION_NOTES state "Trial boundaries: StartFr to EndFr (including gray space)" and "All frames included: Both running and non-running frames included for temporal continuity."

## 1-e. How are trials filtered based on quality controls?

i. Trials are only skipped if they have fewer than 2 frames (`n_trial_frames < 2`). No filtering of long outlier trials is performed. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
if n_trial_frames < 2:
    skipped_trials += 1
    continue
# ...
if len(neural_trials) < 2:
    print(f"  WARNING: Session has fewer than 2 valid trials, skipping")
    continue
```

iii. The AI's CONVERSION_NOTES state "Very long trials (up to 5621 frames): Included, not filtered" under edge cases. No justification was given for not filtering outlier-length trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (deconvolved calcium traces), concatenated across imaging planes. Brain regions come from `iarea` in retinotopy files.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
dtrans = np.load(...)
iarea = retino['iarea']
```

iii. The AI noted this matches the reference code's `load_spk` function.

## 2-b. How is the `neural` data processed?

i. Neural data is sliced per trial (`spk[:, s_fr:e_fr]`) and stored as float16. Additionally, neurons are randomly subsampled to a maximum of 5000 per session using a fixed random seed.

ii.
```python
MAX_NEURONS = 5000
if n_total > MAX_NEURONS:
    rng = np.random.RandomState(42)
    neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
    spk = spk[neuron_idx]
    iarea_arr = iarea_arr[neuron_idx]
# ...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The AI justified subsampling: "Subsampled to 5000 neurons per session (from 20K-90K) to keep file size manageable. Decoder uses PCA to 100 components."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI maps `iarea` codes to brain region names (V1, mHV, lHV, aHV) but also retains neurons with area codes not matching any known region, labeling them "unassigned". All neurons (including unassigned) are kept. This differs from the reference which drops neurons outside the four visual areas.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
# ...
def neu_area_ID(iarea):
    area_map = {8: 'V1', 0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV', 5: 'lHV', 6: 'lHV', 3: 'aHV', 4: 'aHV'}
    regions = []
    for a in iarea:
        a_int = int(a)
        regions.append(area_map.get(a_int, 'unassigned'))
    return regions
```

iii. The AI noted "No explicit filtering. All Suite2p-detected cells included." and the brain regions include "unassigned". The verification output shows 53,664 unassigned neurons out of 445,000.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials start at `StartFr` (corridor entry), which the instructions specify as the alignment event. Neural data is sliced from `StartFr` to `EndFr`.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The AI set `temporal_alignment_event: 'Trial start (corridor entry)'` and `off_start: 0.0`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate is 3.17 Hz, giving a bin size of ~315.5 ms. Each frame is one time bin.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. Consistent with the reference. The imaging frame is the finest resolution available.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and the frame indices of the trial.

ii.
```python
sound_fr = np.array(beh['SoundFr'])
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The AI uses `SoundFr` directly, converting frame differences to seconds by dividing by `FS`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Computed as `(frame_index - SoundFr[trial]) / FS`. This gives a value that is **negative before** the sound cue and **positive after**. The sign convention is opposite to the instruction name "Time **to** sound cue" (which implies positive before, negative after, as in the reference). Additionally, the AI divides frame differences by the frame rate (3.17 Hz), whereas the reference interpolates `SoundFr` onto a real-time axis derived from `ft` (MATLAB datenum timestamps) and computes actual time differences in seconds.

ii.
```python
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The AI's comment says "Time to sound cue (seconds) - negative before, positive after" which contradicts the variable name "time_to_sound_cue" (should be positive before, 0 at cue, negative after).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices (`s_fr` to `e_fr`) used for the neural data of that trial.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. Same frame range as neural data ensures alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `db['sess#']` or `db['days']` fields in the experiment info metadata, falling back to 0 if neither is present.

ii.
```python
def get_day_of_training(db):
    if 'sess#' in db:
        return float(db['sess#'])
    elif 'days' in db:
        return float(db['days'])
    else:
        return 0.0
```

iii. The AI uses the metadata field directly rather than counting sessions per mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The raw value from `sess#` or `days` is used directly as a float, broadcast across all time bins of a trial. The reference instead counts how many recording sessions each mouse has had up to that point (0-indexed), giving values 0 to at most 7.

ii.
```python
day_of_training = get_day_of_training(db)
day_input = np.full(n_trial_frames, day_of_training, dtype=np.float64)
```

iii. The verification output shows `day_of_training` ranges from 0 to 15 across the dataset. This suggests `sess#` may count differently from the reference's session counting approach (which goes up to 7).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the frame at which the trial starts) and the frame indices of the trial.

ii.
```python
start_fr = np.array(beh['StartFr'])
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. Uses `StartFr` directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - StartFr[trial]) / FS`, converting frame differences to seconds by dividing by the frame rate. The reference instead interpolates `StartFr` onto a real-time axis from `ft` timestamps and computes actual time differences.

ii.
```python
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. Using frame differences / FS is an approximation. The reference uses the actual `ft` timestamps which could differ slightly due to irregular frame timing.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from the same frame indices as the neural data of that trial.

ii.
```python
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. Same frame range ensures alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, which indicates whether a trial is in the rewarded corridor.

ii.
```python
is_rew = np.array(beh['isRew']).astype(float)
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
```

iii. Directly from `isRew`, same as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float, then broadcast across all frames of the trial. No further processing.

ii.
```python
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
```

iii. Same as reference approach.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']`, which gives the texture name for each trial.

ii.
```python
wall_name = np.array(beh['WallName'])
stim_name = str(wall_name[tr])
stim_idx = stim_to_idx.get(stim_name, 0)
```

iii. The AI also collects all unique stimuli from `UniqWalls` across all sessions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses all 15 individual wall texture names (circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5) as separate categories, rather than grouping them into the 4 base categories (circle, leaf, rock, wood) as the reference does and as the instructions specify ("Visual stimulus category. e.g. circle, leaf, etc.").

ii.
```python
all_stim = set()
for exp_type, sessions in info.items():
    # ...
    for w in beh_cache[exp_type][beh_key]['UniqWalls']:
        all_stim.add(str(w))
stim_list = sorted(all_stim)
# stim_list = ['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', ...]
```

iii. The AI treats each wall texture variant as a separate stimulus category. The verification output confirms 15 output values for visual_stimulus with range [0, 14].

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']` (frame numbers of licks) and `beh['LickTrind']` (trial index of each lick).

ii.
```python
lick_fr = np.array(beh['LickFr']).astype(float)
lick_trind = np.array(beh['LickTrind']).astype(int)
```

iii. The AI uses both `LickFr` and `LickTrind` to associate licks with specific trials.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI finds licks whose `LickTrind` matches the trial index, rounds the `LickFr` values to nearest integer, subtracts the trial start frame, and sets those indices to 1 in a binary vector. The reference instead creates a single binary vector for all frames by truncating `LickFr` to int and indexing directly, then slices per trial.

ii.
```python
lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
if len(lick_trind) > 0:
    trial_lick_mask = lick_trind == tr
    if np.any(trial_lick_mask):
        trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int)
        trial_lick_frames = trial_lick_frames - s_fr
        valid = (trial_lick_frames >= 0) & (trial_lick_frames < n_trial_frames)
        if np.any(valid):
            lick_binary[trial_lick_frames[valid]] = 1
```

iii. The approach is functionally similar but uses `round` instead of `int` truncation and maps licks per trial via `LickTrind` rather than globally.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to indices relative to the trial start frame, so they align with the neural data's trial window.

ii.
```python
trial_lick_frames = trial_lick_frames - s_fr
```

iii. Same trial frame range as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position at each imaging frame in decimeters.

ii.
```python
ft_pos = np.array(beh['ft_Pos'])[:nfr]
pos = ft_pos[s_fr:e_fr]
```

iii. Directly from `ft_Pos`, same variable as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 dm (1 m per bin) and clipped to range [0, 3] giving 4 bins. However, since trials extend into gray space (position > 40 dm), positions beyond 3m are clipped into bin 3. This makes bin 3 disproportionately large (48.6% of data per verification output).

ii.
```python
POS_BIN_SIZE_DM = TEXTURE_LENGTH_DM / N_POS_BINS  # 10 decimeters = 1 meter
bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
```

iii. The AI noted "Position bins: 4 bins of 1m covering 4m corridor. Gray space (>4m) assigned to bin 3."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 dm, then clipped to [0, 3]. Bin 0: 0-1m, Bin 1: 1-2m, Bin 2: 2-3m, Bin 3: 3m+ (including gray space up to 6m).

ii.
```python
bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
```

iii. Same binning logic as reference, but applied over a wider range because trials include gray space.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Sliced from the same frame range (`s_fr:e_fr`) as the neural data.

ii.
```python
pos = ft_pos[s_fr:e_fr]
```

iii. Same frame range as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame.

ii.
```python
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
speed = ft_run_speed[s_fr:e_fr]
```

iii. Directly from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using **global** quartile boundaries computed via `np.percentile` across ALL frames in ALL sessions (not per-session). The boundaries are computed once at the start. The reference computes rank-based quartiles per session, over only the kept frames.

ii.
```python
def compute_global_speed_quartiles(info):
    all_speeds = []
    # ... collect all speeds across all sessions
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles

def discretize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The AI computed global quartile boundaries: [0.0, 9.67, 31.46]. This means bin 0 includes all speeds below 0 and up to 0 (a large chunk since many frames are at 0 speed). The verification output shows bin distribution is highly uneven: Q1=9.2%, Q2=40.9%, Q3=25.0%, Q4=24.9%.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with the three global percentile boundaries. This gives 4 bins: bin 0 (below 25th percentile), bin 1 (25th-50th), bin 2 (50th-75th), bin 3 (above 75th).

ii.
```python
bins = np.digitize(speed, quartiles)
```

iii. Since the 25th percentile is 0.0 and many frames have speed=0, `np.digitize` assigns speed=0 to bin 1 (not bin 0) because `digitize` returns the bin where `speed >= boundary`. This causes the highly skewed distribution.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Sliced from the same frame range as neural data.

ii.
```python
speed = ft_run_speed[s_fr:e_fr]
speed_binned = discretize_speed(speed, speed_quartiles)
```

iii. Same frame range as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles neuron count mismatches between spike and retinotopy files by truncating to the minimum. Behavior data is truncated to the number of neural frames. Empty lick arrays are handled. Sessions with <2 valid trials are skipped. Exceptions during session processing are caught and the session is skipped with a warning.

ii.
```python
if n_neu_spk != n_neu_ret:
    min_n = min(n_neu_spk, n_neu_ret)
    spk = spk[:min_n]
# ...
ft_pos = np.array(beh['ft_Pos'])[:nfr]
# ...
lick_fr = np.array(beh['LickFr']).astype(float) if len(beh['LickFr']) > 0 else np.array([])
```

iii. Error handling is defensive with try/except around session processing.

## 12-a. What are the most time-consuming steps of the code?

i. Loading neural data files is the most time-consuming step, as shown by per-session timing in the conversion output (e.g., "Neural data loaded: (89577, 20021) in 7.9s"). Total conversion took 589s for 89 sessions.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
```

iii. I/O dominated, same as reference.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_session` iterates through all trials sequentially, computing inputs and outputs for each. The `neu_area_ID` function uses a Python for-loop over all neurons. The global speed quartile computation loads all behavior files a second time.

ii.
```python
for tr in range(ntrials):
    # ... per-trial processing
```
```python
def neu_area_ID(iarea):
    regions = []
    for a in iarea:
        regions.append(area_map.get(a_int, 'unassigned'))
```

iii. The per-trial loop could be partially vectorized. The `neu_area_ID` loop over neurons is unnecessary when vectorized operations could be used.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded multiple times: once for computing global speed quartiles (`compute_global_speed_quartiles`), once for collecting unique stimuli, and once during actual session processing. The retinotopy area mapping is done per-session even though it could be cached.

ii.
```python
# First pass: global speed quartiles
def compute_global_speed_quartiles(info):
    for exp_type, sessions in info.items():
        beh_cache[exp_type] = load_beh(exp_type)
# Second pass: unique stimuli
for exp_type, sessions in info.items():
    beh_cache[exp_type] = load_beh(exp_type)
# Third pass: actual processing
beh_cache[exp_type] = load_beh(exp_type)
```

iii. Loading behavior files three times is wasteful, though behavior files are much smaller than spike files.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes "unassigned" neurons that would not be used in analyses focused on visual areas. The subsampling to 5000 neurons discards the majority of recorded neurons. Including gray space frames adds data that the reference processing explicitly excludes. The global speed quartile precomputation loads all behavior data an extra time.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
MAX_NEURONS = 5000
```

iii. The 53,664 unassigned neurons (12% of total) are included but would likely be excluded in downstream analyses focused on visual cortex.
