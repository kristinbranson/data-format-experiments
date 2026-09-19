# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `Imaging_Exp_info.npy` as the master index, iterates over experiment types and entries to build a map of unique sessions. For each session it loads neural data from `spk/` via `load_spk()`, behavior from `beh/Beh_{exp_type}.npy` via `load_behavior_for_session()`, and retinotopy from `retinotopy/` via `load_retino()`. Behavior files are re-loaded per session rather than grouped and loaded once per file.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
# ...
for exp_type in exp_info:
    for ndb in exp_info[exp_type]:
        key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
        if key not in session_map:
            session_map[key] = (exp_type, ndb)
```
```python
beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
beh_data = np.load(beh_file, allow_pickle=True).item()
```

iii. The AI's CONVERSION_NOTES.md documents understanding of the data structure: "Imaging_Exp_info.npy contains 23 experiment types" and "Same physical session can appear in multiple experiment types."

## 1-b. How are the data split into subjects?

i. Subjects are identified by `ndb['mname']`. The unique sorted set of subject names is built from all session results. 19 subjects are identified.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
```

iii. The AI notes "Subjects identified (19)" in CONVERSION_NOTES.md Step 0.

## 1-c. How are the data split into sessions?

i. A session is uniquely identified by `{mname}_{datexp}_{blk}`. Duplicate entries across experiment types are deduplicated by keeping only the first occurrence. 89 unique sessions result.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_map:
    session_map[key] = (exp_type, ndb)
```

iii. CONVERSION_NOTES.md Step 4: "Same physical session can appear in multiple experiment types... Use each physical session once."

## 1-d. How are the data split into trials?

i. The AI defines trials using `StartFr` (corridor entry frame) to `GrayFr` (grey space entry frame) for each trial index. It rounds these fractional frame numbers to nearest integers and takes the contiguous frame range `spk[:, sfr:gfr]`. This differs from the reference which uses `ft_trInd` and `ft_CorrSpc` boolean masks to identify frames belonging to each trial in corridor space.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
# ...
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. CONVERSION_NOTES.md Step 5: "Trial window: StartFr to GrayFr (corridor entry to grey space entry) - captures full textured corridor portion."

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal filtering: trials with invalid frame ranges (`sfr < 0 or gfr > n_frames or sfr >= gfr`) or fewer than 2 frames are skipped. No trial length outlier filtering is applied. The reference solution drops trials exceeding the 99th percentile of trial lengths across the whole dataset.

ii.
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. CONVERSION_NOTES.md Step 5: "All trials included: No trial filtering." CONVERSION_NOTES.md Step 10 notes: "Very long trials (max T=5607 frames ~ 29 min): Valid - mouse stopped running, VR stationary. No filtering needed as reference code doesn't filter by trial length."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/{session_id}_neural_data.npy`, which contains lists of per-plane neuron-by-frame arrays. These are concatenated across planes. The brain region assignment comes from `iarea` in the retinotopy files.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```

iii. CONVERSION_NOTES.md Step 1: "Neural data is loaded via load_spk(): concatenates spks list across imaging planes."

## 2-b. How is the `neural` data processed?

i. The neural traces are not further processed. For each trial, the contiguous frame range `sfr:gfr` is extracted and stored as float16.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. CONVERSION_NOTES.md Step 1: "spks are deconvolved fluorescence traces from Suite2p (not raw dF/F)."

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by brain region. All neurons are kept, including those with iarea values -1 and 7 (classified as 'other'). The reference solution drops neurons outside V1, mHV, lHV, and aHV (i.e., drops 'other'). The AI's data includes 585,641 extra 'other' neurons and reports 5 brain regions instead of 4.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']

def iarea_to_region_idx(iarea):
    region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    return region_idx
```

iii. CONVERSION_NOTES.md Step 5: "All neurons included: No filtering, consistent with reference code" and Step 1: "No neuron filtering/curation in the reference code - all Suite2p-detected neurons used." However, the reference code's `neu_area_ID` function does exclude non-visual-cortex neurons from region-based analyses, and the reference solution explicitly drops them.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Each trial's neural data starts at the `StartFr` (rounded to nearest int) and ends at `GrayFr`. Variable-length trials are stored as-is.

ii.
```python
sfr = start_frs[trial_idx]
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. CONVERSION_NOTES.md metadata: `'temporal_alignment_event': 'Trial start (corridor entry)'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame is the time bin, at approximately 3.17 Hz (~315 ms). The AI computes the actual frame duration per session from timestamps and uses the median across sessions.

ii.
```python
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
# ...
median_dt = np.median([r['dt_sec'] for r in session_results])
'time_bin_size': float(median_dt * 1000),  # in ms
```

iii. CONVERSION_NOTES.md Step 3: "Frame rate: ~3.17 Hz."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame number of the sound cue for each trial) and frame indices. The AI computes `(SoundFr - frame_index) * dt_sec`.

ii.
```python
sound_frs = beh['SoundFr']
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. CONVERSION_NOTES.md Step 5: "SoundFr - frame_idx... Continuous, time-varying, positive before cue."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI multiplies the difference between SoundFr and frame indices by `dt_sec` (median frame duration). The reference interpolates SoundFr onto the actual time axis derived from MATLAB datenum timestamps, which properly handles non-uniform frame spacing.

ii.
```python
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. CONVERSION_NOTES.md Step 5: "(SoundFr - frame_idx) * dt_sec."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The frame indices used for time_to_sound match the same sfr:gfr range as the neural data, so alignment is consistent within the AI's framework.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Implicit from using the same frame range.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the date string in each session's metadata), parsed into a datetime object.

ii.
```python
date = datetime.strptime(datexp, '%Y_%m_%d')
mouse_sessions[mname].append((sess_key, date))
```

iii. CONVERSION_NOTES.md Step 5: "Days since mouse's first recording."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes calendar day difference from each mouse's first recording date. This produces values ranging from 0 to 92. The reference counts session ordinal number (0-indexed), producing values from 0 to 7. These are very different scales.

ii.
```python
def compute_day_of_training(session_map):
    # ...
    for mname, sessions in mouse_sessions.items():
        sessions.sort(key=lambda x: x[1])
        first_date = sessions[0][1]
        for sess_key, date in sessions:
            day_map[sess_key] = (date - first_date).days
    return day_map
```

iii. CONVERSION_NOTES.md Step 5: "day_of_training: Days since mouse's first recording" and "Day of training: calendar day difference from mouse's first recording date."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the frame indices and `StartFr` (corridor entry frame), multiplied by dt_sec.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. CONVERSION_NOTES.md Step 5: "(frame_idx - StartFr) * dt_sec."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI computes `(frame_index - StartFr_rounded) * dt_sec`. Since sfr is the rounded StartFr, this starts at 0 and increments by dt_sec per frame. The reference uses actual interpolated timestamps.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Implicit from the code.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame range (sfr:gfr) as the neural data.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Implicit from using the same frame range.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating whether the corridor is rewarded.

ii.
```python
is_rew = beh['isRew']
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. CONVERSION_NOTES.md Step 5: "isRew... 1 if rewarded corridor, 0 otherwise."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No processing beyond converting to float32 (1.0 or 0.0) and broadcasting to all frames in the trial.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
np.full(n_trial_frames, rew_val, dtype=np.float32)
```

iii. No special processing noted.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
wall_names = beh['WallName']
stim_name = wall_names[trial_idx]
```

iii. CONVERSION_NOTES.md Step 5: "WallName... Map to category index."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the raw WallName strings directly as category labels (e.g., 'circle1', 'circle2', 'leaf1', 'leaf1_swap1', etc.), resulting in 15 stimulus categories. The reference groups these into 4 base categories (circle, leaf, rock, wood) using a TEXTURE mapping dictionary. This is a major difference: 15 categories vs 4.

ii.
```python
stim_names, stim_to_idx = build_stimulus_mapping(session_results)
# stim_names ends up being the sorted unique WallName values: 15 categories
```

iii. CONVERSION_NOTES.md Step 5: "Stimulus categories: Use WallName directly (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, plus rock/brick variants)."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame numbers of lick events, and `LickTrind`.

ii.
```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. CONVERSION_NOTES.md Step 5: "LickFr, LickTrind... Binary per frame."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are rounded to the nearest integer (reference truncates via `.astype(int)`). A binary array is created for the trial's frame range, with 1 where any lick falls. The AI builds the lick binary per-trial within the StartFr:GrayFr range; the reference builds it for the whole session and then indexes by trial frames.

ii.
```python
def build_lick_binary(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_binary = np.zeros(n_frames, dtype=np.float32)
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        frame_offsets = lick_frs_int[mask] - start_fr
        lick_binary[frame_offsets] = 1.0
    return lick_binary
```

iii. CONVERSION_NOTES.md Step 5: "Binary 1 at any frame that has >=1 lick."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick binary is computed for the same StartFr:GrayFr frame range as the neural data.

ii.
```python
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. Same frame range as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in decimeters at each imaging frame.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
```

iii. CONVERSION_NOTES.md Step 5: "ft_Pos... Discretize into 4 bins."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized into 4 bins using `np.digitize` with bin edges from `np.linspace(0, 40, 5)` = [0, 10, 20, 30, 40]. The reference uses `np.clip(ft_Pos // 10, 0, 3)`. These produce equivalent results for values in [0, 40).

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```

iii. CONVERSION_NOTES.md Step 5: "Discretize into 4 bins of 10 dm each (0-10, 10-20, 20-30, 30-40)."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal bins of 10 decimeters (1 meter) each: 0-10, 10-20, 20-30, 30-40. Values at bin edges are handled by np.digitize.

ii.
```python
bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)  # [0, 10, 20, 30, 40]
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. CONVERSION_NOTES.md Step 5: "4 equal 1m bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same StartFr:GrayFr frame range as the neural data.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
```

iii. Same frame range as neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
trial_speed = ft_run_speed[sfr:gfr]
```

iii. CONVERSION_NOTES.md Step 5: "ft_RunSpeed... Discretize into 4 quartile bins."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global quartile thresholds (25th, 50th, 75th percentiles) across all corridor frames in the entire dataset, then uses `np.digitize` with `right=True` to bin each frame's speed. The reference computes rank-based quartiles per session using only the kept frames of that session, ensuring exactly 25% of data per bin within each session. The AI's approach results in uneven distribution (30.2%, 19.8%, 24.9%, 25.1%) because many frames have speed exactly 0, creating ties that percentile-based thresholds cannot split evenly.

ii.
```python
def collect_all_corridor_speeds(session_map, max_sessions=None):
    for sess_key, (exp_type, ndb) in sessions:
        beh = load_behavior_for_session(sess_key, exp_type, ndb)
        corr_mask = beh['ft_CorrSpc'][:n_frames]
        speeds = beh['ft_RunSpeed'][:n_frames]
        all_speeds.append(speeds[corr_mask])
    return np.concatenate(all_speeds)

speed_quartiles = compute_speed_bin_edges(all_speeds)  # np.percentile at [25, 50, 75]

def discretize_speed(speed, quartiles):
    binned = np.digitize(speed, quartiles, right=True)
    return np.clip(binned, 0, 3)
```

iii. CONVERSION_NOTES.md Step 5: "Running speed quartiles: Computed across ALL corridor frames in the dataset." Step 10: "Speed quartile distribution skewed (Q1=9.8%): Fixed by adding right=True to np.digitize."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global percentile thresholds (25th=0.0, 50th=8.37, 75th=30.19 from the output) applied via np.digitize. The reference uses rank-based splitting per session.

ii.
```python
quartiles = np.percentile(valid, [25, 50, 75])
binned = np.digitize(speed, quartiles, right=True)
```

iii. Speed quartiles reported as `[0.0, 8.37, 30.19]` in conversion output.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same StartFr:GrayFr frame range as the neural data.

ii.
```python
trial_speed = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles)
```

iii. Same frame range as neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI clips behavior arrays to `min(n_frames_neural, n_frames_beh)`, validates frame ranges before processing, skips trials with invalid ranges, and clips lick frames to valid range. One trial is documented as skipped due to invalid frame range.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
# ...
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
# ...
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```

iii. CONVERSION_NOTES.md Step 9: "1 trial skipped (TX83_2022_08_31_1: invalid frame range)."

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural spike files, which are very large. The AI reports ~15-44 seconds per session, totaling ~33 minutes for all 89 sessions. Additionally, the AI loads all behavior files a second time to compute global speed quartiles (8.5s overhead).

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```

iii. CONVERSION_NOTES.md Step 7: "Process session ~15s... Total estimate ~25-30 min."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop within `process_session` iterates over all trials. Additionally, `collect_all_corridor_speeds` reads behavior files a second time (separate from main processing) to compute quartiles, which could have been done in the main processing loop. The `build_lick_binary` is called per-trial rather than built once per session.

ii.
```python
for trial_idx in range(ntrials):
    # ... all per-trial processing
```
```python
def collect_all_corridor_speeds(session_map, ...):
    for sess_key, ... in sessions:
        beh = load_behavior_for_session(...)  # re-loads behavior
```

iii. No specific discussion in CONVERSION_NOTES.md.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are loaded twice: once to compute global speed quartiles in `collect_all_corridor_speeds()`, and again during main session processing. The `get_unique_sessions()` function is also called twice (once for the main map, once for `all_session_map` used in day computation).

ii.
```python
all_session_map = get_unique_sessions()  # called a second time
all_speeds = collect_all_corridor_speeds(
    session_map if sample_mode else all_session_map, ...
)
```

iii. No explicit documentation of this redundancy.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The 'other' brain region neurons (iarea -1 and 7) are processed and stored but are not in the standard visual cortex regions used in the reference analyses. This adds 585,641 neurons worth of data. Global speed quartile computation loads all behavior data separately before the main processing loop.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
```

iii. CONVERSION_NOTES.md Step 1: "Outside visual cortex: iarea in {-1, 7} (excluded from density maps but not from dprime)."
