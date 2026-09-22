# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, builds a deduplicated session list, then loads all behavior files at once via `load_all_behavior()`, stripping `_swap1`/`_swap2` suffixes. Neural data (`spk/`) and retinotopy (`retinotopy/`) are loaded per-session in a two-pass approach: Pass 1 collects speed statistics, Pass 2 builds the full dataset.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
all_beh = load_all_behavior(exp_info)
sessions = build_session_list(exp_info)
```

```python
def load_all_behavior(exp_info):
    all_beh = {}
    for exp_type in exp_info.keys():
        beh_path = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
        beh = np.load(beh_path, allow_pickle=True).item()
        for k, v in beh.items():
            base_key = k
            for suffix in ('_swap1', '_swap2'):
                if base_key.endswith(suffix):
                    base_key = base_key[:-len(suffix)]
                    break
            if base_key not in all_beh:
                all_beh[base_key] = v
    return all_beh
```

iii. The AI noted that sessions can appear under multiple experiment types and deduplicates them. It loads all behavior upfront for efficiency.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is extracted from each session entry. A sorted list of unique subjects is built and `subject_idx` maps each session to its subject.

ii.
```python
subject_list = sorted(set(s['mname'] for s in sessions))
subject_idx_list.append(subject_list.index(sess['mname']))
```

iii. The subject is directly available from the index entries.

## 1-c. How are the data split into sessions?

i. A session is identified by the triple `mname_datexp_blk`. The AI deduplicates sessions that appear under multiple experiment types using a dictionary keyed by this triple. This yields 89 unique sessions.

ii.
```python
def build_session_list(exp_info):
    session_dict = {}
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_dict:
                session_dict[key] = { ... }
```

iii. The AI correctly identified that 142 entries map to 89 unique sessions, matching the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd` (trial index per frame) and `ft_CorrSpc` (corridor space flag). For each trial index, frames where both conditions are met are collected. Trials with fewer than 2 frames are discarded.

ii.
```python
def extract_trial_frames(beh, nfr):
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ntrials = beh['ntrials']
    trials = []
    for n in range(ntrials):
        frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
        if len(frames) >= 2:
            trials.append(frames)
        else:
            trials.append(None)
    return trials
```

iii. The AI uses `ft_CorrSpc` to restrict to corridor frames, matching the reference approach.

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters trials with fewer than 2 corridor frames. There is **no maximum trial length filter** — extremely long trials (e.g., 5607 frames from a stalled animal) are kept.

ii.
```python
if len(frames) >= 2:
    trials.append(frames)
else:
    trials.append(None)
```

iii. The AI's CONVERSION_NOTES.md notes the 5607-frame outlier but states "Not filtered per reference code convention." The AI did not implement any trial length filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (deconvolved calcium traces), concatenated across imaging planes. Brain area assignments come from `iarea` in the retinotopy files.

ii.
```python
def load_spk(mname, datexp, blk, root=DATA_ROOT):
    data = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in data['spks']], 0)
    return spk
```

iii. Matches the reference approach of concatenating across planes.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed. Per-trial neural data is extracted by indexing into the spike array with the trial's corridor frame indices. Data is stored as float32.

ii.
```python
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The AI correctly identified that the data is already deconvolved and needs no further processing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons with `iarea` in `{-1, 7}` are excluded. All other neurons (V1, mHV, lHV, aHV) are kept.

ii.
```python
EXCLUDED_AREAS = {-1, 7}
AREA_MAP = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}

def get_neuron_mask_and_regions(iarea):
    iarea_int = iarea.astype(int)
    mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
    region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
    return mask, region_idx
```

iii. Matches the reference approach. The AI noted this matches `Get_density_map` in the reference utils.py.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry. Each trial's neural data starts at the first corridor frame (where `ft_CorrSpc` is True and `ft_trInd` matches). Trials are variable length — no fixed window or padding.

ii.
```python
frames = np.where((ft_trInd == n) & ft_CorrSpc)[0]
neural = spk_filtered[:, frames].astype(np.float32)
```

iii. The AI correctly uses corridor entry as the alignment event, consistent with the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The native imaging frame rate of ~3.17 Hz is used, giving ~315 ms bins. The time bin size is computed from the median inter-frame interval.

ii.
```python
dt_sec = np.median(np.diff(ft)) * 24 * 3600 if len(ft) > 1 else 1.0/FRAME_RATE
...
'time_bin_size': median_dt * 1000,  # in ms
```

iii. The AI correctly preserves the native temporal resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the sound cue frame for each trial) and the frame indices of the trial.

ii.
```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec     # time to sound cue
```

iii. The AI uses `SoundFr` as the sound cue timing variable.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(frame_index - SoundFr) * dt_sec`. This uses frame-index-based timing (multiplying frame differences by a constant dt), rather than interpolating `SoundFr` onto the actual timestamp axis. The sign convention gives negative values before the cue and positive after, which is `time_since_sound_cue`, the opposite of "time_to_sound_cue".

ii.
```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec     # time to sound cue
```

iii. The AI documented this as "time to sound cue" but the sign is inverted relative to the name. The reference computes `cue_time - frame_time` (positive before cue).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the same `frames` array (corridor frame indices) as the neural data, ensuring temporal alignment.

ii.
```python
input_arr[0, :] = (frames - SoundFr[n]) * dt_sec
```

iii. Aligned via shared frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the chronological ordering of sessions per mouse, based on `datexp` (date of experiment).

ii.
```python
def build_session_list(exp_info):
    ...
    for m, sess_list in mouse_sessions.items():
        sess_list.sort(key=lambda s: s['datexp'])
        for i, s in enumerate(sess_list):
            s['day_index'] = i
```

iii. Sessions are sorted by date within each mouse and assigned a 0-indexed day number.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The day index is the 0-based chronological session number per mouse. It is broadcast as a constant across all time bins of every trial in that session.

ii.
```python
input_arr[1, :] = float(sess['day_index'])
```

iii. Simple sequential counting of sessions per mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the frame indices of the trial. The first frame of the trial is treated as time 0.

ii.
```python
input_arr[2, :] = (frames - frames[0]) * dt_sec       # time since trial start
```

iii. Uses frame indices rather than actual timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(frame_index - first_frame_index) * dt_sec`. This assumes uniform frame spacing (constant dt), whereas the reference interpolates `StartFr` onto the actual timestamp axis to handle non-uniform frame timing.

ii.
```python
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. Uses constant-dt approximation rather than actual timestamps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same `frames` array as neural data.

ii.
```python
input_arr[2, :] = (frames - frames[0]) * dt_sec
```

iii. Aligned via shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavior data, which indicates whether each trial is in a rewarded corridor.

ii.
```python
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. Direct use of the reward flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (1.0 or 0.0) and broadcast across all time bins of the trial.

ii.
```python
input_arr[3, :] = 1.0 if isRew[n] else 0.0
```

iii. No complex processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavior data, which names the wall texture for each trial.

ii.
```python
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. Uses `WallName` rather than `TrialStim` (which is masked in swap sessions).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps each of the 15 individual stimulus names (circle1, circle2, leaf1, leaf1_swap1, etc.) to a unique integer index 0-14. This produces **15 output categories** rather than the 4 grouped categories (circle, leaf, rock, wood) used by the reference.

ii.
```python
ALL_STIMULI = sorted(['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', ...])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
output_arr[0, :] = STIM_TO_IDX[str(WallName[n])]
```

iii. The AI's CONVERSION_NOTES.md states: "15 stimulus categories: Use individual stimulus names rather than grouped categories, preserving maximum information." However, the instructions say "Visual stimulus category. e.g. circle, leaf, etc." suggesting grouped categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) in the behavior data.

ii.
```python
lick_fr = beh['LickFr'].astype(int) if len(beh['LickFr']) > 0 else np.array([], dtype=int)
lick_indicator = np.zeros(nfr, dtype=np.int64)
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
if valid_mask.any():
    lick_indicator[lick_fr[valid_mask]] = 1
```

iii. Directly converts lick frame numbers to a binary indicator.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array is created: 1 if a lick occurs at that frame, 0 otherwise. Lick frames are truncated to integer and filtered to be within the valid frame range.

ii.
```python
lick_indicator = np.zeros(nfr, dtype=np.int64)
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
if valid_mask.any():
    lick_indicator[lick_fr[valid_mask]] = 1
```

iii. Matches the reference approach of converting fractional lick frames to integer indices.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick indicator array is indexed by the same frame indices as the neural data.

ii.
```python
lick_arrays.append(lick_indicator[frames].copy())
output_arr[1, :] = lick_arrays[n]
```

iii. Aligned via shared frame indices.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, which gives position in decimeters at each imaging frame.

ii.
```python
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. Direct use of the frame-level position variable.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is integer-divided by 10 to get 1-meter bins, then clipped to [0, 3].

ii.
```python
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. Matches the reference approach.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Floor division by 10 (since position is in decimeters) gives 4 bins: [0-1m), [1-2m), [2-3m), [3-4m). Values are clipped to [0, 3].

ii.
```python
np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. Four equal-length 1-m spatial bins as specified in the instructions.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses the same `frames` array as neural data.

ii.
```python
output_arr[2, :] = np.clip(ft_Pos[frames] // 10, 0, 3).astype(np.int64)
```

iii. Aligned via shared frame indices.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. Direct use of the frame-level speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes **global** speed quartile boundaries (25th, 50th, 75th percentiles) across all corridor frames of all sessions in Pass 1, then uses `np.digitize` with these boundaries in Pass 2. This results in highly uneven bins: Q1 has ~9.8% of data, Q2 has ~40.2%, Q3 and Q4 each ~25%.

ii.
```python
# Pass 1: Collect all corridor speeds
all_speeds_flat = np.concatenate(all_speeds)
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])

# Pass 2: Apply
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. The AI's CONVERSION_NOTES.md notes "Speed bin Q1 has only 9.8%: Not a bug - global 25th percentile is exactly 0.0 (many stopped frames)." The reference uses rank-based per-session quartiles to ensure exactly 25% per bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with three global percentile boundaries, producing 4 bins. Because many frames have exactly 0 speed, value-based thresholds produce very uneven bins.

ii.
```python
speed_quantiles = np.percentile(all_speeds_flat, [25, 50, 75])
np.digitize(ft_RunSpeed[frames], speed_quantiles)
```

iii. The uneven distribution (9.8%, 40.2%, 25%, 25%) is a consequence of the value-based rather than rank-based approach.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Uses the same `frames` array as neural data.

ii.
```python
output_arr[3, :] = np.digitize(ft_RunSpeed[frames], speed_quantiles).astype(np.int64)
```

iii. Aligned via shared frame indices.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior data is truncated to the number of neural frames (`nfr = spk.shape[1]`). Lick frames beyond the neural frame count are excluded via `(lick_fr >= 0) & (lick_fr < nfr)`. Trials with fewer than 2 corridor frames are skipped. Behavior keys with swap suffixes are stripped to find the base session.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr]
ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
...
valid_mask = (lick_fr >= 0) & (lick_fr < nfr)
```

iii. The AI handles the frame count mismatch between behavior and neural data, matching the reference approach.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the neural data files (`spk/`) is the most time-consuming step, as each file contains tens of thousands of neurons. The AI does this twice: once in Pass 1 (just to get frame counts) and once in Pass 2 (full loading).

ii.
```python
# Pass 1 - loads neural data just for shape
data = np.load(path, allow_pickle=True).item()
nfr = data['spks'][0].shape[1]
...
# Pass 2 - loads again for full processing
spk = load_spk(mname, datexp, blk)
```

iii. The two-pass approach doubles I/O for neural data files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The neuron mask computation in `get_neuron_mask_and_regions` uses a Python list comprehension rather than vectorized numpy operations:

ii.
```python
mask = np.array([a not in EXCLUDED_AREAS for a in iarea_int])
region_idx = np.array([AREA_MAP[int(a)] for a in iarea_int[mask]], dtype=np.int64)
```

iii. These could be vectorized using `np.isin` or similar. The trial frame extraction loop could also be vectorized with `np.unique` grouping.

## 12-c. What processing does the code repeat multiple times?

i. The neural data files are loaded twice: once in Pass 1 (`collect_speeds_from_behavior`) to get frame counts, and once in Pass 2 (`process_session_full`) for full processing. The retinotopy files are also loaded twice (once per pass).

ii.
```python
# Pass 1
data = np.load(path, allow_pickle=True).item()
nfr = data['spks'][0].shape[1]
iarea = load_area_ids(mname, datexp)
...
# Pass 2
spk = load_spk(mname, datexp, blk)
iarea = load_area_ids(mname, datexp)
```

iii. This doubles I/O for the largest files in the dataset.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The two-pass approach computes global speed quartiles, but since the resulting bins are very uneven (9.8%, 40.2%, 25%, 25%), the effort does not achieve the goal of equal-sized bins. The first pass loads all neural data files just to get frame counts, which could be avoided by using behavior data frame counts instead.

ii.
```python
# Pass 1 loads neural data just for frame count
data = np.load(path, allow_pickle=True).item()
nfr = data['spks'][0].shape[1]
```

iii. The frame count could be approximated from behavior data, avoiding the need to load all neural files in Pass 1.
