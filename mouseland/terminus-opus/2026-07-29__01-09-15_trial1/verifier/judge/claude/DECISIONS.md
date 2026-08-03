# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from three subdirectories: `data/beh/` for behavior, `data/spk/` for neural traces, and `data/retinotopy/` for brain region assignment. It first reads the master index `Imaging_Exp_info.npy`, builds a list of all unique sessions (by `mname_datexp_blk`), then for each session loads the behavioral data from the appropriate `Beh_<exp_type>.npy` file, the neural data from `<session_id>_neural_data.npy`, and retinotopy from `<mname>_<datexp>_trans.npz`. The script does two passes: Pass 1 collects running speeds for global quartile computation, Pass 2 processes all sessions.

ii.
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
# Per session:
spk = load_spk(mname, datexp, blk)  # np.concatenate(dat['spks'], 0)
retino = load_retino(mname, datexp)  # dtrans['iarea']
beh_all = np.load(f'data/beh/Beh_{exp_type}.npy', allow_pickle=True).item()
```

iii. The AI documented that it follows the reference code's loading approach (load_spk concatenates planes, load_retino gets iarea). The two-pass approach was chosen to compute global speed quartile edges before processing.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the `mname` field in the experiment info. All unique mouse names are collected from processed sessions and sorted. Each session is assigned a `subject_idx` mapping into this list.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. The mouse name is directly available in the session metadata, so no derivation is needed.

## 1-c. How are the data split into sessions?

i. A session is identified by `mname_datexp_blk`. The AI deduplicates sessions that appear under multiple experiment types by keeping only the first occurrence. However, swap sessions (which have different trial structures but share neural data) are skipped — the AI only processes 76 of 89 sessions.

ii.
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if sess_id not in sessions:
    sessions[sess_id] = { ... }
sessions[sess_id]['exp_types'].append(exp_type)
```

iii. The AI noted that 13 swap sessions were skipped because their behavioral data is stored with `_swap1`/`_swap2` suffixes. The AI decided these were "not critical for the decoder task" and focused on the 76 non-swap sessions.

## 1-d. How are the data split into trials?

i. Trials are split using `StartFr` and `GrayFr` frame indices from the behavioral data. For each trial index (0 to ntrials-1), the AI extracts frames from `StartFr[t]` (rounded to int) to `GrayFr[t]` (rounded to int), giving the corridor portion of each trial. Trials with fewer than 2 frames are skipped. Trial length is variable (not fixed).

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
avail_frames = gray_fr - start_fr
end_fr = min(start_fr + n_timepoints, gray_fr)
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI chose to use `StartFr` to `GrayFr` as the trial window, reasoning this is "when visual stimuli are present and the task is active." The reference code instead uses `ft_trInd` and `ft_CorrSpc` boolean masks to identify frames belonging to each trial within the textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. The only filter is that trials with fewer than 2 available frames (between StartFr and GrayFr) are skipped. Sessions with fewer than 2 valid trials are also skipped.

ii.
```python
if avail_frames < 2:  # need at least 2 frames
    return None
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. No quality-based trial filtering is applied, consistent with the reference approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the neural data files (`<session_id>_neural_data.npy`), which contains deconvolved calcium traces organized by imaging plane. The planes are concatenated into a single neurons-by-frames matrix.

ii.
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```

iii. The AI documented this matches the reference code's `load_spk` function.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed — no dF/F or additional deconvolution. Per-trial neural data is extracted by slicing columns from StartFr to GrayFr and cast to float16. Trials are variable length (no padding to fixed size).

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI noted the data already contains deconvolved traces from Suite2p, so no further processing is needed. float16 was chosen to reduce file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps ALL neurons, including those outside the four visual areas (V1, mHV, lHV, aHV). These extra neurons are labeled "unassigned" (index 4 in the brain_regions list). The reference code drops neurons outside these four areas.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
for i, region in enumerate(brain_regions[:-1]):
    if region in region_map:
        neuron_region_idx[region_map[region]] = i
```

iii. The AI reasoned that "the reference code doesn't filter neurons" and kept all Suite2p-detected neurons. However, the reference code's `load_spikes` function explicitly filters with `keep = region >= 0` and `return spikes[keep], region[keep]`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to corridor entry (trial start) by extracting frames starting at `StartFr`. The trial length is variable — from `StartFr` to `GrayFr` — with no fixed window or padding. The reference uses a fixed window of N_FRAMES=32 with zero-padding for shorter trials.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI stated it preserves "natural trial length" and that "variable-length trials are fine for the decoder."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate of 3.17 Hz is preserved, giving a time bin of ~315.5 ms. This matches the reference.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. The imaging frame is the finest available resolution and all behavioral streams are already on the same grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the frame at which the sound cue was played) and `StartFr` (the trial start frame). The difference gives a frame offset, which is converted to time.

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]  # keep as float for precision
sound_frame_offset = sound_fr - start_fr
```

iii. The AI computes a frame-based offset rather than interpolating onto the timestamp axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `time_to_cue = (frame_index - sound_frame_offset) / FRAME_RATE`. This gives negative values before the cue and positive values after. The reference computes `cue_time - frame_time` (positive before cue, negative after) using interpolated timestamps from the `ft` array.

ii.
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI's sign convention is the opposite of the reference. The instructions say "Time to sound cue" which implies time remaining until the cue (positive before, zero at cue, negative after), matching the reference's convention.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same frame indices used for the neural data (StartFr to GrayFr), so it is naturally aligned.

ii.
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. All data streams share the same frame indexing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `datexp` field (date string) of each session entry in the experiment info, collected across all experiment types for each mouse.

ii.
```python
def get_training_day(mname, datexp, exp_info):
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
    sorted_dates = sorted(dates)
    return sorted_dates.index(datexp)
```

iii. The date string is used to determine the chronological ordering of sessions for each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. All unique recording dates for a mouse are collected across all experiment types, sorted chronologically, and the 0-based index of the current session's date gives the training day. This is broadcast across all timepoints of each trial. The reference computes the same thing but counts sessions (not unique dates) in sorted order; since session IDs contain the date and sort by date within a mouse, the result is similar but may differ if a mouse has multiple sessions on the same date or multiple blocks.

ii.
```python
training_day = get_training_day(mname, datexp, exp_info)
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The AI uses unique dates rather than session count. The reference counts sessions in sorted session_id order, which may produce different values when there are multiple blocks on the same day.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Computed from the frame index within the trial, divided by the frame rate. The reference uses `ft` timestamps and `StartFr` interpolated onto the time axis.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI uses a simple frame-counting approach rather than actual timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each frame's time since trial start is `frame_index / FRAME_RATE` in seconds, starting at 0. The reference interpolates `StartFr` onto the `ft` timestamp axis and subtracts, which accounts for any irregular frame timing.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI assumes perfectly regular frame spacing at 3.17 Hz. The reference uses actual frame timestamps.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. It uses the same frame count as the neural data for each trial, so alignment is inherent.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. All data streams use the same frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, a per-trial flag indicating whether the corridor is rewarded.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Directly available in the behavioral data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew` is cast to float (0.0 or 1.0) and broadcast across all timepoints of the trial. This matches the reference approach.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. No additional processing needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the texture name for each trial. The AI uses the raw wall name strings directly (e.g., "circle1", "leaf2") without grouping.

ii.
```python
stim_name = str(wall_names[t])
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
```

iii. The AI simply uses `WallName` as-is.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps all 13 distinct wall names as separate stimulus categories and encodes them as integer indices. The reference groups the 15 wall names into 4 base textures (circle, leaf, rock, wood) using a hardcoded mapping dictionary. The instructions say "Visual stimulus category. e.g. circle1, leaf2, etc." which could support either approach, but the reference paper and code consistently use 4 texture categories.

ii.
```python
stim_name = str(wall_names[t])
# ...
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx  # visual stimulus (per-trial, repeated)
```

iii. The AI noted the resulting 13-class visual stimulus had "very unbalanced weights" but did not consider grouping into 4 categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) and `LickTrind` (trial index of each lick). The reference uses only `LickFr`.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The AI filters licks by trial using `LickTrind` before placing them.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, licks are filtered by `LickTrind == trial_idx`, then each lick's frame is rounded to int, offset from `StartFr`, and marked as 1 in a binary array. The reference creates a single licking array for the entire session by setting `licking[LickFr.astype(int)] = 1`, then slices per trial.

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. Both approaches produce binary licking indicators, but the mechanism differs.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking uses the same StartFr-to-GrayFr frame window as the neural data, so alignment is inherent.

ii.
```python
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. Both the neural data and licking are extracted from the same frame range.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position at each imaging frame in VR units (decimeters).

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. Directly available per frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in VR units is discretized into 4 bins using `np.digitize` with edges at [10, 20, 30, 40] VR units (i.e., 1m bins). The reference uses floor division by 10 and clips to 0-3.

ii.
```python
def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. Both approaches produce 4 bins of 1m each.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Using `np.digitize` with bin edges [10, 20, 30, 40] in VR units, giving bins 0-3 for [0-10), [10-20), [20-30), [30-40+] in VR units (i.e., [0-1m), [1-2m), [2-3m), [3-4m+]).

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. Equivalent to the reference's floor division approach for the corridor range.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position uses the same StartFr-to-GrayFr frame indices as the neural data.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. All data streams share the same frame window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. Directly available per frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global speed quartile edges across ALL sessions in Pass 1 using `np.percentile`, then discretizes each session's speeds using these fixed edges. The reference computes quartiles per-session using rank-based splitting on kept frames only, ensuring exactly 25% of data in each bin even with many tied values (e.g., zero speed).

ii.
```python
def compute_speed_bin_edges(all_speeds):
    flat_speeds = np.concatenate(all_speeds)
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges

def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. The AI chose global quartiles for consistency across sessions but this differs from the reference's per-session rank-based approach.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with globally computed percentile edges at 25%, 50%, 75%. The reference uses a rank-based split per session.

ii.
```python
bins = np.digitize(speed, bin_edges[1:-1])  # 0,1,2,3
bins = np.clip(bins, 0, N_SPEED_BINS - 1)
```

iii. Using percentile thresholds rather than rank-based splitting can produce unequal bin sizes when many values are tied (e.g., zero speed).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed uses the same StartFr-to-GrayFr frame indices as the neural data.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. All data streams share the same frame window.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavioral data arrays are not explicitly cut to the number of imaged frames (unlike the reference which uses `n_frames = spikes.shape[1]`). Trials with fewer than 2 frames between StartFr and GrayFr are dropped. Sessions with fewer than 2 valid trials are skipped. LickFr values are rounded and checked against bounds. The reference cuts all behavioral streams to the number of neural frames first.

ii.
```python
if avail_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

iii. The AI handles edge cases through bounds checking but does not explicitly synchronize behavioral and neural data lengths.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the neural data files (which total hundreds of GB). The AI also does a two-pass approach (Pass 1 for speed collection, Pass 2 for full processing), which doubles the behavioral data I/O but not the neural data I/O (neural data is only loaded in Pass 2).

ii.
```python
spk = load_spk(mname, datexp, blk)  # Loading large .npy files
```

iii. The I/O cost of loading neural data dominates processing time.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking computation uses a Python for-loop over individual lick frames per trial:
```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
This could be vectorized using array indexing as the reference does. The time_to_cue and time_since_start also use list comprehensions instead of vectorized numpy operations.

ii.
```python
# Could be vectorized:
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)])
# Better: np.arange(actual_frames) / FRAME_RATE
```

iii. These loops are small relative to I/O cost but are unnecessarily slow for a vectorized language.

## 12-c. What processing does the code repeat multiple times?

i. The behavioral data files are loaded twice — once in Pass 1 for speed collection and again in Pass 2 for full processing. The reference loads each behavior file once and processes all its sessions together.

ii.
```python
# Pass 1:
speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
# Pass 2:
result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

iii. The two-pass approach was chosen to compute global speed quartiles, but it doubles the behavioral file I/O.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes "unassigned" neurons (those outside V1/mHV/lHV/aHV), which the reference filters out. These extra neurons (~484k out of ~4.7M total) increase file size and processing time but may not contribute meaningful visual cortex signal for decoding. The two-pass speed approach also collects speeds from corridor frames defined by StartFr/GrayFr rather than from the frames actually kept in the final dataset.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
```

iii. The AI reasoned the reference code "doesn't filter neurons" but misread the reference's `load_spikes` which does filter.
