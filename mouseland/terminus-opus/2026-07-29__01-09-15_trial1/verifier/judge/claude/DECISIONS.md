# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment info from `data/beh/Imaging_Exp_info.npy`, builds a mapping of unique session IDs to experiment types via `build_all_sessions_info()`, then iterates over each session. For each session it loads neural data via `load_spk()`, retinotopy via `load_retino()`, and behavioral data via `load_beh_for_session()` which tries each experiment type file and looks up the session by its plain `sess_id`. The behavioral data loading has a bug: it looks for `sess_id` directly in each behavior file, but some sessions (swap sessions) are stored with keys that include a `_stimtype` suffix (e.g., `sess_id + '_swap1'`). The AI's `load_beh_for_session()` does check for `_swap` suffixed keys, but it first checks if the base `sess_id` exists, and for swap-only sessions (where the base key does not exist in the behavior file), it relies on finding `_swap` prefixed keys. However, 13 sessions are still lost because their behavioral data keys don't match what the code searches for.

ii. Loading experiment info and building session list:
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
```

Loading behavior for a session:
```python
def load_beh_for_session(sess_id, exp_types):
    results = []
    for exp_type in exp_types:
        beh_path = f'data/beh/Beh_{exp_type}.npy'
        if os.path.exists(beh_path):
            beh_all = np.load(beh_path, allow_pickle=True).item()
            if sess_id in beh_all:
                results.append((beh_all[sess_id], sess_id))
            for key in sorted(beh_all.keys()):
                if key.startswith(sess_id + '_swap'):
                    results.append((beh_all[key], key))
    ...
```

iii. The AI documented that 13 sessions were missing because "behavioral data is stored with _swap1/_swap2 suffixes." The AI attempted to handle swap sessions but the lookup logic failed to find behavior for 13 sessions. The reference solution uses `beh_key = session_id + ('_' + entry['stimtype'] if 'stimtype' in entry else '')` which correctly constructs the behavior key including the stimulus type suffix from the experiment info entry.

## 1-b. How are the data split into subjects?

i. The AI extracts `mname` from each session's entry in `exp_info` and collects unique mouse names. Subject indices are assigned as sorted unique mouse names.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
```

iii. This follows the same approach as the reference: subjects are identified by `mname` and sorted. The AI correctly produces 19 subjects.

## 1-c. How are the data split into sessions?

i. A session is identified by `mname_datexp_blk`. The AI builds unique sessions from `exp_info` using `build_all_sessions_info()` which deduplicates by session ID. This produces 89 unique sessions, but 13 are skipped during processing because their behavioral data cannot be found, resulting in 76 sessions.

ii.
```python
def build_all_sessions_info(exp_info):
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if sess_id not in sessions:
                sessions[sess_id] = { ... }
            sessions[sess_id]['exp_types'].append(exp_type)
    return sessions
```

iii. The session identification is correct (same triple as reference), but the code fails to load all 89 sessions due to the behavioral data key mismatch described in 1-a.

## 1-d. How are the data split into trials?

i. The AI extracts trial data using `StartFr` and `GrayFr` frame indices. For each trial, it rounds `StartFr` and `GrayFr` to integers and extracts frames from `start_fr` to `gray_fr`. This differs from the reference, which uses `ft_trInd` (trial index per frame) and `ft_CorrSpc` (inside corridor flag) to identify trial frames.

ii.
```python
def extract_trial_data(spk, beh, trial_idx, n_timepoints):
    start_fr = int(np.round(beh['StartFr'][trial_idx]))
    gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
    ...
    end_fr = min(start_fr + n_timepoints, gray_fr)
    actual_frames = end_fr - start_fr
    neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI uses `StartFr` to `GrayFr` to define the corridor portion, capped at 1000 frames. The reference uses `ft_trInd` and `ft_CorrSpc` boolean masks, which label each frame with its trial and whether it's inside the corridor space. These approaches are similar but not identical — `ft_trInd`/`ft_CorrSpc` is the authors' own labeling of which frames belong to which trial and corridor region, while `StartFr`/`GrayFr` are the start/end frame numbers.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials with fewer than 2 frames (`avail_frames < 2`) and caps trial length at 1000 frames via the `n_timepoints=1000` parameter in `extract_trial_data`. There is no percentile-based filtering of excessively long trials. Sessions with fewer than 2 valid trials are skipped entirely.

ii.
```python
trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)  # large max
if trial_data is None:
    continue
```

```python
if avail_frames < 2:
    return None
end_fr = min(start_fr + n_timepoints, gray_fr)
```

iii. The reference solution computes the 99th percentile of trial lengths across the entire dataset and drops trials exceeding that limit (238.9 frames), removing 382 trials that represent animals sitting still. The AI instead uses a hard cap of 1000 frames, which means extremely long trials (e.g., 5000+ frames of a stationary mouse) are truncated rather than dropped. This is a concerning difference: it includes large amounts of near-stationary data that the reference explicitly removes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the per-session neural data files (`data/spk/<session>_neural_data.npy`), which contains deconvolved calcium traces per imaging plane, concatenated across planes. The retinotopy data (`iarea`) comes from `data/retinotopy/<mouse>_<datexp>_trans.npz`.

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    dat = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk
```

iii. Same variables as reference.

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed — no dF/F computation. Per-trial data is extracted by slicing the concatenated spike matrix from `start_fr` to `end_fr` and cast to float16.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. Consistent with reference — no additional processing needed since data is already deconvolved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does NOT filter neurons by brain region. It keeps all neurons including those in the "unassigned" region (neurons outside V1, mHV, lHV, aHV). The AI includes a 5th brain region called "unassigned" and assigns neurons with unknown areas to it.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
for i, region in enumerate(brain_regions[:-1]):
    if region in region_map:
        neuron_region_idx[region_map[region]] = i
```

iii. The reference solution drops neurons outside the four visual areas (`keep = region >= 0`), keeping 4,105,393 of 4,691,034 neurons. The AI keeps all neurons including 483,862 "unassigned" neurons, totaling 4,047,169 across 76 sessions. This is a significant difference — the reference explicitly excludes non-visual-area neurons.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial data is aligned to corridor entry (`StartFr`). Each trial starts at its `StartFr` frame and extends to `GrayFr` (or 1000 frames, whichever is less). Trials are variable length.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
end_fr = min(start_fr + n_timepoints, gray_fr)
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. Both solutions align to corridor entry (trial start). The AI's approach is similar but uses `StartFr`/`GrayFr` directly rather than the `ft_trInd`/`ft_CorrSpc` frame labels.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate of 3.17 Hz is used directly, giving a time bin of ~315.5 ms.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. Same as reference — no rebinning needed since all data streams are already on the imaging frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (frame number of the sound cue for each trial) and the frame index within the trial.

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]
sound_offset = trial_data['sound_frame_offset']  # = sound_fr - start_fr
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
```

iii. The reference uses `SoundFr` and `ft` (frame timestamps) while the AI uses `SoundFr` and frame indices divided by `FRAME_RATE`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes `(frame_index - sound_frame_offset) / FRAME_RATE`, which gives time since sound cue (positive after cue, negative before). The variable is named `time_to_sound_cue` but the sign convention gives time *since* the cue, not time *to* the cue. The reference computes `cue_time - frame_time`, which is positive before the cue and negative after.

ii.
```python
sound_offset = trial_data['sound_frame_offset']  # sound_fr - start_fr
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The sign is inverted compared to the reference. The reference computes `cue[trial] - time` (positive = cue is in the future), while the AI computes `frame_index - sound_offset` (positive = cue is in the past). Additionally, the AI uses frame_index/FRAME_RATE as an approximation of time rather than actual frame timestamps from `ft`, which may introduce small timing inaccuracies.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed for the same frame range (`start_fr` to `end_fr`) as the neural data, so alignment is by construction.

ii.
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
```

iii. Same alignment approach as reference — computed over the same frames as neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (recording date) for each mouse across all sessions in `exp_info`.

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

iii. Same approach as reference — count unique recording dates per mouse in chronological order.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Unique recording dates for each mouse are sorted chronologically, and the day index is the position in this sorted list (0-based). The value is broadcast across all frames of every trial in that session.

ii.
```python
sorted_dates = sorted(dates)
return sorted_dates.index(datexp)
```

```python
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The AI computes training day based on unique dates across ALL experiment types in `exp_info`, which is the same approach as the reference. Both count the number of recording days for each mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the frame index within the trial and the frame rate.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The reference uses `ft` (actual frame timestamps) and `StartFr` interpolated onto the time axis. The AI approximates with `frame_index / FRAME_RATE`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Simply divides frame index by frame rate: `frame_index / FRAME_RATE`. This assumes perfectly uniform frame spacing, whereas the reference interpolates actual timestamps.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The reference computes `time - start[trial]` using interpolated real timestamps from `ft`, accounting for actual frame timing. The AI's approximation assumes uniform spacing, which may not perfectly reflect reality but should be close.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed for the same frame range as neural data.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. Same alignment approach — computed over same frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` in the behavioral data, which indicates whether the trial was in a rewarded corridor.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Same as reference.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to float and broadcast across all frames of the trial. No other processing.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Same as reference.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` in the behavioral data, which names the texture on the corridor walls for each trial.

ii.
```python
stim_name = str(wall_names[t])
```

iii. Same source variable as reference.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses raw `WallName` values directly as stimulus categories, resulting in 13 distinct categories (e.g., circle1, circle2, circle3, leaf1, leaf2, leaf3, leaf1_swap1, etc.). The reference groups these into 4 base texture categories (circle, leaf, rock, wood) using a mapping table.

ii.
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
# In output construction:
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx
```

Output values:
```python
output_values = [
    all_stim_names,  # visual stimulus categories - 13 raw names
    ...
]
```

iii. The reference groups wall textures into 4 categories: `TEXTURE = {'circle1': 'circle', 'circle2': 'circle', ...}` with `STIMULUS = ['circle', 'leaf', 'rock', 'wood']`. The AI keeps all 13 raw names, which makes the classification task much harder (13-way vs 4-way) and doesn't match the instruction's "visual stimulus category, e.g. circle, leaf, etc."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame numbers of licks) and `LickTrind` (trial index of each lick).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The reference uses only `LickFr` and creates a session-wide binary array. The AI additionally uses `LickTrind` to filter licks by trial.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frames are filtered to the current trial using `LickTrind`, then each lick frame is rounded and converted to a frame offset relative to trial start. A binary array is set to 1 at each lick frame.

ii.
```python
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. The reference creates a session-wide lick vector and then slices it per trial. The AI filters licks per trial using `LickTrind`. Both approaches should produce similar results, but the reference approach is simpler and avoids potential edge cases with `LickTrind` indexing.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to offsets relative to `start_fr`, and only licks within the trial's frame range are included. This ensures alignment with the neural data.

ii.
```python
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. Aligned by the same frame range as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in VR units at each imaging frame.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. Same source as reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is discretized using `np.digitize` with bin edges at [0, 10, 20, 30, 40] VR units (corresponding to 0-1m, 1-2m, 2-3m, 3-4m), then clipped to 0-3.

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]

def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. The reference uses `np.clip(ft_Pos // 10, 0, 3)`. The AI uses `np.digitize` with bin edges, which should produce equivalent results for the corridor portion. Both divide into 4 equal 1m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. 4 bins of 1m each: [0-10), [10-20), [20-30), [30-40] VR units, corresponding to [0-1m), [1-2m), [2-3m), [3-4m].

ii.
```python
bins = np.digitize(position, bin_edges[1:])  # edges at 10, 20, 30, 40
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. Same 4-bin scheme as reference.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Extracted from the same frame range (`start_fr:end_fr`) as the neural data.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. Same alignment approach.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. Same source as reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI collects running speeds from all corridor frames across all sessions in a first pass, computes global percentile-based quartile edges, then discretizes each trial's speed using `np.digitize` with these global edges.

ii.
```python
def compute_speed_bin_edges(all_speeds):
    flat_speeds = np.concatenate(all_speeds)
    flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges

def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. The reference computes quartiles per session using rank-based splitting (`quartiles()` function), which guarantees exactly 25% of frames in each bin within each session. The AI uses global percentile thresholds, which means individual sessions may have very unequal bin counts. The verification output confirms this: speed bin Q1 has only 9.9% of data while Q2 has 39.4%, indicating the bins are far from equal. The reference's per-session rank approach ensures 25% per bin per session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Global quartile edges computed from all sessions' corridor speeds, then `np.digitize` to assign bins 0-3.

ii.
```python
edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
# Speed quartile edges: [-34.31, 0.0, 7.88, 30.0, 163.49]
```

iii. The global percentile approach gives uneven bins per session. The 25th percentile is at 0 speed, meaning the first bin captures all zero-speed frames, but its global fraction is only 9.9% because some sessions have more stationary frames than others.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Extracted from the same frame range as neural data.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. Same alignment approach.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: trials with fewer than 2 frames are skipped; frame indices are rounded with `np.round()`; sessions with missing behavioral data are skipped with a warning (13 sessions); sessions with fewer than 2 valid trials are skipped. Trial length is hard-capped at 1000 frames rather than using a data-driven threshold.

ii.
```python
if avail_frames < 2:
    return None

if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. The reference cuts all streams to the number of imaged frames (`n_frames = spikes.shape[1]`), drops empty trials, and removes trials beyond the 99th percentile length. The AI skips sessions where behavior isn't found, but doesn't truncate behavior streams to neural frame count (it relies on StartFr/GrayFr being within bounds). The AI's frame rounding with `np.round()` differs from the reference's integer truncation with `.astype(int)`.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files is the most time-consuming step. Each session's neural data file is large, and `load_spk()` takes several seconds per session. The AI also does a separate first pass to collect running speeds, adding ~14 seconds.

ii.
```python
spk = load_spk(mname, datexp, blk)  # Load time reported per session
```

iii. Same bottleneck as reference — I/O for neural data dominates.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking computation uses a Python for-loop over each lick frame:
```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
This could be vectorized. The time-to-cue and time-since-start computations also use list comprehensions instead of numpy vectorized operations.

ii.
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)])
```

iii. These could be replaced with `np.arange(actual_frames) / FRAME_RATE` and similar vectorized operations.

## 12-c. What processing does the code repeat multiple times?

i. The AI does two full passes over all sessions: Pass 1 collects running speeds for quartile computation, Pass 2 processes all sessions. Pass 1 reloads behavioral data that will be reloaded again in Pass 2. The behavioral data files are loaded once per session per pass rather than being cached across sessions that share the same behavior file.

ii.
```python
# Pass 1
for i, sess_id in enumerate(session_ids):
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)

# Pass 2
for i, sess_id in enumerate(session_ids):
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

iii. The reference avoids this by computing speed quartiles per-session (not globally), so only one pass is needed. Also, the reference groups sessions by behavior file and loads each behavior file only once for all its sessions.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes "unassigned" neurons (those outside the four visual areas) in the neural data, which adds ~483,862 neurons that the reference discards. These neurons may add noise without useful signal for the decoder. The two-pass approach for global speed quartiles is also unnecessary compared to per-session computation.

ii.
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
```

iii. Including unassigned neurons inflates the dataset and may reduce decoder performance by including neurons from non-visual areas that may not carry relevant information for the task.
