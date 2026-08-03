# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment info from `data/beh/Imaging_Exp_info.npy`, which contains session metadata organized by experiment type (e.g., `unsup_train1_before_learning`, `sup_test1`, etc.). It builds a list of all unique sessions from this info, then for each session loads neural data from `data/spk/`, behavioral data from `data/beh/Beh_<exp_type>.npy`, and retinotopy data from `data/retinotopy/`. 13 sessions are skipped because no matching behavioral data file is found, resulting in 76 out of 89 sessions processed.

ii.
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)

# In build_all_sessions_info:
for exp_type, sess_list in exp_info.items():
    for s in sess_list:
        sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
        if sess_id not in sessions:
            sessions[sess_id] = { ... }
        sessions[sess_id]['exp_types'].append(exp_type)
```

iii. The AI noted that the reference code uses `Imaging_Exp_info.npy` and the `Beh_<exp_type>.npy` files. It documented that 13 sessions were skipped due to missing behavioral data. The agent's trajectory (Step 280) confirms: "All 19 subjects are represented across 76 sessions with 31,442 total trials."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mname` field in each session's metadata. All unique mouse names are collected and sorted, forming the `subjects` list. Each session is mapped to its mouse via `subject_idx`.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
# ...
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. The AI documented 19 subjects matching the paper's stated number. The subject names come from the experiment info metadata field `mname`.

## 1-c. How are the data split into sessions?

i. Sessions are identified by the composite key `{mname}_{datexp}_{blk}` (mouse name, date, block). Each unique combination defines one session. Sessions that appear in multiple experiment types are deduplicated. The AI processes 76 out of 89 sessions (13 skipped due to missing behavioral data).

ii.
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if sess_id not in sessions:
    sessions[sess_id] = {
        'mname': s['mname'],
        'datexp': s['datexp'],
        'blk': s['blk'],
        'exp_types': [],
        ...
    }
```

iii. The AI noted that sessions span multiple experiment types and used deduplication to avoid processing the same neural data twice. The 13 skipped sessions are documented in CONVERSION_NOTES.md.

## 1-d. How are the data split into trials?

i. Trials are defined by the behavioral data arrays `StartFr` and `GrayFr`, which mark the start frame and grey-space entry frame for each trial. The number of trials per session comes from `beh['ntrials']`. Each trial spans from `StartFr[t]` to `GrayFr[t]` (corridor portion only).

ii.
```python
ntrials = beh['ntrials']
for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
    # ...

# In extract_trial_data:
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
```

iii. The AI used the corridor portion (StartFr to GrayFr) as the trial window, consistent with the reference code's focus on the texture corridor portion. This is aligned to trial start (corridor entry) as specified in the instructions.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered minimally: a trial is excluded if `GrayFr - StartFr < 2` (fewer than 2 available frames), or if `start_fr < 0` or `end_fr > spk.shape[1]` (out of neural data bounds). Sessions with fewer than 2 valid trials are skipped. One trial total was excluded (in TX83_2022_08_31_1). Additionally, a hard cap of 1000 frames is imposed via `n_timepoints=1000` in `extract_trial_data`.

ii.
```python
avail_frames = gray_fr - start_fr
if avail_frames < 2:
    return None
end_fr = min(start_fr + n_timepoints, gray_fr)  # n_timepoints=1000
actual_frames = end_fr - start_fr
if actual_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

iii. The AI documented that the reference code does not have explicit trial filtering for quality. The 1000-frame cap is an implicit maximum that affects some very long trials (where the mouse stopped running for extended periods).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`, which contains a dictionary with key `'spks'` — a list of arrays for each imaging plane. Planes are concatenated along the neuron axis.

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    dat = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk
```

iii. The AI directly replicated the reference code's `load_spk` function, which concatenates spike data across imaging planes.

## 2-b. How is the `neural` data processed?

i. The raw deconvolved calcium traces (from Suite2p with 0.75s decay timescale) are used directly without further processing. The data is sliced per-trial from `StartFr` to `GrayFr` and cast to `float16` for storage efficiency. No additional normalization, z-scoring, or rebinning is applied.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI noted: "Neural data: deconvolved Ca2+ traces from Suite2p (decay timescale 0.75s)" and "No explicit neuron quality filtering in reference code." The float16 casting is a storage optimization not present in the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level quality filtering is applied. All neurons from all imaging planes are included. The AI explicitly noted that the reference code does not filter neurons for quality.

ii. No filtering code — all neurons are kept:
```python
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
# No filtering applied
```

iii. CONVERSION_NOTES.md states: "No neuron filtering: Consistent with reference code."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by extracting frames starting at `StartFr[t]`. The first frame of each trial's neural data corresponds to the corridor entry time.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
neural = spk[:, start_fr:end_fr]
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The AI uses `StartFr` which corresponds to corridor entry. This matches the specification.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate of 3.17 Hz, giving a time bin size of ~315.5 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. The AI documented that the frame rate is 3.17 Hz, matching the paper and reference code. The data is kept at native temporal resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundFr']` (the frame at which the sound cue occurs) and `beh['StartFr']` (the trial start frame).

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]
start_fr = int(np.round(beh['StartFr'][trial_idx]))
sound_frame_offset = sound_fr - start_fr
```

iii. The AI correctly identified that `SoundFr` provides the sound cue timing in frame units.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in the trial, the signed time difference from that frame to the sound cue frame is computed: `(frame_index - sound_frame_offset) / FRAME_RATE`. This gives negative values before the cue and positive values after.

ii.
```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The AI designed this as a continuous, time-varying signal representing signed time from each frame to the sound cue, in seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Both use the same frame indexing: frame 0 corresponds to `StartFr` (corridor entry). `time_to_cue[i]` corresponds to `neural[:, i]`. The alignment is inherent because both are indexed relative to the trial start.

ii.
```python
# Both use same frame index within trial
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
neural = spk[:, start_fr:end_fr]
# Both have length actual_frames
```

iii. Frame-level alignment is maintained by using the same frame indices for neural and input data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from `datexp` (date string) in the session metadata from `Imaging_Exp_info.npy`.

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

iii. The AI collects all recording dates for a given mouse across all experiment types, sorts them chronologically, and returns the 0-based index of the current session's date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. All unique dates for a given mouse are collected from the experiment info, sorted, and the index of the current date in that sorted list is used as the day of training (0-based). This value is constant across all timepoints within a trial.

ii.
```python
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. This provides a per-trial constant value broadcast across time. It represents the chronological session index for this mouse, not the actual calendar day count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the frame index within the trial and the frame rate constant `FRAME_RATE = 3.17 Hz`.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. No raw data variable is directly used — it is computed from the frame index.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each frame index `i` is divided by the frame rate (3.17 Hz) to convert to seconds. Frame 0 corresponds to time 0 (corridor entry).

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. Simple linear time axis starting at 0 seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Both share the same frame indexing. Frame `i` of neural data and `time_since_start[i]` both correspond to the same timepoint (i frames after corridor entry).

ii. Same frame indexing as neural data — both arrays have length `actual_frames`.

iii. Alignment is inherent by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a boolean array of length `ntrials` indicating whether each trial is a reward trial.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The AI correctly identified `isRew` as the reward indicator.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean value `isRew[t]` is cast to float (0.0 or 1.0) and broadcast across all timepoints in the trial.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Per-trial constant, matching the instructions specification "1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, an array of stimulus name strings for each trial (e.g., 'circle1', 'leaf1', 'rock1', etc.).

ii.
```python
wall_names = beh['WallName']
stim_name = str(wall_names[t])
```

iii. The AI uses the raw stimulus names directly as categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names are collected across all sessions, sorted alphabetically, and mapped to integer indices. The resulting 13 categories are: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood2, wood5. This integer index is broadcast across all timepoints in the trial.

ii.
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
# ...
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx
```

iii. 13 stimulus categories result, matching the actual diversity of stimuli in the data.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (frame indices of lick events) and `beh['LickTrind']` (trial indices corresponding to each lick).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The AI filters lick events by trial index to get per-trial lick frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary time series is constructed: for each frame in the trial, if a lick occurred at that frame, the value is 1; otherwise 0. Lick frames are converted to offsets relative to trial start and checked against bounds.

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. Creates a binary per-frame licking signal. The AI noted that 96.5% of frames are no-lick, consistent with most sessions being unsupervised (no licking).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames (`LickFr`) are absolute frame indices in the same coordinate system as neural data frames. The offset `lf_int - start_fr` converts to trial-relative frame index, matching the neural data indexing.

ii.
```python
frame_offset = lf_int - start_fr  # Same start_fr used for neural data
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. Both neural and licking data are indexed from `StartFr`, ensuring temporal alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, which contains the VR position for each neural frame.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. `ft_Pos` provides frame-aligned position in VR units (0-40 in the texture corridor).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw position values in VR units are extracted for the trial's frame range and then discretized into bins.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
pos_bins = discretize_position(trial_data['position'])
```

iii. No interpolation or smoothing is applied — raw frame-level positions are used directly.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is discretized into 4 equal-length bins using VR unit edges [0, 10, 20, 30, 40], corresponding to 4 bins of 1 meter each (since 1 VR unit = 0.1m): [0-1m), [1-2m), [2-3m), [3-4m].

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. The instructions specify "4 equal-length, 1-m-long spatial bins." With the corridor being 4m long (40 VR units), 4 bins of 10 VR units each (1m each) matches the specification.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is already frame-aligned (one position value per imaging frame). Extracting `ft_Pos[start_fr:end_fr]` gives the same frame range as the neural data.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
neural = spk[:, start_fr:end_fr]
# Both have same length
```

iii. Both use the same frame indexing, ensuring alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, which contains the running speed for each neural frame.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. `ft_RunSpeed` provides frame-aligned running speed values.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is extracted per-trial and discretized into 4 quartile bins. The quartile edges are computed globally across ALL sessions in a first pass, then applied in a second pass.

ii.
```python
# Pass 1: Collect all speeds from corridor frames
for t in range(ntrials):
    start_fr = int(np.round(beh['StartFr'][t]))
    gray_fr = int(np.round(beh['GrayFr'][t]))
    trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
    speeds.append(trial_speed)

speed_bin_edges = compute_speed_bin_edges(all_speeds)
# edges: np.percentile(flat_speeds, [0, 25, 50, 75, 100])
```

iii. Two-pass approach: first collect all speeds for global quartile computation, then discretize.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 quartile bins using `np.digitize` with edges computed from the 25th, 50th, and 75th percentiles of all running speeds globally.

ii.
```python
def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])  # 0,1,2,3
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. The instructions specify "4 bins, each corresponding to 25% of the data." The actual distribution shows Q1=9.9%, Q2=39.4%, Q3=25.3%, Q4=25.3%, with Q1 being much smaller than 25%. This is because the speed distribution includes many zero-speed frames (mouse stationary), and the 25th percentile edge is at 0, collapsing Q1.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is frame-aligned. Extracting `ft_RunSpeed[start_fr:end_fr]` gives the same frame range as neural data.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. Same frame indexing as neural data ensures alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- **Missing behavioral data**: 13 sessions are skipped (warning printed) when no matching `Beh_<exp_type>.npy` file contains the session ID.
- **Float frame indices**: `StartFr`, `GrayFr` are float64 — rounded to int with `np.round()`.
- **Short trials**: Trials with fewer than 2 frames are excluded (1 trial excluded total).
- **Out-of-bounds frames**: Trials where `start_fr < 0` or `end_fr > spk.shape[1]` are excluded.
- **Empty lick arrays**: Handled gracefully (unsupervised sessions have no licks).
- **Long trials**: Capped at 1000 frames.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
if avail_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

iii. CONVERSION_NOTES.md documents: "Float frame indices rounded to int," "Trials with <2 frames excluded," and "Maximum trial length capped at 1000 frames."

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading neural data (`load_spk`), which involves reading large `.npy` files (up to 90k neurons) from disk. Some sessions take 50-90 seconds just for loading. The total conversion takes about 30-40 minutes for all 89 sessions.

ii.
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
# Output shows: Load: 91.1s for some sessions
```

iii. The conversion output shows load times ranging from 3-91 seconds per session, with later sessions taking longer (possibly due to memory pressure).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking construction loop iterates over individual lick frames to set binary values — this could be vectorized with array indexing. The time_to_cue and time_since_start use list comprehensions instead of `np.arange`.

ii.
```python
# Could be vectorized:
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0

# Could use np.arange:
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)])
```

iii. These are minor inefficiencies given the small number of lick events per trial and the small number of frames per trial.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavioral data files twice: once in Pass 1 (speed collection) and again in Pass 2 (full processing). This means each `Beh_<exp_type>.npy` file is loaded multiple times from disk. The `get_training_day` function is called for every session and iterates over all experiment types/sessions each time.

ii.
```python
# Pass 1: loads beh for speed collection
speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)

# Pass 2: loads beh again for full processing
result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

iii. The two-pass design is intentional (need global speed statistics before processing), but the behavioral data files could have been cached in memory.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `get_training_day` function collects and sorts all dates for a mouse every time it's called, which is redundant computation. The `load_beh_for_session` function checks for swap variants in behavioral data files even though swap sessions are handled as separate sessions. The `build_session_to_beh_map` function is defined but never called (dead code). Also, the code stores neural data as float16, which may lose precision unnecessarily for downstream decoder training.

ii.
```python
def build_session_to_beh_map(exp_info):
    """Build a mapping from session_id to (beh_file, session_key)."""
    # This function is never called in main()
```

iii. The `build_session_to_beh_map` function at lines 82-96 is dead code that was likely part of an earlier approach.
