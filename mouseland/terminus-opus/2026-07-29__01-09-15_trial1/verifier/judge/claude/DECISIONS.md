# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment info from `data/beh/Imaging_Exp_info.npy`, which contains session metadata organized by experiment type. It builds a unified list of all unique sessions by iterating over all experiment types in `exp_info`. For each session, it loads neural data from `data/spk/` via `load_spk()`, behavioral data from `data/beh/Beh_{exp_type}.npy` via `load_beh_for_session()`, and retinotopy data from `data/retinotopy/` via `load_retino()`. Processing occurs in two passes: Pass 1 collects running speeds for quartile computation, Pass 2 processes each session fully.

ii.
```python
# Load experiment info
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()

# Build session list
all_sessions = build_all_sessions_info(exp_info)  # iterates exp_info.keys()

# For each session:
spk = load_spk(mname, datexp, blk)  # loads and concatenates planes
beh_results = load_beh_for_session(sess_id, exp_types)  # loads from Beh_{exp_type}.npy
retino = load_retino(mname, datexp)  # loads retinotopy
```

iii. The AI followed the same loading pattern as the reference code (`utils.load_spk`, `utils.load_exp_beh`, `utils.load_retino`). The approach of building unique sessions from `exp_info` is consistent with how the reference `data_process_script.ipynb` iterates through experiment types.

## 1-b. How are the data split into subjects (mice)?

i. Each session is associated with a mouse name (`mname`) from the experiment info. The AI collects all unique mouse names from successfully processed sessions, creates a mapping from mouse name to index, and assigns each session a `subject_idx`.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
# For each session:
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. This follows the natural structure of the data where `mname` identifies each mouse. The AI found 19 subjects, matching the paper's "19 mice."

## 1-c. How are the data split into sessions?

i. Sessions are identified by the composite key `{mname}_{datexp}_{blk}` (mouse name, experiment date, block). The AI deduplicates sessions that appear under multiple experiment types. Each unique session becomes one entry in the output data structure.

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

iii. The AI identified 89 unique sessions from exp_info, matching the paper's "89 recordings." However, 13 sessions were skipped due to missing behavioral data, resulting in 76 sessions in the final output.

## 1-d. How are the data split into trials?

i. For each session, trials are defined by the behavioral data's `ntrials` count. Each trial spans from `StartFr` (corridor entry frame) to `GrayFr` (gray space entry frame). The AI iterates through all trials in each session.

ii.
```python
ntrials = beh['ntrials']
for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
    # StartFr to GrayFr defines the corridor portion
    start_fr = int(np.round(beh['StartFr'][trial_idx]))
    gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
```

iii. Using `StartFr` to `GrayFr` extracts the corridor (texture) portion of each trial, excluding the gray inter-trial interval. This is consistent with the reference code's focus on corridor activity.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: (1) trials with fewer than 2 frames between StartFr and GrayFr are excluded, (2) trials where StartFr or the end frame falls outside the neural data range are excluded, (3) sessions with fewer than 2 valid trials are excluded. There is no filtering based on running/movement.

ii.
```python
avail_frames = gray_fr - start_fr
if avail_frames < 2:  # need at least 2 frames
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
# ...
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. The CONVERSION_NOTES.md mentions only 1 trial was excluded across all sessions (in TX83_2022_08_31_1). The AI noted that the reference code does not have explicit neuron quality filtering, though the reference code does consistently filter for running frames (`VRmove = beh['ft_move']>0`), which the AI did NOT implement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` field in the neural data `.npy` files (deconvolved calcium traces from Suite2p). Multiple imaging planes are concatenated.

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    dat = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk
```

iii. This matches the reference `utils.load_spk()` which also concatenates planes from `dat['spks']`.

## 2-b. How is the `neural` data processed?

i. The AI extracts raw deconvolved calcium traces per trial (from StartFr to GrayFr) and stores them as float16. No normalization, z-scoring, or spatial interpolation is applied. The reference code's main processing step is spatial interpolation (`get_interpPos_spk`) which converts neural data from time-domain to position-domain, but this is not appropriate for the temporal decoder format.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The AI chose to keep raw deconvolved traces aligned to time rather than position-interpolating, since the target format requires temporal alignment. However, storing as float16 reduces precision of the calcium traces.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies NO neuron-level quality filtering. All neurons from all planes are included. No filtering based on running/movement frames is applied.

ii.
```python
# No neuron filtering code exists. All neurons are included:
n_neurons = spk.shape[0]
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The CONVERSION_NOTES.md states "No explicit neuron quality filtering in reference code" which is true for neuron selection. However, the reference code consistently filters for running FRAMES (`VRmove = beh['ft_move']>0`) - the AI did not implement this frame-level filter. The methods text explicitly states: "We only considered timepoints during running for analysis, which removed time periods when the task mice stopped to collect water rewards."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by extracting frames from `StartFr` onwards. The instructions specify "Temporally aligned based on trial start (corridor entry)."

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The alignment to corridor entry (StartFr) matches the instructions. The trial end is defined by GrayFr (entering gray space), so each trial covers the full corridor traversal.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate of 3.17 Hz, giving a time bin of ~315.5 ms. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
```

iii. This is consistent with the reference paper's recording frame rate. The reference code also uses native frame rate without rebinning.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundFr']` (the neural frame index of the sound cue delivery) and `beh['StartFr']` (the corridor entry frame).

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]  # keep as float for precision
sound_frame_offset = sound_fr - start_fr
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. SoundFr gives the frame when the sound cue was delivered in each trial. The offset from trial start is computed and converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame `i` in the trial, the time to cue is computed as `(i - sound_frame_offset) / FRAME_RATE` in seconds. This is a signed value: negative before the cue, positive after. The sound_frame_offset is the number of frames from trial start to the sound cue.

ii.
```python
sound_offset = trial_data['sound_frame_offset']  # frame offset of sound from trial start
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. This creates a continuous time-varying signal representing the time elapsed since (or remaining until) the sound cue, consistent with the instruction "Time to sound cue, continuous, time-varying."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time-to-cue array has the same number of timepoints as the neural data for each trial (both derived from the same StartFr to GrayFr frame range). Frame index `i` in time_to_cue corresponds to frame index `i` in the neural data.

ii.
```python
# Both share actual_frames = end_fr - start_fr
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
neural = spk[:, start_fr:end_fr]  # same actual_frames length
```

iii. Direct frame-level alignment ensures temporal consistency.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field (experiment date string) in the session metadata within `exp_info`.

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

iii. The AI collects all experiment dates for each mouse and computes a 0-based chronological index.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, all unique experiment dates are collected from `exp_info`, sorted chronologically, and the current session's date is mapped to its index (0-based). This index is then replicated as a constant across all timepoints in each trial.

ii.
```python
training_day = get_training_day(mname, datexp, exp_info)
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. This produces a session-level ordinal day index (e.g., 0, 1, 2, ..., 6 for a mouse with 7 sessions). The instruction says "Day of training, continuous, per-trial."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did NOT include "Environment type" as a separate input variable. The instructions list only 4 inputs: time to sound cue, day of training, time since trial start, and reward availability. Environment type (supervised vs unsupervised vs grating) is not among them.

ii. No code for environment type input exists. The AI's `exptype` field in session info is used only for logging/display, not as a decoder input.

iii. The instructions do not list environment type as a decoder input, so this omission is consistent with the task specification.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable - environment type was not included as an input variable. See 4-a above.

ii. N/A

iii. N/A

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived implicitly from the frame index within each trial. No raw data variable is directly used; instead, the frame index `i` (starting from 0 at StartFr) is divided by the frame rate.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. This gives a linearly increasing time signal from 0 seconds at trial start.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Each frame index `i` (0-indexed from trial start) is divided by the frame rate (3.17 Hz) to convert to seconds.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. Straightforward conversion from frame index to time in seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The array length matches the neural data (both use `actual_frames = end_fr - start_fr`). Frame 0 in both corresponds to the StartFr of the trial.

ii.
```python
# Same actual_frames length for both
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)])
neural = spk[:, start_fr:end_fr]
```

iii. Direct frame-level alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a boolean array indicating whether each trial is a reward trial.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. `isRew` is a standard behavioral variable in the dataset indicating reward corridor trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean `isRew[t]` value is converted to float (0.0 or 1.0) and replicated as a constant across all timepoints in the trial.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Consistent with instructions: "Reward availability: 1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which contains the stimulus name for each trial (e.g., 'leaf1', 'circle1', 'rock1').

ii.
```python
wall_names = beh['WallName']
stim_name = str(wall_names[t])
```

iii. `WallName` contains the per-trial wall texture name, consistent with the reference code's usage.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each unique `WallName` string is collected across all sessions, sorted alphabetically, and mapped to an integer index (0 to 12 for 13 categories). This integer is replicated across all timepoints as a per-trial constant.

ii.
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx  # repeated across time
```

iii. The 13 categories found are: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood2, wood5. The instruction says "Visual stimulus category, e.g. circle1, leaf2, etc., per-trial."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (neural frame indices of lick events) and `beh['LickTrind']` (trial index of each lick event).

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. These are the standard licking variables in the behavioral data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick events are identified by filtering `LickFr` to the current trial using `LickTrind`. Each lick frame is converted to a frame offset from trial start. A binary (0/1) time series is constructed where frames with lick events are set to 1.

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. This creates a binary time-varying signal as specified: "Licking, binary, time-varying. 0 = not licking, 1 = licking."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to offsets from StartFr, matching the neural data's frame indexing. The licking array has the same length as the neural data timepoints.

ii.
```python
frame_offset = lf_int - start_fr  # offset from trial start, same as neural indexing
```

iii. Direct frame-level alignment through shared StartFr reference.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, the VR position for each neural frame.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. `ft_Pos` gives the position inside the VR corridor at each imaging frame.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Raw position values (in VR units, where 1 VR unit = 0.1m) are extracted for the trial frames (StartFr to GrayFr), then discretized into bins.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
pos_bins = discretize_position(trial_data['position'])
```

iii. The position is extracted directly from the behavioral data at imaging frame resolution.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position (in VR units, 0-40 for the 4m corridor) is discretized into 4 equal bins of 10 VR units (1m) each: [0-10), [10-20), [20-30), [30-40].

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]  # 4 bins of 1m each in VR units
def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. Consistent with instruction: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed with the same frame range (start_fr:end_fr) as the neural data, ensuring frame-by-frame alignment.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
neural = spk[:, start_fr:end_fr]
```

iii. Direct frame-level alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, the running speed for each neural frame.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. `ft_RunSpeed` provides running speed at each imaging frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed values are extracted for trial frames, then global quartile edges are computed across all sessions (Pass 1) and used to discretize speed into 4 bins. The quartile edges are computed from ALL frames in the corridor (StartFr to GrayFr), including non-running frames.

ii.
```python
# Pass 1: collect speeds
trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
speeds.append(trial_speed)
# Compute quartiles
speed_bin_edges = compute_speed_bin_edges(all_speeds)  # [0, 25, 50, 75, 100] percentiles
# Pass 2: discretize
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. The reported speed bin edges are `[-34.3, 0.0, 7.9, 30.0, 163.5]`, where the first quartile includes negative and zero speeds, suggesting non-running frames are included. The reference code filters for running frames only (`VRmove = beh['ft_move']>0`).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 bins using global quartile edges (25th, 50th, 75th percentiles across all data).

ii.
```python
def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])  # 0,1,2,3
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. Consistent with instruction: "Running speed discretized into 4 bins, each corresponding to 25% of the data." However, because non-running frames are included, the Q1 bin (lowest 25%) contains mostly stationary/negative speed values, and the distribution is skewed: Q1=9.9%, Q2=39.4%, Q3=25.3%, Q4=25.3%.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is indexed with the same frame range (start_fr:end_fr) as neural data.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
neural = spk[:, start_fr:end_fr]
```

iii. Direct frame-level alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (1) Float frame indices (StartFr, SoundFr, GrayFr) are rounded to integers. (2) Trials with fewer than 2 frames are excluded. (3) Sessions without behavioral data are skipped (13 sessions). (4) Sessions with fewer than 2 valid trials are skipped. (5) Trial data extraction is capped at 1000 frames maximum per trial.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
if avail_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
# Max cap:
trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
end_fr = min(start_fr + n_timepoints, gray_fr)
```

iii. The 1000-frame cap means very long trials (where the mouse stopped for extended periods) are truncated rather than excluded. The verification output shows some sessions have max T of 1000, confirming truncation occurs. The 13 missing sessions are documented in CONVERSION_NOTES.md.

## 12-a. What are the most time-consuming steps of the code?

i. Neural data loading (`load_spk`) is the primary bottleneck, taking 2-21 seconds per session. The conversion log shows total processing takes ~10-30 seconds per session, with loading dominating. Total conversion time for 76 sessions was documented.

ii.
```python
# Timing logged per session:
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
# Example output: Load: 5.2s, Total: 8.1s
```

iii. The neural data files are large (20K-90K neurons x thousands of frames), making I/O the bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The licking construction loop iterates over individual lick frames per trial, which could be vectorized using numpy array operations. The time_to_cue and time_since_start construction loops could also be vectorized with `np.arange`.

ii.
```python
# Licking loop (could be vectorized):
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0

# Time arrays (could use np.arange):
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)])
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)])
```

iii. These are minor inefficiencies since trial lengths are small (~20-1000 frames). The main bottleneck is file I/O.

## 12-c. What processing does the code repeat multiple times?

i. The code makes two full passes over all sessions: Pass 1 loads behavioral data to collect running speeds, Pass 2 reloads behavioral data for full processing. Behavioral data files (`Beh_{exp_type}.npy`) are loaded multiple times for different sessions from the same experiment type.

ii.
```python
# Pass 1: collect speeds
for sess_id in session_ids:
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
# Pass 2: full processing
for sess_id in session_ids:
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```
And within `load_beh_for_session`:
```python
beh_all = np.load(beh_path, allow_pickle=True).item()  # loaded for each session
```

iii. The two-pass approach is necessary for computing global speed quartiles before discretization, but the behavioral data loading is repeated unnecessarily.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `n_timepoints=1000` parameter in `extract_trial_data` sets a maximum trial length, but trials are variable-length anyway, so this cap is unnecessary for most trials. (2) The `_swap` session detection code in `load_beh_for_session` searches for swap variants but only uses the first result. (3) The `build_session_to_beh_map` function is defined but never called.

ii.
```python
# Unused function:
def build_session_to_beh_map(exp_info):
    ...  # Never called

# Swap detection produces results that are ignored:
for key in sorted(beh_all.keys()):
    if key.startswith(sess_id + '_swap'):
        results.append((beh_all[key], key))
# But only first is used:
beh, beh_key = beh_results[0]
```

iii. The unused function and swap detection add code complexity without contributing to the output.
