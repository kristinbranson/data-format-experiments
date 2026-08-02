# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading `Imaging_Exp_info.npy`, which contains 23 experiment types, each with a list of database entries (sessions). It builds a map of unique sessions keyed by `{mname}_{datexp}_{blk}`, deduplicating across experiment types. For each unique session, it loads: (1) neural data via `load_spk()` from `data/spk/`, (2) behavior data from the corresponding `Beh_{exp_type}.npy` file in `data/beh/`, and (3) retinotopy data from `data/retinotopy/`. Sessions are processed one at a time in `process_session()`.

ii.
```python
def get_unique_sessions():
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
        allow_pickle=True
    ).item()
    session_map = {}
    for exp_type in exp_info:
        for ndb in exp_info[exp_type]:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_map:
                session_map[key] = (exp_type, ndb)
    return session_map
```

iii. The AI documented in CONVERSION_NOTES.md that `Imaging_Exp_info.npy` contains 23 experiment types and 142 total entries mapping to 89 unique physical recordings. The deduplication ensures each physical session is used once.

## 1-b. How are the data split into subjects (mice)?

i. The AI identifies subjects from the `mname` field of each session's metadata. It collects all unique mouse names, sorts them alphabetically, and creates a `subject_idx` array mapping each session to a subject.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx.append(subject_to_idx[result['subject']])
```

iii. The AI noted 19 unique subjects in both CONVERSION_NOTES.md and the output logs, matching the paper's statement of "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique combination of `{mname}_{datexp}_{blk}` defines one session. The AI deduplicates across experiment types so that the same physical recording is not counted multiple times. This yields 89 unique sessions.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_map:
    session_map[key] = (exp_type, ndb)
```

iii. The AI noted in CONVERSION_NOTES Step 4 that "Same physical session can appear in multiple experiment types" and decided to "Use each physical session once."

## 1-d. How are the data split into trials?

i. Within each session, trials are defined by the behavior data's `ntrials` count. For each trial, the frame window is `StartFr[trial_idx]` to `GrayFr[trial_idx]`, corresponding to corridor entry through the start of the grey inter-trial interval.

ii.
```python
ntrials = beh['ntrials']
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
for trial_idx in range(ntrials):
    sfr = start_frs[trial_idx]
    gfr = gray_frs[trial_idx]
    # ...
    trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The AI documented that the trial window covers "corridor entry to grey space entry" capturing the full textured corridor portion.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal trial filtering: trials are skipped if (a) `sfr < 0`, (b) `gfr > n_frames`, (c) `sfr >= gfr`, or (d) the trial has fewer than 2 frames. Only 1 trial was skipped across the full dataset. No filtering based on running behavior (ft_move) or other quality criteria is applied.

ii.
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
n_trial_frames = gfr - sfr
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. The AI noted "No trial filtering" in the mapping plan and documented that the reference code doesn't explicitly filter trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` key in the neural data files (`{mname}_{datexp}_{blk}_neural_data.npy`). These are deconvolved fluorescence traces from Suite2p, concatenated across imaging planes.

ii.
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_path = os.path.join(root, fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The AI documented that "spks are deconvolved fluorescence traces from Suite2p (not raw dF/F)" and that the loading matches the reference `load_spk()` function.

## 2-b. How is the `neural` data processed?

i. The neural data is extracted per trial by slicing the full spike matrix from `StartFr` to `GrayFr`. It is cast to float16 for storage efficiency. No other processing is applied — no normalization, no filtering for running frames, no position interpolation.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The AI chose to use raw deconvolved traces without additional processing. The reference code applies different processing depending on the analysis (e.g., position interpolation for coding direction), but for the decoder format, raw frame-rate data was used.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All Suite2p-detected neurons are included. The AI verified neuron counts match the retinotopy data via an assertion.

ii.
```python
assert len(region_idx) == n_neurons, \
    f"Retinotopy ({len(region_idx)}) != neural ({n_neurons}) for {session_key}"
```

iii. The AI documented: "No neuron filtering/curation in the reference code - all Suite2p-detected neurons used."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by extracting frames starting at `StartFr`. The first frame of each trial's neural data corresponds to the moment of corridor entry.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
# where sfr = round(beh['StartFr'][trial_idx])
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)" and the AI implemented this directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native imaging frame rate, approximately 3.17 Hz (~315 ms per frame). No temporal rebinning is applied. The actual frame duration is computed per-session from the behavior timestamps.

ii.
```python
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600  # convert days to seconds
# ...
'time_bin_size': float(median_dt * 1000),  # in ms
```

iii. The AI documented the frame rate as ~3.17 Hz and noted no rebinning was needed since the reference code works at the native frame rate.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `SoundFr` (the frame at which the sound cue occurs in each trial) and the frame indices within the trial window.

ii.
```python
sound_frs = beh['SoundFr']  # keep as float for precise time computation
# ...
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The AI kept `SoundFr` as float for precise time computation and converted the frame difference to seconds using the per-session frame duration.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in the trial, time to sound cue is computed as `(SoundFr - current_frame) * dt_sec`. This produces a continuous, time-varying signal in seconds that is positive before the cue and negative after.

ii.
```python
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Documented in the mapping plan as "positive before cue, negative after."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time_to_sound signal is computed for the same frame indices (sfr to gfr) as the neural data, so they are naturally aligned frame-by-frame.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Same trial window ensures alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field (experiment date string in format `YYYY_MM_DD`) from the session metadata in `Imaging_Exp_info.npy`.

ii.
```python
def compute_day_of_training(session_map):
    mouse_sessions = defaultdict(list)
    for sess_key, (exp_type, ndb) in session_map.items():
        mname = ndb['mname']
        datexp = ndb['datexp']
        date = datetime.strptime(datexp, '%Y_%m_%d')
        mouse_sessions[mname].append((sess_key, date))
```

iii. The AI chose to use the experiment date to compute training day relative to each mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, sessions are sorted by date and the first session's date is used as the reference. Day of training is the calendar day difference (in integer days) from the mouse's first recording. This is a per-trial scalar (same for all timepoints within a trial).

ii.
```python
day_map = {}
for mname, sessions in mouse_sessions.items():
    sessions.sort(key=lambda x: x[1])
    first_date = sessions[0][1]
    for sess_key, date in sessions:
        day_map[sess_key] = (date - first_date).days
```

iii. The AI documented this as "Calendar day difference from mouse's first recording date."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The AI did NOT include "Environment type" as a decoder input. The instructions list exactly 4 decoder inputs: Time to sound cue, Day of training, Time since trial start, and Reward availability. Environment type is not among them.

ii. N/A — no code for this variable.

iii. The AI followed the instructions literally, which do not list Environment type as a decoder input.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Not applicable — Environment type was not included as a decoder input.

ii. N/A

iii. N/A

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from frame indices within the trial window and `StartFr` (the frame of corridor entry).

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Documented in the mapping plan.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, time since trial start is `(current_frame - StartFr) * dt_sec`, producing a continuous, time-varying signal starting at 0 seconds at trial onset and increasing linearly.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Straightforward computation aligned with the trial window definition.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same frame indices as neural data (sfr to gfr), so alignment is inherent. The first timepoint is 0 seconds (at corridor entry).

ii. Same frame range as neural data extraction.

iii. Natural alignment through shared frame indices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `isRew` in the behavior data, which is a per-trial boolean indicating whether the current corridor is rewarded.

ii.
```python
is_rew = beh['isRew']
# ...
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The AI documented this as "1 if rewarded corridor, 0 otherwise."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Simple binary conversion: 1.0 if the trial's `isRew` is true, 0.0 otherwise. This is a per-trial scalar broadcast to all timepoints in the input array.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
# ...
np.full(n_trial_frames, rew_val, dtype=np.float32),
```

iii. Matches the instruction: "Reward availability: 1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `WallName` in the behavior data, which gives the name of the visual texture for each trial (e.g., "circle1", "leaf1", "leaf2").

ii.
```python
wall_names = beh['WallName']
# ...
stim_name = wall_names[trial_idx]
```

iii. The AI used `WallName` directly rather than `stim_id`, noting this in the mapping decisions.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique `WallName` values across all sessions are collected, sorted alphabetically, and mapped to integer indices. Each trial's stimulus is encoded as an integer constant across all timepoints. 15 unique stimulus categories were identified.

ii.
```python
def build_stimulus_mapping(session_results):
    all_stim_names = set()
    for result in session_results:
        for (_, stim_name) in result['output']:
            all_stim_names.add(stim_name)
    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
    return stim_names, stim_to_idx
```

iii. The AI documented 15 stimulus categories and chose alphabetical sorting for consistency.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `LickFr` (lick frame times) in the behavior data.

ii.
```python
lick_frs = beh['LickFr']
# ...
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. The AI used `LickFr` which gives the absolute frame times at which licks occurred.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary time-varying array is created for each trial. `LickFr` values are rounded to the nearest integer frame, and any frame that falls within the trial window (StartFr to GrayFr) is marked as 1, all others as 0.

ii.
```python
def build_lick_binary(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_binary = np.zeros(n_frames, dtype=np.float32)
    lick_frs = beh['LickFr']
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        frame_offsets = lick_frs_int[mask] - start_fr
        frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
        lick_binary[frame_offsets] = 1.0
    return lick_binary
```

iii. The AI chose per-frame binary licking rather than per-trial summary, matching the instruction "Licking, binary, time-varying."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed for the same frame window (StartFr to GrayFr) as neural data, so they are inherently aligned.

ii.
```python
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. Same trial window ensures alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `ft_Pos` in the behavior data, which gives the position of the mouse in decimeters for each frame.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
# ...
trial_pos = ft_pos[sfr:gfr]
```

iii. `ft_Pos` represents position within the corridor in decimeters (0-40 dm for the texture area).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are extracted for the trial frame window and then discretized into 4 equal spatial bins.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)
```

iii. Direct extraction from the per-frame position data.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position (0-40 dm) is divided into 4 equal bins of 10 dm (1 m) each using `np.linspace(0, 40, 5)` to create edges, then `np.digitize` assigns each position to a bin. Values are clipped to [0, 3].

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```

iii. This produces bins [0-10 dm], [10-20 dm], [20-30 dm], [30-40 dm], matching the instruction "4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted for the same trial frame window (StartFr to GrayFr) as neural data.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
```

iii. Same frame range ensures alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `ft_RunSpeed` in the behavior data, which gives the running speed for each frame.

ii.
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
# ...
trial_speed = ft_run_speed[sfr:gfr]
```

iii. `ft_RunSpeed` contains the speed of the mouse at each imaging frame.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed quartiles are first computed across all corridor frames (using `ft_CorrSpc` mask) from all sessions. Then per-trial speeds are discretized using these global quartiles.

ii.
```python
def collect_all_corridor_speeds(session_map, max_sessions=None):
    all_speeds = []
    for sess_key, (exp_type, ndb) in sessions:
        beh = load_behavior_for_session(sess_key, exp_type, ndb)
        n_frames = len(beh['ft'])
        corr_mask = beh['ft_CorrSpc'][:n_frames]
        speeds = beh['ft_RunSpeed'][:n_frames]
        all_speeds.append(speeds[corr_mask])
    all_speeds = np.concatenate(all_speeds)
    return all_speeds
```

iii. Global quartiles ensure consistent binning across all sessions and trials.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three quartile boundaries (25th, 50th, 75th percentiles) are computed from all corridor frame speeds. `np.digitize` with `right=True` assigns each speed to one of 4 bins (Q1-Q4).

ii.
```python
def compute_speed_bin_edges(all_speeds):
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
    return quartiles

def discretize_speed(speed, quartiles):
    binned = np.digitize(speed, quartiles, right=True)
    binned = np.clip(binned, 0, 3)
    return binned
```

iii. The AI initially had a skewed distribution and fixed it by adding `right=True` to `np.digitize`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is extracted for the same trial frame window (StartFr to GrayFr) as neural data.

ii.
```python
trial_speed = ft_run_speed[sfr:gfr]
```

iii. Same frame range ensures alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases: (a) Trials with invalid frame ranges (sfr < 0, gfr > n_frames, sfr >= gfr, or < 2 frames) are skipped. (b) The minimum of neural and behavior frame counts is used when they differ. (c) Fractional frame indices (StartFr, GrayFr) are rounded to nearest integer. (d) NaN/inf values are excluded from speed quartile computation. (e) Lick frame offsets are clipped to valid range. Only 1 trial was skipped in the full dataset.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
# ...
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
# ...
frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
```

iii. The AI documented the single skipped trial: "TX83_2022_08_31_1: invalid frame range."

## 12-a. What are the most time-consuming steps of the code?

i. According to the AI's timing: (a) Processing sessions (loading neural data and extracting trial data) takes ~15s per session, ~22 min total. (b) Saving the 148 GB pickle file takes ~3 min. (c) Speed quartile computation takes ~20s total. Total conversion time was 33.3 minutes.

ii. Timing information is printed throughout the code via `time.time()` calls.

iii. The AI documented timing estimates in CONVERSION_NOTES Step 7.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main trial loop in `process_session()` iterates over trials one at a time, building numpy arrays for each trial individually. Since trials have variable lengths, full vectorization is difficult, but operations like lick binary construction and position extraction could use advanced indexing.

ii.
```python
for trial_idx in range(ntrials):
    # ... per-trial processing
```

iii. The AI noted the trial loop but accepted the per-trial structure since trial lengths vary.

## 12-c. What processing does the code repeat multiple times?

i. (a) The full session map is loaded twice — once in `convert_data()` and once via `all_session_map = get_unique_sessions()` for day computation. (b) Behavior data is loaded twice for each session — once during `collect_all_corridor_speeds()` for quartile computation and once in `process_session()`. (c) `get_unique_sessions()` is called twice.

ii.
```python
session_map = get_unique_sessions()
# ...
all_session_map = get_unique_sessions()  # need full map for day computation
day_map = compute_day_of_training(all_session_map)
# ...
all_speeds = collect_all_corridor_speeds(...)  # loads behavior again
```

iii. The AI did not document this redundancy explicitly.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (a) All neurons are stored (~52K per session average), but the decoder subsamples to 2000 neurons per session. Storing all neurons creates a 148 GB file when only a fraction is needed. (b) The code collects speed quartiles from ALL sessions even in sample mode, though this was later corrected to use only sample sessions. (c) Neural data is stored as float16 which loses precision from the original float32/float64 values, though this may not significantly impact downstream analysis.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The AI documented the OOM issue during decoder training caused by the large data size.
