# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data by first reading `Imaging_Exp_info.npy` to get all experiment metadata, then builds a map of unique sessions (keyed by `mname_datexp_blk`). For each session, it loads neural data from `spk/` directory via `load_spk()`, behavior data from `beh/Beh_{exp_type}.npy`, and retinotopy from `retinotopy/` directory. The same physical session can appear across multiple experiment types in `Imaging_Exp_info.npy`; the AI deduplicates by keeping only the first occurrence of each `mname_datexp_blk` key.

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

iii. The agent noted that `Imaging_Exp_info.npy` contains 23 experiment types with 142 total entries but only 89 unique physical recordings. Each physical session is used once to avoid duplication. This approach iterates over all experiment types and takes the first occurrence.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the `mname` field of each session's metadata (from `Imaging_Exp_info.npy`). The AI collects all unique mouse names across sessions, sorts them, and creates a `subject_idx` array mapping each session to its subject.

ii.
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
# ...
subject_idx.append(subject_to_idx[result['subject']])
```

iii. The agent identified 19 unique subjects from the data, consistent with the paper's statement "89 recordings in 19 mice."

## 1-c. How are the data split into sessions?

i. Each unique `mname_datexp_blk` combination constitutes one session. The AI processes 89 unique sessions. Each session has its own neural data file, behavior data entry, and retinotopy file.

ii.
```python
for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(sess_key, exp_type, ndb, ...)
    session_results.append(result)
```

iii. The agent confirmed 89 sessions matching the paper's count.

## 1-d. How are the data split into trials?

i. Within each session, trials are defined by the behavior data's `ntrials` field, with each trial spanning from `StartFr` (corridor entry frame) to `GrayFr` (grey space entry frame). The AI iterates over trial indices 0 to `ntrials-1` and extracts per-trial data slices.

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

iii. The agent chose `StartFr` to `GrayFr` as the trial window, capturing the textured corridor portion (0-4m). This aligns with the reference code's use of corridor frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies minimal filtering: trials are skipped only if `sfr < 0`, `gfr > n_frames`, `sfr >= gfr`, or `n_trial_frames < 2`. Only 1 trial was skipped across the entire dataset (TX83_2022_08_31_1 had an invalid frame range).

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

iii. The agent noted that the reference code does not filter trials based on quality, so minimal filtering was applied. This is consistent with the reference approach.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from the `spks` field within the `{mname}_{datexp}_{blk}_neural_data.npy` files. These contain deconvolved fluorescence traces from Suite2p, stored as a list of arrays (one per imaging plane).

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

iii. The agent confirmed these are deconvolved fluorescence traces, consistent with the paper: "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. Neural data is loaded by concatenating across imaging planes (typically 3), then sliced per trial from `StartFr` to `GrayFr`. The data is cast to float16 for memory efficiency. No additional processing (normalization, smoothing, z-scoring) is applied.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
# ...
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The agent noted that the reference code uses raw deconvolved traces without additional processing for dprime analyses, so no extra processing was applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron filtering is applied. All Suite2p-detected neurons are included, consistent with the reference code. The neuron count per session ranges from 20,547 to 89,577, matching the paper.

ii. No filtering code exists; all neurons from `load_spk()` are used directly.

iii. The agent confirmed: "No additional neuron filtering beyond Suite2p. All detected neurons included."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to trial start (corridor entry) by slicing from `StartFr` (the frame when the animal enters the corridor). Each trial's neural data starts at frame index `StartFr` and ends at `GrayFr`.

ii.
```python
sfr = start_frs[trial_idx]
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. The instructions specify "Temporally aligned based on trial start (corridor entry)." The agent sets `off_start: 0.0` in metadata, confirming alignment at corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native calcium imaging frame rate (~3.17 Hz, ~315 ms per frame). No temporal rebinning is applied. The frame duration is computed per-session from the `ft` timestamps.

ii.
```python
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600  # convert days to seconds
# ...
median_dt = np.median([r['dt_sec'] for r in session_results])
# metadata: 'time_bin_size': float(median_dt * 1000),  # in ms
```

iii. The agent reported a median time bin of 314.7 ms. No rebinning is applied; the native frame rate is used directly.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from `beh['SoundFr']` (the frame index when the sound cue was delivered) and the current frame index within the trial.

ii.
```python
sound_frs = beh['SoundFr']  # keep as float for precise time computation
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The agent uses SoundFr (per-trial) minus the current frame index, multiplied by the frame duration to convert to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in the trial, the time to sound cue is computed as `(SoundFr[trial] - frame_index) * dt_sec`. This produces a decreasing time series: positive values before the cue, negative after. The result is in seconds.

ii.
```python
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The agent chose to represent this as a continuous, time-varying signal with positive values before the cue and negative after, consistent with the instructions specifying "Time to sound cue, continuous, time-varying."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time to sound cue is computed at the same frame indices as the neural data (from `StartFr` to `GrayFr`), so it is inherently aligned frame-by-frame.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. Both neural and input data use the same frame indices, ensuring alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from the `datexp` field (date string in `YYYY_MM_DD` format) of each session's metadata entry in `Imaging_Exp_info.npy`.

ii.
```python
def compute_day_of_training(session_map):
    mouse_sessions = defaultdict(list)
    for sess_key, (exp_type, ndb) in session_map.items():
        mname = ndb['mname']
        datexp = ndb['datexp']
        date = datetime.strptime(datexp, '%Y_%m_%d')
        mouse_sessions[mname].append((sess_key, date))
    day_map = {}
    for mname, sessions in mouse_sessions.items():
        sessions.sort(key=lambda x: x[1])
        first_date = sessions[0][1]
        for sess_key, date in sessions:
            day_map[sess_key] = (date - first_date).days
    return day_map
```

iii. The agent computes day of training as the calendar day difference from each mouse's first recording date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, all session dates are sorted chronologically. The day of training is computed as the number of days since the mouse's first recording. This value is constant per trial (scalar broadcast to all timepoints).

ii.
```python
day_val = np.float32(day_of_training)
# In input stack:
np.full(n_trial_frames, day_val, dtype=np.float32),
```

iii. Day of training ranges from 0 to 92 days across the dataset.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from the frame indices within each trial relative to `StartFr`, multiplied by the frame duration `dt_sec`.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Uses the same frame indices and timing as the neural data.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame in the trial, the time since trial start is `(frame_index - StartFr) * dt_sec`. This starts at 0 and increases linearly. The result is in seconds.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. A simple linear ramp from 0, consistent with the instructions specifying "Time since trial start, continuous, time varying."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Aligned frame-by-frame using the same frame indices as the neural data.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Same alignment mechanism as all other time-varying signals.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Derived from `beh['isRew']`, a boolean array indicating whether each trial is a rewarded trial.

ii.
```python
is_rew = beh['isRew']
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The agent uses the per-trial `isRew` flag directly.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Converted to a binary value: 1.0 if the trial is rewarded, 0.0 otherwise. This is broadcast as a constant across all timepoints in the trial.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
np.full(n_trial_frames, rew_val, dtype=np.float32),
```

iii. Consistent with instructions: "Reward availability: 1 if in rewarded corridor, 0 if not, discrete, per-trial."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Derived from `beh['WallName']`, which contains the stimulus name for each trial (e.g., 'circle1', 'leaf1', 'rock1', etc.).

ii.
```python
wall_names = beh['WallName']
stim_name = wall_names[trial_idx]
```

iii. The agent uses `WallName` rather than `stim_id` to preserve the full stimulus identity including swap variants.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. All unique stimulus names across all sessions are collected, sorted alphabetically, and mapped to integer indices 0-14 (15 categories total). The stimulus index is broadcast as a constant across all timepoints in the trial.

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
# ...
out_data[0, :] = stim_idx  # constant per trial
```

iii. 15 stimulus categories were identified: circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Derived from `beh['LickFr']` (frame indices when licks occurred) and `beh['LickTrind']` (trial index for each lick). Note: while `LickTrind` is loaded, it is not actually used in the filtering logic.

ii.
```python
def build_lick_binary(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_binary = np.zeros(n_frames, dtype=np.float32)
    lick_frs = beh['LickFr']
    lick_trinds = beh['LickTrind']
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        frame_offsets = lick_frs_int[mask] - start_fr
        frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
        lick_binary[frame_offsets] = 1.0
    return lick_binary
```

iii. The agent constructs a binary lick signal by checking which `LickFr` values fall within the trial's frame range (`StartFr` to `GrayFr`).

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary array of length `n_trial_frames` is created. For each lick frame (`LickFr`) that falls within the trial's frame range, the corresponding position in the binary array is set to 1. Lick frames are rounded to the nearest integer.

ii.
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
if mask.any():
    frame_offsets = lick_frs_int[mask] - start_fr
    frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
    lick_binary[frame_offsets] = 1.0
```

iii. Consistent with instructions: "Licking, binary, time-varying. 0 = not licking, 1 = licking." Overall 3.5% of frames have licking, reflecting that many sessions are from unsupervised (non-water-restricted) mice.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by using the same frame indices as the neural data. `LickFr` values are compared against `StartFr:GrayFr` range, and offsets are computed relative to `StartFr`.

ii.
```python
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. Same frame-based alignment as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Derived from `beh['ft_Pos']`, which contains the position inside the corridor for each neural frame, in decimeters (0-40 dm for the textured corridor).

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[sfr:gfr]
```

iii. `ft_Pos` is the per-frame position variable documented in the reference data_process_script.ipynb.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position values are extracted for the trial's frame range and discretized into 4 equal bins.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)
```

iii. Straightforward extraction and discretization.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position (0-40 dm) is discretized into 4 equal bins of 10 dm each (1 meter) using `np.digitize` with bin edges at [0, 10, 20, 30, 40]. Values are clipped to range [0, 3].

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)  # [0, 10, 20, 30, 40]
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```

iii. Consistent with instructions: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted from `ft_Pos` at the same frame indices as the neural data (`sfr:gfr`), ensuring frame-by-frame alignment.

ii.
```python
trial_pos = ft_pos[sfr:gfr]
```

iii. Same alignment approach as all time-varying signals.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Derived from `beh['ft_RunSpeed']`, which contains the running speed for each neural frame.

ii.
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. `ft_RunSpeed` is documented in data_process_script.ipynb as "running speed for each neural frame."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed quartiles are pre-computed across all corridor frames from all sessions (using `ft_CorrSpc` mask to select corridor frames). Then per-trial speed values are discretized into 4 bins using these quartiles.

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

iii. The agent correctly uses only corridor frames for quartile computation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Speed is discretized into 4 quartile bins using `np.digitize` with 3 quartile edges (25th, 50th, 75th percentiles). The `right=True` parameter is used so values exactly at 0 go into Q1.

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

iii. Consistent with instructions: "Running speed discretized into 4 bins, each corresponding to 25% of the data." The actual distribution is 30.2%, 19.8%, 24.9%, 25.1% -- not perfectly 25% each, likely due to the `right=True` flag behavior and the fact that many speeds are exactly 0 (when the mouse is stationary).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is extracted from `ft_RunSpeed` at the same frame indices as the neural data (`sfr:gfr`).

ii.
```python
trial_speed = ft_run_speed[sfr:gfr]
```

iii. Same frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases:
- Frame count mismatch between neural and behavior data: uses `min(n_frames_neural, n_frames_beh)`
- Invalid trial frame ranges (`sfr < 0`, `gfr > n_frames`, `sfr >= gfr`): trial is skipped
- Very short trials (`n_trial_frames < 2`): trial is skipped
- Empty lick arrays: returns all-zeros binary array
- Non-finite speed values: filtered out when computing quartiles
- Fractional frame indices: rounded to nearest integer

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
# ...
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
if n_trial_frames < 2:
    skipped += 1
    continue
# ...
if len(lick_frs) == 0:
    return lick_binary
```

iii. Only 1 trial was skipped across 38,110 total trials. The agent documented this in CONVERSION_NOTES.md.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. Processing each session (~10-44 seconds per session, dominated by loading and slicing neural data arrays of 20K-90K neurons)
2. Saving the final pickle file (113 seconds for 148 GB)
3. Collecting all corridor speeds for quartile computation (8.5 seconds)
Total conversion time: 33.3 minutes for 89 sessions.

ii.
```python
# Per-session timing:
print(f"  {session_key}: {len(neural_trials)} trials, {n_neurons} neurons, {t1-t0:.1f}s")
# Total:
print(f"Total conversion time: {total_time:.1f}s ({total_time/60:.1f} min)")
```

iii. The agent documented timing in CONVERSION_NOTES.md and estimated ~25-30 min, which was close to actual (33.3 min).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main per-trial loop in `process_session()` iterates over each trial sequentially. While the neural slicing is already vectorized (array slicing), the construction of input/output arrays involves per-trial Python loops that could potentially be partially vectorized. The `build_lick_binary()` function is called per-trial but uses vectorized numpy operations internally.

ii.
```python
for trial_idx in range(ntrials):
    # ... per-trial processing
    trial_neural = spk[:, sfr:gfr].astype(np.float16)
    # ... construct inputs and outputs
```

iii. The agent noted the main bottleneck is I/O (loading large neural arrays) rather than computation, so vectorization of the trial loop would yield limited speedup.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavior data twice for each session in the full conversion mode:
1. Once in `collect_all_corridor_speeds()` to compute speed quartiles
2. Once in `process_session()` for actual data extraction

Additionally, `get_unique_sessions()` is called twice (once for the main session map, once for computing day of training across all sessions).

ii.
```python
# In convert_data():
all_session_map = get_unique_sessions()  # called a second time
day_map = compute_day_of_training(all_session_map)
# Speed collection also loads behavior:
all_speeds = collect_all_corridor_speeds(session_map if sample_mode else all_session_map)
```

iii. The repeated behavior file loading doubles the I/O time for behavior files, though this is minor compared to neural data loading.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code stores neural data as float16, which reduces precision from the original float32/float64. However, the decoder training script converts everything back to float32 tensors, so the float16 storage introduces unnecessary precision loss without benefiting downstream analysis.

The code also computes and stores `dt_sec` per session but then uses a global `median_dt` for the metadata time_bin_size, discarding per-session timing information.

Speed quartiles are computed across all sessions even in sample mode (if `--full` is default), though the code handles this correctly by using the session_map for sample mode.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)  # precision loss
# ...
median_dt = np.median([r['dt_sec'] for r in session_results])  # per-session dt discarded
```

iii. The float16 casting was done for memory efficiency (148 GB output vs ~296 GB at float32), which was necessary to avoid OOM. The decoder still achieved good accuracy despite the precision loss.
