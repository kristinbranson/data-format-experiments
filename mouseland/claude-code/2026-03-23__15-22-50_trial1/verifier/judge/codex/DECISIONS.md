# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads the master session index from `data/beh/Imaging_Exp_info.npy`, deduplicates recordings into unique `mname_datexp_blk` session keys, then processes each session individually. For each session it reloads the matching behavior file from `data/beh`, loads the session's spike file from `data/spk`, and loads retinotopy from `data/retinotopy`.

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
```
```python
def load_behavior_for_session(session_key, exp_type, ndb):
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()
```
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. In `CONVERSION_NOTES.md`, the agent says the dataset is organized as behavior files, per-session spike files, and per-session retinotopy files, and that there are 89 unique physical recordings across 19 subjects. It explicitly chose to use `Imaging_Exp_info.npy` as the session index and each physical session only once.

## 1-b. How are the data split into subjects?

i. Subjects are identified by the mouse name `mname`, carried through each session result as `subject`, then unique-sorted into `subjects`. `subject_idx` maps each session to that subject list.

ii.
```python
for sess_key, (exp_type, ndb) in session_map.items():
    mname = ndb['mname']
    datexp = ndb['datexp']
    mouse_sessions[mname].append((sess_key, date))
```
```python
result = {
    'subject': mname,
    'session_key': session_key,
```
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
subject_idx.append(subject_to_idx[result['subject']])
```

iii. The notes state that the dataset contains 19 subjects and that `mname` is the mouse identifier from the experiment index, so no extra inference step is needed.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` triple, encoded as `"{mname}_{datexp}_{blk}"`. If a session appears in multiple experiment types, only the first occurrence is kept.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_map:
    session_map[key] = (exp_type, ndb)
```
```python
for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(
        sess_key, exp_type, ndb,
        day_of_training=day_map[sess_key],
        speed_quartiles=speed_quartiles,
        show_processing=show_processing
    )
```

iii. In the notes, the agent says there are 89 total unique physical recordings and that the same physical session can appear in multiple experiment types, so each physical session is used once.

## 1-d. How are the data split into trials?

i. Trials are split using rounded `StartFr` and `GrayFr` frame indices. Each trial is treated as one contiguous frame slice from corridor entry to grey-space entry, with variable length `gfr - sfr`.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
```
```python
for trial_idx in range(ntrials):
    sfr = start_frs[trial_idx]
    gfr = gray_frs[trial_idx]
    n_trial_frames = gfr - sfr
```
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_pos = ft_pos[sfr:gfr]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. In `CONVERSION_NOTES.md` Step 5, the agent explicitly records the key decision: "Trial window: StartFr to GrayFr (corridor entry to grey space entry) - captures full textured corridor portion."

## 1-e. How are trials filtered based on quality controls?

i. Trials are skipped if the start or end frame is invalid relative to the available data, if `sfr >= gfr`, or if the resulting trial has fewer than 2 frames. Otherwise they are kept; there is no additional quality-control filter.

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

iii. The notes describe "All trials included" as a key decision, but also report one skipped trial with an invalid frame range during full conversion. That matches the code's light structural filtering rather than any neuroscience-specific QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from the concatenated `spks` arrays in each session's `*_neural_data.npy` file. Brain-region labels come from `iarea` in the corresponding retinotopy `.npz`.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(root, fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes say the neural data are deconvolved fluorescence traces from Suite2p and that retinotopy supplies the visual-area assignments.

## 2-b. How is the `neural` data processed?

i. The agent does not further transform the traces beyond slicing each trial window from `StartFr` to `GrayFr` and casting to `float16`. It keeps the original frame grid and variable trial lengths.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. In Step 6 of the notes, the agent says neural data are stored as `float16` to reduce file size. In Step 1 it also notes that the source traces are already deconvolved fluorescence, so no extra dF/F or deconvolution step is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The agent does not drop neurons by region or quality. It assigns each neuron to one of five region labels, including `'other'`, and keeps all neurons as long as retinotopy length matches the spike matrix.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']
```
```python
region_idx = np.full(len(iarea), 4, dtype=int)
region_idx[iarea == 8] = 0
region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1
region_idx[(iarea == 5) | (iarea == 6)] = 2
region_idx[(iarea == 3) | (iarea == 4)] = 3
```
```python
assert len(region_idx) == n_neurons
```

iii. The notes justify this by saying "No neuron filtering/curation in the reference code - all Suite2p-detected neurons used" and list "All neurons included: No filtering" as a key decision.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's neural data are aligned to trial start at `StartFr` and run forward until `GrayFr`. The alignment event is corridor entry; no pre-event window is included.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs = np.round(beh['GrayFr']).astype(int)
```
```python
sfr = start_frs[trial_idx]
gfr = gray_frs[trial_idx]
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes say the decoder should be "Temporally aligned based on trial start (corridor entry)" and that the chosen trial window is `StartFr` to `GrayFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The agent keeps the native imaging-frame resolution. It estimates frame duration per session from the median difference of `beh['ft']`, then reports the median across sessions in metadata. No temporal rebinning is applied.

ii.
```python
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
```
```python
median_dt = np.median([r['dt_sec'] for r in session_results])
'time_bin_size': float(median_dt * 1000),
'frame_rate_hz': 1.0 / median_dt,
```

iii. The notes state the frame rate is about 3.17 Hz and the converted time bin is about 314.7 ms, indicating the agent intended to preserve the original framewise resolution.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` for the cue timing and from frame indices within the trial, with seconds per frame estimated from `beh['ft']`.

ii.
```python
sound_frs = beh['SoundFr']
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
```
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. In Step 5 of the notes, the mapping table states `SoundFr - frame_idx` is used and multiplied by `dt_sec` to create continuous time-to-cue.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The agent computes the difference between the sound frame and each frame index in the trial, then multiplies by seconds per frame. Positive values mean before the cue and negative values after the cue.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The notes explicitly justify this as "Time to sound cue (positive before cue, negative after)" and record the same formula in the variable-mapping table.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the exact same trial frame indices `sfr:gfr` used to slice neural activity.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. The notes frame this as a frame-aligned conversion: all time-varying variables are computed on the same trial window as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from each session's mouse name `mname` and date string `datexp` from the experiment index.

ii.
```python
for sess_key, (exp_type, ndb) in session_map.items():
    mname = ndb['mname']
    datexp = ndb['datexp']
    date = datetime.strptime(datexp, '%Y_%m_%d')
    mouse_sessions[mname].append((sess_key, date))
```

iii. The notes say "Day of training: Calendar day difference from mouse's first recording date," so the source fields are the mouse id and session date.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are grouped by mouse, sorted by recording date, and each session gets the calendar-day offset from that mouse's first recording. That scalar is then broadcast across every time bin in the trial.

ii.
```python
for mname, sessions in mouse_sessions.items():
    sessions.sort(key=lambda x: x[1])
    first_date = sessions[0][1]
    for sess_key, date in sessions:
        day_map[sess_key] = (date - first_date).days
```
```python
np.full(n_trial_frames, day_val, dtype=np.float32)
```

iii. The notes explicitly call this out as a key decision: "Day of training: Calendar day difference from mouse's first recording date."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and per-frame trial indices, with frame duration estimated from `beh['ft']`.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600
```
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes map `frame_idx - StartFr` to `time_since_trial_start`, again scaled by `dt_sec`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent rounds `StartFr` to an integer frame, computes each frame's offset from that start frame, and converts the offsets to seconds.

ii.
```python
sfr = start_frs[trial_idx]
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes describe this as the continuous time since corridor entry, derived from frame offset times the frame duration.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same frame range `sfr:gfr` used for the neural slice, so the input vector and neural matrix share the same time axis.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. The notes consistently describe the conversion as frame-aligned to trial start, with all time-varying streams following the same per-trial frame window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew`, the per-trial rewarded-corridor indicator.

ii.
```python
is_rew = beh['isRew']
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes' variable-mapping table lists `isRew` directly as the source for reward availability.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent converts the per-trial boolean into `0.0` or `1.0` and broadcasts it across every frame of the trial.

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
np.full(n_trial_frames, rew_val, dtype=np.float32)
```

iii. The notes treat this as a direct mapping: "1 if rewarded corridor, 0 otherwise."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, the per-trial wall texture name.

ii.
```python
wall_names = beh['WallName']
stim_name = wall_names[trial_idx]
```

iii. The notes state "Stimulus categories: Use WallName directly" and specifically reject other behavior fields as the main source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent collects every unique `WallName` seen across sessions, sorts them, assigns integer category ids, and fills each trial's whole time axis with that trial's stimulus index. It keeps the fine-grained wall names rather than collapsing them to four texture families.

ii.
```python
def build_stimulus_mapping(session_results):
    all_stim_names = set()
    for result in session_results:
        for (_, stim_name) in result['output']:
            all_stim_names.add(stim_name)
    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
```
```python
for i, (out_data, stim_name) in enumerate(result['output']):
    stim_idx = stim_to_idx[stim_name]
    out_data[0, :] = stim_idx
```

iii. In Step 5, the notes record the key decision: "Stimulus categories: Use WallName directly (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, plus rock/brick variants)."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`; `LickTrind` is loaded but not actually used.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
```
```python
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. The notes state that licking is generated as a binary framewise variable using `LickFr`, rounded to the nearest frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame times are rounded to the nearest integer frame. Within each trial window, any frame containing one or more licks is set to 1 and all others to 0.

ii.
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
if mask.any():
    frame_offsets = lick_frs_int[mask] - start_fr
    frame_offsets = np.clip(frame_offsets, 0, n_frames - 1)
    lick_binary[frame_offsets] = 1.0
```

iii. The notes justify this as a binary output: "Licking: Binary 1 at any frame that has >=1 lick (using LickFr rounded to nearest integer frame)."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is computed for the exact frame interval `[sfr, gfr)` used for the trial's neural matrix.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
lick_binary = build_lick_binary(beh, sfr, gfr)
```

iii. The notes repeatedly describe the pipeline as frame-aligned, with licking represented per frame over the same trial window as neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`, the framewise position signal.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
trial_pos = ft_pos[sfr:gfr]
```

iii. The notes map `ft_Pos` directly to the position output and describe the corridor as 0-40 dm for the textured portion.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position samples within each trial are discretized into 4 equal bins spanning 0-40 dm. The agent applies `np.digitize` on equally spaced bin edges and clips the result to 0-3.

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```
```python
pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)
```

iii. The notes justify this with "Position discretization: 4 equal bins of 10dm (0-40dm)."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `[0, 10, 20, 30, 40]` decimeters, yielding categories 0-3 after subtracting 1 from `np.digitize` and clipping to the valid range.

ii.
```python
bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```

iii. The notes specify four 1 m bins across the 4 m corridor, which is exactly the threshold scheme encoded here.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position uses the same trial slice `sfr:gfr` as the neural data, so it is aligned sample-by-sample with the neural frames in each trial.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_pos = ft_pos[sfr:gfr]
pos_binned = discretize_position(trial_pos, POSITION_BINS)
```

iii. The notes present all time-varying variables as extracted on the same corridor-entry-aligned frame window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`. The quartile thresholds are estimated using all corridor-frame speeds gathered from behavior files.

ii.
```python
speeds = beh['ft_RunSpeed'][:n_frames]
all_speeds.append(speeds[corr_mask])
```
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
trial_speed = ft_run_speed[sfr:gfr]
```

iii. The notes say "Running speed quartiles: Computed across ALL corridor frames in the dataset."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent concatenates all finite corridor speeds across sessions, computes the 25th/50th/75th percentiles, then bins each trial's speed samples against those global thresholds.

ii.
```python
def compute_speed_bin_edges(all_speeds):
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
    return quartiles
```
```python
all_speeds = collect_all_corridor_speeds(
    session_map if sample_mode else all_session_map,
    max_sessions=None
)
speed_quartiles = compute_speed_bin_edges(all_speeds)
```
```python
speed_binned = discretize_speed(trial_speed, speed_quartiles).astype(np.float32)
```

iii. The notes justify this choice directly and also mention a later fix: `right=True` was added so values exactly at zero would fall into the lowest quartile bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global percentile cut points returned by `np.percentile(valid, [25, 50, 75])`. `np.digitize(..., right=True)` then assigns bins 0-3, clipped to the 4-category range.

ii.
```python
quartiles = np.percentile(valid, [25, 50, 75])
```
```python
binned = np.digitize(speed, quartiles, right=True)
binned = np.clip(binned, 0, 3)
```

iii. The notes explicitly mention the `right=True` change as a deliberate thresholding fix to improve quartile balance when many values are exactly zero.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sliced over the exact same frame interval `sfr:gfr` as the neural data and then discretized, so alignment is by shared imaging frames.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
trial_speed = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles)
```

iii. The notes describe the dataset as frame-aligned throughout, with speed treated as a time-varying output on the neural frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent trims behavior to `min(n_frames_neural, n_frames_beh)`, ignores licks outside the current trial bounds, filters non-finite speeds before computing quartiles, and skips structurally invalid trials. It also falls back from `session_key` to `session_key_stimtype` when looking up behavior sessions.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
```
```python
if session_key in beh_data:
    return beh_data[session_key]
...
key_with_stim = f"{session_key}_{stimtype}"
if key_with_stim in beh_data:
    return beh_data[key_with_stim]
```
```python
valid = all_speeds[np.isfinite(all_speeds)]
```
```python
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```

iii. The notes mention one skipped trial due to an invalid frame range and describe the behavior lookup fallback for swap sessions. They also describe the dataset as otherwise clean, so the handling is mostly defensive.

## 12-a. What are the most time-consuming steps of the code?

i. The most expensive steps are session-by-session processing, which includes reading the large neural files, and writing the final very large pickle. The code also performs a full-dataset speed pass before the main conversion pass.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
```
```python
all_speeds = collect_all_corridor_speeds(
    session_map if sample_mode else all_session_map,
    max_sessions=None
)
```
```python
with open(output_file, 'wb') as f:
    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In the notes, the agent estimates about 15 seconds per session, about 22 minutes total session processing for 89 sessions, and several minutes to save the full output, which identifies the heavy steps.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The dominant explicit Python loop is the per-trial loop inside `process_session`. Within it, licking is rebuilt trial-by-trial and outputs are assembled one trial at a time. There is also a full per-session loop to gather speeds.

ii.
```python
for trial_idx in range(ntrials):
    sfr = start_frs[trial_idx]
    gfr = gray_frs[trial_idx]
    ...
    lick_binary = build_lick_binary(beh, sfr, gfr)
    ...
    neural_trials.append(trial_neural)
    input_trials.append(trial_input)
    output_trials.append((trial_output, stim_name))
```
```python
for sess_key, (exp_type, ndb) in sessions:
    beh = load_behavior_for_session(sess_key, exp_type, ndb)
    ...
    all_speeds.append(speeds[corr_mask])
```

iii. The notes do not give an explicit vectorization analysis; this conclusion comes primarily from the code structure.

## 12-c. What processing does the code repeat multiple times?

i. The code rebuilds the unique-session map twice, reloads behavior data once for speed-threshold estimation and again during session processing, and makes a second pass over outputs to replace stored `(trial_output, stim_name)` tuples with arrays after building the stimulus mapping.

ii.
```python
session_map = get_unique_sessions()
...
all_session_map = get_unique_sessions()
```
```python
all_speeds = collect_all_corridor_speeds(...)
...
result = process_session(...)
```
```python
for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        stim_idx = stim_to_idx[stim_name]
        out_data[0, :] = stim_idx
        result['output'][i] = out_data
```

iii. The notes do not explicitly justify this repeated processing. The repetition is visible in the script structure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script imports and supports optional plotting machinery that is not needed for the saved dataset, loads `LickTrind` without using it, and temporarily stores stimulus names in `output_trials` only to overwrite them in a later pass.

ii.
```python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
```
```python
lick_trinds = beh['LickTrind']
```
```python
output_trials.append((trial_output, stim_name))
...
for i, (out_data, stim_name) in enumerate(result['output']):
    stim_idx = stim_to_idx[stim_name]
    out_data[0, :] = stim_idx
    result['output'][i] = out_data
```

iii. The notes justify the plotting path as a visual sanity-check option (`--show-processing`), but they do not justify the unused `LickTrind` read or the temporary tuple storage beyond what is apparent in the code.
