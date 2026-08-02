# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads experiment/session metadata from `data/beh/Imaging_Exp_info.npy`, deduplicates sessions by `mname_datexp_blk`, then processes each session by loading neural data from `data/spk/*_neural_data.npy`, behavioral data from one of the `Beh_*.npy` files, and retinotopy from `data/retinotopy/*_trans.npz`. Trials are then extracted inside `process_session()`.

ii. 
```python
def load_experiment_info():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    session_meta = {}
    for exp_type, sessions in exp_info.items():
        for db in sessions:
            sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
```

```python
def load_spk(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)

def load_beh(session_id, beh_files, beh_keys):
    for beh_file in beh_files:
        beh_all = np.load(beh_path, allow_pickle=True).item()
```

```python
session_meta = load_experiment_info()
all_session_ids = sorted(session_meta.keys())
for i, sid in enumerate(session_ids):
    neural_trials, input_trials, output_trials, region_idx = process_session(
        sid, meta, speed_quartiles, show_processing=args.show_processing
    )
```

iii. In `CONVERSION_NOTES.md`, the AI says this matches the reference loaders `load_spk`, `load_exp_beh`, and `load_retino`, and that handling multiple behavioral key formats is needed for `stimtype` sessions.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `mname`. The AI collects unique mouse names across the selected sessions, sorts them, stores them in `subjects`, and records one `subject_idx` per session.

ii. 
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

```python
session_subject_map.append(subject_to_idx[meta['mname']])
...
'subjects': subjects_list,
'subject_idx': np.array(session_subject_map, dtype=np.int64),
```

iii. The notes justify this by treating mice as the unique `mname` values and checking that the converted dataset contains 19 subjects, matching the paper.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `mname_datexp_blk` combinations from `Imaging_Exp_info.npy`. Duplicate appearances across experiment-type tables are merged into one session entry with aggregated candidate behavioral files/keys.

ii. 
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if sid not in session_meta:
    session_meta[sid] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
        ...
        'beh_files': [],
        'beh_keys': [],
    }
```

```python
all_session_ids = sorted(session_meta.keys())
...
session_ids = all_session_ids
```

iii. The AI’s notes say this resolves the fact that there are 142 metadata entries across experiment types but only 89 unique sessions, matching the paper’s 89 recordings.

## 1-d. How are the data split into trials?

i. Trials are split by the frame-aligned trial index `beh['ft_trInd']`. For each integer trial id from `0` to `ntrials - 1`, the AI takes all matching frame indices as that trial’s time axis.

ii. 
```python
ntrials = beh['ntrials']
ft_trInd = beh['ft_trInd'][:n_frames]
...
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The notes say this was chosen to match the frame-to-trial mapping used in the dataset and reference code.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies very light trial filtering: it skips trials with fewer than 2 aligned frames, and also skips a trial if clipping its frame indices to the neural recording length leaves fewer than 2 frames.

ii. 
```python
if len(frame_indices) < 2:
    skipped += 1
    continue

frame_indices = frame_indices[frame_indices < n_frames_spk]
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. The notes justify this as “include all trials” except trials with `< 2` frames, and mention frame-count mismatch handling by truncating to the shorter stream.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` field in each `*_neural_data.npy` file, concatenated across imaging planes.

ii. 
```python
def load_spk(mname, datexp, blk):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk
```

iii. The notes explicitly say these are deconvolved fluorescence traces from Suite2p and that no new dF/F computation is needed.

## 2-b. How is the `neural` data processed?

i. The AI concatenates imaging planes, filters neurons by retinotopy label, subsamples to at most 2000 neurons per session, and then slices the neural matrix into per-trial frame windows without additional normalization, running-only masking, or position interpolation.

ii. 
```python
spk = load_spk(mname, datexp, blk)
iarea = load_retinotopy(mname, datexp)
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
```

```python
for trial_idx in range(ntrials):
    ...
    trial_neural = spk[:, frame_indices].astype(np.float32)
    neural_trials.append(trial_neural)
```

iii. The notes justify the area filtering as matching the reference code. They justify the 2000-neuron cap as a practical decoder decision, saying it is acceptable because the decoder later uses PCA/SVD.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only neurons with `iarea != -1` and `iarea != 7`, then region-stratified subsamples if more than 2000 remain. It does not apply an explicit neuron-quality filter beyond the retinotopy mask.

ii. 
```python
valid_mask = (iarea != -1) & (iarea != 7)
valid_indices = np.where(valid_mask)[0]
...
if n_valid <= max_neurons:
    selected_indices = valid_indices
else:
    ...
    sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
```

iii. The AI says this copies the reference visual-area exclusion rule and adds a reproducible 2000-neuron cap to keep the dataset manageable.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start/corridor entry by using the frames assigned to each `ft_trInd` trial and setting metadata `temporal_alignment_event` to `Trial start (corridor entry)`.

ii. 
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
    trial_neural = spk[:, frame_indices].astype(np.float32)
```

```python
'metadata': {
    ...
    'temporal_alignment_event': 'Trial start (corridor entry)',
    'off_start': 0.0,
    'off_end': None,
}
```

iii. The notes explicitly state “Trial alignment: Align to corridor entry (trial start). Use `ft_trInd` for frame-to-trial mapping.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses the native calcium-imaging frame rate, `FS = 3.17 Hz`, corresponding to `1000 / 3.17 ≈ 315.5 ms` bins. It does not rebin in time.

ii. 
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

```python
'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'frame_rate_hz': FS,
}
```

iii. The notes say “Time bin: Native frame rate (~315 ms). No resampling.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `beh['SoundFr']` and the current trial’s frame indices.

ii. 
```python
sound_fr = beh['SoundFr']
...
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The notes map this variable as `SoundFr - frame_idx`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI computes a signed continuous time series in seconds by subtracting each frame index from the trial’s sound-cue frame and dividing by `FS`.

ii. 
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
...
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The notes describe this as a direct time-varying transform into seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same `frame_indices` used to slice the neural matrix for each trial, so it is framewise aligned.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
...
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The notes justify this by using the same frame-level trial segmentation for neural and input streams.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata in `Imaging_Exp_info.npy`, using `days` when present and `sess#` otherwise.

ii. 
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
...
'day_of_training': day_val,
```

```python
day_of_training = float(meta['day_of_training'])
```

iii. The notes say the day value comes directly from experiment-info metadata.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI casts the per-session day value to `float` and broadcasts it across all timepoints of every trial in that session.

ii. 
```python
day_of_training = float(meta['day_of_training'])
...
trial_input = np.zeros((4, n_tp), dtype=np.float32)
trial_input[1, :] = day_of_training
```

iii. The notes justify this as a per-trial contextual variable from the decoder task.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not included in the converted `input` at all. The AI does load session metadata fields `exptype` and `rewType`, but it never maps either of them into the final decoder inputs.

ii. 
```python
session_meta[sid] = {
    ...
    'exptype': db.get('exptype', ''),
    'rewType': db.get('rewType', ''),
    ...
}
```

```python
'input_names': ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability'],
```

iii. The notes do not justify a separate environment-type input; the AI appears to have followed the decoder-task list literally and omitted this variable.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The metadata is read and stored transiently in `session_meta`, but no downstream processing creates an environment-type feature.

ii. 
```python
session_meta[sid]['exp_types'].append(exp_type)
...
trial_input = np.zeros((4, n_tp), dtype=np.float32)
```

iii. No explicit justification was recorded beyond following the four requested decoder inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `beh['StartFr']` and each trial’s frame indices.

ii. 
```python
start_fr = beh['StartFr']
...
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The notes map this variable as `frame_idx - StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The AI subtracts the trial start frame from each frame index and divides by the frame rate, yielding seconds since corridor entry.

ii. 
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The notes justify this as the requested continuous time-varying alignment variable.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same per-trial `frame_indices` as `trial_neural`, so it is aligned frame by frame to the neural trace.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The AI’s notes explicitly say trial start is the common alignment event.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavioral variable `beh['isRew']`.

ii. 
```python
is_rew = beh['isRew']
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes map reward availability directly from `isRew`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the Boolean rewarded/unrewarded trial label into `0.0` or `1.0` and broadcasts it across all timepoints in that trial.

ii. 
```python
trial_input = np.zeros((4, n_tp), dtype=np.float32)
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes justify this as a discrete per-trial input matching the decoder specification.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`, `beh['UniqWalls']`, and `beh['stim_id']` when available. The AI first maps wall names to canonical stimulus names, then builds a global sorted stimulus vocabulary.

ii. 
```python
wall_name = beh['WallName']
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
...
wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
```

```python
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
...
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
```

iii. The notes say the intent was to use `stim_id` when possible and fall back to wall names when `stim_id` is missing or `NaN`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI assigns each trial a single stimulus name, constructs a global sorted list of unique stimulus names, converts each name to an integer category id, and repeats that id across all timepoints in the trial.

ii. 
```python
def build_stimulus_mapping(all_output_trials):
    all_stim_names = set()
    ...
    sorted_stim = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(sorted_stim)}
```

```python
trial_output[0, :] = -1  # placeholder
...
stim_idx = stim_to_idx[stim_name]
output_arr[0, :] = stim_idx
```

iii. The notes justify this as a categorical per-trial output derived from the experiment labels.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh['LickFr']` and `beh['LickTrind']`.

ii. 
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
...
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
```

iii. The notes say this was chosen to create the requested time-varying lick output rather than the reference paper’s per-trial lick summaries.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the AI initializes a zero vector over trial frames and sets a frame to `1` if a lick occurred at that frame. Lick frame numbers are rounded to integers before matching.

ii. 
```python
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. The notes justify this as the time-varying binary representation requested by the decoder task.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned by matching `LickFr` values against the exact `frame_indices` used for `trial_neural`, yielding a per-neural-frame binary series.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
...
pos = np.searchsorted(frame_indices, lf_int)
if pos < n_tp and frame_indices[pos] == lf_int:
    lick_binary[pos] = 1
```

iii. The notes say the lick series is frame-aligned to the calcium traces.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `beh['ft_Pos']` at the selected trial frames.

ii. 
```python
ft_Pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_Pos[frame_indices]
```

iii. The notes explicitly map position from `ft_Pos`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI takes the per-frame VR position values for the trial, clips them to `[0, 39.999]`, floors after division by `10`, and stores the result as one of four bins.

ii. 
```python
trial_pos = ft_Pos[frame_indices]
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The notes justify this as “4 bins of 1m (10 VR units each)” and explicitly say gray-space frames are clipped into the final bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `0-10`, `10-20`, `20-30`, and `30-40` VR units, encoded as integer bins `0-3`. Any value above the 4 m corridor is clipped into bin `3`.

ii. 
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
...
pos_bin_names = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The notes explicitly record the gray-space clipping as a design choice.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is sampled at the same `frame_indices` used to build `trial_neural`, so position and neural activity are aligned framewise within each trial.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_pos = ft_Pos[frame_indices]
```

iii. The notes justify this by using the frame-aligned behavioral streams (`ft_*` variables).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['ft_RunSpeed']`.

ii. 
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The notes explicitly map running speed from `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first computes global quartile boundaries from concatenated `ft_RunSpeed` values across all selected sessions, then digitizes each trial’s per-frame speeds against those boundaries.

ii. 
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    all_speeds = []
    for sid in session_ids:
        ...
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
```

```python
trial_speed = ft_RunSpeed[frame_indices]
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The notes justify this as “4 global quartile bins” and acknowledge that all frames, including stationary ones, were included in the quartile computation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the 25th, 50th, and 75th percentiles of the global `ft_RunSpeed` distribution. `np.digitize` maps each frame to bins `0-3`, labeled `Q1-Q4`.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
...
speed_bin_names = ['Q1', 'Q2', 'Q3', 'Q4']
```

iii. The notes explicitly describe the output as “running speed quartile (4 bins).”

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same per-trial `frame_indices` as the neural data, so it is aligned frame by frame.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_speed = ft_RunSpeed[frame_indices]
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
```

iii. The notes justify this by using the frame-aligned `ft_RunSpeed` stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several edge cases defensively: it tries multiple behavioral files/keys, falls back to wall names when `stim_id` is missing or `NaN`, truncates frame-aligned arrays to the shorter of neural and behavioral streams, and skips too-short trials.

ii. 
```python
for beh_file in beh_files:
    ...
    if session_id in beh_all:
        return beh_all[session_id]
    for key in beh_keys:
        if key in beh_all:
            return beh_all[key]
```

```python
n_frames = min(n_frames_spk, n_frames_beh)
...
if not np.isnan(sid_val):
    wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
else:
    wall_to_stim[wall] = wall
```

iii. The notes explicitly list these cases as handled edge cases and say the key-format fallback was added after finding `_swap1` / `_swap2` sessions.

## 12-a. What are the most time-consuming steps of the code?

i. The AI’s code is dominated by loading very large `spk` arrays and scanning all sessions once more to compute global running-speed quartiles. Trial slicing itself is comparatively light.

ii. 
```python
print("Computing global running speed quartiles...")
for sid in session_ids:
    ...
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
    speeds = beh['ft_RunSpeed']
```

```python
print(f"  Loading neural data...")
spk = load_spk(mname, datexp, blk)
print(f"    Raw: {n_neurons_raw} neurons x {n_frames_spk} frames ...")
```

iii. The notes’ runtime table says load-neural-data is the main cost and that full conversion time is driven mostly by per-session I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The explicit per-trial loop in `process_session()`, the per-lick loop used to build `lick_binary`, and the repeated Python loops in neuron subsampling and stimulus remapping could all have been reduced or vectorized further.

ii. 
```python
for trial_idx in range(ntrials):
    ...
```

```python
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
```

```python
for region in BRAIN_REGIONS:
    ...
for session_outputs in all_output_raw:
    for output_arr, stim_name in session_outputs:
```

iii. The AI did not give a strong justification here beyond saying the script was “efficient enough”; the code itself still contains several Python-level loops.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavioral files once to compute global speed quartiles and then again during session processing; it also makes a full second pass over all trial outputs to build stimulus ids after storing placeholder outputs.

ii. 
```python
speed_quartiles = compute_running_speed_quartiles(session_meta, session_ids)
...
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

```python
trial_output[0, :] = -1  # placeholder
...
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
...
for session_outputs in all_output_raw:
    for output_arr, stim_name in session_outputs:
        output_arr[0, :] = stim_idx
```

iii. The notes acknowledge the separate quartile pass and the later global stimulus-mapping pass but do not treat them as problems.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns `selected_idx` from neuron subsampling but never uses it later, stores `ft_CorrSpc` but never applies it, and does summary/plotting work that is not part of the saved dataset.

ii. 
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
...
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
```

```python
print("\nOutput distributions:")
for out_idx, out_name in enumerate(data['output_names']):
    all_vals = np.concatenate([trial[out_idx] for session in all_output for trial in session])
```

```python
if args.show_processing:
    plot_processing(data, session_ids[:2])
```

iii. The AI’s notes focus on validation and visualization, not on minimizing discarded work, so these extra computations appear to be incidental rather than deliberate.
