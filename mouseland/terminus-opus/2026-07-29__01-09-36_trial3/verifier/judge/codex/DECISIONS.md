# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads `data/beh/Imaging_Exp_info.npy` first and builds a `session_meta` dictionary keyed by `mname_datexp_blk`. For each session, it records a list of candidate behavior files and keys. Later, each session is processed by loading one spike file from `data/spk`, one retinotopy file from `data/retinotopy`, and the first behavior entry found by searching those candidate behavior files/keys.

ii. 
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
session_meta = {}
for exp_type, sessions in exp_info.items():
    for db in sessions:
        sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
```

```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

```python
for beh_file in beh_files:
    beh_all = np.load(beh_path, allow_pickle=True).item()
    if session_id in beh_all:
        return beh_all[session_id]
    for key in beh_keys:
        if key in beh_all:
            return beh_all[key]
```

iii. In `CONVERSION_NOTES.md`, the AI says the master index is `Imaging_Exp_info.npy`, that sessions can appear in multiple behavior files, and that `stimtype` suffixes require trying multiple behavior keys. The trajectory shows it intentionally deduplicated sessions by session id and then loaded spikes and retinotopy per session.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The AI collects unique mouse names from session metadata, sorts them, and maps each session to an integer `subject_idx`.

ii. 
```python
subjects_set = set()
for sid in session_ids:
    subjects_set.add(session_meta[sid]['mname'])
subjects_list = sorted(subjects_set)
subject_to_idx = {name: idx for idx, name in enumerate(subjects_list)}
```

```python
'subjects': subjects_list,
'subject_idx': np.array(session_subject_map, dtype=np.int64),
```

iii. The AI's notes say there are 19 unique mouse names and explicitly describe `mname` as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is defined as the triple `(mname, datexp, blk)` encoded as `sid = "{mname}_{datexp}_{blk}"`. Duplicate appearances of the same recording across experiment types are merged because `session_meta` is keyed by `sid`.

ii. 
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"

if sid not in session_meta:
    session_meta[sid] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
        ...
    }
```

```python
all_session_ids = sorted(session_meta.keys())
print(f"  Found {len(all_session_ids)} unique sessions")
```

iii. In `CONVERSION_NOTES.md`, the AI states there are 142 index entries across experiment types but only 89 unique sessions, so it intentionally deduplicated them at the session-id level.

## 1-d. How are the data split into trials?

i. Trials are split using `ft_trInd`: every frame whose `ft_trInd` equals `trial_idx` is assigned to that trial. The AI includes the full trial as encoded in `ft_trInd`, which its trajectory describes as corridor plus gray-space frames, not just corridor frames.

ii. 
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The trajectory explicitly says the AI chose to "include both corridor and gray space frames (full trial)" and later reasoned that each trial "includes BOTH corridor ... and gray space."

## 1-e. How are trials filtered based on quality controls?

i. The AI only filters out trials with fewer than 2 surviving frames after truncating to the overlap of neural and behavioral data. It does not drop extreme-length trials.

ii. 
```python
frame_indices = np.where(trial_mask)[0]

if len(frame_indices) < 2:
    skipped += 1
    continue

frame_indices = frame_indices[frame_indices < n_frames_spk]
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. `CONVERSION_NOTES.md` says "Trial curation rules: Include all trials. Skip trials with < 2 frames." The notes and trajectory do not mention the 99th-percentile long-trial filter used in the reference solution.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from `spks` in each session's spike file, concatenated across imaging planes. Brain-region labels are derived from `iarea` in the retinotopy file.

ii. 
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

```python
dtrans = np.load(ret_path, allow_pickle=True)
return dtrans['iarea']
```

iii. The AI's notes say the neural data are deconvolved fluorescence traces stored in `spks`, and that retinotopy `iarea` is used for brain-region assignment.

## 2-b. How is the `neural` data processed?

i. After loading and filtering/subsampling neurons, the AI extracts each trial's columns by frame index and stores them as `float32`. It does not perform additional temporal resampling or padding.

ii. 
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
...
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes say the reference data already contain deconvolved traces and that no dF/F or deconvolution is needed. The trajectory shows the switch to `float32` was a practical decoder/memory decision made after experimenting with file size.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy codes: anything with `iarea == -1` or `iarea == 7` is removed. If more than 2000 neurons remain, the AI subsamples to at most 2000 neurons per session, approximately stratified by the four coarse visual regions.

ii. 
```python
valid_mask = (iarea != -1) & (iarea != 7)
valid_indices = np.where(valid_mask)[0]
```

```python
if n_valid <= max_neurons:
    selected_indices = valid_indices
else:
    region_masks = neu_area_ID(iarea[valid_indices])
    ...
    n_sample = max(1, int(np.round(max_neurons * n_region / total_in_regions)))
    sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
```

iii. `CONVERSION_NOTES.md` justifies this as matching the reference region filter and then subsampling for feasibility because the decoder reduces to about 2000 neurons anyway. The trajectory repeatedly cites file size and decoder PCA/SVD limits as the reason for the 2000-neuron cap.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI treats trial start / corridor entry as the alignment event conceptually, but in code it aligns by taking all frames belonging to the trial according to `ft_trInd`. It does not explicitly crop the neural array from `StartFr` onward to corridor-only frames.

ii. 
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
...
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The trajectory says "Use ft_trInd to identify frames belonging to each trial" and "Include both corridor and gray space frames (full trial)" while still describing the overall alignment event as corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native imaging frame rate, 3.17 Hz, so each time bin is about 315.5 ms. No temporal rebinning or resampling is applied.

ii. 
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

```python
'time_bin_size': TIME_BIN_MS,
```

iii. The notes state "Time bin: Native frame rate (~315 ms). No resampling."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives `time_to_sound_cue` from `SoundFr` and the per-frame imaging indices of the current trial. It does not use the behavioral timestamp array `ft`.

ii. 
```python
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The notes map this variable as `SoundFr - frame_idx`, and the trajectory describes it as "compute as (SoundFr - frame_index) / fs in seconds."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the AI subtracts the frame index from the trial's `SoundFr` value and divides by the frame rate, yielding seconds until the sound cue. This preserves fractional `SoundFr` values but assumes constant frame spacing.

ii. 
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
...
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. `CONVERSION_NOTES.md` describes the transform exactly as `(SoundFr - frame) / FS`, and the trajectory presents that as the chosen time-varying input representation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same `frame_indices` array used to extract the neural data for the trial, so it is one value per neural time bin.

ii. 
```python
frame_indices = np.where(trial_mask)[0]
...
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The AI's general alignment rationale in the notes is that behavioral variables are frame-aligned and therefore can be indexed with the same trial frames as the neural matrix.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives `day_of_training` from metadata fields in `Imaging_Exp_info.npy`: it prefers `days`, otherwise falls back to `sess#`.

ii. 
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
...
'day_of_training': day_val,
```

iii. `CONVERSION_NOTES.md` explicitly lists `days/sess#` as the source variable and describes the transform as direct.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. There is effectively no computation beyond reading the metadata value, casting it to float, and broadcasting it across every time bin of the trial.

ii. 
```python
day_of_training = float(meta['day_of_training'])
...
trial_input[1, :] = day_of_training
```

iii. The notes say "day_of_training: Direct value" and the trajectory shows the AI checked how to interpret this field but settled on using it directly rather than reconstructing training day from session order.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives `time_since_trial_start` from `StartFr` and the per-frame trial indices. It does not use `ft`.

ii. 
```python
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The notes map this variable as `(frame_idx - StartFr) / FS`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame in a trial, the AI subtracts `StartFr` from the frame index and divides by 3.17 Hz, producing seconds since the start frame.

ii. 
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
...
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The AI's notes describe this as a direct frame-difference-to-seconds conversion.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is evaluated on exactly the same `frame_indices` used for the trial's neural columns.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The AI consistently treated all time-varying variables as frame-aligned to the neural bins selected for the trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `reward_availability` is taken directly from `isRew`, one value per trial.

ii. 
```python
is_rew = beh['isRew']
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes list `isRew` as the source and describe it as the rewarded-corridor flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI converts the per-trial boolean/numeric flag to float and broadcasts it across all time bins in the trial.

ii. 
```python
trial_input = np.zeros((4, n_tp), dtype=np.float32)
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes describe this variable as direct with no special processing beyond using it as a per-trial broadcast input.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus label from `WallName`, `UniqWalls`, and optionally a session-level `stim_id` mapping if present in the behavioral dict. It then gathers all distinct resulting labels across sessions into a global category list.

ii. 
```python
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
wall_to_stim = {}
```

```python
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
...
for _, stim_name in session_outputs:
    all_stim_names.add(stim_name)
```

iii. The notes say "stim_id -> STIM_NAMES | wall name otherwise" and justify this with the swap-session stimulus naming issue.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI maps wall names to specific stimulus identities such as `circle1`, `leaf2`, or `leaf1_swap1`, not to the four broad texture classes. It initially stores the string label, then in a second pass creates a sorted global mapping from label to integer index and fills that integer into every time bin of the trial.

ii. 
```python
STIM_NAMES = {0: 'circle1', 1: 'circle2', 2: 'leaf1', 3: 'leaf2',
              4: 'leaf3', 5: 'leaf1_swap1', 6: 'leaf1_swap2'}
```

```python
trial_output[0, :] = -1  # placeholder
...
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
...
output_arr[0, :] = stim_idx
```

iii. `README.md` and `CONVERSION_NOTES.md` both describe the shipped output as multi-category stimulus identity rather than the four-category texture grouping used by the reference.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `licking` is derived from `LickFr` and `LickTrind`. The AI uses `LickTrind` to select licks assigned to a given trial and `LickFr` to place them on frame bins.

ii. 
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
...
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
```

iii. The notes identify `LickFr, LickTrind` as the source variables and say licking is converted to a binary per-frame series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI initializes a zero vector for the trial, rounds each lick frame to the nearest integer frame, then uses `searchsorted` to check whether that frame exists in the trial's `frame_indices`; if so, that time bin is marked as 1.

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

iii. The trajectory shows the AI explicitly revisited lick alignment and decided to verify it trial-by-trial. Its notes describe the result as a binary time series aligned to frames.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick vector is defined over the same per-trial `frame_indices` used for the neural matrix, so it has the same number of time bins as the neural data for that trial.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
...
lick_binary = np.zeros(n_tp, dtype=np.int64)
```

iii. The notes repeatedly state that the relevant behavioral variables are frame-aligned, and the code uses the same frame list for both neural extraction and lick binning.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI derives position from `ft_Pos` on the selected trial frames.

ii. 
```python
ft_Pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_Pos[frame_indices]
```

iii. The notes identify `ft_Pos` as the position variable in VR units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI clips positions to `[0, 39.999]`, divides by 10, floors to integers, and then clips to `[0, 3]`. Because its trial windows include gray-space frames, all positions beyond 40 are forced into the last corridor bin.

ii. 
```python
trial_pos = ft_Pos[frame_indices]
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The trajectory shows the AI noticed the gray-space issue but chose to keep the full-trial structure and clip gray-space positions into bin 3.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The category thresholds are fixed 10-unit intervals corresponding to 1 m each: `[0,10)`, `[10,20)`, `[20,30)`, `[30,40)`, with any higher value clipped into category 3.

ii. 
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. `CONVERSION_NOTES.md` states "4 bins of 10 VR units each," and the trajectory explicitly discusses clipping gray-space positions into the last bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same `frame_indices` as the neural data, so it is one value per neural time bin in the trial.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_pos = ft_Pos[frame_indices]
```

iii. The AI's general alignment rule was to index time-varying behavioral variables with the same trial frames as the neural array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` on the selected trial frames, with session-wide bin thresholds computed from all `ft_RunSpeed` values across the selected sessions.

ii. 
```python
speeds = beh['ft_RunSpeed']
all_speeds.append(speeds)
```

```python
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The notes identify `ft_RunSpeed` as the source and say the output is discretized into four global quartile bins.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI concatenates all speeds across the chosen sessions, computes the 25th/50th/75th percentiles with `np.percentile`, and then digitizes each trial's framewise speeds against those global thresholds.

ii. 
```python
all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
```

```python
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. `CONVERSION_NOTES.md` explicitly calls this "4 global quartile bins," and the trajectory says the AI decided to include all frames so each bin would correspond to 25% of the data.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the dataset-wide numeric quartiles returned by `np.percentile`; `np.digitize` assigns categories 0-3 according to those boundaries, and the result is clipped to `[0, 3]`.

ii. 
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The notes and conversion logs report the actual quartile boundaries and describe them as the thresholds used for speed categories.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is indexed with the same `frame_indices` as the neural data, so each neural time bin gets one speed category.

ii. 
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The AI's alignment strategy for all time-varying variables was shared frame indexing with the neural matrix.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavioral streams to the overlap between neural and behavioral recording lengths with `n_frames = min(n_frames_spk, n_frames_beh)`. It skips trials with fewer than 2 surviving frames, tries multiple behavior-file keys to handle naming inconsistencies, and falls back to wall names if stimulus IDs are missing or `NaN`.

ii. 
```python
n_frames_beh = len(beh['ft'])
n_frames = min(n_frames_spk, n_frames_beh)

ft_trInd = beh['ft_trInd'][:n_frames]
ft_Pos = beh['ft_Pos'][:n_frames]
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
```

```python
if len(frame_indices) < 2:
    skipped += 1
    continue
```

```python
if not np.isnan(sid_val):
    wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
else:
    wall_to_stim[wall] = wall
```

iii. The notes list these cases explicitly under edge-case handling: behavior/spike frame mismatches, `stimtype` suffixes, `NaN stim_id`, and short trials.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is loading and concatenating the very large per-session spike files. The full conversion log shows most session time is spent before any trial-wise processing begins.

ii. 
```python
spk = load_spk(mname, datexp, blk)
n_neurons_raw, n_frames_spk = spk.shape
print(f"    Raw: {n_neurons_raw} neurons x {n_frames_spk} frames ({time.time()-t_load:.1f}s)")
```

iii. `CONVERSION_NOTES.md` estimates neural-data loading at 1-30 seconds per session and attributes most runtime to loading those files.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The code loops over trials, then loops over licks within each trial, and separately loops over all sessions to compute global speed quartiles. Those operations could have been more vectorized or grouped.

ii. 
```python
for trial_idx in range(ntrials):
    ...
    for lf in trial_lick_frames:
        ...
```

```python
for sid in session_ids:
    meta = session_meta[sid]
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)
```

iii. The AI's notes focus on I/O, but the final code still contains several Python loops over large structures that the reference avoided or handled more economically.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly loads entire behavior files. It loads behavior once during `compute_running_speed_quartiles` and again during `process_session`, and `load_beh` reads the full `.npy` file anew on each call instead of grouping sessions by behavior file.

ii. 
```python
def compute_running_speed_quartiles(session_meta, session_ids):
    for sid in session_ids:
        meta = session_meta[sid]
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

```python
def process_session(sid, meta, speed_quartiles, show_processing=False):
    ...
    beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. The trajectory shows the AI knew behavior files contained multiple sessions, but the final implementation still performs repeated per-session loads rather than the grouped access pattern used in the reference.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and returns `selected_idx` from neuron subsampling but never uses it downstream, loads `ft_CorrSpc` but never uses it in trial selection, and optionally builds large diagnostic plots that are not part of the converted dataset. It also does a second pass to build stimulus mappings after first storing placeholder outputs.

ii. 
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
...
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
```

```python
trial_output[0, :] = -1  # placeholder
...
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
...
output_arr[0, :] = stim_idx
```

```python
if args.show_processing:
    plot_processing(data, session_ids[:2])
```

iii. The notes describe the plotting and summary checks as validation utilities. The unused `selected_idx` and unused `ft_CorrSpc` are visible directly in the final code.
