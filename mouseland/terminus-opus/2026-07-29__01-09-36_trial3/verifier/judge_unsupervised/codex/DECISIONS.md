# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a session table from `data/beh/Imaging_Exp_info.npy`, then for each unique session ID loads neural data from `data/spk/<mname>_<datexp>_<blk>_neural_data.npy`, behavioral data from one of the `Beh_*.npy` files, and retinotopy from `data/retinotopy/<mname>_<datexp>_trans.npz`. Trials are then extracted by iterating `trial_idx` from `0` to `ntrials - 1`.

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
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk_data = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    return spk
```

```python
def load_beh(session_id, beh_files, beh_keys):
    for beh_file in beh_files:
        beh_path = os.path.join(DATA_ROOT, 'beh', beh_file)
        beh_all = np.load(beh_path, allow_pickle=True).item()
        if session_id in beh_all:
            return beh_all[session_id]
        for key in beh_keys:
            if key in beh_all:
                return beh_all[key]
```

iii. In `CONVERSION_NOTES.md`, the agent says this matches the reference loaders in `utils.py`: `load_spk`, `load_exp_beh`, and `load_retino`. The trajectory also shows the agent copied those loader patterns from the original `code/utils.py`.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. The script collects unique `mname` values across selected sessions, sorts them, and stores a per-session index into that subject list.

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
```

iii. The agent’s notes identify “subjects = 19 unique mouse names” and treat `mname` as the subject identifier, consistent with the experiment metadata.

## 1-c. How are the data split into sessions?

i. Sessions are split by the composite key `mname_datexp_blk`. The agent deduplicates sessions across experiment types by storing them in `session_meta` under that key.

ii.
```python
sid = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if sid not in session_meta:
    session_meta[sid] = {
        'mname': db['mname'],
        'datexp': db['datexp'],
        'blk': db['blk'],
```

```python
all_session_ids = sorted(session_meta.keys())
```

iii. The notes justify this as necessary because the same physical session can appear under multiple experiment-type entries in `Imaging_Exp_info.npy`; the goal was to recover the 89 unique recordings reported in the paper.

## 1-d. How are the data split into trials?

i. Trials are split using the framewise trial index `ft_trInd`. For each `trial_idx` in `range(ntrials)`, the code selects all behavioral/neural frames whose `ft_trInd` equals that trial number.

ii.
```python
for trial_idx in range(ntrials):
    trial_mask = (ft_trInd == trial_idx)
    frame_indices = np.where(trial_mask)[0]
```

iii. The agent states in `CONVERSION_NOTES.md` that trial alignment is based on corridor entry and uses `ft_trInd` as the frame-to-trial mapping, which it considered the closest match to the reference code.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no explicit trial-level QC. The script keeps all trials except ones with fewer than 2 mapped frames, or trials whose mapped frames all fall beyond the neural recording length after truncation to `n_frames_spk`.

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

iii. The notes say “Include all trials. Skip trials with < 2 frames.” The agent treated this as a pragmatic minimal QC rule rather than a biological curation rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` field in each `*_neural_data.npy` file, concatenated across imaging planes.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate(spk_data['spks'], axis=0)
```

iii. The notes explicitly say the source is the deconvolved fluorescence output stored in `spks`, not raw fluorescence or dF/F.

## 2-b. How is the `neural` data processed?

i. The script concatenates `spks`, filters neurons by retinotopy label, randomly subsamples to at most 2000 neurons per session with region-stratified sampling, and then slices the selected neural matrix by each trial’s frame indices. It does not compute dF/F, deconvolution, running-only filtering, or position interpolation itself.

ii.
```python
spk = load_spk(mname, datexp, blk)
iarea = load_retinotopy(mname, datexp)
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
```

```python
trial_neural = spk[:, frame_indices].astype(np.float32)
```

```python
if n_valid <= max_neurons:
    selected_indices = valid_indices
else:
    ...
    selected_indices = valid_indices[selected_local]
```

iii. The justification in `CONVERSION_NOTES.md` is that the reference paper used deconvolved traces and that subsampling to 2000 neurons was acceptable because the downstream decoder uses PCA/SVD anyway. That subsampling decision is the agent’s own addition; it is not described as coming from the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered by retinotopy label only: keep neurons with `iarea != -1` and `iarea != 7`, which the agent interprets as excluding cells outside visual cortex. No additional cell-quality metrics are used in the conversion script.

ii.
```python
valid_mask = (iarea != -1) & (iarea != 7)
valid_indices = np.where(valid_mask)[0]
```

iii. The agent repeatedly cites the original `utils.py` pattern `(arid!=-1) & (arid != 7)` and treats this as the main neural QC rule from the reference code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural matrix is aligned to trial start/corridor entry by taking the contiguous set of frames assigned to that trial by `ft_trInd`. There is no additional temporal shifting.

ii.
```python
trial_mask = (ft_trInd == trial_idx)
frame_indices = np.where(trial_mask)[0]
trial_neural = spk[:, frame_indices].astype(np.float32)
```

iii. The notes say “Trial alignment: Align to corridor entry (trial start). Use `ft_trInd` for frame-to-trial mapping,” matching the decoder instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses the native imaging frame rate `FS = 3.17 Hz`, i.e. `1000 / 3.17 = 315.46 ms` bins. No temporal rebinning is applied.

ii.
```python
FS = 3.17  # Frame rate in Hz
TIME_BIN_MS = 1000.0 / FS  # ~315.5 ms
```

iii. The notes justify this as matching the paper’s imaging sampling rate and state explicitly: “Time bin: Native frame rate (~315 ms). No resampling.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from behavioral frame indices and the per-trial sound-cue frame `SoundFr`.

ii.
```python
sound_fr = beh['SoundFr']
...
trial_sound_fr = sound_fr[trial_idx]
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The notes map this input as `SoundFr - frame_idx`, in seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every frame in a trial, the script computes signed time until cue onset in seconds as `(SoundFr[trial] - frame_index) / FS`. Values are negative after the cue.

ii.
```python
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
trial_input[0, :] = time_to_sound.astype(np.float32)
```

iii. The agent describes this as a direct continuous time-varying covariate and does not mention additional smoothing, clipping, or binary encoding.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the exact same `frame_indices` used to slice `trial_neural`, so it is frame-aligned to the neural trace within each trial.

ii.
```python
n_tp = len(frame_indices)
trial_neural = spk[:, frame_indices].astype(np.float32)
time_to_sound = (trial_sound_fr - frame_indices.astype(np.float64)) / FS
```

iii. The agent’s justification is implicit: all trial-level inputs and outputs are built off the same `frame_indices` vector.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata in `Imaging_Exp_info.npy`, using either the `days` field or the `sess#` field depending on which exists.

ii.
```python
day_key = 'days' if 'days' in db else 'sess#'
day_val = db.get(day_key, 0)
...
'day_of_training': day_val,
```

iii. The notes describe this as `days/sess# -> day_of_training` and treat the experiment-info file as the authoritative source.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The script takes the chosen metadata field as-is, converts it to `float`, and broadcasts the same value across all frames of every trial in that session.

ii.
```python
day_of_training = float(meta['day_of_training'])
...
trial_input[1, :] = day_of_training
```

iii. The agent justifies this as a direct per-trial/session covariate. There is no normalization or harmonization between `days` and `sess#`; the notes simply say “Direct value.”

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the per-trial start frame `StartFr` and the trial’s frame indices.

ii.
```python
start_fr = beh['StartFr']
...
trial_start = start_fr[trial_idx]
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The notes map this as `frame_idx - StartFr`, in seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame in a trial, the script computes elapsed time since corridor entry as `(frame_index - StartFr[trial]) / FS`, then writes that continuous trace into the input matrix.

ii.
```python
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
trial_input[2, :] = time_since_start.astype(np.float32)
```

iii. The agent’s notes say this is a direct continuous time-varying input with no further processing.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by construction because it uses the same `frame_indices` vector that indexes the neural matrix for that trial.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
time_since_start = (frame_indices.astype(np.float64) - trial_start) / FS
```

iii. The justification is the same shared-frame construction used for all time-varying trial variables.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the behavioral per-trial boolean `isRew`.

ii.
```python
is_rew = beh['isRew']
...
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The notes describe this as `isRew -> reward_availability`, broadcast over the trial.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts `isRew[trial_idx]` to a float `0.0/1.0` and fills the full time axis of that trial with the same value.

ii.
```python
trial_input[3, :] = float(is_rew[trial_idx])
```

iii. The agent’s justification is that reward availability is a per-trial discrete variable in the decoder specification.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial wall identity `WallName`, together with the session’s `stim_id` map and `UniqWalls` to translate wall labels into canonical stimulus names when possible.

ii.
```python
uniq_walls = beh['UniqWalls']
stim_id_map = beh.get('stim_id', None)
wall_to_stim = {}
```

```python
stim_name = wall_to_stim.get(wall_name[trial_idx], wall_name[trial_idx])
```

iii. The notes say the agent initially had a swap-stimulus naming bug and fixed it so IDs 5 and 6 map to `leaf1_swap1` and `leaf1_swap2`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script first builds a session-local map from wall labels to canonical stimulus names using `stim_id` when available, then after all sessions are processed it builds one global sorted stimulus vocabulary and fills output channel 0 with the corresponding integer ID for every frame of a trial.

ii.
```python
if stim_id_map is not None:
    for i, wall in enumerate(uniq_walls):
        if i < len(stim_id_map):
            sid_val = stim_id_map[i]
            if not np.isnan(sid_val):
                wall_to_stim[wall] = STIM_NAMES.get(int(sid_val), wall)
```

```python
stim_to_idx, stim_names_sorted = build_stimulus_mapping(all_output_raw)
...
for output_arr, stim_name in session_outputs:
    stim_idx = stim_to_idx[stim_name]
    output_arr[0, :] = stim_idx
```

iii. The justification in the notes is to preserve canonical stimulus identity across sessions while falling back to wall names for labels that are present in the data but absent from `STIM_NAMES`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from eventwise lick-frame and lick-trial arrays: `LickFr` and `LickTrind`.

ii.
```python
lick_fr = beh['LickFr']
lick_trind = beh['LickTrind']
```

iii. The notes explicitly map `LickFr, LickTrind -> licking`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the script finds all lick events with `LickTrind == trial_idx`, rounds each lick frame to the nearest integer, and sets the corresponding frame in a binary per-frame vector to `1` if that frame exists in the trial.

ii.
```python
trial_lick_mask = (lick_trind == trial_idx)
trial_lick_frames = lick_fr[trial_lick_mask]
lick_binary = np.zeros(n_tp, dtype=np.int64)
for lf in trial_lick_frames:
    lf_int = int(np.round(lf))
    pos = np.searchsorted(frame_indices, lf_int)
    if pos < n_tp and frame_indices[pos] == lf_int:
        lick_binary[pos] = 1
    elif pos > 0 and frame_indices[pos-1] == lf_int:
        lick_binary[pos-1] = 1
```

iii. The agent’s notes justify this as the way to turn sparse lick events into the requested time-varying binary output. It also notes that multiple licks can collapse onto one imaging frame at 3.17 Hz.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licks are aligned by matching rounded lick frame numbers against the same per-trial `frame_indices` used to index the neural trace.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
...
pos = np.searchsorted(frame_indices, lf_int)
if pos < n_tp and frame_indices[pos] == lf_int:
    lick_binary[pos] = 1
```

iii. The justification is implicit in the implementation: lick events are projected directly onto the neural frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise position `ft_Pos`.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_Pos[frame_indices]
```

iii. The notes map `ft_Pos -> position_bin`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script takes framewise position values for a trial, clips them to `[0, 39.999]`, converts them to 10-unit bins with `floor(pos / 10)`, and keeps four categories `0..3`.

ii.
```python
trial_pos = ft_Pos[frame_indices]
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The notes justify this as “4 bins of 1m (10 VR units each)” and explicitly acknowledge that gray-space frames are clipped into the last bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are fixed at 0, 10, 20, 30, and 40 VR units, corresponding to four 1 m bins. Any position above 40 VR units is forced into category 3 because of clipping.

ii.
```python
pos_clipped = np.clip(trial_pos, 0, 39.999)
pos_bin = np.floor(pos_clipped / 10.0).astype(np.int64)
pos_bin = np.clip(pos_bin, 0, 3)
```

iii. The agent’s own final review calls this a “known trade-off” that makes bin 3 overrepresented because gray-space frames are clipped into it.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by indexing `ft_Pos` with the same `frame_indices` used for the neural data.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_pos = ft_Pos[frame_indices]
```

iii. The justification is the same shared-frame alignment scheme used throughout the script.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise running speed `ft_RunSpeed`.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The notes map `ft_RunSpeed -> running_speed_bin`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before session processing, the script pools all `ft_RunSpeed` values across the selected sessions and computes three global quartile boundaries. For each trial, it digitizes the per-frame speed values into those bins.

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

iii. The notes justify this as “4 global quartile bins,” but the final review admits the resulting distribution is uneven because of the chosen implementation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Category thresholds are the global 25th, 50th, and 75th percentiles of pooled `ft_RunSpeed` over the selected sessions. `np.digitize` maps each frame to bins `0..3`.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bin = np.digitize(trial_speed, speed_quartiles).astype(np.int64)
speed_bin = np.clip(speed_bin, 0, 3)
```

iii. The conversion log shows the actual thresholds were `[0.0, 9.66718583, 31.45552777]` for the full run. The notes later acknowledge that these bins do not end up close to 25% each in the converted dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by indexing `ft_RunSpeed` with the same per-trial `frame_indices` used to slice the neural trace.

ii.
```python
trial_neural = spk[:, frame_indices].astype(np.float32)
trial_speed = ft_RunSpeed[frame_indices]
```

iii. The alignment justification is the same frame-sharing design as for all other time-varying variables.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles a few edge cases explicitly: multiple possible behavioral keys per session, absent behavior files, mismatched neural/behavior frame counts, NaN `stim_id` entries, sessions with too many neurons, and trials with too few frames. It does not do general missing-value imputation.

ii.
```python
for beh_file in beh_files:
    beh_path = os.path.join(DATA_ROOT, 'beh', beh_file)
    if not os.path.exists(beh_path):
        continue
```

```python
n_frames = min(n_frames_spk, n_frames_beh)
...
if np.isnan(sid_val):
    wall_to_stim[wall] = wall
```

```python
if len(frame_indices) < 2:
    skipped += 1
    continue
```

iii. The notes explicitly mention these “edge cases” and say they were added after debugging sessions with stimtype suffixes, NaN stimulus IDs, and frame-count mismatches.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are loading the very large neural matrices from disk and filtering/subsampling them session by session. Computing global speed quartiles also requires loading every behavioral session once before the main loop.

ii.
```python
print("\nComputing running speed quartiles...")
speed_quartiles = compute_running_speed_quartiles(session_meta, session_ids)
...
for i, sid in enumerate(session_ids):
    ...
    neural_trials, input_trials, output_trials, region_idx = process_session(...)
```

```python
spk = load_spk(mname, datexp, blk)
...
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
```

iii. The conversion logs show per-session runtime is dominated by neural-data load times, often several to tens of seconds per session; the notes make the same point in the runtime estimate table.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial loop in `process_session`, the inner per-lick loop that uses `searchsorted` for every lick event, and the per-region sampling loop in `filter_and_subsample_neurons`. The quartile prepass also reloads behavior session by session.

ii.
```python
for trial_idx in range(ntrials):
    ...
    for lf in trial_lick_frames:
        lf_int = int(np.round(lf))
        pos = np.searchsorted(frame_indices, lf_int)
```

```python
for region in BRAIN_REGIONS:
    region_local_idx = np.where(region_masks[region])[0]
    ...
    sampled = rng.choice(region_local_idx, size=n_sample, replace=False)
```

iii. The agent did not discuss vectorization in its notes, but these are the main obvious scalar/loop-heavy sections in the final script.

## 12-c. What processing does the code repeat multiple times?

i. Behavioral data are loaded twice for every session in full mode: once during global speed-quartile computation and again during session processing. The code also repeatedly scans candidate behavior files/keys in `load_beh`, and it repeatedly constructs per-trial arrays from scratch for each trial.

ii.
```python
speed_quartiles = compute_running_speed_quartiles(session_meta, session_ids)
...
beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

```python
def compute_running_speed_quartiles(session_meta, session_ids):
    for sid in session_ids:
        meta = session_meta[sid]
        beh = load_beh(sid, meta['beh_files'], meta['beh_keys'])
```

iii. The agent’s notes do not call this out, but it is a direct consequence of the script structure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and returns `selected_idx` from neuron subsampling but never uses it later; it loads `ft_CorrSpc` without using it; it fills output channel 0 with a placeholder `-1` and overwrites it later; and it supports a `show_processing` argument in `process_session` that is not used inside that function.

ii.
```python
spk, iarea_filtered, selected_idx = filter_and_subsample_neurons(spk, iarea)
...
return spk[selected_indices], iarea[selected_indices], selected_indices
```

```python
ft_CorrSpc = beh['ft_CorrSpc'][:n_frames]
...
trial_output[0, :] = -1  # placeholder
```

iii. These are implementation leftovers rather than documented scientific decisions. The notes do not justify them.
