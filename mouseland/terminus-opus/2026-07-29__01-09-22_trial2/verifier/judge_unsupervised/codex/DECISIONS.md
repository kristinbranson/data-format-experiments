# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script starts from `data/beh/Imaging_Exp_info.npy`, treats each unique `mname_datexp_blk` neural recording as one session, sorts those session ids, and then loads three modalities per session: neural traces from `data/spk`, retinotopy from `data/retinotopy`, and behavior from `data/beh/Beh_<exp_type>.npy`. When a neural recording appears multiple times in `Imaging_Exp_info.npy`, it keeps the first occurrence only.

ii.
```python
def get_all_sessions():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}
    for exp_type in info.keys():
        for entry in info[exp_type]:
            session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
```

iii. `CONVERSION_NOTES.md` Step 5 says “All 89 sessions included: Each unique neural recording = one session.” Step 10 also says “Sessions with stimtype handled (use first occurrence).”

## 1-b. How are the data split into subjects?

i. Subjects are identified by the mouse name `mname`. The output `subjects` list is built in first-seen session order, and each session gets a `subject_idx` pointing into that list.

ii.
```python
all_subjects = []
...
mname = result['mname']
if mname not in all_subjects: all_subjects.append(mname)
subject_idx = all_subjects.index(mname)
...
'subjects': all_subjects, 'subject_idx': np.array(all_subject_idx),
```

iii. The notes repeatedly describe the dataset as “89 sessions, 19 mice” and use `mname` as the subject identifier.

## 1-c. How are the data split into sessions?

i. Sessions are unique `session_id = mname_datexp_blk` recordings, sorted lexicographically. Multiple `Imaging_Exp_info` rows referring to the same neural recording are collapsed into one session before conversion.

ii.
```python
session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
...
for session_id in sorted(session_map.keys()):
    exp_type, entry, beh_key = session_map[session_id]
    sessions.append({'session_id': session_id, ...})
```

iii. `CONVERSION_NOTES.md` Step 5 says “Each unique neural recording = one session.” The trajectory also shows the agent explicitly counting 142 metadata entries but 89 unique neural sessions.

## 1-d. How are the data split into trials?

i. Within each session, the script uses `beh['ntrials']` as the trial count and defines a trial’s usable frames as those where `ft_trInd == trial_idx` and `ft_move > 0`. Each kept trial becomes one `(neurons, timepoints)` neural matrix plus aligned input/output arrays.

ii.
```python
ntrials = beh['ntrials']
ft_trInd = beh['ft_trInd'][:n_frames_neural]
vr_move = ft_move > 0
...
for trial_idx in range(ntrials):
    valid_mask = (ft_trInd == trial_idx) & vr_move
    valid_frame_indices = np.where(valid_mask)[0]
```

iii. The notes say “All trials included, frames filtered for VR-moving,” and the trajectory shows the agent deciding to use frame-level trial segmentation rather than the repo’s position-interpolated representation.

## 1-e. How are trials filtered based on quality controls?

i. There is no explicit trial-quality curation beyond requiring at least two valid moving frames after `ft_move > 0` filtering. Trials with fewer than two such frames are dropped, and sessions with fewer than two remaining trials are dropped.

ii.
```python
if len(valid_frame_indices) < 2:
    continue
...
if valid_count < 2:
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 says “Trial curation: All trials included, frames filtered for VR-moving.” The extra “at least 2 frames / 2 trials” rule appears to come from the decoder format requirement, not from the reference repo.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` entry inside each `*_neural_data.npy` file. Those traces are concatenated across imaging planes.

ii.
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The notes identify `load_spk` from `utils.py` as the matching reference loader and state that the neural data are Suite2p deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. Processing is minimal: concatenate planes, drop neurons outside selected retinotopy labels, restrict to running frames within each trial, slice the deconvolved trace matrix by those frame indices, and cast to `float32`. The script does not perform the reference repo’s position interpolation or normalization.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `spks (concat) -> neural` with “Filter neurons, extract per-trial running frames.” The trajectory explicitly says the agent chose raw frame-level time bins instead of the repo’s `spk_pos_interp/get_interpPos_spk` position-based pipeline because the decoder task is temporally aligned.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are globally filtered by retinotopy label: keep only those with `iarea != -1` and `iarea != 7`, which the agent interprets as excluding neurons outside visual cortex. Frames are also filtered to `ft_move > 0`.

ii.
```python
neuron_mask = (iarea != -1) & (iarea != 7)
spk = spk[neuron_mask]
...
vr_move = ft_move > 0
valid_mask = (ft_trInd == trial_idx) & vr_move
```

iii. The notes and trajectory both cite the reference utility code comment `idx_neu = (arid!=-1) & (arid != 7) # exclude neurons from outside of visual cortex` and “only considered timepoints during running.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned implicitly by trial membership, not by using `Trial_start_time` or `StartFr`. For each trial, the kept neural timeline starts at the first frame satisfying both `ft_trInd == trial_idx` and `ft_move > 0`, so the effective zero point is the first retained moving frame, not an explicit corridor-entry timestamp.

ii.
```python
valid_mask = (ft_trInd == trial_idx) & vr_move
valid_frame_indices = np.where(valid_mask)[0]
...
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The trajectory shows the agent debating trial-start alignment and deciding to use raw frame-level data aligned by `ft_trInd`. There is no note showing use of `Trial_start_time` or `StartFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use the imaging frame rate directly: `FRAME_RATE = 3.17 Hz`, so each bin is about `315.46 ms`. No temporal rebinning is applied.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
...
'time_bin_size': TIME_BIN_MS,
```

iii. `CONVERSION_NOTES.md` Step 3 and Step 9 both cite `3.17 Hz` / `315.5 ms bins`, and the trajectory recovered `fs = 3.17Hz` from the notebook.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `ft`.

ii.
```python
ft = beh['ft'][:n_frames_neural]
sound_fr = beh['SoundFr']
...
s_fr = sound_fr[trial_idx]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `SoundFr, ft -> input[0]: time_to_sound_cue`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The script treats `SoundFr` as a possibly fractional frame index, interpolates that index onto the `ft` timeline to get an absolute sound time, then subtracts that sound time from each kept frame time and converts MATLAB day units to seconds. If `SoundFr` is `NaN`, it fills the whole trial with zeros.

ii.
```python
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
else:
    s_fr_int = int(np.floor(s_fr))
    s_fr_frac = s_fr - s_fr_int
    ...
    sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
    time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes say “Time computations: Using actual frame times (MATLAB datenum).” The trajectory shows the agent inspecting `SoundFr` and `ft` and deciding to use frame-time interpolation rather than only position.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on the exact same `valid_frame_indices` used to slice the neural data, so it has the same per-trial time axis as `neural`.

ii.
```python
ft_trial = ft[valid_frame_indices]
...
time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
...
input_trial[0] = time_to_sound
```

iii. This follows the Step 5 plan in the notes to make time-varying decoder inputs aligned to the frame-level neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata only: `mname` and the recording date string `datexp` from `Imaging_Exp_info.npy`.

ii.
```python
for i, s in enumerate(sessions):
    mouse_sessions[s['mname']].append((i, s))
...
datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]), int(s['datexp'].split('_')[2]))
```

iii. `CONVERSION_NOTES.md` Step 5 maps `datexp -> input[1]: day_of_training`.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script sorts that mouse’s sessions by recording date and assigns `day_of_training = (session_date - first_session_date).days`. That scalar is then copied into every time bin of every trial from that session.

ii.
```python
dates.sort(key=lambda x: x[1])
first_date = dates[0][1]
for idx, date in dates:
    day_of_training[idx] = (date - first_date).days
...
input_trial[1] = np.float32(day_val)
```

iii. The notes justify this as “Days since first session per mouse.” The trajectory does not show a stronger reference-based justification than that heuristic.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the kept frame times `ft_trial`, which themselves come from `ft[valid_frame_indices]`. The script does not use `Trial_start_time` or `StartFr` directly.

ii.
```python
ft = beh['ft'][:n_frames_neural]
...
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft -> input[2]: time_since_trial_start`. The trajectory shows the agent choosing frame-based temporal alignment rather than explicit start-frame anchoring.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script subtracts the first retained frame time from every retained frame time in the trial and converts from MATLAB day units to seconds. This means time zero is the first kept moving frame.

ii.
```python
ft_trial = ft[valid_frame_indices]
time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. The notes describe it as “Elapsed time from trial start,” but the code shows that the implementation is actually “elapsed time from first retained frame.”

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned one-to-one with the neural time axis because both use the same `valid_frame_indices`. The alignment anchor is the first kept moving frame in the trial.

ii.
```python
neural_trial = spk[:, valid_frame_indices].astype(np.float32)
...
input_trial[2] = time_since_start
```

iii. This is consistent with the script’s general design choice, stated in the trajectory, to use the retained frame sequence as the shared timeline for neural, inputs, and outputs.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level boolean `isRew`.

ii.
```python
is_rew = beh['isRew']
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `isRew -> reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts `isRew[trial_idx]` into `1.0` or `0.0` and repeats that scalar across all time bins of the trial.

ii.
```python
input_trial = np.zeros((4, n_t), dtype=np.float32)
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The notes describe it as “1=rewarded, 0=not,” with no additional processing.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from trial-level wall identities: `WallName`, with the global category vocabulary collected from all sessions’ `UniqWalls`.

ii.
```python
for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(wn))
...
wall_name = beh['WallName']
stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `WallName -> visual_stimulus_category` and shows the globally collected stimulus list.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script builds a single global `stim_to_idx` dictionary over all sessions, converts the current trial’s `WallName` to that integer code, and writes the same code into every time bin of the trial.

ii.
```python
all_stim_names = get_all_stim_names(all_sessions)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
output_trial[0] = stim_idx
```

iii. The notes emphasize a global categorical mapping so all sessions share one stimulus index space.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from the session-level lick frame indices `LickFr`.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
lick_trial = lick_binary[valid_frame_indices]
```

iii. `CONVERSION_NOTES.md` Step 5 maps `LickFr -> output[1]: licking`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The script initializes an all-zero frame vector for the session, rounds each `LickFr` to the nearest integer frame index, clips to the valid frame range, and sets those frames to `1`. Multiple licks in the same frame collapse to one binary event.

ii.
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    ...
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
```

iii. The notes only say “Binary per frame.” The trajectory shows the agent inspected `LickFr` and chose frame-index binarization.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-level binary lick vector is sliced by the same `valid_frame_indices` used for `neural`, so licking and neural activity are frame-aligned within each kept trial.

ii.
```python
lick_trial = lick_binary[valid_frame_indices]
...
output_trial[1] = lick_trial
```

iii. This follows the notes’ plan to make licking a time-varying output aligned to the neural frame sequence.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from framewise position `ft_Pos`.

ii.
```python
ft_Pos = beh['ft_Pos'][:n_frames_neural]
...
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_Pos -> output[2]: position_bin`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script samples `ft_Pos` on the retained moving frames and discretizes those positions by dividing by `15.0`, flooring, clipping to `[0, 3]`, and casting to integer.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
...
output_trial[2] = pos_bins
```

iii. The notes justify this as “4 bins of 15dm over 60dm,” explicitly using the full 6 m corridor rather than only the 4 m textured segment.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are `[0,1,2,3]` for the ranges `[0,15)`, `[15,30)`, `[30,45)`, `[45,60]` in the `ft_Pos` units, which the metadata labels as `0-1.5m`, `1.5-3m`, `3-4.5m`, `4.5-6m`.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
...
'output_values': [..., ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ...],
```

iii. `CONVERSION_NOTES.md` Step 5 and the README both state that the agent intentionally used four equal bins over the full 6 m corridor.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is aligned by taking `ft_Pos` values at the same `valid_frame_indices` used for the neural slices, so every retained neural frame gets one position-bin label.

ii.
```python
pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
...
output_trial[2] = pos_bins
```

iii. The notes consistently describe outputs as frame-aligned to the same running-frame trial timeline.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_RunSpeed -> output[3]: running_speed_bin`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first computes global quartile thresholds from all sessions’ `ft_RunSpeed` values restricted to `ft_move > 0`. Within each trial, it samples `ft_RunSpeed` on the retained frames and bins those values with `np.digitize`.

ii.
```python
speeds = beh['ft_RunSpeed'][vr_move]
...
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. `CONVERSION_NOTES.md` Step 5 calls this “Global quartile-based (computed from all sessions).” The trajectory shows the agent choosing this to satisfy the decoder instruction “each corresponding to 25% of the data.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global 25th, 50th, and 75th percentiles of the retained running-speed sample. `np.digitize` then converts each value to categories `0-3`.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. The notes explicitly describe the output as “4 quartile bins.”

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same `valid_frame_indices` as `neural`, so each retained neural frame has one running-speed-bin label.

ii.
```python
speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
...
output_trial[3] = speed_bins
```

iii. This matches the script’s overall shared frame-axis design for neural, inputs, and outputs.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or awkward values are handled by simple fallbacks: missing `LickFr` becomes an empty array and therefore an all-zero lick vector; `NaN` `SoundFr` becomes an all-zero time-to-sound vector; out-of-range `SoundFr` values are clamped to the first or last frame time; behavior arrays are truncated to the neural frame count; very short trials are skipped.

ii.
```python
lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
...
if np.isnan(s_fr):
    time_to_sound = np.zeros(n_t, dtype=np.float32)
...
elif s_fr_int >= len(ft) - 1:
    sound_time = ft[-1]
else:
    sound_time = ft[0]
...
ft = beh['ft'][:n_frames_neural]
```

iii. The notes do not present a strong reference-based rationale here; these are pragmatic safeguards added during script development.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading huge neural arrays from disk, iterating through every trial in every session to slice frame subsets, computing global speed quartiles in a full-dataset pass, serializing each session to temporary pickles, and then reloading those pickles to assemble the final output.

ii.
```python
spk = load_spk(...)
...
for trial_idx in range(ntrials):
    ...
speed_quartiles = compute_global_speed_quartiles(all_sessions)
...
with open(temp_fn, 'wb') as f:
    pickle.dump(...)
...
with open(tf, 'rb') as f:
    sd = pickle.load(f)
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 9 focus on memory pressure and full-run time, and the script prints timing around loading and processing for each session.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorization opportunities are the per-trial session loop in `process_session`, the repeated per-session scans in `compute_global_speed_quartiles` and `get_all_stim_names`, and the small region-assignment loop at the end of `process_session`.

ii.
```python
for trial_idx in range(ntrials):
    ...
for i, session in enumerate(sessions):
    ...
for r_idx, r_name in enumerate(['V1', 'mHV', 'lHV', 'aHV']):
    region_idx[area_map[r_name]] = r_idx
```

iii. The notes say “Vectorize loops” was a goal, but the final implementation still relies heavily on Python loops over trials and repeated session passes.

## 12-c. What processing does the code repeat multiple times?

i. It reloads behavior files multiple times across separate passes: once to build the session list, again to gather global stimulus names, again to compute running-speed quartiles, and again inside every session conversion. It also writes session data to temporary files and then reads them back immediately.

ii.
```python
all_sessions = get_all_sessions()
all_stim_names = get_all_stim_names(all_sessions)
speed_quartiles = compute_global_speed_quartiles(all_sessions)
...
beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
...
with open(temp_fn, 'wb') as f:
    pickle.dump(...)
...
with open(tf, 'rb') as f:
    sd = pickle.load(f)
```

iii. `CONVERSION_NOTES.md` Step 6 mentions some memory optimizations, but not the repeated behavior-file passes; that repetition is visible directly in the code.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It includes work that does not affect the final decoder inputs/outputs: optional plotting, repeated temporary-file serialization just to reassemble the same structures, full-session metadata bookkeeping, and repeated per-time-bin broadcasting of per-trial constants such as stimulus id and reward availability.

ii.
```python
if args.show_processing and i < 2:
    plot_processing(...)
...
with open(temp_fn, 'wb') as f:
    pickle.dump(...)
...
output_trial[0] = stim_idx
...
input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. The trajectory and notes justify plotting as a debugging aid and temp files as a memory workaround, but neither is required by the final decoder once `converted_data.pkl` exists.
