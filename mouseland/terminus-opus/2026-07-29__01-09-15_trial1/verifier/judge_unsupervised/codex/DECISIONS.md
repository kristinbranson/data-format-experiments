# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads `Imaging_Exp_info.npy`, builds one base session id per `(mname, datexp, blk)`, then for each base session loads behavior from one or more `Beh_<exp_type>.npy` files, neural activity from `data/spk/<mname>_<datexp>_<blk>_neural_data.npy`, and retinotopy from `data/retinotopy/<mname>_<datexp>_trans.npz`. For behavior it collects all matching direct and `_swap*` keys, but `process_session()` only uses the first returned behavior entry, so it does not truly load all behavioral variants when multiple exist.

ii. 
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
session_ids = sorted(all_sessions.keys())
```

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
```

```python
beh_results = load_beh_for_session(sess_id, exp_types)
...
beh, beh_key = beh_results[0]
...
spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
```

iii. The notes say the agent intended to “load all sessions from exp_info” and later claimed that 13 swap-only sessions were skipped. The trajectory also shows a later patch justified as “The simplest fix: just use the first result (non-swap variant).” That justification explains why the final code arbitrarily selects the first behavior entry instead of representing every swap variant.

## 1-b. How are the data split into subjects (mice)?

i. Sessions are assigned to mice by the `mname` field from `exp_info`. The final `subjects` list is the sorted set of mouse names from processed sessions, and `subject_idx` maps each session to that subject list.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
...
subject_idx.append(mouse_to_idx[result['mname']])
```

iii. The notes describe “19 subjects” and consistently refer to mouse identity as `mname`. No alternative subject split appears in the trajectory.

## 1-c. How are the data split into sessions?

i. A session is defined as a unique base id string `"{mname}_{datexp}_{blk}"` built from `exp_info`. This collapses all behavior keys sharing the same recording id, including `_swap1` and `_swap2` variants, into one session record.

ii.
```python
for s in sess_list:
    sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
    if sess_id not in sessions:
        sessions[sess_id] = {
            'mname': s['mname'],
            'datexp': s['datexp'],
            'blk': s['blk'],
            'exp_types': [],
```

iii. The notes say “Loads all sessions from exp_info.” The later trajectory patch and the final code show that swap variants are treated as alternate behavior entries under the same base recording, not as separate output sessions.

## 1-d. How are the data split into trials?

i. For each chosen behavior entry, the script uses `beh['ntrials']` and loops over `range(ntrials)`. Each trial is extracted from the neural and behavioral frame streams using `StartFr` as the first frame and `GrayFr` as the corridor end.

ii.
```python
ntrials = beh['ntrials']
...
for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
```

```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
...
end_fr = min(start_fr + n_timepoints, gray_fr)
actual_frames = end_fr - start_fr
```

iii. The notes explicitly say “Extract per-trial from StartFr to GrayFr” and “Extracts corridor portion (StartFr to GrayFr) for each trial.”

## 1-e. How are trials filtered based on quality controls?

i. Trial QC is minimal. A trial is dropped if the corridor segment has fewer than 2 frames, if the computed slice has fewer than 2 frames, or if the frame bounds fall outside the neural array. A whole session is dropped if it ends up with fewer than 2 valid trials.

ii.
```python
avail_frames = gray_fr - start_fr
if avail_frames < 2:
    return None
...
if actual_frames < 2:
    return None
...
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

```python
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. The notes justify this as handling edge cases such as “Trials with <2 frames excluded” and satisfying the decoder requirement that each session must have at least two trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from the `spks` field inside each session’s `*_neural_data.npy` file.

ii.
```python
dat = np.load(spk_path, allow_pickle=True).item()
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
```

iii. The notes identify `load_spk` as the key reference loader and describe the neural source as “spks (concat planes).”

## 2-b. How is the `neural` data processed?

i. The script concatenates all planes/cell groups along the neuron axis, slices the corridor frames for each trial, and casts the trial array to `float16`. It does not compute new fluorescence measures or apply the reference code’s position interpolation.

ii.
```python
spk = np.concatenate([nspk for nspk in dat['spks']], 0)
...
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The notes say the raw data are already “deconvolved Ca2+ traces from Suite2p” and the script note says “Neural data stored as float16 to reduce file size.” The trajectory never shows a justification for skipping interpolation beyond choosing a simpler trial-aligned decoder representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no explicit neuron-level quality filter. Neurons are included as loaded; only trial slices can be dropped for invalid frame ranges.

ii.
```python
spk = load_spk(mname, datexp, blk)
...
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The notes explicitly state “No explicit neuron quality filtering in reference code” and later “No neuron filtering: Consistent with reference code.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry / trial start. For each trial the extracted neural segment begins at `StartFr` and ends at `GrayFr`, so time zero is the first corridor frame.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
...
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,
```

iii. The notes repeatedly justify this as matching the decoder task: “Temporally aligned based on trial start (corridor entry)” and “Trial extraction: Uses StartFr to GrayFr (corridor portion only).”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the native neural-frame resolution, using a fixed frame rate of 3.17 Hz, equivalent to about 315.5 ms per bin. No rebinning is applied.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
...
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. The notes and reference-text summary both cite 3.17 Hz as the frame rate, and the script simply slices raw frames instead of aggregating them.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `StartFr`, plus the implied within-trial frame index.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
sound_fr = beh['SoundFr'][trial_idx]
...
sound_frame_offset = sound_fr - start_fr
```

iii. The notes map “SoundFr - frame” to `input[0]: time_to_sound_cue`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The script computes a signed continuous time axis relative to cue onset: `(current_frame_index - sound_offset) / FRAME_RATE`. Values are negative before the cue and positive after it.

ii.
```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The code comment itself states the intended meaning: “Signed time from current frame to sound cue (negative = before cue, positive = after).” The notes only say “Time in seconds.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same per-trial frame index `0..actual_frames-1` used for the extracted neural slice, so each element corresponds to the same neural frame.

ii.
```python
neural = trial_data['neural']
...
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The alignment follows directly from the shared `actual_frames` and trial-start extraction that the notes describe.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` in `exp_info`, not from a direct per-trial field in the behavior dict.

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

iii. The notes explicitly map `datexp` to `day_of_training` and describe the transform as “Chronological day index.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the script gathers all unique session dates across `exp_info`, sorts them, uses the 0-based index of the current date as the training-day value, and repeats that scalar across every frame of the trial.

ii.
```python
training_day = get_training_day(mname, datexp, exp_info)
...
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The only explicit justification in the notes is “Chronological day index.” The trajectory does not show the agent considering alternatives such as `sess#`.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived implicitly from the trial’s frame index after `StartFr`, together with the constant `FRAME_RATE`. It does not use the raw `ft` timestamps.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
...
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The notes summarize this as “frame - StartFr” mapped to time in seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The script creates a uniformly spaced vector `[0, 1/FRAME_RATE, 2/FRAME_RATE, ...]` for each extracted trial segment.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The justification is implicit: the agent used the nominal frame rate instead of variable `ft` intervals, presumably to keep all trial time axes simple and uniform.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same `actual_frames` length as the neural slice, starting at 0 for the first extracted neural frame.

ii.
```python
neural = trial_data['neural']
...
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. This is part of the same trial-start alignment strategy described in the notes.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `beh['isRew']`.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The notes map `isRew` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts the trial’s reward flag to `float` and repeats it across all timepoints in that trial.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The notes describe this simply as a binary per-trial variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `beh['WallName']`.

ii.
```python
wall_names = beh['WallName']
...
stim_name = str(wall_names[t])
```

iii. The notes map `WallName` to `output[0]: visual_stimulus`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. For each trial the script takes the raw wall/stimulus name string, builds a global sorted list of all stimulus names seen in processed sessions, converts each name to an integer index, and repeats that index across the whole trial.

ii.
```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
stim_idx = stim_to_idx[trial_out['stim_name']]
out[0, :] = stim_idx
```

iii. The notes justify this as treating `WallName` as a categorical label. The trajectory does not show any extra collapsing of categories with `stim_id` or `get_cat_id`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr` and `LickTrind`.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The notes map `LickFr` to a binary per-frame licking output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial the script creates a zero vector over corridor frames, rounds each lick frame to the nearest integer frame, converts it to an offset from `StartFr`, and writes `1.0` at those offsets.

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. The notes justify this as “Binary per frame.” No more complex lick histogramming from the reference code is reused.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick frames are converted to offsets relative to `StartFr` and clipped to the extracted corridor segment, so the licking vector is frame-aligned with the neural slice.

ii.
```python
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0
```

iii. This follows the same trial-start corridor alignment used for all trial-varying variables.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `beh['ft_Pos']`.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The notes map `ft_Pos` to `position_bin`.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The script slices `ft_Pos` over the trial’s corridor frames and then discretizes the resulting per-frame VR positions.

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
...
pos_bins = discretize_position(trial_data['position'])
```

iii. The notes describe this as extracting corridor frames and binning into four 1 m bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is binned with fixed edges `[0, 10, 20, 30, 40]` in VR units, corresponding to four 1 m bins over the 4 m textured corridor.

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
...
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```

iii. The notes explicitly justify these as “4 bins of 1m each,” using `1 VR unit = 0.1m`.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sliced over the exact same `start_fr:end_fr` corridor-frame window as neural activity, so each position bin corresponds to one neural frame.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The notes say the entire conversion is aligned to corridor entry and uses the corridor portion only.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['ft_RunSpeed']`.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. The notes map `ft_RunSpeed` to `speed_bin`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first gathers all corridor-frame speeds from all selected sessions to estimate global quartile edges. During trial extraction it slices per-frame `ft_RunSpeed` over the corridor segment and discretizes it using those global edges.

ii.
```python
for t in range(ntrials):
    start_fr = int(np.round(beh['StartFr'][t]))
    gray_fr = int(np.round(beh['GrayFr'][t]))
    if start_fr < len(beh['ft_RunSpeed']) and gray_fr <= len(beh['ft_RunSpeed']):
        trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
        speeds.append(trial_speed)
```

```python
flat_speeds = np.concatenate(all_speeds)
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
```

iii. The notes justify this as “4 quartile bins,” and the trajectory shows a dedicated first pass for quartile computation.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The script uses the 25th, 50th, and 75th percentiles of all collected speed samples as bin cut points, then maps each frame to one of four quartile bins.

ii.
```python
edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
...
bins = np.digitize(speed, bin_edges[1:-1])
bins = np.clip(bins, 0, N_SPEED_BINS - 1)
```

iii. The notes say the speed output should be “4 quartile bins,” matching the decoder instruction.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sliced over `start_fr:end_fr`, the same frame range used for the neural data.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. This is part of the general per-frame trial alignment described in the notes.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The handling is ad hoc: float frame markers are rounded to integer frames; too-short or out-of-bounds trials are skipped; sessions with no behavior or fewer than two valid trials are skipped; NaNs are removed only when computing global speed-bin edges. There is no dedicated imputation or consistency repair for per-frame missing values.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
...
if avail_frames < 2:
    return None
...
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

```python
flat_speeds = np.concatenate(all_speeds)
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
```

iii. The notes justify these choices via “Edge cases” such as rounded float frame indices and skipped short trials, but they do not describe a more systematic missing-data strategy.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive parts are loading the huge `spk` files, extracting every trial for every session, and writing the enormous pickle. The script explicitly measures per-session load time and total time, and it performs two full passes over the dataset.

ii.
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
```

```python
for i, sess_id in enumerate(session_ids):
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
...
for i, sess_id in enumerate(session_ids):
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

iii. The notes mention a full conversion time, per-session load times, and a 117 GB output file, which all point to I/O-heavy loading and serialization as the dominant costs.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial speed collection loop, per-trial extraction loop, per-lick loop, and list-comprehension style construction of `time_to_cue` and `time_since_start` are all scalar/serial. The final output assembly also loops trial-by-trial.

ii.
```python
for t in range(ntrials):
    ...
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
```

```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

```python
for lf in trial_lick_frs:
    ...
    licking[frame_offset] = 1.0
```

iii. The notes say the code should be efficient, but the final script mostly uses straightforward Python loops rather than vectorized NumPy operations.

## 12-c. What processing does the code repeat multiple times?

i. It loads behavior files repeatedly, makes one full pass just to collect speed samples and another full pass to do the actual conversion, recomputes rounded frame boundaries per trial in both passes, and recomputes training-day date sets for every session.

ii.
```python
# Pass 1
speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
...
# Pass 2
result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```

```python
for exp_type in exp_types:
    beh_path = f'data/beh/Beh_{exp_type}.npy'
    if os.path.exists(beh_path):
        beh_all = np.load(beh_path, allow_pickle=True).item()
```

```python
def get_training_day(mname, datexp, exp_info):
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
```

iii. The notes themselves describe the two-pass structure. The trajectory never shows caching behavior dictionaries or precomputing training-day maps.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The final script contains unused helper functions (`build_session_to_beh_map`, `get_brain_region_indices`, `get_stimulus_category`), keeps unused locals like `n_total_frames`, `subjects`, and `beh_key`, and optionally generates processing plots that are not part of decoder training. These do not change the final converted arrays.

ii.
```python
def build_session_to_beh_map(exp_info):
    ...

def get_brain_region_indices(iarea, brain_regions):
    ...

def get_stimulus_category(wall_name):
    return wall_name
```

```python
n_total_frames = spk.shape[1]
...
subjects = []
...
beh, beh_key = beh_results[0]
```

iii. The trajectory shows the agent adding plotting and documentation features for inspection. Those are useful for validation, but not for the downstream decoder itself.
