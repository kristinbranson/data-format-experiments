# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script loads experiment metadata from `data/beh/Imaging_Exp_info.npy`, builds session IDs as `mname_datexp_blk`, then for each session loads behavior from one or more `Beh_<exp_type>.npy` files, neural activity from `data/spk/<session>_neural_data.npy`, and retinotopy from `data/retinotopy/<mouse>_<date>_trans.npz`. If multiple behavioral variants exist for a session, it gathers them, but `process_session` uses only the first returned behavior dictionary.

ii. ```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)

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
    return unique_results if unique_results else None

spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
beh, beh_key = beh_results[0]
```

iii. `CONVERSION_NOTES.md` and the trajectory show the agent wanted to use all 89 recordings from `Imaging_Exp_info.npy`. The trajectory also shows an initial failure to load swap-only behavior keys, followed by a later code change that added `_swap` lookup. The final code reflects that partial fix, but still keeps only the first behavior variant per base session.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. After session processing, the script takes the unique mouse names present in `all_results`, sorts them, stores them in `subjects`, and maps each session to its mouse with `subject_idx`.

ii. ```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}

for result in all_results:
    subject_idx.append(mouse_to_idx[result['mname']])

data = {
    'subjects': all_mice,
    'subject_idx': np.array(subject_idx),
}
```

iii. The notes say the dataset has 19 mice and the trajectory repeatedly refers to sessions by `mname`, so the agent's intended subject identity was the mouse name.

## 1-c. How are the data split into sessions?

i. Sessions are split by the base key `mname_datexp_blk`. `build_all_sessions_info` deduplicates `exp_info` entries on that base key and aggregates the list of experiment types seen for that recording. It does not create separate output sessions for different `stimtype` variants such as `swap1` and `swap2`.

ii. ```python
def build_all_sessions_info(exp_info):
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if sess_id not in sessions:
                sessions[sess_id] = {
                    'mname': s['mname'],
                    'datexp': s['datexp'],
                    'blk': s['blk'],
                    'exp_types': [],
                    'exptype': s.get('exptype', 'unknown'),
                    'rewType': s.get('rewType', 'None'),
                }
            sessions[sess_id]['exp_types'].append(exp_type)
    return sessions
```

iii. The trajectory shows the agent noticed `stimtype`-based swap sessions in `exp_info`, but the final session-building code still collapses everything onto the base recording key. The notes earlier described 76 processed sessions because of swap handling problems; the final code is a partial correction, not a full session model that preserves `stimtype`.

## 1-d. How are the data split into trials?

i. Within each selected session, trials are split using the behavior dictionary's `ntrials`, `StartFr`, and `GrayFr`. The script loops over `range(ntrials)`, and `extract_trial_data` slices each trial from corridor entry (`StartFr`) to gray-space entry (`GrayFr`).

ii. ```python
ntrials = beh['ntrials']

for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)
    if trial_data is None:
        continue

def extract_trial_data(spk, beh, trial_idx, n_timepoints):
    start_fr = int(np.round(beh['StartFr'][trial_idx]))
    gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
    end_fr = min(start_fr + n_timepoints, gray_fr)
```

iii. The notes explicitly say trial extraction is `StartFr` to `GrayFr`, and the trajectory says the agent aligned trials to corridor entry because the decoder instructions specified trial-start alignment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by simple structural validity checks. A trial is dropped if it has fewer than 2 frames, if the derived slice is shorter than 2 frames, or if the slice would go outside the neural array. Sessions are dropped if they end up with fewer than 2 valid trials. There is no explicit filter for low-running trials, despite the paper saying analyses considered only running timepoints.

ii. ```python
avail_frames = gray_fr - start_fr
if avail_frames < 2:
    return None

end_fr = min(start_fr + n_timepoints, gray_fr)
actual_frames = end_fr - start_fr
if actual_frames < 2:
    return None

if start_fr < 0 or end_fr > spk.shape[1]:
    return None

if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. The notes mention excluding only one very short trial and capping trial length, while the trajectory explicitly debates whether to filter non-running timepoints and then chooses not to. That is the agent's justification for the light trial QC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes directly from the `spks` entry inside each `*_neural_data.npy` file. The script concatenates all imaging planes along the neuron axis.

ii. ```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    dat = np.load(spk_path, allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk
```

iii. The notes identify `load_spk` from the reference `utils.py` as the loading function and describe the source as deconvolved calcium traces from Suite2p.

## 2-b. How is the `neural` data processed?

i. Neural data are not re-normalized or deconvolved again. The script slices the already deconvolved traces for each trial from `StartFr` to `GrayFr` and casts the result to `float16` to reduce file size.

ii. ```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
end_fr = min(start_fr + n_timepoints, gray_fr)
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The notes say the reference analyses used deconvolved traces and no additional neuron-quality filter. The trajectory shows the agent considered the reference interpolation pipeline but decided to keep framewise temporal slices for the decoder and switch to `float16` for storage reasons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no neuron-level quality-control filtering in the conversion script. Neural data are only filtered indirectly when the associated trial fails the structural checks in `extract_trial_data`. The script also does not apply the paper's running-only timepoint filter.

ii. ```python
neural = spk[:, start_fr:end_fr].astype(np.float16)

if avail_frames < 2:
    return None
if actual_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```

iii. The notes explicitly say "No explicit neuron quality filtering in reference code" and the trajectory repeatedly treats "include all neurons" as a requirement. The same trajectory also acknowledges the paper's running-only analysis but does not implement it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each neural trial is aligned to trial start, defined as corridor entry at `StartFr`. Timepoint 0 for a trial is the first neural frame at `StartFr`.

ii. ```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))
neural = spk[:, start_fr:end_fr].astype(np.float16)

data['metadata'] = {
    'temporal_alignment_event': 'Trial start (corridor entry)',
    'off_start': 0.0,
}
```

iii. The notes and trajectory both state that the decoder instructions required temporal alignment to trial start, so the agent anchored trial slices at `StartFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is the native frame rate, `1 / 3.17` s per bin, stored as `TIME_BIN_MS = 1000 / 3.17`, about 315.5 ms. No temporal rebinning is applied. Trials keep their native frame spacing, although very long trials are truncated at 1000 frames.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms

trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)

'metadata': {
    'time_bin_size': TIME_BIN_MS,
    'frame_rate_hz': FRAME_RATE,
}
```

iii. The notes record the 3.17 Hz frame rate from the paper and notebook. The trajectory shows the agent deliberately kept native temporal samples instead of spatially interpolating to fixed position bins.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr` and `StartFr`, with the frame-to-seconds conversion set by `FRAME_RATE`.

ii. ```python
sound_fr = beh['SoundFr'][trial_idx]
start_fr = int(np.round(beh['StartFr'][trial_idx]))
sound_frame_offset = sound_fr - start_fr
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The notes map `SoundFr - frame` to the decoder input "time_to_sound_cue", and the trajectory says the agent chose a trial-start-aligned relative time signal.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The script computes the cue's offset from the trial start in frames, then for every frame index `i` in the trial calculates `(i - sound_offset) / FRAME_RATE`. This yields a signed continuous signal with negative values before the cue and positive values after the cue.

ii. ```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The code comment itself documents the agent's sign convention. The trajectory shows the agent intentionally kept this as a continuous time-varying variable because that matched the decoder specification.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned frame-by-frame on the same per-trial grid as the neural slice. Both start at `StartFr`, and `time_to_sound_cue[k]` corresponds to neural frame `k` within that trial.

ii. ```python
neural = trial_data['neural']
actual_frames = trial_data['actual_frames']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The notes and trajectory both emphasize trial-start alignment, so all time-varying inputs were built on the same frame count `actual_frames`.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the mouse name `mname` and experiment date `datexp` inside `Imaging_Exp_info.npy`.

ii. ```python
def get_training_day(mname, datexp, exp_info):
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
    sorted_dates = sorted(dates)
    return sorted_dates.index(datexp)
```

iii. The notes map `datexp` to day of training, and the trajectory says the agent wanted a chronological day index per mouse rather than an experiment-type label.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each session, the script gathers all unique dates used by that mouse across `exp_info`, sorts them lexicographically, takes the index of the current `datexp`, and repeats that scalar across all trial frames.

ii. ```python
training_day = get_training_day(mname, datexp, exp_info)
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. The notes describe this as a chronological day index, and the trajectory treats that as the intended representation for the decoder input.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. No raw variable is used because the final script does not include an environment-type input at all. The only inputs are `time_to_sound_cue`, `day_of_training`, `time_since_trial_start`, and `reward_availability`.

ii. ```python
input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
```

iii. The decoder instructions in `instruction_reference.md` specify only four inputs, and the notes' mapping table uses exactly those four. The agent therefore omitted environment type entirely.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The script does not compute, encode, or store any environment-type variable.

ii. ```python
input_names = ['time_to_sound_cue', 'day_of_training', 'time_since_trial_start', 'reward_availability']
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The omission is consistent with the notes and the decoder task definition rather than with any separate environment-type design.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the trial's frame index relative to `StartFr` and the constant `FRAME_RATE`. The code does not use the raw timestamp vector `beh['ft']`.

ii. ```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The notes map `frame - StartFr` to `time_since_trial_start`, and the trajectory says the agent wanted a simple trial-start-aligned temporal axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial, the script creates a vector `0, 1/FRAME_RATE, 2/FRAME_RATE, ...` up to the number of extracted frames.

ii. ```python
actual_frames = trial_data['actual_frames']
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The agent's notes explicitly describe this variable as `frame - StartFr` converted to seconds.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned exactly to the neural slice because it is built from the same `actual_frames` length after slicing neural data from `StartFr`.

ii. ```python
neural = trial_data['neural']
actual_frames = trial_data['actual_frames']
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The trajectory says the agent treated `StartFr` as time zero for the entire converted trial structure.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial behavior flag `beh['isRew'][t]`.

ii. ```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The notes map `isRew` directly onto reward availability and note that unsupervised sessions simply have `False` trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The script converts the boolean trial label to `float` and repeats it across every frame in the trial, producing a constant 0/1 time series.

ii. ```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. The trajectory says the agent kept unsupervised and naive sessions by treating `isRew=False` as a valid input value rather than filtering those sessions out.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial stimulus label `beh['WallName'][t]`.

ii. ```python
wall_names = beh['WallName']
stim_name = str(wall_names[t])
```

iii. The notes map `WallName` to visual stimulus category, and the trajectory notes that labels such as `wood1` are kept as they appear in the data.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script keeps the raw string label, collects all unique labels across processed sessions, sorts them, builds an integer lookup table `stim_to_idx`, and repeats the resulting integer code across all frames of that trial in the final output array.

ii. ```python
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}

stim_idx = stim_to_idx[trial_out['stim_name']]
out = np.zeros((4, n_t), dtype=np.int64)
out[0, :] = stim_idx
```

iii. The notes say visual stimulus is a per-trial categorical output. The trajectory also records the agent's decision to keep dataset labels even where paper text used a different informal name such as brick versus wood.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `beh['LickFr']`, `beh['LickTrind']`, and the trial's `StartFr`.

ii. ```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. The notes map `LickFr` to the licking output, and the trajectory says the agent wanted a framewise binary series.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the script initializes an all-zero vector of length `actual_frames`, rounds each lick frame to an integer, converts it to an offset from `StartFr`, and sets that position to 1 if it falls inside the trial window.

ii. ```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. The notes describe licking as "binary per frame", and the trajectory emphasizes making it time-varying rather than a per-trial summary.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Lick times are aligned to neural data by subtracting `StartFr`, so lick frame offset `k` corresponds to neural frame `k` within the same trial.

ii. ```python
frame_offset = lf_int - start_fr
if 0 <= frame_offset < actual_frames:
    licking[frame_offset] = 1.0

out[1, :] = trial_out['licking'].astype(np.int64)
```

iii. The trajectory repeatedly states that all streams were aligned to trial start, and the code uses the same frame index basis as the neural slice.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `beh['ft_Pos']` sampled over the trial window from `StartFr` to `GrayFr`.

ii. ```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. The notes map `ft_Pos` to position output and identify the textured corridor as 0 to 40 VR units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code takes the raw per-frame `ft_Pos` slice for the corridor portion of the trial and passes it directly to `discretize_position` without smoothing or interpolation.

ii. ```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
pos_bins = discretize_position(trial_data['position'])
```

iii. The trajectory says the agent chose framewise temporal data rather than the reference position-interpolated representation, so it kept the native position trace and discretized afterward.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is thresholded into four 1 m bins using edges `[0, 10, 20, 30, 40]` in VR units, where 1 VR unit is assumed to be 0.1 m. `np.digitize` produces bin indices 0 to 3, clipped to the valid range.

ii. ```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]

def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    bins = np.digitize(position, bin_edges[1:])
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```

iii. The notes state the same 0-1 m, 1-2 m, 2-3 m, 3-4 m mapping and justify it from the 4 m textured corridor described in the paper.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position uses the same frame slice `start_fr:end_fr` as the neural data, so each position sample corresponds to the same neural frame index within the trial.

ii. ```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
pos_bins = discretize_position(trial_data['position'])
```

iii. The trial-start-aligned slicing described in the notes is applied identically to both streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `beh['ft_RunSpeed']` over the trial window from `StartFr` to `GrayFr`.

ii. ```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```

iii. The notes map `ft_RunSpeed` to running-speed output and treat it as a framewise behavioral signal.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script makes a first full pass over the dataset to collect all corridor-frame speed samples, removes NaNs, computes global quartile edges with `np.percentile`, and then bins each trial's per-frame speed trace with those edges.

ii. ```python
def compute_speed_bin_edges(all_speeds):
    flat_speeds = np.concatenate(all_speeds)
    flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges

trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. The notes say speed bins should be quartiles over the full dataset, and the trajectory shows the agent deliberately added a first pass just to compute those quartiles.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholding uses the global quartile edges from pass 1. `np.digitize(speed, bin_edges[1:-1])` assigns each frame to one of four bins, clipped to 0 to 3.

ii. ```python
def discretize_speed(speed, bin_edges):
    bins = np.digitize(speed, bin_edges[1:-1])
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```

iii. The notes define the target as four bins corresponding to 25% of the data, which is exactly how the agent justified the percentile-based edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned to the neural slice by using the same frame indices `start_fr:end_fr` and then keeping the same `actual_frames` length through discretization.

ii. ```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
```

iii. The trial-start-aligned frame slicing in the notes applies to speed in the same way as neural data, licking, and position.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script handles data issues with small defensive checks rather than explicit correction. Float frame indices are rounded to integers. Trials with invalid or too-short slices are skipped. NaN speed samples are removed only when computing global speed percentiles. Bin assignments are clipped to valid category ranges. Licks that land outside the extracted trial window are ignored. There is no general imputation for missing values.

ii. ```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr = int(np.round(beh['GrayFr'][trial_idx]))

if avail_frames < 2:
    return None
if actual_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None

flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
bins = np.clip(bins, 0, N_SPEED_BINS - 1)
```

iii. The notes mention rounded frame indices and skipped short trials as explicit edge-case handling, and the trajectory shows the agent saw very long or odd trials but chose not to perform more aggressive cleaning.

## 12-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are the dataset-wide passes over all sessions and trials, the session-scale loading of massive neural matrices from `load_spk`, the per-trial extraction of large `(neurons x time)` arrays, and the final pickle serialization of the huge output structure.

ii. ```python
for i, sess_id in enumerate(session_ids):
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)

for i, sess_id in enumerate(session_ids):
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)

spk = load_spk(mname, datexp, blk)

with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The notes say full conversion created an extremely large pickle, and the trajectory repeatedly comments on long conversion, save, verification-load, and training startup times.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The pass-1 trial loop over `StartFr:GrayFr`, the pass-2 trial loop that repeatedly builds per-trial arrays, the per-lick loop inside each trial, and the repeated loop over all `exp_info` sessions in `get_training_day` are the clearest vectorization candidates.

ii. ```python
for t in range(ntrials):
    start_fr = int(np.round(beh['StartFr'][t]))
    gray_fr = int(np.round(beh['GrayFr'][t]))
    trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]

for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)

for lf in trial_lick_frs:
    lf_int = int(np.round(lf))

for exp_type, sessions in exp_info.items():
    for s in sessions:
        if s['mname'] == mname:
            dates.add(s['datexp'])
```

iii. The trajectory frames these loops as the expensive parts of processing and discusses large trial counts, so these are the agent's main efficiency weak points.

## 12-c. What processing does the code repeat multiple times?

i. The code scans the dataset twice: once to collect all running speeds for quartile thresholds and again to build final trials. It also repeatedly reloads behavior files per session, recomputes training-day date sets for every session of a mouse, and re-derives trial boundaries separately in pass 1 and pass 2.

ii. ```python
for i, sess_id in enumerate(session_ids):
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)

for i, sess_id in enumerate(session_ids):
    result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)

training_day = get_training_day(mname, datexp, exp_info)
```

iii. The notes explicitly describe the two-pass design, and the trajectory says this first pass was added solely for quartile computation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script collects all behavior variants for a session but discards every variant except `beh_results[0]`. It also defines helper functions that are never used in the final path, such as `build_session_to_beh_map`, `get_brain_region_indices`, and `get_stimulus_category`. Inside `process_session`, `n_total_frames` is computed but not used.

ii. ```python
beh_results = load_beh_for_session(sess_id, exp_types)
beh, beh_key = beh_results[0]

def build_session_to_beh_map(exp_info):
    ...

def get_brain_region_indices(iarea, brain_regions):
    ...

def get_stimulus_category(wall_name):
    return wall_name

n_total_frames = spk.shape[1]
```

iii. The trajectory shows the swap-variant logic was added late and never fully integrated into session creation. The unused helpers and unused local variables are visible directly in the final script.
