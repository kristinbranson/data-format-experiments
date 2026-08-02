# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads experiment metadata from `Imaging_Exp_info.npy`, builds a deduplicated list of physical sessions, loads one behavior dictionary per experiment type, loads neural data from `data/spk/*_neural_data.npy`, loads retinotopy from `data/retinotopy/*_trans.npz`, and then processes every trial inside each session with `process_session()`.

ii. ```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
).item()

beh_cache[et] = load_beh(et)
spk = load_spk(session['db_entry'])
iarea = load_retino(session['db_entry'])
neural_trials, input_trials, output_trials, brain_reg_idx, elapsed = \
    process_session(session, beh, day, speed_quartiles, stim_to_idx, frame_period)
```

iii. In `CONVERSION_NOTES.md`, the agent says the same physical session appears in multiple experiment types and should be included once, and that the decoder dataset should therefore be built from the 89 unique `(mname, datexp, blk)` sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by unique mouse name `mname`. The subject list is `sorted(set(s['mname'] for s in sessions))`, and each session gets a `subject_idx` from that mapping.

ii. ```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx_list.append(subject_to_idx[session['mname']])
```

iii. The notes say the dataset contains 19 unique mice and that subject identity should be the mouse name used throughout the original metadata.

## 1-c. How are the data split into sessions?

i. Sessions are defined as unique `(mname, datexp, blk)` tuples from `Imaging_Exp_info.npy`. If the same physical session appears under multiple experiment types, later duplicates are skipped.

ii. ```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
...
sessions.sort(key=lambda s: (s['mname'], s['datexp']))
```

iii. The notes explicitly justify this as a deduplication step: the agent believed repeated appearances across experiment types were different analysis views of the same recording and should not become duplicated sessions in the decoder dataset.

## 1-d. How are the data split into trials?

i. Within each session, trials are split using behavioral frame indices `StartFr[t]` and `EndFr[t]`. Each trial becomes the frame slice `start:end` of neural and behavioral arrays.

ii. ```python
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)
...
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
    neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say the chosen trial window is `StartFr` to `EndFr`, which the agent interpreted as corridor entry through the end of the trial.

## 1-e. How are trials filtered based on quality controls?

i. The agent only applies basic validity checks: skip trials whose frame bounds are invalid, clip `end` to available behavior-array length, and skip trials shorter than 2 timepoints. There is no reference-style filtering to running-only frames or corridor-only frames.

ii. ```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1
    continue

end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue

n_tp = end - start
if n_tp < 2:
    skipped += 1
    continue
```

iii. The notes frame this as format/QC protection for malformed trials. They do not cite a paper- or code-based trial curation rule beyond keeping trials usable for the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the Suite2p deconvolved traces stored as the `spks` list inside each `*_neural_data.npy` file.

ii. ```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```

iii. The notes say deltaF/F is not computed because the reference data already contain deconvolved traces and the paper states analyses used deconvolved fluorescence traces.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes, leaves the traces otherwise raw, slices each trial by frame index, and casts the result to `float16` to reduce memory. It does not apply the reference code’s running filter or position interpolation.

ii. ```python
spk = load_spk(session['db_entry'])
...
neural = spk[:, start:end].astype(np.float16)
# Using float16 to reduce memory
```

iii. In the notes, the agent explicitly chose “raw frame-level data (~3.17 Hz)” and described that as a decoder-oriented simplification, despite also documenting that the reference analyses primarily use position-interpolated activity.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is no neuron-quality filtering. The only checks are that retinotopy length matches neuron count and that trials have valid frame bounds. Neurons outside named visual areas are retained and labeled as `other`.

ii. ```python
assert len(iarea) == n_neurons, \
    f"Neuron count mismatch: spk={n_neurons}, retinotopy={len(iarea)}"

idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
for area_val, region_name in AREA_MAP.items():
    mask = iarea == area_val
    idx[mask] = region_to_idx[region_name]
```

iii. The notes justify this by saying there is “no explicit quality filtering of neurons” in the reference and that, for the decoder, all neurons should be included and the model should learn which are relevant.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. It is aligned by trial start (`StartFr`) in raw frame time. Trial timepoint 0 is the first frame at corridor entry, and the trial runs to `EndFr`.

ii. ```python
start = StartFr[t]
end = EndFr[t]
...
neural = spk[:, start:end].astype(np.float16)
...
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
```

iii. The notes repeatedly state that the dataset is “temporally aligned based on trial start (corridor entry)” and that this motivated the raw frame-based representation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin is the native imaging frame period, estimated from one session’s `ft` timestamps by taking the median frame-to-frame interval. No temporal rebinning is applied.

ii. ```python
ft = sample_beh['ft']
dt = np.diff(ft) * 24 * 3600
frame_period = float(np.nanmedian(dt))
...
'time_bin_size': frame_period * 1000,
```

iii. The notes say “Time bin = frame rate ~315 ms, no additional binning” and justify this as preserving the raw frame-level representation.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from trial-level `SoundFr` and the per-frame indices between `StartFr` and `EndFr`.

ii. ```python
SoundFr = beh['SoundFr']
...
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The notes map “SoundFr - current_frame” to the decoder input `time_to_sound_cue`.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each trial, the agent constructs a frame index vector and computes signed seconds relative to the cue. Values are positive before the cue and negative after it.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The notes explicitly describe this variable as continuous seconds to the cue, using the sign convention “positive before, negative after.”

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It uses the exact same `start:end` frame grid as the neural slice, so it is framewise aligned to the trial-start-aligned neural matrix.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0)
```

iii. The justification is implicit in the code and notes: all time-varying inputs are generated on the same per-trial frame grid as the neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata, specifically mouse name `mname` and session date `datexp`, not from a direct behavioral time-series variable.

ii. ```python
mouse_sessions[s['mname']].append((s['datexp'], i))
...
sess_list.sort(key=lambda x: x[0])
for day_idx, (datexp, global_idx) in enumerate(sess_list):
    days[global_idx] = float(day_idx)
```

iii. The notes justify this as “ordinal session index (by date) within each mouse.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The agent sorts each mouse’s sessions chronologically, assigns day indices `0, 1, 2, ...`, and then broadcasts the session’s scalar day value across all timepoints in each trial.

ii. ```python
training_days = compute_training_days(sessions)
...
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. The notes say this should be a continuous per-trial variable and therefore use a constant trial-wide broadcast of the chronological session index.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. The agent does not create an `Environment type` input at all. The final dataset only contains four inputs: time to sound cue, day of training, time since trial start, and reward availability.

ii. ```python
'input_names': ['time_to_sound_cue', 'day_of_training',
               'time_since_trial_start', 'reward_availability'],
```

iii. The notes justify the chosen input set by mirroring the decoder-input list from the instructions; they do not mention adding environment type as an extra input.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. There is no code path that computes, stores, or names an environment-type variable.

ii. ```python
inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0).astype(np.float32)
```

iii. The omission appears intentional: the notes’ “Variable Mapping” section lists only the four instructed decoder inputs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and the frame indices spanning each trial.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes describe this as `(frame_idx - StartFr) * frame_period_sec`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent converts frame offset from trial start into seconds, then stores that continuous trajectory as the third input channel.

ii. ```python
time_since_start = (frame_indices - start) * frame_period
inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0)
```

iii. The notes justify it as a time-varying input aligned to trial start, with no additional smoothing or binning.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the exact same `start:end` frame grid used for the neural slice, so alignment is one value per neural frame.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The justification is the same trial-start frame alignment used throughout the script.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial behavioral flag `isRew`.

ii. ```python
isRew = beh['isRew']
...
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes explicitly map `isRew` to reward availability and define it as `1` in rewarded corridors and `0` otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The agent turns each trial’s `isRew[t]` into a constant float vector of length `n_tp`, so reward availability is a trial-wide time-varying channel with the same constant value at every frame.

ii. ```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0)
```

iii. The notes justify this as a per-trial discrete variable broadcast across the trial to match decoder input shape.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`, with a session-global category mapping built from all unique `UniqWalls` entries seen across sessions.

ii. ```python
for wn in beh['UniqWalls']:
    all_stim.add(str(wn))
...
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
```

iii. The notes map `WallName` to the visual-stimulus output and state that the agent keeps all unique stimulus names as categories.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent sorts all unique stimulus names, maps each trial’s `WallName` to an integer index, and then broadcasts that index across all timepoints in the trial.

ii. ```python
all_stimuli = get_all_stimuli(sessions, beh_cache)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
...
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The notes justify this as a per-trial categorical output. They also note that 15 unique stimuli were kept, rather than collapsing categories.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`, using the trial frame boundaries `StartFr` and `EndFr`.

ii. ```python
lick_frs = beh['LickFr']
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```

iii. The notes justify this as a binary per-frame lick vector, with lick times assigned to the nearest frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent rounds fractional lick-frame timestamps to integers, selects licks that fall inside the trial window, clips them to valid indices, and writes `1` into a binary vector at those frame positions.

ii. ```python
lick_vec = np.zeros(n_frames, dtype=np.int64)
lick_frs_int = np.round(lick_frs).astype(int)
...
trial_lick_frs = lick_frs_int[mask] - start_fr
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
lick_vec[trial_lick_frs] = 1
```

iii. The notes explicitly mention “Round `LickFr` to nearest int frame, create binary vector per trial.”

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned on the same raw frame grid as the neural slice for that trial: one lick value per neural frame from `start` to `end`.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
lick = make_lick_vector(beh, start, end)
out = np.stack([stim_out, lick, pos_bin, speed_bin], axis=0)
```

iii. The justification is implicit in the shared `start:end` indexing used for both neural activity and outputs.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the frame-level behavioral position trace `ft_Pos`.

ii. ```python
ft_Pos = beh['ft_Pos']
...
pos = ft_Pos[start:end]
```

iii. The notes map `ft_Pos` directly to the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent slices `ft_Pos` over the trial’s frame window and discretizes each frame’s position value with `np.digitize`.

ii. ```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)
...
pos_bin = digitize_position(pos)
```

iii. The notes justify this as four 1 m bins over the 4 m texture corridor, but also state that the last bin includes gray space.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The thresholds are `[10, 20, 30]` dm, producing categories `[0,1,2,3]` corresponding to `[0,10)`, `[10,20)`, `[20,30)`, and `[30,60+]`.

ii. ```python
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]
POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
...
bins = np.digitize(pos, [10, 20, 30])
```

iii. The notes explicitly say “Position bin 3 includes gray space: [30,60) dm covers 3-4 m texture + 2 m gray.”

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. It is taken from the same `start:end` frame slice as the neural data, so each neural frame gets one position-bin label.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. The agent’s justification is the same raw framewise trial alignment used across all variables.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the frame-level behavioral variable `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed']
...
speed = ft_RunSpeed[start:end]
```

iii. The notes map `ft_RunSpeed` directly to running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first computes global quartile cutoffs across all framewise `ft_RunSpeed` values from all sessions, then slices each trial’s speed trace and digitizes it against those quartiles.

ii. ```python
all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes justify this as “Speed quartiles: computed globally across all frames in all sessions.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. It is thresholded by the global 25th, 50th, and 75th percentiles of `ft_RunSpeed`, producing four bins labeled `Q1_slow`, `Q2`, `Q3`, and `Q4_fast`.

ii. ```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
speed_labels = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']
bins = np.digitize(speed, quartiles)
```

iii. The justification in the notes is that the decoder instructions asked for running speed discretized into four bins corresponding to 25% of the data.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. It is aligned framewise using the same `start:end` slice as the neural data.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes imply shared alignment by building every time-varying variable on the same per-trial frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent handles malformed trials conservatively: it skips trials with bad indices or fewer than 2 timepoints, clips `end` to available behavior-array length, asserts retinotopy/neuron consistency, rounds lick frames, and clips lick indices. It does not impute missing values or explicitly handle NaNs beyond using `nanmedian` for frame period.

ii. ```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
...
if n_tp < 2:
    skipped += 1
    continue
...
dt = np.diff(ft) * 24 * 3600
frame_period = float(np.nanmedian(dt))
```

iii. The notes treat these as sanity checks for format robustness rather than as scientifically motivated curation.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive steps are loading the very large `spk` arrays, iterating trial-by-trial through every session, constructing per-trial neural/input/output arrays, and serializing the final very large pickle.

ii. ```python
spk = load_spk(session['db_entry'])
...
for t in range(ntrials):
    ...
    neural_trials.append(neural)
...
pickle.dump(data, f, protocol=5)
```

iii. The trajectory and notes emphasize runtime and memory repeatedly, including estimates of minutes per full conversion and hundreds of GB of output.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `process_session()`, the repeated call to `make_lick_vector()` for every trial, and the per-session loop used to gather all running speeds for quartiles are the clearest vectorization targets. Trial-wide broadcasts of day/reward and the repeated `np.arange(start, end)` construction are also scalarized inside Python loops.

ii. ```python
for t in range(ntrials):
    ...
    frame_indices = np.arange(start, end, dtype=np.float64)
    ...
    lick = make_lick_vector(beh, start, end)
```

iii. The notes mention memory pressure and runtime concerns but do not report a vectorized rewrite; the implementation remains mostly Python-loop based.

## 12-c. What processing does the code repeat multiple times?

i. The code repeatedly constructs per-trial frame-index arrays, repeatedly broadcasts scalar trial labels across timepoints, repeatedly digitizes position and speed one trial at a time, and repeatedly scans lick timestamps against each trial window.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
day = np.full(n_tp, training_day, dtype=np.float32)
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
pos_bin = digitize_position(pos)
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. This repetition is not explicitly justified in the notes; it follows from the agent’s simple trial-by-trial implementation strategy.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script optionally produces diagnostic plots, computes memory summaries during conversion, builds the unused `subjects_list` variable, and includes gray-space frames in the final trial representation even though the reference paper/code mainly analyze running in the texture corridor. It also stores raw-frame neural data at huge size rather than a more compact reference-style representation.

ii. ```python
subjects_list = []
...
if args.show_processing and i < 2:
    plot_processing(...)
...
if (i + 1) % 10 == 0:
    gc.collect()
    total_neural_mb = ...
```

iii. The notes justify the plots and memory tracking as debugging/sanity checks. They do not justify the extra gray-space inclusion beyond the agent’s own decision to keep the full `StartFr:EndFr` window.
