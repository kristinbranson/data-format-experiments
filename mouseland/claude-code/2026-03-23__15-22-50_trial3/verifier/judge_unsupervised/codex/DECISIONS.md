# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script first loads `/app/data/beh/Imaging_Exp_info.npy`, builds a unique session table from the experiment metadata, then loads only the needed behavior files (`Beh_{exp_type}.npy`) into a cache. For each session it loads the neural file from `data/spk/` and the retinotopy file from `data/retinotopy/`, then iterates through all trials in that behavior record.

ii. ```python
def build_session_list():
    exp_info = np.load(
        os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
    ).item()

    seen = set()
    sessions = []

    for exp_type in exp_info:
        for db in exp_info[exp_type]:
            key = (db['mname'], db['datexp'], db['blk'])
            if key in seen:
                continue
            seen.add(key)
            ...
            sessions.append({
                'mname': db['mname'],
                'datexp': db['datexp'],
                'blk': db['blk'],
                'exp_type': exp_type,
                'db_entry': db,
                'beh_key': beh_key,
            })

exp_types_needed = set(s['exp_type'] for s in sessions)
for et in exp_types_needed:
    beh_cache[et] = load_beh(et)

for i, session in enumerate(sessions):
    beh = beh_cache[session['exp_type']][session['beh_key']]
    neural_trials, input_trials, output_trials, brain_reg_idx, elapsed =                     process_session(session, beh, day, speed_quartiles, stim_to_idx, frame_period)
```

iii. In `CONVERSION_NOTES.md`, the agent says the same physical session appears in multiple experiment types and therefore should be included once. The trajectory shows it explicitly chose to deduplicate by physical session and treat the duplicate behavior files as alternative views of the same recording.

## 1-b. How are the data split into subjects?

i. Subjects are defined entirely by mouse name (`mname`). The script builds `subjects = sorted(set(s["mname"] for s in sessions))` and a `subject_idx` entry per session.

ii. ```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
...
subject_idx_list.append(subject_to_idx[session['mname']])
```

iii. The notes describe “19 unique mouse names” and use mouse name as the subject identity across sessions.

## 1-c. How are the data split into sessions?

i. A session is defined as a unique `(mname, datexp, blk)` tuple. After deduplication, sessions are sorted by `(mname, datexp)` and each one becomes one entry in the output lists.

ii. ```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
...
sessions.sort(key=lambda s: (s['mname'], s['datexp']))
```

iii. The agent documented that the same physical session can appear in multiple experiment types, and decided to treat those as one recording session for decoding.

## 1-d. How are the data split into trials?

i. Within each session, trials are taken from the behavior record using `ntrials`, `StartFr`, and `EndFr`. Each trial is the raw frame slice `[StartFr[t], EndFr[t])`, so trials have variable length and include everything between corridor entry and corridor exit.

ii. ```python
ntrials = beh['ntrials']
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)
...
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
    ...
    n_tp = end - start
    ...
    neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say “Trial window: StartFr to EndFr (corridor + gray space)” and the trajectory shows the agent chose trial-start alignment with variable-length trials instead of forcing a fixed size.

## 1-e. How are trials filtered based on quality controls?

i. There is only minimal trial QC. The script skips trials with out-of-bounds frame indices, `end <= start`, or fewer than 2 time points after clipping `end` to the available behavioral arrays. It does not filter trials by running, corridor occupancy, or behavioral quality.

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

iii. The notes emphasize that the agent chose to keep all trial frames for decoder continuity and only added safety checks for malformed windows.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the `spks` entry in each `{mouse}_{date}_{blk}_neural_data.npy` file. Retinotopy `iarea` is loaded separately only to populate `brain_region_idx`.

ii. ```python
def load_spk(db):
    fn = '%s_%s_%s_neural_data.npy' % (db['mname'], db['datexp'], db['blk'])
    spk_path = os.path.join(DATA_ROOT, 'spk', fn)
    spk = np.concatenate(
        [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
    )
    return spk
```

iii. The notes explicitly state the data are already Suite2p deconvolved traces and that the neural array comes from concatenating the stored `spks` planes.

## 2-b. How is the `neural` data processed?

i. The script concatenates imaging planes, slices raw frame-level trial segments, and casts each trial matrix to `float16` for memory savings. It does not compute dF/F, re-deconvolve, normalize, smooth, or perform the reference code’s position interpolation to 60 bins.

ii. ```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
...
neural = spk[:, start:end].astype(np.float16)
# Using float16 to reduce memory (decoder uses PCA, so precision is fine)
```

iii. `CONVERSION_NOTES.md` says the traces are already deconvolved and that the agent deliberately used raw frame-level data instead of the reference position-interpolated representation.

## 2-c. How is the `neural` data filtered based on quality controls?

i. There is effectively no neural quality filtering. All neurons are kept, including `iarea == -1` and `iarea == 7`, which are grouped into an `other` region instead of excluded. The code also keeps stationary and gray-space frames inside the trial window.

ii. ```python
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
...
neural = spk[:, start:end].astype(np.float16)
```

iii. The agent explicitly documented “All neurons included” and “No ft_move filtering” as deliberate differences from the reference analyses.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial’s neural matrix starts at `StartFr[t]`, which the script interprets as trial start / corridor entry, and ends at `EndFr[t]`.

ii. ```python
start = StartFr[t]
end = EndFr[t]
...
neural = spk[:, start:end].astype(np.float16)
...
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
```

iii. The notes repeatedly state “Temporally aligned based on trial start (corridor entry)” and that the trial window is `StartFr` to `EndFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data stay at the original imaging frame resolution. The script estimates a single global frame period from the first session’s `ft` timestamps and uses that for all sessions; no temporal rebinning is applied.

ii. ```python
sample_beh = beh_cache[sessions[0]['exp_type']][sessions[0]['beh_key']]
ft = sample_beh['ft']
dt = np.diff(ft) * 24 * 3600  # datenum to seconds
frame_period = float(np.nanmedian(dt))
...
'time_bin_size': frame_period * 1000,
```

iii. The notes say the raw data are at ~3.17 Hz (~315 ms) and that the agent chose “Time bin = frame rate: ~315 ms, no additional binning.”

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from the per-trial sound frame index `SoundFr[t]`, the trial frame indices from `StartFr[t]:EndFr[t]`, and the estimated frame period from `ft`.

ii. ```python
SoundFr = beh['SoundFr']
...
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The notes map this input to `SoundFr - current_frame`, converted from frames to seconds using the frame period.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in the trial, the code subtracts the current absolute frame index from the absolute cue frame and multiplies by seconds per frame. The sign convention is positive before cue and negative after cue.

ii. ```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period  # positive before, negative after
```

iii. The trajectory explicitly mentions this sign convention and checks that `SoundFr` is an absolute per-trial frame index.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same `start:end` frame indices used to slice the neural data, so it has one value per neural frame in the trial.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_to_sound = (sound_fr - frame_indices) * frame_period
...
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say all time-varying inputs are built directly on the frame axis of the neural trial window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The code derives it only from session metadata: mouse name `mname` and recording date `datexp`. It does not use the explicit `sess#` or `days` fields mentioned in the metadata notes.

ii. ```python
def compute_training_days(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((s['datexp'], i))
```

iii. The notes say the agent considered `sess#` / `days` but ultimately chose chronological session order within each mouse as the cleanest universal rule.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Sessions are grouped by mouse, sorted by recording date, and assigned ordinal values `0, 1, 2, ...` within each mouse. That scalar is then broadcast across every frame of every trial in the session.

ii. ```python
for mname, sess_list in mouse_sessions.items():
    sess_list.sort(key=lambda x: x[0])
    for day_idx, (datexp, global_idx) in enumerate(sess_list):
        days[global_idx] = float(day_idx)
...
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. This exact procedure is described in the notes under the day-of-training mapping.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr[t]`, the absolute frame indices in the trial window, and the global frame period estimated from `ft`.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes map this input to `(frame_idx - StartFr) * frame_period`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each frame, the code subtracts the trial’s start frame and converts the result from frames to seconds.

ii. ```python
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes describe it as a straightforward elapsed-time-from-start feature.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed over the same `start:end` frames as the neural data, beginning at 0 seconds on the first neural frame of the trial.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes state that all time-varying inputs are framewise and aligned to the neural slices.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean `isRew` in the behavior data.

ii. ```python
isRew = beh['isRew']
...
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes explicitly map reward availability to `isRew` with 1 for rewarded corridor and 0 otherwise.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The code converts `isRew[t]` to float and broadcasts it across all time points of the trial.

ii. ```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes describe this as a discrete per-trial input rather than a time-varying event signal.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName` for each trial, with the global category vocabulary built from `UniqWalls` across sessions.

ii. ```python
def get_all_stimuli(sessions, beh_cache):
    all_stim = set()
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        for wn in beh['UniqWalls']:
            all_stim.add(str(wn))
    return sorted(all_stim)
...
WallName = beh['WallName']
```

iii. The notes map visual stimulus directly to `WallName` and mention 15 unique stimulus strings in the full conversion.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The script collects all unique stimulus names, sorts them, maps each name to an integer, and then broadcasts the trial’s category index across every frame in that trial.

ii. ```python
all_stimuli = get_all_stimuli(sessions, beh_cache)
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
...
stim_name = str(WallName[t])
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. The notes describe the stimulus output as per-trial but broadcast over time so every output shares the same per-frame shape.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. In the final code it is derived from `LickFr` alone. Although `LickTrind` exists in the raw data, the script does not use it when constructing per-trial licking vectors.

ii. ```python
lick_frs = beh['LickFr']
# Round to nearest integer frame
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. The trajectory explicitly notes that the code uses frame-range matching on `LickFr` rather than the trial labels in `LickTrind`.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The code rounds fractional lick frame indices to the nearest integer, keeps only licks whose integer frame lies within the current trial window, shifts them into trial-relative coordinates, clips to valid indices, and writes 1s into a binary vector.

ii. ```python
def make_lick_vector(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_vec = np.zeros(n_frames, dtype=np.int64)

    lick_frs = beh['LickFr']
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        trial_lick_frs = lick_frs_int[mask] - start_fr
        trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
        lick_vec[trial_lick_frs] = 1
```

iii. The notes say “Round LickFr to nearest int frame, create binary vector per trial,” and the trajectory mentions the edge case that this could differ slightly from using `LickTrind` directly.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The licking vector is created over the same `[start:end)` frame window as the neural slice, with lick times converted to trial-relative frame offsets from `start`.

ii. ```python
lick = make_lick_vector(beh, start, end)
...
neural = spk[:, start:end].astype(np.float16)
```

iii. The code and notes both describe licking as a framewise output aligned to the neural trial window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from the framewise position trace `ft_Pos` in the behavior data.

ii. ```python
ft_Pos = beh['ft_Pos']
...
pos = ft_Pos[start:end]
```

iii. The notes map the position output directly to `ft_Pos` and document the corridor as 0–60 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code takes the raw per-frame position values from `ft_Pos[start:end]` and passes them to `np.digitize` using fixed edges at 10, 20, and 30 dm.

ii. ```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)
...
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. The notes record the intended 1 m bins and the later choice to collapse gray-space frames into the last category.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The categories are `0:[0,10)`, `1:[10,20)`, `2:[20,30)`, and `3:[30,60+)` in decimeters. In practice, the last category merges the final 1 m of texture corridor with the entire 2 m gray space.

ii. ```python
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]
POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
...
bins = np.digitize(pos, [10, 20, 30])
```

iii. The trajectory shows the agent debated what to do with the 40–60 dm gray space and finally chose to fold it into the last position bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is aligned frame by frame by slicing `ft_Pos` over the same `[start:end)` neural trial window.

ii. ```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes treat position as a time-varying output defined on the neural frame axis.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from the framewise running-speed trace `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed']
...
speed = ft_RunSpeed[start:end]
```

iii. The notes explicitly map running speed to `ft_RunSpeed`.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The script first concatenates `ft_RunSpeed` from all selected sessions, computes global 25th/50th/75th percentile thresholds, then slices each trial’s speed trace and digitizes it with those thresholds.

ii. ```python
def compute_speed_quartiles(sessions, beh_cache):
    all_speeds = []
    for s in sessions:
        beh = beh_cache[s['exp_type']][s['beh_key']]
        speeds = beh['ft_RunSpeed']
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
...
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes say “Speed quartiles: Computed globally across all frames in all sessions,” which is exactly what the code does.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the global quartiles returned by `np.percentile(all_speeds, [25, 50, 75])`; `np.digitize` then assigns categories 0–3 relative to those cut points.

ii. ```python
def digitize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```

iii. The trajectory even double-checks the global quartile values and notes some imbalance because many frames have identical speeds at the quartile boundary.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned frame by frame by slicing `ft_RunSpeed` over the same `[start:end)` window as the neural trial.

ii. ```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes treat speed as another time-varying output on the neural frame axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses only light defensive handling: it asserts that retinotopy and neural neuron counts match, clips `end` to the available behavioral arrays, skips invalid or very short trials, rounds and clips lick indices, and otherwise proceeds without explicit NaN repair or special handling for cue times outside a trial.

ii. ```python
assert len(iarea) == n_neurons,                 f"Neuron count mismatch: spk={n_neurons}, retinotopy={len(iarea)}"
...
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
...
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
```

iii. The notes describe these as sanity/safety checks rather than a full missing-data policy, and the trajectory says there was no explicit validation for out-of-window `SoundFr`.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are loading and slicing the enormous session-wide spike matrices for all 89 sessions, iterating through every trial to build neural/input/output arrays, and serializing the final 202 GB pickle. Computing global speed quartiles also requires scanning all framewise speed traces, but that is smaller than the neural I/O and per-trial extraction.

ii. ```python
for i, session in enumerate(sessions):
    ...
    neural_trials, input_trials, output_trials, brain_reg_idx, elapsed =                     process_session(session, beh, day, speed_quartiles, stim_to_idx, frame_period)
    ...
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=5)
```

iii. The notes report ~33 minutes for full conversion and repeatedly emphasize file size/memory pressure. The per-session output log prints elapsed time and cumulative neural memory, which highlights the same bottlenecks.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial Python loop in `process_session()` is the main vectorization target: it repeatedly allocates `np.arange`, `np.full`, slices raw arrays, digitizes position/speed, and builds lick vectors trial by trial. `make_lick_vector()` also re-filters the full lick list for each trial. Smaller loops like `get_brain_region_idx()` over area values and `compute_training_days()` over sessions are also vectorizable but less important.

ii. ```python
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
    ...
    frame_indices = np.arange(start, end, dtype=np.float64)
    ...
    lick = make_lick_vector(beh, start, end)
    pos = ft_Pos[start:end]
    pos_bin = digitize_position(pos)
    speed = ft_RunSpeed[start:end]
    speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. This follows directly from the code structure and from the notes’ emphasis on per-session runtime and memory use.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly rebuilds trial-relative frame vectors, broadcasts constants (`day`, `reward`, `stimulus`) for every trial, scans the session-wide lick list for every trial, and slices/digitizes position and speed one trial at a time. At the session level it also reconstructs the region-to-index mapping inside `get_brain_region_idx()` each call.

ii. ```python
frame_indices = np.arange(start, end, dtype=np.float64)
...
day = np.full(n_tp, training_day, dtype=np.float32)
...
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
...
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
...
lick = make_lick_vector(beh, start, end)
```

iii. This is visible in `process_session()` and explains part of the runtime overhead noted in the conversion logs.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion stores every neuron from every session even though the downstream decoder later subsamples or projects high-dimensional neural activity, so a large amount of converted neural data is not used directly during training. Within `convert_data.py` itself, optional plotting, memory accounting, and summary statistics are extra work that are not part of the final decoder inputs/outputs.

ii. ```python
neural = spk[:, start:end].astype(np.float16)
# Using float16 to reduce memory (decoder uses PCA, so precision is fine)
...
if args.show_processing and i < 2:
    plot_processing(...)
...
neural_mb = sum(t.nbytes for t in neural_trials) / 1e6
print(f"    {n_trials} trials, {n_neurons} neurons, mean {mean_tp:.0f} frames/trial, {neural_mb:.0f}MB, {elapsed:.1f}s")
```

iii. `CONVERSION_NOTES.md` states that full decoder training later subsampled neurons to 3000 per session to fit memory. That means much of the saved full-resolution neural data is not used directly by the downstream training step.
