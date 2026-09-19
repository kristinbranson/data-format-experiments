# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `data/beh/Imaging_Exp_info.npy`, deduplicates physical sessions by `(mname, datexp, blk)`, loads the needed behavior dictionaries `Beh_<exp_type>.npy` into a cache, and then loads spike and retinotopy files per session during processing.

ii. 
```python
exp_info = np.load(
    os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True
).item()
```
```python
fn = os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type)
return np.load(fn, allow_pickle=True).item()
```
```python
spk_path = os.path.join(DATA_ROOT, 'spk', fn)
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md` the AI says the dataset should be loaded once per physical session and that the same session can appear in multiple experiment types, so it should be paired with one behavior file and one spike file.

## 1-b. How are the data split into subjects?

i. Sessions are assigned to subjects by mouse name `mname`. The final `subjects` list is `sorted(set(...))`, and each kept session gets a `subject_idx` from that mapping.

ii.
```python
subjects = sorted(set(s['mname'] for s in sessions))
subject_to_idx = {s: i for i, s in enumerate(subjects)}
```
```python
subject_idx_list.append(subject_to_idx[session['mname']])
```

iii. The notes say there are 19 unique mice and repeatedly describe subject identity as coming directly from the session metadata and filenames.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` tuple. Duplicate appearances across experiment types are skipped after the first occurrence.

ii.
```python
key = (db['mname'], db['datexp'], db['blk'])
if key in seen:
    continue
seen.add(key)
```
```python
sessions.append({
    'mname': db['mname'],
    'datexp': db['datexp'],
    'blk': db['blk'],
    'exp_type': exp_type,
    'db_entry': db,
    'beh_key': beh_key,
    'session_key': key,
})
```

iii. The notes explicitly justify "use each physical session once" because the same recording can appear in several experiment-type groupings.

## 1-d. How are the data split into trials?

i. The AI splits trials with contiguous frame spans `StartFr[t]:EndFr[t]`. That means each trial includes corridor frames plus gray-space frames, instead of using `ft_trInd` and `ft_CorrSpc`.

ii.
```python
StartFr = beh['StartFr'].astype(int)
EndFr = beh['EndFr'].astype(int)
```
```python
for t in range(ntrials):
    start = StartFr[t]
    end = EndFr[t]
```

iii. In the trajectory and notes, the AI says its chosen trial window is "StartFr to EndFr (corridor + gray space)" and treats that as the trial segment for decoder inputs and outputs.

## 1-e. How are trials filtered based on quality controls?

i. Trials are only skipped if they are out of bounds, have non-positive length after clipping, or have fewer than 2 timepoints. The AI does not apply the reference 99th-percentile long-trial filter.

ii.
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1
    continue
```
```python
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
...
if n_tp < 2:
    skipped += 1
    continue
```

iii. The notes say "reference code uses all trials in a session" and the AI's own validation notes emphasize that there were no skipped trials in the full conversion, so it deliberately used only minimal integrity checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from concatenated `spks` arrays in the per-session spike file, and brain-region labels come from `iarea` in the retinotopy file.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. The notes say the data are already Suite2p deconvolved traces and do not need dF/F or deconvolution.

## 2-b. How is the `neural` data processed?

i. The AI concatenates planes, then slices a contiguous per-trial block from `start:end` and stores it as `float16`. It does not interpolate, pad, or otherwise transform the neural signal.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
neural = spk[:, start:end].astype(np.float16)
```

iii. The trajectory says the AI wanted raw frame-level data at about 3.17 Hz and later changed the storage dtype to `float16` to reduce memory.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not exclude neurons by area. It keeps all neurons and maps `iarea` values `-1` and `7` into an `"other"` region instead of dropping them.

ii.
```python
AREA_MAP = {
    8: 'V1',
    0: 'mHV', 1: 'mHV', 2: 'mHV', 9: 'mHV',
    5: 'lHV', 6: 'lHV',
    3: 'aHV', 4: 'aHV',
    -1: 'other', 7: 'other',
}
```
```python
idx = np.full(len(iarea), region_to_idx['other'], dtype=np.int64)
for area_val, region_name in AREA_MAP.items():
    mask = iarea == area_val
    idx[mask] = region_to_idx[region_name]
```

iii. The notes repeatedly justify this as "include ALL neurons" so the decoder can learn which neurons matter, and list this as a deliberate difference from the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry by starting each neural segment at `StartFr`, but the AI keeps the segment through `EndFr`, so the aligned window extends through the gray space.

ii.
```python
StartFr = beh['StartFr'].astype(int)
...
start = StartFr[t]
end = EndFr[t]
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say the temporal alignment event is trial start / corridor entry, and the mapping plan explicitly records the chosen trial window as `StartFr to EndFr`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses one imaging frame per bin, with a single `frame_period` estimated from the median frame interval of the first sample session. No temporal rebinning is applied.

ii.
```python
dt = np.diff(ft) * 24 * 3600
frame_period = float(np.nanmedian(dt))
```
```python
'time_bin_size': frame_period * 1000,
```

iii. The notes say "Time bin = frame rate: ~315ms, no additional binning" and describe the conversion as using raw frame-level data.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives it from `SoundFr` and the integer frame indices for the chosen trial window, together with a session-level `frame_period`.

ii.
```python
SoundFr = beh['SoundFr']
```
```python
sound_fr = SoundFr[t]
frame_indices = np.arange(start, end, dtype=np.float64)
```

iii. The mapping notes state `SoundFr - current_frame`, multiplied by frame period in seconds, was the intended construction.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every trial, the AI computes `(sound_fr - frame_indices) * frame_period`, yielding positive values before the cue and negative values after the cue. It does not interpolate onto actual `ft` timestamps.

ii.
```python
time_to_sound = (sound_fr - frame_indices) * frame_period
```

iii. The trajectory says the AI intentionally used raw frame-rate timing and a single frame period from MATLAB datenums rather than per-frame interpolation.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is generated on the exact same `start:end` frame index grid that is used to slice the neural matrix for that trial.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
...
neural = spk[:, start:end].astype(np.float16)
inp = np.stack([time_to_sound, day, time_since_start, rew], axis=0)
```

iii. The AI's notes describe all converted signals as raw frame-level time series sharing the same per-trial frame window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Day of training is derived from the ordering of sessions by `mname` and `datexp`. The AI does not use the `days` or `sess#` fields from the metadata.

ii.
```python
mouse_sessions[s['mname']].append((s['datexp'], i))
```
```python
sess_list.sort(key=lambda x: x[0])
for day_idx, (datexp, global_idx) in enumerate(sess_list):
    days[global_idx] = float(day_idx)
```

iii. The notes explicitly say to compute an ordinal session index by date within each mouse.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI assigns session numbers `0, 1, 2, ...` within each mouse after sorting by date, then broadcasts the resulting scalar across all time bins of each trial.

ii.
```python
days = np.zeros(len(sessions), dtype=np.float32)
...
days[global_idx] = float(day_idx)
```
```python
day = np.full(n_tp, training_day, dtype=np.float32)
```

iii. The notes justify this as "ordinal session index (by date) within each mouse."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr` and per-bin integer frame indices, with the same session-level `frame_period`.

ii.
```python
StartFr = beh['StartFr'].astype(int)
```
```python
frame_indices = np.arange(start, end, dtype=np.float64)
time_since_start = (frame_indices - start) * frame_period
```

iii. The notes describe this variable as `frame_idx - StartFr`, converted into seconds.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each trial the AI subtracts the integer trial start frame from every frame index in the window and multiplies by `frame_period`. This starts exactly at `0` for the first kept bin.

ii.
```python
time_since_start = (frame_indices - start) * frame_period
```

iii. The trajectory and notes describe a simple frame-based conversion rather than interpolation on `ft`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same contiguous `start:end` frame grid used for the neural slice.

ii.
```python
frame_indices = np.arange(start, end, dtype=np.float64)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes state that all decoder variables are extracted on the raw per-trial frame window.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived directly from the per-trial `isRew` flag.

ii.
```python
isRew = beh['isRew']
...
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The mapping notes list `isRew` as the source and describe it as `1 if rewarded corridor, 0 otherwise`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond casting to float and broadcasting across all bins of the trial.

ii.
```python
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes treat this as a direct per-trial variable with no extra processing.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, one value per trial.

ii.
```python
WallName = beh['WallName']
...
stim_name = str(WallName[t])
```

iii. The notes say this output should come from corridor wall identity.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI collects all unique `UniqWalls` across the selected sessions, assigns each distinct wall label its own category index, and then broadcasts that index across all bins of a trial. It does not collapse variants into 4 base textures.

ii.
```python
for wn in beh['UniqWalls']:
    all_stim.add(str(wn))
return sorted(all_stim)
```
```python
stim_to_idx = {s: i for i, s in enumerate(all_stimuli)}
...
stim_idx = stim_to_idx[stim_name]
stim_out = np.full(n_tp, stim_idx, dtype=np.int64)
```

iii. In the trajectory the AI explicitly decided to keep all 15 unique stimulus names, and the validation notes later interpret visual-stimulus chance performance as `1/15`.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, the session-level list of lick frame numbers.

ii.
```python
lick_frs = beh['LickFr']
```

iii. The notes and trajectory treat `LickFr` as the core licking source variable and do not rely on `LickTrind` for assignment.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds lick frame numbers to the nearest integer, keeps the licks whose rounded frame falls in the current trial's `start:end` range, shifts them to trial-relative coordinates, clips them, and marks those bins as `1`.

ii.
```python
lick_frs_int = np.round(lick_frs).astype(int)
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
...
trial_lick_frs = lick_frs_int[mask] - start_fr
trial_lick_frs = np.clip(trial_lick_frs, 0, n_frames - 1)
lick_vec[trial_lick_frs] = 1
```

iii. The trajectory says the AI chose rounding because `LickFr` is fractional and thought frame-range assignment would usually put licks in the correct trial.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is built on the same `start:end` frame span used for the neural slice of that trial.

ii.
```python
lick = make_lick_vector(beh, start, end)
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes describe licking as time-varying, frame-based output on the raw trial window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from `ft_Pos`, sampled over the chosen per-trial frame span.

ii.
```python
ft_Pos = beh['ft_Pos']
...
pos = ft_Pos[start:end]
```

iii. The notes identify `ft_Pos` as the raw position stream in decimeters.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI uses `np.digitize` with cut points at 10, 20, and 30 dm on `ft_Pos[start:end]`, so the last category covers `30-60` dm and therefore includes gray-space frames.

ii.
```python
def digitize_position(pos):
    bins = np.digitize(pos, [10, 20, 30])
    return bins.astype(np.int64)
```
```python
pos = ft_Pos[start:end]
pos_bin = digitize_position(pos)
```

iii. The notes justify this by saying position bin 3 should include the gray space and label it as `3m+`.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are hard-coded as `[10, 20, 30]` dm, producing 4 bins: `[0,10)`, `[10,20)`, `[20,30)`, and `[30,60+]`.

ii.
```python
POS_BIN_EDGES = [0, 10, 20, 30, 60.01]
POS_BIN_LABELS = ['0-1m', '1-2m', '2-3m', '3m+']
```
```python
bins = np.digitize(pos, [10, 20, 30])
```

iii. The trajectory shows the AI debated whether gray-space frames should be excluded, then chose to fold them into the last bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position bins are taken from the same contiguous `start:end` trial window as the neural data.

ii.
```python
pos = ft_Pos[start:end]
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes say all outputs should stay temporally continuous with the chosen trial segment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed`.

ii.
```python
ft_RunSpeed = beh['ft_RunSpeed']
...
speed = ft_RunSpeed[start:end]
```

iii. The notes identify `ft_RunSpeed` as the source variable and describe the output as quartile-binned running speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI computes global percentile thresholds over all `ft_RunSpeed` values from all selected sessions, then digitizes each trial's raw `ft_RunSpeed[start:end]` against those thresholds.

ii.
```python
all_speeds = []
for s in sessions:
    beh = beh_cache[s['exp_type']][s['beh_key']]
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)
all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
```
```python
speed = ft_RunSpeed[start:end]
speed_bin = digitize_speed(speed, speed_quartiles)
```

iii. The notes explicitly say "Speed quartiles: computed globally across all frames in all sessions."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are determined by `np.digitize(speed, quartiles)` using those three global percentile cutoffs.

ii.
```python
def digitize_speed(speed, quartiles):
    bins = np.digitize(speed, quartiles)
    return bins.astype(np.int64)
```
```python
speed_labels = ['Q1_slow', 'Q2', 'Q3', 'Q4_fast']
```

iii. The validation notes discuss the resulting uneven quartile counts and justify them as a consequence of many exact zero-speed frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled on the same `start:end` frame window as the neural data.

ii.
```python
speed = ft_RunSpeed[start:end]
neural = spk[:, start:end].astype(np.float16)
```

iii. The notes describe speed as a time-varying output extracted from the same frame-level trial segment as the neural input.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates `StartFr` and `EndFr` to integers, skips trials that are out of neural-frame bounds, clips `end` to the available behavioral arrays, skips empty or 1-bin trials, and asserts that retinotopy and spike neuron counts match. It does not globally trim all behavior streams to the imaged frame count before trial extraction.

ii.
```python
assert len(iarea) == n_neurons, \
    f"Neuron count mismatch: spk={n_neurons}, retinotopy={len(iarea)}"
```
```python
if start < 0 or end > n_total_frames or end <= start:
    skipped += 1
    continue
end = min(end, len(ft_Pos), len(ft_RunSpeed))
if end <= start:
    skipped += 1
    continue
```

iii. In the trajectory the AI justified integer truncation of `StartFr` and `EndFr` as consistent with the reference code, and argued that frame-based lick assignment should be adequate except near boundaries.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are repeated loading of the large per-session spike arrays and constructing / saving the very large full pickle. The per-session processing loop also repeatedly slices huge neural matrices trial by trial.

ii.
```python
spk = np.concatenate(
    [nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0
)
```
```python
for t in range(ntrials):
    ...
    neural = spk[:, start:end].astype(np.float16)
```
```python
with open(args.outfile, 'wb') as f:
    pickle.dump(data, f, protocol=5)
```

iii. The notes emphasize file size and runtime throughout: sample conversion size, full conversion size, and the 33.4 minute full run.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest candidates are the per-trial loop in `process_session`, which repeatedly slices neural and behavioral arrays, and the repeated per-trial scanning of all lick frames inside `make_lick_vector`.

ii.
```python
for t in range(ntrials):
    ...
    neural = spk[:, start:end].astype(np.float16)
    ...
    lick = make_lick_vector(beh, start, end)
```
```python
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
```

iii. The AI only vectorized the retinotopy mapping, suggesting it noticed some efficiency issues but left the much larger trial-level loops in scalar Python form.

## 12-c. What processing does the code repeat multiple times?

i. It repeatedly scans the same lick array once per trial, repeatedly allocates `np.arange(start, end)` and `np.full(...)` arrays trial by trial, and performs separate full passes over behavior just to collect stimuli and speed thresholds before the main session loop.

ii.
```python
for s in sessions:
    beh = beh_cache[s['exp_type']][s['beh_key']]
    speeds = beh['ft_RunSpeed']
    all_speeds.append(speeds)
```
```python
for s in sessions:
    beh = beh_cache[s['exp_type']][s['beh_key']]
    for wn in beh['UniqWalls']:
        all_stim.add(str(wn))
```
```python
frame_indices = np.arange(start, end, dtype=np.float64)
day = np.full(n_tp, training_day, dtype=np.float32)
rew = np.full(n_tp, float(isRew[t]), dtype=np.float32)
```

iii. The notes focus on correctness and validation rather than efficiency, so these repeated passes appear to be incidental consequences of the implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code carries some small extra work that is not needed downstream: it builds `subjects_list` but never uses it, computes memory / summary statistics only for logging, and includes optional plotting code used only for manual inspection.

ii.
```python
subjects_list = []
```
```python
neural_mb = sum(t.nbytes for t in neural_trials) / 1e6
...
total_neural_mb = sum(sum(t.nbytes for t in s) for s in neural_all) / 1e6
```
```python
if args.show_processing and i < 2:
    plot_processing(...)
```

iii. The notes show the AI put substantial emphasis on debugging, visualization, and runtime monitoring, which explains these non-essential extras.
