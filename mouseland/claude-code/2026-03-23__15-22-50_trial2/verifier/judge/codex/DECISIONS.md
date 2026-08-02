# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent builds a session table from `data/beh/Imaging_Exp_info.npy`, deduplicates by `{mname}_{datexp}_{blk}`, chooses one behavior key per neural recording, then loads retinotopy from `data/retinotopy`, neural traces from `data/spk`, and behavior from the matching `Beh_{exp_type}.npy` file for each session.

ii. ```python
def build_session_map():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                       allow_pickle=True).item()
    for exp_type, db_list in exp_info.items():
        for ndb in db_list:
            spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
```

```python
iarea = load_retino(mname, datexp)
spk = load_spk_filtered(mname, datexp, blk, valid_mask)
beh = load_beh(session_info)
```

iii. In `CONVERSION_NOTES.md` Step 4 the agent says sessions can appear in multiple experiment types but each physical recording should be used once; in trajectory step 49 it explicitly reasons that behavior files contain the full trial set for a recording and should be paired once per neural file.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name `mname`. The converter assigns each new `mname` the next integer index and stores session-to-subject membership in `subject_idx`.

ii. ```python
subjects_seen = {}
...
mname = info['db']['mname']
if mname not in subjects_seen:
    subjects_seen[mname] = len(subjects_seen)
...
subject_idx_list.append(subjects_seen[mname])
subjects = sorted(subjects_seen.keys(), key=lambda x: subjects_seen[x])
```

iii. `CONVERSION_NOTES.md` Step 5 lists mouse name as the source for `subjects` and `subject_idx`, and Step 2 reports the expected 19 mice.

## 1-c. How are the data split into sessions?

i. Sessions are split by unique neural recording key `{mname}_{datexp}_{blk}`. If multiple experiment-info entries point to the same recording, the agent keeps one entry and prefers the entry without `stimtype`.

ii. ```python
spk_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
beh_key = f"{spk_key}_{ndb['stimtype']}" if 'stimtype' in ndb else spk_key
if spk_key not in session_map or 'stimtype' not in ndb:
    session_map[spk_key] = {
        'exp_type': exp_type, 'beh_key': beh_key, 'db': dict(ndb)
    }
```

iii. In `CONVERSION_NOTES.md` Step 4 the agent says there are 89 unique sessions and that sessions with `stimtype` variants should still be represented once per physical recording. Trajectory step 49 gives the same justification.

## 1-d. How are the data split into trials?

i. Trials are defined from `StartFr[i]` to `StartFr[i+1]`, or to the end of the session for the last trial. The agent clips the frame bounds to valid range and skips trials shorter than 2 frames.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
start = max(0, start)
end = min(nfr_use, end)
n_frames = end - start
if n_frames < 2:
    continue
```

iii. `CONVERSION_NOTES.md` Step 5 says the planned trial length is from `StartFr` to the start of the next trial so that corridor and gray space are both kept. Trajectory steps 69, 72, and 74 show the agent explicitly choosing this wider trial window while thinking about file size.

## 1-e. How are trials filtered based on quality controls?

i. There is almost no trial-quality filtering. The agent only drops trials shorter than 2 frames and discards whole sessions that end up with fewer than 2 valid trials.

ii. ```python
if n_frames < 2:
    continue
...
if len(neural_trials) < 2:
    print(f"WARNING: <2 valid trials for {spk_key}")
    return None
```

iii. `CONVERSION_NOTES.md` Step 3 says there is no explicit trial filtering in the reference code for basic loading, and Step 5 lists "use all trials (no filtering)" as the plan for decoder conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the per-session `spks` arrays stored in `data/spk/{mname}_{datexp}_{blk}_neural_data.npy`, with retinotopy `iarea` from `data/retinotopy/{mname}_{datexp}_trans.npz` used to decide which neurons to keep.

ii. ```python
spk_data = np.load(os.path.join(DATA_ROOT, 'spk', fn), allow_pickle=True).item()
planes = spk_data['spks']
...
dtrans = np.load(os.path.join(DATA_ROOT, 'retinotopy', fn), allow_pickle=True)
return dtrans['iarea']
```

iii. `CONVERSION_NOTES.md` Step 1 identifies the neural source as Suite2p deconvolved calcium traces in `spks`, and Step 3 states that no extra dF/F computation is needed because the stored data are already deconvolved traces.

## 2-b. How is the `neural` data processed?

i. The agent concatenates planes after filtering, converts traces to `float16`, truncates to the available behavioral frame count, and slices the result into per-trial `(n_neurons, n_timepoints)` matrices without temporal resampling.

ii. ```python
for plane in planes:
    n = plane.shape[0]
    plane_mask = valid_mask[offset:offset+n]
    filtered.append(plane[plane_mask].astype(np.float16))
...
spk = np.concatenate(filtered, 0)
...
nfr_use = min(nfr, len(beh['ft']))
spk = spk[:, :nfr_use]
trial_spk = spk[:, start:end].copy()
```

iii. `CONVERSION_NOTES.md` Step 6 says the script stores deconvolved calcium traces as `float16` and filters per plane before concatenation as a memory optimization. Trajectory steps 72, 79, and 81 show that the main justification was keeping the full dataset small enough to save and load.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are filtered only by retinotopy area code. Any neuron with `iarea` equal to `-1` or `7` is excluded, and all remaining neurons are assigned to `V1`, `mHV`, `lHV`, or `aHV`.

ii. ```python
EXCLUDED_AREAS = {-1, 7}
...
valid_mask = np.array([int(ia) not in EXCLUDED_AREAS for ia in iarea])
region_idx = np.array([
    BRAIN_REGIONS.index(AREA_MAP.get(int(ia), 'V1'))
    for ia in iarea[valid_mask]
], dtype=np.int64)
```

iii. `CONVERSION_NOTES.md` Steps 1, 3, and 5 all describe this as the intended neuron curation rule, citing the reference retinotopy mapping and the paper's "inside visual cortex" restriction.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Neural data are aligned to corridor entry by starting each trial at `StartFr[i]`. The agent does not re-center around the cue; it keeps a forward-running trial segment beginning at trial start.

ii. ```python
StartFr = beh['StartFr'].astype(int)
...
start = StartFr[i]
end = StartFr[i + 1] if i < ntrials - 1 else nfr_use
trial_spk = spk[:, start:end].copy()
```

iii. The task instructions asked for alignment to trial start, and `CONVERSION_NOTES.md` Step 5 says "align to StartFr". The same note also states that the agent intentionally kept corridor plus gray-space frames after the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data keep the native imaging frame rate: `1000 / 3.17 ≈ 315.5 ms` per bin. No temporal rebinning is applied.

ii. ```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
...
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says the time bin size should be the native frame rate and that no resampling is needed.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the frame indices spanning that trial.

ii. ```python
SoundFr = beh['SoundFr']
...
frame_idx = np.arange(start, end)
sound_fr = SoundFr[i]
```

iii. `CONVERSION_NOTES.md` Step 5 maps "Time to SoundFr" to `input[0]` and says it should be computed relative to the current frame.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each frame in a trial, the agent computes `(SoundFr[i] - frame_idx) / fs` in seconds. If `SoundFr[i]` is `NaN`, the whole trace is replaced by zeros.

ii. ```python
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
else:
    time_to_sound = ((sound_fr - frame_idx) / fs).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 states exactly this formula. The `NaN -> zeros` fallback is an implementation choice rather than a documented reference rule.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is sliced over the exact same per-trial frame window as `neural`, so each neural frame gets one time-to-cue value.

ii. ```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The agent's notes treat this as a framewise aligned input derived from the same `StartFr`-anchored trial window used for neural slicing.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from session metadata fields in `Imaging_Exp_info.npy`, preferring `days` and falling back to `sess#`.

ii. ```python
def get_session_day(db):
    for key in ['days', 'sess#']:
        if key in db:
            return int(db[key])
    return 0
```

iii. `CONVERSION_NOTES.md` Step 5 says the source is the session metadata and that `sess#` should be used if explicit `days` is missing.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The chosen day value is converted to `float32` once per session and then broadcast across all frames in each trial.

ii. ```python
day = np.float32(get_session_day(db))
...
np.full(n_frames, day, dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly lists it as a per-trial scalar input and notes the fallback behavior.

## 4-c. What variables in the raw data is `input` *Environment type* derived from?

i. It is not derived at all. The converter does not create an environment-type input, and there is no environment-type channel in `input_names`.

ii. ```python
'input_names': ['time_to_sound_cue', 'day_of_training',
                'time_since_trial_start', 'reward_availability'],
```

iii. The task instructions only required four decoder inputs, and the agent's Step 5 variable map contains those same four inputs and no environment-type field.

## 4-d. What processing is involved in computing `input` *Environment type*?

i. None. The variable is omitted entirely.

ii. ```python
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. The omission follows the agent's interpretation of the decoder task in `CONVERSION_NOTES.md` Step 5.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from the per-trial frame count, using `StartFr[i]` as time zero.

ii. ```python
start = StartFr[i]
...
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `current_frame - StartFr` to `time_since_trial_start`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent creates a linear ramp `0, 1/fs, 2/fs, ...` within each trial and stores it as `float32`.

ii. ```python
(np.arange(n_frames) / fs).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 describes it as `(current_frame - StartFr) / fs`, starting at zero.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It uses the same trial segmentation and frame count as `neural`, so every neural frame is paired with a time-since-start value.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
inp = np.stack([
    time_to_sound,
    np.full(n_frames, day, dtype=np.float32),
    (np.arange(n_frames) / fs).astype(np.float32),
    np.full(n_frames, float(isRew[i]), dtype=np.float32),
], axis=0)
```

iii. Trajectory step 212 shows the agent validating that the first element of this channel is zero for every trial, which was its sanity check for alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the trial-level behavioral variable `isRew`.

ii. ```python
isRew = beh['isRew']
...
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `isRew` directly to the reward-availability input.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial-level reward flag is cast to float and broadcast across all frames in that trial.

ii. ```python
np.full(n_frames, float(isRew[i]), dtype=np.float32)
```

iii. The notes describe it as a per-trial scalar, and trajectory step 90 discusses supervised versus unsupervised sessions in exactly those terms.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial `WallName`.

ii. ```python
WallName = beh['WallName']
...
stim = standardize_stim_name(str(WallName[i]))
```

iii. `CONVERSION_NOTES.md` Step 5 lists `WallName` as the source variable for visual stimulus.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent standardizes stimulus names with a hand-written mapping (`rock* -> circle*`, `wood* -> leaf*`, `brick* -> circle*/leaf3`, `wood5 -> leaf3`, `rock5 -> circle3`), stores per-trial names during session processing, then globally sorts all names and replaces the placeholder output row with integer category IDs.

ii. ```python
STIM_CATEGORY_MAP = {
    'rock1': 'circle1', 'rock2': 'circle2',
    'wood1': 'leaf1', 'wood2': 'leaf2',
    'brick1': 'circle1', 'brick2': 'circle2',
    'brick5': 'leaf3', 'wood5': 'leaf3', 'rock5': 'circle3',
}
...
stim = standardize_stim_name(str(WallName[i]))
...
output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. `CONVERSION_NOTES.md` Step 5 says the agent wanted "standardized categories", and trajectory step 234 notes that it explicitly fixed `wood5 -> leaf3` late in the process.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from frame-level lick timestamps `LickFr`.

ii. ```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
if 'LickFr' in beh and len(beh['LickFr']) > 0:
    lick_fr = beh['LickFr'].astype(int)
    valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
    lick_binary[lick_fr[valid_lick]] = 1
```

iii. `CONVERSION_NOTES.md` Step 5 maps `LickFr/LickTrind` to a binary per-frame lick output, and trajectory step 90 shows the agent checking whether all-zero licking in some sample sessions was due to unsupervised behavior rather than a parsing bug.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The agent creates a session-length binary vector initialized to zero and marks every valid lick frame with `1`.

ii. ```python
lick_binary = np.zeros(nfr_use, dtype=np.int64)
...
lick_binary[lick_fr[valid_lick]] = 1
```

iii. `CONVERSION_NOTES.md` Step 5 says the intended output is "binary per frame - check if any lick occurred in that frame's time window."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-level lick vector is sliced with the same `[start:end]` trial window used for neural data.

ii. ```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. The notes treat all outputs as frame-aligned to the same trial window defined from `StartFr`.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from frame-level position `ft_Pos`, with `ft_CorrSpc` used to tell corridor frames from non-corridor frames.

ii. ```python
ft_Pos = beh['ft_Pos'][:nfr_use]
ft_CorrSpc = beh['ft_CorrSpc'][:nfr_use].astype(bool)
```

iii. `CONVERSION_NOTES.md` Step 5 maps `ft_Pos` to the position output and notes the 0-4 m texture corridor plus gray space structure.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent initializes all frames to a gray-space class, then overwrites corridor frames with one of four 10-decimeter bins using `ft_Pos`.

ii. ```python
pos_bins = np.full(nfr_use, 4, dtype=np.int64)  # default: gray
for b in range(4):
    mask = ft_CorrSpc & (ft_Pos >= b * 10) & (ft_Pos < (b + 1) * 10)
    pos_bins[mask] = b
pos_bins[ft_CorrSpc & (ft_Pos >= 40)] = 3
```

iii. `CONVERSION_NOTES.md` Step 5 explicitly says the agent planned four 1 m texture bins plus an extra gray-space category, even though it noticed that the task only asked for four bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Categories are: `0-1m`, `1-2m`, `2-3m`, `3-4m`, and a fifth `gray` category for all non-corridor frames.

ii. ```python
'output_values': [
    all_stim_sorted,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
    ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
],
```

iii. The notes justify this by saying gray-space frames still exist in the chosen trial window and therefore need a class label.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. The framewise position labels are sliced with the same trial bounds as neural data.

ii. ```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. `CONVERSION_NOTES.md` Step 5 says position should be time-varying and aligned to the same `StartFr`-anchored trial frames.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from frame-level running speed `ft_RunSpeed`.

ii. ```python
ft_RunSpeed = beh['ft_RunSpeed'][:nfr_use]
...
speeds = beh['ft_RunSpeed']
```

iii. `CONVERSION_NOTES.md` Step 5 lists `ft_RunSpeed` as the source for the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first pools all session-level `ft_RunSpeed` values, computes quartiles on the subset with `speed > 0`, then assigns every frame to one of four bins by thresholding against those three cut points.

ii. ```python
all_speeds = np.concatenate(all_speeds)
valid = all_speeds > 0
return np.percentile(all_speeds[valid], [25, 50, 75])
...
speed_bins = np.zeros(nfr_use, dtype=np.int64)
speed_bins[ft_RunSpeed >= speed_quartiles[0]] = 1
speed_bins[ft_RunSpeed >= speed_quartiles[1]] = 2
speed_bins[ft_RunSpeed >= speed_quartiles[2]] = 3
```

iii. `CONVERSION_NOTES.md` Step 5 says quartiles should be computed across running frames. Trajectory step 193 explicitly defends putting non-running frames into the slowest bin after computing quartiles on positive speeds only.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the 25th, 50th, and 75th percentiles of all positive `ft_RunSpeed` values pooled across sessions; labels are `Q1_slow`, `Q2`, `Q3`, and `Q4_fast`.

ii. ```python
return np.percentile(all_speeds[valid], [25, 50, 75])
...
'output_values': [
    ...,
    ['Q1_slow', 'Q2', 'Q3', 'Q4_fast'],
],
```

iii. The agent documents this directly in `CONVERSION_NOTES.md` Step 5 and Step 10, and trajectory step 193 explains why this leads to the first bin containing many stopped frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. The framewise speed-bin vector is sliced with the same `[start:end]` trial window as the neural data.

ii. ```python
out = np.stack([
    np.full(n_frames, 0, dtype=np.int64),
    lick_binary[start:end],
    pos_bins[start:end],
    speed_bins[start:end],
], axis=0)
```

iii. The notes treat running speed as a time-varying aligned output just like licking and position.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The converter uses silent fallbacks rather than explicit errors: `SoundFr = NaN` becomes an all-zero time-to-sound trace, lick frames are clipped to valid bounds, trial bounds are clipped to `[0, nfr_use]`, neural data are truncated to the length of `beh['ft']`, missing day metadata becomes `0`, and unknown stimulus names are passed through unchanged.

ii. ```python
nfr_use = min(nfr, len(beh['ft']))
...
if np.isnan(sound_fr):
    time_to_sound = np.zeros(n_frames, dtype=np.float32)
...
valid_lick = (lick_fr >= 0) & (lick_fr < nfr_use)
...
for key in ['days', 'sess#']:
    if key in db:
        return int(db[key])
return 0
```

iii. The notes describe these mainly as practical robustness choices rather than decisions from the reference code. Trajectory step 212 shows the agent doing ad hoc sanity checks after discovering one indexing mistake.

## 12-a. What are the most time-consuming steps of the code?

i. The agent identifies loading huge neural files and then slicing them trial-by-trial as the main runtime bottlenecks; computing speed quartiles from behavior only is treated as comparatively cheap.

ii. ```python
speed_quartiles = collect_speed_quartiles(session_map, keys)
...
for idx, spk_key in enumerate(keys):
    result = process_session(spk_key, info, speed_quartiles)
```

iii. `CONVERSION_NOTES.md` Step 7 and trajectory step 106 both estimate that neural I/O and per-session trial processing dominate the runtime.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization targets are the per-trial loop inside `process_session`, the per-session loop in `convert_data`, the per-session stimulus re-encoding pass, and the per-plane filtering loop in `load_spk_filtered`.

ii. ```python
for plane in planes:
    ...
for i in range(ntrials):
    ...
for idx, spk_key in enumerate(keys):
    ...
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. Trajectory step 106 discusses avoiding extra copies during per-plane filtering, and the structure of the converter makes the repeated loops explicit.

## 12-c. What processing does the code repeat multiple times?

i. The code loads behavior twice for every session: once in `collect_speed_quartiles` and again in `process_session`. It also traverses all trials once to build outputs and then traverses all trials again to write the final stimulus indices.

ii. ```python
def collect_speed_quartiles(session_map, keys):
    for spk_key in keys:
        info = session_map[spk_key]
        beh = load_beh(info)
...
for idx, spk_key in enumerate(keys):
    result = process_session(spk_key, info, speed_quartiles)
...
for si in range(len(output_all)):
    for ti in range(len(output_all[si])):
        output_all[si][ti][0, :] = stim_to_idx[all_stim_per_session[si][ti]]
```

iii. `CONVERSION_NOTES.md` Step 6 calls out speed quartile collection as a separate pass over all sessions, and the code clearly performs a second pass for stimulus ID insertion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It builds optional verification plots that are not needed for the saved dataset, keeps an intermediate `stim_names` list only to do a later re-encoding pass, copies every trial slice with `.copy()`, and stores a gray-space position class even though the decoder specification asked for four corridor bins.

ii. ```python
trial_spk = spk[:, start:end].copy()
...
stim_names.append(stim)
...
if show_processing:
    plot_processing(data)
...
'output_values': [
    all_stim_sorted,
    ['no_lick', 'lick'],
    ['0-1m', '1-2m', '2-3m', '3-4m', 'gray'],
```

iii. The notes justify plotting as a validation aid and justify the gray class as a consequence of keeping gray-space frames, but both choices add processing or categories beyond what the downstream decoder task strictly asked for.
