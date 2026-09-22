# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script reads `beh/Imaging_Exp_info.npy` as the master index, deduplicates sessions by `mname_datexp_blk`, then loads each session's behavior from the experiment-type behavior file, spikes from the per-session spike file, and retinotopy from the per-date retinotopy file.

ii.
```python
info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
               allow_pickle=True).item()
...
if exp_type not in beh_cache:
    beh_cache[exp_type] = load_beh(exp_type)
beh = beh_cache[exp_type][beh_key]
spk = load_spk(db)
retino = load_retino(db)
```

iii. The notes say this “uses same load_spk pattern” and the same session-key format as the reference. The trajectory also emphasizes that behavior is cached per experiment type and that the reference loading pattern was intentionally copied.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by `db['mname']`. During assembly, the script assigns each new mouse the next integer in `subjects_set` and appends that index to `subject_idx_list` for each processed session.

ii.
```python
mname = db['mname']
if mname not in subjects_set:
    subjects_set[mname] = len(subjects_set)
subject_idx_list.append(subjects_set[mname])
```

iii. The notes only state that 19 unique subjects were recovered; they do not explicitly justify the ordering choice. The subject split is inferred directly from `mname` in the session metadata.

## 1-c. How are the data split into sessions?

i. A session is defined as one `(mname, datexp, blk)` triple. The code builds `key = f'{mname}_{datexp}_{blk}'` and keeps only the first occurrence when the same recording appears under multiple experiment types.

ii.
```python
key = f"{db['mname']}_{db['datexp']}_{db['blk']}"
if key not in seen_keys:
    seen_keys.add(key)
    all_sessions.append((exp_type, db, key))
```

iii. The notes report 89 unique sessions and explicitly treat duplicate listings across experiment types as the same recording.

## 1-d. How are the data split into trials?

i. Trials are cut from `StartFr` to `EndFr` for each trial after flooring both boundaries to integer frames and clipping them to the imaged range. The code does not use `ft_trInd` or `ft_CorrSpc` to define the trial window.

ii.
```python
start_fr = np.array(beh['StartFr'])
end_fr = np.array(beh['EndFr'])
...
start_fr_int = np.floor(start_fr).astype(int)
end_fr_int = np.floor(end_fr).astype(int)
...
s_fr = start_fr_int[tr]
e_fr = end_fr_int[tr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Step 5 of the notes says “Trial boundaries: StartFr to EndFr (including gray space)” and Step 10 says “We include all frames (not just running). This maintains temporal continuity for time-varying decoder.”

## 1-e. How are trials filtered based on quality controls?

i. The code drops only trials with fewer than 2 frames after clipping the integer `StartFr`/`EndFr` bounds. It does not remove long trials, stationary outliers, or trials that become empty after `ft_CorrSpc` masking because it never applies that mask.

ii.
```python
n_trial_frames = e_fr - s_fr
if n_trial_frames < 2:
    skipped_trials += 1
    continue
```

iii. The notes explicitly defend keeping very long trials: “Very long trials (up to 5621 frames): Included, not filtered.”

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is taken from the `spks` arrays in the session spike file, concatenated across imaging planes. Neuron region labels are taken from retinotopy `iarea` values.

ii.
```python
spk = np.concatenate([nspk for nspk in
                      np.load(spk_path, allow_pickle=True).item()['spks']], 0)
...
iarea = retino['iarea']
```

iii. The notes describe this as matching the reference `load_spk` and `load_retino` functions.

## 2-b. How is the `neural` data processed?

i. Per trial, the code slices the deconvolved traces by the chosen frame window and stores them as `float16`. Before that, it may truncate neurons to the smaller of spike/retinotopy counts and randomly subsample each session to at most 5000 neurons.

ii.
```python
if n_neu_spk != n_neu_ret:
    min_n = min(n_neu_spk, n_neu_ret)
    spk = spk[:min_n]
...
if n_total > MAX_NEURONS:
    neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
    spk = spk[neuron_idx]
...
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. Step 5 of the notes calls out a key decision: “Neuron subsampling: 5000 neurons per session ... to keep file size manageable. Decoder uses PCA to 100 components.” It also justifies `float16` as a size reduction.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code does not filter neural units down to the four visual areas. Instead, it maps unknown `iarea` codes to an `unassigned` category and keeps them; the only hard cap is random subsampling to 5000 neurons per session.

ii.
```python
regions.append(area_map.get(a_int, 'unassigned'))
...
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
...
if n_total > MAX_NEURONS:
    neuron_idx = np.sort(rng.choice(n_total, MAX_NEURONS, replace=False))
```

iii. The notes list `unassigned` as a kept brain region and only justify filtering in terms of file-size-driven subsampling, not visual-area selection.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural trials are aligned to corridor entry by starting each window at integer-floored `StartFr`, and they run until integer-floored `EndFr`. Trial lengths are variable and include gray-space frames after the textured corridor.

ii.
```python
s_fr = start_fr_int[tr]
e_fr = end_fr_int[tr]
neural = spk[:, s_fr:e_fr].astype(np.float16)
```

iii. The notes explicitly say the alignment event is trial start/corridor entry and that the script keeps “all frames” through the trial for temporal continuity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data uses the imaging frame rate of 3.17 Hz, so each bin is `1000 / 3.17` ms. No additional temporal rebinning or resampling is applied.

ii.
```python
FS = 3.17
TIME_BIN_MS = 1000.0 / FS
...
'time_bin_size': TIME_BIN_MS,
```

iii. The notes and README both describe the frame rate as 3.17 Hz and the bin size as about 315.5 ms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The code derives it from `SoundFr` and the integer frame indices of the extracted trial window. It does not use the recorded frame timestamps `ft`.

ii.
```python
sound_fr = np.array(beh['SoundFr'])
...
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
```

iii. The notes map this variable as “`SoundFr - frame_idx` / fs”, and the README describes it as time relative to the sound cue in seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. It computes `(frame_index - SoundFr) / FS`, giving a value that is negative before the cue and positive after the cue.

ii.
```python
# Input 0: Time to sound cue (seconds) - negative before, positive after
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The notes and README both explicitly describe the sign convention as negative before cue and positive after cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on the same per-trial `s_fr:e_fr` frame window that is used to slice the neural matrix, so each neural time bin has one corresponding sound-cue time value.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_to_cue = (frame_indices - sound_fr[tr]) / FS
```

iii. The notes repeatedly justify keeping all streams on the same trial window for “temporal continuity.”

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The code derives day of training from per-session metadata fields in `Imaging_Exp_info.npy`, preferring `sess#`, then `days`, and otherwise defaulting to 0.

ii.
```python
def get_day_of_training(db):
    if 'sess#' in db:
        return float(db['sess#'])
    elif 'days' in db:
        return float(db['days'])
    else:
        return 0.0
```

iii. The Step 5 mapping table in the notes explicitly says “`sess#` or `days` -> day_of_training.”

## 4-b. What processing is involved in computing `input` *Day of training*?

i. No per-mouse reordering is done. The selected metadata value is simply broadcast across all time bins of the trial.

ii.
```python
day_of_training = get_day_of_training(db)
...
day_input = np.full(n_trial_frames, day_of_training, dtype=np.float64)
```

iii. The notes treat this as a direct metadata field rather than a value to derive from session chronology.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The code derives it from `StartFr` and the integer frame indices of the extracted trial window, not from the recorded `ft` timestamps.

ii.
```python
start_fr = np.array(beh['StartFr'])
...
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
```

iii. The notes map this input as “`frame_idx - StartFr` / fs”.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. It computes `(frame_index - StartFr) / FS` on the `StartFr`→`EndFr` window. Because `StartFr` is fractional and the window starts at `floor(StartFr)`, the first bin can be slightly negative.

ii.
```python
s_fr = start_fr_int[tr]
...
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. The notes explicitly record this behavior: “Negative StartFr (first trial): Clipped to 0.” The verification output also showed small negative values for `time_since_trial_start`.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same `s_fr:e_fr` bins used to slice the neural data, so it is time-locked to the AI’s chosen trial window.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
frame_indices = np.arange(s_fr, e_fr, dtype=np.float64)
time_since_start = (frame_indices - start_fr[tr]) / FS
```

iii. The notes describe all time-varying streams as kept on the same full-trial window for continuity.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Reward availability is taken directly from the trial-level `isRew` flag.

ii.
```python
is_rew = np.array(beh['isRew']).astype(float)
```

iii. The notes map `isRew` directly to `reward_availability`.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The per-trial reward flag is broadcast across all time bins of that trial and stored as a float input channel.

ii.
```python
reward_input = np.full(n_trial_frames, is_rew[tr], dtype=np.float64)
```

iii. The notes treat this as a direct trial label and do not describe additional processing.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The visual-stimulus output is derived from trial-level `WallName`, with the allowed label set collected beforehand from each session’s `UniqWalls`.

ii.
```python
for w in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(w))
...
wall_name = np.array(beh['WallName'])
```

iii. The notes and README both state that stimulus labels come from the raw wall names, not from a collapsed texture-family mapping.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The code treats each distinct wall name as its own category. It builds a sorted `stim_list` from all observed `UniqWalls`, maps each trial’s `WallName` to its index in that list, and broadcasts that index across the whole trial.

ii.
```python
stim_list = sorted(all_stim)
...
stim_to_idx = {s: i for i, s in enumerate(stim_list)}
stim_name = str(wall_name[tr])
stim_idx = stim_to_idx.get(stim_name, 0)
stim_output = np.full(n_trial_frames, stim_idx, dtype=np.int64)
```

iii. The Step 5 mapping table says “15 categories,” and the README explicitly documents 15 stimulus labels such as `circle1`, `leaf1_swap1`, etc.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr` and, additionally, `LickTrind` so the script can assign lick events to individual trials before rasterizing them within each trial window.

ii.
```python
lick_fr = np.array(beh['LickFr']).astype(float) if len(beh['LickFr']) > 0 else np.array([])
lick_trind = np.array(beh['LickTrind']).astype(int) if len(beh['LickTrind']) > 0 else np.array([], dtype=int)
```

iii. The notes map licking from `LickFr, LickTrind` to a binary per-frame output.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, the code selects licks with `LickTrind == tr`, rounds the lick-frame numbers to integers, converts them to trial-relative indices by subtracting `s_fr`, clips to the valid trial range, and marks those bins as 1 in a binary vector.

ii.
```python
lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
trial_lick_mask = lick_trind == tr
trial_lick_frames = np.round(lick_fr[trial_lick_mask]).astype(int)
trial_lick_frames = trial_lick_frames - s_fr
valid = (trial_lick_frames >= 0) & (trial_lick_frames < n_trial_frames)
lick_binary[trial_lick_frames[valid]] = 1
```

iii. The notes only say “Binary per frame”; they do not explicitly justify the choice to round rather than truncate lick times.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is rasterized on the same `s_fr:e_fr` trial window as the neural data, so the binary lick vector has one entry per neural time bin.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
...
lick_binary = np.zeros(n_trial_frames, dtype=np.int64)
```

iii. The notes emphasize that all time-varying streams share the same full trial window.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the framewise behavioral variable `ft_Pos`.

ii.
```python
ft_pos = np.array(beh['ft_Pos'])[:nfr]
```

iii. The notes map `ft_Pos` directly to the position output.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The code slices `ft_Pos` on the `StartFr`→`EndFr` window, divides by 10 decimeters, clips the result into 4 bins, and stores the integer bin per frame.

ii.
```python
pos = ft_pos[s_fr:e_fr]
pos_binned = discretize_position(pos)
```

iii. The notes state that position uses 4 bins of 1 m and that gray-space frames are intentionally retained in the last bin.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Bins are computed as `floor(position_dm / 10)` and clipped to the range 0-3, so 0-10 dm maps to bin 0, 10-20 dm to bin 1, 20-30 dm to bin 2, and everything at 30 dm or above maps to bin 3.

ii.
```python
def discretize_position(position_dm):
    bins = np.clip(position_dm // POS_BIN_SIZE_DM, 0, N_POS_BINS - 1).astype(np.int64)
    return bins
```

iii. The notes explicitly defend assigning gray-space frames to the last position bin.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is sampled on the same `s_fr:e_fr` frame window as the neural data, yielding one position-bin label per neural time bin.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
pos = ft_pos[s_fr:e_fr]
```

iii. The notes justify using a common full-trial window for all time-varying signals.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from the framewise behavioral variable `ft_RunSpeed`.

ii.
```python
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
```

iii. The notes map `ft_RunSpeed` directly to the running-speed output.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The code first computes global speed quartile boundaries across all behavior frames in all unique sessions using `np.percentile`, then discretizes each trial’s `ft_RunSpeed` values with `np.digitize` against those fixed global thresholds.

ii.
```python
all_speeds = np.concatenate(all_speeds)
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
bins = np.digitize(speed, quartiles)
```

iii. The notes explicitly call out “Speed quartiles: Computed across all frames in all sessions,” and later defend the uneven bin counts that follow from many near-zero speeds.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Thresholds are the 25th, 50th, and 75th percentiles of the concatenated global speed distribution, and `np.digitize` maps each frame into bins 0-3.

ii.
```python
quartiles = np.percentile(all_speeds, [25, 50, 75])
...
bins = np.digitize(speed, quartiles)
```

iii. The notes state that the bins are quartiles computed across all frames and all sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sliced on the same `s_fr:e_fr` frame window as the neural data, so each neural time bin gets one discretized speed label.

ii.
```python
neural = spk[:, s_fr:e_fr].astype(np.float16)
speed = ft_run_speed[s_fr:e_fr]
```

iii. The notes justify keeping all streams on the same trial window.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles edge cases ad hoc: it clips `StartFr`/`EndFr` to valid frame indices, truncates framewise behavior arrays to the number of neural frames, skips behavior sessions whose key is missing, truncates spike or retinotopy arrays to the smaller neuron count if they disagree, masks out lick bins outside the trial window, and skips trials with fewer than 2 frames or sessions with fewer than 2 surviving trials.

ii.
```python
ft_pos = np.array(beh['ft_Pos'])[:nfr]
ft_run_speed = np.array(beh['ft_RunSpeed'])[:nfr]
start_fr_int = np.clip(start_fr_int, 0, nfr - 1)
end_fr_int = np.clip(end_fr_int, 0, nfr)
...
if n_neu_spk != n_neu_ret:
    min_n = min(n_neu_spk, n_neu_ret)
    spk = spk[:min_n]
...
if beh_key not in beh_cache[exp_type]:
    continue
```

iii. The notes explicitly mention clipped negative `StartFr`, retained long trials, and all-zero licking/reward in unsupervised sessions. The rest of the error handling is present in code but not substantially justified in the notes.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant costs are per-session spike-file loading/concatenation and the extra full-dataset behavior passes used to gather unique stimuli and compute global speed quartiles. Optional plotting is also nontrivial in `--show-processing` mode.

ii.
```python
spk = np.concatenate([nspk for nspk in
                      np.load(spk_path, allow_pickle=True).item()['spks']], 0)
...
for exp_type, sessions in info.items():
    ...
    all_speeds.append(speeds)
```

iii. The logs and notes report per-session timing and explicitly print a global “Computing global speed quartiles...” phase before session processing. The notes do not isolate a single bottleneck beyond these timings.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain non-vectorized: the per-neuron `iarea` mapping loop in `neu_area_ID`, the per-trial loop in `process_session`, the per-trial lick-event masking, and the repeated full-dataset scans over behavior files for stimuli and speed thresholds.

ii.
```python
for a in iarea:
    ...
for tr in range(ntrials):
    ...
    trial_lick_mask = lick_trind == tr
...
for exp_type, sessions in info.items():
    for db in sessions:
```

iii. The AI did not explicitly discuss vectorization opportunities beyond general efficiency goals, so this characterization is inferred from the code structure itself.

## 12-c. What processing does the code repeat multiple times?

i. The script traverses the whole dataset multiple times: once to build the unique session list, once to gather unique stimulus names, once to compute global speed quartiles, and then again to do the actual session conversion. Behavior caches are also rebuilt for different passes.

ii.
```python
for exp_type, sessions in info.items():
    ... all_sessions.append((exp_type, db, key))
...
for exp_type, sessions in info.items():
    ... all_stim.add(str(w))
...
for exp_type, sessions in info.items():
    ... all_speeds.append(speeds)
```

iii. The notes justify the repeated passes indirectly by choosing 15 stimulus labels and global speed quartiles, both of which require whole-dataset scans in this implementation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The core conversion does some extra work that is not needed by downstream decoding: optional `--show-processing` plotting, end-of-run summary statistics, and whole-dataset collection of 15 wall-name labels rather than collapsing them to the 4 decoder categories used in the reference. It also carries a few unused values such as `nneu` and `session_idx`.

ii.
```python
nneu = spk.shape[0]
...
if show_processing and len(neural_trials) > 0:
    ... plt.savefig(f'/app/processing_{session_id}.png', dpi=100)
...
for w in beh_cache[exp_type][beh_key]['UniqWalls']:
    all_stim.add(str(w))
```

iii. The notes explicitly require the processing plots and richer label reporting for sanity checking, so the extra work appears deliberate rather than accidental.
