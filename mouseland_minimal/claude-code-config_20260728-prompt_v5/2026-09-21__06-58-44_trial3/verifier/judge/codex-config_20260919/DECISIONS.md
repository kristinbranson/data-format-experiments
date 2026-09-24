# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy` as the master index, deduplicates physical recordings, caches one `Beh_<exp_type>.npy` dictionary per experiment type, and loads each session's spike and retinotopy files. A preliminary pass also loads behavior and the first spike plane to collect speed values.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
beh_cache[exp_type] = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. The trajectory says all experiment types should be included, while duplicate listings represent the same physical recording and should be used only once. It chose the first matching behavior source and cached behavior files.

## 1-b. How are the data split into subjects?

i. `mname` defines a subject. Subjects are accumulated in first-encounter order, and each retained session receives the index of its mouse in that list.

ii.
```python
if mname not in subjects_set:
    subjects_set.append(mname)
subject_idx_all.append(subjects_set.index(mname))
```

iii. The trajectory identified the unique mice from the experiment index; no derived subject split was needed.

## 1-c. How are the data split into sessions?

i. A session is the tuple `(mname, datexp, blk)`. Listings under multiple experiment types or stimulus variants are deduplicated, with the first listing retained.

ii.
```python
key = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if key not in unique_sessions:
    unique_sessions[key] = {...}
```

iii. The agent inspected duplicate entries and concluded that they share the same physical neural recording and essentially the same behavior, so one copy should be used.

## 1-d. How are the data split into trials?

i. For each integer trial `t` from `0` to `ntrials-1`, it selects imaged frames whose `ft_trInd` equals `t` and whose `ft_CorrSpc` flag is true. Thus trials span the textured corridor and remain variable length.

ii.
```python
for t in range(ntrials):
    mask = (ft_trInd == t) & ft_CorrSpc
    frames = np.where(mask)[0]
```

iii. The trajectory reasoned that the frame-level trial index is preferable to fractional boundary fields, and that gray-space frames should be excluded because the requested position labels cover the 4 m textured corridor.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than three corridor frames are discarded. Afterward, an entire session is discarded if fewer than two trials remain. There is no upper-duration/outlier filter.

ii.
```python
MIN_FRAMES_PER_TRIAL = 3
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
if len(neural_trials) < 2:
    continue
```

iii. The code header states the three-frame and two-trial rules. The trajectory says it saw no explicit trial exclusions and wanted at least minimally usable trials, but it did not investigate or justify long stopped-animal trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from each session file's `spks` list, with imaging planes concatenated along the neuron axis. `iarea` from retinotopy is used to select and label neurons.

ii.
```python
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. The agent identified `spks` as already-deconvolved calcium traces and verified that concatenated neuron order matched retinotopy labels.

## 2-b. How is the `neural` data processed?

i. Planes are concatenated, valid-area rows are selected, and trial-frame columns are sliced. No normalization, smoothing, padding, or temporal transformation is applied; the original numeric dtype is retained.

ii.
```python
spk_filtered = spk[neuron_indices, :]
neural_trial = spk_filtered[:, frames]
```

iii. The trajectory says the supplied activity is already Suite2p deconvolved output, so further deconvolution or fluorescence processing is unnecessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only neurons assigned to V1, mHV, lHV, or aHV are retained. Unmapped codes, including `-1` and `7`, are excluded; sessions with no retained neurons are skipped.

ii.
```python
valid_neurons = area_idx['V1'] | area_idx['mHV'] | area_idx['lHV'] | area_idx['aHV']
neuron_indices = np.where(valid_neurons)[0]
if len(neuron_indices) == 0:
    continue
```

iii. The agent explicitly chose the four visual-cortex groupings used by the paper's helper and noted that cell classification had already occurred upstream.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each variable-length trial begins with the first frame labeled as belonging to that trial and inside the textured corridor (corridor entry). It ends with that trial's last corridor frame and is neither padded nor truncated to a shared length.

ii.
```python
frames = np.where((ft_trInd == t) & ft_CorrSpc)[0]
neural_trial = spk_filtered[:, frames]
```

iii. The trajectory chose corridor entry as trial start and retained all corridor frames to preserve continuous temporal sampling and running-speed variation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The metadata bin size is the median difference in `ft` from the first indexed session, converted from MATLAB days to milliseconds (about 315 ms), and that single duration is reused for all sessions.

ii.
```python
dt_days = np.median(np.diff(ft))
time_bin_ms = dt_days * 24 * 3600 * 1000
'time_bin_size': float(time_bin_ms)
```

iii. The agent measured an approximately 3.17 Hz imaging rate and treated one imaging frame as one decoder bin.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr`, the selected absolute frame indices, and the single bin duration estimated from the first session's `ft`.

ii.
```python
sound_fr = SoundFr[t]
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The trajectory identified `SoundFr` as a fractional frame index and intended positive values before the cue and negative values after it.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every selected frame, its absolute frame number is subtracted from the cue frame and multiplied by a constant seconds-per-frame. Fractional `SoundFr` is preserved, but actual per-frame timestamps are not used or interpolated.

ii.
```python
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The agent chose a direct frame-distance conversion after estimating the imaging interval, with the sign convention required by “time to” the cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the `frames` used as neural columns, so it has one value per neural time point.

ii.
```python
neural_trial = spk_filtered[:, frames]
time_to_cue = (sound_fr - frames) * time_bin_s
```

iii. The trajectory emphasized keeping all streams on the same frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and the calendar date encoded by `datexp` for every deduplicated session.

ii.
```python
d = get_date(info['datexp'])
first_d = mouse_first_date[info['mname']]
days[key] = (d - first_d).days
```

iii. The agent considered session-number fields ambiguous and chose elapsed calendar days relative to each mouse's first recording.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Dates are parsed as datetimes, the earliest date per mouse is subtracted, and the resulting integer elapsed days are repeated at every time point of every trial in the session.

ii.
```python
return datetime.strptime(datexp, '%Y_%m_%d')
day_arr = np.full(n_frames, day_of_training, dtype=np.float32)
```

iii. The trajectory explicitly selected elapsed calendar time as its interpretation of training day.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses the selected frame indices, specifically the first retained corridor frame, plus the global seconds-per-frame estimate. It does not use raw `StartFr` directly.

ii.
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The agent regarded the first `ft_CorrSpc` frame as corridor entry/trial start and favored frame-level flags over fractional boundaries.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The first retained frame index is subtracted from every retained index and multiplied by the constant frame duration, forcing the first value to exactly zero.

ii.
```python
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The trajectory sought a simple elapsed-time series on the imaging grid.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. The elapsed-time values are generated from the same frame-index vector used to slice the neural matrix.

ii.
```python
neural_trial = spk_filtered[:, frames]
time_since_start = (frames - frames[0]) * time_bin_s
```

iii. The agent's stated goal was common frame-based alignment across streams.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from the trial-level `isRew` array.

ii.
```python
isRew = beh['isRew']
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. The trajectory identified `isRew` as the reward-availability flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The trial's value is cast to float and repeated across every time point; no other transformation is applied.

ii.
```python
reward_avail = np.full(n_frames, float(isRew[t]), dtype=np.float32)
```

iii. The agent treated reward availability as a binary per-trial contextual input and broadcast it to keep a uniform input matrix.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Categories are derived from the union of raw `UniqWalls` names, and each trial is labeled using its raw `WallName`.

ii.
```python
for name in beh['UniqWalls']:
    all_stim_names.add(name)
stim_to_idx = {name: i for i, name in enumerate(sorted(all_stim_names))}
stim_cat = stim_to_idx[WallName[t]]
```

iii. The trajectory chose `WallName` over masked analysis-specific `stim_id`, but decided to make every unique raw wall name a global class.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Raw wall names are sorted, assigned integer indices, and the per-trial index is repeated over all time points. Variants and shuffled versions are not collapsed to circle/leaf/rock/wood.

ii.
```python
stim_names = sorted(all_stim_names)
stim_arr = np.full(n_frames, stim_to_idx[WallName[t]], dtype=int)
```

iii. The agent wanted globally consistent labels across mice and sessions and repeated scalar outputs to produce a uniform `(n_output, n_timepoints)` array.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It uses `LickFr` for lick frame locations and `LickTrind` to restrict licks to the current trial.

ii.
```python
trial_lick_mask = (LickTrind.astype(int) == t)
trial_lick_frames = np.round(LickFr[trial_lick_mask]).astype(int)
```

iii. The trajectory inspected both arrays and concluded that a binary flag should be built by matching lick frames to trial frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are rounded to the nearest integer. A selected corridor frame is marked 1 if a rounded lick lands there, otherwise 0; multiple licks in one frame remain 1.

ii.
```python
lick_arr = np.zeros(n_frames, dtype=int)
frame_to_local = {f: i for i, f in enumerate(frames)}
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. The code header and trajectory explicitly say lick frames are rounded and converted to a per-frame binary output.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Absolute lick frames are mapped to local positions in the exact corridor-frame vector used for neural slicing.

ii.
```python
frame_to_local = {f: i for i, f in enumerate(frames)}
neural_trial = spk_filtered[:, frames]
```

iii. The agent chose direct frame-number matching because behavior and neural activity share the imaging-frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from frame-level `ft_Pos` at the selected corridor frames.

ii.
```python
ft_Pos = beh['ft_Pos'][:nfr]
pos_bins = discretize_position(ft_Pos[frames])
```

iii. Inspection showed corridor position spans 0–40 decimeters, matching a 4 m textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is integer-floor-divided by 10 decimeters and converted to integer labels, with results clipped to 0–3.

ii.
```python
bins = np.clip(positions // 10, 0, 3).astype(int)
```

iii. The agent interpreted the requested four equal 1 m bins as four 10-decimeter intervals.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The bins are `[0,10)`, `[10,20)`, `[20,30)`, and `[30,40]` decimeters, labeled 0–3; clipping sends any boundary/outlier values to the nearest valid category.

ii.
```python
np.clip(positions // 10, 0, 3).astype(int)
```

iii. This thresholding follows directly from the known 40 dm corridor length and requested 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is indexed with the same absolute `frames` used for the neural columns.

ii.
```python
neural_trial = spk_filtered[:, frames]
pos_bins = discretize_position(ft_Pos[frames])
```

iii. The agent established that frame-level behavior is already on the imaging grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed` at corridor frames. The global thresholds are calculated in a first pass over all indexed sessions, truncating each behavior stream to the first spike plane's frame count.

ii.
```python
corridor_speeds = ft_RunSpeed[ft_CorrSpc]
quartiles = np.percentile(np.concatenate(all_speeds), [25, 50, 75])
```

iii. The trajectory interpreted “four bins, each corresponding to 25% of the data” as global speed quartiles.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Three global percentile values are computed, then `np.digitize` assigns each retained speed to labels 0–3. Values tied at a threshold are placed together, so category counts need not each be 25%.

ii.
```python
def discretize_speed(speeds, quartiles):
    return np.digitize(speeds, quartiles)
```

iii. The agent wanted a single set of boundaries shared by all sessions and reported them in metadata.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Categories are separated by the global 25th, 50th, and 75th percentile speed values over all corridor frames, using NumPy's default right-open digitization convention.

ii.
```python
speed_quartiles = np.percentile(all_speeds, [25, 50, 75])
speed_bins = np.digitize(ft_RunSpeed[frames], speed_quartiles)
```

iii. The trajectory selected global quartile boundaries to make labels comparable across the complete dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Once thresholds are calculated, speed values from the same `frames` used for the neural columns are digitized one-for-one.

ii.
```python
neural_trial = spk_filtered[:, frames]
speed_bins = discretize_speed(ft_RunSpeed[frames], speed_quartiles)
```

iii. The agent retained all corridor frames so stationary periods and the full speed distribution remained temporally aligned.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Frame-level behavior is truncated to neural length. Too-short trials, sessions with fewer than two usable trials, sessions with no mapped neurons, and sessions whose spike or retinotopy load raises any exception are silently skipped after a message. Licks outside retained frames are ignored by membership testing. There is no explicit NaN cleanup.

ii.
```python
ft_trInd = beh['ft_trInd'][:nfr]
try:
    spk = load_spk(...)
except Exception as e:
    continue
if len(neural_trials) < 2:
    continue
```

iii. The trajectory noticed minor NaN-position differences in duplicate behavior files and treated them as effectively identical. The broad skip behavior was added defensively but was not substantively justified.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating the very large spike files dominates. It happens once in the speed-threshold prepass (although only to inspect a plane's shape) and again in full during session conversion; neural trial slicing/copying is also expensive.

ii.
```python
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
neural_trial = spk_filtered[:, frames]
```

iii. The trajectory's conversion stalled while loading the large data and the agent explicitly suspected large-file I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Trial-frame discovery scans all session frames once per trial; lick creation loops over lick events and builds Python sets/dictionaries per trial; subject lookup is a linear list search. These could be replaced by grouped indices, vectorized membership/index mapping, and a subject-to-index dictionary.

ii.
```python
for t in range(ntrials):
    frames = np.where((ft_trInd == t) & ft_CorrSpc)[0]
for lf in trial_lick_frames:
    if lf in frame_to_local:
        lick_arr[frame_to_local[lf]] = 1
```

iii. The trajectory did not discuss vectorization; these are evident implementation hotspots, though much smaller than spike-file I/O.

## 12-c. What processing does the code repeat multiple times?

i. Spike files are opened in both the speed-quartile prepass and main pass. Behavior files are loaded into one cache during the speed pass and then loaded into a new cache during stimulus/session processing. Trial masks are recomputed separately for every trial, and `LickTrind.astype(int)` is repeated inside that loop.

ii.
```python
beh_cache = {}  # in compute_speed_quartiles
...
beh_cache = {}  # recreated in main
trial_lick_mask = (LickTrind.astype(int) == t)
```

iii. The trajectory planned a two-pass global-quartile design and did not identify its repeated I/O and conversions as an optimization issue.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores unused `exp_info_entry`, assigns unused `nneu`, builds an unused `frame_set`, and repeatedly converts all `LickTrind` values. The speed pass deserializes spike arrays solely to obtain the first plane's frame count. Full raw wall variants are also collected as separate labels even though the reference downstream target uses four base textures.

ii.
```python
'exp_info_entry': s,
nneu, nfr = spk.shape
frame_set = set(frames.tolist())
spk_data = np.load(spk_path, allow_pickle=True).item()['spks']
```

iii. The trajectory contains no justification for these discarded intermediates; they are incidental implementation artifacts.
