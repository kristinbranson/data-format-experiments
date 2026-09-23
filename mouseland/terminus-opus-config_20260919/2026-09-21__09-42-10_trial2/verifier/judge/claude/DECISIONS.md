# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `beh/Imaging_Exp_info.npy` as the master index listing every recording grouped by experiment type. Each `Beh_<exp_type>.npy` behaviour file is loaded once for all sessions it contains (grouped by exp_type). Neural data are loaded from `spk/<session_id>_neural_data.npy` and retinotopy from `retinotopy/<mname>_<datexp>_trans.npz`. The loading is done in two passes: first behaviour (pass 1), then neural (pass 2) with multiprocessing.

ii.
```python
exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type),
            allow_pickle=True).item()
fn = os.path.join(DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
spks = np.load(fn, allow_pickle=True).item()['spks']
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
              allow_pickle=True)
```

iii. The AI documented in CONVERSION_NOTES.md that the data organization mirrors the Figshare layout expected by the reference code, with `beh/`, `spk/`, and `retinotopy/` subdirectories.

## 1-b. How are the data split into subjects?

i. The mouse name (`mname`) is taken from the session key tuple and used to identify subjects. The sorted unique mouse names form the subjects list. 19 mice are identified.

ii.
```python
def session_key(db):
    return (db['mname'], db['datexp'], str(db['blk']))
...
subjects = sorted({k[0] for k, _, _, _ in sessions})
data['subject_idx'].append(subjects.index(key[0]))
```

iii. The AI verified 19 subjects matching the paper's "19 mice."

## 1-c. How are the data split into sessions?

i. A session is identified by the tuple (mname, datexp, blk). The `unique_sessions()` function deduplicates across the 23 experiment types, keeping only the first occurrence, yielding 89 unique sessions.

ii.
```python
def unique_sessions(exp_info):
    reps = {}
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            key = session_key(db)
            if key not in reps:
                reps[key] = (exp_type, db)
    ordered = sorted(reps.keys())
    return [(k, reps[k][0], reps[k][1], exp_types[k]) for k in ordered]
```

iii. The AI verified that 142 database entries across 23 experiment types map to 89 unique sessions, matching the paper's "89 recordings." Behaviour was confirmed identical across duplicate entries.

## 1-d. How are the data split into trials?

i. For each trial, frames are selected where `ft_trInd == t` AND `ft_CorrSpc` (inside the 4 m texture corridor) AND `ft_move > 0` (VR moving, i.e. the mouse ran above the 6 cm/s threshold). Frames with NaN `ft_trInd` are also excluded via `np.isfinite(ftr)`. Only running corridor frames are retained.

ii.
```python
move = beh['ft_move'][:nfr_beh] > 0
corr = beh['ft_CorrSpc'][:nfr_beh].astype(bool)
ftr = beh['ft_trInd'][:nfr_beh].astype(float)
valid = move & corr & np.isfinite(ftr)
vidx = np.where(valid)[0]
vtr = ftr[vidx].astype(int)
```

iii. The AI justified this by referencing the paper's statement "We only considered timepoints during running for analysis" and the reference code's `fr_valid = (ft_move>0) & ft_CorrSpc` mask used in `Get_dprime_selective_neuron`. The AI noted this also removes long stationary periods when mice stop to collect rewards.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have 0 retained running frames (empty after the `ft_move > 0` + `ft_CorrSpc` filter), or fewer than `MIN_FRAMES_PER_TRIAL = 5` retained frames, or if their frames fall beyond the neural recording length. Sessions with fewer than 2 usable trials are also skipped. No percentile-based length filter is applied.

ii.
```python
MIN_FRAMES_PER_TRIAL = 5
...
for t in range(ntrials):
    frames = vidx[lo[t]:hi[t]]
    if len(frames) == 0:
        continue
    trials.append(...)
...
frames = tr['frames'][tr['frames'] < nfr_spk]
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
```

iii. The AI noted in the conversion output that 0 trials were dropped across the full dataset (38,110 kept of 38,110 recorded), because the running-frame filter itself handles the very long stationary trials by removing the non-running frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in the `spk/<session_id>_neural_data.npy` files, which contain a list of per-imaging-plane arrays concatenated along axis 0 (as in `utils.load_spk`). The visual area of each neuron comes from `iarea` in the retinotopy files.

ii.
```python
fn = os.path.join(DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
spks = np.load(fn, allow_pickle=True).item()['spks']
...
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)))
iarea = ret['iarea']
```

iii. The AI documented that the neural data are suite2p deconvolved traces, matching the paper's "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. The deconvolved traces are not further processed (no z-scoring, no dF/F). However, the AI **subsamples neurons to at most 2,000 per session**, stratified proportionally across the four brain areas, using a fixed random seed (SEED=2025 + session index). The data are stored as float32.

ii.
```python
MAX_NEURONS = 2000
...
def select_neurons(iarea, rng):
    masks = neu_area_ID(iarea)
    per_region = [np.where(masks[r])[0] for r in BRAIN_REGIONS]
    counts = np.array([len(x) for x in per_region])
    total = counts.sum()
    if total > MAX_NEURONS:
        alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
        ...
        pick = rng.choice(per_region[r], size=alloc[r], replace=False)
    ...
```

iii. The AI justified the 2,000 neuron cap by noting the full dataset is ~183 GB of running-frame activity and that "the reference decoder projects each session onto 100 PCs and itself random-projects to at most 2,000 neurons for its SVD initialisation, so 2,000 neurons/session is decoder-saturating."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their `iarea` maps to one of the four visual areas (V1=8, mHV=0,1,2,9, lHV=5,6, aHV=3,4). Neurons with `iarea` in {-1, 7} (12.5% of neurons) are dropped. Then the remaining neurons are subsampled to at most 2,000 per session.

ii.
```python
def neu_area_ID(iarea):
    idx = {}
    for ar in area_name:
        if ar == 'V1':
            idx[ar] = iarea == 8
        elif ar == 'mHV':
            idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV':
            idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV':
            idx[ar] = (iarea == 3) | (iarea == 4)
    return idx
```

iii. The AI copied the `neu_area_ID` function directly from `utils.py` and noted that "neurons with `iarea` outside those codes are never used by the reference analyses and have no area label, so those neurons are dropped."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start, `StartFr`). Trials have variable length because mice run at different speeds. Each trial starts at corridor entry and ends at corridor exit, containing only the running frames within the texture corridor. No fixed-length window is imposed; `off_start = 0.0`, `off_end = None`.

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
...
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
...
data['metadata'] = dict(
    temporal_alignment_event='trial start = entry into the virtual-reality corridor',
    off_start=0.0,
    off_end=None,
)
```

iii. The AI noted that trials end at corridor exit and have variable duration (median ~21 frames ~ 6.6 s).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame interval (~314.7 ms, 3.178 Hz) is the time bin, computed as the median of `np.diff(ft)` across sessions.

ii.
```python
dts = np.array([s['dt'] for s in session_info])
data['metadata'] = dict(
    time_bin_size=float(np.median(dts) * 1000.0),
    frame_rate_hz=float(1.0 / np.median(dts)),
)
```

iii. The AI noted the frame rate matches the value stated in `data_process_script.ipynb` ("fs = 3.17 Hz").

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional frame index of the sound cue for each trial) and `ft` (the timestamp of every imaging frame, in MATLAB datenums/days).

ii.
```python
t_frame = beh['ft'] * 86400.0  # datenum (days) -> seconds
t_cue = np.interp(beh['SoundFr'], frame_idx, t_frame)
```

iii. Documented in the variable mapping table in CONVERSION_NOTES.md Step 5.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `SoundFr` is a fractional frame index, so it is interpolated onto the absolute frame time axis. The input is computed as `t_cue - t_frame` (cue time minus frame time), so it is positive before the cue and negative after.

ii.
```python
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```

iii. The AI referenced `utils.spk_2_cue` (cue alignment in the reference code) as the source of the cue-frame relationship.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The same frame indices (`frames`) used for the neural data are used to index `t_frame`, so the time-to-cue values are naturally aligned with the neural data on a per-frame basis.

ii.
```python
frames = vidx[lo[t]:hi[t]]
...
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32),
```

iii. "behaviour is natively on the neural-frame clock (`ft_*`), so alignment = indexing by neural frame."

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the session date (`datexp` field, e.g. `2023_03_25`), computing calendar days since each mouse's first imaging session. The first session dates are computed over all 89 sessions.

ii.
```python
def first_session_dates(all_sessions):
    first_date = {}
    for key, _, _, _ in all_sessions:
        m, date = key[0], key[1]
        d = datetime.date(*map(int, date.split('_')))
        if m not in first_date or d < first_date[m]:
            first_date[m] = d
    return first_date
...
day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
```

iii. The AI noted that `exp_info['days']` exists for only 8 of 142 entries, so it cannot be used as a dataset-wide axis. Calendar days since first imaging session was chosen as a continuous training-day proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The calendar date of each session is parsed from the session key, and the number of days since that mouse's first imaging session is computed. This value is a per-trial scalar broadcast across all time bins. The range is 0-92 days.

ii.
```python
day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
...
day_of_training=np.float32(day_of_training),
...
inp[1] = tr['day_of_training']  # broadcast scalar
```

iii. The AI initially computed day_of_training from only the selected sessions (bug found in Step 7), then fixed it to use all 89 sessions so values are consistent between `--sample` and `--full` runs.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (the fractional frame index of corridor entry for each trial) and `ft` (frame timestamps).

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
```

iii. Documented in CONVERSION_NOTES.md Step 5 variable mapping.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `StartFr` is interpolated onto the absolute frame time axis. The input is `t_frame - t_start` (frame time minus start time), giving elapsed time in seconds since corridor entry. It is always >= 0.

ii.
```python
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```

iii. The AI asserts `time_since_trial_start >= 0` as a sanity check.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame indices as the neural data are used to compute `t_frame[frames]`.

ii.
```python
frames = vidx[lo[t]:hi[t]]
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32),
```

iii. All data streams are aligned by neural frame index.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, which marks whether each trial was in the rewarded corridor.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(bool)
...
reward_available=np.float32(1.0 if is_rew[t] else 0.0),
```

iii. The AI noted that `isRew` is exactly the set of trials in the rewarded wall, and that 61 of 89 sessions have no rewarded trials (unsupervised/naive mice).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast from boolean to float (1.0 for rewarded, 0.0 otherwise). It is a per-trial scalar broadcast across all time bins.

ii.
```python
inp[3] = tr['reward_available']  # scalar broadcast
```

iii. No processing beyond type conversion.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, which names the wall texture for each trial.

ii.
```python
wall = np.asarray(beh['WallName'])
stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
```

iii. The AI noted that `TrialStim` is masked in swap sessions, so `WallName` is the reliable source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 distinct wall names are mapped to 4 texture families: circle, leaf, rock, and brick (the AI uses 'brick' for what the data files call 'wood'). The `texture_family()` function strips numeric suffixes and replaces 'wood' with 'brick'. The category index is per-trial and broadcast across all time bins.

ii.
```python
STIM_CATEGORIES = ['circle', 'leaf', 'rock', 'brick']

def texture_family(wall_name):
    base = wall_name.split('_')[0]
    base = ''.join(ch for ch in base if not ch.isdigit())
    if base == 'wood':
        base = 'brick'
    return base
...
out[0] = tr['stim_cat']  # per-trial scalar broadcast
```

iii. The AI justified using the paper's terminology ('brick') rather than the data's ('wood'), citing the paper: "four large texture images: circle, leaf, rock and brick."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the frame indices of every lick in the session.

ii.
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
```

iii. The AI referenced `utils.spk_2_cue`/`spk_2_firstLick` which bin licks by frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A frame is marked as 1 (licking) if at least one lick falls in it, 0 otherwise. Lick frame indices are floored to integer frames. NaN/infinite values in LickFr are filtered out, and licks beyond the recording are dropped.

ii.
```python
lick_bin = np.zeros(nfr_beh, dtype=bool)
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
...
lick=lick_bin[frames].astype(np.int64),
```

iii. The AI noted that licking is identically 0 in the 61 unsupervised/naive sessions (mice were not water restricted), which is correct rather than missing data.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes the imaging frames, so `lick_bin` is on the same frame grid as the neural data. The same `frames` array is used to index both.

ii.
```python
lick=lick_bin[frames].astype(np.int64),
```

iii. Aligned by frame index.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the position in the corridor at each imaging frame, in decimeters (0-40 dm in the texture, up to 60 dm through grey space).

ii.
```python
pos = beh['ft_Pos'][:nfr_beh].astype(float)
```

iii. Documented in CONVERSION_NOTES.md.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position in decimeters is divided by 10 (the bin width corresponding to 1 m), floored, and clipped to 0-3, yielding 4 equal-length 1-m spatial bins over the 4-m texture corridor.

ii.
```python
TEXTURE_LENGTH_DM = 40.0
N_POS_BINS = 4
...
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
```

iii. Position bins are 0-1m, 1-2m, 2-3m, 3-4m as specified in the decoder task.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `floor(position_dm / 10)` with clipping to [0, 3]. Boundaries are at 1m, 2m, and 3m. The distribution is approximately uniform (0.250/0.249/0.250/0.252) because the VR moves at constant speed while the mouse is running.

ii.
```python
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
...
np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)), 0, N_POS_BINS - 1)
```

iii. Documented in the conversion output.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is on the neural frame clock, so the same frame indices are used.

ii.
```python
pos=pos[frames],  # in process_behaviour, same frames as neural
```

iii. All data streams are aligned by frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse at each imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr_beh].astype(float)
```

iii. Documented in CONVERSION_NOTES.md Step 5.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. **Global** speed quartile edges are computed from the 25th, 50th, and 75th percentiles of running speed across ALL retained frames from ALL sessions. These edges are then applied per-frame using `np.digitize`. Each bin holds approximately 25% of all data points.

ii.
```python
speeds = np.concatenate([tr['speed'] for res in beh_results for tr in res['trials']])
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. The AI computed global quartile edges [12.422, 25.353, 40.855] cm/s across 821,579 retained frames. The output distribution confirmed exact 25%/25%/25%/25%.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Using `np.digitize` with 3 global edges, producing bins 0-3. Values below the 25th percentile go to bin 0, values above the 75th percentile go to bin 3.

ii.
```python
SPEED_VALUES = ['Q1 (slowest 25%)', 'Q2', 'Q3', 'Q4 (fastest 25%)']
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. The AI noted the global approach ensures identical speed bins between `--sample` and `--full` runs and across all sessions.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is on the neural frame clock, so the same frame indices are used.

ii.
```python
speed=speed[frames],  # same frames as neural
```

iii. Aligned by frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple edge cases are handled: (1) Behaviour arrays longer than the neural recording are truncated to the neural frame count. (2) NaN values in `ft_trInd` are excluded via `np.isfinite(ftr)`. (3) NaN/Inf values in `LickFr` are filtered. (4) Lick frames beyond the recording are dropped. (5) Fractional frame indices (`StartFr`, `SoundFr`, `LickFr`) are handled by interpolation or flooring. (6) Sessions appearing in multiple experiment types are converted once. (7) Sessions with `stimtype` (swap sessions) have the suffix included in the behaviour key.

ii.
```python
valid = move & corr & np.isfinite(ftr)
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
frames = tr['frames'][tr['frames'] < nfr_spk]
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10 Check 5, noting "behaviour is 1-3 frames longer -> truncate to `nfr = spk.shape[1]` (as the reference does)."

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files (89 files totalling 405 GB). The neural pass takes ~40 seconds with 8 parallel workers. The entire conversion runs in under 1 minute.

ii.
```python
fn = os.path.join(DATA_ROOT, 'spk', '%s_%s_%s_neural_data.npy' % (mname, datexp, blk))
spks = np.load(fn, allow_pickle=True).item()['spks']
```

iii. The AI measured I/O throughput at ~2.2 GB/s and noted this is the fundamental bottleneck.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI **already vectorized** the per-trial frame grouping using `np.searchsorted` instead of per-trial boolean masks. The per-trial loop for assembling inputs/outputs could potentially be further vectorized but is not the bottleneck.

ii.
```python
order = np.argsort(vtr, kind='stable')
vidx, vtr = vidx[order], vtr[order]
lo = np.searchsorted(vtr, np.arange(ntrials), side='left')
hi = np.searchsorted(vtr, np.arange(ntrials), side='right')
```

iii. The AI noted "~100x faster than per-trial boolean masks" in the CONVERSION_NOTES.md.

## 12-c. What processing does the code repeat multiple times?

i. The behaviour files are loaded once per experiment type (not per session), avoiding redundant reads. Global speed edges are computed once in the behaviour pass. No significant repeated processing was identified.

ii.
```python
by_exp = collections.defaultdict(list)
for i, (key, exp_type, db, ets) in enumerate(sessions):
    by_exp[exp_type].append(i)
for exp_type in sorted(by_exp):
    B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type), ...)
```

iii. The AI structured the code to minimize redundant file reads.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI subsamples neurons to 2,000 per session with stratified random selection. This is extra processing not performed by the reference code. Additionally, the `ft_move > 0` filter involves computing and applying a running mask that the reference solution does not use.

ii.
```python
if total > MAX_NEURONS:
    alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
    ...
    pick = rng.choice(per_region[r], size=alloc[r], replace=False)
```

iii. The AI justified neuron subsampling as necessary for dataset size but this is an additional processing step.
