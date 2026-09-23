# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from the same three directories: `beh/` for behavior, `spk/` for deconvolved calcium traces, and `retinotopy/` for visual area assignments. It first loads `beh/Imaging_Exp_info.npy` as the master index, then iterates over all 23 `Beh_<exp_type>.npy` files in a single pass to build a behavior dictionary keyed by session. Spike data and retinotopy are loaded per session. The AI merges duplicate behavior keys (swap1/swap2 suffixes) by taking the union of their `stim_id` mappings.

ii.
```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

# Single pass over all behavior files
for f in sorted(glob.glob(os.path.join(BEH_DIR, 'Beh_*.npy'))):
    B = np.load(f, allow_pickle=True).item()
    for k, beh in B.items():
        base = '_'.join(k.split('_')[:5])
        if base not in beh_by_session:
            beh_by_session[base] = beh
        sid = np.asarray(beh['stim_id'], dtype=float)
        for w, s in zip(list(beh['UniqWalls']), sid):
            if not np.isnan(s):
                stim_map[base][str(w)] = int(s)

# Per session: spikes and retinotopy
planes = np.load(path, allow_pickle=True).item()['spks']
iarea = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (mname, datexp)))['iarea']
```

iii. The AI loads all behavior files in one pass for efficiency and merges the stim_id labellings across experiment types so that swap stimuli are properly labeled.

## 1-b. How are the data split into subjects?

i. Mouse name (`mname`) is extracted from the session index entries. Subjects are the sorted unique mouse names across all sessions.

ii.
```python
sessions = {}
for exp_type, dbs in exp_info.items():
    for db in dbs:
        key = session_key(db['mname'], db['datexp'], db['blk'])
        rec = sessions.setdefault(key, {'mname': db['mname'], ...})

subjects = sorted(set(r['mname'] for r in results))
```

iii. The mouse name is directly available from the index entries.

## 1-c. How are the data split into sessions?

i. A session is defined as one unique `(mname, datexp, blk)` combination, yielding 89 unique sessions. When the same session appears under multiple experiment types, the entries are merged (keeping one behavior dict and the union of stim_id mappings).

ii.
```python
def session_key(mname, datexp, blk):
    return '%s_%s_%s' % (mname, datexp, blk)

sessions = {}
for exp_type, dbs in exp_info.items():
    for db in dbs:
        key = session_key(db['mname'], db['datexp'], db['blk'])
        rec = sessions.setdefault(key, {...})
        rec['exp_types'].append(exp_type)
```

iii. Sessions are deduplicated by their composite key; duplicate entries across experiment types are merged rather than duplicated.

## 1-d. How are the data split into trials?

i. Trials are defined by `ft_trInd` (trial index per frame). The AI applies a **running filter**: only frames where `ft_move > 0` (mouse is running) AND `ft_CorrSpc` (inside the textured corridor) are kept. Frames are grouped by trial index, sorted, and split at boundaries. This means stationary frames within a trial are excluded, creating potentially non-contiguous frame selections per trial.

ii.
```python
ft_corr = np.asarray(beh['ft_CorrSpc'], dtype=bool)[:nfr]
ft_move = np.asarray(beh['ft_move'], dtype=float)[:nfr] > 0

valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))

idx_valid = np.where(valid)[0]
tr_valid = ft_trind[idx_valid].astype(int)
order = np.argsort(tr_valid, kind='stable')
# ... split at boundaries
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
    tr = int(tt[0])
    fr = np.sort(fr)
```

iii. The AI justifies the running filter by citing the reference paper ("We only considered timepoints during running for analysis") and the reference code's `fr_valid = VRmove & isCorridor` mask used in `Get_dprime_selective_neuron`.

## 1-e. How are trials filtered based on quality controls?

i. Trials are dropped if they have fewer than 5 valid (running + corridor) frames (`MIN_FRAMES_PER_TRIAL = 5`), if their wall name has no canonical `stim_id` mapping (309 `circle3` trials), or if the sound cue time or trial start time is NaN. There is no trial length outlier filter (no 99th percentile cutoff).

ii.
```python
MIN_FRAMES_PER_TRIAL = 5

if len(fr) < MIN_FRAMES_PER_TRIAL:
    n_drop_short += 1
    continue
wname = str(wall[tr])
if wname not in smap:
    n_drop_stim += 1
    continue
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]):
    n_drop_cue += 1
    continue
```

iii. The AI documents that 309 `circle3` trials are dropped because they have no canonical `stim_id` in the dataset. No outlier-length filter is applied.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy` (list of per-plane arrays, concatenated) and `iarea` from `retinotopy/<mname>_<datexp>_trans.npz` for area assignment.

ii.
```python
planes = np.load(path, allow_pickle=True).item()['spks']
sizes = np.array([p.shape[0] for p in planes])
# ... concatenation via plane offsets
iarea = ret['iarea']
```

iii. Same source as the reference code's `load_spk` and `load_retino`.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps not in the reference conversion: (1) subsamples to 1,000 neurons per session (seeded, proportionally across visual areas), (2) z-scores each neuron over the included (running + corridor) frames, and (3) stores as float32. The reference stores raw deconvolved traces as float16 with all visual-cortex neurons.

ii.
```python
N_NEURONS_PER_SESSION = 1000

# Neuron subsampling
rows, area_idx = select_neurons(iarea, N_NEURONS_PER_SESSION, rng)
spk, n_spk_neurons = load_spk_rows(key, rows)

# Z-scoring
mu = spk[:, valid].mean(axis=1, keepdims=True)
sd = spk[:, valid].std(axis=1, keepdims=True)
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
spk = (spk - mu) / np.where(sd > 0, sd, 1.0)
spk = spk[keep_neu]
```

iii. The AI justifies subsampling because the full dataset is 405 GB and the decoder projects to 100 PCs anyway. Z-scoring is justified by reference to `get_kfold_reward_response` in the reference code which z-scores traces before population analyses.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside visual cortex (iarea in {-1, 7}) are excluded, same as the reference. Additionally, zero-variance neurons are removed after z-scoring. Neurons are then subsampled to 1,000 per session with proportional stratification across V1/mHV/lHV/aHV.

ii.
```python
AREA_OF_IAREA = {8: 0, 0: 1, 1: 1, 2: 1, 9: 1, 5: 2, 6: 2, 3: 3, 4: 3}

def select_neurons(iarea, n_target, rng):
    area_idx = np.full(len(iarea), -1, dtype=np.int64)
    for ia, a in AREA_OF_IAREA.items():
        area_idx[iarea == ia] = a
    valid = np.where(area_idx >= 0)[0]
    if len(valid) <= n_target:
        return valid, area_idx[valid]
    # proportional allocation across areas
    counts = np.array([(area_idx[valid] == a).sum() for a in range(len(AREA_NAMES))])
    alloc = np.floor(counts / counts.sum() * n_target).astype(int)
    # ... random sampling
```

iii. The area mapping matches the reference's `neu_area_ID`. Subsampling is justified by the 405 GB data size.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry. However, because of the running filter (`ft_move > 0`), the neural data only includes frames where the mouse was running. This means frames within a trial may not be temporally contiguous -- pauses within a corridor traversal create gaps in the time series.

ii.
```python
valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
# ...
'neural': np.ascontiguousarray(spk[:, fr]),
```

iii. The AI states alignment is to corridor entry and cites the reference `fr_valid` mask. The non-contiguous frames are a consequence of the running filter.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied. The imaging frame rate (~3.17 Hz, ~315 ms per bin) is the temporal resolution. The time_bin_size in metadata is computed as the mean dt across sessions.

ii.
```python
dt_ms = float(np.mean([r['dt'] for r in results]) * 1000.0)
# where r['dt'] = float(np.median(np.diff(ft)) * SEC_PER_DAY)
```

iii. Same native imaging frame rate as the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `beh['SoundTime']` (MATLAB datenum of the sound cue for each trial) and `beh['ft']` (MATLAB datenum of each imaging frame).

ii.
```python
t_sound = np.asarray(beh['SoundTime'], dtype=float)
ft = np.asarray(beh['ft'], dtype=float)[:nfr]
```

iii. The AI uses the time-based variables (SoundTime, ft) rather than the frame-based variables (SoundFr) used by the reference.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The time to the sound cue is computed as `(SoundTime[trial] - ft[frame]) * SEC_PER_DAY`, converting from MATLAB datenum (days) to seconds. Positive values mean the cue is in the future, negative means it has already occurred.

ii.
```python
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. Direct time arithmetic in the MATLAB datenum domain, converted to seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The time_to_cue is computed for the same frame indices (`fr`) used for the neural data of that trial.

ii.
```python
trials_out.append({
    'frames': fr,
    'neural': np.ascontiguousarray(spk[:, fr]),
    'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
})
```

iii. All signals use the same frame indices, so alignment is automatic.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the date string in the session key), parsed as a calendar date. The first imaging date of each mouse is found, and day_of_training is the number of calendar days elapsed since that first date.

ii.
```python
def parse_date(datexp):
    y, m, d = datexp.split('_')
    return date(int(y), int(m), int(d))

first_day = {}
for r in results:
    d = parse_date(r['datexp'])
    if r['mname'] not in first_day or d < first_day[r['mname']]:
        first_day[r['mname']] = d

day = (parse_date(r['datexp']) - first_day[r['mname']]).days
```

iii. The AI computes actual calendar days elapsed rather than an ordinal session count.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days since the first session of that mouse. This differs from an ordinal count: e.g., if sessions are on days 1, 5, and 10, the values are 0, 4, 9 rather than 0, 1, 2. The value is per-trial, broadcast across all timepoints.

ii.
```python
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
inp = np.stack([
    ...
    np.full(T, float(day), dtype=np.float32),
    ...
])
```

iii. The AI does not explicitly justify calendar days vs ordinal count.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `beh['Trial_start_time']` (MATLAB datenum of corridor entry for each trial) and `beh['ft']` (frame timestamps).

ii.
```python
t_start = np.asarray(beh['Trial_start_time'], dtype=float)
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. Uses the time-based variable `Trial_start_time` rather than the frame-based `StartFr`.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Computed as `(ft[frame] - Trial_start_time[trial]) * SEC_PER_DAY`, giving seconds since corridor entry. Positive values (increasing within a trial).

ii.
```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. Direct time difference converted to seconds.

## 5-c. How is `input` *Time since trial start* aligned with the neural data?

i. Computed for the same frame indices as the neural data.

ii.
```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. Same frame-based alignment as all other signals.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `beh['isRew']`, which marks whether each trial's corridor is the rewarded one.

ii.
```python
is_rew = np.asarray(beh['isRew'], dtype=bool)
'is_rew': int(is_rew[tr]),
```

iii. Directly from the behavior data.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to int (0 or 1) and broadcast across all timepoints of the trial.

ii.
```python
np.full(T, float(t['is_rew']), dtype=np.float32),
```

iii. No processing needed; it is 0 for all trials in unsupervised and naive mice.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `beh['WallName']` (wall texture name per trial), `beh['stim_id']` and `beh['UniqWalls']` (mapping wall names to the paper's 7 canonical stimulus IDs).

ii.
```python
sid = np.asarray(beh['stim_id'], dtype=float)
for w, s in zip(list(beh['UniqWalls']), sid):
    if not np.isnan(s):
        stim_map[base][str(w)] = int(s)
# ...
wname = str(wall[tr])
if wname not in smap:
    n_drop_stim += 1
    continue
```

iii. The AI uses the paper's role-based canonical stimulus labeling system from `stim_id`.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Wall names are mapped to 7 canonical stimulus IDs (circle1, circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2) via the `stim_id` mapping. This is a role-based labeling where rock fills the circle role and wood/brick fills the leaf role. Trials with unmapped wall names (309 `circle3` trials) are dropped. The value is per-trial, broadcast across all timepoints.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']

'output_values': [CANONICAL_STIM, ...]

np.full(T, t['stim'], dtype=np.int64),
```

iii. The AI states this follows the paper's pooling convention: "we denote the stimuli as leaf and circle, even though other visual stimuli were also used."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `beh['LickFr']`, the neural frame index of each lick event.

ii.
```python
lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
lick_bin = np.zeros(nfr, dtype=np.int64)
li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
li = li[(li >= 0) & (li < nfr)]
lick_bin[li] = 1
```

iii. Directly from the lick frame data.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary per-frame variable: 1 if at least one lick falls in that frame, 0 otherwise. Lick frames are floored to integer frame indices and out-of-range values are dropped. NaN lick frames are filtered out.

ii.
```python
lick_bin = np.zeros(nfr, dtype=np.int64)
if lick_fr.size:
    li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
    li = li[(li >= 0) & (li < nfr)]
    lick_bin[li] = 1
```

iii. Same approach as the reference; handles edge cases (NaN, out-of-range) more explicitly.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes neural frames, and the lick binary is indexed by the same frame indices used for neural data.

ii.
```python
'lick': lick_bin[fr],
```

iii. Frame-based alignment, same as all other signals.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `beh['ft_Pos']`, the position in the corridor at each imaging frame, in decimeters.

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
'pos_dm': ft_pos[fr],
```

iii. Directly from the per-frame position data.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is discretized into 4 bins using `np.digitize` with edges at 10, 20, 30 decimeters (corresponding to 1, 2, 3 meters).

ii.
```python
POS_BIN_EDGES_DM = np.array([10.0, 20.0, 30.0])
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. The 4 bins correspond to 0-1m, 1-2m, 2-3m, 3-4m as required by the instructions.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. 4 equal-length 1-meter bins: [0-10dm) -> bin 0, [10-20dm) -> bin 1, [20-30dm) -> bin 2, [30-40dm) -> bin 3. Uses `np.digitize` with edges [10, 20, 30].

ii.
```python
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. Equivalent to the reference's `ft_Pos // 10` approach.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` provides one position per imaging frame, and the same frame indices are used for position and neural data.

ii.
```python
'pos_dm': ft_pos[fr],
```

iii. Frame-based alignment.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `beh['ft_RunSpeed']`, the running speed at each imaging frame in cm/s.

ii.
```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
'speed': ft_speed[fr],
```

iii. Directly from the per-frame speed data.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Running speed is discretized into 4 bins using **global** value-based quartiles computed across all included (running) timepoints of all sessions. `np.quantile` computes the 25th, 50th, and 75th percentile values, and `np.digitize` assigns each speed to a bin.

ii.
```python
all_speed = np.concatenate([np.concatenate([t['speed'] for t in r['trials']])
                            for r in results])
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])

spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The AI computes quartiles globally so each bin holds 25% of all data, as the instruction requires. The reference computes quartiles per session using rank-based ordering.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Value-based quartiles from `np.quantile` across all sessions: speeds below the 25th percentile -> bin 0 (slowest), 25-50th -> bin 1, 50-75th -> bin 2, above 75th -> bin 3 (fastest).

ii.
```python
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. Since the running filter excludes stationary frames, there are fewer ties at zero speed, making value-based quartiles more balanced than they would be without the filter.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` provides one speed per imaging frame, and the same frame indices are used for speed and neural data.

ii.
```python
'speed': ft_speed[fr],
```

iii. Frame-based alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple edge cases are handled: behavior arrays are truncated to the number of neural frames (`nfr`); NaN values in `ft_trInd`, `ft_Pos`, and `ft_RunSpeed` are excluded by the `np.isfinite` check in the valid mask; `LickFr` is handled with `np.atleast_1d` for empty arrays and NaN filtering; lick frames outside [0, nfr) are dropped; trials with NaN cue or start times are dropped; sessions with < 2 usable trials would be dropped (none occurred).

ii.
```python
nfr = min(spk.shape[1], len(beh['ft']))
spk = spk[:, :nfr]
ft = np.asarray(beh['ft'], dtype=float)[:nfr]

valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))

lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
li = li[(li >= 0) & (li < nfr)]
```

iii. The AI documents that behavior can run past imaging, and handles empty/NaN lick data for non-water-restricted mice.

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (405 GB total). The AI mitigates this by only loading the sampled neuron rows and using multiprocessing.

ii.
```python
def load_spk_rows(key, rows):
    planes = np.load(path, allow_pickle=True).item()['spks']
    # ... only copies the sampled rows
```

iii. I/O is the bottleneck; the AI uses parallel processing and selective row loading.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI vectorized trial segmentation using sort + split instead of looping over trials:

ii.
```python
tr_valid = ft_trind[idx_valid].astype(int)
order = np.argsort(tr_valid, kind='stable')
tr_sorted = tr_valid[order]
idx_sorted = idx_valid[order]
bounds = np.where(np.diff(tr_sorted) != 0)[0] + 1
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
    ...
```

iii. The vectorized approach avoids scanning the entire frame index once per trial.

## 12-c. What processing does the code repeat multiple times?

i. The behavior files are loaded once and shared across sessions via multiprocessing fork (copy-on-write). No redundant processing is identified.

ii. N/A

iii. N/A

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI adds an extra input variable `sound_cue_onset` (binary, 1 on the first frame at/after the sound cue) that was not specified in the decoder task instructions. This is a 5th input beyond the 4 requested.

ii.
```python
cue_onset = np.zeros(T, dtype=np.float32)
after = np.where(t['time_to_cue'] <= 0)[0]
if after.size:
    cue_onset[after[0]] = 1.0
inp = np.stack([
    t['time_to_cue'].astype(np.float32),
    cue_onset,  # extra input not in instructions
    np.full(T, float(day), dtype=np.float32),
    t['time_since_start'].astype(np.float32),
    np.full(T, float(t['is_rew']), dtype=np.float32),
])
```

iii. The AI justifies this as providing "a scale-free version of the cue timing" to supplement the time_to_sound_cue input.
