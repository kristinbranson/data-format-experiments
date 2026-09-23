# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads `Imaging_Exp_info.npy` as the master index, then loads **all** 23 `Beh_<exp_type>.npy` files upfront in `load_session_registry()`, collapsing the 142 experiment-type records into 89 unique recordings. For each unique recording, it merges `stim_id` across experiment types (taking the union, asserting no conflicts). The spike files and retinotopy files are loaded later per-session in `process_session()`.

ii. Loading the registry and all behavior files:
```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
# ...
for exp_type, db in exp_info.items():
    beh_all = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
                      allow_pickle=True).item()
    for d in db:
        key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
        # merge stim_id across experiment types
```

Loading neural data per session:
```python
path = os.path.join(root, 'spk', '%s_%s_%s_neural_data.npy'
                    % (entry['mname'], entry['datexp'], entry['blk']))
d = np.load(path, allow_pickle=True).item()
```

iii. The AI's CONVERSION_NOTES document that "The 142 records collapse to 89 unique recordings" and that behavior payloads are byte-identical across experiment-type views, with only `stim_id` differing. The AI merges `stim_id` across views to recover a complete canonical stimulus identity per trial.

## 1-b. How are the data split into subjects?

i. The mouse name is taken from `mname` in the experiment info records. After building the registry, subjects are the sorted unique mouse names across all sessions selected for processing.

ii.
```python
subjects = sorted({sess_beh[k]['mname'] for k in keys})
data['subject_idx'].append(subjects.index(sb['mname']))
```

iii. The AI notes "19 unique names" from the data, matching the paper's "19 mice."

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. The 142 experiment-type records are collapsed to 89 unique recordings. All 89 sessions are included (none are dropped based on quality).

ii.
```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
# ...
if key not in reg:
    reg[key] = dict(...)
else:
    # merge stim_id, assert consistency
```

iii. CONVERSION_NOTES: "89 unique recordings from 142 experiment-type records."

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal (one value of `ft_trInd`). Within each trial, only samples satisfying `ft_CorrSpc & (ft_move > 0)` are retained -- i.e., inside the 0-4 m texture region AND the VR was moving. The AI additionally applies a boundary guard requiring samples to fall between `StartFr` and `GrayFr` (with 1 frame of slack) to remove 21 mislabelled edge samples.

ii.
```python
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
kept = np.flatnonzero(keep)
# boundary guard
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
kept = kept[inside]
```

iii. CONVERSION_NOTES: "Samples: corridor (0-4 m) AND VR moving. Directly from the paper ('only ... inside the 0-4-m region', 'excluded the data points in which the animal was not running') and from `Get_dprime_selective_neuron`'s `fr_valid = VRmove & isCorridor`."

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials based on two criteria: (1) trials whose wall texture has no canonical `stim_id` (i.e. `circle3` trials, 309 of 38110) are dropped because `stim_id` is NaN in every experiment-type view, and (2) trials with fewer than `MIN_SAMPLES_PER_TRIAL = 2` retained samples are dropped (though this never fires in practice since the observed minimum is 11). No trial-length-based filtering is applied.

ii.
```python
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
# ...
if T < MIN_SAMPLES_PER_TRIAL:
    continue
```

iii. CONVERSION_NOTES: "`circle3` trials excluded (309 trials, 0.81%, 4 sessions). This wall has `stim_id = NaN` in every experiment-type view, i.e. the reference assigns it no role and excludes it from every analysis."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, which contains per-plane deconvolved calcium traces concatenated across imaging planes. The visual area of each neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
# ...
region_all = load_region_index(entry['mname'], entry['datexp'], root=root)
```

iii. Same as the reference -- Suite2p deconvolved traces.

## 2-b. How is the `neural` data processed?

i. The AI applies three processing steps not in the reference: (1) per-neuron z-scoring across retained samples, (2) subsampling to at most 2000 neurons per session (stratified by visual region), and (3) dropping zero-variance neurons. The data is stored as float32.

ii.
```python
# z-scoring
Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)
Xs = Xs.astype(np.float32, copy=False)
# subsampling
sel = stratified_subsample(region_all, max_neurons, rng)
Xs = np.ascontiguousarray(X[sel])
```

iii. CONVERSION_NOTES: "Per-neuron z-scoring across the retained samples of the session. Suite2p deconvolved amplitudes span 0 - ~8000 with wildly different per-neuron scales; the reference itself z-scores neurons before any population read-out. ... Subsample to <= 2000 neurons per session ... keeping all 4,105,393 visual-cortex neurons would produce a 151.5 GB array."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three filters: (1) neurons outside visual cortex (`iarea in {-1, 7}`) are dropped, (2) zero-variance neurons are dropped, and (3) a stratified subsample of up to 2000 neurons per session is selected.

ii.
```python
region_all[sd_all == 0] = -1  # mark zero-variance neurons
sel = stratified_subsample(region_all, max_neurons, rng)
```

iii. CONVERSION_NOTES: "Additionally drop neurons with zero variance over the retained samples (they carry no information and would produce 0/0 in z-scoring)."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Aligned to corridor entry (trial start). Each trial's neural data consists of the retained samples (corridor + running) from that trial, starting at the first retained frame after corridor entry. Trials have variable length. No padding is applied.

ii.
```python
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. CONVERSION_NOTES: "Alignment event = trial start = corridor entry."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame per time bin (~314.9 ms, fs = 3.176 Hz). No rebinning is applied. The time bin size is reported as the mean frame interval across sessions.

ii.
```python
dt = float(np.median(np.diff(ft)) * SEC_PER_DAY)  # ~0.3149 s
# ...
'time_bin_size': float(np.mean(dts) * 1000.0),
```

iii. CONVERSION_NOTES: "Time bin = one imaging frame, 314.85 ms."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (frame number of the sound cue per trial) and `ft_move` (to construct the "running clock" time axis).

ii.
```python
tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)
# ...
'time_to_cue': (tau_cue[trial] - tau[kept]).astype(np.float32),
```

iii. CONVERSION_NOTES: "running-clock time of the cue minus running-clock time of the sample."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The AI constructs a "running clock" `tau` that only accumulates time during running frames (`ft_move > 0`). The sound cue frame `SoundFr` is interpolated onto this running clock, and the input is `tau(SoundFr) - tau(current_frame)` -- positive before the cue, negative after.

ii.
```python
running = ft_move > 0
tau = np.zeros(n)
if n > 1:
    np.cumsum(running[:-1] * dt, out=tau[1:])
# ...
tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)
'time_to_cue': (tau_cue[trial] - tau[kept]).astype(np.float32),
```

iii. CONVERSION_NOTES: "Time variables are measured on the 'running clock'... After the paper's non-running samples are removed, wall-clock elapsed time has a pathological tail (median trial duration 7.1 s, 99th pct 74 s, max 1765 s)."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed from the same retained sample indices (`kept`) as the neural data, so neural and time-to-cue arrays share the same frame indices and lengths.

ii.
```python
frame_idx = sb['frame_idx']  # = kept
# neural uses: X[sel][:, frame_idx] then split by trial slices
# time_to_cue uses: tau_cue[trial] - tau[kept]
```

iii. Alignment is by shared frame index.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp` (the session date string, e.g. "2022_07_19") in the experiment info records.

ii.
```python
def training_day_lookup(reg):
    first = {}
    for entry in reg.values():
        d = date(*map(int, entry['datexp'].split('_')))
        first[entry['mname']] = min(first.get(entry['mname'], d), d)
    return {key: float((date(*map(int, e['datexp'].split('_'))) - first[e['mname']]).days)
            for key, e in reg.items()}
```

iii. CONVERSION_NOTES: "Day of training = days since that mouse's first imaging session (0-92)."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The AI computes calendar days since the mouse's first imaging session. This gives values ranging from 0 to 92. The value is per-session and is broadcast (tiled) across all time bins of each trial.

ii.
```python
d = date(*map(int, entry['datexp'].split('_')))
first[entry['mname']] = min(first.get(entry['mname'], d), d)
# day = (session_date - first_session_date).days
```

iii. CONVERSION_NOTES: "`exp_info` only carries an explicit `days` field for a handful of records, so the session date is the only definition available for all 89 sessions."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr` (corridor entry frame per trial) and `ft_move` (to construct the running clock).

ii.
```python
tau_start = np.interp(start_fr_all, grid, tau)
'time_since_start': (tau[kept] - tau_start[trial]).astype(np.float32),
```

iii. CONVERSION_NOTES step 5 mapping table lists `StartFr, ft_move` as source variables.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Using the running clock `tau`, the corridor entry frame `StartFr` is interpolated onto the running clock axis, and the input is `tau(current_frame) - tau(StartFr)` in seconds. This gives non-negative, monotonically increasing values within a trial, but only counts time during running frames.

ii.
```python
tau_start = np.interp(start_fr_all, grid, tau)
'time_since_start': (tau[kept] - tau_start[trial]).astype(np.float32),
```

iii. Same running-clock justification as time_to_sound_cue.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same mechanism as time_to_sound_cue: computed from the same `kept` indices as the neural data.

ii. Same as 3-c.

iii. Alignment is by shared frame index.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` (boolean per trial) and `WallName` (wall texture name per trial).

ii.
```python
is_rew = np.asarray(beh['isRew'], dtype=bool)[:ntrials]
if is_rew.any():
    rew_wall = str(wall_name[is_rew][0])
    trial_rew = (wall_name == rew_wall).astype(np.float32)
else:
    trial_rew = np.zeros(ntrials, dtype=np.float32)
```

iii. CONVERSION_NOTES: "Reward availability from `isRew`. `utils.get_cat_id` identifies the rewarded wall as `WallName[isRew][0]`; a trial is 'in the rewarded corridor' iff its wall is that wall."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI identifies the rewarded wall name as `WallName[isRew][0]`, then sets reward_available=1 for all trials whose wall matches that rewarded wall, and 0 otherwise. For sessions with no rewards (unsupervised/naive), all trials get 0. The value is per-trial and tiled across time bins.

ii. Same code as 6-a.

iii. CONVERSION_NOTES: "Unsupervised/naive sessions have `isRew` all-False -> all zeros, which is correct."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (wall texture name per trial), `UniqWalls` (unique wall names in the session), and the merged `stim_id` (canonical stimulus role code from the experiment info).

ii.
```python
wall2id = {w: s for w, s in zip(entry['uniq_walls'], entry['stim_id'])}
wall_name = np.asarray(beh['WallName'])[:ntrials]
trial_stim = np.array([wall2id[w] for w in wall_name], dtype=float)
```

iii. CONVERSION_NOTES: "Stimulus output = the paper's canonical `stim_id` (7 classes)."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI uses the paper's canonical `stim_id` role codes, producing 7 stimulus classes: circle1 (0), circle2 (1), leaf1 (2), leaf2 (3), leaf3 (4), leaf1_swap1 (5), leaf1_swap2 (6). These are role-based, not texture-name-based, meaning they are aligned across mice that saw different physical textures. Trials with NaN `stim_id` (circle3) are excluded. The value is per-trial and broadcast across time bins.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
OUTPUT_VALUES = [STIM_NAMES, ...]
# ...
out[0] = int(sb['trial_stim'][t])
```

iii. CONVERSION_NOTES: "This is the reference's own stimulus variable and is role-aligned across mice ... Using raw `WallName` would give mutually incompatible labels across mice."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (frame indices of each lick in the session).

ii.
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
lf = lf[(lf >= 0) & (lf < n)]
lick_bin[lf] = True
```

iii. CONVERSION_NOTES: "Licking binned with `floor(LickFr)`."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary per-frame variable: 1 if at least one lick falls in that frame (using `floor(LickFr)` to assign fractional lick frame indices to integer frames), 0 otherwise. Lick frames outside the imaging range are excluded.

ii.
```python
lick_bin = np.zeros(n, dtype=bool)
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
lf = lf[(lf >= 0) & (lf < n)]
lick_bin[lf] = True
```

iii. CONVERSION_NOTES: "frame k covers [t_k, t_{k+1}), so floor() is the correct bin for a fractional lick frame index."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so the lick binary is on the same frame grid as the neural data. It is indexed by the same `kept` indices used for neural data.

ii.
```python
'lick': lick_bin[kept].astype(np.int64),
```

iii. Alignment by shared frame index.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos` (position in decimeters per imaging frame).

ii.
```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:n]
# ...
pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                  0, N_POSITION_BINS - 1)
```

iii. Same variable as the reference.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 (to get 1 m bins), giving 4 equal-length bins covering 0-4 m. Values are clipped to [0, 3].

ii.
```python
TEXTURE_LENGTH_DM = 40.0
N_POSITION_BINS = 4
pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                  0, N_POSITION_BINS - 1)
```

iii. CONVERSION_NOTES: "position_bin = `min(int(ft_Pos/10), 3)` -> 4 x 1 m bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. The continuous position (0-40 dm) is divided by 10 dm and floored to integer, then clipped to [0, 3], giving 4 bins: 0-1m, 1-2m, 2-3m, 3-4m. This is the same as the reference.

ii. Same as 9-b.

iii. Position bins are ~25% each by construction since VR speed is approximately constant.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is per imaging frame, indexed by the same `kept` frame indices as the neural data.

ii.
```python
'position': ft_pos[kept],
```

iii. Alignment by shared frame index.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed` (running speed per imaging frame).

ii.
```python
ft_spd = np.asarray(beh['ft_RunSpeed'], dtype=float)[:n]
# ...
'speed': ft_spd[kept],
```

iii. Same variable as the reference.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speed is discretized into 4 bins using global quartile edges computed over all retained samples across all 89 sessions (not per session). The global edges are computed once in the behavior-only pass so sample and full runs share identical boundaries.

ii.
```python
all_speed = np.concatenate([sb['speed'] for sb in sess_beh.values()])
speed_edges = np.percentile(all_speed, [25, 50, 75])
# ...
spd_bin = np.digitize(sb['speed'], speed_edges).astype(np.int64)
```

iii. CONVERSION_NOTES: "Speed quartiles computed globally (over every retained sample of the whole dataset), not per session, so that class q2 means the same thing in every session for the shared read-out."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` with three global percentile edges (25th, 50th, 75th percentiles of running speed across all retained samples). This produces 4 bins where each contains approximately 25% of the global data, but within individual sessions the distribution can be unequal.

ii. Same as 10-b.

iii. CONVERSION_NOTES confirms the speed bin fractions are approximately 25% each globally.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is per imaging frame, indexed by the same `kept` frame indices as the neural data.

ii.
```python
'speed': ft_spd[kept],
```

iii. Alignment by shared frame index.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) behavior arrays are truncated to the number of neural imaging frames (`[:nframes_neural]`), matching the reference code's `[:nfr]` truncation. (2) Lick frames outside the imaging range are excluded. (3) A boundary guard removes 21 mislabelled edge samples where `ft_trInd` carries the previous trial's index near corridor transitions. (4) Trials with NaN `stim_id` are excluded. (5) Zero-variance neurons are excluded.

ii.
```python
n = len(ft_full) if nframes_neural is None else min(len(ft_full), int(nframes_neural))
# ...
lf = lf[(lf >= 0) & (lf < n)]
# boundary guard
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
```

iii. CONVERSION_NOTES: "The reference code always truncates the behaviour arrays with `[:nfr]`."

## 12-a. What are the most time-consuming steps of the code?

i. Loading the spike files (405 GB total). The AI reports stage 2 (neural loading) takes ~120s with 8 parallel workers, compared to 1.3s for behavior loading.

ii.
```python
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
# ...
chunks.append(p[:, frame_idx])
```

iii. CONVERSION_NOTES: "stage 2 (neural), 8 workers: ~120 s"

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial input/output array construction in `finalize_trials` loops over trials to build individual arrays. The wall-to-stimulus-id mapping iterates over wall names. However, the dominant cost is I/O so these are minor.

ii.
```python
for t, a, b in zip(uniq, starts, stops):
    T = b - a
    inp = np.empty((4, T), dtype=np.float32)
    # ...
```

iii. The code already optimizes the expensive parts (column-slicing planes before concatenation, parallel workers).

## 12-c. What processing does the code repeat multiple times?

i. The behavior is processed twice per session: once in `build_session_behavior` during stage 1 (without knowing the neural frame count), and again in `process_session` during stage 2 (with the true frame count from the spike file). This double computation is intentional -- the stage 1 pass computes the global speed quartile edges.

ii.
```python
# stage 1: behavior-only pass
sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                       for key, entry in reg.items())
# stage 2: re-derive with neural frame count
sb = build_session_behavior(key, entry, day, nframes_neural=nframes)
```

iii. CONVERSION_NOTES notes that the global speed quartiles need all sessions' behavior data before processing neural data.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The behavior is processed twice as noted above. Additionally, the `build_session_behavior` function computes a full running clock `tau` and all behavioral masks even during the first pass when the neural frame count is unknown -- these are recomputed in the second pass. The `stratified_subsample` function performs random selection that discards information from the majority of neurons.
