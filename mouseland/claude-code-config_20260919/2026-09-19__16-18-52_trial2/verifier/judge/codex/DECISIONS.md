# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads `beh/Imaging_Exp_info.npy`, then eagerly reads every `Beh_<exp_type>.npy` file and merges repeated experiment-type views into one registry keyed by `(mname, datexp, blk)`. Later, each session loads its spike planes from `spk/<mname>_<datexp>_<blk>_neural_data.npy` and its retinotopy labels from `retinotopy/<mname>_<datexp>_trans.npz`.

ii. ```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
...
beh_all = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=True).item()
...
path = os.path.join(root, 'spk', '%s_%s_%s_neural_data.npy'
                    % (entry['mname'], entry['datexp'], entry['blk']))
d = np.load(path, allow_pickle=True).item()
...
d = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)),
            allow_pickle=True)
```

iii. In `CONVERSION_NOTES.md`, the agent says the 142 behavior records are experiment-type views of 89 unique recordings, so it collapses them and merges `stim_id` across views before processing.

## 1-b. How are the data split into subjects?

i. Subjects are split by mouse name (`mname`). The final subject list is the sorted set of all mouse names, and each session’s `subject_idx` is the index of its mouse in that list.

ii. ```python
return OrderedDict(sorted(reg.items(),
                          key=lambda kv: (kv[1]['mname'], kv[1]['datexp'], kv[1]['blk'])))
...
subjects = sorted({sess_beh[k]['mname'] for k in keys})
...
data['subject_idx'].append(subjects.index(sb['mname']))
```

iii. The notes state that a session is keyed by `(mname, datexp, blk)` and that there are 19 unique mice.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. Multiple experiment-type entries for the same recording are merged into a single session record, leaving 89 sessions.

ii. ```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
...
if key not in reg:
    reg[key] = dict(
        beh={f: beh[f] for f in BEH_FIELDS if f in beh},
        uniq_walls=uw, stim_id=sid.copy(),
        mname=d['mname'], datexp=d['datexp'], blk=d['blk'],
        exptypes=[], reward_type=d.get('rewType', 'unknown'),
        cohort=d.get('exptype', 'naive'))
```

iii. The notes explicitly justify this as collapsing 142 experiment-type records into the 89 unique recordings reported by the paper.

## 1-d. How are the data split into trials?

i. Trials are not taken as full corridor traversals. Instead, the agent first keeps only samples inside the corridor and while the VR was moving, then groups those retained samples by `ft_trInd`. The resulting trial arrays contain only the retained samples for each trial, and trials with fewer than 2 retained samples are skipped.

ii. ```python
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
kept = np.flatnonzero(keep)
...
trial = tr_of_frame[kept]
...
uniq, starts = np.unique(trial, return_index=True)
stops = np.append(starts[1:], len(trial))
...
for t, a, b in zip(uniq, starts, stops):
    T = b - a
    if T < MIN_SAMPLES_PER_TRIAL:
        continue
```

iii. The notes say the agent intentionally defined a trial as a corridor traversal restricted to `ft_CorrSpc & (ft_move > 0)`, because it wanted corridor-only, running-only samples for the decoder.

## 1-e. How are trials filtered based on quality controls?

i. The agent filters out samples outside the corridor, non-running samples, samples with invalid trial indices, samples from trials with missing canonical `stim_id` (`circle3`), and 21 boundary-mislabeled samples using a `StartFr`/`GrayFr` guard. Entire trials are skipped if they end up with fewer than 2 retained samples. It does not implement the reference’s 99th-percentile long-trial filter.

ii. ```python
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
kept = np.flatnonzero(keep)
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
...
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
kept = kept[inside]
...
if T < MIN_SAMPLES_PER_TRIAL:
    continue
```

iii. The notes justify these filters by appealing to the paper’s running-only corridor analysis, excluding `circle3` because it has `stim_id = NaN` everywhere, and removing a small number of mislabelled boundary frames.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from the deconvolved spike-like traces in `spks` from each session’s spike file, plus `iarea` from the matching retinotopy file to assign brain regions and decide which neurons to keep.

ii. ```python
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
...
iarea = np.asarray(d['iarea'])
```

iii. The notes describe `spks` as the Suite2p deconvolved traces and `iarea` as the retinotopy-based visual-area label per neuron.

## 2-b. How is the `neural` data processed?

i. The agent column-slices each imaging plane to the retained frame indices, concatenates planes, removes zero-variance neurons, stratified-subsamples to at most 2000 neurons per session, z-scores each retained neuron over the retained samples, stores the data as `float32`, and finally cuts it into per-trial matrices.

ii. ```python
while planes:
    p = planes.pop(0)
    chunks.append(p[:, frame_idx])
...
X = np.concatenate(chunks, axis=0)
...
sel = stratified_subsample(region_all, max_neurons, rng)
...
if zscore:
    Xs = (Xs - Xs.mean(axis=1, keepdims=True)) / Xs.std(axis=1, keepdims=True)
Xs = Xs.astype(np.float32, copy=False)
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The notes justify z-scoring as matching how the reference code standardizes neurons before population analyses, and justify the 2000-neuron cap as a practical memory/runtime concession.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are kept only if their retinotopy code maps into one of four visual regions, then any zero-variance neurons are dropped, and the remaining neurons are stratified-subsampled by region to at most 2000 per session.

ii. ```python
region = np.full(len(iarea), -1, dtype=np.int64)
for code, ridx in IAREA_TO_REGION.items():
    region[iarea == code] = ridx
...
sd_all = X.std(axis=1)
region_all[sd_all == 0] = -1
...
sel = stratified_subsample(region_all, max_neurons, rng)
```

iii. The notes say the visual-cortex filter comes from the reference, while the zero-variance exclusion and subsampling were added for numerical stability and tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The intended alignment event is corridor entry (`StartFr`). However, the stored neural trials are built only from retained running/corridor samples, so they begin near corridor entry but are indexed on a compressed retained-sample axis rather than on all contiguous imaging frames.

ii. ```python
start_fr_all = np.asarray(beh['StartFr'], dtype=float)[:ntrials]
...
keep = ft_corr & running & (tr_of_frame >= 0) & (tr_of_frame < ntrials)
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The notes state that the alignment event is corridor entry and that only corridor-plus-running samples are kept, with no padding.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one time bin. The frame interval is estimated from `ft` as the median inter-frame spacing, about 0.3149 s, and no temporal rebinning is applied.

ii. ```python
dt = float(np.median(np.diff(ft)) * SEC_PER_DAY)       # ~0.3149 s
...
time_bin_size=float(np.mean(dts) * 1000.0),
```

iii. The notes repeatedly say the agent kept the native imaging-frame resolution and did not interpolate onto a new temporal grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from `SoundFr`, plus the session frame times `ft`, with `ft_move` used to convert wall-clock time into the agent’s running-clock time base.

ii. ```python
ft = ft_full[:n]
...
running = ft_move > 0
...
tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)
```

iii. The notes say the cue time is measured on the running clock rather than directly from wall-clock frame times.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The agent computes a running-clock `tau` by accumulating `dt` only over frames where `ft_move > 0`, interpolates each trial’s `SoundFr` onto that clock, and stores `tau_cue[trial] - tau[kept]` at each retained sample.

ii. ```python
tau = np.zeros(n)
if n > 1:
    np.cumsum(running[:-1] * dt, out=tau[1:])
...
tau_cue = np.interp(np.asarray(beh['SoundFr'], dtype=float)[:ntrials], grid, tau)
...
time_to_cue=(tau_cue[trial] - tau[kept]).astype(np.float32)
```

iii. The notes justify this as removing the very long wall-clock pauses caused by stopped animals and keeping the time variable on a scale the decoder can use.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is aligned by indexing it with the same retained-sample slices that are used to cut the neural activity into trials.

ii. ```python
inp[0] = sb['time_to_cue'][a:b]
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The agent’s code treats the retained sample axis as the common alignment axis for inputs, outputs, and neural data.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `mname` and `datexp` in the session registry.

ii. ```python
for entry in reg.values():
    d = date(*map(int, entry['datexp'].split('_')))
    first[entry['mname']] = min(first.get(entry['mname'], d), d)
```

iii. The notes say the explicit `days` field is missing for many records, so the agent used session dates instead.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the agent finds the earliest recording date and computes each session’s day as calendar days since that first imaging session. That scalar is then broadcast across all bins of each trial.

ii. ```python
return {key: float((date(*map(int, e['datexp'].split('_'))) - first[e['mname']]).days)
        for key, e in reg.items()}
...
inp[1] = sb['day']
```

iii. The notes explicitly say this is “days since that mouse’s first imaging session” and report a range of 0–92 days.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It is derived from `StartFr`, `ft`, and `ft_move`, via the same running-clock construction used for time-to-cue.

ii. ```python
running = ft_move > 0
...
tau_start = np.interp(start_fr_all, grid, tau)
```

iii. The notes say this variable is also measured on the running clock.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. The agent interpolates `StartFr` onto the running clock and stores `tau[kept] - tau_start[trial]` at each retained sample.

ii. ```python
tau_start = np.interp(start_fr_all, grid, tau)
...
time_since_start=(tau[kept] - tau_start[trial]).astype(np.float32)
```

iii. The notes justify this with the same argument used for time-to-cue: pauses are excluded, so elapsed time is accumulated only across running frames.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned on the same retained-sample slices as the neural data.

ii. ```python
inp[2] = sb['time_since_start'][a:b]
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The agent uses one retained-sample axis for the per-trial neural, input, and output arrays.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from `isRew` and `WallName`.

ii. ```python
is_rew = np.asarray(beh['isRew'], dtype=bool)[:ntrials]
...
rew_wall = str(wall_name[is_rew][0])
trial_rew = (wall_name == rew_wall).astype(np.float32)
```

iii. The notes say the rewarded wall is identified as `WallName[isRew][0]`, and then every trial in that corridor is labeled reward-available.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. If a session has any rewarded trials, the agent identifies the rewarded wall from the first rewarded trial and marks every trial with that wall as `1`; otherwise it assigns all zeros. The result is then broadcast across each trial’s bins.

ii. ```python
if is_rew.any():
    rew_wall = str(wall_name[is_rew][0])
    trial_rew = (wall_name == rew_wall).astype(np.float32)
else:
    rew_wall = None
    trial_rew = np.zeros(ntrials, dtype=np.float32)
...
inp[3] = sb['trial_rew'][t]
```

iii. The notes justify this as implementing the instruction “1 if in rewarded corridor, 0 if not,” including all non-task sessions as all-zero.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from `WallName`, `UniqWalls`, and merged `stim_id` values from the behavior files.

ii. ```python
sid = np.atleast_1d(np.array(beh['stim_id'], dtype=float))
...
wall2id = {w: s for w, s in zip(entry['uniq_walls'], entry['stim_id'])}
wall_name = np.asarray(beh['WallName'])[:ntrials]
trial_stim = np.array([wall2id[w] for w in wall_name], dtype=float)
```

iii. The notes say the agent wanted the paper’s canonical `stim_id` role code so labels would be comparable across mice and experiment views.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The agent merges `stim_id` across experiment-type views, maps each trial’s `WallName` to that canonical 7-class `stim_id`, drops trials where that id is `NaN` (notably `circle3`), and broadcasts the resulting class index across all retained bins of the trial.

ii. ```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
trial_stim = np.array([wall2id[w] for w in wall_name], dtype=float)
...
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
...
out[0] = int(sb['trial_stim'][t])
```

iii. The notes explicitly justify using 7 canonical stimulus-role classes instead of the broader texture categories, and explain that `circle3` is excluded because the reference never assigns it a role.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from `LickFr`.

ii. ```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
```

iii. The notes state that lick timing is available directly as neural-frame indices.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frame indices are floored to integer frame bins, out-of-range licks are discarded, and a binary lick vector is created over frames before being restricted to the retained samples.

ii. ```python
lick_bin = np.zeros(n, dtype=bool)
lf = np.floor(np.asarray(beh['LickFr'], dtype=float)).astype(np.int64)
lf = lf[(lf >= 0) & (lf < n)]
lick_bin[lf] = True
```

iii. The notes justify `floor(LickFr)` by saying frame `k` covers `[t_k, t_{k+1})`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is aligned by first indexing the lick vector on the retained sample set and then cutting trials with the same `(a, b)` slices used for the neural data.

ii. ```python
lick=lick_bin[kept].astype(np.int64),
...
out[1] = sb['lick'][a:b]
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The code uses one shared retained-sample axis for licking and neural data.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It is derived from `ft_Pos`.

ii. ```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:n]
...
position=ft_pos[kept],
```

iii. The notes describe `ft_Pos` as the position signal sampled once per imaging frame, in decimetres.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The agent keeps the retained `ft_Pos` values and discretizes them into 4 equal 1 m bins during trial finalization.

ii. ```python
pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                  0, N_POSITION_BINS - 1)
```

iii. The notes say this was chosen to match the decoder requirement of four equal-length bins across the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Position is divided by 10 decimetres (1 m), converted to integers, and clipped to category indices 0–3 corresponding to `0-1m`, `1-2m`, `2-3m`, `3-4m`.

ii. ```python
OUTPUT_VALUES = [STIM_NAMES,
                 ['no_lick', 'lick'],
                 ['0-1m', '1-2m', '2-3m', '3-4m'],
                 ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']]
...
pos_bin = np.clip((sb['position'] / (TEXTURE_LENGTH_DM / N_POSITION_BINS)).astype(np.int64),
                  0, N_POSITION_BINS - 1)
```

iii. The notes say the corridor is 40 dm long and the output must be 4 equal-length, 1 m bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is first restricted to retained samples and then cut per trial with the same slices as the neural data.

ii. ```python
position=ft_pos[kept],
...
out[2] = pos_bin[a:b]
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The code uses the same retained-sample axis for position and neural activity.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from `ft_RunSpeed`.

ii. ```python
ft_spd = np.asarray(beh['ft_RunSpeed'], dtype=float)[:n]
...
speed=ft_spd[kept],
```

iii. The notes identify `ft_RunSpeed` as the per-frame running-speed signal.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent first gathers retained speeds from every session in a behavior-only prepass, computes global 25th/50th/75th percentile edges across the whole dataset, and then discretizes each retained sample by applying those edges with `np.digitize`.

ii. ```python
all_speed = np.concatenate([sb['speed'] for sb in sess_beh.values()])
speed_edges = np.percentile(all_speed, [25, 50, 75])
...
spd_bin = np.digitize(sb['speed'], speed_edges).astype(np.int64)
```

iii. The notes say this was done so that a given speed-bin label would mean the same thing across all sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The thresholds are the global dataset-wide quartile edges of the retained running-speed distribution, and each retained sample is assigned to one of four bins by `np.digitize`.

ii. ```python
speed_edges = np.percentile(all_speed, [25, 50, 75])
...
spd_bin = np.digitize(sb['speed'], speed_edges).astype(np.int64)
```

iii. The notes explicitly contrast this with a per-session definition and defend using one global set of edges.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is restricted to retained samples and then split into trials with the same slices as the neural data.

ii. ```python
speed=ft_spd[kept],
...
out[3] = spd_bin[a:b]
...
trials = [np.ascontiguousarray(Xs[:, a:b]) for (a, b) in slices]
```

iii. The code uses the common retained-sample axis for speed and neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent truncates behavior to the imaged frame count, converts `NaN` trial indices to `-1` and drops them, removes out-of-range licks, filters out samples from trials lacking a canonical `stim_id`, applies a boundary guard around `StartFr` and `GrayFr`, and excludes zero-variance neurons. Trials with fewer than 2 retained samples are skipped.

ii. ```python
n = len(ft_full) if nframes_neural is None else min(len(ft_full), int(nframes_neural))
...
tr_of_frame = np.where(np.isfinite(ft_tr), ft_tr, -1).astype(np.int64)
...
lf = lf[(lf >= 0) & (lf < n)]
...
kept = kept[np.isfinite(trial_stim[tr_of_frame[kept]])]
...
inside = (kept >= start_fr_all[tr_kept] - 1.0) & (kept <= gray_fr_all[tr_kept] + 1.0)
...
region_all[sd_all == 0] = -1
```

iii. The notes justify these as robustness fixes for known quirks: behavior sometimes outlasts imaging, a few boundary frames are mislabeled, `circle3` never receives a canonical `stim_id`, and some neurons have zero variance after sample filtering.

## 12-a. What are the most time-consuming steps of the code?

i. The expensive part is stage 2, where each session’s large spike file is opened, column-sliced, concatenated, and processed. The code explicitly treats behavior loading as cheap and spike loading as the expensive phase.

ii. ```python
# stage 2 -- neural data (expensive; loads /app/data/spk)
...
d = np.load(path, allow_pickle=True).item()
planes = d['spks']
...
X = np.concatenate(chunks, axis=0)
```

iii. The notes say the neural stage dominates runtime because the spike files total hundreds of GB, while stage 1 behavior loading is only about 1.3 s.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious non-vectorized loops are the per-trial assembly loop in `finalize_trials`, the per-region quota loop in `stratified_subsample`, and the registry merge loop over experiment types and records. The agent’s notes imply it viewed many of the larger costs as I/O-bound rather than CPU-loop-bound.

ii. ```python
for t, a, b in zip(uniq, starts, stops):
    ...
    inputs.append(inp)
    outputs.append(out)
...
for r in range(len(BRAIN_REGIONS)):
    p = pool[region[pool] == r]
    picked.append(p if quota[r] >= len(p)
                  else rng.choice(p, size=int(quota[r]), replace=False))
...
for exp_type, db in exp_info.items():
    ...
    for d in db:
```

iii. In Step 6 of the notes, the agent focuses its efficiency discussion on removing redundant reads and big temporary arrays, which suggests it considered the remaining Python loops less important than I/O and memory movement.

## 12-c. What processing does the code repeat multiple times?

i. The behavior-derived session state is computed twice: once in the stage-1 prepass for every session so the code can compute global speed edges, and again inside `process_session` after the true neural frame count is known. In `--show-processing` mode, the first sessions are processed yet again for plotting.

ii. ```python
sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                       for key, entry in reg.items())
...
sb = build_session_behavior(key, entry, day, nframes_neural=nframes)
...
res = process_session(reg[key], key, day_lut[key], speed_edges,
                      args.max_neurons, return_raw=True,
                      zscore=not args.no_zscore)
```

iii. The code and notes both show a cheap behavior-only pass followed by a full per-session pass, with optional extra diagnostic processing for plotting.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does extra diagnostic work that is not needed for the final converted dataset: an initial full-session behavior prepass, optional raw-copy retention and plotting for `--show-processing`, and metadata/summary calculations used only for validation logs. The stage-1 `sess_beh` objects also exist mainly to derive global speed edges and sample selection statistics before being superseded by per-session recomputation.

ii. ```python
sess_beh = OrderedDict((key, build_session_behavior(key, entry, day_lut[key]))
                       for key, entry in reg.items())
...
raw = Xs.copy() if return_raw else None
...
if args.show_processing:
    ...
    show_processing(key, reg[key], res, speed_edges)
...
cat_out = np.concatenate([o for s in data['output'] for o in s], axis=1)
cat_in = np.concatenate([o for s in data['input'] for o in s], axis=1)
```

iii. The notes emphasize validation, diagnostics, and processing plots, so some work is intentionally done for checking and demonstration rather than for the downstream decoder input itself.
