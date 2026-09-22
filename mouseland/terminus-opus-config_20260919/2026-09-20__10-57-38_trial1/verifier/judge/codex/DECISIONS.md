# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Loads the master `Imaging_Exp_info.npy`, deduplicates recordings by `(mname,datexp,blk)`, scans all `Beh_*.npy` files once, merges stimulus maps across duplicate experiment-type entries, and loads each session's retinotopy and selected spike rows. Full mode processes every behavior-backed session.

ii. ```python
exp_info = np.load(os.path.join(BEH_DIR, 'Imaging_Exp_info.npy'), allow_pickle=True).item()
...
for f in sorted(glob.glob(os.path.join(BEH_DIR, 'Beh_*.npy'))):
    B = np.load(f, allow_pickle=True).item()
...
ret = np.load(os.path.join(RET_DIR, '%s_%s_trans.npz' % (info['mname'], info['datexp'])))
spk, n_spk_neurons = load_spk_rows(key, rows)
```

iii. The notes say this covers 89 unique sessions, reads 23 behavior files in one pass, merges complementary swap labels, and avoids materializing the full 405-GB concatenated spike dataset.

## 1-b. How are the data split into subjects?

i. Uses `mname` as the subject identifier. The final sorted unique mouse names form `subjects`; each session gets its mouse's index.

ii. ```python
subjects = sorted(set(r['mname'] for r in results))
'subject_idx': np.array([subjects.index(r['mname']) for r in results], dtype=np.int64)
```

iii. The master index directly supplies mouse identity; the notes report 19 mice.

## 1-c. How are the data split into sessions?

i. Defines a session as unique mouse/date/block, deduplicating repeated experiment-type listings. Behavior keys with suffixes are reduced to their five-part base session key.

ii. ```python
key = session_key(db['mname'], db['datexp'], db['blk'])
rec = sessions.setdefault(key, {...})
...
base = '_'.join(k.split('_')[:5])
```

iii. The notes explain that `_swap1`/`_swap2` behavior entries are duplicate views of one recording and are merged, yielding 89 sessions.

## 1-d. How are the data split into trials?

i. Builds a global valid-frame mask requiring corridor, movement, and finite trial/position/speed values; groups those frame indices by integer `ft_trInd`. Thus a stored trial contains only its valid running frames and can have gaps relative to the raw traversal.

ii. ```python
valid = (ft_corr & ft_move & np.isfinite(ft_trind) & np.isfinite(ft_pos)
         & np.isfinite(ft_speed))
...
tr_valid = ft_trind[idx_valid].astype(int)
...
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
```

iii. The agent cites the paper's “only considered timepoints during running” statement and reference analysis mask `fr_valid = VRmove & isCorridor`.

## 1-e. How are trials filtered based on quality controls?

i. Drops trials with fewer than five valid frames, missing canonical stimulus labels, non-finite sound/start times, or invalid trial indices; later drops sessions with fewer than two surviving trials.

ii. ```python
if len(fr) < MIN_FRAMES_PER_TRIAL: ... continue
if wname not in smap: ... continue
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]): ... continue
...
results = [r for r in results if len(r['trials']) >= 2]
```

iii. The notes call these usability checks and report 309 unlabeled `circle3` trials dropped; the minimum length was intended to remove trials with too little usable running data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Uses per-plane `spks` arrays from each session's neural file; `iarea` from retinotopy determines visual region and selected rows.

ii. ```python
planes = np.load(path, allow_pickle=True).item()['spks']
...
iarea = ret['iarea']
rows, area_idx = select_neurons(iarea, N_NEURONS_PER_SESSION, rng)
```

iii. The notes identify these as Suite2p deconvolved traces and the reference retinotopy mapping.

## 2-b. How is the `neural` data processed?

i. Selects at most 1,000 neurons, computes each selected neuron's mean/SD over valid frames, removes zero/non-finite-SD neurons, z-scores its whole trace, slices trial frames, and stores float32.

ii. ```python
mu = spk[:, valid].mean(axis=1, keepdims=True)
sd = spk[:, valid].std(axis=1, keepdims=True)
keep_neu = (sd[:, 0] > 0) & np.isfinite(sd[:, 0])
spk = (spk - mu) / np.where(sd > 0, sd, 1.0)
...
'neural': np.ascontiguousarray(spk[:, fr])
```

iii. The agent says z-scoring mirrors a reference population-analysis helper and helps PCA, while 1,000 neurons makes conversion/training tractable.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Drops neurons outside V1/mHV/lHV/aHV, proportionally random-subsamples to 1,000 per session, then removes selected neurons with zero or non-finite SD.

ii. ```python
valid = np.where(area_idx >= 0)[0]
...
chosen.append(rng.choice(pool, size=min(alloc[a], len(pool)), replace=False))
...
spk = spk[keep_neu]
```

iii. The notes justify visual-cortex filtering from reference `iarea` handling and subsampling from the 405-GB source size. The RNG is described as seeded and area-stratified.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are identified relative to corridor entry, but stored neural columns are only the running/corridor frames assigned to that trial; no padding or fixed-duration window is used.

ii. ```python
fr = np.sort(fr)
...
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
'neural': np.ascontiguousarray(spk[:, fr]),
```

iii. The agent calls corridor entry (`Trial_start_time`/`StartFr`) the alignment event and retains variable trial lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Uses native imaging frames with no rebinning. Metadata uses the mean session median frame interval, about 314.85 ms.

ii. ```python
'dt': float(np.median(np.diff(ft)) * SEC_PER_DAY)
...
dt_ms = float(np.mean([r['dt'] for r in results]) * 1000.0)
```

iii. The notes state 3.17 Hz (~315 ms) and say behavioral streams are already frame-aligned.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. Derived from per-trial `SoundTime` and per-frame timestamps `ft`.

ii. ```python
t_sound = np.asarray(beh['SoundTime'], dtype=float)
...
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY
```

iii. The agent treats `SoundTime` as the timestamp equivalent of cue-frame alignment.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Subtracts each selected frame timestamp from trial sound time and converts MATLAB-day units to seconds, positive before and negative after the cue. It also creates an extra binary cue-onset channel.

ii. ```python
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY
...
after = np.where(t['time_to_cue'] <= 0)[0]
if after.size: cue_onset[after[0]] = 1.0
```

iii. The continuous signed difference follows the requested meaning; the extra onset series was added because the format guidance says time onsets may be binary.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. Computes it at exactly the same `fr` indices used to slice neural activity.

ii. ```python
'neural': np.ascontiguousarray(spk[:, fr]),
'time_to_cue': (t_sound[tr] - ft[fr]) * SEC_PER_DAY,
```

iii. The agent documents all time-varying streams as sharing the imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. Derived from each session's `datexp` and the earliest retained session date for that mouse.

ii. ```python
d = parse_date(r['datexp'])
...
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
```

iii. The notes interpret day of training as elapsed days since the mouse's first imaging session.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Parses calendar dates, subtracts the mouse's earliest date in the processed results, and broadcasts the elapsed integer days over every trial frame.

ii. ```python
day = (parse_date(r['datexp']) - first_day[r['mname']]).days
...
np.full(T, float(day), dtype=np.float32)
```

iii. The agent explicitly chose elapsed calendar days rather than count of recorded sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. Derived from frame timestamps `ft` and per-trial `Trial_start_time`.

ii. ```python
t_start = np.asarray(beh['Trial_start_time'], dtype=float)
...
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY
```

iii. The notes identify `Trial_start_time` as corridor entry, equivalent to `StartFr` on the time axis.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Subtracts trial-start time from each selected frame timestamp and converts days to seconds.

ii. ```python
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY
```

iii. The agent intended a continuous signed elapsed-time variable starting around zero.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Uses the same selected frame indices `fr` as neural data.

ii. ```python
'neural': np.ascontiguousarray(spk[:, fr]),
'time_since_start': (ft[fr] - t_start[tr]) * SEC_PER_DAY,
```

iii. The notes describe common frame-grid alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. Uses the per-trial raw `isRew` field.

ii. ```python
is_rew = np.asarray(beh['isRew'], dtype=bool)
...
'is_rew': int(is_rew[tr])
```

iii. The agent says this directly marks rewarded-corridor trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Casts the trial boolean to 0/1 and broadcasts it over all stored frames.

ii. ```python
np.full(T, float(t['is_rew']), dtype=np.float32)
```

iii. The notes say 1 means rewarded corridor and 0 otherwise, including all-zero non-task sessions.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. Uses `WallName`, `UniqWalls`, and `stim_id`; mappings from repeated behavior-file entries are merged per session.

ii. ```python
sid = np.asarray(beh['stim_id'], dtype=float)
for w, s in zip(list(beh['UniqWalls']), sid):
    if not np.isnan(s): stim_map[base][str(w)] = int(s)
...
wname = str(wall[tr])
```

iii. The agent chose the paper/reference notebook's role-based stimulus IDs.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Maps wall names to seven canonical role labels and broadcasts the integer label across the trial; unlabeled walls are dropped.

ii. ```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
...
np.full(T, t['stim'], dtype=np.int64)
```

iii. The notes say this pools physical rock/wood patterns into canonical circle/leaf roles and preserves swap identities; 309 `circle3` trials lacked a mapping and were removed.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Uses raw lick frame numbers `LickFr`.

ii. ```python
lick_fr = np.atleast_1d(np.asarray(beh['LickFr'], dtype=float))
```

iii. The notes describe `LickFr` as imaging-frame lick events.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Floors finite lick frame numbers to integers, discards out-of-range events, and marks a frame 1 if at least one lick falls there.

ii. ```python
li = np.floor(lick_fr[np.isfinite(lick_fr)]).astype(int)
li = li[(li >= 0) & (li < nfr)]
lick_bin[li] = 1
```

iii. The agent says this creates the requested binary per-frame output; sessions with no licks remain all zero.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Slices the framewise lick vector using the same trial frame indices as neural activity.

ii. ```python
'neural': np.ascontiguousarray(spk[:, fr]),
'lick': lick_bin[fr],
```

iii. The shared imaging-frame indices provide alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Uses framewise `ft_Pos`, interpreted in decimeters.

ii. ```python
ft_pos = np.asarray(beh['ft_Pos'], dtype=float)[:nfr]
...
'pos_dm': ft_pos[fr]
```

iii. The notes infer 0–40 dm is the four-meter textured corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Slices raw position at selected frames, then digitizes it into categorical bins during assembly.

ii. ```python
'pos_dm': ft_pos[fr]
...
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. The agent follows the requested four one-meter position categories.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Uses thresholds 10, 20, and 30 dm and clips indices to 0–3, corresponding to 0–1, 1–2, 2–3, and 3–4 m.

ii. ```python
POS_BIN_EDGES_DM = np.array([10.0, 20.0, 30.0])
pos_bin = np.clip(np.digitize(t['pos_dm'], POS_BIN_EDGES_DM), 0, 3)
```

iii. The notes state these are four equal-length one-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Uses `ft_Pos[fr]` for the exact same `fr` columns stored in neural.

ii. ```python
'neural': np.ascontiguousarray(spk[:, fr]),
'pos_dm': ft_pos[fr],
```

iii. The agent relies on the common imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Uses framewise `ft_RunSpeed` at retained frames.

ii. ```python
ft_speed = np.asarray(beh['ft_RunSpeed'], dtype=float)[:nfr]
...
'speed': ft_speed[fr]
```

iii. The raw stream is already frame-aligned and expressed as speed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Concatenates all retained speeds from all retained sessions/trials, computes global 25th/50th/75th quantiles, and digitizes every trial with those common thresholds.

ii. ```python
all_speed = np.concatenate([np.concatenate([t['speed'] for t in r['trials']]) for r in results])
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
...
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The agent chose global edges so each bin contains approximately 25% of converted timepoints and thresholds are consistent across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Applies global value thresholds at the three dataset-wide empirical quartiles via `np.digitize`; tied values are not rank-split.

ii. ```python
speed_edges = np.quantile(all_speed, [0.25, 0.5, 0.75])
spd_bin = np.clip(np.digitize(t['speed'], speed_edges), 0, 3)
```

iii. The notes report exactly 25% per bin on the converted running-only data and store the edges in metadata.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Extracts speed at the same `fr` indices used for neural columns, then thresholds without changing length/order.

ii. ```python
'neural': np.ascontiguousarray(spk[:, fr]),
'speed': ft_speed[fr],
```

iii. The common frame indices ensure within-dataset alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Truncates neural and behavior to their common frame count; requires finite trial, position, and speed values; ignores invalid/out-of-range lick frames; drops trials with missing event times or labels and short trials; asserts spike/retinotopy neuron counts match.

ii. ```python
nfr = min(spk.shape[1], len(beh['ft']))
valid = (... & np.isfinite(ft_trind) & np.isfinite(ft_pos) & np.isfinite(ft_speed))
...
li = li[(li >= 0) & (li < nfr)]
...
if not np.isfinite(t_sound[tr]) or not np.isfinite(t_start[tr]): continue
```

iii. The notes say truncation matches reference behavior and the additional exclusions prevent malformed decoder examples.

## 12-a. What are the most time-consuming steps of the code?

i. Identifies loading multi-gigabyte spike files and session processing as the bottleneck; parallelizes sessions and copies only selected rows.

ii. ```python
planes = np.load(path, allow_pickle=True).item()['spks']
...
with ctx.Pool(args.nworkers) as pool:
    results = pool.map(process_session, keys, chunksize=1)
```

iii. The notes call the 405-GB spike source I/O-bound and report 2–9 seconds/session with multiprocessing.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent explicitly vectorized trial segmentation by sorting valid trial indices and splitting at boundaries. Remaining small loops include plane row extraction, area allocation, and per-trial assembly.

ii. ```python
order = np.argsort(tr_valid, kind='stable')
bounds = np.where(np.diff(tr_sorted) != 0)[0] + 1
for fr, tt in zip(np.split(idx_sorted, bounds), np.split(tr_sorted, bounds)):
```

iii. The notes claim this replaces a loop that scans all frames separately for every trial; remaining loops are small or operate on ragged outputs.

## 12-c. What processing does the code repeat multiple times?

i. Per-trial assembly repeatedly allocates constant arrays and casts/copies neural arrays; optional diagnostics concatenate the ragged dataset again. No major repeated scientific transform is identified in the notes.

ii. ```python
np.full(T, float(day), dtype=np.float32)
np.full(T, float(t['is_rew']), dtype=np.float32)
...
neural_s.append(t['neural'].astype(np.float32))
```

iii. The agent's efficiency discussion does not flag repeated processing beyond necessary ragged per-trial construction.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion creates an extra `sound_cue_onset` input not requested by the decoder task and computes extensive metadata/sanity summaries; plotting is optional. Most of this is retained or used for validation rather than literally discarded.

ii. ```python
cue_onset = np.zeros(T, dtype=np.float32)
...
'input_names': ['time_to_sound_cue', 'sound_cue_onset', ...]
...
durations = np.concatenate([...])
```

iii. The extra onset channel was justified from generic format guidance, while diagnostics and plots were included to validate conversion. The notes do not identify any discarded core processing.


