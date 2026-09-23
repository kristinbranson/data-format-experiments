# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `beh/Imaging_Exp_info.npy` as the master index, deduplicates recordings, loads each selected `Beh_<exp_type>.npy` once, then loads each session's spike and retinotopy files. The full mode processes 89 unique recordings.

ii.
```python
exp_info = load_exp_info()
sessions = unique_sessions(exp_info)
B = np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
spks = np.load(fn, allow_pickle=True).item()['spks']
ret = np.load(os.path.join(DATA_ROOT, 'retinotopy', '%s_%s_trans.npz' % (mname, datexp)))
```

iii. The agent documented that 142 index entries reduce to 89 unique recordings because sessions recur across experiment types, and it verified repeated behavior copies were identical. Grouped behavior loading and one neural read per session were chosen for efficiency.

## 1-b. How are the data split into subjects?

i. Subjects are mouse IDs (`mname`). The output contains sorted unique mouse names, while each retained session receives the corresponding index.

ii.
```python
subjects = sorted({k[0] for k, _, _, _ in sessions})
data['subject_idx'].append(subjects.index(key[0]))
```

iii. The agent treated the explicit mouse identifier in the experiment index as authoritative; the full data contain 19 mice.

## 1-c. How are the data split into sessions?

i. A session is the unique tuple `(mname, datexp, blk)`. Duplicate appearances under multiple experiment types are represented once, using the first occurrence.

ii.
```python
def session_key(db):
    return (db['mname'], db['datexp'], str(db['blk']))
if key not in reps:
    reps[key] = (exp_type, db)
```

iii. The notes say this yields the paper's 89 recordings and that duplicate behavior entries were checked for equality.

## 1-d. How are the data split into trials?

i. Per-frame `ft_trInd` supplies trial membership. For each trial, the agent keeps only frames in the 4 m corridor for which `ft_move > 0`, groups them by trial, and later truncates them to the neural recording length.

ii.
```python
valid = move & corr & np.isfinite(ftr)
vidx = np.where(valid)[0]
vtr = ftr[vidx].astype(int)
frames = vidx[lo[t]:hi[t]]
```

iii. The agent cited the paper's running-timepoint curation and the reference mask `(ft_move>0) & ft_CorrSpc`, arguing that it removes long stationary reward/stop periods.

## 1-e. How are trials filtered based on quality controls?

i. Empty trials are omitted in the behavior pass. After truncation to available neural frames, trials with fewer than five retained moving-corridor frames are dropped; sessions with fewer than two usable trials are skipped.

ii.
```python
if len(frames) == 0:
    continue
frames = tr['frames'][tr['frames'] < nfr_spk]
if len(frames) < MIN_FRAMES_PER_TRIAL:
    continue
if len(trials) < 2:
    continue
```

iii. The agent called fewer than five frames too short to be informative or indicative of end-of-recording truncation. It reported that all full-run trials survived this rule.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from each plane's deconvolved `spks` array in the session neural file; `iarea` from retinotopy determines neuron region and selection.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
iarea = ret['iarea']
rows, region, region_counts, n_labelled = select_neurons(iarea, rng)
```

iii. The agent noted that these are already Suite2p-deconvolved traces, so neither dF/F calculation nor another deconvolution is required.

## 2-b. How is the `neural` data processed?

i. The agent selects area-labelled neurons, randomly subsamples them to at most 2,000 per session in approximate regional proportions, extracts only retained moving-corridor frames, and stores contiguous float32 neuron-by-time trial arrays. It applies no normalization or temporal rebinning.

ii.
```python
pick = rng.choice(per_region[r], size=alloc[r], replace=False)
out[filled:filled + len(sel)] = plane[sel][:, frames]
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. The cap was justified as making the full conversion tractable while exceeding the decoder's effective 100-PC representation; no z-scoring was used because it was analysis-specific rather than universal reference preprocessing.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons outside V1, mHV, lHV, and aHV are dropped. Remaining neurons are capped at 2,000 per session by seeded, region-stratified random selection. No d-prime or additional cell-quality filter is applied.

ii.
```python
per_region = [np.where(masks[r])[0] for r in BRAIN_REGIONS]
if total > MAX_NEURONS:
    alloc = np.floor(counts / total * MAX_NEURONS).astype(int)
rng = np.random.default_rng(SEED + isess)
```

iii. Unlabelled areas cannot be represented in `brain_regions`; the additional cap was a size/decoder-efficiency choice, not a biological quality criterion.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are described as aligned to corridor entry (`StartFr`), but neural columns are the original frames satisfying the moving-corridor mask rather than a fixed contiguous window. Trial arrays have variable length; actual frame timestamps preserve elapsed time across omitted stationary frames.

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
valid = move & corr & np.isfinite(ftr)
neural.append(np.ascontiguousarray(act[:, cols], dtype=np.float32))
```

iii. The agent argued that `StartFr` is the requested event and variable-length trials avoid padding; it explicitly acknowledged that retained frames need not be contiguous in clock time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin, approximately 315 ms at 3.17 Hz. No temporal rebinning or resampling is performed.

ii.
```python
dt=float(np.median(np.diff(t_frame)))
time_bin_size=float(np.median(dts) * 1000.0)
```

iii. Behavior is already on the imaging-frame clock, so the agent considered native frames the highest-resolution common grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-trial `SoundFr` and the imaging frame timestamps `ft`.

ii.
```python
t_frame = beh['ft'] * 86400.0
t_cue = np.interp(beh['SoundFr'], frame_idx, t_frame)
```

iii. `SoundFr` was chosen as the actual cue timing rather than a delayed reward proxy.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Fractional cue frames are interpolated onto the timestamp axis, and each retained frame is assigned cue time minus frame time in seconds (positive before cue, negative after).

ii.
```python
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32)
```

iii. The sign follows the literal meaning “time to” the cue, and conversion from MATLAB-day units supplies seconds.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is indexed using exactly the same absolute retained frame indices as the neural trial.

ii.
```python
frames=frames,
time_to_cue=(t_cue[t] - t_frame[frames]).astype(np.float32)
```

iii. The agent states that behavior is natively sampled on the neural frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the calendar date embedded in each session key and the earliest imaging-session date for that mouse across all 89 sessions.

ii.
```python
first_date = first_session_dates(sessions)
day = (datetime.date(*map(int, key[1].split('_'))) - first_date[key[0]]).days
```

iii. The agent found the index's explicit `days` field incomplete and chose elapsed calendar days as a continuous training axis.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It subtracts the mouse's first recording date from the current date and broadcasts that integer day difference over every retained frame in the trial.

ii.
```python
day_of_training=np.float32(day_of_training)
inp[1] = tr['day_of_training']
```

iii. Computing first dates over the complete session set keeps sample and full conversions consistent.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial corridor-entry frame `StartFr` and frame timestamps `ft`.

ii.
```python
t_start = np.interp(beh['StartFr'], frame_idx, t_frame)
```

iii. `StartFr` directly represents the instructed corridor-entry alignment event.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Fractional start frames are interpolated to timestamps; start time is subtracted from every retained frame time in seconds.

ii.
```python
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32)
```

iii. Actual elapsed times were retained so gaps from removed stationary frames are represented correctly.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed at the same absolute frames used as the trial's neural columns.

ii.
```python
frames=frames,
time_since_start=(t_frame[frames] - t_start[t]).astype(np.float32)
```

iii. The shared frame index provides direct alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial `isRew` field.

ii.
```python
is_rew = np.asarray(beh['isRew']).astype(bool)
```

iii. The notes identify `isRew` as exactly the rewarded-corridor trials.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is converted to 0.0 or 1.0 and broadcast over all retained trial frames.

ii.
```python
reward_available=np.float32(1.0 if is_rew[t] else 0.0)
inp[3] = tr['reward_available']
```

iii. This is naturally a per-trial binary decoder input; non-rewarded cohorts remain zero.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from the per-trial `WallName`.

ii.
```python
wall = np.asarray(beh['WallName'])
stim_cat = np.array([STIM_CATEGORIES.index(texture_family(w)) for w in wall])
```

iii. `WallName` remains informative in swap sessions and explicitly identifies the displayed texture.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Numeric and swap suffixes are stripped, variants are pooled into four texture families, and raw `wood` is renamed `brick`. The category index is broadcast over time.

ii.
```python
base = wall_name.split('_')[0]
base = ''.join(ch for ch in base if not ch.isdigit())
if base == 'wood': base = 'brick'
out[0] = tr['stim_cat']
```

iii. The paper calls the four images circle, leaf, rock, and brick; pooling crop/swap variants makes labels comparable across mice.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It is derived from session lick frame numbers in `LickFr`.

ii.
```python
lf = np.floor(np.asarray(beh['LickFr'], dtype=float))
```

iii. The reference alignment utilities likewise bin lick events by imaging frame.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame values are floored, nonfinite and out-of-range entries are removed, and a frame is marked 1 if any lick falls within it, otherwise 0.

ii.
```python
lf = lf[np.isfinite(lf)].astype(int)
lf = lf[(lf >= 0) & (lf < nfr_beh)]
lick_bin[lf] = True
```

iii. The agent interpreted constant zeros in non-task cohorts as genuine absence of licking, not missing observations.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The framewise lick indicator is sliced using the identical retained frame indices used for neural activity.

ii.
```python
lick=lick_bin[frames].astype(np.int64)
```

iii. `LickFr` already uses neural-frame coordinates.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from framewise `ft_Pos`, measured in decimeters.

ii.
```python
pos = beh['ft_Pos'][:nfr_beh].astype(float)
```

iii. The agent verified that corridor frames cover the 0–40 dm texture area.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The retained raw positions are divided into four equal 10 dm (1 m) bins, floored, and clipped to category indices 0–3.

ii.
```python
out[2] = np.clip(np.floor(tr['pos'] / (TEXTURE_LENGTH_DM / N_POS_BINS)),
                 0, N_POS_BINS - 1).astype(np.int64)
```

iii. This implements the instructed four equal-length spatial bins over the 4 m corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 1, 2, 3, and 4 m, represented as decimeter divisions at 0, 10, 20, 30, and 40.

ii.
```python
TEXTURE_LENGTH_DM = 40.0
N_POS_BINS = 4
position_bin_edges_m=[0.0, 1.0, 2.0, 3.0, 4.0]
```

iii. Equal 1 m bins follow directly from the requested output definition.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is extracted at the same retained absolute frames as neural activity.

ii.
```python
pos=pos[frames]
```

iii. `ft_Pos` is already on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from framewise `ft_RunSpeed` for all retained moving-corridor frames.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr_beh].astype(float)
```

iii. This is the dataset's direct running-speed measure.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The agent pools speeds from all behavior-pass trials across all selected sessions, calculates the 25th, 50th, and 75th percentiles, and uses those common numerical edges to assign bins.

ii.
```python
speeds = np.concatenate([tr['speed'] for res in beh_results for tr in res['trials']])
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. Global edges were chosen to give a shared interpretation across sessions and approximately 25% of the full dataset in each bin.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Values below the global 25th percentile are category 0, then the 25–50%, 50–75%, and at/above 75% ranges become categories 1–3 via `np.digitize`.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
np.digitize(tr['speed'], speed_edges)
```

iii. The agent preferred shared value thresholds; it reported exactly quartered counts in validation (subject to ties/selection details).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Raw speed is first sliced at the same retained frame indices and its resulting values are categorized without changing length.

ii.
```python
speed=speed[frames]
out[3] = np.digitize(tr['speed'], speed_edges).astype(np.int64)
```

iii. `ft_RunSpeed` is sampled on the same imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior-selected frames beyond the neural recording are discarded; trials then shorter than five frames are removed. Nonfinite and invalid lick indices are discarded. Assertions check aligned shapes. Behavior arrays otherwise remain on their native lengths until neural truncation.

ii.
```python
frames = frames[frames < nfr]
frames = tr['frames'][tr['frames'] < nfr_spk]
lf = lf[np.isfinite(lf)]
assert data['input'][i][tr].shape == (4, T)
```

iii. The agent documented that behavior is typically 1–3 frames longer than neural data and cited reference truncation. The minimum-length rule was intended to reject uninformative/truncated trials.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and slicing the multi-gigabyte per-session spike files dominates. The agent parallelizes this neural pass over up to six processes.

ii.
```python
spks = np.load(fn, allow_pickle=True).item()['spks']
pool = mp.Pool(nproc, maxtasksperchild=1)
it = pool.imap_unordered(convert_session, jobs)
```

iii. Notes estimate 2–8 GB per neural file and identify the neural pass as the dominant I/O/memory work.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The agent already vectorizes trial grouping with stable sorting and `searchsorted`. Remaining loops over planes, regions, trials, and sessions are needed for ragged arrays/file layout, though the frame-to-column dictionary lookup per trial could be vectorized with sorted-index searches.

ii.
```python
order = np.argsort(vtr, kind='stable')
lo = np.searchsorted(vtr, np.arange(ntrials), side='left')
hi = np.searchsorted(vtr, np.arange(ntrials), side='right')
cols = np.array([frame_pos[f] for f in frames], dtype=np.int64)
```

iii. The notes explicitly claim roughly 100× improvement over scanning every frame separately for every trial.

## 12-c. What processing does the code repeat multiple times?

i. It avoids most expensive repetition: behavior files are loaded once per experiment type and each spike file once per session. It does traverse trials in behavior extraction, neural extraction, assembly, plotting (if requested), and checks; it also constructs per-session trial arrays in separate neural and assembly passes.

ii.
```python
for t in range(ntrials):
for tr in trials:
for tr in trials:
```

iii. The agent emphasized avoiding repeated behavior and spike reads; remaining repeated lightweight passes support staged extraction, assembly, and validation.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The conversion computes diagnostic counts, cohort/session metadata, and optional six-panel plots that are not decoder features. It also reads/records some fields such as experimental cohort and raw-count summaries solely for documentation and checks.

ii.
```python
region_counts_recorded=region_counts.tolist()
session_info.append(dict(... cohort=cohort_of(db, ets), ...))
if args.show_processing:
    plot_processing(...)
```

iii. These were intentionally retained for provenance, sanity checking, and visualization; optional plotting is disabled unless requested.
