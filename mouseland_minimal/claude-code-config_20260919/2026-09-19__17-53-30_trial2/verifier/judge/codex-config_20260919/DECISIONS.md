# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy`, eagerly loads every `Beh_<experiment type>.npy`, merges repeated index entries into one recording keyed by mouse/date/block, and loads that recording's plane-wise spikes and retinotopy separately.

ii.
```python
beh_files = {et: np.load(os.path.join(DATA_ROOT, 'beh', 'Beh_%s.npy' % et),
                         allow_pickle=True).item() for et in exp_info}
rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)
return np.load(path, allow_pickle=True)['iarea']
```

iii. The trajectory says duplicate experiment-type entries share behavior but expose different stimulus mappings, so they are merged to recover an identity for every trial and each recording is converted once.

## 1-b. How are the data split into subjects?

i. Subject identity is `mname`; unique names are sorted and each session receives an integer `subject_idx`.

ii.
```python
subjects = sorted({s['mname'] for s in sessions.values()})
'subject_idx': np.array([subjects.index(i['mouse']) for i in session_info], dtype=np.int64)
```

iii. The index explicitly supplies mouse identity, so the agent used it directly.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` recording. Repeated experiment-type entries are merged in an ordered dictionary rather than emitted as duplicate sessions.

ii.
```python
rec = '%s_%s_%s' % (ndb['mname'], ndb['datexp'], ndb['blk'])
entry = sessions.setdefault(rec, {'mname': ndb['mname'], 'datexp': ndb['datexp'],
                                  'blk': ndb['blk'], ...})
```

iii. The agent observed that the same recording occurs under several analysis labels and reported converting all 89 unique recordings once.

## 1-d. How are the data split into trials?

i. Frames are assigned using `ft_trInd`, restricted to the textured corridor (`ft_CorrSpc`), running frames (`ft_move > 0`), and non-NaN trial indices. Thus stationary frames inside a traversal are removed and retained trial samples need not be temporally contiguous.

ii.
```python
mask = (beh['ft_CorrSpc'][:nframes] &
        (beh['ft_move'][:nframes] > 0) &
        ~np.isnan(beh['ft_trInd'][:nframes]))
```

iii. The agent cited the paper's “only considered timepoints during running” statement and a released-code `corr_fr` mask, while retaining variable-length corridor-entry-to-exit trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only when they have at least five retained running frames and a recoverable canonical stimulus. It does not apply the reference's dataset-wide 99th-percentile maximum-duration filter.

ii.
```python
if len(fr) < MIN_FRAMES_PER_TRIAL or names[t] is None:
    continue
```

iii. The agent says five frames removes truncated traversals and missing mappings remove trials without a paper stimulus ID; it reports 309 of 38,110 trials removed.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane's `spks` array in `<session>_neural_data.npy`; `iarea` from retinotopy supplies each neuron's region.

ii.
```python
return np.concatenate(np.load(path, allow_pickle=True).item()['spks'], 0)
iarea = load_areas(entry['mname'], entry['datexp'])
```

iii. The agent identified `spks` as already-deconvolved Suite2p activity and `iarea` as ordered consistently with the concatenated planes.

## 2-b. How is the `neural` data processed?

i. After filtering, every neuron is z-scored across the whole recording, converted to float32, then sliced at retained trial frames. By default at most 2,000 neurons are randomly retained per session.

ii.
```python
activity = spk[usable]
activity = (activity - activity.mean(axis=1, keepdims=True)) / sd[usable][:, None]
activity = activity.astype(np.float32)
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The agent cited one paper utility that z-scores activity and argued that 2,000-neuron subsampling saves about 23-fold space without harming pilot decoder accuracy because the decoder later compresses sessions.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It retains only V1/mHV/lHV/aHV neurons, excludes zero-variance neurons, and uniformly samples 2,000 if more remain.

ii.
```python
usable = np.where((region >= 0) & (sd > 0))[0]
if nsub is not None and len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))
```

iii. The visual-area exclusion follows the paper; silent cells cannot be z-scored; subsampling was justified by storage and a four-session performance comparison.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to corridor entry and retain their variable end, but only running frames within the textured corridor are stored; no padding is used.

ii.
```python
fr = frames[t]
neural.append(np.ascontiguousarray(activity[:, fr]))
'off_start': 0.0,
'off_end': None,
```

iii. The agent reasoned that corridor entry is trial start, variable-length arrays are supported, and the running-only rule follows the paper.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning is applied: one retained column is one imaging frame. Metadata uses the median observed frame interval, about 315 ms (~3.17 Hz).

ii.
```python
'frame_period_s': float(np.median(np.diff(ft)) * 86400.0)
'time_bin_size': float(np.median([i['frame_period_s'] for i in session_info]) * 1000.0)
```

iii. The agent treated behavior and neural streams as already sharing the imaging-frame grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived from per-frame MATLAB datenum timestamps `ft` and per-trial cue timestamp `SoundTime`.

ii.
```python
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
```

iii. The agent preferred physical timestamps so time gaps remain represented after stationary frames are removed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue time minus each retained frame time is multiplied by 86,400 to produce seconds; values are positive before and negative after the cue.

ii.
```python
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
inp[0] = t_to_cue
```

iii. The trajectory describes seconds-until-cue in physical units, preserving gaps caused by the running mask.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the `fr` indices used to slice neural activity.

ii.
```python
t_to_cue = (beh['SoundTime'][t] - ft[fr]) * 86400.0
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The agent says both streams use the common imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from `datexp` and the earliest imaging date found for the same `mname`.

ii.
```python
date = datetime.date(*map(int, entry['datexp'].split('_')))
return float((date - first_day).days)
```

iii. The agent noted that actual VR-training start dates are absent and selected first imaging session as a proxy.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar-day elapsed time from a mouse's first included imaging session is computed, then broadcast over every retained frame of each trial.

ii.
```python
inp[1] = day
```

iii. The agent claimed this is literally day zero for mice with a “before learning” session and a documented proxy otherwise.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses frame timestamps `ft` and per-trial `Trial_start_time`.

ii.
```python
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
```

iii. The agent identifies `Trial_start_time`/`StartFr` with corridor entry.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start timestamp is subtracted from every retained frame timestamp and converted from days to seconds.

ii.
```python
inp[2] = t_since_start
```

iii. Physical timestamps were used so removed stationary intervals still appear as elapsed time gaps.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is indexed by the same `fr` array as neural data and starts near zero at corridor entry.

ii.
```python
t_since_start = (ft[fr] - beh['Trial_start_time'][t]) * 86400.0
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The common frame indices provide direct alignment.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes directly from per-trial `isRew`.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. The agent interprets it as whether water is available in that corridor, false for unsupervised and naive animals.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is cast to float and broadcast across the trial's retained frames.

ii.
```python
inp[3] = float(beh['isRew'][t])
```

iii. No transformation beyond broadcasting was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It merges `UniqWalls` and `stim_id` mappings from all duplicate session entries, then looks up each trial's `WallName`.

ii.
```python
for wall, sid in zip(beh['UniqWalls'], beh['stim_id']):
    entry['wall2stim'][str(wall)] = int(sid)
return [STIM_NAMES[wall2stim[str(w)]] if str(w) in wall2stim else None
        for w in entry['beh']['WallName']]
```

iii. The agent found that `TrialStim` can be a placeholder in swap comparisons, so it reconstructed labels from wall mappings.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. It encodes seven canonical identities (`circle1/2`, `leaf1/2/3`, and two leaf swaps) as 0–6 and broadcasts the code over the trial, rather than reducing wall names to four base textures.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
out[0] = STIM_NAMES.index(names[t])
```

iii. The agent argued that individual test stimuli are central to the paper and broad grouping remains recoverable from the seven-way label.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from session-level lick frame numbers `LickFr`.

ii.
```python
lick_fr = np.asarray(beh['LickFr'], dtype=float)
```

iii. The agent treated these as imaging-frame coordinates.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick-frame values are rounded to the nearest integer, bounds-checked, and marked in a boolean frame vector; multiple licks in one frame remain one.

ii.
```python
idx = np.round(lick_fr).astype(int)
idx = idx[(idx >= 0) & (idx < nframes)]
licks[idx] = True
```

iii. The agent's code defensively handles NaN and out-of-range values; the trajectory offers no further explicit rationale for rounding.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The per-frame lick vector is sliced at the same retained frame indices as neural activity.

ii.
```python
out[1] = licks[fr]
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The common imaging-frame grid is the stated alignment mechanism.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It comes from per-imaging-frame corridor position `ft_Pos`.

ii.
```python
pos = beh['ft_Pos'][:nframes]
```

iii. The agent identified these values as decimeters along the corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is divided by 10 decimeters and converted to integer category indices.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, N_POSITION_BINS - 1)
```

iii. Ten decimeters equals the requested one-meter category width.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Values map to four bins `[0,1)`, `[1,2)`, `[2,3)`, and `[3,4]` meters, clipped to codes 0–3.

ii.
```python
POSITION_BIN_DM = 10.0
np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, 3)
```

iii. The four-meter textured corridor directly motivates four equal one-meter bins.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sliced with the same `fr` indices as the neural matrix.

ii.
```python
out[2] = np.clip((pos[fr] / POSITION_BIN_DM).astype(int), 0, 3)
```

iii. Both are sampled on the imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from per-frame `ft_RunSpeed` at retained trial frames.

ii.
```python
speed = beh['ft_RunSpeed'][:nframes]
```

iii. The agent used the released per-frame running-speed stream directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Speeds from all retained trials and sessions are pooled, global 25th/50th/75th value percentiles are calculated, and each value is assigned with `searchsorted`.

ii.
```python
edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
```

iii. The agent reasoned that pooling all retained timepoints makes the four bins contain 25% of the entire converted dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global numeric percentile edges (reported as about 12.4, 25.3, and 40.8 cm/s) define codes 0–3; values equal to an edge enter the higher bin.

ii.
```python
np.searchsorted(speed_edges, speed[fr], side='right')
```

iii. The global edges were chosen to give dataset-level quartiles, though tied values can prevent exact 25% occupancy.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed categories are evaluated at the exact `fr` indices used for neural columns.

ii.
```python
out[3] = np.searchsorted(speed_edges, speed[fr], side='right')
neural.append(np.ascontiguousarray(activity[:, fr]))
```

iii. The streams already share imaging-frame coordinates.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The code truncates behavior to neural frame count; removes NaN trial indices and nonfinite lick frames; bounds-checks licks; drops trials with unknown stimuli or fewer than five retained frames; clips position; excludes zero-variance neurons; and asserts retinotopy/neuron length equality. It has no per-session exception recovery and does not enforce at least two trials before saving.

ii.
```python
in_trial = ~np.isnan(beh['ft_trInd'][:nframes])
lick_fr = lick_fr[np.isfinite(lick_fr)]
assert len(iarea) == nneurons_total, rec
if len(fr) < MIN_FRAMES_PER_TRIAL or names[t] is None:
    continue
```

iii. The agent explicitly describes missing stimulus IDs and short/truncated trials as exclusions and treats silent neurons as unusable for z-scoring.

## 12-a. What are the most time-consuming steps of the code?

i. Loading and concatenating very large spike matrices, computing full-recording standard deviations/means and z-scores, copying trial slices, and serializing the 6.5-GB pickle dominate. Session work is multiprocessing-parallelized.

ii.
```python
with mp.get_context('fork').Pool(args.workers) as pool:
    for i, res in enumerate(pool.imap(_worker, jobs)):
        ...
activity = (activity - activity.mean(axis=1, keepdims=True)) / sd[usable][:, None]
```

iii. The trajectory includes timed pilots, memory checks, background conversion, and neuron subsampling specifically to reduce the roughly 150-GB all-neuron result.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Loops over experiment entries, wall mappings, area-code mappings, sessions, trials, and per-trial speed collection remain. Trial assembly is difficult to eliminate because outputs are variable length, but area assignment and speed collection could be more vectorized; repeated `subjects.index` could use a dictionary.

ii.
```python
for code, ridx in AREA_CODE_TO_REGION.items():
    region[iarea == code] = ridx
for t in keep:
    speeds.append(beh['ft_RunSpeed'][:nframes][frames[t]])
for t in range(int(beh['ntrials'])):
    ...
```

iii. The trajectory does not explicitly discuss vectorization; it instead chose process-level parallelism across sessions.

## 12-c. What processing does the code repeat multiple times?

i. `trial_frames` and `session_stimulus_names` are computed once in `collect_behavior` and again in `convert_session`; behavior is traversed twice; speed samples are concatenated in the first pass; each session recomputes frame masks.

ii.
```python
# collect_behavior
frames = trial_frames(beh, nframes)
names = session_stimulus_names(entry)
# convert_session
frames = trial_frames(beh, nframes)
names = session_stimulus_names(entry)
```

iii. The agent calls the first pass necessary for global speed quartiles but does not explain why its already-computed retained-trial/frame information is not reused.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. `collect_behavior` constructs `per_session` retained-trial lists that are used only for printed counts, not conversion. It also stores extensive session metadata, cohort labels, and trial IDs not consumed by the decoder. Most notably it z-scores every usable neuron before randomly discarding all but 2,000, so statistics are computed for discarded neurons.

ii.
```python
sd = spk.std(axis=1)
usable = np.where((region >= 0) & (sd > 0))[0]
if len(usable) > nsub:
    usable = np.sort(rng.choice(usable, size=nsub, replace=False))
per_session[rec] = keep
```

iii. The trajectory focuses on dataset-size reduction and validation, but does not identify these discarded intermediate results as an optimization target.
