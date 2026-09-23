# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads `Imaging_Exp_info.npy` as the master index, de-duplicates recordings, loads each selected `Beh_<exp_type>.npy` once, and loads one spike and retinotopy file per session. All 89 unique recordings are included.

ii.
```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
B = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=True).item()
spks = np.load(spk_path, allow_pickle=True).item()['spks']
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
```

iii. The trajectory says the 142 index entries represent 89 unique recordings and that duplicate behavior dictionaries were byte-identical, so it used one session per recording and no session/mouse exclusion.

## 1-b. How are the data split into subjects?

i. Subjects are the sorted unique `mname` values; each session receives its mouse's index in that list.

ii.
```python
subjects = sorted({s['mname'] for s in sessions.values()})
data['subject_idx'].append(subjects.index(s['mname']))
```

iii. The trajectory identifies 19 mice and treats the explicit mouse name as the subject identifier.

## 1-c. How are the data split into sessions?

i. A session is the unique `(mname, datexp, blk)` tuple. An `OrderedDict` de-duplicates repeated experiment-type entries.

ii.
```python
def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
sessions.setdefault(kn, {...})
```

iii. The agent justified this by matching the 89 unique tuples to the paper's 89 recordings and available spike files.

## 1-d. How are the data split into trials?

i. Frames are grouped by `ft_trInd`, but only frames satisfying `ft_move > 0` and `ft_CorrSpc` are retained. Thus a stored trial is a possibly non-contiguous set of running frames in the textured corridor.

ii.
```python
valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
frames = np.nonzero(valid)[0]
tr = ft_trInd[frames].astype(np.int64)
```

iii. The trajectory cites the paper's running-only analyses and `fr_valid = VRmove & isCorridor`, arguing the decoder treats bins independently and timestamps preserve elapsed time across gaps.

## 1-e. How are trials filtered based on quality controls?

i. Trials with fewer than two retained running/corridor frames, non-finite start/cue time, or any non-finite time, position, or speed value are dropped. No long-trial percentile filter is used; the produced data retained all 38,110 trials according to the final report.

ii.
```python
if len(fr) < 2: continue
if not np.isfinite(t0) or not np.isfinite(tcue): continue
if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))): continue
```

iii. The agent reported that all trials had at least 11 retained frames and none was dropped; the checks were defensive missing-data handling.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from each plane's `spks`; neuron regions come from retinotopy `iarea`.

ii.
```python
spks = np.load(spk_path, allow_pickle=True).item()['spks']
region = neu_area_idx(np.asarray(ret['iarea']))
```

iii. The agent describes `spks` as already-deconvolved Suite2p fluorescence and `iarea` as the paper's area assignment.

## 2-b. How is the `neural` data processed?

i. Selected neuron rows and selected frame columns are copied as float32, concatenated across planes, and sliced back into trial arrays. No normalization, deconvolution, padding, or temporal rebinning is applied.

ii.
```python
out = np.empty((len(rows), len(frames)), dtype=np.float32)
out[sel] = plane[rows[sel] - offset][:, frames]
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The trajectory says the files already contain the deconvolved signal used by the paper, so further signal processing was unnecessary.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only V1, mHV, lHV, and aHV neurons are candidates. If a session has more than 2,000 candidates, a seeded random sample of 2,000 is retained.

ii.
```python
candidates = np.nonzero(region >= 0)[0]
if max_neurons is not None and len(candidates) > max_neurons:
    rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
```

iii. Area filtering was justified as matching paper analyses. The 2,000 cap was justified by file size and pilot decoder behavior: the trajectory says the decoder switches away from exact SVD above 2,000 and accuracy declined.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial start is corridor entry, but stored neural columns are only the running corridor frames for that `ft_trInd`; stopped frames are omitted and trials remain variable length.

ii.
```python
frames_out.append(fr)
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The agent says `Trial_start_time`/`StartFr` is the alignment event, and real timestamps keep temporal covariates correct despite omitted stopped frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One imaging frame is one bin; no rebinning occurs. The metadata uses the median session frame rate, about 314.7 ms per bin.

ii.
```python
frame_rates.append(1.0 / (np.median(np.diff(b['ft'])) * 86400.0))
time_bin_size = 1000.0 / float(np.median(frame_rates))
```

iii. The agent treated imaging frames as the native common grid for neural and behavior data.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. It is derived directly from per-trial `SoundTime` and per-frame `ft` timestamps.

ii.
```python
tcue = beh['SoundTime'][trial]
to_cue = (tcue - ft[fr]) * 86400.0
```

iii. The agent chose timestamps rather than frame indices to retain true elapsed seconds across removed frames.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Frame time is subtracted from cue time and MATLAB datenums are converted from days to seconds, producing positive values before and negative values after the cue.

ii.
```python
to_cue = (tcue - ft[fr]) * 86400.0
inp[0] = to_cue
```

iii. The trajectory explicitly interprets the requested name as seconds until cue.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `fr` frame indices used for neural columns.

ii.
```python
to_cue = (tcue - ft[fr]) * 86400.0
frames_out.append(fr)
```

iii. The agent states all streams share imaging-frame indices.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. It is derived from the date encoded in each session's `datexp` and the earliest recorded date for that mouse.

ii.
```python
d = datetime.date(*map(int, s['datexp'].split('_')))
first_date[s['mname']] = min(first_date.get(s['mname'], d), d)
```

iii. The trajectory says the released data lacks the absolute training-start date, so recording dates were used.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. It computes elapsed calendar days since that mouse's first recording and broadcasts the result over every retained frame.

ii.
```python
day_of_training = {kn: (date - first_date[s['mname']]).days ...}
inp[1] = day_of_training
```

iii. The agent describes this as the best available proxy for day of training.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. It uses per-trial `Trial_start_time` and per-frame `ft`.

ii.
```python
t0 = beh['Trial_start_time'][trial]
t = (ft[fr] - t0) * 86400.0
```

iii. The agent used real timestamps so stopped intervals remain reflected even though their frames are omitted.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Trial-start time is subtracted from each selected frame timestamp and days are converted to seconds.

ii.
```python
t = (ft[fr] - t0) * 86400.0
inp[2] = t
```

iii. The trajectory calls this real elapsed time since corridor entry.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed at the identical selected frame indices `fr` used for neural columns.

ii.
```python
t = (ft[fr] - t0) * 86400.0
frames_out.append(fr)
```

iii. The shared frame index is the alignment mechanism.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It comes from the per-trial `isRew` flag.

ii.
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. The agent notes it is always zero for unsupervised and naive cohorts.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The flag is converted to 0.0/1.0 and broadcast across the trial's retained frames.

ii.
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. No additional derivation was considered necessary.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. It is derived from per-trial `WallName`, session `UniqWalls`, and `stim_id` arrays in all matching experiment-index entries.

ii.
```python
for wall, sid in zip(uniq, stim_id):
    if not np.isnan(sid): mapping[wall] = int(sid)
out[0] = stim_map[wall]
```

iii. The agent wanted to preserve the paper's canonical stimulus identities and merge partial mappings across analysis entries.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Raw wall names are mapped to seven canonical `stim_id` identities; otherwise-unlabelled `circle3` becomes an eighth category. The per-trial ID is broadcast across frames.

ii.
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
EXTRA_STIM = ['circle3']
out[0] = stim_map[wall]
```

iii. The trajectory says this mirrors how the paper pools alternate texture sets and avoids discarding 390 `circle3` trials.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. It comes from session lick frame numbers in `LickFr`.

ii.
```python
lf = np.round(np.asarray(beh['LickFr'], dtype=float))
lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. The agent describes `LickFr` as directly indexing imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Finite lick frame values are rounded to nearest integer; in-range frames become 1 and all others 0.

ii.
```python
lf = lf[np.isfinite(lf)].astype(np.int64)
out[1] = lick[fr].astype(np.int64)
```

iii. The trajectory defines the output as whether at least one lick occurred in an imaging frame.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The framewise lick mask is indexed by the same `fr` used for neural data.

ii.
```python
out[1] = lick[fr].astype(np.int64)
frames_out.append(fr)
```

iii. The agent also sanity-checked that licking peaked around the cue and differed by rewarded corridor.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. It uses framewise `ft_Pos` and session `Texture_Length`.

ii.
```python
pos = beh['ft_Pos'][:nfr]
texture_len = float(beh['Texture_Length'])
```

iii. The agent treats the texture length (40 raw units) as the requested 4 m corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The corridor is divided by four; position divided by that bin length is integer-cast and clipped to categories 0–3.

ii.
```python
bin_len = texture_len / len(POSITION_VALUES)
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, 3)
```

iii. This directly implements four equal 1 m spatial bins.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are 0, 10, 20, 30, and 40 raw position units (equivalent to 0–1, 1–2, 2–3, and 3–4 m); values are clipped into 0–3.

ii.
```python
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```

iii. The choice follows the explicit four-equal-length-bin requirement.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is selected at the same `fr` frame indices as neural activity.

ii.
```python
p = pos[fr]
out[2] = pos_bin
```

iii. The agent relies on the common imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. It is derived from framewise `ft_RunSpeed` at retained frames.

ii.
```python
speed = beh['ft_RunSpeed'][:nfr]
s = speed[fr]
```

iii. The agent uses the dataset's direct running-speed measurement.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first behavior-only pass collects all retained speeds across the full dataset, computes global 25th/50th/75th percentiles, and assigns each speed with `searchsorted`.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.searchsorted(speed_edges, s, side='right')
```

iii. The agent says global thresholds make category meanings consistent across sessions.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three global edges (reported as about 10.4, 20.2, and 32.0 cm/s) define four bins; values equal to an edge enter the higher bin.

ii.
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
np.searchsorted(speed_edges, s, side='right')
```

iii. The trajectory interprets “each corresponding to 25% of the data” globally across kept timepoints.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Speed values are selected by the same trial frame indices as neural columns.

ii.
```python
s = speed[fr]
out[3] = np.searchsorted(speed_edges, s, side='right')
```

iii. The common frame indices provide alignment.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Behavior is bounded to available imaging frames; invalid trial indices are excluded; invalid/out-of-range licks are ignored; trials with too few frames or non-finite required variables are skipped; mismatched neuron counts and bad/conflicting stimulus maps raise errors.

ii.
```python
nfr = min(p.shape[1] for p in spks)
valid = ... & ~np.isnan(ft_trInd)
if not np.isfinite(t0) or not np.isfinite(tcue): continue
if n_file != n_total: raise ValueError(...)
```

iii. The trajectory characterizes the dataset as clean and these as defensive checks; it reports no trial was ultimately lost.

## 12-a. What are the most time-consuming steps of the code?

i. Loading hundreds of gigabytes of spike planes and copying selected neuron/frame values dominate; full conversion is parallelized by session.

ii.
```python
with Pool(args.workers, maxtasksperchild=1) as pool:
    for res in pool.imap_unordered(process_session, jobs): ...
spks = np.load(spk_path, allow_pickle=True).item()['spks']
```

iii. The agent's process and reference both identify neural-file I/O as dominant; the trajectory also spent substantial time on pilot conversions/decoder training to select the neuron cap.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Most frame grouping is vectorized. Remaining per-trial construction, per-plane extraction, subject lookup via `list.index`, stimulus mapping, and session-level first passes could be streamlined, though trial arrays inherently require per-trial objects.

ii.
```python
return [frames[bounds[i]:bounds[i + 1]] for i in range(int(beh['ntrials']))]
for trial, fr in enumerate(per_trial): ...
for plane in spks: ...
```

iii. The trajectory does not explicitly discuss vectorization; its optimization focus was parallel session processing and avoiding full neuron concatenation.

## 12-c. What processing does the code repeat multiple times?

i. `session_behaviour_arrays` is run once over every session to collect speed statistics and again inside each worker to create final inputs/outputs. That repeats stimulus-map construction, trial-frame grouping, lick masking, validation, and behavioral arrays.

ii.
```python
arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
...
beh_arrays = session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges)
```

iii. The repetition is intentional because global speed edges must be known before final speed categories can be assigned; the trajectory calls this a behavior-only pass.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The first pass constructs full input/output trial arrays even though only retained speeds and frame rates are used; its provisional speed output is all zeros. It also calculates/stores extensive metadata and performs pilot decoder experiments not required by conversion itself.

ii.
```python
if speed_edges is not None:
    out[3] = np.searchsorted(speed_edges, s, side='right')
else:
    out[3] = 0
speeds.append(arrays['speeds'])
```

iii. The agent used the pass to establish global quartiles and used pilot analyses to justify 2,000 neurons, but much of the pass's allocated output is discarded.
