# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master index from `beh/Imaging_Exp_info.npy`, builds one session record per unique `(mname, datexp, blk)` recording, then loads one behavior dictionary per experiment type and one spike file plus one retinotopy file per session. Unlike the reference, it also keeps all `stim_id` entries for a recording so it can merge them later into a canonical stimulus map.

ii. 
```python
exp_info = np.load(os.path.join(root, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=True).item()
```
```python
for exp_type, db in exp_info.items():
    for d in db:
        kn = session_key(d)
        s = sessions.setdefault(kn, {
            'kn': kn, 'mname': d['mname'], 'datexp': d['datexp'], 'blk': d['blk'],
            'exp_types': [], 'stim_map': {}, 'beh_source': (exp_type, beh_key(d)),
        })
```
```python
B = np.load(os.path.join(root, 'beh', 'Beh_%s.npy' % exp_type),
            allow_pickle=True).item()
spks = np.load(spk_path, allow_pickle=True).item()['spks']
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
```

iii. In the trajectory, the AI explicitly justified de-duplicating the 142 index entries down to 89 recordings because they matched the 89 spike files, and because duplicated behavior dictionaries were “byte-identical” across experiment types (steps 20, 38, 100). It also said it was following `utils.py` and preserving all `stim_id` tables for later stimulus remapping (steps 34, 100).

## 1-b. How are the data split into subjects (mice)?

i. Subjects are split by the `mname` field from the experiment index. The final `subjects` list is the sorted set of unique mouse names, and `subject_idx` is the index of each session’s mouse in that sorted list.

ii. 
```python
def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
```
```python
subjects = sorted({s['mname'] for s in sessions.values()})
...
data['subject_idx'].append(subjects.index(s['mname']))
```

iii. The trajectory shows the AI counted 19 mice and grouped sessions per mouse from the index itself (steps 56, 84, 100). The AI did not derive subject identity from any secondary source.

## 1-c. How are the data split into sessions?

i. A session is one unique recording identified by `(mname, datexp, blk)`. The AI keeps only one session entry for each unique key even when the same recording appears in multiple experiment-type tables.

ii. 
```python
def session_key(d):
    return '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
```
```python
kn = session_key(d)
s = sessions.setdefault(kn, {
    'kn': kn, 'mname': d['mname'], 'datexp': d['datexp'], 'blk': d['blk'],
    'exp_types': [], 'stim_map': {}, 'beh_source': (exp_type, beh_key(d)),
})
```

iii. The AI justified this in the trajectory by noting that `Imaging_Exp_info.npy` contains 142 rows but only 89 unique recordings, matching the paper’s “89 recordings in 19 mice,” and by verifying that duplicate behavior records were identical (steps 20, 38, 100).

## 1-d. How are the data split into trials?

i. The AI treats each trial as one traversal of the textured corridor, but in code it does not keep every frame from entry to grey-space entry. Instead, within each trial it keeps only imaging frames where `ft_move > 0`, `ft_CorrSpc` is true, and `ft_trInd` is not NaN. This means the stored trials are sparse, running-only subsets of the traversal rather than contiguous full traversals.

ii. 
```python
def trial_frames(beh, nfr):
    ft_trInd = beh['ft_trInd'][:nfr]
    valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
    frames = np.nonzero(valid)[0]
    tr = ft_trInd[frames].astype(np.int64)
    ...
    return [frames[bounds[i]:bounds[i + 1]] for i in range(int(beh['ntrials']))]
```

iii. The AI’s justification was that this matches `utils.py`’s `fr_valid = VRmove & isCorridor` selection and the paper’s statement “We only considered timepoints during running for analysis” (steps 3, 11, 40, 100). It explicitly argued that dropping stopped frames was acceptable because the decoder predicts each time bin independently (step 100).

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s long-trial outlier filter. Instead, it keeps a trial if it has at least 2 selected frames and finite `Trial_start_time`, `SoundTime`, per-frame time-to-start, time-to-cue, position, and running-speed values. In the final run it reported that all 38,110 trials survived.

ii. 
```python
for trial, fr in enumerate(per_trial):
    if len(fr) < 2:
        continue
    t0 = beh['Trial_start_time'][trial]
    tcue = beh['SoundTime'][trial]
    ...
    if not np.isfinite(t0) or not np.isfinite(tcue):
        continue
    ...
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(to_cue))):
        continue
    ...
    if not (np.all(np.isfinite(p)) and np.all(np.isfinite(s))):
        continue
```

iii. In the trajectory, the AI inspected frame counts and concluded that every trial had at least 11 selected frames, so “none were dropped” in the final dataset (steps 54, 84, 100). Its quality-control logic was therefore driven by validity checks on the selected frames, not by the reference’s percentile-based length filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data are derived from `spks` in each session’s spike file and `iarea` from the corresponding retinotopy file. The AI concatenates imaging planes logically by indexing across the list of plane arrays and uses `iarea` only to decide which neurons to keep and how to label their region.

ii. 
```python
spks = np.load(spk_path, allow_pickle=True).item()['spks']
```
```python
ret = np.load(os.path.join(root, 'retinotopy', '%s_%s_trans.npz' %
                           (session['mname'], session['datexp'])), allow_pickle=True)
region = neu_area_idx(np.asarray(ret['iarea']))
```
```python
for plane in spks:
    ...
    out[sel] = plane[rows[sel] - offset][:, frames]
```

iii. The AI said in the trajectory that it was using the paper’s “Suite2p deconvolved fluorescence” traces directly and using `utils.neu_area_ID` to restrict neurons to the four visual-area groups analyzed in the paper (steps 40, 50, 100).

## 2-b. How is the `neural` data processed?

i. The AI leaves the neural signal in its deconvolved form with no normalization or temporal rebinning, but it does two extra processing steps absent from the reference: it randomly subsamples visual-area neurons to at most 2,000 per session, and it stores the selected neural matrix as `float32` rather than `float16`.

ii. 
```python
candidates = np.nonzero(region >= 0)[0]
...
if max_neurons is not None and len(candidates) > max_neurons:
    rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
```
```python
out = np.empty((len(rows), len(frames)), dtype=np.float32)
...
neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The trajectory shows the AI justified the 2,000-neuron cap empirically: it ran pilot conversions at several neuron counts and argued that decoder accuracy worsened above 2,000 because `train_decoder` switches to a lossy random projection when `svd_max_neurons > 2000` (steps 29, 48, 75, 79, 81, 100). It also said the cap reduced output size from about 175 GB to 6.6 GB (steps 48, 100).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Neurons are first filtered to those whose `iarea` falls in one of four visual-region groups (`V1`, `mHV`, `lHV`, `aHV`). After that, if more than 2,000 neurons remain, the AI randomly subsamples 2,000 of them per session with a seeded RNG.

ii. 
```python
idx = np.full(len(iarea), -1, dtype=np.int64)
idx[iarea == 8] = 0
idx[np.isin(iarea, [0, 1, 2, 9])] = 1
idx[np.isin(iarea, [5, 6])] = 2
idx[np.isin(iarea, [3, 4])] = 3
```
```python
candidates = np.nonzero(region >= 0)[0]
...
rows = np.sort(rng.choice(candidates, size=max_neurons, replace=False))
```

iii. The AI justified the area filter by citing `utils.neu_area_ID` and stating that no paper analysis used neurons with `iarea == -1` or `7` (steps 11, 50, 100). It justified the additional random subsampling only as a decoder/memory optimization, not as a paper-derived quality control (steps 75, 81, 100).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says the alignment event is trial start / corridor entry, but the stored neural arrays begin at the first selected running frame inside the corridor for that trial, not necessarily the literal first corridor-entry frame. After that, the trial contains only selected running frames until the end of the textured corridor.

ii. 
```python
'temporal_alignment_event':
    'trial start = entry into the textured virtual-reality corridor '
    '(beh["Trial_start_time"] / "StartFr")',
```
```python
valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
```
```python
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI defended this in the trajectory by saying that `ft_CorrSpc & (ft_move > 0)` was the frame selection used in `utils.py` and that using real frame timestamps preserved time-to-cue and time-since-start despite the gaps (steps 40, 100).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is one imaging frame. The AI estimates frame rate from the median difference of `ft` and stores the median implied bin size in milliseconds. It does not rebin or resample in time.

ii. 
```python
frame_rates.append(1.0 / (np.median(np.diff(b['ft'])) * 86400.0))
...
time_bin_size = 1000.0 / float(np.median(frame_rates))
```
```python
'time_bin_size': float(time_bin_size),
```

iii. The AI checked in the trajectory that frame rates were tightly clustered around 3.17 Hz and reported a final time bin of about 314.7 ms (steps 44, 54, 100). It did not justify any resampling because it did none.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives `time_to_sound_cue` from `SoundTime` and per-frame timestamps `ft`. It does not use `SoundFr`.

ii. 
```python
ft = beh['ft'][:nfr]
...
tcue = beh['SoundTime'][trial]
...
to_cue = (tcue - ft[fr]) * 86400.0
```

iii. In the trajectory, the AI described this variable as “seconds until the sound cue (`SoundTime`, positive before the cue)” and emphasized that it used real timestamps so the values remained valid even after dropping stopped frames (step 100).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every kept frame in a trial, the AI subtracts that frame’s timestamp from the trial’s `SoundTime` and converts MATLAB-day units to seconds. The result is continuous, positive before cue onset and negative after cue onset.

ii. 
```python
to_cue = (tcue - ft[fr]) * 86400.0
...
inp[0] = to_cue
```

iii. The AI justified the sign convention and real-time computation in the final trajectory summary, where it explicitly described the variable as “seconds until the sound cue (`SoundTime`, positive before the cue)” (step 100).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. The AI aligns this input by computing it on the exact same per-trial frame index array `fr` used to extract neural activity for that trial.

ii. 
```python
t = (ft[fr] - t0) * 86400.0
to_cue = (tcue - ft[fr]) * 86400.0
```
```python
out[sel] = plane[rows[sel] - offset][:, frames]
...
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI’s trajectory justification was that all streams were aligned through shared imaging-frame timestamps, even though it removed non-running frames (step 100).

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives this variable from session dates `datexp` and mouse identity `mname`, not from a per-trial field. It computes the earliest recording date seen for each mouse and then measures each session relative to that.

ii. 
```python
for kn, s in sessions.items():
    d = datetime.date(*map(int, s['datexp'].split('_')))
    first_date[s['mname']] = min(first_date.get(s['mname'], d), d)
```

iii. The AI justified this in code comments and in the trajectory by saying the released dataset does not contain the absolute start of training, so it used days since the first recording of that mouse (lines 297-305; step 100).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI computes a calendar-day difference between a session’s recording date and that mouse’s earliest recording date in the release. It then broadcasts that scalar across all time bins in each trial of the session.

ii. 
```python
day_of_training = {kn: (datetime.date(*map(int, s['datexp'].split('_')))
                        - first_date[s['mname']]).days
                   for kn, s in sessions.items()}
```
```python
inp[1] = day_of_training
```

iii. The trajectory explicitly says “day of training = days since that mouse’s first recording” and notes that the data do not expose an absolute training start date (step 100).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives this input from `Trial_start_time` and the per-frame timestamps `ft`. It does not use `StartFr`.

ii. 
```python
t0 = beh['Trial_start_time'][trial]
ft = beh['ft'][:nfr]
...
t = (ft[fr] - t0) * 86400.0
```

iii. The AI justified this in the trajectory as “seconds since corridor entry,” explicitly using real elapsed time rather than frame indices (step 100).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame, the AI subtracts the trial’s `Trial_start_time` from the frame timestamp `ft` and converts from days to seconds. Because it uses the real timestamps of the kept frames, the variable includes waiting/stopping time even though the stopped frames themselves are omitted.

ii. 
```python
t = (ft[fr] - t0) * 86400.0
...
inp[2] = t
```

iii. In the trajectory, the AI highlighted this as “real elapsed time, so it carries the animal’s stopping behaviour,” even though the dataset stores only running frames (step 100).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is aligned by evaluating it on the same `fr` frame indices that define the neural trial window.

ii. 
```python
t = (ft[fr] - t0) * 86400.0
...
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI’s trajectory justification was the same as for the cue input: shared frame timestamps keep the input aligned with the selected neural bins (step 100).

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. It is derived from the per-trial boolean/indicator `isRew`.

ii. 
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. The AI said in the trajectory that reward availability is `isRew`, and also noted that it is always `0` for the unsupervised and naive cohorts (step 100).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. No transformation beyond converting `isRew` to `1.0` or `0.0` and broadcasting that per-trial value across the kept time bins of the trial.

ii. 
```python
inp[3] = 1.0 if beh['isRew'][trial] else 0.0
```

iii. The AI did not cite a paper-derived transformation here; the trajectory treats this as a direct trial label taken from the raw behavior (step 100).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus label from the trial’s `WallName`, but it also consults the `stim_id` arrays stored in `Imaging_Exp_info.npy` for all entries corresponding to that recording. It merges those `stim_id` tables into a session-specific wall-name-to-canonical-stimulus mapping.

ii. 
```python
s['stim_id_entries'] = s.get('stim_id_entries', []) + \
    [(exp_type, beh_key(d), np.asarray(d['stim_id'], dtype=float))]
```
```python
for exp_type, key, stim_id in session['stim_id_entries']:
    ...
    for wall, sid in zip(uniq, stim_id):
        if not np.isnan(sid):
            ...
            mapping[wall] = sid
```
```python
wall = beh['WallName'][trial]
out[0] = stim_map[wall]
```

iii. The AI justified this in the trajectory by arguing that the paper pooled different wall-name sets through the authors’ `stim_id` tables, especially for rock/brick/wood cohorts and swap sessions (steps 34, 36, 100).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI does not collapse textures to four broad classes. Instead, it preserves up to eight canonical stimulus identities: `circle1`, `circle2`, `leaf1`, `leaf2`, `leaf3`, `leaf1_swap1`, `leaf1_swap2`, and a fallback `circle3`. It assigns one of these categories per trial and broadcasts it across the kept time bins.

ii. 
```python
CANONICAL_STIM = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
                  'leaf1_swap1', 'leaf1_swap2']
EXTRA_STIM = ['circle3']
STIM_VALUES = CANONICAL_STIM + EXTRA_STIM
```
```python
for wall in uniq:
    if wall not in mapping:
        if wall not in EXTRA_STIM:
            raise ValueError(...)
        mapping[wall] = len(CANONICAL_STIM) + EXTRA_STIM.index(wall)
```
```python
out[0] = stim_map[wall]
```

iii. The AI justified this by saying `circle3` is never labeled by the paper and that keeping it as an eighth category avoids discarding 390 trials; it also said the canonical `stim_id` scheme is how the paper pools equivalent rock/brick/wood and leaf/circle stimuli (steps 46, 84, 100).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. Licking is derived from `LickFr`, which is interpreted as imaging-frame indices for lick events.

ii. 
```python
lick = np.zeros(nfr, dtype=bool)
if len(beh['LickFr']) > 0:
    lf = np.round(np.asarray(beh['LickFr'], dtype=float))
    lf = lf[np.isfinite(lf)].astype(np.int64)
    lick[lf[(lf >= 0) & (lf < nfr)]] = True
```

iii. The AI treated licking as directly available from frame-indexed raw behavior and later used trajectory checks on lick-rate profiles to validate that this stream was aligned sensibly (steps 58, 90, 100).

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each lick frame to the nearest integer frame, drops non-finite or out-of-range lick indices, and builds a binary per-frame vector where `True` means at least one lick landed in that frame. Trial outputs then index that vector by the kept frame list.

ii. 
```python
lf = np.round(np.asarray(beh['LickFr'], dtype=float))
lf = lf[np.isfinite(lf)].astype(np.int64)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
...
out[1] = lick[fr].astype(np.int64)
```

iii. The AI did not provide a long explicit justification for rounding, but its trajectory validation focused on whether the resulting licking output reproduced expected reward- and cue-related structure, which it reported that it did (step 90).

## 8-c. How is `output` *Licking* aligned with the neural data?

i. It is aligned by sampling the session-wide per-frame lick vector on the same per-trial frame index array `fr` used for neural extraction.

ii. 
```python
out[1] = lick[fr].astype(np.int64)
```
```python
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI’s trajectory validation checked that lick probability peaked around the cue and differed between rewarded and non-rewarded corridors, which it used as evidence that neural and lick alignment was sensible (step 90).

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. Position is derived from the per-frame position stream `ft_Pos`.

ii. 
```python
pos = beh['ft_Pos'][:nfr]
```

iii. In the trajectory, the AI confirmed that corridor positions on selected frames spanned roughly `0` to `40`, matching the textured 4 m corridor, and it used that to justify its 4-bin discretization (steps 42, 100).

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI uses `Texture_Length` to infer corridor length in the position units, divides by four to get equal bin size, and then converts each selected frame’s position into a bin index.

ii. 
```python
texture_len = float(beh['Texture_Length'])          # 40 units == 4 m
bin_len = texture_len / len(POSITION_VALUES)
...
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```

iii. The AI justified this as producing four equal 1 m bins over the textured corridor, consistent with the decoder task and with the data’s `Texture_Length = 40` units (steps 42, 100).

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded into four equal-width bins over the textured corridor by dividing by `Texture_Length / 4` and clipping to the range `0..3`.

ii. 
```python
POSITION_VALUES = ['0-1m', '1-2m', '2-3m', '3-4m']
...
pos_bin = np.clip((p / bin_len).astype(np.int64), 0, len(POSITION_VALUES) - 1)
```

iii. The trajectory repeatedly described this as “the 4-m corridor in four 1-m bins,” and the AI later checked that the resulting distribution was close to uniform across bins (steps 84, 100).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is already frame-indexed, and the AI aligns it by indexing `ft_Pos` with the same `fr` frame list used for that trial’s neural data.

ii. 
```python
p = pos[fr]
...
out[2] = pos_bin
```
```python
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI justified alignment through shared imaging frames and later checked that position remained monotonic within example trials, which it used as a sanity check on the alignment (step 90).

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. Running speed is derived from `ft_RunSpeed` on the selected imaging frames.

ii. 
```python
speed = beh['ft_RunSpeed'][:nfr]
...
speeds=np.concatenate([speed[f] for f in frames_out]) if frames_out else np.zeros(0)
```

iii. The AI justified this directly in the trajectory as using `ft_RunSpeed` and then discretizing it into quartiles over kept timepoints (steps 68, 100).

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI performs a first pass over all sessions to collect running speeds from all kept timepoints, computes global 25th/50th/75th percentiles, and then applies those three thresholds session-by-session to convert each frame’s speed into one of four bins.

ii. 
```python
for kn, s in sessions.items():
    ...
    arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
    speeds.append(arrays['speeds'])
...
speed_edges = np.percentile(speeds, [25, 50, 75])
```
```python
if speed_edges is not None:
    out[3] = np.searchsorted(speed_edges, s, side='right')
```

iii. The AI justified this in the trajectory by saying that “global quartiles” make the speed classes mean the same thing in every session, and it reported the final thresholds as roughly `10.4 / 20.2 / 32.0 cm/s` (steps 64, 100).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses three global percentile edges and `np.searchsorted(..., side='right')` to threshold speed into four ordinal categories `speed_q1` to `speed_q4`.

ii. 
```python
SPEED_VALUES = ['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']
...
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.searchsorted(speed_edges, s, side='right')
```

iii. The AI’s justification was that the quartiles were computed over all kept timepoints in the dataset so that the category definitions were shared across sessions (steps 64, 84, 100).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is aligned by indexing `ft_RunSpeed` on the same selected frame list `fr` that defines the neural and other per-trial arrays.

ii. 
```python
s = speed[fr]
...
out[3] = np.searchsorted(speed_edges, s, side='right')
```
```python
for fr in beh_arrays['frames']:
    neural.append(np.ascontiguousarray(out[:, start:start + len(fr)]))
```

iii. The AI’s trajectory justification again relied on shared imaging-frame indexing for all streams (step 100).

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI guards against several data problems: it truncates behavior-derived arrays to the number of available imaging frames, removes NaN trial indices when selecting frames, drops non-finite or out-of-range lick indices, and skips trials with too few selected frames or non-finite timing/position/speed values. It also checks that retinotopy and spike files agree on neuron count.

ii. 
```python
valid = (beh['ft_move'][:nfr] > 0) & beh['ft_CorrSpc'][:nfr] & ~np.isnan(ft_trInd)
```
```python
lf = lf[np.isfinite(lf)].astype(np.int64)
lick[lf[(lf >= 0) & (lf < nfr)]] = True
```
```python
if len(fr) < 2:
    continue
...
if not np.isfinite(t0) or not np.isfinite(tcue):
    continue
...
if n_file != n_total:
    raise ValueError(...)
```

iii. The trajectory shows the AI explicitly inspected NaNs in `ft_trInd`, licks, trial lengths, and time ranges (steps 42, 52, 54), then encoded those checks as skip conditions rather than attempting to impute missing values.

## 12-a. What are the most time-consuming steps of the code?

i. The dominant cost is reading and slicing the huge spike files, especially because the dataset contains millions of neurons across 89 sessions. The AI also added a full first behavioral pass to compute global speed quartiles before the main spike-loading pass.

ii. 
```python
spks = np.load(spk_path, allow_pickle=True).item()['spks']
...
for plane in spks:
    ...
    out[sel] = plane[rows[sel] - offset][:, frames]
```
```python
for kn, s in sessions.items():
    ...
    arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
    speeds.append(arrays['speeds'])
```

iii. In the trajectory, the AI measured the all-neuron conversion at about 175 GB and emphasized that the neural read/extract path was the heavy part, which is why it introduced the 2,000-neuron cap and multiprocessing (steps 48, 82, 100).

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized one loop that the reference leaves explicit: it groups valid frames for all trials in one pass using sorting and `searchsorted` instead of scanning all frames separately for each trial. Remaining obvious loops are the per-trial loop in `session_behaviour_arrays` and the per-plane extraction loop in `process_session`.

ii. 
```python
frames = np.nonzero(valid)[0]
tr = ft_trInd[frames].astype(np.int64)
order = np.argsort(tr, kind='stable')
frames, tr = frames[order], tr[order]
bounds = np.searchsorted(tr, np.arange(int(beh['ntrials']) + 1))
```
```python
for trial, fr in enumerate(per_trial):
    ...
```
```python
for plane in spks:
    ...
```

iii. The trajectory does not discuss vectorization explicitly, but the written code shows that the AI intentionally replaced the slower reference-style per-trial frame search with a grouped implementation while leaving the trial-assembly and per-plane loops in place.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats `session_behaviour_arrays` for every session: once in a behavior-only first pass to gather speed values and estimate frame rates, and again inside `process_session` to rebuild the same frame lists, inputs, and outputs with the finalized speed thresholds.

ii. 
```python
for kn, s in sessions.items():
    ...
    arrays = session_behaviour_arrays(s, b, nfr, day_of_training[kn])
    speeds.append(arrays['speeds'])
```
```python
beh_arrays = session_behaviour_arrays(session, beh, nfr, day_of_training, speed_edges)
```

iii. The trajectory justifies the first pass by the desire to compute global speed quartiles (steps 64, 100), but it does not mention any attempt to cache and reuse the already computed per-trial frame selections or input/output arrays.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Relative to the decoder task, the AI performs substantial extra work for metadata and optimization experiments: it stores experiment-type provenance, wall-name-to-stimulus mappings, reward-mode strings, trial-count summaries, and neuron-count summaries for every session; it also ran pilot conversions at several neuron caps to choose 2,000 neurons/session. Most of that work is not used by downstream decoding.

ii. 
```python
stim_counts = Counter(STIM_VALUES[int(o[0, 0])] for o in r['output'])
session_info.append({
    'session': kn, 'subject': s['mname'], 'date': s['datexp'], 'block': s['blk'],
    'experiment_types': s['exp_types'],
    'day_of_training': day_of_training[kn],
    'n_trials': len(r['neural']),
    'n_timepoints': int(sum(x.shape[1] for x in r['neural'])),
    'n_neurons': r['n_neurons'],
    'n_neurons_recorded': r['n_neurons_total'],
    'n_neurons_in_visual_areas': r['n_neurons_in_areas'],
    'wall_name_to_stimulus': {w: STIM_VALUES[i] for w, i in r['stim_map'].items()},
    'stimulus_trial_counts': dict(stim_counts),
    'rewarded_trials': int(sum(int(x[3, 0]) for x in r['input'])),
    'reward_mode': str(beh[kn]['Reward_Mode']),
})
```

iii. The trajectory shows that the AI also did several pilot conversions purely to choose the neuron cap (steps 72, 75, 77, 79, 81, 100). Those experiments informed its design choice, but they are not part of the actual downstream analysis format.
