# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads the master session index from `beh/Imaging_Exp_info.npy`, iterates over every behavior file `Beh_<exp_type>.npy`, and collapses duplicate recordings into a single `recordings[key]` entry keyed by `mname_datexp_blk`. For each recording it stores behavior arrays in memory, merges `stim_id` labels across experiment types into a `wallmap`, and later loads spikes from `spk/<session>_neural_data.npy` and retinotopy from `retinotopy/<mouse>_<date>_trans.npz` session by session.

ii. 
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'),
                   allow_pickle=1).item()

for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type),
                  allow_pickle=1).item()
```

```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
behkey = key + ('_%s' % d['stimtype'] if 'stimtype' in d else '')
beh = Beh[behkey]
```

```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
iarea = np.load(fn, allow_pickle=True)['iarea']
```

iii. In the final trajectory summary, the agent said that a recording can appear in up to five experiment types, that the behavior arrays are identical across copies except for `stim_id`, and that it therefore read behavior once per recording and merged `stim_id` across experiment types to “fully label every trial.”

## 1-b. How are the data split into subjects?

i. The AI uses `mname` as the subject id. Sessions are sorted by `(mname, datexp, blk)`, `subjects` is populated from first occurrence of each `mname`, and `subject_idx` stores the index of that subject for each session.

ii. 
```python
rec.update(dict(
    mname=d['mname'], datexp=d['datexp'], blk=d['blk'],
```

```python
keys = sorted(recordings.keys(),
              key=lambda k: (recordings[k]['mname'], recordings[k]['datexp'],
                             recordings[k]['blk']))
```

```python
if rec['mname'] not in subjects:
    subjects.append(rec['mname'])
subject_idx.append(subjects.index(rec['mname']))
```

iii. The final trajectory summary explicitly states “89 recordings from 19 mice” and defines a session as one `mname_datexp_blk` recording, implying that `mname` is the subject split.

## 1-c. How are the data split into sessions?

i. The AI defines one session as one unique recording `mname_datexp_blk`. If the same recording appears under several experiment types, it is merged into one session entry.

ii. 
```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
rec['exp_types'].append(exp_type)
```

```python
keys = sorted(recordings.keys(),
              key=lambda k: (recordings[k]['mname'], recordings[k]['datexp'],
                             recordings[k]['blk']))
```

iii. The final trajectory summary says “Session = one recording (`mname_datexp_blk`)" and justifies merging duplicates because the same recording appears in multiple experiment types with identical behavior arrays except for `stim_id`.

## 1-d. How are the data split into trials?

i. The AI treats each trial as one corridor traversal, but it does not keep all frames in that traversal. It keeps only imaging frames that are simultaneously assigned to a trial, inside the textured corridor, and marked as moving (`ft_move > 0`). Trial `t` is the subset of those kept frames whose `ft_trInd` equals `t`.

ii. 
```python
keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
frames = np.nonzero(keep)[0]
return frames, trind[frames].astype(int)
```

```python
for t in range(rec['ntrials']):
    f = frames[trials == t]
    if len(f) == 0:
        continue
```

iii. The final trajectory summary says “Trial = one corridor traversal, aligned to corridor entry,” and justifies the extra running filter as the paper’s standard selection, saying it makes traversals comparable by reducing them from roughly 11-600 frames to 11-30 frames.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not apply the reference solution’s long-trial percentile filter. Instead, it drops trials if they have no kept frames after the corridor-and-running mask or if their stimulus cannot be labeled in the merged canonical `stim_id` scheme (`stim_of_trial[t] < 0`, reported as unlabeled `circle3` trials).

ii. 
```python
if stim_of_trial[t] < 0:
    n_dropped_stim += 1
    continue
f = frames[trials == t]
if len(f) == 0:
    continue
```

```python
data['metadata'] = {
    ...
    'ntrials_dropped_unlabelled_stimulus': n_dropped_stim,
```

iii. The final trajectory summary says the AI dropped 309 trials with an unlabeled stimulus because “the paper never labels” that third variant on the common scale. No explicit trajectory justification was given for omitting a long-trial outlier filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural arrays are derived from `spks` in each session’s spike file, and neuron region labels are derived from `iarea` in the matching retinotopy file.

ii. 
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
```

```python
fn = os.path.join(ROOT, 'retinotopy', '%s_%s_trans.npz' % (rec['mname'], rec['datexp']))
iarea = np.load(fn, allow_pickle=True)['iarea']
```

iii. The final trajectory summary says the AI kept neurons assigned by the retinotopy maps to V1/mHV/lHV/aHV, and the code comments written in the trajectory state that `utils.load_spk` and the retinotopy `iarea` ordering match plane-by-plane.

## 2-b. How is the `neural` data processed?

i. The AI concatenates or, more precisely, reconstructs a selected subset of neurons across imaging planes into one session matrix, keeps only selected neurons, leaves time at native frame resolution, and stores each trial as `spk[:, f]` with `float32` dtype. It does not cast neural data to `float16`.

ii. 
```python
out = np.empty((len(sel), nframes), dtype=np.float32)
for p, plane in enumerate(planes):
    m = (sel >= offsets[p]) & (sel < offsets[p + 1])
    if m.any():
        out[m] = plane[sel[m] - offsets[p]]
```

```python
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says the AI used native ~3.18 Hz imaging frames and randomly subsampled neurons to 2000 per session to avoid a dataset “hundreds of GB” in size and because the reference decoder’s exact SVD initialization does not benefit from more than 2000 neurons.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI filters neurons to those whose retinotopy labels map to `V1`, `mHV`, `lHV`, or `aHV`, then randomly subsamples those valid neurons down to at most 2000 per session.

ii. 
```python
area_idx = np.full(len(iarea), -1, dtype=np.int64)
for a, name in enumerate(AREA_NAMES):
    area_idx[np.isin(iarea, AREA_CODES[name])] = a

valid = np.nonzero(area_idx >= 0)[0]
if len(valid) > NEURONS_PER_SESSION:
    valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))
```

iii. The final trajectory summary says the AI dropped `iarea` values outside the four visual areas and added uniform random subsampling because sessions contain 20k-90k neurons and otherwise the output would be too large.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI says neural data are aligned to corridor entry, but operationally each trial contains only the kept frames for that trial after corridor and running filtering. The arrays therefore start at the first kept imaging frame for the traversal and remain variable-length.

ii. 
```python
keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
frames = np.nonzero(keep)[0]
```

```python
for t in range(rec['ntrials']):
    f = frames[trials == t]
    ...
    neural_s.append(spk[:, f])
```

iii. The final trajectory summary says trials are “aligned to corridor entry” and justified keeping only frames in the 4 m textured corridor and during running because that matches the paper’s standard selection and produces comparable traversals.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI keeps the native two-photon imaging frame grid and applies no temporal rebinning. It estimates the time bin size from the median frame interval in each session and stores the dataset-level mean in metadata.

ii. 
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
dts.append(dt)
```

```python
'time_bin_size': float(np.mean(dts)) * 1000.0,
```

iii. The final trajectory summary says “Timepoints are the native ~3.18 Hz imaging frames” and does not mention any resampling.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The AI derives it from the per-trial sound-cue frame `SoundFr`, the selected imaging frame indices `f`, and a per-session frame duration `dt` estimated from `ft`.

ii. 
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
```

```python
time_to_cue = (rec['SoundFr'][t] - f) * dt
```

iii. The final trajectory summary says the AI used “signed seconds to the sound cue,” and the code computes that directly from sound-cue frame indices converted to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame in a trial, the AI subtracts that frame index from the trial’s fractional `SoundFr` and multiplies by `dt`, producing seconds until cue that are positive before the cue and negative after it. It does not interpolate cue time onto the actual `ft` timestamps.

ii. 
```python
# time to the sound cue: positive before the cue, negative after it
time_to_cue = (rec['SoundFr'][t] - f) * dt
```

iii. The final trajectory summary explicitly states “signed seconds to the sound cue (positive before)” and says these times were left unmodified despite a rare long tail caused by dropping idle frames.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same kept frame indices `f` used to slice the neural matrix for that trial.

ii. 
```python
time_to_cue = (rec['SoundFr'][t] - f) * dt
...
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says the AI used native imaging frames for trial timepoints, so all time-varying inputs and outputs are aligned on that selected frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The AI derives day-of-training from each session’s `datexp` and `mname`.

ii. 
```python
date = {k: datetime.date(*map(int, recordings[k]['datexp'].split('_'))) for k in keys}
```

```python
m = recordings[k]['mname']
first[m] = min(first.get(m, date[k]), date[k])
```

iii. The final trajectory summary says the date of each recording is “the only training-time info in the metadata,” which is why the AI based day-of-training on session dates.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the AI finds the earliest recording date and assigns every session a continuous value equal to calendar days elapsed since that first recording. That scalar is then broadcast across all timepoints of the trial.

ii. 
```python
for k in keys:
    recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)
```

```python
day = np.full(len(f), rec['day'])
```

iii. The final trajectory summary says “Day of training: days since that mouse’s first recording” and specifically notes that this makes naive and “before learning” sessions day 0.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The AI derives it from the per-trial start frame `StartFr`, the kept frame indices `f`, and the per-session frame duration `dt` estimated from `ft`.

ii. 
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
time_since_start = (f - rec['StartFr'][t]) * dt
```

iii. The final trajectory summary says the AI used “real elapsed seconds from corridor entry,” which is what `StartFr` is intended to represent.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. For each kept frame, the AI subtracts the trial’s fractional corridor-entry frame `StartFr[t]` from the frame index `f` and converts the difference to seconds with `dt`. Because only running frames are kept, the resulting series is sampled on a sparse subset of the original frame grid.

ii. 
```python
time_since_start = (f - rec['StartFr'][t]) * dt
```

iii. The final trajectory summary says the AI used “real elapsed seconds from corridor entry” and left those times unchanged even when some trials had long tails.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is computed on the same frame indices `f` used to index the neural data, so each timepoint of the input corresponds to one selected neural frame.

ii. 
```python
time_since_start = (f - rec['StartFr'][t]) * dt
...
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says the dataset uses native imaging frames as timepoints, so `time_since_trial_start` is aligned by construction to those neural frames.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The AI does not use `isRew` directly as the final reward-availability label. It derives reward availability from `WallName` together with `isRew`: it identifies the wall texture that ever received reward in the session and marks all trials with that wall as reward-available.

ii. 
```python
rewarded_walls = set(rec['WallName'][rec['isRew']])
assert len(rewarded_walls) <= 1, (k, rewarded_walls)
rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))
```

iii. The final trajectory summary explicitly justifies this change: `isRew` indicates actual reward delivery, which in “active after cue” sessions depends on licking, so using it directly would leak the licking output into the decoder input.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The AI computes the set of rewarded wall textures in a session, asserts there is at most one, creates a per-trial boolean `rew_of_trial`, and broadcasts that scalar across all timepoints of each trial.

ii. 
```python
rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))
...
rew = np.full(len(f), 1.0 if rew_of_trial[t] else 0.0)
```

iii. The final trajectory summary says this represents “the corridor in which water was available in that session,” notes that the rewarded corridor is always a single texture, and says unsupervised and naive sessions therefore get zeros.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The AI derives the stimulus label from `WallName`, but only after building a merged `wallmap` from `UniqWalls` and `stim_id` across experiment types. The final per-trial label is looked up as `wallmap.get(wall_name, -1)`.

ii. 
```python
for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
    if not np.isnan(cat):
        rec['wallmap'][str(wall)] = int(cat)
```

```python
wallmap = rec['wallmap']
stim_of_trial = np.array([wallmap.get(w, -1) for w in rec['WallName']])
```

iii. The final trajectory summary says the AI used the paper’s “canonical 7-category labelling (`beh['stim_id']`)” so that mice trained on different texture pairs are labeled by comparable roles.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI keeps the paper’s canonical `stim_id` category codes rather than collapsing textures into four broad classes. Trials whose wall texture never received a canonical label in the merged `wallmap` are dropped.

ii. 
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
```

```python
if stim_of_trial[t] < 0:
    n_dropped_stim += 1
    continue
...
out[0] = stim_of_trial[t]
```

iii. The final trajectory summary says this 7-category scheme makes stimuli comparable across mice and reports that 309 trials were dropped because the paper never assigns that third variant a canonical category.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. The AI derives licking from `LickFr`, the lick frame indices for the session.

ii. 
```python
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
licks[lick_fr] = True
```

iii. The final trajectory summary includes a licking sanity check and treats licking as a binary per-frame output on the imaging grid.

## 8-b. What processing is involved in computing `output` *Licking*?

i. The AI rounds each lick frame to the nearest integer imaging frame, clips to valid frame range, sets those frames to `True` in a session-wide boolean vector, and then slices that vector per trial.

ii. 
```python
licks = np.zeros(nframes, dtype=bool)
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
licks[lick_fr] = True
```

```python
out[1] = licks[f]
```

iii. No separate detailed justification was given beyond the final trajectory sanity check that licking behaved as expected in rewarded versus unrewarded corridors.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. Licking is first represented on the session’s imaging-frame grid, then indexed with the same trial-specific kept frames `f` used for the neural data.

ii. 
```python
out[1] = licks[f]
...
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says all time-varying streams use the native selected imaging frames, which is the alignment mechanism here as well.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. The AI derives position from `ft_Pos`.

ii. 
```python
pos = rec['ft_Pos']
...
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. The final trajectory summary says the 4 m textured corridor is divided into 4 x 1 m bins, which is the rationale for using position along the corridor.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The AI thresholds each kept position value using fixed edges at 10, 20, and 30, producing four 1 m categories across the 4 m corridor.

ii. 
```python
POSITION_EDGES = [10.0, 20.0, 30.0]
...
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. The final trajectory summary links the 4 x 1 m binning to the decision to keep only frames inside the 4 m textured corridor.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. It is thresholded with `np.digitize` against `[10.0, 20.0, 30.0]`, yielding bins for 0-1 m, 1-2 m, 2-3 m, and 3-4 m.

ii. 
```python
POSITION_EDGES = [10.0, 20.0, 30.0]
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

```python
['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. The final trajectory summary explicitly describes the position output as “four 1 m bins.”

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. Position is read on the session frame grid and then sliced at the same kept frame indices `f` as the neural data.

ii. 
```python
out[2] = np.digitize(pos[f], POSITION_EDGES)
...
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says the trial timepoints are the selected native imaging frames, which is the shared alignment for both position and neural data.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. The AI derives running-speed output from `ft_RunSpeed`.

ii. 
```python
speeds.append(rec['ft_RunSpeed'][frames])
...
speed = rec['ft_RunSpeed']
```

iii. The final trajectory summary says running speed is binned from all included timepoints, implying `ft_RunSpeed` is the raw source.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The AI first gathers all running-speed samples from the selected corridor-and-running frames across all sessions, computes global 25th/50th/75th percentiles, and then applies those same thresholds in every session.

ii. 
```python
speeds = []
for k in keys:
    rec = recordings[k]
    frames, _ = add_frame_selection(rec, len(rec['ft']))
    speeds.append(rec['ft_RunSpeed'][frames])
speeds = np.concatenate(speeds)
speed_edges = np.percentile(speeds, [25, 50, 75])
```

```python
out[3] = np.digitize(speed[f], speed_edges)
```

iii. The final trajectory summary explicitly says “Speed bins: quartile edges computed once over all included timepoints, so a bin means the same thing in every session.”

## 10-c. How is `output` *Running speed* thresholded into categories?

i. The AI uses `np.digitize` with the global percentile thresholds `speed_edges`, producing four bins named `speed_q1` through `speed_q4`.

ii. 
```python
speed_edges = np.percentile(speeds, [25, 50, 75])
...
out[3] = np.digitize(speed[f], speed_edges)
```

```python
['speed_q1', 'speed_q2', 'speed_q3', 'speed_q4']
```

iii. The final trajectory summary justifies these as globally consistent quartile bins across the entire dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Running speed is sampled at the same kept frame indices `f` used for the trial’s neural data.

ii. 
```python
out[3] = np.digitize(speed[f], speed_edges)
...
neural_s.append(spk[:, f])
```

iii. The final trajectory summary says timepoints are native selected imaging frames, which is the shared alignment for speed and neural data.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI truncates behavior arrays to the number of imaged frames with `n = min(nframes, len(rec['ft']))`, ignores frames with `NaN` trial indices, clips lick frames to valid bounds, skips trials with no kept frames, and drops trials whose stimulus cannot be labeled in its merged canonical mapping.

ii. 
```python
n = min(nframes, len(rec['ft']))
trind = rec['ft_trInd'][:n]
keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
```

```python
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
```

```python
if stim_of_trial[t] < 0:
    n_dropped_stim += 1
    continue
if len(f) == 0:
    continue
```

iii. The code comments written in the trajectory say the behavior arrays are sometimes one frame longer than imaging and should therefore be truncated the same way as the paper’s code. The final trajectory summary separately justifies dropping unlabeled-stimulus trials.

## 12-a. What are the most time-consuming steps of the code?

i. The likely bottleneck is loading large spike files and copying selected neurons plane by plane into a new dense array for every session. A secondary full-dataset pass is used to gather running-speed samples for global quartile edges.

ii. 
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
```

```python
out = np.empty((len(sel), nframes), dtype=np.float32)
for p, plane in enumerate(planes):
    m = (sel >= offsets[p]) & (sel < offsets[p + 1])
    if m.any():
        out[m] = plane[sel[m] - offsets[p]]
```

```python
for k in keys:
    rec = recordings[k]
    frames, _ = add_frame_selection(rec, len(rec['ft']))
    speeds.append(rec['ft_RunSpeed'][frames])
```

iii. The code comments written in the trajectory say the full neuron-by-frame matrix “can be 10 GB,” and the final trajectory summary says keeping all neurons would make the dataset hundreds of GB.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop repeatedly builds `f = frames[trials == t]`, which rescans the `trials` array once per trial. The per-plane neuron-copy loop could also be replaced with a precomputed gather or concatenation strategy, and the global speed pass is a separate full scan over behavior data.

ii. 
```python
for t in range(rec['ntrials']):
    ...
    f = frames[trials == t]
```

```python
for p, plane in enumerate(planes):
    m = (sel >= offsets[p]) & (sel < offsets[p + 1])
    if m.any():
        out[m] = plane[sel[m] - offsets[p]]
```

iii. No explicit trajectory justification was given for keeping these loops unvectorized.

## 12-c. What processing does the code repeat multiple times?

i. The AI repeats whole-dataset behavior scanning: once in `load_behaviour()` to assemble recordings and merge stimulus labels, then again across all recordings to compute global speed quartiles. Within each session it also repeatedly filters `frames[trials == t]` inside the trial loop.

ii. 
```python
recordings = load_behaviour()
```

```python
speeds = []
for k in keys:
    rec = recordings[k]
    frames, _ = add_frame_selection(rec, len(rec['ft']))
    speeds.append(rec['ft_RunSpeed'][frames])
```

```python
for t in range(rec['ntrials']):
    f = frames[trials == t]
```

iii. No explicit trajectory justification was given for this repeated processing, beyond the stated desire to define one global running-speed binning scheme.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI does extra work to merge experiment-type-specific `stim_id` labels, compute and store extensive metadata, and carry fields such as `texture_length`, `exp_types`, `cohort`, and `reward_mode` even though the downstream decoder only consumes `neural`, `input`, `output`, and indexing metadata. It also computes the dataset-average `time_bin_size` from `dts` only for metadata.

ii. 
```python
rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
rec['exp_types'].append(exp_type)
for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
    if not np.isnan(cat):
        rec['wallmap'][str(wall)] = int(cat)
```

```python
rec.update(dict(
    ...
    texture_length=float(beh['Texture_Length']),
    reward_mode=str(beh['Reward_Mode']),
))
```

```python
dts.append(dt)
...
'time_bin_size': float(np.mean(dts)) * 1000.0,
```

iii. No explicit trajectory justification was given for these extra metadata-oriented computations, aside from the agent’s general aim to preserve paper-specific provenance and canonical stimulus labels.
