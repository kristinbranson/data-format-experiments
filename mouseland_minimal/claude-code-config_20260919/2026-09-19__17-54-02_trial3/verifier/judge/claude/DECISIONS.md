# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Everything comes from the three folders under `/app/data`: `beh` (behaviour), `spk` (deconvolved
traces) and `retinotopy` (visual area of each neuron). `beh/Imaging_Exp_info.npy` is the master
index; it is a dict of 23 experiment types, each a list of entries. For every experiment type the
matching `Beh_<exp_type>.npy` is read **once**, and every entry in the index is turned into a
recording key `mname_datexp_blk` (with the behaviour keyed by `key + '_' + stimtype` for swap
sessions). Because a recording can be listed under up to five experiment types, the behaviour
arrays are copied into a compact per-recording dict only the *first* time the recording is seen,
while the wall-name → `stim_id` map is merged over *all* experiment types the recording appears in.
The spike file and the retinotopy file are then read per session inside the main loop
(`load_spikes` / `select_neurons`). Result: 89 recordings, 19 mice.

ii.
```python
exp_info = np.load(os.path.join(ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=1).item()
recordings = collections.OrderedDict()
for exp_type, db in exp_info.items():
    Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
    for d in db:
        key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
        behkey = key + ('_%s' % d['stimtype'] if 'stimtype' in d else '')
        beh = Beh[behkey]
        rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
        rec['exp_types'].append(exp_type)
        for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
            if not np.isnan(cat):
                rec['wallmap'][str(wall)] = int(cat)
        if 'ft' in rec:
            continue
        rec.update(dict(mname=d['mname'], datexp=d['datexp'], blk=d['blk'], ...))
```
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
iarea = np.load(os.path.join(ROOT, 'retinotopy',
                '%s_%s_trans.npz' % (rec['mname'], rec['datexp'])), allow_pickle=True)['iarea']
```

iii. The agent verified (step 22/24/30 of the trajectory) that the 23 experiment types contain 89
unique `(mname, datexp, blk)` triples matching the 89 files in `spk/`, and that when a recording
appears in several experiment types the behaviour arrays are byte-identical, only `stim_id` differs
("it is the stimulus subset used by that analysis"). Hence "I read behaviour once per recording and
merged `stim_id` across types to fully label every trial." It also checked that every needed
retinotopy file exists (step 46) and that `len(iarea)` equals the neuron count of the spike file
(step 48, plus a run-time `assert`).

## 1-b. How are the data split into subjects?

i. By `mname` from the index entry, carried on every recording record. The session keys are sorted
by `(mname, datexp, blk)` and the subject list is built in order of first appearance, so it is
effectively alphabetical; `subject_idx` is the index of the session's mouse into that list. 19
subjects, 89 sessions, with the same sessions-per-subject counts as the expert solution.

ii.
```python
keys = sorted(recordings.keys(),
              key=lambda k: (recordings[k]['mname'], recordings[k]['datexp'], recordings[k]['blk']))
...
if rec['mname'] not in subjects:
    subjects.append(rec['mname'])
subject_idx.append(subjects.index(rec['mname']))
```

iii. No justification is needed beyond "the index already names the mouse"; the agent printed
`'%d recordings, %d mice'` as a sanity check and reported "89 sessions (all 89 recordings, 19 mice)".

## 1-c. How are the data split into sessions?

i. One session = one recording = one `(mname, datexp, blk)` triple, which is also the name of the
spike file. Duplicates across experiment types are collapsed by `OrderedDict.setdefault`, so each
recording appears exactly once; all 89 are kept (no session is dropped).

ii.
```python
key = '%s_%s_%s' % (d['mname'], d['datexp'], d['blk'])
rec = recordings.setdefault(key, {'exp_types': [], 'wallmap': {}})
rec['exp_types'].append(exp_type)
if 'ft' in rec:
    continue          # behaviour arrays already stored for this recording
```

iii. From the final summary: "**Session = one recording** (`mname_datexp_blk`). A recording appears
in up to 5 experiment types; the behaviour arrays are identical in each copy, only `stim_id`
differs." The agent explicitly counted the multiplicities (step 24) before deciding to deduplicate.

## 1-d. How are the data split into trials?

i. A trial is one corridor traversal, exactly as the data declares it: the behaviour labels every
imaging frame with its trial in `ft_trInd`, and there are `ntrials` of them. The frames that
constitute a trial are the frames of that trial which are (a) inside the textured part of the
corridor (`ft_CorrSpc`, i.e. 0–4 m, excluding the 2 m grey gap) **and** (b) acquired while the
virtual reality was moving (`ft_move > 0`, "during running"). Frames with `NaN` trial index are
excluded. Trials keep their native, variable length (11–178 bins, median 21).

The running filter is the one substantive difference from the expert solution: it removes ~33% of
the in-corridor frames (815,506 kept vs ≈1.23 M for the expert), and because idle frames are
excised from the *middle* of traversals, 46% of trials contain at least one gap larger than one
frame and 24% contain a gap larger than 10 frames, i.e. successive columns of a trial are not always
adjacent in time.

ii.
```python
def add_frame_selection(rec, nframes):
    n = min(nframes, len(rec['ft']))
    trind = rec['ft_trInd'][:n]
    keep = (rec['ft_CorrSpc'][:n] & (rec['ft_move'][:n] > 0) & ~np.isnan(trind))
    frames = np.nonzero(keep)[0]
    return frames, trind[frames].astype(int)
```
```python
frames, trials = add_frame_selection(rec, nframes)
for t in range(rec['ntrials']):
    ...
    f = frames[trials == t]
    if len(f) == 0:
        continue
```

iii. The agent quotes the Methods directly — "We only considered timepoints during running for
analysis, which removed time periods when the task mice stopped to collect water rewards" — and
notes the paper's own code does the same everywhere (`utils.py`: `corr_fr = beh['ft_CorrSpc'][:nfr]
& VRmove`, with `VRmove = beh['ft_move'][:nfr] > 0`). It also gives a data-driven argument: "it is
also what makes traversals comparable (11–30 frames, median 21 ≈ 4 m at 60 cm/s) instead of
11–600". Step 42 shows the measurement it relied on: for one session, frames per trial in the
corridor are min/med/max 20/34/675 without the filter and 20/24/33 with it. `ft_CorrSpc` was checked
to span exactly 0–39.99 position units while the grey space spans 40–60, which is what makes four
1 m position bins tile the trial.

## 1-e. How are trials filtered based on quality controls?

i. Two filters, neither of which is a length/stall control:
  * a trial whose stimulus has no canonical `stim_id` in the merged wall map is dropped (309 trials,
    0.8% — the `circle3` variant in four naive/unsup sessions);
  * a trial left with no frames after the frame selection is skipped.
No trial is dropped for being too long. The running filter shortens stalled traversals instead of
removing them, so the pathological trials survive as short arrays with a long real-time span: the
worst trial still spans 5,607 imaging frames (≈29 min of standing still) but is stored as ~100 bins,
and 1.2% of trials span more than 200 frames. As a result `time_since_trial_start` reaches 1,763 s
and `time_to_sound_cue` spans [−1762, +723] s, against a median traversal of ~7 s.

ii.
```python
for t in range(rec['ntrials']):
    if stim_of_trial[t] < 0:
        # A few naive/unsupervised sessions also showed a third variant
        # ("circle3") that the paper never assigns a category to; those trials
        # cannot be labelled on the common scale and are dropped.
        n_dropped_stim += 1
        continue
    f = frames[trials == t]
    if len(f) == 0:
        continue
```

iii. For the dropped stimuli: "309 trials (0.8%) showing a third variant the paper never labels
('circle3') were dropped" — they cannot be placed on the canonical `stim_id` scale the agent chose
for the stimulus output. For the long tail the agent did not filter but measured: step 82 quantifies
`frac tss>60 = 1.0%`, `frac |cue|>60 = 0.5%`, and step 84–87 ran the real decoder on a 10-session
subset with and without clipping the two time inputs at 60 s: 0.835/0.854/0.858/0.644 unclipped vs
0.845/0.827/0.858/0.648 clipped. Conclusion in the final message: "I checked that clipping them
changes accuracy by <0.02, so they are left unmodified."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session>_neural_data.npy`, a list of one (neurons × frames) array per
imaging plane, which `load_spikes` treats as a single neuron axis in plane order. The area of each
neuron comes from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`, which indexes the neurons in
that same order.

ii.
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
sizes = [p.shape[0] for p in planes]
nneurons, nframes = int(np.sum(sizes)), planes[0].shape[1]
sel, area = neuron_getter(nneurons, nframes)
offsets = np.concatenate([[0], np.cumsum(sizes)])
out = np.empty((len(sel), nframes), dtype=np.float32)
for p, plane in enumerate(planes):
    m = (sel >= offsets[p]) & (sel < offsets[p + 1])
    if m.any():
        out[m] = plane[sel[m] - offsets[p]]
```

iii. "utils.load_spk concatenates the per-plane blocks in order, and the retinotopy `iarea` follows
that same order, so a global neuron index maps to (plane, row) by cumulative plane size." The agent
confirmed the correspondence empirically (step 48) and guards it with
`assert len(iarea) == nneurons`.

## 2-b. How is the `neural` data processed?

i. Not processed at all. The selected rows of the deconvolved traces are sliced by the trial's frame
indices and stored as `float32`; no normalisation, smoothing, dF/F or deconvolution is applied, and
trials are left at their native variable length (no padding).

ii.
```python
neural_s.append(spk[:, f])
```
with `spk` produced as `np.empty((len(sel), nframes), dtype=np.float32)` in `load_spikes`.

iii. Metadata records "deconvolved two-photon calcium fluorescence (Suite2p), sampled at ~3.18 Hz,
not normalised". This follows the Methods: "All our analyses were based on deconvolved fluorescence
traces."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two steps. (1) Area filter, identical to the expert's: a neuron is kept only if its `iarea` falls
in V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4); codes −1 and 7 are dropped. (2) **A uniform random
subsample to at most 2,000 neurons per session** (fixed seed). No other curation is applied (the
authors already ran the Suite2p cell classifier). The subsample is the big deviation: every session
ends up with exactly 2,000 neurons, 178,000 in total, against ~4.1 M (mean 46,128/session) in the
expert conversion — 4.3% of the neurons. Area proportions are preserved (V1 44% in both).

ii.
```python
AREA_NAMES = ['V1', 'mHV', 'lHV', 'aHV']
AREA_CODES = {'V1': (8,), 'mHV': (0, 1, 2, 9), 'lHV': (5, 6), 'aHV': (3, 4)}
NEURONS_PER_SESSION = 2000

def select_neurons(rec, nneurons, rng):
    iarea = np.load(fn, allow_pickle=True)['iarea']
    assert len(iarea) == nneurons, (fn, len(iarea), nneurons)
    area_idx = np.full(len(iarea), -1, dtype=np.int64)
    for a, name in enumerate(AREA_NAMES):
        area_idx[np.isin(iarea, AREA_CODES[name])] = a
    valid = np.nonzero(area_idx >= 0)[0]
    if len(valid) > NEURONS_PER_SESSION:
        valid = np.sort(rng.choice(valid, NEURONS_PER_SESSION, replace=False))
    return valid, area_idx[valid]
```

iii. The area codes are taken verbatim from `utils.neu_area_ID`. For the subsample the agent gives
two reasons: "Recordings hold 20k–90k neurons — keeping all would be hundreds of GB, and 2000 is the
reference decoder's `svd_max_neurons`, above which it stops computing an exact SVD initialisation";
the code comment adds "Neurons are drawn uniformly at random (fixed seed), which keeps the relative
representation of the four visual areas unbiased." (The second reason is only partly right:
`svd_max_neurons` caps the *initialisation* of the projection — `decoder.py:958-963` falls back to a
random Gaussian projection — while the trained `nn.Linear(nneurons, npcs)` still sees every neuron.)

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. To corridor entry (trial start). A trial's columns are the frames the mouse spent running inside
that corridor, in increasing frame order, starting at the first such frame after `StartFr`. Trials
are variable length and are stored that way; nothing is cut or padded, so `off_start = 0.0` and
`off_end = None`.

ii.
```python
f = frames[trials == t]
...
neural_s.append(spk[:, f])
```
```python
'temporal_alignment_event': 'entry into the virtual-reality corridor (start of the trial)',
'off_start': 0.0,
'off_end': None,
```

iii. "**Trial = one corridor traversal**, aligned to corridor entry." The agent noted in the metadata
that "Trials therefore start at the alignment event (off_start=0) but have variable length … so
off_end is N/A", and checked with `--plot-samples` (step 110/112) that position rises monotonically
from bin 0 within each trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: one bin = one native two-photon frame, ~315 ms (3.18 Hz). The bin
size written to the metadata is the mean over sessions of the per-session median inter-frame
interval, 314.85 ms (the expert used the nominal 1000/3.17 = 315.46 ms). All behavioural streams are
already sampled on the same frame grid, so nothing has to be resampled. Because non-running frames
are removed, successive bins within a trial are not always contiguous in absolute time, although
every bin is the same 315 ms wide.

ii.
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)  # datenum -> s
dts.append(dt)
...
'time_bin_size': float(np.mean(dts)) * 1000.0,
```

iii. The frame is the finest resolution available and the paper analyses data at frame resolution;
the agent measured the interval from `ft` (step 42: `dt sec 0.3147`) rather than assuming it, and
records the per-session rate in `session_info['frame_rate_hz']`.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame index at which the sound cue was played on each
trial) and the frame grid itself; the frame interval `dt` is measured from the timestamps `ft`.

ii.
```python
dt = float(np.median(np.diff(rec['ft'][:nframes])) * 86400.0)
...
time_to_cue = (rec['SoundFr'][t] - f) * dt
```

iii. `SoundFr` is the cue expressed in the same frame units as the neural data, so no cross-stream
interpolation is needed; `ft` is a MATLAB datenum, hence the ×86400 conversion to seconds.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each kept frame of the trial, `(SoundFr[trial] − frame_index) × dt`, in seconds: positive
before the cue, negative after it — the same sign convention as the expert. It is a continuous,
time-varying input, stored as `float32`. No clipping is applied, so the tail runs to
[−1762, +723] s on stalled trials (see 1-e).

ii.
```python
time_to_cue = (rec['SoundFr'][t] - f) * dt
input_s.append(np.stack([time_to_cue, day, time_since_start, rew]).astype(np.float32))
```
```python
'time_to_sound_cue': 'seconds until the sound cue (positive before the cue, negative after it)',
```

iii. "**Times**: real elapsed seconds from corridor entry, and signed seconds to the sound cue
(positive before)." The long tail was measured and tested rather than removed (see 1-e iii).

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the frame indices `f` that index the neural columns of that trial, so
it has the same length and the same alignment by construction.

ii.
```python
f = frames[trials == t]
time_to_cue = (rec['SoundFr'][t] - f) * dt
...
neural_s.append(spk[:, f])
```

iii. Every stream in this dataset lives on the imaging-frame grid, so using the same frame index
vector for neural, inputs and outputs is sufficient for alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the recording date in the index entry (and therefore in the session key), parsed
into a calendar date.

ii.
```python
date = {k: datetime.date(*map(int, recordings[k]['datexp'].split('_'))) for k in keys}
```

iii. "the date of each recording is the only training-time information in the metadata" — the agent
checked the index entries for any explicit training-day field (step 32/34, looking for `days`,
`artLick`, `Note`) and found none.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The number of **calendar days elapsed since that mouse's first recording**: 0 for a mouse's first
session and up to 92 for the last one. It is a per-trial scalar broadcast across all bins of the
trial. (The expert instead used the ordinal index of the session within the mouse, 0–7.)

ii.
```python
first = {}
for k in keys:
    m = recordings[k]['mname']
    first[m] = min(first.get(m, date[k]), date[k])
for k in keys:
    recordings[k]['day'] = float((date[k] - first[recordings[k]['mname']]).days)
...
day = np.full(len(f), rec['day'])
```

iii. "We express it as days elapsed since that mouse's first recording (= its first session in the
virtual-reality corridors), which is 0 for every 'naive'/'before learning' session and grows through
training." The agent printed the resulting per-mouse timelines (step 58) and confirmed that day 0 is
always the `*_before_learning` / `naive_test1` session and that the ordering matches the experimental
progression.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `StartFr`, the fractional imaging-frame index of corridor entry for each trial, and the
frame interval `dt` measured from `ft`.

ii.
```python
time_since_start = (f - rec['StartFr'][t]) * dt
```

iii. Same rationale as 3-a: `StartFr` is already expressed in imaging-frame units.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(frame_index − StartFr[trial]) × dt` in seconds, so it starts just above 0 at the first kept
frame and increases; positive after entry, negative before (never happens here since the first kept
frame follows entry). Continuous, time-varying, `float32`, unclipped.

ii.
```python
time_since_start = (f - rec['StartFr'][t]) * dt
input_s.append(np.stack([time_to_cue, day, time_since_start, rew]).astype(np.float32))
```

iii. "real elapsed seconds from corridor entry". Note that because idle frames are dropped this is
*true* elapsed time rather than bin index × dt, which the agent regarded as the correct behaviour
("Because idle frames are dropped these have a rare long tail (~0.5–1% beyond 60 s); I checked that
clipping them changes accuracy by <0.02, so they are left unmodified").

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: computed on the trial's frame index vector `f`, which is also the column index of
the neural matrix.

ii.
```python
f = frames[trials == t]
time_since_start = (f - rec['StartFr'][t]) * dt
neural_s.append(spk[:, f])
```

iii. All streams share the imaging-frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew` **and** `WallName`: the set of wall textures on which a reward was ever delivered in
that session is taken from `isRew`, and reward availability is then set for *every* trial run in
that corridor — not for the trials on which water was actually obtained. It is 0 for all unsupervised
and naive sessions.

ii.
```python
rewarded_walls = set(rec['WallName'][rec['isRew']])
assert len(rewarded_walls) <= 1, (k, rewarded_walls)
rew_of_trial = np.isin(rec['WallName'], list(rewarded_walls))
```

iii. From the code comment / final message: "`beh['isRew']` marks the trials on which water was
actually delivered, which in the 'active after cue' sessions requires the mouse to have licked —
using it directly would leak the licking output into the decoder's input." The agent verified this
on `TX108_2023_03_25_1` (step 97): in the rewarded corridor `isRew` is true on only 109/143 trials,
`RewTime` is non-NaN for exactly the `isRew` trials, and every `isRew` trial contains a lick, while
only 32% of the non-`isRew` trials of the *same* corridor do. It then checked all 89 sessions (step
99) that the rewarded corridor is a single wall texture, that it always carries canonical stimulus
category 2, and that no unsupervised/naive session has any reward.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. A per-trial 0/1 flag broadcast across the trial's bins as a `float32` row of the input matrix.

ii.
```python
rew = np.full(len(f), 1.0 if rew_of_trial[t] else 0.0)
input_s.append(np.stack([time_to_cue, day, time_since_start, rew]).astype(np.float32))
```

iii. The task specifies "1 if in rewarded corridor, 0 if not", i.e. a property of the corridor, which
is what `rew_of_trial` encodes. The agent cross-checked the result inside the converted file (step
95): in every task session, `reward_availability == 1` exactly on the trials whose stimulus category
is 2.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName` (the texture of each trial's corridor) mapped through `UniqWalls` → `stim_id`,
the paper's canonical stimulus code, merged across every experiment type in which the recording
appears.

ii.
```python
for wall, cat in zip(beh['UniqWalls'], np.asarray(beh['stim_id'], float)):
    if not np.isnan(cat):
        rec['wallmap'][str(wall)] = int(cat)
...
wallmap = rec['wallmap']
stim_of_trial = np.array([wallmap.get(w, -1) for w in rec['WallName']])
```

iii. "the paper's canonical 7-category labelling (`beh['stim_id']`), so a mouse trained on rock/brick
is labelled by the matching role and categories are comparable across mice". The agent chose
`WallName` over `TrialStim` implicitly — inspection at step 30 shows `TrialStim` is masked to
`'stimulus_of_trial'` for the stimuli an experiment type does not use, and that `stim_id` is NaN for
the same ones, which is why it merges the map over all experiment types (step 56 shows the merge
leaves only `circle3` unmapped, in 4 sessions).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial gets the integer `stim_id` of its wall (0–6), broadcast across the trial's bins as an
`int64` row, with `output_values` named
`['circle1','circle2','leaf1','leaf2','leaf3','leaf1_swap1','leaf1_swap2']`. Trials whose wall has no
`stim_id` (`circle3`, 309 trials) are dropped. This yields **7** categories with frequencies
0.32/0.06/0.34/0.17/0.06/0.03/0.03, against the expert's **4** physical texture families
(circle/leaf/rock/wood).

`stim_id` is a *role* code, not a texture identity: verification over all 89 sessions shows
`circle1 → 0` in 62 sessions but `→ 2` in 6 (the mice for which circle was the rewarded texture),
`rock1 → 0` in 19 sessions, `→ 2` in 2 and `→ 4` in 1, `wood1 → 2` in 19, `wood5 → 4` in 7. Code 2 is
always the trained/rewarded texture. The chosen value names are therefore only literally accurate
for the majority circle/leaf mice.

ii.
```python
STIM_NAMES = ['circle1', 'circle2', 'leaf1', 'leaf2', 'leaf3',
              'leaf1_swap1', 'leaf1_swap2']
...
out = np.empty((4, len(f)), dtype=np.int64)
out[0] = stim_of_trial[t]
```
```python
'stimulus_category': 'wall texture of the corridor, mapped onto the canonical '
                     'categories used in the paper (beh["stim_id"]); mice '
                     'trained on rock/brick are labelled by the matching role',
```

iii. The stated aim is cross-mouse comparability: "the paper maps each mouse's stimuli onto this
common set of roles so that stimuli are comparable across mice, and we decode that common set"
(code comment), and the caveat about rock/brick mice is recorded in the metadata.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame index of every detected lick in the session.

ii.
```python
licks = np.zeros(nframes, dtype=bool)
lick_fr = np.round(rec['LickFr']).astype(int)
lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nframes)]
licks[lick_fr] = True
```

iii. `LickFr` is already expressed on the imaging-frame grid, so it can be turned directly into a
per-frame flag. The agent checked (step 56) that licks occur only in the 28 task sessions and are
zero in all 46 unsupervised and 15 naive sessions.

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary per-frame flag: 1 if at least one lick falls in that frame, 0 otherwise. The fractional
lick frame is **rounded** to the nearest frame (the expert truncated), and licks outside the imaged
range are discarded. Per-frame flags are then sliced by the trial's frames. 3.5% of bins are licks.

ii. See 8-a, plus
```python
out[1] = licks[f]
...
['no_lick', 'lick'],
'licking': '1 if at least one lick was detected in this imaging frame',
```

iii. Rounding assigns each lick to the frame whose timestamp is nearest the lick time. The agent
sanity-checked the resulting variable against the paper's behavioural result (final message):
"licking occurs only in task mice and is 21% of frames in the rewarded corridor vs 4.4% elsewhere
(anticipatory licking, as in the paper)".

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The flag lives on the full session frame grid and is indexed with the same `f` used for the
neural columns, so it has the trial's length and alignment.

ii.
```python
out[1] = licks[f]
neural_s.append(spk[:, f])
```

iii. Common frame grid for all streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the virtual-reality position at each imaging frame, in decimetres (0–40 across the
textured corridor, 40–60 through the grey space).

ii.
```python
pos = rec['ft_Pos']
...
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. Step 42 verified the units and coverage directly: `pos range corr 0.0 39.99`, `pos range gray
40.0 59.99`, which is what justifies "the corridor is 40 units (=4 m) long -> 1 m bins".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. None beyond the binning — the raw `ft_Pos` value of each kept frame is discretised. Stored as an
`int64` row of the output matrix; resulting bin occupancies are 0.250/0.249/0.250/0.252.

ii.
```python
POSITION_EDGES = [10.0, 20.0, 30.0]      # corridor is 40 units (=4 m) long -> 1 m bins
out[2] = np.digitize(pos[f], POSITION_EDGES)
```

iii. As in 9-a; the trial's frames are all inside `ft_CorrSpc`, so positions never exceed 40 and the
four bins tile the trial.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1 m bins with `np.digitize` at edges 10, 20, 30 decimetres → categories
`['0-1m','1-2m','2-3m','3-4m']`, exactly the discretisation requested by the task and used by the
expert.

ii.
```python
POSITION_EDGES = [10.0, 20.0, 30.0]
out[2] = np.digitize(pos[f], POSITION_EDGES)
...
['0-1m', '1-2m', '2-3m', '3-4m'],
```

iii. Directly prescribed by the instructions ("4 equal-length, 1-m-long spatial bins") and made
possible by restricting trials to the textured 4 m.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame; it is indexed with the same `f` as the neural columns.

ii.
```python
out[2] = np.digitize(pos[f], POSITION_EDGES)
neural_s.append(spk[:, f])
```

iii. Common frame grid; the agent additionally eyeballed sample trials (`--plot-samples`, steps
110–112) and reported "position is monotonic within trials".

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the ball-tracking running speed at each imaging frame.

ii.
```python
speed = rec['ft_RunSpeed']
...
out[3] = np.digitize(speed[f], speed_edges)
```

iii. Directly available per frame; no derivation needed.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw speed is discretised into quartile bins. The three quartile edges are computed **once,
globally**, over the running frames of *all* sessions (12.42 / 25.35 / 40.85 a.u., from 821,579
frames), so a speed bin means the same thing in every session. The expert instead ranked the frames
*within each session* and split the ranks into four equal groups. The achieved global occupancies
are 0.2505 / 0.2508 / 0.2501 / 0.2486.

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

iii. "Running speed is discretised into 4 bins each holding 25% of the data. The bin edges are
computed once over all timepoints that enter the dataset, so a speed bin means the same thing in
every session." The running filter removes the mass of exactly-zero speeds that would otherwise make
value-based edges impossible (step 42: 25th percentile of in-corridor speed is 0.0 without the
filter, 8.6 with it).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global quartile edges → `['speed_q1','speed_q2','speed_q3',
'speed_q4']`, i.e. four bins each holding ~25% of the data as the task requires. The edges are also
written into the metadata.

ii.
```python
N_SPEED_BINS = 4
speed_edges = np.percentile(speeds, [25, 50, 75])
out[3] = np.digitize(speed[f], speed_edges)
...
'speed_bin_edges': speed_edges.tolist(),
```

iii. See 10-b; the requirement "4 bins, each corresponding to 25% of the data" is met globally
(0.2486–0.2508 per bin).

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is indexed with the same `f` as the neural
columns.

ii.
```python
out[3] = np.digitize(speed[f], speed_edges)
neural_s.append(spk[:, f])
```

iii. Common frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several small guards, all silent:
  * the behaviour can be longer than the imaging, so every stream is truncated to
    `n = min(nframes, len(rec['ft']))`, with `nframes` taken from the spike file;
  * frames with `NaN` `ft_trInd` (outside any trial) are excluded by `~np.isnan(trind)`;
  * licks whose frame index is negative or beyond the imaged range are discarded;
  * trials with no surviving frames are skipped, as are trials whose wall has no `stim_id`;
  * `assert len(iarea) == nneurons` guards the neuron/retinotopy correspondence, and
    `assert len(rewarded_walls) <= 1` guards the reward assumption.
No NaN/Inf reaches the output: the format verifier reported "Data format is valid, no errors or
warnings."

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

iii. The truncation is explicitly copied from the paper's code: "`nframes` is the number of imaging
frames actually present in the spike file; the behaviour arrays are sometimes one frame longer (the
paper's code truncates the same way, e.g. `utils.Get_dprime_selective_neuron`)." Step 42 measured
that 0.17% of frames have a NaN trial index and 24 in-corridor frames have none.

## 12-a. What are the most time-consuming steps of the code?

i. Disk I/O dominates: (1) reading the 89 spike files, 405 GB in total, in `load_spikes` — each
`np.load(...).item()` unpickles every imaging plane of a session; (2) reading the 23 behaviour files,
6.6 GB, in `load_behaviour`. Everything else (the per-plane row gather, the per-trial slicing, the
binning) is negligible by comparison. The whole conversion ran in ≈7 minutes on this machine, only
because 933 GB of page cache was available.

ii.
```python
planes = np.load(os.path.join(ROOT, 'spk', '%s_neural_data.npy' % key),
                 allow_pickle=True).item()['spks']
```
```python
Beh = np.load(os.path.join(ROOT, 'beh', 'Beh_%s.npy' % exp_type), allow_pickle=1).item()
```

iii. Not discussed explicitly, but the code is organised around this cost: each behaviour file is
read exactly once for all the recordings it holds, and the neuron selection is applied plane by
plane so that "Selecting inside the loop avoids ever materialising the full (nneurons x nframes)
matrix, which can be 10 GB" — that avoids the concatenated copy the paper's `utils.load_spk` makes,
although the planes themselves are still fully unpickled.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial frame lookup `frames[trials == t]` rescans the whole kept-frame index once per
trial, i.e. O(ntrials × nframes) per session (up to ~700 trials × ~25k frames); one
`np.split(frames, np.searchsorted(trials, np.arange(1, ntrials)))` would give all trials in a single
pass. Similarly `np.digitize` for position and speed is called once per trial although it could be
computed once per session and then sliced (as the expert does), and `stim_of_trial` is built with a
Python dict lookup per trial. All are negligible next to the I/O.

ii.
```python
for t in range(rec['ntrials']):
    ...
    f = frames[trials == t]
    ...
    out[2] = np.digitize(pos[f], POSITION_EDGES)
    out[3] = np.digitize(speed[f], speed_edges)
```

iii. Not discussed by the agent.

## 12-c. What processing does the code repeat multiple times?

i. `add_frame_selection` is executed twice for every recording — once in the speed-quartile pre-pass
and once in the main loop (with a slightly different frame limit: `len(rec['ft'])` vs the spike
file's `nframes`). The wall-name → `stim_id` merge loop is re-run for each experiment type a
recording appears in (up to five times). The `np.digitize` calls are repeated per trial rather than
per session. The expert's code also makes a global pre-pass over the behaviour, so the pattern is
comparable.

ii.
```python
for k in keys:                                    # pre-pass
    frames, _ = add_frame_selection(rec, len(rec['ft']))
...
for k in keys:                                    # main loop
    frames, trials = add_frame_selection(rec, nframes)
```

iii. The pre-pass is deliberate: the global quartile edges have to be known before any trial is
written, which is exactly why the frame selection has to be done twice.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Mostly storage-format waste rather than computation:
  * the neural arrays are stored as `float32` where `float16` would have been enough for deconvolved
    traces (the expert used `float16`) — that doubles the 6.5 GB file;
  * the output arrays are `int64` for values in 0–6, 8× larger than the `int8` the expert used;
  * `texture_length` is copied out of every behaviour file and never used;
  * a number of metadata fields (`cohort`, `experiment_types`, `reward_mode`, `frame_rate_hz`,
    `ntimepoints`) are computed per session and ignored by the decoder, though they are legitimate
    documentation;
  * the speed pre-pass computes the frame selection over the *behaviour-length* arrays and over
    trials that are later dropped for having no stimulus label, so ~6,000 of the 821,579 frames used
    to set the quartile edges are not in the final dataset.
Also note the metadata string "variable length (11-30 frames, median 21)" is stale — the actual
range in the delivered file is 11–178 bins.

ii.
```python
out = np.empty((len(sel), nframes), dtype=np.float32)   # float16 would suffice
out = np.empty((4, len(f)), dtype=np.int64)             # int8 would suffice
texture_length=float(beh['Texture_Length']),            # never used
```

iii. Not discussed by the agent; the file size it was optimising for was addressed by subsampling
neurons instead of by narrowing dtypes.
