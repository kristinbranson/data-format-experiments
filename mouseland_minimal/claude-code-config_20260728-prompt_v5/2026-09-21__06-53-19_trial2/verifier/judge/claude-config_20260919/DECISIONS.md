# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All data are read from the three subfolders of `data/`: `beh/` (behaviour), `spk/` (deconvolved
traces) and `retinotopy/` (visual area per neuron). `beh/Imaging_Exp_info.npy` is used as the master
index; the AI loops over its 23 experiment types, loads each `Beh_<exp_type>.npy` once, and builds a
dict `sessions` keyed by `mname_datexp_blk` holding `{ndb, beh, exp_type}` for every unique recording
(first occurrence wins). The behaviour key gets a `_<stimtype>` suffix when the index entry has one.
Spikes (`spk/<session>_neural_data.npy`, `spks` concatenated over imaging planes) and retinotopy
(`retinotopy/<mouse>_<date>_trans.npz`) are read per session inside `process_session`. All 89 unique
sessions / 19 mice are converted; nothing is subsetted by experiment type. Note the behaviour
sub-dicts of every session are kept alive in RAM for the whole run, and the spike files are read
*twice* (once in `compute_speed_quartiles` via `get_nfr`, once in `process_session`).

ii.
```python
def collect_sessions():
    """Collect all unique recording sessions across all experiment types."""
    exp_info = np.load(os.path.join(BEH_ROOT, 'Imaging_Exp_info.npy'), allow_pickle=True).item()

    sessions = {}
    for exp_type in exp_info:
        beh_file = os.path.join(BEH_ROOT, f'Beh_{exp_type}.npy')
        Beh = np.load(beh_file, allow_pickle=True).item()

        for ndb in exp_info[exp_type]:
            session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if session_key in sessions:
                continue

            if 'stimtype' in ndb:
                beh_key = f"{session_key}_{ndb['stimtype']}"
            else:
                beh_key = session_key

            if beh_key in Beh:
                sessions[session_key] = {'ndb': ndb, 'beh': Beh[beh_key], 'exp_type': exp_type}
        del Beh
```
```python
def load_spk(ndb):
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
    del spk_data
    return spk
```
```python
    fn = f"{ndb['mname']}_{ndb['datexp']}_trans.npz"
    ret = np.load(os.path.join(RET_ROOT, fn))
    brain_reg_idx = map_brain_regions(ret['iarea'])
```

iii. From the trajectory (step 27/31/35): "Include ALL 89 unique sessions across all experiment
types. This gives us the maximum data. The decoder inputs include 'reward availability' which will be
0 for unsupervised/naive mice and vary for supervised mice." The AI verified in step 30 that 33
recordings appear under several experiment types and concluded "for the decoder I just need all
trials from each recording, using WallName labels... so I should find, for each unique session, the
first experiment type containing it and pull from that file". It also checked (step 33) that the
behaviour of a `stimtype`-split session has the same `ntrials`/`UniqWalls`, so either copy works, and
verified (step 40) that the spike neuron count matches the retinotopy `iarea` length for all sessions.

## 1-b. How are the data split into subjects?

i. The subject is `ndb['mname']` carried through the session record. Sessions are processed in sorted
`session_key` order and a mouse is appended to `subjects` the first time it is seen, so `subjects`
ends up alphabetically ordered (19 mice); `subject_idx` is the index of each session's mouse into that
list. No further splitting is derived — the index file already names the mouse.

ii.
```python
        mname = info['ndb']['mname']
        ...
        if mname not in subject_to_idx:
            subject_to_idx[mname] = len(subjects_list)
            subjects_list.append(mname)
        session_subject_idx.append(subject_to_idx[mname])
```
```python
        'subjects': subjects_list,
        'subject_idx': np.array(session_subject_idx, dtype=np.int64),
```

iii. Not discussed at length; the AI treated `mname` as the given subject identity (step 25 verified
19 unique mice and 89 unique sessions against the 89 spike files).

## 1-c. How are the data split into sessions?

i. A session is one recording = one mouse × one date × one block, keyed `mname_datexp_blk`, which is
also the name of the spike file. Recordings listed under more than one experiment type (33 of them)
are kept only on first encounter, and a recording listed twice within the same experiment type under
two `stimtype`s (swap sessions) is likewise kept once. This yields 89 sessions. Sessions producing
fewer than 2 usable trials would be dropped (none actually were).

ii.
```python
            session_key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if session_key in sessions:
                continue
```
```python
        if result is None:
            print(f"  Skipped: < 2 valid trials", flush=True)
            continue
```

iii. Step 31: "89 unique sessions, but some sessions appear in multiple experiment types because the
same recording was analyzed for different stimuli. But the neural data and trials are the same - just
different stim_id assignments. For the decoder, I should use each recording only once." Step 33
confirmed the swap-`stimtype` duplicates share identical trial structure.

## 1-d. How are the data split into trials?

i. The split is taken from the data: each imaging frame is labelled with its trial in `ft_trInd` and
with whether it is inside the texture part of the corridor in `ft_CorrSpc`. A trial is the set of
frames with `ft_trInd == trial_idx & ft_CorrSpc`, i.e. the 4 m texture traversal from corridor entry
to entry into the grey space; the 2 m grey space is excluded. No frames are removed inside a trial
(stationary frames are kept), so trials are contiguous but of variable length (11 to 5607 frames,
median 32).

ii.
```python
    for trial_idx in range(ntrials):
        mask = (ft_trInd == trial_idx) & ft_CorrSpc
        frame_indices = np.where(mask)[0]

        if len(frame_indices) < 2:
            continue

        T = len(frame_indices)
```

iii. Step 27: "the 4m texture area splits naturally into 4 one-meter bins... I'll define each trial
as running from corridor entry to gray space entry, keeping position naturally within the 0-40dm
range." On keeping non-running frames (step 35): "including non-running frames fits that exception
since the decoder needs contiguous time series for its temporal inputs and to predict running speed
across all states. The paper's filtering serves a different purpose — computing summary statistics
like d-prime." The AI verified empirically (step 29) that `ft_CorrSpc` frames of a trial span
positions 0–39.1 dm and that gray frames span 40.5–59.4 dm.

## 1-e. How are trials filtered based on quality controls?

i. Almost no filtering. A trial is dropped only if fewer than 2 of its frames were imaged inside the
corridor (`len(frame_indices) < 2`); a session is dropped if it ends up with fewer than 2 trials (no
session did). 38,110 of ~38,111 behavioural trials survive. In particular **no upper limit on trial
length is applied**: trials in which the mouse stopped for minutes are retained, including one of
5607 frames (~29 min, mouse parked at 1.4 m). The AI explicitly noticed these outliers and considered
either a running-frame filter or a max-length cap, but ultimately kept everything.

ii.
```python
        if len(frame_indices) < 2:
            continue
```
```python
    if len(neural_trials) < 2:
        return None
```

iii. Step 87: "checking the max trial length, some trials have 5607 frames — about 29 minutes at this
sampling rate — which is way too long and suggests the mouse simply wasn't running... Very long trials
with mostly stationary periods would dominate the training data and could degrade decoder
performance." The AI weighed filtering to running frames ("matches the paper") against contiguity and
decided: "I'm leaning toward including all frames to preserve temporal continuity", because dropping
non-running frames would put gaps into `time_since_trial_start` / `time_to_sound_cue`. A cap on trial
length was raised ("maybe I should just cap trial length at some maximum frame count instead") but
never implemented.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of one (neurons × frames) array
per imaging plane, concatenated along the neuron axis. The per-neuron visual area comes from `iarea`
in `retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    spk = np.concatenate(spk_data['spks'], axis=0)
```
```python
    ret = np.load(os.path.join(RET_ROOT, fn))
    brain_reg_idx = map_brain_regions(ret['iarea'])
```

iii. Module docstring: "Use deconvolved fluorescence traces as stated in the paper ('All our analyses
were based on deconvolved fluorescence traces')." The concatenation order was taken from the reference
`utils.load_spk`, and step 40 verified the concatenated neuron count equals `len(iarea)` for all 89
sessions.

## 2-b. How is the `neural` data processed?

i. No processing at all: the columns of the concatenated `spks` matrix belonging to a trial are sliced
out and cast to `float32`. No normalisation, no smoothing, no padding, no z-scoring; trials keep their
native variable length. The resulting pickle is 296 GB.

ii.
```python
        # Neural data: (n_neurons, T) as float32
        neural_trial = spk[:, frame_indices].astype(np.float32)
```

iii. The file already contains deconvolved traces, which the paper states all analyses are based on,
so nothing further is needed. The AI repeatedly worried about the resulting size (step 76: "276GB
pickle file! That's extremely large") and considered reducing sessions, filtering neurons, or
switching to position-interpolated bins, but concluded (step 90) "the file size is just what it is, so
I should let it finish and see if the decoder trains properly", having confirmed 1 TB RAM / 3.4 TB disk
were available. It did not consider a smaller dtype.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. Every Suite2p-curated cell is kept (4,691,034 total). `iarea` is mapped to
five labels — V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4) and a fifth catch-all `unassigned` for
`iarea` −1/7 and anything unexpected — so the 585,641 neurons outside the four identified visual areas
are retained and merely labelled. Mean neurons/session is 52,708 (vs 46,128 if the non-visual cells
were dropped).

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
BRAIN_REGION_MAP = {
    8: 0,                  # V1
    0: 1, 1: 1, 2: 1, 9: 1,  # mHV (medial higher visual)
    5: 2, 6: 2,            # lHV (lateral higher visual)
    3: 3, 4: 3,            # aHV (anterior higher visual)
    -1: 4, 7: 4,           # unassigned
}

def map_brain_regions(iarea):
    return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)
```

iii. Step 27: "For the decoder, I'm deciding to include all neurons rather than filtering by area,
since different analysis functions in the codebase handle this inconsistently anyway — the decoder can
learn which neurons carry useful signal on its own." The region map itself was copied from the
reference `utils.neu_area_ID` (module docstring: "Brain regions from retinotopy data: V1, mHV, lHV,
aHV, plus 'unassigned' for neurons outside identified visual areas (iarea == -1 or 7)"). Step 59
revisited the idea of area filtering purely as a size optimisation and rejected it: "There's no need to
filter neurons during conversion either, since PCA in the decoder handles that automatically."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry (trial start). Each trial's array starts at the first imaged
frame inside the texture corridor for that trial and runs to the last one; nothing is trimmed to a
common window and nothing is padded, so trials have different lengths. `off_start` is 0.0 and
`off_end` is `None` in the metadata.

ii.
```python
        mask = (ft_trInd == trial_idx) & ft_CorrSpc
        frame_indices = np.where(mask)[0]
        ...
        neural_trial = spk[:, frame_indices].astype(np.float32)
```
```python
            'temporal_alignment_event': 'Corridor entry (trial start)',
            'off_start': 0.0,
            'off_end': None,
```

iii. Module docstring: "Temporal alignment: trial start = corridor entry (StartFr)." Step 20/27: the
AI compared `StartFr`/`GrayFr`/`EndFr` against the per-frame flags and chose the per-frame flags
because "there are per-frame arrays (timestamp, trial index, corridor flag, gray-space flag) that let
me directly select which frames belong to which trial... which is much cleaner than working from
start/end frame numbers" (the frame-number fields are fractional). Variable lengths were accepted
because "the data format already supports" them (verified against `train_decoder.py`).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame; no rebinning, resampling or interpolation is done. The bin size is
measured once from the median inter-frame interval of the first session's `ft` (MATLAB datenums →
seconds) and written to metadata as 314.69 ms (3.178 Hz). The same scalar `dt_seconds` is then used for
every session's time computations, although the AI measured that the per-session interval ranges from
0.3144 to 0.3154 s.

ii.
```python
    first_key = sorted(sessions.keys())[0]
    ft = sessions[first_key]['beh']['ft']
    dt_seconds = float(np.median(np.diff(ft)) * 86400)
    dt_ms = dt_seconds * 1000
```
```python
            'time_bin_size': dt_ms,
            'frame_rate_hz': 1.0 / dt_seconds,
```

iii. Step 35: "the calcium imaging rate is around 3.17 Hz, so I could just take the median frame
interval from the timestamps and convert to seconds, but since this could vary slightly by session, I
should probably pick the most common value across sessions for metadata purposes." Step 41 then
verified across all sessions: "Frame intervals (seconds): mean=0.3148, std=0.000286, min=0.3144,
max=0.3154 ... All ~same: True", justifying a single global bin size. The frame is the finest
resolution available and all behavioural streams are already on that grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundFr` (the fractional imaging-frame number of the sound cue in each trial) together with the
integer frame indices of the trial and the global `dt_seconds` derived from `ft`.

ii.
```python
    SoundFr = beh['SoundFr']
    ...
        time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. Step 27: "For time-to-sound, I'm computing frame time minus sound onset time in seconds"; the AI
noted `SoundFr` is fractional and decided to "rely on the floating point SoundFr frame indices
directly rather than rounding them" (step 39).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every frame of the trial: `(SoundFr[trial] - frame_index) * dt_seconds`, in seconds. The sign
convention is positive before the cue and negative after, matching the name "time **to** sound cue".
Frame times are approximated as `frame_index * dt` with a single global `dt` rather than read from the
per-frame `ft` timestamps (error ≤0.3%, i.e. <0.01 s on a typical 25-frame trial). Stored as float32,
one value per bin. Observed range is −1762 s to +723 s, driven by the un-capped stationary trials.

ii.
```python
        time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
        ...
        input_trial = np.stack([time_to_sound, day_train, time_since_start, reward_avail], axis=0)
```

iii. Step 27: "Now I'm settling on time-to-sound-cue meaning time remaining until the cue: sound frame
minus current frame divided by frame rate, positive before the cue occurs and negative afterwards."
Using a constant `dt` was justified by the step-41 measurement that the frame interval is essentially
constant across the whole dataset.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed directly from `frame_indices`, the very same frame list used to slice the neural
matrix, so it is sample-for-sample aligned with the neural columns and has the trial's length.
`SoundFr` is already expressed in imaging-frame units, so no cross-clock conversion is needed.

ii.
```python
        mask = (ft_trInd == trial_idx) & ft_CorrSpc
        frame_indices = np.where(mask)[0]
        T = len(frame_indices)
        neural_trial = spk[:, frame_indices].astype(np.float32)
        time_to_sound = ((SoundFr[trial_idx] - frame_indices) * dt_seconds).astype(np.float32)
```

iii. All behavioural streams in this dataset are indexed by imaging frame, so alignment is by frame
number; the AI used one `frame_indices` vector for every stream of a trial.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `ndb['datexp']` (the recording date string, e.g. `2022_07_19`) and `ndb['mname']`, taken from
`Imaging_Exp_info.npy`. The alternative index fields `sess#` and `days` were inspected and rejected.

ii.
```python
def parse_date(datexp):
    return datetime.strptime(datexp, '%Y_%m_%d')
```
```python
    for session_key, info in sessions.items():
        mname = info['ndb']['mname']
        dt = parse_date(info['ndb']['datexp'])
        mouse_dates.setdefault(mname, []).append(dt)
```

iii. Step 35: "For each mouse, I'll compute days since its first recording by parsing dates from
datexp rather than relying on inconsistent fields like `ndb['days']` or `ndb['sess#']`, since date
parsing seems more reliable across sessions" (the index entries do not all carry the same fields).

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Calendar days elapsed between the session's date and the earliest recording date of that mouse
(so the first session of a mouse is 0). The value is per-session, cast to float32 and broadcast across
all bins of every trial of that session. The resulting range is 0–92 days.

ii.
```python
def compute_days_of_training(sessions):
    """For each mouse, compute calendar days from first recording."""
    ...
    mouse_first_date = {m: min(dates) for m, dates in mouse_dates.items()}

    return {
        sk: (parse_date(info['ndb']['datexp']) - mouse_first_date[info['ndb']['mname']]).days
        for sk, info in sessions.items()
    }
```
```python
        day_train = np.full(T, day_of_training, dtype=np.float32)
```

iii. Step 39: "calendar days between sessions computed from the sorted recording dates for each mouse
seems most accurate — parsing dates from the datexp field and taking days elapsed from each mouse's
first session gives a clean monotonic value." Broadcasting per-trial scalars across time was chosen
(step 39) so that every input array has a uniform `(4, T)` shape: "the simplest approach is to make
everything (4, T) shaped, avoiding mixed formats."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the trial's own frame indices (`ft_trInd`/`ft_CorrSpc` masks) and `dt_seconds`. The fractional
`StartFr` field is *not* used: the reference point is the first imaged corridor frame of the trial,
which is the frame-grid realisation of corridor entry.

ii.
```python
        time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. Step 27: "Time-since-trial-start is just frame index relative to trial start times dt" — having
already decided that the trial begins at the first `ft_CorrSpc` frame of that trial (the AI noted
`StartFr` is fractional and preferred the per-frame flags; step 20/27).

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(frame_index - first_frame_of_trial) * dt_seconds`, in seconds, so the first bin of every trial is
exactly 0 and the value increases by ~0.3147 s per bin. Positive throughout (no pre-trial baseline is
included). Stored as float32 per bin; range 0 to 1764 s (again inflated by the uncapped long trials).

ii.
```python
        time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. Same as 5-a: with a fixed frame interval, elapsed time is simply the frame offset times `dt`;
the AI verified the frame interval is constant across the dataset (step 41).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. It is built from the identical `frame_indices` array used to slice the neural data, so it is
aligned bin-for-bin and has the same length as the trial's neural matrix.

ii.
```python
        neural_trial = spk[:, frame_indices].astype(np.float32)
        ...
        time_since_start = ((frame_indices - frame_indices[0]) * dt_seconds).astype(np.float32)
```

iii. Alignment is by imaging-frame index throughout, which is the common clock for all streams in this
dataset.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
    isRew = beh['isRew']
    ...
        reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
```

iii. Step 27/54: the flag is available directly; the AI kept unsupervised and naive cohorts in the
dataset knowing `isRew` is false for them, on the grounds that "the decoder inputs (reward
availability, day of training) capture condition differences" (module docstring).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool → float cast and broadcasting the per-trial scalar across the trial's bins to
keep the input array shape `(4, T)`.

ii.
```python
        reward_avail = np.full(T, float(isRew[trial_idx]), dtype=np.float32)
        input_trial = np.stack([time_to_sound, day_train, time_since_start, reward_avail], axis=0)
```

iii. Step 39: per-trial values are broadcast over time so that all four inputs share one uniform
`(4, T)` layout rather than mixing `(d,)` and `(d, T)` shapes.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the per-trial name of the wall texture. `TrialStim`/`stim_id` were examined and
deliberately not used.

ii.
```python
    WallName = beh['WallName']
    ...
        stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
```

iii. Step 31: "I'll rely on WallName directly as the stimulus category, since that's independent of
stim_id — stim_id is just used for the paper's specific comparisons" (the same recording appears under
several experiment types with different `stim_id` arrays).

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The AI enumerated all 15 distinct `WallName` values occurring anywhere in the dataset
(circle1/2/3, leaf1/2/3 and two leaf1 swaps, rock1/2, wood1/2/5 and two wood1 swaps), sorted them, and
uses the index into that global list as the label — i.e. **15 categories at crop level, not the 4
texture categories (circle, leaf, rock, wood/brick) used by the paper**. Individual crops of the same
photograph (leaf1, leaf2, leaf3) and the spatially shuffled controls (leaf1_swap1/2) therefore become
separate classes. The value is per trial, broadcast across the trial's bins as int64. Each session
only contains 4–8 of the 15 classes.

ii.
```python
ALL_STIMULI = sorted([
    'circle1', 'circle2', 'circle3',
    'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2', 'leaf3',
    'rock1', 'rock2',
    'wood1', 'wood1_swap1', 'wood1_swap2', 'wood2', 'wood5',
])
STIM_TO_IDX = {s: i for i, s in enumerate(ALL_STIMULI)}
```
```python
        stim_idx = STIM_TO_IDX[str(WallName[trial_idx])]
        stim_out = np.full(T, stim_idx, dtype=np.int64)
```

iii. Step 31/35: the AI collected the wall names empirically ("All unique stimuli: circle1, circle2,
circle3, leaf1, ... wood5") and weighed the two options: "broad categories lose information, but
including every specific stimulus name means some sessions won't have all of them. The practical
solution is to use per-session stimulus names since the decoder predicts on a per-session basis"; it
then settled on a fixed global 15-name list so that the label index is consistent across sessions
("I count 15 distinct stimuli across sessions... so I need a consistent global index mapping for
stimulus names", step 39). It also noted "the paper mentions rock and bricks, but in the data they
seem to be wood instead of bricks".

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']` (fractional imaging-frame number of every lick) together with `beh['LickTrind']`
(the trial each lick belongs to), which is used to restrict the licks considered to the current trial.

ii.
```python
def compute_lick_per_frame(beh, trial_idx, frame_indices):
    """Compute binary licking array for given frames of a trial."""
    lick_mask = beh['LickTrind'] == trial_idx
    if not np.any(lick_mask):
        return np.zeros(len(frame_indices), dtype=np.float32)

    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```

iii. Module docstring: "Licking derived per frame from LickFr/LickTrind." Step 27: "I'll check each
frame against LickFr/LickTrind for lick occurrence".

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick's fractional frame number is **rounded** to the nearest integer frame; a frame of the
trial is 1 if at least one of that trial's licks rounds onto it, else 0. Multiple licks in a bin
collapse to a single 1. Implemented as a Python-level membership test per frame against a `set`.
Result stored as int64; overall lick rate in the converted data is 3.5% of bins.

ii.
```python
    lick_frames_set = set(np.round(beh['LickFr'][lick_mask]).astype(int))
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```
```python
        lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices).astype(np.int64)
```

iii. Step 39: "since LickFr values may fall between frames, I'll assign each lick to its floor frame
index... Since LickFr appears to be interpolated positions in the frame time series rather than exact
indices, using round() to map each lick to its nearest integer frame makes more sense. Multiple licks
landing in the same frame interval is fine since licking is just binary presence."

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already in imaging-frame units, and the binary flag is evaluated exactly on the trial's
`frame_indices`, the same list used to slice the neural matrix, giving one lick value per neural bin.

ii.
```python
        neural_trial = spk[:, frame_indices].astype(np.float32)
        ...
        lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices).astype(np.int64)
```

iii. Alignment by frame index, as for all other streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the virtual-corridor position at each imaging frame, in decimeters (0–40 across the
texture, continuing to 60 through the grey space), trimmed to the number of imaged frames.

ii.
```python
    ft_Pos = beh['ft_Pos'][:nfr]
    ...
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. Step 22/29: the AI verified `ft_Pos` ranges 0–60 with `Texture_Length = 40`,
`Gray_Space_length = 20`, and that corridor frames of a trial cover 0–39.1 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Integer-divide the decimeter position by 10 to get 1-metre bins, clip to [0, 3], store as int64 per
bin. No interpolation or spatial resampling. Because only `ft_CorrSpc` frames are kept, the position
never exceeds 40 dm, so the clip is only a safety net.

ii.
```python
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```
```python
        ['0-1m', '1-2m', '2-3m', '3-4m'],
```

iii. Step 27: "The 4m texture area splits naturally into 4 one-meter bins... discretize into bins
[0,10), [10,20), [20,30), [30,40]", which is exactly what the decoder task specifies.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-width thresholds at 10, 20 and 30 dm (1 m, 2 m, 3 m) — the four bins are the four
1-m segments required by the instructions, not data-driven quantiles. The realised occupancies are
28.5% / 23.3% / 23.6% / 24.6%, the first bin being fuller because the mouse can stall at position 0.

ii.
```python
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. Directly mandated by the decoder task ("Position in corridor discretized into 4 equal-length,
1-m-long spatial bins"); the AI explicitly matched the trial definition (texture corridor only) to
this requirement.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one entry per imaging frame and is indexed with the trial's `frame_indices`, the same
frames used for the neural columns, so it is aligned bin-for-bin.

ii.
```python
        neural_trial = spk[:, frame_indices].astype(np.float32)
        ...
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
```

iii. Alignment by frame index, as for all other streams.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed of the mouse at each imaging frame (trimmed to the imaged
frames). `ft_move` (the VR displacement) was considered and rejected.

ii.
```python
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
    ...
        speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. Step 35: "For the speed signal itself, I'll use the raw running speed variable rather than the VR
movement delta, since the latter is more binary and less representative of actual running behavior."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass over all 89 sessions pools `ft_RunSpeed` over every corridor frame
(`ft_CorrSpc & ~isnan(ft_trInd)`, i.e. including frames of trials later skipped) and computes the 25th,
50th and 75th percentiles of that global pool. Those three thresholds are then applied to every session
with `np.digitize`. The thresholds come out as 0.0, 8.33 and 30.16.

ii.
```python
def compute_speed_quartiles(sessions):
    """Compute running speed quartile edges across all corridor frames."""
    all_speeds = []
    for session_key in sorted(sessions.keys()):
        ...
        nfr = get_nfr(ndb)
        ft_trInd = beh['ft_trInd'][:nfr]
        ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
        ft_RunSpeed = beh['ft_RunSpeed'][:nfr]

        corridor_mask = ft_CorrSpc & ~np.isnan(ft_trInd)
        all_speeds.append(ft_RunSpeed[corridor_mask])

    all_speeds = np.concatenate(all_speeds)
    q25, q50, q75 = np.percentile(all_speeds, [25, 50, 75])
    return np.array([q25, q50, q75])
```

iii. Step 35: "the spec wants it discretized into quartiles, so I'll compute the 25th/50th/75th
percentiles across all frames including zero-speed ones, since a 'not running' bin is a valid
category... that percentile computation should be restricted to corridor frames only, to stay
consistent with the rest of the dataset." A global (rather than per-session) pool was chosen so the
bin labels mean the same thing in every session.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [q25, q50, q75])`, labelled `Q1 (slowest)`…`Q4 (fastest)`. Because the pooled
25th percentile is exactly 0.0 and a large mass of frames sit at exactly zero speed, `digitize`
(left-closed) sends all zero-speed frames to bin 1 and leaves only strictly negative speeds in bin 0.
The realised bin occupancies are **9.8% / 40.2% / 25.0% / 25.0%**, not the four equal 25% bins the task
specifies, and bin 0 ("slowest") in fact contains only backwards ball motion.

ii.
```python
        speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```
```python
        ['Q1 (slowest)', 'Q2', 'Q3', 'Q4 (fastest)'],
```
```python
    # metadata
    'speed_quartile_edges': speed_quartiles.tolist(),   # [0.0, 8.327, 30.157]
```

iii. The AI's stated intent was quartiles ("Running speed quartiles computed across all corridor frames
in all sessions", module docstring) and it was aware that many frames are stationary ("including
zero-speed ones, since a 'not running' bin is a valid category", step 35), but it did not check the
realised bin sizes and did not consider that the ties at zero make threshold-based binning unable to
produce 25% bins. The printed edge `Q25=0.00` was visible in its own run log.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is indexed with the trial's `frame_indices` — the
same frames used for the neural columns — so it is aligned bin-for-bin.

ii.
```python
        neural_trial = spk[:, frame_indices].astype(np.float32)
        ...
        speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. Alignment by frame index, as for all other streams.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) The behaviour arrays can run past the imaging, so every per-frame behavioural stream is trimmed
to the number of imaged frames `nfr` before use. (b) Frames not assigned to a trial (`ft_trInd` NaN)
fall out naturally because `NaN == trial_idx` is False, and are also explicitly excluded from the speed
pool. (c) Trials with fewer than 2 imaged corridor frames are dropped, and sessions left with fewer
than 2 trials are dropped. (d) Sessions whose behaviour key is absent from the behaviour file are
skipped in `collect_sessions`. (e) Unknown `iarea` codes fall back to the `unassigned` label via
`dict.get(..., 4)`. There is no `try/except` around per-session processing, so any unexpected failure
would abort the whole (multi-hour) run, and licks whose frame rounds past `nfr` are not explicitly
removed (they simply never match a kept frame index).

ii.
```python
    spk = load_spk(ndb)
    n_neurons, nfr = spk.shape
    ...
    ft_trInd = beh['ft_trInd'][:nfr]
    ft_CorrSpc = beh['ft_CorrSpc'][:nfr]
    ft_Pos = beh['ft_Pos'][:nfr]
    ft_RunSpeed = beh['ft_RunSpeed'][:nfr]
```
```python
        corridor_mask = ft_CorrSpc & ~np.isnan(ft_trInd)
```
```python
        if len(frame_indices) < 2:
            continue
```

iii. Step 59: the AI observed the mismatch directly — "I did notice a minor discrepancy where the
behavior array has one more frame than the neural spike data (22476 vs 22475), but that single-frame
difference is negligible... For the actual processing step, I need to trim to the neural frame count."
Otherwise the dataset was found to be clean (the format verifier reported no errors or warnings).

## 12-a. What are the most time-consuming steps of the code?

i. Reading the spike files, which total ~405 GB — and the code reads them **twice**: once in
`compute_speed_quartiles`, where `get_nfr` unpickles the entire `<session>_neural_data.npy` just to
read `spks[0].shape[1]`, and once in `process_session`. `get_nfr`'s docstring claims the opposite
("Get number of neural frames without loading full data"), but `np.load(..., allow_pickle=True).item()`
materialises every plane. Second most expensive is pickling/writing the 296 GB output file (the AI's
log shows the save alone taking tens of minutes). Everything else — masks, percentiles, per-trial
slicing — is negligible by comparison.

ii.
```python
def get_nfr(ndb):
    """Get number of neural frames without loading full data."""
    fn = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}_neural_data.npy"
    spk_data = np.load(os.path.join(SPK_ROOT, fn), allow_pickle=True).item()
    nfr = spk_data['spks'][0].shape[1]
    del spk_data
    return nfr
```
```python
        nfr = get_nfr(ndb)          # in compute_speed_quartiles
        ...
    spk = load_spk(ndb)             # again in process_session
```
```python
    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(data, f, protocol=4)
```

iii. The AI diagnosed this itself (step 54): "the issue is that `compute_speed_quartiles` loads neural
data just to get frame count, which is very wasteful... Really I could just use the behavior data
length as an approximation for the neural frame count, or peek at just the shape of the first plane
instead of loading everything", and announced the fix (step 60): "The issue was my script loaded
neural data twice (for speed quartiles and processing). Let me rewrite it efficiently." The rewrite
only moved the offending lines into `get_nfr`; the second full read was never removed.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two loops. (1) The per-trial mask `(ft_trInd == trial_idx) & ft_CorrSpc` rescans the whole frame
axis once per trial (~450 trials × 89 sessions); a single grouping pass (e.g. `np.argsort`/`np.unique`
on `ft_trInd`) would do the same work once. (2) `compute_lick_per_frame` builds a Python `set` and
tests membership frame by frame in a list comprehension, for every trial; this could be a single
vectorised `licking[lick_frames] = 1` boolean array per session, as the flag only needs to be built
once per session rather than once per trial. `map_brain_regions` is likewise a Python loop over up to
90,000 neurons (`np.isin` would vectorise it). All are small next to the I/O cost.

ii.
```python
    for trial_idx in range(ntrials):
        mask = (ft_trInd == trial_idx) & ft_CorrSpc
        frame_indices = np.where(mask)[0]
```
```python
    return np.array([1.0 if f in lick_frames_set else 0.0 for f in frame_indices], dtype=np.float32)
```
```python
    return np.array([BRAIN_REGION_MAP.get(int(ia), 4) for ia in iarea], dtype=np.int64)
```

iii. Not discussed in the trajectory; the AI's optimisation attention went entirely to memory and to
the neural-data I/O.

## 12-c. What processing does the code repeat multiple times?

i. (1) Every spike file is fully unpickled twice (frame-count pass + processing pass) — a duplicated
405 GB read. (2) The trial/corridor mask over the frame axis is recomputed per trial. (3) The lick set
is rebuilt for every trial from the full `LickTrind`/`LickFr` arrays instead of once per session.
(4) Behaviour files are read once each, but every session's behaviour dict is then retained in RAM for
the whole run.

ii.
```python
        nfr = get_nfr(ndb)      # first full read of the spike file
```
```python
    spk = load_spk(ndb)         # second full read of the same spike file
```
```python
    lick_mask = beh['LickTrind'] == trial_idx      # recomputed per trial
```

iii. The duplicate read was acknowledged and nominally "fixed" (steps 54/60, see 12-a) but remains in
the delivered script. The other repeats were not mentioned.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) Neural traces are stored as `float32` where `float16` would hold the deconvolved values
adequately — this alone doubles the 296 GB pickle. (2) All 585,641 neurons with no assigned visual
area are converted and stored, which the paper's own analyses never use. (3) Outputs are stored as
`int64` although they take only 4 values (`int8` suffices). (4) The whole second pass over the spike
files (12-a) produces nothing but a frame count that could have come from the behaviour arrays.
(5) The 29-minute stationary trials are converted in full, contributing thousands of bins that all
carry the same position label. The combination made the output file 296 GB, requiring ~300 GB of RAM to
load, and the decoder run took several hours — while the decoder itself immediately reduces each
session to 100 principal components.

ii.
```python
        neural_trial = spk[:, frame_indices].astype(np.float32)
```
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
    -1: 4, 7: 4,           # unassigned
```
```python
        stim_out = np.full(T, stim_idx, dtype=np.int64)
        lick_out = compute_lick_per_frame(beh, trial_idx, frame_indices).astype(np.int64)
        pos_out = np.clip(ft_Pos[frame_indices] // 10, 0, 3).astype(np.int64)
        speed_out = np.digitize(ft_RunSpeed[frame_indices], speed_quartiles).astype(np.int64)
```

iii. The AI was aware of the size problem throughout (steps 54, 59, 76, 87, 90) and considered
dropping sessions, filtering to visual-cortex neurons, restricting to running frames, and
position-interpolating, but rejected each in turn — "There's no need to filter neurons during
conversion either, since PCA in the decoder handles that automatically — I just need to keep everything
in float32 to save memory" (step 59) — after confirming the machine had 1 TB RAM and 3.4 TB disk
(step 60). The int64 outputs were introduced deliberately to fix a `list indices must be integers`
crash in the verifier's summary printing (step 81/83), without considering a narrower integer type.
