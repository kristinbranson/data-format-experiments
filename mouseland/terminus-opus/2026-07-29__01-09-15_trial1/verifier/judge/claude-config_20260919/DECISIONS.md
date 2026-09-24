# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads everything from three subfolders of `data/`: `beh/` (behavior), `spk/` (deconvolved
traces) and `retinotopy/` (visual area per neuron), all with relative paths from `/app`.
`data/beh/Imaging_Exp_info.npy` is read first as the master index; it is a dict keyed by experiment
type, each holding a list of recording entries. `build_all_sessions_info()` flattens this into a
dict keyed by `mname_datexp_blk`, recording the list of experiment types each recording appears
under. For every recording the AI then (a) loads behavior by opening `data/beh/Beh_<exp_type>.npy`
for each of that recording's experiment types and looking for the key `sess_id` or any key starting
with `sess_id + '_swap'`, (b) loads `data/spk/<sess_id>_neural_data.npy` and concatenates the
per-plane `spks` arrays, (c) loads `data/retinotopy/<mname>_<datexp>_trans.npz` for `iarea`.
The whole dataset is traversed **twice**: pass 1 loads only behavior to pool running speeds for the
global quartile edges, pass 2 loads behavior + spikes + retinotopy and builds the trials.

**Important**: the delivered `converted_data.pkl` contains only **76 of the 89 recordings**. The 13
recordings whose behavior is stored only under `_swap1`/`_swap2` keys were reported as
`WARNING: No behavioral data found ... SKIPPED` during the full run. The agent diagnosed this at
trajectory step 161-162 and patched `load_beh_for_session` to find the swap keys, but then decided
(step 163) *not* to finish the refactor or re-run the conversion. So the `convert_data.py` that
ships with the solution finds those keys (verified by executing it), while the shipped
`converted_data.pkl` was produced by the older version and is missing those 13 sessions.

ii.
```python
exp_info = np.load('data/beh/Imaging_Exp_info.npy', allow_pickle=True).item()
all_sessions = build_all_sessions_info(exp_info)
```
```python
def build_all_sessions_info(exp_info):
    sessions = {}
    for exp_type, sess_list in exp_info.items():
        for s in sess_list:
            sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
            if sess_id not in sessions:
                sessions[sess_id] = {'mname': s['mname'], 'datexp': s['datexp'], 'blk': s['blk'],
                                     'exp_types': [], 'exptype': s.get('exptype', 'unknown'),
                                     'rewType': s.get('rewType', 'None')}
            sessions[sess_id]['exp_types'].append(exp_type)
    return sessions
```
```python
def load_beh_for_session(sess_id, exp_types):
    results = []
    for exp_type in exp_types:
        beh_path = f'data/beh/Beh_{exp_type}.npy'
        if os.path.exists(beh_path):
            beh_all = np.load(beh_path, allow_pickle=True).item()
            if sess_id in beh_all:
                results.append((beh_all[sess_id], sess_id))
            for key in sorted(beh_all.keys()):
                if key.startswith(sess_id + '_swap'):
                    results.append((beh_all[key], key))
    ...
```
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    dat = np.load(os.path.join(root, f'{mname}_{datexp}_{blk}_neural_data.npy'),
                  allow_pickle=True).item()
    return np.concatenate([nspk for nspk in dat['spks']], 0)

def load_retino(mname, datexp, root='data/retinotopy'):
    dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
    return {'iarea': dtrans['iarea'], 'neu_ar_idx': neu_area_ID(dtrans['iarea'])}
```

iii. `load_spk`, `load_retino` and `neu_area_ID` are explicitly copied from the reference
`code/utils.py` (documented in CONVERSION_NOTES Step 1). For the 13 dropped recordings the agent
wrote: *"The swap sessions add complexity but only affect 13 out of 89 sessions... The swap sessions
are test3 experiments which introduce leaf1_swap stimuli - they're important for the paper's
analysis but not critical for the decoder task."* CONVERSION_NOTES Step 9 records the mismatch as
"Partial" and attributes the raised minimum neuron count (24,383 vs the paper's 20,547) to the
skipped sessions.

## 1-b. How are the data split into subjects?

i. The subject is `mname` from the master index entry, carried on each session record as
`result['mname']`. After processing, the unique mouse names are sorted into `subjects` and each
session gets `subject_idx` = index into that list. Result: 19 subjects, matching the paper.

ii.
```python
all_mice = sorted(set(r['mname'] for r in all_results))
mouse_to_idx = {m: i for i, m in enumerate(all_mice)}
...
subject_idx.append(mouse_to_idx[result['mname']])
...
'subjects': all_mice,
'subject_idx': np.array(subject_idx),
```

iii. The index already names the mouse; no split has to be derived. CONVERSION_NOTES Step 9 lists
"Subjects 19 / 19 / Yes" as a passing consistency check against the paper.

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block, i.e. `f"{mname}_{datexp}_{blk}"`, which is also
the name of the spike file. `build_all_sessions_info` de-duplicates with `if sess_id not in
sessions`, so a recording listed under several experiment types is created once and the extra
experiment types are only appended to its `exp_types` list. When a recording has several behavior
variants (swap1/swap2), only the first is used (`beh, beh_key = beh_results[0]`), so one recording
is always one output session. Sessions with fewer than 2 valid trials are dropped.

ii.
```python
sess_id = f"{s['mname']}_{s['datexp']}_{s['blk']}"
if sess_id not in sessions:
    sessions[sess_id] = {...}
sessions[sess_id]['exp_types'].append(exp_type)
```
```python
# Use the first (primary) behavioral data
beh, beh_key = beh_results[0]
```
```python
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. CONVERSION_NOTES Step 2 notes 89 neural files and that the behavior is keyed by
`mouse_date_block`. The agent observed (trajectory step 27) that "99 unique session keys in
behavioral data, 89 neural data files" and that the extras are the `_swap` variants of the same
recording, so the spike file defines the session.

## 1-d. Are the data correctly split into trials / how are the data split into trials?

i. Trials are the `ntrials` traversals declared in the behavior. For each trial the AI takes the
*frame window* `[round(StartFr[t]), round(GrayFr[t]))`, i.e. from corridor entry up to the grey
space, capped at 1000 frames and at the number of imaged frames. It does **not** use `ft_trInd` /
`ft_CorrSpc` (the per-frame trial and in-texture labels the reference uses).

Because `StartFr` and `GrayFr` are fractional, `np.round` disagrees with the authors' own
`ft_CorrSpc` labelling: checked over the 3,022 trials of `Beh_unsup_test1.npy`, the AI's window
starts one frame **earlier** than the true first in-corridor frame on 1,743/3,022 trials (58%) and
ends one frame **earlier** than the true last in-corridor frame on 1,653/3,022 trials (55%). The
extra leading frame is the tail of the previous traversal's grey space (`ft_Pos` ≈ 59), so 1.4% of
all emitted frames lie outside the 4 m texture.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
gray_fr  = int(np.round(beh['GrayFr'][trial_idx]))
...
end_fr = min(start_fr + n_timepoints, gray_fr)     # n_timepoints = 1000
actual_frames = end_fr - start_fr
...
neural   = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
speed    = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```
```python
for t in range(ntrials):
    trial_data = extract_trial_data(spk, beh, t, n_timepoints=1000)  # large max
```

iii. CONVERSION_NOTES Step 5 maps "spks (concat planes) → neural, Extract per-trial from StartFr to
GrayFr". Trajectory step 29: *"Texture area: positions 0-40 (4m), where ft_CorrSpc=True; Grey space:
positions 40-60 (2m)... I should include the corridor portion (from StartFr to GrayFr) since that's
when the visual stimuli are presented."* Step 10 Check 3 claims "Trial extraction: Uses StartFr to
GrayFr (corridor portion only)". Trials are kept at their natural, variable length (no padding).

## 1-e. How are trials filtered based on quality controls?

i. Four rules, all inside `extract_trial_data`, and one at session level:
* `GrayFr - StartFr < 2` → trial dropped (needs at least 2 frames);
* `start_fr < 0` or `end_fr > spk.shape[1]` → trial dropped entirely (this is what removes the one
  trial with `StartFr = -0.66` in `TX83_2022_08_31_1`);
* trial length truncated to at most `n_timepoints = 1000` frames;
* sessions left with fewer than 2 trials are dropped.

There is **no** outlier-length filtering. Measured over all behavior files, 0.9% of traversals are
longer than the reference's 99th-percentile limit (~239 frames) yet those trials hold **14% of all
corridor frames** — these are animals that stopped, not animals running slowly. 54 traversals exceed
1000 frames and are silently truncated at the 1000-frame cap. The consequence is visible in the
verification output: `T ... max: 1000`, `time_since_trial_start: [0.0, 315.1]` s (a 5-minute
"traversal" of a 4 m corridor).

ii.
```python
avail_frames = gray_fr - start_fr
if avail_frames < 2:          # need at least 2 frames
    return None
end_fr = min(start_fr + n_timepoints, gray_fr)
actual_frames = end_fr - start_fr
if actual_frames < 2:
    return None
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```
```python
if len(trial_neural) < 2:
    print(f'  WARNING: Session {sess_id} has fewer than 2 valid trials, skipping')
    return None
```

iii. CONVERSION_NOTES Step 10 Check 5 lists this as the edge-case handling: *"Float frame indices
(StartFr, SoundFr) rounded to int; Trials with <2 frames excluded (1 trial in TX83_2022_08_31_1);
Maximum trial length capped at 1000 frames (from extract_trial_data)."* The 1000-frame value is
introduced in the code only as the comment `# large max` and is never justified. The agent noticed
`max: 1000` in the verification output (trajectory step 159, "Check the max T of 1000 - this seems
like a cap from the extract_trial_data function") but never followed up.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `data/spk/<sess_id>_neural_data.npy` — a list of one (neurons × frames) array per
imaging plane, concatenated along the neuron axis. The area of each neuron comes from `iarea` in
`data/retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
def load_spk(mname, datexp, blk, root='data/spk'):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    dat = np.load(os.path.join(root, fn), allow_pickle=True).item()
    spk = np.concatenate([nspk for nspk in dat['spks']], 0)
    return spk
```
```python
spk = load_spk(mname, datexp, blk)
retino = load_retino(mname, datexp)
```

iii. CONVERSION_NOTES Step 1 identifies `load_spk` (utils.py:338) and `load_retino` (utils.py:330)
as the reference loaders and the functions are copied verbatim. Step 1 Notes: *"Neural data:
deconvolved Ca2+ traces from Suite2p (decay timescale 0.75 s)"*.

## 2-b. How is the `neural` data processed?

i. No processing at all. The frames of the trial window are sliced out of the concatenated `spks`
matrix and cast to `float16`. No dF/F, no deconvolution, no normalisation, no z-scoring, no padding.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
```

iii. The stored traces are already deconvolved (CONVERSION_NOTES Step 1/Step 3: "Deconvolution 0.75 s
decay", "Suite2p deconvolved fluorescence traces"), so nothing further is needed. `float16` is used
"to reduce file size" (CONVERSION_NOTES Step 6). README documents neural as float16.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are filtered.** Every neuron in the spike file is kept. The `iarea` code is mapped
to one of the four visual areas with the reference's `neu_area_ID`, and any neuron whose `iarea` is
not covered (`iarea == -1`, "no area", and `iarea == 7`) is assigned to a fifth pseudo-region called
`'unassigned'` rather than being dropped. The full dataset therefore reports 5 brain regions with
483,862 of 4,047,169 neurons (12%) in `'unassigned'`.

ii.
```python
def neu_area_ID(iarea):
    area_name = ['V1', 'mHV', 'lHV', 'aHV']
    idx = {}
    for ar in area_name:
        if ar == 'V1':    idx[ar] = iarea == 8
        elif ar == 'mHV': idx[ar] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
        elif ar == 'lHV': idx[ar] = (iarea == 5) | (iarea == 6)
        elif ar == 'aHV': idx[ar] = (iarea == 3) | (iarea == 4)
    return idx
```
```python
brain_regions = ['V1', 'mHV', 'lHV', 'aHV', 'unassigned']
neuron_region_idx = np.full(n_neurons, len(brain_regions) - 1, dtype=int)  # default: unassigned
for i, region in enumerate(brain_regions[:-1]):
    if region in region_map:
        neuron_region_idx[region_map[region]] = i
```

iii. CONVERSION_NOTES Step 1: *"No explicit neuron quality filtering in reference code"*, and Step 10
Check 3: *"No neuron filtering: Consistent with reference code"*. Trajectory step 20 explicitly
enumerates `iarea == -1: unassigned` and `iarea == 7: not assigned to any group in neu_area_ID`, so
the choice to retain them under a fifth label was deliberate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry, taken as `round(StartFr[t])`; that frame becomes index 0
of the trial and `time_since_trial_start` = 0 there. Trials are kept at their own natural length
(variable), nothing is padded, `off_start = 0.0` and `off_end = None` in the metadata. Every other
stream (position, speed, licking, inputs) is sliced with the identical `[start_fr:end_fr]` window,
so the streams are internally consistent.

The weakness is the alignment event itself: `StartFr` is fractional and `np.round` puts the origin
one frame before the true corridor-entry frame on ~58% of trials (see 1-d), so on those trials
timepoint 0 is the previous traversal's grey space.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
...
neural   = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
speed    = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,
'off_end': None,  # variable trial length
```

iii. The Decoder Task section specifies "Temporally aligned based on trial start (corridor entry)".
CONVERSION_NOTES Step 5 maps `frame - StartFr → time_since_trial_start`. Variable lengths were kept
because the format only requires a constant bin *size*, and README states "Trials are aligned to
corridor entry (trial start). Each trial spans from the start of the textured corridor to the grey
space transition."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame at 3.17 Hz = 315.46 ms. No rebinning, resampling or interpolation is
performed — all behavioural streams in this dataset are already sampled on the imaging frame grid
(`ft_*` variables), so the native grid is used throughout.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE  # ~315.5 ms
...
'time_bin_size': TIME_BIN_MS,
'frame_rate_hz': FRAME_RATE,
```

iii. CONVERSION_NOTES Step 3 lists "Frame rate 3.17 Hz | Notebook cell 4" as a value taken from the
reference notebook, and Step 9 records "Frame rate 3.17 Hz / 3.17 Hz / Yes" as a passing consistency
check. The imaging frame is the finest resolution available.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundFr'][t]` (the fractional imaging-frame number at which the sound cue was played on
that trial) and `round(beh['StartFr'][t])`. It is converted to seconds using the **nominal** frame
rate constant (3.17 Hz), not the per-frame timestamps `beh['ft']`.

ii.
```python
sound_fr = beh['SoundFr'][trial_idx]  # keep as float for precision
...
sound_frame_offset = sound_fr - start_fr
```
```python
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)],
                       dtype=np.float32)
```

iii. Trajectory step 13: *"beh['SoundFr'] = neural frame when sound cue delivered"*.
CONVERSION_NOTES Step 5 maps "SoundFr - frame → input[0]: time_to_sound_cue, Time in seconds".
The agent kept `SoundFr` as a float "for precision" while rounding `StartFr` to an int.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For bin `i` of a trial, the value is `(i - sound_frame_offset) / 3.17` seconds, i.e. **time
elapsed since the cue** — negative before the cue, positive after. This is the *negative* of the
reference's convention (`cue_time - frame_time`, positive before the cue), despite the name
`time_to_sound_cue`. The in-code comment states the convention the code actually implements.
Values range over [-404.1, +314.3] s in the full dataset; the extreme values come from the
unfiltered stopped-animal trials of 1-e, not from the cue itself.

ii.
```python
# Input 0: time_to_sound_cue (continuous, time-varying)
# Signed time from current frame to sound cue (negative = before cue, positive = after)
sound_offset = trial_data['sound_frame_offset']
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)],
                       dtype=np.float32)
```

iii. CONVERSION_NOTES Step 10 Check 2 offers the sanity check *"time_to_sound_cue ranges are
consistent with SoundPos distribution (0.5-3.5 m)"*. README documents input[0] as "signed time to
sound cue (seconds)". No justification is given for the sign convention, and the instruction's
wording ("Time to sound cue") is never reconciled with the implemented sign.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on the same frame index range as the neural slice: bin `i` of the input corresponds
to frame `start_fr + i`, exactly the columns taken from `spk`. Both arrays therefore have
`actual_frames` columns and index 0 of both is frame `start_fr`.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
...
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)], ...)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. Everything in this dataset is indexed by imaging frame (`ft`, `ft_Pos`, `SoundFr`, `LickFr`,
`StartFr` are all frame numbers), so using a single frame window for every stream is the alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `datexp` (the recording date string) from the master index, collected across **all** entries of
`Imaging_Exp_info.npy` for that mouse — including the 13 recordings that end up being skipped — so
the ordinal is computed over the complete set of recordings, not just the converted ones.

ii.
```python
def get_training_day(mname, datexp, exp_info):
    """Compute the training day index for a session.
    Returns the chronological day index (0-based) for this mouse."""
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
    sorted_dates = sorted(dates)
    return sorted_dates.index(datexp)
```

iii. CONVERSION_NOTES Step 5 maps "datexp → input[1]: day_of_training, Chronological day index".
The `YYYY_MM_DD` date string sorts chronologically as a string, which is the only field that orders
a mouse's recordings.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The index of the session's date in the sorted list of that mouse's unique recording dates (0 for
the first recorded day). It is a per-trial scalar, broadcast across every bin of the trial as
float32. Observed range in the full dataset: [0.0, 6.0].

ii.
```python
training_day = get_training_day(mname, datexp, exp_info)
...
# Input 1: day_of_training (continuous, per-trial)
day_val = np.full(actual_frames, training_day, dtype=np.float32)
```

iii. It counts *recorded* days rather than calendar days since the recording days of a mouse are not
consecutive. No mouse in this dataset has two blocks on the same date (verified), so counting unique
dates and counting recordings give identical results here.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['StartFr'][t]`, rounded to the nearest integer frame, together with the bin index within the
trial; converted to seconds with the nominal 3.17 Hz constant rather than the per-frame timestamps
`beh['ft']`.

ii.
```python
start_fr = int(np.round(beh['StartFr'][trial_idx]))
```
```python
# Input 2: time_since_trial_start (continuous, time-varying)
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5: "frame - StartFr → input[2]: time_since_trial_start, Time in seconds".

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `i / 3.17` seconds for bin `i`, so the first bin of every trial is exactly 0 and the spacing is a
constant 315.5 ms. Positive throughout, monotonically increasing. Observed range [0.0, 315.1] s
(the upper end is a consequence of the unfiltered stopped-animal trials, 1-e).

Two approximations relative to the reference: the actual frame interval is not constant (measured
`dt` in one session ranges 0.207–0.710 s around a mean of 0.3148 s), and the fractional part of
`StartFr` is thrown away by rounding, so the value is a nominal rather than a measured time.

ii.
```python
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```

iii. The frame rate is documented as 3.17 Hz in the reference notebook and is used as the single time
constant throughout the script (`TIME_BIN_MS = 1000.0 / FRAME_RATE`), so times and the declared bin
size are self-consistent.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame window as the neural slice — bin `i` is frame `start_fr + i`, and the array length is
`actual_frames`, identical to `neural.shape[1]`. Timepoint 0 is the alignment event itself.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
...
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
inputs = np.stack([time_to_cue, day_val, time_since_start, reward_avail], axis=0)
```

iii. As in 3-c, one frame window drives all streams, so the alignment is by construction.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew'][t]`, the boolean flag marking trials run in the rewarded corridor.

ii.
```python
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. Trajectory step 13: *"beh['isRew'] = whether trial is rewarded"*. CONVERSION_NOTES Step 5 maps
"isRew → input[3]: reward_availability, Binary".

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool→float cast and broadcasting the per-trial scalar across all bins of the trial.
Observed range [0.0, 1.0], as expected (unsupervised and naive mice have `isRew` False throughout;
supervised mice have both).

ii.
```python
# Input 3: reward_availability (discrete, per-trial)
reward_avail = np.full(actual_frames, float(beh['isRew'][t]), dtype=np.float32)
```

iii. The Decoder Task specifies "Reward availability: 1 if in rewarded corridor, 0 if not, discrete,
per-trial", which `isRew` provides directly.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName'][t]`, the name of the texture on the corridor walls for that trial.

ii.
```python
wall_names = beh['WallName']
...
stim_name = str(wall_names[t])
```

iii. Trajectory step 13: *"beh['WallName'] = stimulus name per trial"*. CONVERSION_NOTES Step 5 maps
"WallName → output[0]: visual_stimulus, Categorical".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. **The raw wall names are used as the categories, with no grouping into base textures.** The set of
distinct `WallName` strings encountered across all processed sessions is collected, sorted, and each
is assigned an index; the index is broadcast across all bins of the trial. This yields **13
categories**: `['circle1', 'circle2', 'circle3', 'leaf1', 'leaf1_swap1', 'leaf1_swap2', 'leaf2',
'leaf3', 'rock1', 'rock2', 'wood1', 'wood2', 'wood5']`, with a very skewed distribution
(leaf1 0.300, circle1 0.283 … circle3 0.010, leaf1_swap1 0.002, leaf1_swap2 0.003) and a chance
level of 1/13 = 0.077 rather than 1/4 = 0.25.

ii.
```python
def get_stimulus_category(wall_name):
    """Get the visual stimulus category name."""
    return wall_name       # (defined but never called)
```
```python
stim_name = str(wall_names[t])
...
all_stim_names.add(trial_out['stim_name'])
...
all_stim_names = sorted(all_stim_names)
stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
...
out[0, :] = stim_idx  # visual stimulus (per-trial, repeated)
```
```python
output_values = [all_stim_names, ['no_lick', 'lick'],
                 ['0-1m', '1-2m', '2-3m', '3-4m'],
                 ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']]
```

iii. No justification is given for using the raw names rather than the four base textures. The agent
had seen the reference `get_cat_id` function (CONVERSION_NOTES Step 1: "get_cat_id | utils.py:137 |
PROCESSING | Assign stimulus category IDs") and knew the stim_id mapping (trajectory step 13:
"0=circle1, 1=circle2, 2=leaf1, 3=leaf2, 4=leaf3, 5=leaf1_swap1, 6=leaf1_swap2"). Post-hoc,
CONVERSION_NOTES Step 12 rationalises the result: *"visual_stimulus has 13 classes (many rare), so
0.49 balanced accuracy is good"*. README describes the mice as running through corridors with
"naturalistic texture patterns (leaf, circle, rock, wood/brick)" — i.e. four textures — while the
output has 13 levels.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']` (the fractional imaging-frame number of each lick) together with
`beh['LickTrind']` (the trial index of each lick), which is used to select only the licks belonging
to the current trial.

ii.
```python
lick_frs = beh['LickFr']
lick_trinds = beh['LickTrind']
trial_lick_frs = lick_frs[lick_trinds == trial_idx]
```

iii. Trajectory step 13: *"beh['LickFr'] = neural frame of licks"*. CONVERSION_NOTES Step 5 maps
"LickFr → output[1]: licking, Binary per frame".

## 8-b. What processing is involved in computing `output` *Licking*?

i. A binary flag per bin: each of the trial's lick frames is **rounded** to the nearest integer
frame, offset by `start_fr`, and if it falls inside the trial window that bin is set to 1. Multiple
licks in one bin collapse to 1. Frames with no lick are 0. Result over the full dataset:
no_lick 0.965, lick 0.035. (The reference truncates rather than rounds the fractional lick frame;
rounding shifts roughly half the licks one bin later.)

ii.
```python
licking = np.zeros(actual_frames, dtype=np.float32)
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
```python
out[1, :] = trial_out['licking'].astype(np.int64)
```

iii. CONVERSION_NOTES Step 10 Check 2 sanity check: *"Licking: 96.5% no_lick, 3.5% lick - consistent
with most sessions being unsupervised (no licking)"*. Step 12 reports licking decoding at 0.874
validation balanced accuracy and calls it "Very good".

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The lick flag array is created with length `actual_frames` and indexed by `lick_frame -
start_fr`, i.e. on exactly the same frame window used for the neural columns, so bin `i` of `licking`
and column `i` of `neural` are the same imaging frame.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
...
licking = np.zeros(actual_frames, dtype=np.float32)
...
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```

iii. `LickFr` is already expressed in imaging frames, so subtracting `start_fr` puts it on the
trial's own frame grid.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the VR position of the mouse at every imaging frame, in VR units of 0.1 m
(0–40 across the 4 m texture, 40–60 through the 2 m grey space).

ii.
```python
position = beh['ft_Pos'][start_fr:end_fr].copy()
```
```python
CORRIDOR_LENGTH_VR = 40.0  # texture corridor in VR units (4m)
GRAY_LENGTH_VR = 20.0      # grey space in VR units (2m)
VR_UNIT_TO_METERS = 0.1    # 1 VR unit = 0.1m
```

iii. Trajectory step 29: *"Corridor total length = 60 VR units = 6 m (4 m texture + 2 m grey);
Texture area: positions 0-40 (4 m), where ft_CorrSpc=True; ... So 1 VR unit = 0.1 m"*.
CONVERSION_NOTES Step 2 records the same under "Spatial Structure".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The raw VR position is passed straight to `np.digitize` against the edges [10, 20, 30, 40] and the
result clipped to 0–3, giving four 1-m bins. No smoothing or interpolation. Full-dataset
distribution: 0-1 m 0.287, 1-2 m 0.229, 2-3 m 0.235, 3-4 m 0.249.

The `np.clip` is load-bearing in an unintended way: because the trial window is `round(StartFr)`
rather than the first `ft_CorrSpc` frame, ~1.4% of emitted frames have `ft_Pos` between 40 and 60
(grey space, 4–6 m) and are silently folded into the "3-4 m" bin instead of being excluded.

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]  # 4 bins of 1m each in VR units
N_POSITION_BINS = 4

def discretize_position(position, bin_edges=POSITION_BIN_EDGES):
    """4 bins of 1m each: [0-10), [10-20), [20-30), [30-40]"""
    bins = np.digitize(position, bin_edges[1:])  # 0,1,2,3
    bins = np.clip(bins, 0, N_POSITION_BINS - 1)
    return bins.astype(np.int64)
```
```python
pos_bins = discretize_position(trial_data['position'])
...
out[2, :] = trial_out['position_bin']
```

iii. The Decoder Task requires "Position in corridor discretized into 4 equal-length, 1-m-long
spatial bins". CONVERSION_NOTES Step 10 Check 2 sanity check: *"Position bins roughly equally
distributed (~25% each)"*.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed geometric thresholds at 1, 2 and 3 m (VR units 10, 20, 30), i.e. equal-length spatial bins,
exactly as the instruction specifies — not data-driven quantiles. Values ≥ 40 VR units are clipped
into the top bin; values < 0 would be clipped into bin 0.

ii.
```python
POSITION_BIN_EDGES = [0, 10, 20, 30, 40]
bins = np.digitize(position, bin_edges[1:])
bins = np.clip(bins, 0, N_POSITION_BINS - 1)
```
```python
'position_bin_edges_vr_units': POSITION_BIN_EDGES,
'position_bin_edges_meters': [e * VR_UNIT_TO_METERS for e in POSITION_BIN_EDGES],
'corridor_length_m': CORRIDOR_LENGTH_VR * VR_UNIT_TO_METERS,
```

iii. Directly from the Decoder Task spec ("4 equal-length, 1-m-long spatial bins") plus the
VR-unit-to-metre conversion established in trajectory step 29; the edges are also recorded in the
metadata so the discretisation is self-documenting.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is sampled once per imaging frame, and it is sliced with the identical
`[start_fr:end_fr]` window used for the neural columns, so bin `i` of the position output and column
`i` of `neural` are the same frame.

ii.
```python
neural   = spk[:, start_fr:end_fr].astype(np.float16)
position = beh['ft_Pos'][start_fr:end_fr].copy()
```

iii. All streams are on the imaging-frame grid, so a single slice aligns them.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed at every imaging frame.

ii.
```python
speed = beh['ft_RunSpeed'][start_fr:end_fr].copy()
```
```python
trial_speed = beh['ft_RunSpeed'][start_fr:gray_fr]   # pass 1, for the quartile edges
```

iii. Trajectory step 13: *"beh['ft_RunSpeed'] = running speed per neural frame"*.
CONVERSION_NOTES Step 5 maps "ft_RunSpeed → output[3]: speed_bin, 4 quartile bins".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A dedicated first pass over the whole dataset loads only the behaviour files, concatenates the
corridor-window speeds of every trial of every session into one pooled vector, drops NaNs, and takes
`np.percentile(..., [0,25,50,75,100])` to get **one global set of bin edges** shared by all sessions
and mice. Pass 2 then discretises each trial's speeds against those edges. The pooled speeds are
collected from all 89 recordings, including the 13 that are later skipped, and from trials that are
later rejected.

ii.
```python
def compute_speed_bin_edges(all_speeds):
    """Compute quartile bin edges for running speed."""
    flat_speeds = np.concatenate(all_speeds)
    flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
    edges = np.percentile(flat_speeds, [0, 25, 50, 75, 100])
    return edges
```
```python
# === PASS 1: Collect running speeds for quartile computation ===
for i, sess_id in enumerate(session_ids):
    speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
    if speeds is not None:
        all_speeds.extend(speeds)
speed_bin_edges = compute_speed_bin_edges(all_speeds)
```

iii. The Decoder Task asks for "Running speed discretized into 4 bins, each corresponding to 25% of
the data". A global (rather than per-session) split makes the bin labels mean the same thing in
every session. The edges are stored in the metadata (`'speed_bin_edges'`). No justification is given
for keeping the pass-1 population different from the finally converted population.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three interior percentile edges, clipped to 0–3. The computed edges are
`[-34.31, 0.0, 7.88, 29.99, 163.49]`. Because 44% of frames sit at **exactly 0** (measured in
`TX83_2022_08_31_1`) and the 25th percentile is therefore exactly 0, `np.digitize` (right-open,
`0 >= 0`) sends every zero-speed frame into bin 1. Bin 0 ends up containing **only negative
speeds**. The realised distribution is therefore **Q1 0.099 / Q2 0.394 / Q3 0.253 / Q4 0.253**,
not the 25%/25%/25%/25% the task requires and the code's own docstring claims.

ii.
```python
N_SPEED_BINS = 4

def discretize_speed(speed, bin_edges):
    """Discretize speed into quartile bins. Returns bin indices 0-3."""
    bins = np.digitize(speed, bin_edges[1:-1])  # 0,1,2,3
    bins = np.clip(bins, 0, N_SPEED_BINS - 1)
    return bins.astype(np.int64)
```
```python
Speed quartile edges: [-34.30872239   0.           7.88457407  29.99136575 163.4872794 ]
```

iii. No justification for the imbalance is given. The agent explicitly observed it — trajectory step
159: *"Speed bins: Q1 9.9%, Q2 39.4%, Q3 25.3%, Q4 25.3%"* — and CONVERSION_NOTES Step 12 records
speed_bin at 0.4675 validation balanced accuracy and labels it "Good", without revisiting the
binning. `output_values` still names the bins `['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']`.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled per imaging frame and sliced with the same `[start_fr:end_fr]` window as
the neural data, so bin `i` of the speed output and column `i` of `neural` are the same frame.

ii.
```python
neural = spk[:, start_fr:end_fr].astype(np.float16)
speed  = beh['ft_RunSpeed'][start_fr:end_fr].copy()
...
spd_bins = discretize_speed(trial_data['speed'], speed_bin_edges)
out[3, :] = trial_out['speed_bin']
```

iii. Same single-window principle as every other stream.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled:
* trials whose rounded start frame is negative, or whose end frame exceeds the number of imaged
  frames, are **dropped whole** (`if start_fr < 0 or end_fr > spk.shape[1]: return None`) — this is
  the one trial the notes mention in `TX83_2022_08_31_1`;
* trials with fewer than 2 frames are dropped; sessions with fewer than 2 trials are dropped;
* fractional `StartFr`/`GrayFr`/`LickFr` are rounded to integers;
* NaNs are removed before the speed percentiles;
* a session with no behaviour is warned about and skipped;
* `warnings.filterwarnings('ignore')` is set globally at import.

Not handled:
* the 13 recordings whose behaviour lives under `_swap1`/`_swap2` keys were skipped in the delivered
  dataset (15% of recordings), a known, documented, unrepaired gap;
* the behaviour arrays are never truncated to `spk.shape[1]` (the reference's `beh[...][:nfr]`);
  instead whole trials that run past the imaging are discarded;
* only the first behaviour variant of a recording is used (`beh_results[0]`), so with the shipped
  `load_beh_for_session` the second swap variant of a recording would be silently discarded;
* the 1000-frame cap silently truncates 54 traversals rather than flagging or dropping them.

ii.
```python
if start_fr < 0 or end_fr > spk.shape[1]:
    return None
```
```python
beh_results = load_beh_for_session(sess_id, exp_types)
if beh_results is None:
    print(f'  WARNING: No behavioral data found for {sess_id}')
    return None
```
```python
flat_speeds = flat_speeds[~np.isnan(flat_speeds)]
```
```python
import warnings
warnings.filterwarnings('ignore')
```

iii. CONVERSION_NOTES Step 9 "Notes on Missing Sessions": *"13 sessions skipped because behavioral
data is stored with _swap1/_swap2 suffixes (test3 experiments with leaf1_swap stimuli). These
sessions share neural recordings with base sessions but have different trial structures."* and
trajectory step 163: *"Given the practical constraints (250 GB file, long load times), let me focus
on making the current 76-session dataset work well rather than adding more sessions."* Step 10
Check 5 lists the rounding and the <2-frame rule as the edge cases considered.

## 12-a. What are the most time-consuming steps of the code?

i. Reading and concatenating the spike files. The per-session timing printed by the script shows
`Load:` dominating `Total:` (e.g. `Load: 91.1s, Total: 135.6s` for `VR2_2021_04_11_1`), and the full
run took 3533 s ≈ 59 min. After the spike I/O, the next largest costs are pickling the 125 GB output
and the SVD/format-verification on load in the decoder. Pass 1 (behaviour only) is cheap at 13.6 s
total.

ii.
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk)
t_load = time.time() - t0
...
print(f'  Neurons: {result["n_neurons"]}, Trials: ..., Load: {result["load_time"]:.1f}s, '
      f'Total: {t_total:.1f}s')
```
```python
with open(args.output, 'wb') as f:
    pickle.dump(data, f, protocol=4)
```

iii. The script prints per-session load/total timings precisely to expose this bottleneck
(instruction Step 6: "Print timing information to find bottlenecks"). The I/O is irreducible — every
spike file has to be read once — but the output size is a self-inflicted part of the cost: keeping
all neurons (including 12% unassigned-area) and all long trials produced a 125 GB pickle.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Several, all small relative to the spike I/O:
* the per-lick Python loop in `extract_trial_data` (a `np.round`/mask/`np.clip` assignment would do
  it in one shot);
* the two list comprehensions building `time_to_cue` and `time_since_start` per trial — these are
  `np.arange(actual_frames) / FRAME_RATE` shifted by a scalar;
* `get_training_day`, which rescans the entire `exp_info` structure once per session instead of
  building a mouse→dates table once;
* the per-trial Python loop itself; the reference likewise loops per trial, so this is comparable.

ii.
```python
for lf in trial_lick_frs:
    lf_int = int(np.round(lf))
    frame_offset = lf_int - start_fr
    if 0 <= frame_offset < actual_frames:
        licking[frame_offset] = 1.0
```
```python
time_to_cue = np.array([(i - sound_offset) / FRAME_RATE for i in range(actual_frames)],
                       dtype=np.float32)
time_since_start = np.array([i / FRAME_RATE for i in range(actual_frames)], dtype=np.float32)
```
```python
def get_training_day(mname, datexp, exp_info):
    dates = set()
    for exp_type, sessions in exp_info.items():
        for s in sessions:
            if s['mname'] == mname:
                dates.add(s['datexp'])
```

iii. CONVERSION_NOTES Step 6 claims the script is efficient but does not identify any of these; the
agent's efficiency effort went into the two-pass design and `float16` storage instead. Given the
measured per-session overhead (2–3 s of non-I/O work against 5–91 s of I/O), vectorising these would
not change the runtime materially.

## 12-c. What processing does the code repeat multiple times?

i. Four repetitions:
* **Two full passes over the behaviour**: pass 1 loads every session's behaviour purely to pool
  running speeds; pass 2 loads it all again to build the trials;
* **`Beh_<exp_type>.npy` is re-read from disk for every session**, and once per experiment type that
  the session appears under — a behaviour file holding many sessions is therefore unpickled dozens
  of times (the reference groups sessions by behaviour file and reads each file once);
* **`neu_area_ID` is computed twice per session**: once inside `load_retino` (stored as
  `neu_ar_idx`, never used) and again in `process_session`;
* **`get_training_day` rescans all of `exp_info`** once per session.

ii.
```python
# === PASS 1: Collect running speeds for quartile computation ===
speeds = process_session(sess_id, sess_info, exp_info, collect_speeds=True)
...
# === PASS 2: Processing sessions ===
result = process_session(sess_id, sess_info, exp_info, speed_bin_edges=speed_bin_edges)
```
```python
for exp_type in exp_types:
    beh_path = f'data/beh/Beh_{exp_type}.npy'
    if os.path.exists(beh_path):
        beh_all = np.load(beh_path, allow_pickle=True).item()
```
```python
def load_retino(mname, datexp, root='data/retinotopy'):
    dtrans = np.load(...)
    ix = neu_area_ID(dtrans['iarea'])
    return {'iarea': dtrans['iarea'], 'neu_ar_idx': ix}
...
    region_map = neu_area_ID(iarea)          # in process_session, recomputed
```

iii. The two-pass design is a deliberate and necessary trade-off: global speed quartiles cannot be
computed without seeing all the data first, and the agent kept pass 1 behaviour-only precisely so it
would be cheap (13.6 s measured). The repeated behaviour-file reads and the duplicated
`neu_area_ID` are not justified anywhere; they are simply artefacts of the loader's structure.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
* `load_retino` computes and returns `neu_ar_idx`, which no caller ever reads;
* `get_brain_region_indices`, `build_session_to_beh_map` and `get_stimulus_category` are defined and
  never called; `from scipy import interpolate` is imported and never used;
* pass 1 pools the running speeds of the 13 recordings that are subsequently skipped, and of trials
  that pass 2 later rejects, so the quartile edges are computed on a population that is not the
  converted population;
* `load_beh_for_session` collects *all* swap variants of a recording, then `process_session` uses
  only `beh_results[0]` and discards the rest;
* neural data is written for 483,862 neurons (12% of the total) whose retinotopic area is unassigned,
  which the reference analyses exclude, and for the stopped-animal trials of 1-e (~14% of corridor
  timepoints) — together a substantial share of the 125 GB output that carries little task signal.

ii.
```python
from scipy import interpolate                     # never used
```
```python
def get_brain_region_indices(iarea, brain_regions):   # never called
def build_session_to_beh_map(exp_info):               # never called
def get_stimulus_category(wall_name):                 # never called
```
```python
return {'iarea': dtrans['iarea'], 'neu_ar_idx': ix}   # 'neu_ar_idx' never read
```
```python
beh, beh_key = beh_results[0]                         # remaining variants discarded
```

iii. None of this is justified in CONVERSION_NOTES; the dead code and the unused `neu_ar_idx` are
leftovers from copying the reference utilities wholesale. The decision to retain the
unassigned-area neurons *is* justified indirectly (Step 10 Check 3: "No neuron filtering: Consistent
with reference code") — but it is the main reason the output pickle is as large as it is, and the
decoder never distinguishes those neurons.
