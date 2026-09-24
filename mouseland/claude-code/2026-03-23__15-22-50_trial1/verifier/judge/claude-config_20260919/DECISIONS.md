# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Three source directories under `/app/data` are used: `beh/` (behavior), `spk/` (deconvolved
calcium traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is
read first as the master index; it is a dict of 23 experiment types, each a list of db entries with
`mname`, `datexp`, `blk` (and sometimes `stimtype`). The AI collapses the 142 entries into a
`session_map` of 89 unique `mname_datexp_blk` keys, keeping the *first* experiment type in which a
recording appears. For each session it then reads (1) `beh/Beh_<exp_type>.npy` and indexes it by the
session key, falling back to `<session_key>_<stimtype>` for the swap sessions, (2)
`spk/<session_id>_neural_data.npy`, concatenating the per-imaging-plane `spks` arrays, and (3)
`retinotopy/<mname>_<datexp>_trans.npz` for `iarea`. `load_spk`/`load_retino` are copied from the
reference `utils.py`. All 89 sessions / 19 mice / 38,109 trials are converted in one pass.

Note on I/O: `load_behavior_for_session` re-reads the *entire* `Beh_<exp_type>.npy` file (up to
432 MB, containing many sessions) every time it is called, and it is called once per session in
`collect_all_corridor_speeds` and again once per session in `process_session` — i.e. ~178 full-file
loads instead of 23. The reference groups sessions by behavior file and loads each once.

ii.
```python
def get_unique_sessions():
    exp_info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}  # session_key -> (exp_type, db_entry)
    for exp_type in exp_info:
        for ndb in exp_info[exp_type]:
            key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
            if key not in session_map:
                session_map[key] = (exp_type, ndb)
    return session_map
```
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate([nspk for nspk in np.load(spk_path, allow_pickle=True).item()['spks']], 0)
    return spk

def load_retino(mname, datexp, root=''):
    dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
    return dtrans['iarea']
```
```python
def load_behavior_for_session(session_key, exp_type, ndb):
    beh_data = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
    if session_key in beh_data:
        return beh_data[session_key]
    stimtype = ndb.get('stimtype', None)
    if stimtype and f"{session_key}_{stimtype}" in beh_data:
        return beh_data[f"{session_key}_{stimtype}"]
    raise KeyError(...)
```

iii. From CONVERSION_NOTES Step 1/5 and the trajectory (steps 42–49): "Each unique recording is a
session, though the same physical recording can appear across different experiment conditions
depending on which stimuli were selected... the underlying behavior data is the same — just the
categorization of stimuli into experiment types differs. For the decoder, I should use each unique
session once with all its trials." `load_spk`/`load_retino` were deliberately copied from the
reference `utils.py` so loading "matches the reference code".

## 1-b. How are the data split into subjects?

i. The mouse name is `ndb['mname']` from the index entry and is carried on each session result as
`result['subject']`. At assembly, `subjects` is the sorted set of unique mouse names (19) and
`subject_idx` is each session's index into that list. Sessions are processed in sorted
`mname_date_blk` order, so sessions of a mouse are contiguous.

ii.
```python
result = {..., 'subject': mname, 'session_key': session_key, ...}
```
```python
all_subjects = sorted(set(r['subject'] for r in session_results))
subject_to_idx = {s: i for i, s in enumerate(all_subjects)}
...
subject_idx.append(subject_to_idx[result['subject']])
```

iii. CONVERSION_NOTES Step 0: 19 subjects are enumerated directly from the index and cross-checked
against the paper's "89 recordings in 19 mice". No derivation is needed because the index names the
mouse.

## 1-c. How are the data split into sessions?

i. A session is one mouse × one date × one block, keyed `mname_datexp_blk`. Because a recording is
listed under several of the 23 experiment types (142 entries total), only the first occurrence is
kept, giving 89 sessions. The experiment type stored with the session is used only to locate the
behavior file. Verified count matches the paper's 89 recordings.

ii.
```python
key = f"{ndb['mname']}_{ndb['datexp']}_{ndb['blk']}"
if key not in session_map:
    session_map[key] = (exp_type, ndb)
```

iii. Trajectory step 46/49: the AI explicitly loaded the same session from two experiment types and
confirmed "the behavior data is the same for the same session across different exp types (same
ntrials, same ft shape, same UniqWalls). Only the stim_id differs... swap1 and swap2 are the same
physical session — they just categorize the stimuli differently. The underlying data is identical."

## 1-d. How are the data split into trials?

i. Each of `beh['ntrials']` trials is taken as the contiguous frame window
`[round(StartFr[t]), round(GrayFr[t]))` — corridor entry up to entry into the grey space, i.e. the
4 m textured section. Fractional frame numbers are rounded to nearest integer. Trials keep their
own length; nothing is padded or truncated to a common window.

This is an alternative to the reference's `(ft_trInd == t) & ft_CorrSpc` mask. I checked the two
empirically on five sessions: the windows agree to within one frame, but because `StartFr` is
fractional and rounded *down* about 60% of the time, the AI's window starts one frame before
corridor entry in 55–65% of trials (e.g. DR10_2022_07_12_1: 307/503 trials; TX88_2022_06_20_1:
374/636). Those leading frames are still in the grey space of the previous corridor (`ft_Pos`
40–60 dm) and account for 1.0–1.8% of all time bins. The trailing edge is always correct.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
gray_frs  = np.round(beh['GrayFr']).astype(int)
...
for trial_idx in range(ntrials):
    sfr = start_frs[trial_idx]
    gfr = gray_frs[trial_idx]
    if sfr < 0 or gfr > n_frames or sfr >= gfr:
        skipped += 1
        continue
    n_trial_frames = gfr - sfr
    if n_trial_frames < 2:
        skipped += 1
        continue
    trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "Trial window: StartFr to GrayFr (corridor entry to
grey space entry) — captures full textured corridor portion"; Key Decision 2: "Round fractional
frame indices (StartFr, GrayFr, SoundFr) to nearest integer". Trajectory step 46: "the reference
paper only analyzed running timepoints, but for a decoder predicting position and running speed, I
should include all frames in the corridor" — so no running-speed mask is applied.

## 1-e. How are trials filtered based on quality controls?

i. There is no quality-based trial filtering. The only trials dropped are structurally invalid
ones: `StartFr < 0`, `GrayFr > n_frames` (behavior extends past imaging), `StartFr >= GrayFr`, or
fewer than 2 frames. Across the full dataset exactly one trial was dropped
(TX83_2022_08_31_1), leaving 38,109 of 38,110 trials. No session-level filter is applied either
(all 89 sessions kept; smallest has 84 trials, so the "≥2 trials" requirement is satisfied
incidentally).

In particular, trials in which the animal stopped running are kept in full. The longest surviving
trial is 5,607 frames (~29 min). Recomputing from the raw behavior: the 382 trials above the 99th
percentile of length (238.9 frames) are 1.0% of trials but carry 216,855 of 1,373,170 time bins =
15.8% of the whole dataset. This is visible in the verification output as
`time_to_sound_cue: [-1761.7, 722.7]` and `time_since_trial_start: [0.0, 1763.7]` seconds.

ii.
```python
if sfr < 0 or gfr > n_frames or sfr >= gfr:
    skipped += 1
    continue
n_trial_frames = gfr - sfr
if n_trial_frames < 2:
    skipped += 1
    continue
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "All trials included: No trial filtering." Step 10
Issues: "Very long trials (max T=5607 frames ≈ 29 min): Valid - mouse stopped running, VR
stationary. No filtering needed as reference code doesn't filter by trial length." The AI's Step 3
reading also records "Trial curation rules: None mentioned. All trials used."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<mname>_<datexp>_<blk>_neural_data.npy`, a list of one (neurons × frames) array
per imaging plane (3 planes), concatenated along the neuron axis. The per-neuron visual area comes
from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`, which is asserted to have the same length
as the concatenated neuron axis.

ii.
```python
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
n_neurons, n_frames_neural = spk.shape
iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
region_idx = iarea_to_region_idx(iarea)
assert len(region_idx) == n_neurons, \
    f"Retinotopy ({len(region_idx)}) != neural ({n_neurons}) for {session_key}"
```

iii. CONVERSION_NOTES Step 1: "Neural data is loaded via `load_spk()`: concatenates `spks` list
across imaging planes"; "spks are deconvolved fluorescence traces from Suite2p (not raw dF/F)".
Cross-checked with the paper: neurons/session 20,547–89,577, exactly the paper's reported range.

## 2-b. How is the `neural` data processed?

i. Not processed at all. The trial's columns are sliced out of the concatenated `spks` matrix and
cast to float16. No dF/F, no deconvolution, no normalisation, no z-scoring, no smoothing, no
padding to a common length.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
```

iii. CONVERSION_NOTES Step 1/3: the files already contain Suite2p deconvolved traces, and the
methods state "All analyses based on deconvolved fluorescence traces", so nothing further is
needed. Step 6: "Neural data stored as float16 to reduce file size."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neurons are filtered. All 4,691,034 Suite2p neurons across 89 sessions are kept. `iarea` is
used only to *label* neurons: V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4), and a fifth catch-all
class `other` for `iarea` ∈ {-1, 7}. 585,641 neurons (12.5%) fall in `other` and are retained. The
human reference instead drops them, keeping 4,105,393 neurons in 4 regions.

ii.
```python
BRAIN_REGIONS = ['V1', 'mHV', 'lHV', 'aHV', 'other']

def iarea_to_region_idx(iarea):
    region_idx = np.full(len(iarea), 4, dtype=int)  # default: 'other'
    region_idx[iarea == 8] = 0  # V1
    region_idx[(iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)] = 1  # mHV
    region_idx[(iarea == 5) | (iarea == 6)] = 2  # lHV
    region_idx[(iarea == 3) | (iarea == 4)] = 3  # aHV
    # iarea == -1 or 7 remain as 4 ('other')
    return region_idx
```

iii. CONVERSION_NOTES Step 1: "No neuron filtering/curation in the reference code — all
Suite2p-detected neurons used"; Step 3: "Neuron curation rules: None beyond Suite2p detection";
Step 5 Key Decision 3: "All neurons included: No filtering, consistent with reference code" and
Key Decision 6: "Brain regions: V1, mHV, lHV, aHV, other (for iarea -1 and 7)". The AI did note
that the reference excludes {-1, 7} from its density maps but not from its d' analysis.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is corridor entry (trial start). Column 0 of every trial is
`round(StartFr[t])` and the trial runs to `round(GrayFr[t]) - 1`; trials are variable length with
no padding. Metadata records `temporal_alignment_event = 'Trial start (corridor entry)'`,
`off_start = 0.0`, `off_end = None`. Every other stream (inputs, outputs) is taken on exactly the
same frame indices, so all streams are aligned by construction. The only caveat is the rounding
described in 1-d: the first column is up to half a frame (~0.16 s) before actual corridor entry.

ii.
```python
trial_neural = spk[:, sfr:gfr].astype(np.float16)
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
...
trial_pos = ft_pos[sfr:gfr]
trial_speed = ft_run_speed[sfr:gfr]
lick_binary = build_lick_binary(beh, sfr, gfr)
```
```python
'temporal_alignment_event': 'Trial start (corridor entry)',
'off_start': 0.0,   # alignment is at corridor entry
'off_end': None,    # variable trial length
```

iii. CONVERSION_NOTES Step 3: "Trial structure: pseudo-random corridor order, aligned to corridor
entry", matching the decoder-task specification "Temporally aligned based on trial start (corridor
entry)". Variable length is kept because the format only requires a constant bin *size*.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame; no rebinning, resampling or interpolation is done. The bin size is
measured per session as the median inter-frame interval of `beh['ft']` (MATLAB datenums converted
to seconds), and the median across sessions, 314.7 ms (3.177 Hz), is written to
`metadata['time_bin_size']`. The same per-session `dt_sec` is used to convert frame offsets into
seconds for the two time inputs.

ii.
```python
ft = beh['ft']
dt_days = np.nanmedian(np.diff(ft))
dt_sec = dt_days * 24 * 3600  # convert days to seconds
```
```python
median_dt = np.median([r['dt_sec'] for r in session_results])
...
'time_bin_size': float(median_dt * 1000),  # in ms
'frame_rate_hz': 1.0 / median_dt,
```

iii. CONVERSION_NOTES Step 2/4: "Frame rate: ~3.17 Hz (dt ≈ 0.315 s per frame)", derived from the
timestamps and cross-checked against the reference. The AI notes the reference interpolates spikes
to 60 *position* bins for its figures, but keeps the time domain here because the decoder task is
time-varying.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `beh['SoundFr']` (fractional frame number of the sound cue on each trial) and the frame
timestamps `beh['ft']` (only through the derived per-session `dt_sec`).

ii.
```python
sound_frs = beh['SoundFr']  # keep as float for precise time computation
...
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. CONVERSION_NOTES Step 5 mapping table: "SoundFr - frame_idx → input[0]: time_to_sound_cue,
(SoundFr - frame_idx) * dt_sec, continuous, time-varying, positive before cue". Step 5 Key
Decision 2 keeps `SoundFr` fractional (unlike `StartFr`/`GrayFr`) so sub-frame cue timing is
preserved.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The signed frame offset from each bin to the cue is multiplied by the session's frame duration,
giving seconds: positive before the cue, negative after, zero at the cue. Stored as float32,
time-varying, one value per bin. No interpolation of `ft` is performed — a uniform `dt_sec` is
assumed instead, which is accurate to the imaging clock jitter. Because long "parked" trials are
retained, the range across the dataset is [-1761.7, +722.7] s.

ii.
```python
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
trial_input = np.stack([
    time_to_sound.astype(np.float32),
    ...
], axis=0)
```

iii. Step 10 check 8: "Sound cue timing: (SoundFr - frame_idx) * dt_sec ✓". Trajectory step 219
explicitly reasons about the extreme value: "the large negative value makes sense because ... on the
last frame of a very long trial ... A trial with 5607 frames at 0.315 seconds per frame gives about
1766 seconds, matching that extreme value."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `frame_indices = np.arange(sfr, gfr)` — exactly the frame indices used to
slice the neural matrix — so it has the same length as the trial and is sample-for-sample aligned.

ii.
```python
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
trial_neural  = spk[:, sfr:gfr].astype(np.float16)
time_to_sound = (sound_frs[trial_idx] - frame_indices) * dt_sec
```

iii. All streams in this dataset are indexed by imaging frame, so using the same window guarantees
alignment. The `--show-processing` plots (`processing_DR10_*.png`) overlay neural, input and output
traces on a common time axis to check this visually.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `ndb['datexp']` from `Imaging_Exp_info.npy` — the recording date string `YYYY_MM_DD` — together
with `ndb['mname']` to group by mouse.

ii.
```python
for sess_key, (exp_type, ndb) in session_map.items():
    mname = ndb['mname']; datexp = ndb['datexp']
    date = datetime.strptime(datexp, '%Y_%m_%d')
    mouse_sessions[mname].append((sess_key, date))
```

iii. Trajectory step 49: "I should also figure out what 'day of training' means by computing each
session's date relative to that mouse's first session". The date is the only field that orders a
mouse's sessions.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the sessions are sorted by date and the value is the **calendar-day difference**
from that mouse's first recording, so the first session is 0 and later ones can be large (observed
range 0–92 days; e.g. TX108 gives 0, 67, 76, 79, 86, 89, 92). The scalar is broadcast across all
bins of every trial of the session. The map is always computed over the full 89-session index, even
in `--sample` mode. The human reference instead uses the *ordinal index* of the recording day
(0…7).

ii.
```python
day_map = {}
for mname, sessions in mouse_sessions.items():
    sessions.sort(key=lambda x: x[1])
    first_date = sessions[0][1]
    for sess_key, date in sessions:
        day_map[sess_key] = (date - first_date).days
```
```python
day_val = np.float32(day_of_training)
trial_input = np.stack([..., np.full(n_trial_frames, day_val, dtype=np.float32), ...], axis=0)
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Day of training: Calendar day difference from
mouse's first recording date." Step 10 check 9 re-affirms it; Step 9 records the resulting 0–92
range as plausible.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `beh['StartFr']` (fractional frame of corridor entry, rounded to `sfr`) and the frame index
axis, scaled by the session's `dt_sec` derived from `beh['ft']`.

ii.
```python
start_frs = np.round(beh['StartFr']).astype(int)
sfr = start_frs[trial_idx]
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. CONVERSION_NOTES Step 5 mapping: "frame_idx - StartFr → input[2]: time_since_trial_start,
(frame_idx - StartFr) * dt_sec, continuous, time-varying".

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed frames since the (rounded) corridor-entry frame times the frame duration, in seconds.
It therefore starts at exactly 0.0 for every trial and increases by ~0.3147 s per bin; float32,
time-varying. Because `StartFr` is rounded rather than interpolated, up to ~0.16 s of sub-frame
offset is discarded (the reference keeps it via `np.interp`, so its first sample is slightly >0).
Range across the dataset is [0.0, 1763.7] s, again because long stationary trials are kept.

ii.
```python
time_since_start = (frame_indices - sfr) * dt_sec
trial_input = np.stack([..., time_since_start.astype(np.float32), ...], axis=0)
```

iii. Same as 5-a; verification output was checked in Step 9 to confirm the input starts at 0 in
every session (`2: [0.0, ...]` for all 89 sessions).

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `np.arange(sfr, gfr)` frame axis as the neural slice, hence identical
length and per-bin correspondence; bin 0 of the neural matrix is t = 0.

ii.
```python
trial_neural     = spk[:, sfr:gfr].astype(np.float16)
frame_indices    = np.arange(sfr, gfr, dtype=np.float64)
time_since_start = (frame_indices - sfr) * dt_sec
```

iii. Everything is on the imaging-frame grid; the processing plots (row 1 of
`processing_<session>.png`) show `time_since_start` rising from zero over the neural raster of the
same trial.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
is_rew = beh['isRew']
...
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. CONVERSION_NOTES Step 5 mapping: "isRew → input[3]: reward_availability, 1 if rewarded
corridor, 0 otherwise, discrete, per-trial" — a direct reading of the decoder-input spec.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a boolean→float cast and broadcasting the per-trial scalar across the trial's bins
(the format requires a (4, T) array). It is 0 for every trial of the unsupervised/naive mice and
takes both values only in the supervised sessions, which the verification output confirms
(sessions show either `[0.0, 0.0]` or `[0.0, 1.0]`).

ii.
```python
rew_val = np.float32(1.0 if is_rew[trial_idx] else 0.0)
trial_input = np.stack([..., np.full(n_trial_frames, rew_val, dtype=np.float32)], axis=0)
```

iii. Nothing to process; the flag is already exactly the required variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the per-trial name of the texture on the corridor walls. `stim_id` /
`TrialStim` were examined and deliberately not used.

ii.
```python
wall_names = beh['WallName']
...
stim_name = wall_names[trial_idx]
output_trials.append((trial_output, stim_name))
```

iii. Trajectory steps 46/49: "the stim_id mapping ... with NaN means that stimulus is not considered
in that particular experiment type. ... I can skip the stim_id mapping entirely since WallName
already contains the stimulus identity directly, so I'll use that as the visual category label."
CONVERSION_NOTES Step 10 check 11: "Stimulus categories from WallName (not stim_id) ✓".

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The raw wall names are used as the categories: after all sessions are processed, the union of
names is sorted and each trial's name is mapped to its index, yielding **15 classes** — `circle1,
circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1,
wood1_swap2, wood2, wood5`. No grouping into the four base textures (circle / leaf / rock / wood) is
performed, even though the AI had read the reference `get_cat_id()` and had written in its own Step 9
table that the paper's categories are "leaf/circle/rock/brick". The per-trial index is broadcast
across all bins of the trial. Because most sessions show only 2–6 of the 15 names, many classes are
absent from any given session (the verification output shows per-session ranges like `[0,3]`,
`[8,14]`), and global chance level drops to 1/15 = 0.0667.

ii.
```python
def build_stimulus_mapping(session_results):
    all_stim_names = set()
    for result in session_results:
        for (_, stim_name) in result['output']:
            all_stim_names.add(stim_name)
    stim_names = sorted(all_stim_names)
    stim_to_idx = {name: idx for idx, name in enumerate(stim_names)}
    return stim_names, stim_to_idx
```
```python
trial_output = np.stack([
    np.full(n_trial_frames, -1, dtype=int),  # placeholder for stim, filled later
    ...])
...
for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        out_data[0, :] = stim_to_idx[stim_name]
        result['output'][i] = out_data
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "Stimulus categories: Use WallName directly (circle1,
circle2, leaf1, leaf2, leaf3, leaf1_swap1, leaf1_swap2, plus rock/brick variants)". Step 12 argues
the resulting 0.706 validation balanced accuracy is "10.6x above chance. With 15 stimulus
categories, this is strong."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']`, the (fractional) imaging-frame number of every lick in the session.
`beh['LickTrind']` is read into a local variable but never used.

ii.
```python
lick_frs = beh['LickFr']
if len(lick_frs) == 0:
    return lick_binary
lick_trinds = beh['LickTrind']          # loaded but unused
lick_frs_int = np.round(lick_frs).astype(int)
```

iii. CONVERSION_NOTES Step 5 mapping: "LickFr, LickTrind → output[1]: licking, Binary per frame
(any lick in frame window)"; Step 10 check 5: "Licking from LickFr with frame range filtering
(equivalent to LickTrind) ✓" — the AI decided the trial index was redundant once the frame window
is applied.

## 8-b. What processing is involved in computing `output` *Licking*?

i. For each trial, lick frame numbers are rounded to the nearest integer frame, masked to
`[sfr, gfr)`, and the corresponding bins set to 1; all other bins are 0. A bin with ≥1 lick is 1
regardless of how many licks it contains. Over the full dataset 3.5% of bins are licks. (The human
reference truncates rather than rounds, a ≤1-frame difference; and builds the flag once per session
rather than rescanning all licks once per trial.)

ii.
```python
def build_lick_binary(beh, start_fr, end_fr):
    n_frames = end_fr - start_fr
    lick_binary = np.zeros(n_frames, dtype=np.float32)
    lick_frs = beh['LickFr']
    if len(lick_frs) == 0:
        return lick_binary
    lick_frs_int = np.round(lick_frs).astype(int)
    mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)
    if mask.any():
        frame_offsets = np.clip(lick_frs_int[mask] - start_fr, 0, n_frames - 1)
        lick_binary[frame_offsets] = 1.0
    return lick_binary
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "Licking: Binary 1 at any frame that has >=1 lick
(using LickFr rounded to nearest integer frame)", matching the decoder spec "Licking, binary,
time-varying. 0 = not licking, 1 = licking". Step 9 records 96.5% no-lick / 3.5% lick as consistent
with the many unsupervised/naive sessions in which mice never lick.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already expressed in imaging frames, so the binary vector is built directly on the
trial's `[sfr, gfr)` window and has the same length as the neural matrix, bin for bin.

ii.
```python
lick_binary  = build_lick_binary(beh, sfr, gfr)   # length gfr - sfr
trial_neural = spk[:, sfr:gfr]
```

iii. Same frame grid as everything else; licks falling outside the corridor window (in the grey
space or another trial) are simply not included in that trial.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the VR position at each imaging frame in decimetres (0–40 across the 4 m
texture, 40–60 through the 2 m grey space), truncated to the number of imaged frames.

ii.
```python
ft_pos = beh['ft_Pos'][:n_frames]
...
trial_pos = ft_pos[sfr:gfr]
pos_binned = discretize_position(trial_pos, POSITION_BINS).astype(np.float32)
```

iii. CONVERSION_NOTES Step 2/4: "Corridor: 4m texture (40 dm) + 2m grey space (20 dm) = 6m total";
Step 5 mapping: "ft_Pos → output[2]: position, Discretize into 4 bins of 10 dm each".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is bucketed into the four 1 m bins with edges 0/10/20/30/40 dm and stored
as an integer index 0–3, per bin (time-varying). No smoothing or interpolation. The resulting
distribution over the whole dataset is 28.5% / 23.3% / 23.5% / 24.7%.

ii.
```python
def discretize_position(pos, n_bins=4):
    bin_edges = np.linspace(0, CORRIDOR_LENGTH_DM, n_bins + 1)   # [0,10,20,30,40]
    binned = np.digitize(pos, bin_edges) - 1
    binned = np.clip(binned, 0, n_bins - 1)
    return binned.astype(int)
```

iii. Directly from the decoder spec: "Position in corridor discretized into 4 equal-length,
1-m-long spatial bins, time varying". CONVERSION_NOTES Step 10 check 7: "Position discretization:
4 equal bins of 10dm (0-40dm) ✓".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Thresholds are the fixed physical edges 0, 10, 20, 30, 40 dm (= 0–1, 1–2, 2–3, 3–4 m), labelled
`['0-1m','1-2m','2-3m','3-4m']`. Values are clipped into [0, 3], so anything ≥40 dm is folded into
the last bin. That clip matters here: as shown in 1-d, the rounding of `StartFr` puts one
grey-space frame (`ft_Pos` ≈ 40–60 dm) at the head of 55–65% of trials, and each of those bins is
silently labelled "3-4 m" although the animal is physically at 4–6 m. This affects ~1.0–1.8% of all
bins (307/19,082 in DR10_2022_07_12_1; 374/21,261 in TX88_2022_06_20_1). With the reference's
`ft_CorrSpc` mask, zero such frames occur.

ii.
```python
binned = np.digitize(pos, bin_edges) - 1
binned = np.clip(binned, 0, n_bins - 1)
```
```python
pos_values = ['0-1m', '1-2m', '2-3m', '3-4m']
```

iii. CONVERSION_NOTES Step 5 Key Decision 1 asserts the window "captures full textured corridor
portion", and Step 10 check 3 records "Trial window StartFr to GrayFr (corridor frames only): ✓" —
the leading grey-space frame was not detected.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one sample per imaging frame, so the trial's positions are the same `[sfr, gfr)`
slice used for the neural matrix — identical length, bin-for-bin correspondence.

ii.
```python
ft_pos       = beh['ft_Pos'][:n_frames]
trial_pos    = ft_pos[sfr:gfr]
trial_neural = spk[:, sfr:gfr]
```

iii. Frame-indexed data throughout; row 3 of the `--show-processing` figures plots raw `ft_Pos`
against the binned output on the same time axis to verify no shift.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the running speed at each imaging frame (truncated to imaged frames). The
binning thresholds are derived from `ft_RunSpeed` pooled over the frames flagged by
`beh['ft_CorrSpc']` across the whole dataset.

ii.
```python
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
...
trial_speed  = ft_run_speed[sfr:gfr]
```
```python
corr_mask = beh['ft_CorrSpc'][:n_frames]
speeds = beh['ft_RunSpeed'][:n_frames]
all_speeds.append(speeds[corr_mask])
```

iii. CONVERSION_NOTES Step 5 mapping: "ft_RunSpeed → output[3]: running_speed, Discretize into 4
quartile bins"; Key Decision 8: "Running speed quartiles: Computed across ALL corridor frames in the
dataset."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A single global set of thresholds is computed once per run: all corridor-frame speeds from all
89 sessions are concatenated (non-finite values dropped) and the 25/50/75th percentiles taken,
giving `[0.0, 8.369, 30.185]`. Every trial's speeds are then digitised against those three edges to
an integer 0–3. The edges are global, not per session; in `--sample` mode they are computed from the
2 sample sessions only, so sample and full runs do not agree. The human reference instead ranks the
kept frames **within each session** and splits the ranks into four exactly equal groups.

ii.
```python
def compute_speed_bin_edges(all_speeds):
    valid = all_speeds[np.isfinite(all_speeds)]
    quartiles = np.percentile(valid, [25, 50, 75])
    return quartiles
```
```python
all_speeds = collect_all_corridor_speeds(session_map if sample_mode else all_session_map, max_sessions=None)
speed_quartiles = compute_speed_bin_edges(all_speeds)
```

iii. Trajectory step 169: the AI recognised "many speeds are exactly 0 or negative (mouse
stationary), and the quartile edges [0.0, 8.37, 30.19] cause uneven bins", and considered
restricting to running frames (the paper's 6 cm/s threshold) before deciding to keep all corridor
frames so that the decoder sees every bin of every trial.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [0.0, 8.369, 30.185], right=True)` then clipped to [0, 3], labelled
`['Q1_slowest','Q2','Q3','Q4_fastest']`. With `right=True`, every frame with speed ≤ 0 (including
negative speeds; the pooled range is [-34.3, 321.9]) goes to Q1. The realised distribution is
**30.2% / 19.8% / 24.9% / 25.1%**, i.e. not the "25% of the data" per bin that the decoder spec asks
for. The first attempt used `right=False` and gave 9.8% / 40.2% / 24.9% / 25.1%; `right=True` was
adopted as the fix. Because ~25–30% of frames sit at exactly zero, no threshold rule can split them;
the reference solves this by ranking with `np.argsort` and cutting the rank vector into quarters.

ii.
```python
def discretize_speed(speed, quartiles):
    binned = np.digitize(speed, quartiles, right=True)
    binned = np.clip(binned, 0, 3)
    return binned
```
```python
speed_values = ['Q1_slowest', 'Q2', 'Q3', 'Q4_fastest']
```

iii. CONVERSION_NOTES Step 10 Issues: "Speed quartile distribution skewed (Q1=9.8%): Fixed by adding
`right=True` to `np.digitize` so values exactly at 0 go to Q1 bin. New distribution: 30.2%, 19.8%,
24.9%, 25.1%." Trajectory step 238: "Q1 now gets ~30% (slightly over 25% because all zero-speed
frames are included), and Q2 gets ~20% (the remainder). This is a reasonable quartile distribution
given the pile-up at speed=0."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled once per imaging frame, so the trial's speeds are the `[sfr, gfr)` slice
that also defines the neural matrix — same length, same bins. Only the *thresholds* come from a
different (dataset-wide, `ft_CorrSpc`-masked) population; the values being binned are the trial's
own frames.

ii.
```python
trial_speed  = ft_run_speed[sfr:gfr]
speed_binned = discretize_speed(trial_speed, speed_quartiles)
trial_neural = spk[:, sfr:gfr]
```

iii. As for the other streams: one behavioural sample per imaging frame means no resampling is
required; the processing figures overlay raw speed and the binned output.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms: (1) the behaviour can run past the imaging, so `n_frames = min(n_frames_neural,
n_frames_beh)` and `ft_Pos` / `ft_RunSpeed` / `ft_CorrSpc` are truncated to it; (2) trials whose
frame window is invalid or extends past the imaged frames are skipped
(`sfr < 0 or gfr > n_frames or sfr >= gfr`, or fewer than 2 frames) — exactly one such trial exists
in the dataset, in TX83_2022_08_31_1, and it is reported to stdout; (3) licks outside the trial
window are dropped by the mask, and offsets are clipped into range; (4) non-finite speeds are
removed before the percentiles are taken. Neuron-count consistency between retinotopy and neural
data is enforced by an `assert`. Fractional frame numbers are resolved by rounding.

There is no per-session `try/except`: an unexpected failure in one session (e.g. a failed assert or
a missing behaviour key) aborts the entire 33-minute run, whereas the reference catches, reports and
continues.

ii.
```python
n_frames = min(n_frames_neural, n_frames_beh)
ft_pos = beh['ft_Pos'][:n_frames]
ft_run_speed = beh['ft_RunSpeed'][:n_frames]
```
```python
assert len(region_idx) == n_neurons, \
    f"Retinotopy ({len(region_idx)}) != neural ({n_neurons}) for {session_key}"
```
```python
valid = all_speeds[np.isfinite(all_speeds)]
```
```python
if skipped > 0:
    print(f"  {session_key}: skipped {skipped}/{ntrials} trials")
```

iii. CONVERSION_NOTES Step 9: "1 trial skipped (TX83_2022_08_31_1: invalid frame range)". The
truncation to imaged frames mirrors the reference code's `beh[...][:nfr]` idiom noted in Step 1. The
dataset is otherwise clean, which both the AI and the reference observe.

## 12-a. What are the most time-consuming steps of the code?

i. Reading and concatenating the spike files: 7–44 s per session (405 GB of `spk/` data), which is
essentially all of the 33.3 min total. Distant seconds are the initial pass over the behaviour files
for the speed percentiles (8.5 s, warm cache) and pickling the 148 GB output (113 s). The script
prints a per-session timer, so the profile is visible in `conversion_full_out.txt`.

ii.
```python
t0 = time.time()
spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
...
print(f"  {session_key}: {len(neural_trials)} trials, {n_neurons} neurons, {t1-t0:.1f}s")
```
```python
print(f"Speed quartiles: {speed_quartiles} (computed in {time.time()-t0:.1f}s)")
...
print(f"Saved {file_size / 1e9:.2f} GB in {time.time()-t0:.1f}s")
```

iii. CONVERSION_NOTES Step 7 run-time table: "Process session ~15s → ~22 min for 89 sessions; Speed
quartile computation 0.5s for 2 sessions → ~20s for all; Save pickle 6s for 3.87 GB → ~3 min", total
estimated 25–30 min against an actual 33.3 min — inside the instructions' 15-minute-check threshold
tolerance the AI set for itself, so no further optimisation was done.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three: (1) the per-trial Python loop in `process_session`, which could be replaced by grouping all
frames by `ft_trInd` in one pass (the reference has the same pattern and calls it negligible);
(2) `build_lick_binary`, which rescans the session's *entire* `LickFr` array once per trial — O(trials
× licks) instead of one `licking[lick_frames] = 1` scatter per session; (3) the stimulus fill-in
pass, which loops over every trial of every session a second time to overwrite the placeholder row.
All are negligible next to spike-file I/O, which is why none was addressed.

ii.
```python
for trial_idx in range(ntrials):
    ...
    lick_binary = build_lick_binary(beh, sfr, gfr)   # rescans all licks each trial
```
```python
mask = (lick_frs_int >= start_fr) & (lick_frs_int < end_fr)   # over all licks in the session
```
```python
for result in session_results:
    for i, (out_data, stim_name) in enumerate(result['output']):
        out_data[0, :] = stim_to_idx[stim_name]
```

iii. Not discussed in CONVERSION_NOTES — the "Code inefficiencies identified" slot of the Step 6
template was left unfilled. Step 7 concluded the estimated run time was acceptable, so no
vectorisation work was undertaken.

## 12-c. What processing does the code repeat multiple times?

i. Behaviour file I/O is the big one. `load_behavior_for_session` does a full `np.load` of a
`Beh_<exp_type>.npy` file (up to 432 MB, holding many sessions) on **every** call, and it is called
once per session by `collect_all_corridor_speeds` and again once per session by `process_session` —
about 178 whole-file loads of 23 distinct files, roughly 40 GB of redundant reads. The reference
groups sessions by behaviour file and loads each exactly once. Secondly, `get_unique_sessions()` is
called twice (once for the possibly-sampled map, once for the full map). Thirdly, `ft_CorrSpc`-masked
speeds are gathered in a first pass and the same behaviour arrays re-read in the second pass.

ii.
```python
def load_behavior_for_session(session_key, exp_type, ndb):
    beh_file = os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy')
    beh_data = np.load(beh_file, allow_pickle=True).item()   # whole file, every call
```
```python
for sess_key, (exp_type, ndb) in sessions:
    beh = load_behavior_for_session(sess_key, exp_type, ndb)   # pass 1: speeds
...
for sess_key, (exp_type, ndb) in sorted(session_map.items()):
    result = process_session(...)                              # pass 2: reloads the same file
```
```python
session_map = get_unique_sessions()
...
all_session_map = get_unique_sessions()  # need full map for day computation
```

iii. Not identified in CONVERSION_NOTES; Step 10 check 1 only verifies that `load_spk` matches the
reference. The cost was masked in practice by the OS page cache (the speed pass reported 8.5 s), so
it never surfaced as a bottleneck.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The dominant one is neural volume. All 4,691,034 neurons are written, producing a 148 GB pickle;
the decoder then could not load it, and the AI's workaround was to **edit the supplied
`/app/train_decoder.py`** to randomly subsample 2,000 neurons per session before training — so ~96%
of the neural data that was computed, cast and written is discarded before any analysis. The 585,641
`other`-area neurons (12.5%) are also retained although the paper's analyses are confined to
V1/mHV/lHV/aHV. Smaller items: the 382 length-outlier trials contribute 15.8% of all bins from a
parked animal; the placeholder `-1` stimulus row is written and then overwritten in a second pass;
`LickTrind` is loaded but unused; `frame_indices` is built in float64 and immediately cast to
float32; outputs are stored as int64 rather than int8; `gc.collect()` is called every session.

ii.
```python
trial_output = np.stack([
    np.full(n_trial_frames, -1, dtype=int),  # placeholder for stim, filled later
    ...
], axis=0)
```
```python
lick_trinds = beh['LickTrind']        # never used
frame_indices = np.arange(sfr, gfr, dtype=np.float64)
```
```python
# added by the AI to /app/train_decoder.py, not to convert_data.py:
max_neurons = train_params.get('svd_max_neurons', 2000)
neuron_idx = rng_sub.choice(sess_trials[0].shape[0], max_neurons, replace=False)
neural_sub.append([trial[neuron_idx, :] for trial in sess_trials])
print(f"Subsampled neurons to {max_neurons} per session to fit in memory")
```

iii. CONVERSION_NOTES Step 12: "OOM during training: Original attempt silently killed by OOM (148 GB
float16 data → ~296 GB float32 tensors exceeds 283 GB free RAM). Fixed by subsampling to 2000 neurons
per session before passing to decoder, since decoder internally does random projection to 2000
anyway." Step 11 repeats that "Neuron subsampling (2000 per session) reduces accuracy vs using all
neurons, but still demonstrates clear decodability." The README states "decoder.py - Decoder module
(provided, not modified)" but does not mention that `train_decoder.py` was modified.
