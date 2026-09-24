# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the master index `data/beh/Imaging_Exp_info.npy`, which lists every recording grouped by
experiment type, and flattens it into one record per unique recording (`session_id = mname_datexp_blk`),
keeping the first experiment type in which a recording appears. For each session it then reads three
things: the deconvolved traces `data/spk/<session_id>_neural_data.npy` (a list of per-plane arrays that
are concatenated), the retinotopy `data/retinotopy/<mname>_<datexp>_trans.npz` for the visual area of
each neuron, and the behavior dictionary `data/beh/Beh_<exp_type>.npy` keyed by the session id (plus
`_<stimtype>` for swap sessions). 89 sessions are found and all 89 are converted. The behavior file is
re-read from disk inside `process_session` for every session, and is additionally read in full twice more
in two dataset-wide pre-passes (`get_all_stim_names`, `compute_global_speed_quartiles`).

ii.
```python
def get_all_sessions():
    info = np.load(os.path.join(DATA_ROOT, 'beh', 'Imaging_Exp_info.npy'), allow_pickle=True).item()
    session_map = {}
    for exp_type in info.keys():
        for entry in info[exp_type]:
            session_id = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
            beh_key = session_id if 'stimtype' not in entry else f"{session_id}_{entry['stimtype']}"
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
```
```python
def load_spk(mname, datexp, blk, root=''):
    fn = f'{mname}_{datexp}_{blk}_neural_data.npy'
    spk = np.concatenate(
        [nspk for nspk in np.load(os.path.join(root, fn), allow_pickle=True).item()['spks']], 0
    )
    return spk

def load_retino(mname, datexp, root=''):
    dtrans = np.load(os.path.join(root, f'{mname}_{datexp}_trans.npz'), allow_pickle=True)
    return dtrans['iarea']
```
```python
    beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
    beh = beh_all[beh_key]
    del beh_all
```

iii. CONVERSION_NOTES Step 1 records the reference loading pattern verbatim (`load_spk` concatenates
`spks` across planes, `load_retino` gives `iarea`, behavior comes from `Beh_<exp_type>.npy`), and the AI
states it copied that pattern. Step 5 Key Decision 1: "All 89 sessions included: Each unique neural
recording = one session", because "each neural file is unique per session — the experiment type just
categorizes the same recording differently" (trajectory step 21).

## 1-b. How are the data split into subjects?

i. The subject is `entry['mname']` from the index, carried on each session record. `all_subjects` is built
in first-encounter order while looping over sessions (which are themselves sorted by `session_id`, so
subjects appear in alphabetical order), and `subject_idx` is the index of the session's mouse into that
list. 19 subjects result, in the same order and with the same session counts as the reference.

ii.
```python
        mname = result['mname']
        if mname not in all_subjects: all_subjects.append(mname)
        subject_idx = all_subjects.index(mname)
```
```python
        'subjects': all_subjects, 'subject_idx': np.array(all_subject_idx),
```

iii. Not explicitly argued; the notes simply record 19 subjects consistent with the paper's "89 recordings
in 19 mice" (Steps 2-4 consistency table). The mouse name is given by the index, so no split has to be
inferred.

## 1-c. How are the data split into sessions?

i. A session is one recording: mouse + date + block, i.e. one `*_neural_data.npy` file. Because the master
index lists the same recording under several experiment types, the AI keys a dictionary on `session_id` and
keeps only the first occurrence (`if session_id not in session_map`), yielding 89 unique sessions from 142
index entries. Sessions are emitted sorted by `session_id`.

ii.
```python
            if session_id not in session_map:
                session_map[session_id] = (exp_type, entry, beh_key)
    sessions = []
    for session_id in sorted(session_map.keys()):
```

iii. Trajectory step 21: "there are exactly 89 unique sessions matching the paper's claim of 89
recordings. Each session can appear in multiple experiment types... each neural file should be one
session." Notes Step 10 Check 5: "Sessions with stimtype handled (use first occurrence)".

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials the behavior declares, and a frame belongs to trial *t* if
`ft_trInd == t`. Within a trial the AI keeps only frames where the VR was moving, `ft_move > 0`; it does
**not** restrict to the textured corridor (`ft_CorrSpc`), so each trial covers the 4 m texture **and** the
following 2 m grey space (`ft_Pos` runs 0–60 dm). Trials are variable length (median ≈ 31 bins, max 193).
Two consequences: (a) frames stationary in VR are dropped from the middle of a trial, so a trial's columns
are not temporally contiguous — in the session I checked, 295 of 470 trials contain at least one internal
gap, some as long as 65 s; (b) the grey space is part of every trial.

ii.
```python
    vr_move = ft_move > 0
    for trial_idx in range(ntrials):
        valid_mask = (ft_trInd == trial_idx) & vr_move
        valid_frame_indices = np.where(valid_mask)[0]
        if len(valid_frame_indices) < 2:
            continue
```

iii. Notes Step 1: "Frame filter: `ft_move > 0` only running/VR-moving frames", justified by the paper's
statement that the authors "only considered timepoints during running", and by reference `utils.py`, which
builds `VRmove = beh['ft_move'][:nfr] > 0`. Trajectory step 27: "The paper says 'only considered timepoints
during running for analysis'. I should filter for VR-moving frames (ft_move > 0)." Variable-length trials
were chosen because "some trials have more frames (if mouse stops)" (step 26). The grey space was kept
deliberately: after seeing position bin 3 hold 50% of the data, the AI considered excluding grey frames but
rejected it because "that would lose a lot of data" (step 64). The AI saw the long time gaps and accepted
them: "the time values are correct — they represent real elapsed time... This is fine for the decoder"
(step 65). Note that the reference code's own validity mask is `fr_valid = VRmove & isCorridor`, i.e. it
combines the movement filter with the texture-space filter that the AI dropped.

## 1-e. How are trials filtered based on quality controls?

i. Essentially none. A trial is kept if it has at least 2 VR-moving frames; a session is dropped if fewer
than 2 trials survive. In practice nothing is removed: all 38,110 trials and all 89 sessions are kept
(the reference keeps 37,728 after dropping 382 outliers). No trial-length outlier rule, no behavioral
performance rule, no per-session rule beyond the 2-trial minimum.

ii.
```python
        if len(valid_frame_indices) < 2:
            continue
```
```python
    valid_count = len(neural_trials)
    ...
    if valid_count < 2:
        return None
```

iii. Notes Step 3 "Trial curation rules": "All trials included, frames filtered for VR-moving". The AI's
implicit argument is that the `ft_move > 0` filter already removes the pathological stationary periods, so
no separate length cut is needed; indeed the longest converted trial is 193 bins versus 238 in the
reference (which had to explicitly drop 99th-percentile traversals). The AI never measured the trial-length
distribution or looked for parked animals; it did notice the resulting multi-minute within-trial gaps and
chose to keep them (trajectory steps 64–65).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session_id>_neural_data.npy` — a list of one (neurons × frames) array per imaging
plane, concatenated along the neuron axis. The area label of each neuron comes from `iarea` in
`retinotopy/<mname>_<datexp>_trans.npz`.

ii.
```python
    spk = load_spk(mname, datexp, blk, root=os.path.join(DATA_ROOT, 'spk'))
    n_neurons_total, n_frames_neural = spk.shape
    iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
```

iii. Notes Step 1 and Step 3: the data are "Suite2p deconvolved fluorescence traces" and the paper states
"All analyses based on deconvolved fluorescence traces", so `spks` is used directly and no dF/F or
deconvolution step is needed.

## 2-b. How is the `neural` data processed?

i. Not processed at all. For each trial the kept columns of `spks` are gathered and cast to float32
(the source arrays are already float32, so this is a no-op numerically but keeps full precision).
Trials are variable length; nothing is padded, smoothed, normalised or z-scored.

ii.
```python
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. Notes: `neural_data_type`: "Suite2p deconvolved fluorescence traces", `deconvolution_decay_s`: 0.75.
Because the file already holds deconvolved traces, the AI treats them as the final neural signal. No
rationale is given for float32 (the reference uses float16); the resulting pickle is 211–226 GB.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only filter is anatomical: neurons whose retinotopic area code is `-1` (unassigned) or `7` are
dropped; everything else is kept and assigned to V1 (8), mHV (0,1,2,9), lHV (5,6) or aHV (3,4). This keeps
4,105,393 neurons, exactly the same set (and the same per-area counts: V1 1,833,035 / mHV 1,108,860 /
lHV 495,318 / aHV 668,180) as the reference solution. No activity, SNR or selectivity filter is applied.

ii.
```python
    iarea = load_retino(mname, datexp, root=os.path.join(DATA_ROOT, 'retinotopy'))
    neuron_mask = (iarea != -1) & (iarea != 7)
    spk = spk[neuron_mask]
    iarea_filtered = iarea[neuron_mask]
```
```python
def neu_area_ID(iarea):
    idx = {}
    idx['V1'] = iarea == 8
    idx['mHV'] = (iarea == 0) | (iarea == 1) | (iarea == 2) | (iarea == 9)
    idx['lHV'] = (iarea == 5) | (iarea == 6)
    idx['aHV'] = (iarea == 3) | (iarea == 4)
    return idx
```

iii. Notes Step 1: "Neuron filter: `(iarea != -1) & (iarea != 7)` excludes non-visual cortex neurons" —
copied verbatim from reference `utils.py` (`idx_neu = (arid!=-1) & (arid != 7) # exclude neurons from
outside of visual cortex`), and `neu_area_ID` is copied from the reference. The AI notes that Suite2p
already curated the cells, and the reference applies no further filter (the d-prime selectivity filter in
`utils.py` is an analysis-specific step, not a curation step, and is correctly not applied).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is declared as "Trial start (corridor entry)", `off_start = 0.0`, `off_end = None`.
Each trial's matrix starts at the first VR-moving frame carrying that trial's index — normally the frame
right after `StartFr` (I verified first-moving-frame = 6, 71, 116, 175, 211 against
`StartFr` = 4.30, 70.23, 115.59, 174.02, 210.52) — and runs to the last moving frame of the trial, i.e.
through the grey space. Trials are left at their own lengths, nothing is padded or truncated. Because
stationary frames are removed, bin *k* of a trial is not a fixed time after corridor entry, and successive
bins are not necessarily 315 ms apart.

ii.
```python
        valid_mask = (ft_trInd == trial_idx) & vr_move
        valid_frame_indices = np.where(valid_mask)[0]
        ...
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```
```python
            'time_bin_size': TIME_BIN_MS, 'temporal_alignment_event': 'Trial start (corridor entry)',
            'off_start': 0.0, 'off_end': None,
```

iii. Trajectory step 26: "the task says 'temporally aligned based on trial start (corridor entry)', so I
should include frames from corridor entry... I'll use variable-length trials." The instructions allow
`off_end = None`, and the decoder consumes each trial's own length, so no common window is imposed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One imaging frame = one bin, at the 3.17 Hz imaging rate, so
`time_bin_size = 1000/3.17 = 315.46 ms`, identical across all trials and sessions. The
position-interpolated representation used by the paper (60 spatial bins per trial, `get_interpPos_spk`)
was deliberately not used.

ii.
```python
FRAME_RATE = 3.17  # Hz
TIME_BIN_MS = 1000.0 / FRAME_RATE
```
```python
            'time_bin_size': TIME_BIN_MS, ...
            'frame_rate_hz': FRAME_RATE,
```

iii. Notes Step 3/4: frame rate 3.17 Hz from the reference notebook, measured mean frame interval 0.3148 s
in the data — declared "Consistent". The frame is the finest resolution available and all behavioral
streams are already sampled on the same frame grid, so nothing needs resampling. (Caveat the AI does not
flag: since non-moving frames are dropped, consecutive stored bins are sometimes far more than 315 ms
apart, so the declared bin size describes the acquisition rate rather than the spacing of stored bins.)

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundFr` (the fractional imaging-frame number at which the sound cue was played on each trial)
and `ft` (the MATLAB datenum timestamp of every imaging frame, truncated to the number of imaged frames).

ii.
```python
    ft = beh['ft'][:n_frames_neural]
    ...
    sound_fr = beh['SoundFr']
```

iii. Notes Step 5 mapping table: "SoundFr, ft -> input[0]: time_to_sound_cue: Time from frame to sound cue
(seconds)". Trajectory step 26 identifies `SoundFr` as the frame index of the cue per trial.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The fractional cue frame is linearly interpolated onto the frame-time axis (manually: integer part plus
fraction times the inter-frame interval), and the input is `frame_time − cue_time`, converted from days to
seconds. The sign convention is therefore *time since* the cue: negative before the cue, positive after —
the opposite sign to the reference, which stores `cue − time`. If `SoundFr` were NaN the whole trial's
input would be set to zeros, and out-of-range cue frames are clamped to the first/last frame time (both
branches are dead code: I checked all 23 behavior files, 63,177 trials, and found zero NaN or
out-of-range `SoundFr`).

ii.
```python
        s_fr = sound_fr[trial_idx]
        if np.isnan(s_fr):
            time_to_sound = np.zeros(n_t, dtype=np.float32)
        else:
            s_fr_int = int(np.floor(s_fr))
            s_fr_frac = s_fr - s_fr_int
            if 0 <= s_fr_int < len(ft) - 1:
                sound_time = ft[s_fr_int] + s_fr_frac * (ft[s_fr_int + 1] - ft[s_fr_int])
            elif s_fr_int >= len(ft) - 1:
                sound_time = ft[-1]
            else:
                sound_time = ft[0]
            time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. Trajectory step 27, decision 6: "Time to sound cue: For each frame, compute time (in seconds) until
the sound cue. Before sound cue: negative. After: positive." The interpolation is needed because `SoundFr`
is fractional; the `DAYS_TO_SEC` factor because `ft` is a MATLAB datenum (Notes Step 5, decision 6:
"Time computations: Using actual frame times (MATLAB datenum)").

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on `ft[valid_frame_indices]` — exactly the frames whose `spks` columns form that trial's
neural matrix — so the two streams share the same index grid bin for bin.

ii.
```python
        ft_trial = ft[valid_frame_indices]
        ...
            time_to_sound = ((ft_trial - sound_time) * DAYS_TO_SEC).astype(np.float32)
```

iii. All behavioral streams in this dataset are stored per imaging frame, so taking the same frame window
guarantees alignment; the AI's `--show-processing` plots overlay inputs, outputs and neural data per trial
to check this visually.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From `datexp`, the calendar date embedded in each session id (`YYYY_MM_DD`), grouped per mouse.

ii.
```python
        dates = [(idx, datetime(int(s['datexp'].split('_')[0]), int(s['datexp'].split('_')[1]),
                                int(s['datexp'].split('_')[2]))) for idx, s in msessions]
```

iii. Trajectory step 27, decision 7: "The exp_info has 'sess#' which is 0 for before learning, 1 for after
learning. But this doesn't capture the actual day of training. I should use the date to compute days from
the first session for each mouse."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the sessions are sorted by date and the value is the number of **calendar days** between
that session's date and the mouse's first recorded session; the value is 0 for the first session and
ranges up to 92. It is computed over all 89 sessions (so a sample run agrees with a full run) and
broadcast as a constant across every bin of every trial in the session.

ii.
```python
def compute_day_of_training(sessions):
    mouse_sessions = defaultdict(list)
    for i, s in enumerate(sessions):
        mouse_sessions[s['mname']].append((i, s))
    day_of_training = np.zeros(len(sessions))
    for mname, msessions in mouse_sessions.items():
        dates = [...]
        dates.sort(key=lambda x: x[1])
        first_date = dates[0][1]
        for idx, date in dates:
            day_of_training[idx] = (date - first_date).days
    return day_of_training
```
```python
        input_trial[1] = np.float32(day_val)
```

iii. Notes Step 5: "datexp -> input[1]: day_of_training: Days since first session per mouse". The AI's
argument is that the experiment-type label (before/after learning) is too coarse and the elapsed calendar
time is the literal "day of training"; it is a per-trial (constant) input as the Decoder Task specifies.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft` alone — the timestamp of the first retained frame of the trial is used as the trial's origin.
`StartFr` (the fractional corridor-entry frame) is read by nobody in the script; the AI knew of it
("StartFr contains interpolated frame indices for corridor entry", trajectory step 17) but did not use it.

ii.
```python
        ft_trial = ft[valid_frame_indices]
        time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. Notes Step 5: "ft -> input[2]: time_since_trial_start: Elapsed time from trial start". Because the
first frame of the trial window is (within one frame) corridor entry, the AI uses it directly as t = 0,
which also guarantees `off_start = 0.0` exactly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Subtract the first frame's timestamp and convert days to seconds. Values start at exactly 0.0 and
increase monotonically. Because stationary frames were removed but real clock time was kept, the value
accumulates the duration of the removed pauses: the per-session maxima reach 407 s, 836 s and in one case
1769 s, versus a maximum of 74.8 s in the reference.

ii.
```python
        time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
        ...
        input_trial[2] = time_since_start
```

iii. Trajectory step 65: "The problem is that when the mouse stops running, there are large gaps in the
frame sequence. I should use the actual elapsed time (from frame times) rather than frame index
differences... the time values are correct — they represent real elapsed time. The issue is that some
trials have very long pauses. This is fine for the decoder — it's real data."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame window as the neural columns (`ft[valid_frame_indices]`), so it is bin-for-bin aligned.

ii.
```python
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
        ft_trial = ft[valid_frame_indices]
        time_since_start = ((ft_trial - ft_trial[0]) * DAYS_TO_SEC).astype(np.float32)
```

iii. As in 3-c: every stream is indexed by imaging frame, so a shared frame index list is sufficient.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
    is_rew = beh['isRew']
    ...
        input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. Notes Step 5: "isRew -> input[3]: reward_availability: 1=rewarded, 0=not". Trajectory step 21: "The
reward availability input would be 0 for unsupervised/naive sessions and 1/0 for supervised sessions."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a bool → float cast, broadcast as a constant over the trial's bins. The verification output
confirms it is identically 0 for the unsupervised/naive sessions and takes both values in the supervised
ones.

ii.
```python
        input_trial[3] = np.float32(1.0 if is_rew[trial_idx] else 0.0)
```

iii. No processing is needed: the flag is already exactly the quantity the Decoder Task asks for
("1 if in rewarded corridor, 0 if not, discrete, per-trial").

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the wall texture. The label set is the sorted union of
`UniqWalls` over all 89 sessions, gathered in a dataset-wide pre-pass so that the label indices are
consistent between a sample run and a full run.

ii.
```python
def get_all_stim_names(sessions):
    all_stim = set()
    ...
        for wn in beh_cache[exp_type][beh_key]['UniqWalls']:
            all_stim.add(str(wn))
    return sorted(all_stim)
```
```python
        stim_idx = stim_to_idx[str(wall_name[trial_idx])]
```

iii. Notes Step 5: "WallName -> output[0]: visual_stimulus_category: Map to index". `WallName` (not
`TrialStim`) is the per-trial texture identity, which is the same variable the reference uses.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each raw wall name is mapped to its index in the 15-element sorted list
`['circle1','circle2','circle3','leaf1','leaf1_swap1','leaf1_swap2','leaf2','leaf3','rock1','rock2',
'wood1','wood1_swap1','wood1_swap2','wood2','wood5']`, and the index is broadcast over every bin of the
trial. The crops and spatial shuffles of one texture family are therefore **separate classes**; they are
not collapsed to the four base textures (circle / leaf / rock / wood). The reference `utils.py` function
`get_cat_id`, which maps wall names onto categories, was catalogued in the notes but not used. As a result
the decoder faces 15 classes, most of which are absent from any given session (chance 1/15 = 0.067, full
balanced validation accuracy 0.493, versus 4 classes and 0.664 for the reference).

ii.
```python
    all_stim_names = get_all_stim_names(all_sessions)
    stim_to_idx = {name: i for i, name in enumerate(all_stim_names)}
    ...
        output_trial[0] = stim_idx
```
```python
        'output_values': [all_stim_names, ['no_lick', 'lick'], ...],
```

iii. No explicit justification is given in CONVERSION_NOTES or the trajectory for keeping 15 raw names
rather than the base textures; the AI only comments afterwards that "With 15 categories, 49% accuracy is
very good (7.4x chance)" (Notes Step 12), and notes that the sklearn warning about classes absent from
`y_true` arises "because some stimulus categories (like circle2, leaf2, etc.) don't appear in these 2
sessions" (trajectory step 74).

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
    lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```

iii. Notes Step 5: "LickFr -> output[1]: licking: Binary per frame". Trajectory step 26: "LickFr exists -
frame indices for each lick event".

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built: the fractional lick frame is **rounded** to the nearest frame
(the reference truncates) and that frame is set to 1; licks outside `[0, n_frames)` are discarded, and an
empty `LickFr` gives an all-zero vector. A bin is 1 if at least one lick was assigned to it. Because
stationary frames are later dropped, licks emitted while the animal was standing still (common at the
reward point) do not appear in the converted data: the overall lick fraction is 2.7%, versus 4.1% in the
reference.

ii.
```python
def build_lick_per_frame(n_frames, lick_fr):
    lick_binary = np.zeros(n_frames, dtype=np.int32)
    if len(lick_fr) == 0:
        return lick_binary
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
    lick_binary[idx] = 1
    return lick_binary
```

iii. The Decoder Task asks for licking as a binary time series ("If an output is a time such as when a
behavior occurred, represent it as a binary time series"), so lick events are rasterised onto the frame
grid. Rounding/range-clipping is described in Notes Step 10 as edge-case handling.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already expressed in imaging frames, and the per-frame flag is indexed with the same
`valid_frame_indices` used for the neural columns.

ii.
```python
        lick_trial = lick_binary[valid_frame_indices]
        ...
        output_trial[1] = lick_trial
```

iii. Same argument as 3-c: frame indexing is the common clock of all streams.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the VR position of the animal at each imaging frame, in decimetres, running 0–40 inside
the textured corridor and 40–60 through the grey space (truncated to the imaged frames).

ii.
```python
    ft_Pos = beh['ft_Pos'][:n_frames_neural]
```

iii. Notes Step 5: "ft_Pos -> output[2]: position_bin". Step 3 records the corridor geometry: "4m texture +
2m grey", "Corridor_Length = 60" decimetres.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. The position is divided by 15 decimetres and floored, giving 4 bins that tile the **full 6 m** corridor
(texture plus grey space), and clipped to [0, 3]. The value is stored per bin and labelled
`['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m']`. The resulting distribution is near uniform
(0.250/0.251/0.252/0.246) by construction, since the VR advances at a roughly constant rate through both
sections.

ii.
```python
        pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```
```python
        'output_values': [..., ['0-1.5m', '1.5-3m', '3-4.5m', '4.5-6m'], ...],
```

iii. Trajectory steps 63–65: the AI first implemented 1 m bins over the texture (`//10`, clipped), saw that
bin 3 then absorbed all grey-space frames and held 50% of the data, considered the two clean alternatives
("exclude grey space frames entirely, or include grey space as a 5th position category") and rejected both
("that would lose a lot of data"; only 4 `output_values` are allowed), and concluded: "The task says
'4 equal-length, 1-m-long spatial bins' but the corridor is 6m total, so 4 bins of 1.5m each makes more
sense for equal distribution."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed geometric thresholds at 15, 30 and 45 decimetres, i.e. four equal-length bins of **1.5 m**, not
the 1 m specified in the Decoder Task. The thresholds are the same for every trial and session.

ii.
```python
        pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
```

iii. As in 9-b: the bin width was widened from the specified 1 m to 1.5 m, deliberately and with the
specification quoted, in order to spread the grey-space frames evenly across the four classes rather than
drop them.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame and is indexed with the same `valid_frame_indices` as the
neural columns.

ii.
```python
        pos_bins = np.clip(np.floor(ft_Pos[valid_frame_indices] / 15.0), 0, 3).astype(np.int32)
        ...
        output_trial[2] = pos_bins
```

iii. Frame indexing is shared by all streams; the `--show-processing` figure plots position bins together
with the neural raster for the same trial.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the animal's running speed at each imaging frame (treadmill speed, not VR speed).

ii.
```python
    ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
```

iii. Notes Step 5: "ft_RunSpeed -> output[3]: running_speed_bin: 4 quartile bins".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw speed is used, discretised into four bins by three **global** thresholds shared by all sessions. The
thresholds are estimated in a dataset-wide pre-pass: for each of the 89 sessions the speeds at VR-moving
frames are taken, subsampled to at most 5,000 frames with a fixed seed, pooled (~445,000 samples), and the
25/50/75th percentiles computed — 11.79, 24.43, 40.44 cm/s. Because each session contributes an equal
number of samples regardless of length, these are session-weighted rather than data-weighted quantiles.

ii.
```python
def compute_global_speed_quartiles(sessions):
    all_speeds = []
    ...
        vr_move = beh['ft_move'] > 0
        speeds = beh['ft_RunSpeed'][vr_move]
        if len(speeds) > 5000:
            speeds = np.random.RandomState(42+i).choice(speeds, 5000, replace=False)
        all_speeds.append(speeds)
    all_speeds = np.concatenate(all_speeds)
    quartiles = np.percentile(all_speeds, [25, 50, 75])
    return quartiles
```

iii. Notes Step 5, decision 5: "Speed bins: Global quartile-based (computed from all sessions)". Trajectory
step 28: "I need to handle the speed quartile computation properly - should use ALL sessions' data", and
step 66–67 optimise the pre-pass to read behavior only. Step 72 accepts the consequence: "The speed bins
are still skewed for the unsupervised session (95% in Q1) which makes sense since the quartiles are
computed globally and this session has slower running."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize` against the three global thresholds, clipped to [0, 3], labelled `['Q1','Q2','Q3','Q4']`.
Dataset-wide the four classes hold 0.230 / 0.242 / 0.259 / 0.269 of the bins — approximately but not
exactly the "25% of the data" the Decoder Task asks for (the reference's per-session rank split gives
0.2500 / 0.2500 / 0.2500 / 0.2500). Per session the imbalance is severe: session-level fractions range
from (0.951, 0.045, 0.004, 0.000) to (0.029, 0.099, 0.208, 0.664).

ii.
```python
        speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
```

iii. A single global scale is argued to make the label mean the same thing in every session (a "fast" bin
is fast in absolute terms), at the cost of within-session balance, which the AI explicitly noticed and
accepted (trajectory step 72). Note that discarding stationary frames removes the mass of exactly-zero
speeds that would otherwise make threshold-based quartiles impossible (among VR-moving frames the
zero-speed fraction is 0), so thresholding rather than ranking is workable here.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is indexed with the same `valid_frame_indices` as the
neural columns.

ii.
```python
        speed_bins = np.clip(np.digitize(ft_RunSpeed[valid_frame_indices], speed_quartiles), 0, 3).astype(np.int32)
        ...
        output_trial[3] = speed_bins
```

iii. Frame indexing is shared across streams; no interpolation or lag is applied.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four guards. (a) The behavior arrays can be 1–2 frames longer than the imaging, so every behavioral
stream is truncated to `n_frames_neural = spk.shape[1]`. (b) Licks with frame numbers outside
`[0, n_frames)` are dropped and a missing/empty `LickFr` yields an all-zero lick vector. (c) A NaN
`SoundFr` yields an all-zero `time_to_sound_cue` for that trial and an out-of-range `SoundFr` is clamped to
the first/last frame time (both are dead code in this dataset — I verified 0 NaN and 0 out-of-range values
across all 63,177 trials in all 23 behavior files). (d) Trials with fewer than 2 usable frames and sessions
with fewer than 2 usable trials are skipped (neither occurs). No try/except around session processing, so a
hard failure would abort the run; none occurred.

ii.
```python
    n_neurons_total, n_frames_neural = spk.shape
    ...
    ft = beh['ft'][:n_frames_neural]
    ft_trInd = beh['ft_trInd'][:n_frames_neural]
    ft_Pos = beh['ft_Pos'][:n_frames_neural]
    ft_move = beh['ft_move'][:n_frames_neural]
    ft_RunSpeed = beh['ft_RunSpeed'][:n_frames_neural]
    lick_binary = build_lick_per_frame(n_frames_neural, beh.get('LickFr', np.array([])))
```
```python
    idx = np.round(lick_fr).astype(int)
    idx = idx[(idx >= 0) & (idx < n_frames)]
```

iii. Trajectory step 67: "the behavior arrays are only 1-2 frames longer than the neural data... Looking at
the reference code, it uses `nfr = spk.shape[1]` and then `beh['ft_move'][:nfr]`", i.e. the truncation is
copied from the reference. Notes Step 10 Check 5 lists the edge cases considered.

## 12-a. What are the most time-consuming steps of the code?

i. As instrumented by the script itself (per-session `load` and `process` timings), loading and
concatenating the per-plane `spks` arrays dominates: 4–15 s per session on the full run (up to 100–250 s
when the file cache is cold), against 3–7 s for all per-trial processing. Total full conversion 73.1 min.
Two further costs the AI does not account for: every session writes its converted arrays to a per-session
pickle in `cache/` and the combine step reads them all back (~226 GB written and re-read, then 211 GB
written again for the final pickle), and each of the 89 sessions re-loads its whole 100–430 MB behavior
file from disk.

ii.
```python
    t_load = time.time()
    print(f'  Neurons: {n_neurons_total} -> {n_neurons} ({t_load-t0:.1f}s load)')
    ...
    t1 = time.time()
    print(f'  Trials: {valid_count}/{ntrials} ({t1-t_load:.1f}s process)')
```
```python
        temp_fn = f'cache/session_{i:03d}.pkl'
        with open(temp_fn, 'wb') as f:
            pickle.dump({...}, f, protocol=4)
```

iii. Notes Step 6/7: the AI identified the speed-quartile pre-pass as the first bottleneck (it originally
loaded every neural file just to get a frame count) and removed the neural loads from it, cutting the
sample run from ~800 s to 179.6 s. Trajectory step 74: "The main bottleneck is loading the neural data
files." No analysis of the temp-file round trip or repeated behavior loads is given.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Two. (1) The per-trial membership test `(ft_trInd == trial_idx) & vr_move` rescans the whole
frame-length array once per trial — O(ntrials × nframes), where one `np.argsort`/`np.split` pass would
group all frames by trial at once (the reference has the same pattern). (2) The per-trial fancy-index
gather `spk[:, valid_frame_indices]` copies out of a ~50,000 × 20,000 array several hundred times per
session; a single gather of all kept columns followed by `np.split` would touch the matrix once. Minor
ones: `all_subjects.index(mname)` is a linear scan per session, and the per-trial `np.zeros` +
row-assignment for inputs/outputs could be one `np.stack`.

ii.
```python
    for trial_idx in range(ntrials):
        valid_mask = (ft_trInd == trial_idx) & vr_move
        valid_frame_indices = np.where(valid_mask)[0]
        ...
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. The AI did not identify these; Notes Step 6 lists only "save session data to temp files", "speed
quartiles computed from behavior data only" and "garbage collection after each session" as optimisations.
In practice the per-trial loop costs only 3–7 s per session against 4–15 s of I/O, so the omission has
limited impact.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are read from disk repeatedly: once for every session in `get_all_stim_names` (all 23
files), a second time for every session in `compute_global_speed_quartiles` (all 23 files again), and then
once more per session inside `process_session` — 89 separate `np.load` calls of 100–430 MB files, where
the reference reads each behavior file exactly once and processes all its sessions from that one read.
Both pre-passes also hold every loaded behavior file in a `beh_cache` dict simultaneously (~5 GB) instead
of releasing them. In addition, every converted session is serialised to `cache/session_NNN.pkl` and
immediately deserialised again in the combine step, and `neu_area_ID` recomputes the area masks that
`neuron_mask` already derived from the same `iarea` array.

ii.
```python
def get_all_stim_names(sessions):
    beh_cache = {}
    for session in sessions:
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(...).item()
```
```python
def compute_global_speed_quartiles(sessions):
    beh_cache = {}
    for i, session in enumerate(sessions):
        if exp_type not in beh_cache:
            beh_cache[exp_type] = np.load(...).item()
```
```python
    beh_all = np.load(os.path.join(DATA_ROOT, 'beh', f'Beh_{exp_type}.npy'), allow_pickle=True).item()
    beh = beh_all[beh_key]
    del beh_all
```

iii. Not documented. The two pre-passes are themselves justified — the stimulus label set and the speed
thresholds must be identical between a sample run and a full run — but the AI never merged them into one
pass, nor grouped the main loop by behavior file as the reference does.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The main one is the temp-file round trip: each session's converted arrays are pickled to `cache/`,
read back, concatenated into one in-memory dictionary and written again to the output pickle, then the
temp files are deleted. The stated purpose — "avoid memory accumulation" — is not achieved, because the
combine step holds the entire 211 GB dataset in RAM anyway; the only effect is roughly 450 GB of extra
disk traffic. Second, the neural arrays are kept at float32 although the same values in float16 (as in the
reference) would halve the 211 GB file and its write time with no loss that matters for a decoder. Third,
several computations are dead or redundant: the NaN/out-of-range `SoundFr` branches (no such values
exist), `neu_area_ID` recomputing masks already implied by `neuron_mask`, `session_meta`/`brain_regions`
variables assigned and never used, the unused `idx`/`speed_quartiles`/`stim_to_idx` parameters of
`plot_processing`, and `gc.collect()` after each session and each trial-set.

ii.
```python
        temp_fn = f'cache/session_{i:03d}.pkl'
        with open(temp_fn, 'wb') as f:
            pickle.dump({...}, f, protocol=4)
        temp_files.append(temp_fn)
    ...
    for tf in temp_files:
        with open(tf, 'rb') as f:
            sd = pickle.load(f)
        all_neural.append(sd['neural'])
        ...
    for tf in temp_files:
        os.remove(tf)
```
```python
        neural_trial = spk[:, valid_frame_indices].astype(np.float32)
```

iii. Notes Step 6: "Save session data to temp files to avoid memory accumulation during processing" and
"Garbage collection after each session". No justification is offered for float32, and the AI never
re-examined whether the temp-file scheme actually reduced peak memory.
