# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads the master index `beh/Imaging_Exp_info.npy` first and turns it into a map from one
*physical recording* (`mname_datexp_blk`) to exactly one behavior copy (`make_source_map`). It
cross-checks that map against the set of files in `spk/` and aborts if the two disagree. Behavior is
then loaded **one file per experiment type** (`load_behaviors`), and only the fields needed by the
conversion are copied out of each session dict (`compact_behavior`) so the big `Beh_*.npy` dict can
be released before the next file is read. Neural traces (`spk/<sid>_neural_data.npy`, key `spks`, a
list of per-plane arrays) and the retinotopy (`retinotopy/<mouse>_<date>_trans.npz`, key `iarea`)
are read per session inside the main loop. All 89 recordings / 19 mice are converted.

ii.
```python
exp_info = np.load(data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
sources = make_source_map(exp_info, data_root / "spk")
session_ids = sorted(sources)
behaviors = load_behaviors(data_root, sources)
```
```python
    spk_sessions = {path.name.removesuffix("_neural_data.npy")
                    for path in spk_root.glob("*_neural_data.npy")}
    if set(sources) != spk_sessions:
        raise RuntimeError("Experiment table/neural file mismatch: ...")
```
```python
        path = data_root / "beh" / f"Beh_{experiment_type}.npy"
        behavior_file = np.load(path, allow_pickle=True).item()
        for sid, key in requested:
            behaviors[sid] = compact_behavior(behavior_file[key])
        del behavior_file
        gc.collect()
```
```python
        raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
        retino_name = f"{record['mname']}_{record['datexp']}_trans.npz"
        iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
```

iii. From the trajectory (steps 21–54): the agent enumerated the index, found 142 entries but only
89 unique `mname_datexp_blk` triples, and confirmed `mapping 89 spks 89 missing mapping set()
missing spk set()`. The module docstring states: "The paper contains 89 unique imaging recordings,
but several recordings occur in more than one analysis-specific `Beh_*.npy` file. This converter
includes each physical recording once." The per-experiment-type grouping is explicitly to read each
behavior file once (the largest is 173 MB and the spk directory totals 434 GB, which the agent
measured in step 19).

## 1-b. How are the data split into subjects?

i. The subject of a session is the `mname` field of its index entry. `subjects` is the sorted set of
unique `mname` values (19 mice) and `subject_idx` is each session's index into that list. Sessions
are emitted in `sorted(session_ids)` order, which is alphabetical by `mouse_date_block`, so a
mouse's sessions are contiguous and in date order.

ii.
```python
    subjects = sorted({sources[sid]["record"]["mname"] for sid in session_ids})
    subject_to_id = {name: idx for idx, name in enumerate(subjects)}
...
        subject_idx.append(subject_to_id[str(record["mname"])])
```

iii. Not discussed explicitly; the index already names the mouse, so no splitting has to be derived.
The verifier output the agent inspected confirms 19 subjects with the expected per-mouse session
counts.

## 1-c. How are the data split into sessions?

i. A session is one mouse on one date in one block: `f"{mname}_{datexp}_{blk}"`, which is also the
name of the spike file. A recording that appears under several experiment types (or under two
`stimtype` swap keys) is kept **once**, using the first occurrence in the iteration order of
`Imaging_Exp_info.npy` (`sources.setdefault`). The behavior key appends `_{stimtype}` when the entry
has one. 89 sessions result.

ii.
```python
def session_id(record: dict) -> str:
    return f"{record['mname']}_{record['datexp']}_{record['blk']}"

def behavior_key(record: dict) -> str:
    key = session_id(record)
    if "stimtype" in record:
        key += f"_{record['stimtype']}"
    return key
```
```python
            sid = session_id(record)
            sources.setdefault(sid, {"experiment_type": experiment_type,
                                     "behavior_key": behavior_key(record),
                                     "record": record})
```

iii. Docstring of `make_source_map`: "Repeated behavior copies differ only in analysis labels such as
`stim_id`; trial timing, actual `WallName`, and behavioral streams are the same. The first
occurrence in the repository's experiment table is therefore a deterministic canonical copy." The
agent verified this in step 50–51 by printing `UniqWalls`/`stim_id`/`ntrials`/`len(ft)` for the
`_swap1` and `_swap2` copies of the same recording (identical except `stim_id`).

## 1-d. How are the data split into trials?

i. Trials are the trials the data declares: every imaging frame carries its trial in `ft_trInd`. The
AI builds a per-session frame mask = *valid trial index* AND `ft_CorrSpc` (inside the 4 m textured
corridor) AND `ft_move > 0` (virtual reality moving, i.e. the mouse running), then groups the
surviving frames by their `ft_trInd` value. A trial is the set of frames that carry its index, in
frame order; trials therefore have variable length (11–178 frames, mean 22.3). All 38,110 trials in
the dataset retain at least one frame. Grey-space frames and non-running frames are excluded, so the
frames of a trial are **not necessarily consecutive in time**.

ii.
```python
def valid_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    """Reproduce the paper's running-and-textured-corridor frame mask."""
    nframes = min(nframes, len(beh["ft"]), len(beh["ft_trInd"]),
                  len(beh["ft_CorrSpc"]), len(beh["ft_move"]))
    trial = beh["ft_trInd"][:nframes]
    valid_trial = np.isfinite(trial) & (trial >= 0) & (trial < beh["ntrials"])
    return valid_trial & beh["ft_CorrSpc"][:nframes].astype(bool) & (beh["ft_move"][:nframes] > 0)
```
```python
        frame_indices = np.flatnonzero(frame_mask)
        trial_ids = beh["ft_trInd"][: len(frame_mask)][frame_mask].astype(np.int64)
        if np.any(np.diff(trial_ids) < 0):
            raise RuntimeError(f"Trial indices are not chronological in {sid}")
...
        kept_trials = np.unique(trial_ids)
        for trial in kept_trials:
            columns = np.flatnonzero(trial_ids == trial)
            frames = frame_indices[columns]
```

iii. Module docstring: "It uses the same valid-frame definition as
`code/utils.py::Get_dprime_selective_neuron`: the animal must be in the 4 m textured corridor and the
VR must be moving (`ft_move > 0`)." The agent read that function in the trajectory (steps 21/41) and
quoted the reference comment "only use activity inside the texture area plus mouse is running (VR
moving)"; `methods.txt` likewise says "We only considered timepoints during running for analysis,
which removed time periods when the task mice stopped to collect water rewards." Step 25: "I'll
preserve those original imaging frames (rather than spatially interpolating them), because this
decoder is explicitly time-aligned to corridor entry; timestamps will therefore retain any gaps
caused by pauses."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality filter is applied. Every trial that keeps at least one running-in-corridor
frame is written out — 38,110 of 38,110. The only trial-related guards are structural and abort the
run rather than drop data: a session with no valid frames, a session with fewer than two usable
trials, non-chronological trial indices, a retinotopy/neuron-count mismatch, or a non-float32 spike
array all raise. The agent pre-checked in step 53–54 that no session has a trial with zero frames
(`zero_trials: 0`) and that no session falls below two trials.

ii.
```python
        if len(frame_indices) == 0:
            raise RuntimeError(f"No valid running corridor frames in {sid}")
        if np.any(np.diff(trial_ids) < 0):
            raise RuntimeError(f"Trial indices are not chronological in {sid}")
...
        if len(session_neural) < 2:
            raise RuntimeError(f"Fewer than two usable trials in {sid}")
```

iii. Implicit in the design: the `ft_move > 0` mask is treated as the paper's curation step, so
stalled periods are removed frame-by-frame instead of by discarding trials. The agent reported
(step 42) "the conversion ... has retained the intended full trial set—38,110 trials total are
expected—while filtering only non-running frames and neurons outside the four paper-defined
visual-area groups." No justification is given anywhere for keeping trials whose retained frames
still straddle very long pauses (16% of trials contain a within-trial gap > 5 s, 1.3% a gap > 30 s,
the worst spanning 1,765 s).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session_id>_neural_data.npy` — a list of `(n_neurons, n_frames)` float32
deconvolved traces, one per imaging plane. The area label of each neuron comes from `iarea` in
`retinotopy/<mouse>_<date>_trans.npz`, whose length is asserted to equal the total neuron count
across planes.

ii.
```python
        raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
        nframes = min(plane.shape[1] for plane in raw_planes)
        if any(plane.dtype != np.float32 for plane in raw_planes):
            raise TypeError(f"Expected float32 deconvolved traces in {spk_path}")
        iarea = np.load(data_root / "retinotopy" / retino_name)["iarea"]
        raw_neuron_count = sum(plane.shape[0] for plane in raw_planes)
        if len(iarea) != raw_neuron_count:
            raise RuntimeError(...)
```

iii. Module docstring: "The paper's neural arrays are Suite2p non-negative deconvolved fluorescence
traces." Metadata records `"neural_signal": "Suite2p non-negative deconvolved fluorescence"`.

## 2-b. How is the `neural` data processed?

i. No transformation at all: no dF/F, no smoothing, no normalization, no spatial interpolation, no
z-scoring. The retained neurons and the retained frames are selected plane by plane (to avoid
materializing the whole session), the planes are concatenated in file order, and each trial's
columns are copied into a contiguous `(n_neurons, n_timepoints)` float32 array. Values are kept in
the source dtype (float32), giving a 142 GB pickle.

ii.
```python
        selected_planes = []
        offset = 0
        for plane in raw_planes:
            local_keep = np.flatnonzero(keep_neuron[offset : offset + plane.shape[0]])
            selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
            offset += plane.shape[0]
        selected_neural = np.concatenate(selected_planes, axis=0)
```
```python
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. Module docstring: "No additional smoothing, normalization, or spatial interpolation is applied
here: spatial interpolation in the paper was for position-aligned analyses, whereas the requested
decoder alignment is temporal (corridor entry)." Step 30: "It keeps the paper's four retinotopic
visual-area groups, removes atlas-unassigned neurons ... the resulting pickle is expected to be
roughly 150 GB" — the size was anticipated and accepted rather than reduced by down-casting.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron-level filter is the retinotopic area: a neuron is kept iff its `iarea` code is in
V1 `{8}`, medial `{0,1,2,9}`, anterior `{3,4}` or lateral `{5,6}`. Codes `-1` (outside the atlas) and
`7` are dropped. This keeps 4,105,393 of 4,691,034 neurons (V1 1,833,035; medial 1,108,860;
anterior 668,180; lateral 495,318). No further selection (no activity threshold, no SNR cut) is
applied. Frame-level curation is the running/corridor mask of 1-d.

ii.
```python
AREA_CODES = {"V1": (8,), "medial": (0, 1, 2, 9), "anterior": (3, 4), "lateral": (5, 6)}

def area_labels(iarea):
    region = np.full(len(iarea), -1, dtype=np.int8)
    for idx, name in enumerate(list(AREA_CODES)):
        region[np.isin(iarea, AREA_CODES[name])] = idx
    keep = region >= 0
    return keep, region[keep].astype(np.int64)
```

iii. Code comment: "Exact grouping used by `code/utils.py::neu_area_ID`. Area -1 (outside the
retinotopic atlas) and area 7 (not assigned to one of the four visual-area groups used in the paper)
are excluded." Metadata repeats this. The agent checked the `iarea` histograms of several sessions
in step 20 before fixing the grouping. Suite2p's own cell classifier had already curated the ROIs,
so nothing further is done.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry: the first column of a trial is the first retained frame after the
mouse entered the 4 m textured corridor, and the last is the last retained frame before it left.
Trials keep their natural, variable length; nothing is padded or truncated to a common window.
`off_start = 0.0` and `off_end = None`. Because the mask also removes non-running frames, the
columns of a trial are the *running* frames of that traversal and can be separated by arbitrarily
long real-time gaps (median max within-trial gap 0.63 s, but 16% of trials contain a gap > 5 s and
the largest is 1,765 s). The real elapsed time of each column is nevertheless supplied to the decoder
as the `time_since_trial_start_s` input.

ii.
```python
            columns = np.flatnonzero(trial_ids == trial)
            frames = frame_indices[columns]
            times = beh["ft"][frames]
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```
```python
            "temporal_alignment_event": "entry into the 4 m textured corridor",
            "off_start": 0.0,
            "off_end": None,
```

iii. Step 25 (quoted above) is the only explicit justification: the original imaging frames are
preserved rather than resampled, and the agent notes that "timestamps will therefore retain any gaps
caused by pauses". The metadata field `"frame_filter": "ft_CorrSpc & (ft_move > 0) with a valid
trial index"` documents the consequence.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One column per two-photon imaging frame; the nominal rate of 3.17 Hz
is recorded as `time_bin_size = 1000/3.17 = 315.46 ms`, identical for every trial and session. (The
true inter-frame interval varies slightly, 0.26–0.45 s, and dropped frames mean the columns are not
uniformly spaced in time.)

ii.
```python
FRAME_RATE_HZ = 3.17
...
            "time_bin_size": 1000.0 / FRAME_RATE_HZ,
            "nominal_frame_rate_hz": FRAME_RATE_HZ,
```

iii. Not argued explicitly beyond the docstring's statement that no interpolation/resampling is
applied because the requested alignment is temporal rather than positional. The imaging frame is the
finest resolution available and every behavioral stream (`ft_*`) is already on that same grid.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime` (per-trial MATLAB datenum timestamp of the cue) and `ft` (datenum timestamp of every
imaging frame). The frame-number variant `SoundFr` is not used.

ii.
```python
            times = beh["ft"][frames]
            time_to_sound = ((beh["SoundTime"][trial] - times) * SECONDS_PER_DAY).astype(np.float32)
```

iii. The agent compared the two representations in steps 21–22, printing
`(ft[StartFr] - Trial_start_time)*86400` and the equivalent for the grey-space event and finding
agreement to ~0.25 s (sub-frame). Using the recorded event timestamps directly avoids interpolating
a fractional frame number onto the time axis.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `(SoundTime[trial] - ft[frame]) * 86400`, i.e. MATLAB datenum days converted to seconds, stored as
float32 and evaluated for every retained frame of the trial. It is positive before the cue and
negative after it, matching the name "time **to** sound cue". Observed range over the dataset:
−1763.3 s to +723.5 s (the extremes come from the trials with long stalls described in 1-e).

ii.
```python
SECONDS_PER_DAY = 86_400.0
...
            trial_input[0] = time_to_sound
...
            "time_to_sound_sign": "positive before the cue; negative after the cue",
```

iii. The sign convention is documented explicitly in the metadata. No smoothing or clipping is
applied; the agent did not comment on the extreme values.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the frames that supply the neural columns of that trial (`frames`
derives from the same `columns` used to slice `selected_neural`), so it is element-wise aligned and
has the trial's length.

ii.
```python
            columns = np.flatnonzero(trial_ids == trial)
            frames = frame_indices[columns]
            times = beh["ft"][frames]
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
            time_to_sound = ((beh["SoundTime"][trial] - times) * SECONDS_PER_DAY)
```

iii. All streams in this dataset (`ft`, `ft_*`) share the imaging-frame grid, so indexing every
stream with the same frame list guarantees alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. `datexp` (the recording date string) and `mname`, taken from the index entry of each session. The
`days` field that exists on some `Imaging_Exp_info.npy` entries is deliberately *not* used.

ii.
```python
    for sid in session_ids:
        record = sources[sid]["record"]
        date = datetime.strptime(record["datexp"], "%Y_%m_%d").date()
        subject = str(record["mname"])
        parsed[sid] = (subject, date)
        first_date[subject] = min(date, first_date.get(subject, date))
```

iii. Docstring of `elapsed_days_by_session`: "The experiment table does not provide a complete
training-day counter (only selected later recordings have a `days` field). Dates are available for
every recording, so elapsed calendar day is the only uniform, non-imputed continuous measure across
supervised, unsupervised, grating, and naive mice." The agent verified this in step 47–48 by
printing `(exptype, sess#, days, stimtype, ...)` for all 89 recordings and seeing `days` populated
only for the `*_train2_after_learning` entries.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest imaging date is day 0, and every session is the integer number of
calendar days elapsed since then, as a float. The value is per-session and is broadcast across all
bins of all its trials. Observed range 0–92 days.

ii.
```python
    return {sid: float((date - first_date[subject]).days)
            for sid, (subject, date) in parsed.items()}
...
            trial_input[1] = day_by_session[sid]
...
            "training_day_definition": "elapsed calendar days since the subject's earliest imaging session",
```

iii. As above: calendar date is the only field available for every recording. The definition is
written into the metadata so a downstream user knows it is elapsed calendar days, not a count of
training sessions or of the mouse's true training day (training began ~2 weeks before the first
recording, per `methods.txt`).

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `Trial_start_time` (per-trial datenum of corridor entry) and `ft` (per-frame datenum). The frame
variant `StartFr` is not used.

ii.
```python
            time_from_start = ((times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY).astype(np.float32)
```

iii. Same as 3-a: the agent checked in step 22 that `ft[StartFr] - Trial_start_time` is within a
quarter of a frame, so the recorded timestamp and the frame-index representation agree; the
timestamp is used directly.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft[frame] - Trial_start_time[trial]) * 86400`, in seconds, float32, per frame. It is positive
after corridor entry (the first retained frame of a trial is always after entry, so the minimum over
the dataset is +6.0e-5 s). Maximum 1765.2 s, again from stalled trials.

ii.
```python
            trial_input[2] = time_from_start
```

iii. No justification beyond the unit conversion; it is a direct time difference against the trial's
alignment event.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed from `times = beh["ft"][frames]` for the same `frames`/`columns` that produce the neural
matrix, so it is element-wise aligned and of identical length. Because non-running frames are
dropped, consecutive elements are not equally spaced; this input is what tells the decoder the real
elapsed time of each column.

ii.
```python
            frames = frame_indices[columns]
            times = beh["ft"][frames]
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
            trial_input[2] = ((times - beh["Trial_start_time"][trial]) * SECONDS_PER_DAY)
```

iii. Frame-indexed alignment for every stream, as in 3-c.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
            trial_input[3] = float(beh["isRew"][trial])
```

iii. Not discussed; `isRew` is a direct read of the required quantity. (The agent also carried
`Reward_Mode` per session into `session_info` as context.)

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast from bool to float (0.0/1.0) and broadcast across every bin of the trial, as one row of the
`(4, n_timepoints)` float32 input matrix. No other processing. Observed range 0–1.

ii.
```python
            trial_input = np.empty((4, ntime), dtype=np.float32)
            ...
            trial_input[3] = float(beh["isRew"][trial])
```

iii. None given; none needed.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the wall texture. (`WallType`, `ft_WallID`, `UniqWalls` and
`stim_id` are not used; `stim_id` is masked in the swap sessions.)

ii.
```python
def visual_vocabulary(behaviors: dict) -> list[str]:
    return sorted({str(wall) for beh in behaviors.values() for wall in beh["WallName"]})
...
            trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
```

iii. Not argued explicitly. The agent counted the wall names over the whole dataset in step 53–54:
`leaf1 9736, circle1 9393, leaf2 4841, rock1 2743, wood1 2517, circle2 1836, wood2 1592, leaf3 1545,
leaf1_swap2 821, leaf1_swap1 818, wood5 567, rock2 535, wood1_swap2 421, circle3 390, wood1_swap1
355` — 15 distinct names — and took the vocabulary from that count.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The 15 distinct `WallName` strings found across the whole dataset are sorted and used **as-is** as
the 15 category labels; no collapsing to the four base textures (circle / leaf / rock / wood) is
performed, so `circle1`, `circle2`, `circle3` are three separate classes, and `leaf1`,
`leaf1_swap1`, `leaf1_swap2` are three more. The per-trial index into that vocabulary is broadcast
over all bins of the trial. Resulting class frequencies range from 0.8% to 26.4% of bins.

ii.
```python
    stimulus_values = visual_vocabulary(behaviors)
    stimulus_to_id = {name: idx for idx, name in enumerate(stimulus_values)}
...
            trial_output[0] = stimulus_to_id[str(beh["WallName"][trial])]
...
        "output_values": [stimulus_values, ...]
```

iii. No justification is recorded in the code comments, the metadata, or the trajectory for the
choice of granularity; the agent simply adopted the raw `WallName` vocabulary it had enumerated.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
        lick_frame = beh["LickFr"]
        lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
```

iii. Not discussed; `LickFr` is already expressed on the imaging-frame grid, so it is the natural
source.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Non-finite lick entries are dropped, the remaining fractional frame numbers are truncated to the
frame they fall in, and a bin is labelled 1 if any lick falls in it, 0 otherwise — a binary
time series, `output_values = ["not licking", "licking"]`. Licks that fall on frames removed by the
running/corridor mask are simply never matched. Resulting positive rate 3.69% of bins.

ii.
```python
            trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Not discussed. (Note that the `ft_move > 0` mask removes exactly the periods in which task mice
stopped to collect reward, which `methods.txt` describes as the paper's intent, and which removes
some licking frames.)

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `np.isin(frames, lick_frame)` is evaluated on the same absolute frame indices that select the
neural columns, so the lick flag is element-wise aligned with the neural matrix and has the trial's
length.

ii.
```python
            frames = frame_indices[columns]
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
            trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Frame-indexed alignment for every stream, as in 3-c.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the position inside the corridor at each imaging frame, in decimeters, together with the
per-session `Texture_Length` field (40 dm in every session) used to set the bin width.

ii.
```python
    out["texture_length_dm"] = float(beh["Texture_Length"])
...
            position_bin = np.floor(beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)).astype(np.int64)
```

iii. Code comment: "Texture_Length is 40 decimeters. Using the recorded setting keeps the four bins
exactly 1 m even if a future source file differs." Metadata records
`"position_source_units": "decimeters"`. The agent verified in step 22 that within the mask `ft_Pos`
ranges 0.0004–39.998 dm.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by `Texture_Length/4 = 10` dm and floored, giving an integer
0–3; the value is computed per frame (time-varying) and stored as one row of the int64 output
matrix. Resulting bin occupancies 25.00 / 24.87 / 24.96 / 25.17%.

ii.
```python
            position_bin = np.floor(beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0)).astype(np.int64)
            position_bin = np.clip(position_bin, 0, 3)
            trial_output[2] = position_bin
```

iii. As above; four equal 1 m bins over the 4 m textured corridor, as the instructions require. Only
corridor (`ft_CorrSpc`) frames are retained, so positions never legitimately exceed 40 dm.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-width spatial thresholds at 10, 20 and 30 dm (1, 2, 3 m), with a defensive clip to
the range 0–3 so an exact 40.0 dm sample cannot produce a fifth bin. Labels
`["0-1 m", "1-2 m", "2-3 m", "3-4 m"]`.

ii.
```python
            position_bin = np.clip(position_bin, 0, 3)
...
        "output_values": [..., ["0-1 m", "1-2 m", "2-3 m", "3-4 m"], ...]
```

iii. Directly from the decoder-task specification: "Position in corridor discretized into 4
equal-length, 1-m-long spatial bins".

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, indexed with the same `frames` array used for the neural
columns, so it is element-wise aligned and of the trial's length.

ii.
```python
            frames = frame_indices[columns]
            position_bin = np.floor(beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0))
```

iii. Frame-indexed alignment for every stream, as in 3-c.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
        speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
...
            speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False)
```

iii. Not discussed; it is the requested quantity, read directly.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first pass over all 89 sessions concatenates `ft_RunSpeed` over exactly the frames the mask
retains and computes the 25/50/75% quantiles of that pooled distribution — one **dataset-wide** set
of thresholds rather than per-session ones. The run aborts if the three edges are not strictly
increasing. Each frame is then assigned to a bin by `np.digitize`. Resulting occupancies are
25.000% per bin. The edges are stored in the metadata.

ii.
```python
def global_speed_edges(behaviors: dict) -> np.ndarray:
    """Compute dataset-wide quartiles over exactly the retained samples."""
    speeds = []
    for beh in behaviors.values():
        mask = valid_frame_mask(beh, len(beh["ft"]))
        speeds.append(beh["ft_RunSpeed"][: len(mask)][mask].astype(np.float32))
    all_speeds = np.concatenate(speeds)
    edges = np.quantile(all_speeds, (0.25, 0.50, 0.75))
    if not np.all(np.diff(edges) > 0):
        raise RuntimeError(f"Running-speed quartiles are not distinct: {edges}")
    return edges.astype(np.float64)
```
```python
            "running_speed_quartile_edges": speed_edges.tolist(),
```

iii. Step 30: "computes running-speed quartiles globally over the retained samples". The docstring
stresses that the quantiles are taken over "exactly the retained samples", so the four bins each hold
25% of the data that is actually written. The agent measured the pooled quartiles in step 53–54
(`[-19.17, 12.42, 25.35, 40.85, 163.49]`) before committing to the approach, confirming the edges are
distinct (the mask removes most of the zero-speed ties that would otherwise make equal-size
value-threshold bins impossible).

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(..., right=False)` against the three global edges, giving classes
`["lowest 25%", "25-50%", "50-75%", "highest 25%"]`. Thresholds are on *value*, not on per-session
rank, so a given speed means the same class in every session, but individual sessions are not exactly
balanced.

ii.
```python
            speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False).astype(np.int64)
            trial_output[3] = speed_bin
```

iii. Follows the decoder-task requirement "Running speed discretized into 4 bins, each corresponding
to 25% of the data", read as a property of the pooled dataset.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. Indexed with the same `frames` array used for the neural columns, so it is element-wise aligned
and of the trial's length.

ii.
```python
            frames = frame_indices[columns]
            speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False)
```

iii. Frame-indexed alignment for every stream, as in 3-c.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Two classes of handling. (a) Benign truncation/cleaning: the behavior streams can be longer than
the imaging, so every stream is cut to `min(n_imaged_frames, len(ft), len(ft_trInd),
len(ft_CorrSpc), len(ft_move))`; `n_imaged_frames` itself is the minimum frame count across imaging
planes; frames whose `ft_trInd` is NaN, negative or ≥ `ntrials` are dropped; non-finite `LickFr`
entries are dropped; `ft_Pos` bins are clipped to 0–3. (b) Everything else is **fail-fast**: a
missing behavior key, an experiment-table/spike-file mismatch, a retinotopy/neuron count mismatch, a
non-float32 spike array, a session with no valid frames, non-chronological trial indices, fewer than
two usable trials, or non-distinct speed quartiles all raise and abort the whole conversion rather
than skipping the session. In practice none of these fired: all 89 sessions converted.

ii.
```python
    nframes = min(nframes, len(beh["ft"]), len(beh["ft_trInd"]),
                  len(beh["ft_CorrSpc"]), len(beh["ft_move"]))
    valid_trial = np.isfinite(trial) & (trial >= 0) & (trial < beh["ntrials"])
```
```python
        nframes = min(plane.shape[1] for plane in raw_planes)
        lick_frame = lick_frame[np.isfinite(lick_frame)].astype(np.int64)
```
```python
        if key not in behavior_file:
            raise KeyError(f"{key!r} is absent from {path}")
        if len(iarea) != raw_neuron_count:
            raise RuntimeError(f"Retinotopy/neural neuron mismatch in {sid}: ...")
```

iii. The consistency checks are written as hard assertions so that a silent mis-pairing of
retinotopy, behavior and spikes cannot reach the output. The agent's progress notes (steps 33, 38,
42) report each batch of sessions "passed shape and retinotopy consistency checks" — the checks were
used as the running validation of the conversion.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 434 GB of `spk/*_neural_data.npy` files (89 pickled `.npy` dicts, 6–15 GB each) and
writing the 142 GB output pickle. Within a session, the fancy-index selection
`plane[np.ix_(local_keep, frame_indices)]` materializes ~46k neurons × ~9k frames per session and is
the main CPU cost; the per-trial `np.ascontiguousarray` copy then rewrites the same data a second
time. The pre-pass `global_speed_edges` also touches every behavior session before the main loop.

ii.
```python
        raw_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
            selected_planes.append(plane[np.ix_(local_keep, frame_indices)])
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent measured the source volume up front (step 19: "89 434.17" GB) and predicted the
output size (step 30: "roughly 150 GB"), i.e. it knew the run was I/O-bound. The plane-by-plane
selection carries the comment "This avoids first concatenating the full (often multi-gigabyte)
unfiltered recording", an explicit memory/time trade-off.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop rescans the whole session each iteration: `np.flatnonzero(trial_ids == trial)`
is O(n_trials × n_kept_frames) where one `np.argsort`/`np.searchsorted` over `trial_ids` (already
sorted) would give all trial boundaries in one pass. `np.isin(frames, lick_frame)` is likewise
executed once per trial instead of building one per-frame lick flag per session, and
`np.digitize(...)`/`np.floor(ft_Pos/...)` are called per trial instead of once per session and then
sliced. The neuron-keep `np.flatnonzero(keep_neuron[offset:...])` per plane is cheap and fine.

ii.
```python
        for trial in kept_trials:
            columns = np.flatnonzero(trial_ids == trial)
            ...
            position_bin = np.floor(beh["ft_Pos"][frames] / (beh["texture_length_dm"] / 4.0))
            speed_bin = np.digitize(beh["ft_RunSpeed"][frames], speed_edges, right=False)
            trial_output[1] = np.isin(frames, lick_frame).astype(np.int64)
```

iii. Not discussed by the agent. These costs are negligible next to the 434 GB of I/O.

## 12-c. What processing does the code repeat multiple times?

i. `valid_frame_mask` is computed twice for every session — once inside `global_speed_edges` (over
the full behavior length) and again in the main loop (over the imaged length) — and the two passes
do not use the same frame count, so the quartile edges are derived from a slightly larger frame set
than the one actually written. The neural data of each trial is also copied twice (the `[:, columns]`
fancy index already allocates a new array, and `np.ascontiguousarray` then copies it again to change
the memory layout). `compact_behavior` copies every array field with `copy=True`, including fields
that are only read once.

ii.
```python
        mask = valid_frame_mask(beh, len(beh["ft"]))          # pre-pass, behavior length
...
        frame_mask = valid_frame_mask(beh, nframes)           # main loop, imaged length
```
```python
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])
```

iii. Not discussed. The double mask computation is the price of a dataset-wide speed quantile, which
needs a pre-pass; the `copy=True` compaction is deliberate, to let the multi-hundred-megabyte
behavior dicts be freed ("Keep only fields needed by this conversion, detached from the large dict").

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Little in the way of computed-then-discarded values, but several avoidable costs in what is
stored: the neural arrays are kept at the source float32 instead of float16, doubling the pickle to
142 GB and requiring ~151 GB of RAM in the decoder run; the categorical outputs (4 small integers
per bin) are stored as int64, 8× larger than needed; `np.ascontiguousarray` performs a whole extra
copy of every trial's neural matrix purely to change memory layout; `compact_behavior` copies
`Trial_start_time`/`SoundTime`/`isRew`/`WallName` arrays for the whole session when only per-trial
scalars are used; and `session_info` accumulates per-session diagnostics (`n_valid_frames`,
`n_neurons_raw`, `reward_mode`, `source_experiment_type`) that the decoder never reads.

ii.
```python
            trial_neural = np.ascontiguousarray(selected_neural[:, columns])   # float32, 142 GB total
            trial_output = np.empty((4, ntime), dtype=np.int64)
```

iii. No justification given for the dtypes; the agent anticipated the ~150 GB output (step 30) and
accepted it, later noting "Memory use is stable at about 151 GB, consistent with the pickle size"
(step 76). The extra `ascontiguousarray` carries the comment "Each trial owns a contiguous array so
the pickle is self-contained and does not retain a whole-session backing allocation per view" — a
real concern for views, though the fancy-index slice already owns its data.
