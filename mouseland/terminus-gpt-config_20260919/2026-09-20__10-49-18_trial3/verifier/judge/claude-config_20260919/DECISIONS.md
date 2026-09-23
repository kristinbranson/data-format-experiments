# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates recordings by globbing the spike directory (`/app/data/spk/*_neural_data.npy`), asserting exactly 89 files. It then reads **all 23** `beh/Beh_*.npy` behavior dictionaries once at startup (`load_behavior_views`) and groups every behavior key onto a "physical" recording id, obtained by stripping an analysis-view suffix `_swap1`/`_swap2` from the key. Every session must have at least one behavior view or the run aborts. Per session it then loads (a) the spike file (`obj["spks"]`, a list of three plane arrays), and (b) the retinotopy file `retinotopy/<mouse>_<YYYY_MM_DD>_trans.npz` for `iarea`. `beh/Imaging_Exp_info.npy` is read only to attach descriptive metadata (`experiment_group`, `sess#`, `rewType`, `Gender`), not to drive the iteration.

ii.
```python
def neural_files():
    return {
        os.path.basename(p).removesuffix("_neural_data.npy"): p
        for p in glob.glob(os.path.join(ROOT, "spk", "*_neural_data.npy"))
    }

def load_behavior_views(valid_ids):
    """Load all analysis views and group them by unique physical recording."""
    views = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(ROOT, "beh", "Beh_*.npy"))):
        group = os.path.basename(path)[4:-4]
        obj = np.load(path, allow_pickle=True).item()
        for key, beh in obj.items():
            sid = physical_id(key)
            if sid in valid_ids:
                views[sid].append((group, key, beh))
    missing = sorted(set(valid_ids) - set(views))
    if missing:
        raise RuntimeError(f"No behavior found for sessions: {missing}")
    return views
```
```python
    if len(paths_all) != 89:
        raise RuntimeError(f"Expected 89 unique neural sessions, found {len(paths_all)}")
```

iii. From CONVERSION_NOTES Step 2/4: `Imaging_Exp_info.npy` has "23 experiment groups and 142 analysis entries, but only 89 unique physical session IDs"; recordings recur under several task analyses (test1/test2/train2) and test3 exposes duplicate `_swap1`/`_swap2` views. The AI therefore treats the 89 spike files as the authoritative list of recordings and deduplicates behavior onto them, matching the paper's "89 recordings in 19 mice". Behavior files are read once for all their sessions to avoid repeated multi-hundred-MB I/O.

## 1-b. How are the data split into subjects?

i. The subject is the mouse-name prefix of the session id (`sid.split("_")[0]`). `subjects` is the sorted unique list (19 mice); `subject_idx` is the index of each session's mouse into that list, in the same order as `neural`/`input`/`output` (sessions are processed in sorted session-id order, so sessions of a mouse are contiguous and chronological).

ii.
```python
    subjects = sorted({sid.split("_")[0] for sid in selected})
    subject_map = {v: i for i, v in enumerate(subjects)}
    ...
    "subject_idx": np.asarray([subject_map[s.split("_")[0]] for s in selected], dtype=np.int64),
```
A stricter parser is used elsewhere for the same purpose:
```python
def parse_session_id(sid):
    """Parse <mouse>_<YYYY>_<MM>_<DD>_<block> without splitting date underscores."""
    m = re.fullmatch(r"(.+?)_(\d{4}_\d{2}_\d{2})_([^_]+)", sid)
```

iii. Step 2 of CONVERSION_NOTES lists the 19 mice found this way (DR10 … VR2) and notes the exact agreement with the paper's "89 recordings in 19 mice". The mouse name is already embedded in the file name / `mname`, so no derivation is needed. (An early iteration used `rsplit('_', 2)`, which misparsed the underscore-delimited date; this was found in Step 6 and replaced by the regex parser.)

## 1-c. How are the data split into sessions?

i. A session is one physical recording `<mouse>_<YYYY_MM_DD>_<block>` — exactly one spike file. Multiple behavior *views* of the same recording (different experiment groups, and `_swap1`/`_swap2` keys) are collapsed onto that single session; the AI verifies the duplicate views carry identical physical streams (`StartFr`, `GrayFr`, `SoundFr`, `isRew`, `ft_Pos`, `ft_RunSpeed`, same `ntrials` and frame count) and then uses the first view. 89 sessions result, all retained.

ii.
```python
def physical_id(key):
    """Remove analysis-view suffixes, leaving mouse_date_block."""
    return re.sub(r"_swap[12]$", "", key)

def choose_behavior(session_views):
    """Choose one physical stream; duplicate views differ only in semantic labels."""
    b0 = session_views[0][2]
    n = int(b0["ntrials"]); nf = len(b0["ft"])
    for _, _, b in session_views[1:]:
        if int(b["ntrials"]) != n or len(b["ft"]) != nf:
            raise ValueError("Duplicate views differ in physical stream dimensions")
        for key in ("StartFr", "GrayFr", "SoundFr", "isRew", "ft_Pos", "ft_RunSpeed"):
            if not np.allclose(np.asarray(b0[key]), np.asarray(b[key]), equal_nan=True):
                raise ValueError(f"Duplicate views differ in {key}")
    return b0
```

iii. Step 4: "Deduplicate behavior views by physical `<mouse>_<date>_<blk>` ID; retain 89 sessions … These are not additional recordings or trials." The AI additionally *proves* the views are physically identical rather than assuming it, by `np.allclose` on the six core streams.

## 1-d. How are the data split into trials?

i. Each session's trials are the `ntrials` declared by the behavior dictionary. A trial is the contiguous frame window `[ceil(StartFr), ceil(GrayFr))` — corridor entry through the exclusive start of the grey space — clipped to the number of imaged frames. Trials keep their natural, variable length; nothing is padded, truncated to a common window, or cut in the middle. A window with `end <= start` aborts the run.

ii.
```python
def trial_bounds(beh, nfr):
    starts = np.ceil(np.asarray(beh["StartFr"], dtype=np.float64)).astype(np.int64)
    ends = np.ceil(np.asarray(beh["GrayFr"], dtype=np.float64)).astype(np.int64)
    starts = np.clip(starts, 0, nfr)
    ends = np.clip(ends, 0, nfr)
    if np.any(ends <= starts):
        bad = np.flatnonzero(ends <= starts)
        raise ValueError(f"Invalid trial windows: {bad[:20].tolist()}")
    return starts, ends
```
```python
    for i, (a, z) in enumerate(zip(starts, ends)):
        frames = np.arange(a, z, dtype=np.float64)
        T = z - a
```

iii. Step 4/Step 10: "`ceil(StartFr)` is the first frame whose `ft_trInd` belongs to the new trial, while `floor(StartFr)` includes the preceding trial. `ceil(GrayFr)` is the exclusive end of the 4 m visual corridor; this yields source positions approximately 0–40 and excludes the 20-unit grey space." Spot checks confirmed every included frame's `ft_trInd` equals the target trial. All 38,110 windows were audited for positive length.

## 1-e. How are trials filtered based on quality controls?

i. **No trial quality filtering is applied.** All 38,110 trials of all 89 sessions are kept, including trials in which the animal stalled inside the corridor (maximum retained trial = 5,607 frames ≈ 29 min; session-level maxima of 1,000–2,300 frames are common). The only implicit "filter" is structural: a window must have positive length, and any violation raises rather than drops. No session is dropped either (every session has ≥ 84 trials).

ii. There is no filtering code; the loop runs over every trial:
```python
    starts, ends = trial_bounds(beh, nfr)
    ...
    for i, (a, z) in enumerate(zip(starts, ends)):
```
and the documented decision is explicit:
> **No invented filtering**: Retain all 89 sessions, all 38,110 valid trials, all concatenated Suite2p traces, and rare long/stationary trials because neither code nor paper specifies exclusion.

iii. Step 4: "Trial duration | No global duration filter | Median 23 frames; rare stalls up to 5,607 frames | No exclusion rule stated | Retain all 38,110 valid windows; do not invent a duration filter." Step 10 adds: "Extremely long trial windows are genuine stalled runs. They are retained because frame trial IDs remain correct and no paper/code curation rule excludes them."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<session_id>_neural_data.npy`, a list of exactly three (plane) neuron × frame arrays, concatenated along the neuron axis in the released order — the same order the retinotopy `iarea` vector indexes. The per-neuron brain region comes from `iarea` in `retinotopy/<mouse>_<YYYY_MM_DD>_trans.npz`, and the vector length is asserted to equal the concatenated neuron count.

ii.
```python
    obj = np.load(path, allow_pickle=True).item()
    parts = obj["spks"]
    if len(parts) != 3 or len({x.shape[1] for x in parts}) != 1:
        raise ValueError(f"Unexpected neural components for {sid}")
    nneu, nfr = sum(x.shape[0] for x in parts), parts[0].shape[1]
    ...
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
    ...
    with np.load(rp) as r:
        region_idx = map_regions(r["iarea"])
    if len(region_idx) != nneu:
        raise ValueError(f"Retinotopy/neural mismatch for {sid}: {len(region_idx)} != {nneu}")
```

iii. Step 1/2: the reference `load_spk` "concatenates all three supplied arrays"; the third component may have 0–2 extra rows, which is valid because "raw retinotopy length equals the sum of all three components". The resulting per-session neuron range (20,547–89,577) reproduces the paper's stated range exactly, which the AI used as its main loading sanity check.

## 2-b. How is the `neural` data processed?

i. No processing at all: no dF/F, no deconvolution, no normalization, no smoothing, no neuron subsampling. Only the retained trial columns are copied out, stored as **float32**, with variable trial length. The concatenation is done per trial slice rather than on the whole session, which the AI proves is mathematically identical to `concatenate(parts)[:, a:z]` and avoids a multi-GB full-session copy.

ii.
```python
        # Equivalent to reference concatenate(parts, axis=0)[:, a:z], without
        # materializing excluded inter-trial and gray-space frames.
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```

iii. Step 3: "Paper analyses use the supplied non-negative deconvolved fluorescence/spike traces directly. No delta-F/F recomputation is appropriate." Step 10 records a raw `np.allclose` check of converted trial slices against `np.concatenate(raw_spks)[:, start:end]`, which passed exactly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are dropped.** All 4,691,034 released ROIs are kept. `iarea` is mapped with the reference `neu_area_ID` codes (V1=8; mHV=0,1,2,9; lHV=5,6; aHV=3,4) and every ROI that falls outside those codes is assigned to an explicit fifth region called `unassigned` (585,641 neurons) rather than being removed. `brain_regions` therefore has 5 entries.

ii.
```python
REGIONS = ["V1", "mHV", "lHV", "aHV", "unassigned"]

def map_regions(iarea):
    """Exact reference neu_area_ID mapping plus explicit unassigned class."""
    a = np.asarray(iarea)
    out = np.full(a.shape, 4, dtype=np.int16)
    out[a == 8] = 0
    out[np.isin(a, [0, 1, 2, 9])] = 1
    out[np.isin(a, [5, 6])] = 2
    out[np.isin(a, [3, 4])] = 3
    return out
```

iii. Step 1/3: Suite2p already performed cell classification, neuropil correction and deconvolution before release, and "No universal neuron quality filter (`iscell`, SNR threshold, etc.) was found" in the reference code; d-prime/area selections are "analysis selections rather than recording-quality curation". Step 4: "raw `iarea` values outside these mappings should be retained as an explicit unassigned region rather than silently dropping neurons."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry: every trial array begins at the first imaging frame at or after `StartFr` (`ceil(StartFr)`) and runs to the corridor exit (`ceil(GrayFr)`), so column 0 of every trial is corridor entry. Trials are variable length, nothing is padded or truncated to a common window, and metadata declares `off_start = 0.0`, `off_end = None`.

ii.
```python
        "temporal_alignment_event": "visual corridor entry (StartFr; first included frame is ceil(StartFr))",
        "off_start": 0.0,
        "off_end": None,
        "trial_window": "[ceil(StartFr), ceil(GrayFr)); visual 4-m corridor only; variable duration",
```

iii. Step 5 Key Decision 1: "Use visual corridor entry through visual-corridor exit, `[ceil(StartFr), ceil(GrayFr))`, aligned to corridor entry. This excludes inter-trial/gray-space activity and preserves the full 4 m trial requested." `ceil` rather than `floor` was chosen after verifying that `floor(StartFr)` pulls in a frame belonging to the previous trial.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: one time bin = one imaging frame. The bin size is *measured* per session as the median `ft` difference converted from MATLAB datenum days to seconds (and range-checked to 0.2–0.5 s); it comes out at 314.5–315.3 ms per session (≈3.18 Hz). `metadata['time_bin_size']` is the median of the per-session values (≈314.7 ms). The same per-session `dt` is used to build the two time-valued inputs.

ii.
```python
def session_dt_seconds(beh):
    """Convert median MATLAB-datenum frame interval from days to seconds."""
    d = np.diff(np.asarray(beh["ft"], dtype=np.float64))
    d = d[np.isfinite(d) & (d > 0)]
    dt = float(np.median(d) * 86400.0)
    if not (0.2 < dt < 0.5):
        raise ValueError(f"Unexpected imaging frame interval {dt} s")
    return dt
```
```python
    dt_values = [x["frame_bin_ms"] for x in session_stats]
    ... "time_bin_size": float(np.median(dt_values)),
```

iii. Step 3/4: "Native `ft` differences are approximately 3.64e-6 days … multiplying by 86,400 gives about 0.315 s per imaging frame (~3.18 Hz)". Step 5 Key Decision 2: "Keep one bin per imaging frame (~315 ms), because neural and behavior are already synchronized at this rate and no paper temporal rebinning is specified." The paper's 60-position-bin interpolation is deliberately *not* used because the decoder task requires temporal, not spatial, alignment.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From the per-trial cue frame `SoundFr` and the frame index of each bin, scaled by the session frame interval `dt` (derived from `ft`).

ii.
```python
    sound = np.asarray(beh["SoundFr"], dtype=np.float64)
    ...
        frames = np.arange(a, z, dtype=np.float64)
        inp[0] = (sound[i] - frames) * dt
```

iii. Step 5 mapping table: "`SoundFr`, frame index → `input[0]` time to sound cue: `(SoundFr - frame_index) * session_dt_seconds`, signed continuous series … Positive before cue, zero near cue, negative after." The methods state the cue is presented in every imaging trial at a random position 0.5–3.5 m, so it is a per-trial event timestamp in imaging-frame coordinates.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. The fractional cue frame is subtracted from each bin's integer frame index and the difference is converted to seconds using the session's uniform median frame interval (rather than interpolating the actual `ft` timestamps). The result is a continuous, signed, time-varying series, positive before the cue and negative after, stored as float32. Over the whole dataset the range is [-1762.0, +722.7] s — the extremes coming from the un-filtered stalled trials.

ii.
```python
        inp[0] = (sound[i] - frames) * dt
```

iii. The AI treats imaging frames as uniformly sampled (verified: median interval 0.3146 s, range-checked per session), so a frame-index difference times `dt` is the elapsed time. The sign convention is documented as "time *to* cue": positive before, negative after.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is computed on exactly the same frame window `[a, z)` used to slice the neural columns of that trial, so it shares the trial's length and bin grid by construction; the code asserts equal lengths for neural/input/output for every trial.

ii.
```python
        frames = np.arange(a, z, dtype=np.float64)
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0)...
        inp[0] = (sound[i] - frames) * dt
        if not (nt.shape[1] == inp.shape[1] == out.shape[1]):
            raise AssertionError("Trial stream length mismatch")
```

iii. `SoundFr` is expressed in imaging-frame coordinates, so all streams live on the frame grid; the `--show-processing` plots show the cue series crossing zero inside the corridor window.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the recording date embedded in the session id (`<mouse>_<YYYY>_<MM>_<DD>_<block>`), parsed with `datetime.strptime`. `sess#` from `Imaging_Exp_info.npy` is deliberately **not** used (it conflicts between duplicate views and is missing for eight sessions); it is preserved only as descriptive metadata.

ii.
```python
def subject_day_values(session_ids):
    first = {}
    dates = {}
    for sid in session_ids:
        mouse, date, _ = parse_session_id(sid)
        d = datetime.strptime(date, "%Y_%m_%d").date()
        dates[sid] = d
        first[mouse] = min(first.get(mouse, d), d)
    return {sid: float((dates[sid] - first[sid.split("_")[0]]).days) for sid in session_ids}
```

iii. Step 4: "`sess#` is missing for 8 sessions and conflicts across duplicate views … Use continuous elapsed calendar days from each subject's earliest included recording. Preserve experiment-stage metadata separately."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest recording date **over all 89 recordings** is day 0, and every session's value is the number of *calendar* days elapsed since then (0 … 92). The scalar is broadcast across every bin of every trial of that session and stored as float32. Because the reference day set is computed over all sessions (`sorted(paths_all)`), sample and full runs agree.

ii.
```python
    day = subject_day_values(sorted(paths_all))
    ...
        inp[1] = day_value
```

iii. Step 5 Key Decision 5: "Calendar days since each mouse's earliest recording is reproducible, continuous, and complete." The task asks for a *continuous* per-trial "day of training", and elapsed days is the only continuous measure available consistently for all sessions.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From the per-trial corridor-entry frame `StartFr` (used at its raw fractional value) and the frame index of each bin, scaled by the session frame interval `dt`.

ii.
```python
        inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. Step 5 mapping table: "frame index, `StartFr` → `input[2]` time since trial start: `(frame_index - StartFr) * session_dt_seconds`, continuous series. Starts near zero; uses fractional event timing."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Elapsed seconds since corridor entry, computed as (bin frame index − fractional `StartFr`) × `dt`. Because the window starts at `ceil(StartFr)`, the first value is in (0, 0.315] s rather than exactly 0, and the series increases monotonically to the corridor exit. Dataset range: [0.0, 1763.9] s (the upper end again from unfiltered stalled trials). Stored as float32.

ii.
```python
        frames = np.arange(a, z, dtype=np.float64)
        inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. Using the fractional `StartFr` (instead of the ceiled integer) keeps the true sub-frame offset of corridor entry; the `--show-processing` plot shows the monotone ramp starting at ~0.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: it is built from the identical `[a, z)` frame window used for the neural slice, so it is aligned by construction and has the trial's length.

ii.
```python
        frames = np.arange(a, z, dtype=np.float64)
        T = z - a
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0)...
        inp = np.empty((4, T), dtype=np.float32)
        inp[2] = (frames - float(beh["StartFr"][i])) * dt
```

iii. All behavior events are given in imaging-frame coordinates, so everything is on the neural frame grid.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial flag `isRew`, which marks trials run in the rewarded corridor.

ii.
```python
    reward = np.asarray(beh["isRew"], dtype=np.int16)
    ...
        inp[3] = reward[i]
```

iii. Step 5 mapping table: "`isRew` → `input[3]` reward availability: Boolean 0/1 broadcast across trial. Means rewarded corridor, not actual reward delivery." Raw rate 4,336/38,110 = 11.38%, preserved in the conversion.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to numeric and broadcast of the per-trial scalar across all bins of the trial (stored in the float32 input array). Unrewarded cohorts (naive/unsupervised) come out all-zero, as expected.

ii.
```python
        inp[3] = reward[i]
```

iii. Per-trial variables are broadcast to the time axis because "the validator requires neural, input, and output to have identical timepoint counts within each trial" (Step 5 Key Decision 3).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `TrialStim`, merged across all duplicate behavior views of the session. For each trial the AI collects the set of non-placeholder `TrialStim` labels across views; if exactly one concrete label exists it is used; if the views conflict the run aborts; if **all** views hold the placeholder string `stimulus_of_trial`, it falls back to `WallName` (which in practice only happens for 309 `circle3` trials). `WallName` is otherwise not used.

ii.
```python
def merged_stimuli(session_views):
    """Merge semantic TrialStim labels across duplicate analysis views."""
    for i in range(n):
        concrete = {
            str(b["TrialStim"][i]) for _, _, b in session_views
            if str(b["TrialStim"][i]) != "stimulus_of_trial"
        }
        if len(concrete) > 1:
            raise ValueError(f"Conflicting stimulus labels at trial {i}: {concrete}")
        if concrete:
            label = next(iter(concrete))
        else:
            wall = {str(b["WallName"][i]) for _, _, b in session_views}
            label = next(iter(wall))
        if label not in STIM_TO_ID:
            raise ValueError(f"Unknown visual stimulus category {label!r}")
        labels.append(label)
```

iii. Step 4/5: "Test3 uses swap-specific dictionary keys … Merge concrete `TrialStim` labels across views. There are zero conflicts; 309 unresolved placeholders are consistently raw `circle3` and retain that label." Note that the AI's own Step 2 exploration had concluded the opposite — "`WallName` retains concrete visual identity on these trials and is the safer raw visual-category source" — but the final decision uses `TrialStim` as primary.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The merged label string is mapped to an integer index into an 8-value vocabulary `["circle1","circle2","circle3","leaf1","leaf2","leaf3","leaf1_swap1","leaf1_swap2"]`; unknown labels abort. The per-trial integer is broadcast across the trial's bins as `output[0]` (int16). Full-dataset frame fractions: circle1 0.331, circle2 0.055, circle3 0.007, leaf1 0.346, leaf2 0.164, leaf3 0.047, leaf1_swap1 0.025, leaf1_swap2 0.026. No grouping of variants into coarse texture families is done, and no `rock*`/`wood*` categories appear.

ii.
```python
STIM_VALUES = [
    "circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
    "leaf1_swap1", "leaf1_swap2",
]
STIM_TO_ID = {v: i for i, v in enumerate(STIM_VALUES)}
...
        out[0] = STIM_TO_ID[str(stimuli[i])]
```

iii. Step 5 Key Decision 4: "Merge all duplicate behavior views before choosing category. Concrete labels never conflict. Use raw `circle3` only for the 309 trials where all views contain placeholders." The AI treats `TrialStim` as the "semantic" label and keeps each texture crop (circle1/2/3, leaf1/2/3 and the two spatial shuffles of leaf1) as its own class.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the fractional imaging-frame coordinate of every lick in the session.

ii.
```python
    lick_frames = np.asarray(beh["LickFr"], dtype=np.float64)
    lick_session = np.zeros(nfr, dtype=np.int16)
    lick_idx = np.floor(lick_frames).astype(np.int64)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
    if len(lick_idx):
        lick_session[np.unique(lick_idx)] = 1
```

iii. Step 5 mapping table: "`LickFr` → `output[1]` licking: Binary raster; event assigned to `floor(LickFr)`, multiple events/frame collapse to 1."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length binary vector is built once: a frame is 1 if at least one lick falls inside it, 0 otherwise. Lick frames are floored (the imaging interval containing the event) and events outside `[0, n_neural_frames)` are discarded. Lick *counts* are not kept. Full-dataset frame fraction of licking = 0.035.

ii.
```python
        out[1] = lick_session[a:z]
```
(pre-rasterized once per session, see 8-a snippet)

iii. Step 5 Key Decision 6: "`LickFr` is a fractional frame coordinate; `floor` selects the imaging interval containing the lick. Use binary presence, not lick count." Pre-rasterizing once per session (rather than per trial) was one of the documented speed-ups.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The session-level raster is indexed with the same `[a, z)` window as the neural slice, so it is on the neural frame grid and has the trial's length.

ii.
```python
        out[1] = lick_session[a:z]
```

iii. `LickFr` is already expressed in imaging-frame coordinates; the `--show-processing` plot overlays the lick steps on the trial time axis.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the per-imaging-frame corridor position in source units (0–40 across the textured 4 m corridor, continuing to 60 through the 2 m grey space); the stream is truncated to the neural frame count before use.

ii.
```python
    pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    ...
        out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Step 4: "`Corridor_Length=60`, texture corridor ends near 40 … Physical visual corridor is 4 m". The AI verified positions inside its trial windows span ≈0–40.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Only discretization — no smoothing, no interpolation onto spatial bins (the paper's 60-bin interpolation is deliberately not used). Raw positions in the trial window are digitized into 4 classes and stored as int16 per bin, time-varying. Full-dataset fractions: 0.285 / 0.233 / 0.235 / 0.247.

ii.
```python
        # Source visual corridor is 0..40 (= four physical metres).
        out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```
```python
        "output_values": [..., ["0-1 m", "1-2 m", "2-3 m", "3-4 m"], ...],
```

iii. Step 4/5: "use 0–10, 10–20, 20–30, 30–40 source bins for 1 m categories", i.e. four equal-length 1 m bins as the Decoder Task requires. (Note: the metadata string `"source_position_scale": "15 source units per metre; visual corridor 0-40 source units"` is internally inconsistent with the implemented 10-units-per-metre binning; the data — `Texture_Length = 40` for a 4 m corridor — supports the implemented value, so only the metadata sentence is wrong.)

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. `np.digitize(pos, [10, 20, 30])` then `np.clip(..., 0, 3)`: `[0,10) → 0`, `[10,20) → 1`, `[20,30) → 2`, `[30, ∞) → 3`. The clip only matters for the small number of frames at or beyond 40 (≈0.3% of retained frames, where the last frame of a traversal has already crossed into the grey space); those are folded into the 3–4 m class.

ii.
```python
        out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Four equal-length 1 m bins are exactly what the Decoder Task specifies; the AI checked class ranges and the full-data distribution (all four classes well populated, none dominant).

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one sample per imaging frame; it is truncated to the neural frame count and then sliced with the same `[a, z)` window as the neural data, so it is aligned by construction.

ii.
```python
    pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    ...
        out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Step 4: "Behavior versus neural frames … Truncate all frame streams to neural frame count before trial extraction", mirroring the reference's `beh[...][:nfr]`. Step 12 re-audited three trials directly from raw files and confirmed monotone position within each window.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the per-imaging-frame running speed, truncated to the neural frame count.

ii.
```python
    speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    ...
        out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. Step 5 mapping table: "`ft_RunSpeed` → `output[3]` running speed: Global corridor-frame quartile thresholds".

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Before any session is converted, the AI makes a pass over **all selected sessions'** behavior, gathers `ft_RunSpeed` restricted to exactly the retained corridor windows (1,375,142 frames in full mode), checks all values are finite, and computes a single **global** set of thresholds with `np.quantile(values, [0.25, 0.5, 0.75])` = `[0.0, 8.3812388, 30.1893780]`. Negative speeds (tracking noise) are kept and fall in the lowest class. The thresholds are recorded in metadata.

ii.
```python
def compute_speed_thresholds(selected_ids, views, nfr_by_session):
    """Global quartiles over exactly the retained corridor imaging frames."""
    chunks = []
    for sid in selected_ids:
        b = choose_behavior(views[sid])
        nfr = min(int(nfr_by_session[sid]), len(b["ft_RunSpeed"]))
        starts, ends = trial_bounds(b, nfr)
        speed = np.asarray(b["ft_RunSpeed"][:nfr], dtype=np.float64)
        chunks.extend(speed[a:z] for a, z in zip(starts, ends))
    values = np.concatenate(chunks)
    q = np.quantile(values, [0.25, 0.5, 0.75]).astype(np.float64)
    return q, int(values.size), (float(values.min()), float(values.max()))
```

iii. Step 5 Key Decision 7: "Compute global thresholds once from all retained corridor frames, matching 'each corresponding to 25% of the data.' Keep negative estimated speeds in the lowest class rather than altering raw behavior." Computing over exactly the retained frames means excluded (grey-space, inter-trial) frames do not move the boundaries.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, [0, 8.3812, 30.1894])` then `clip(0, 3)`. Because a large point mass sits at exactly 0 and `np.digitize` is right-open, **all zero-speed frames land in class 1**, not class 0; class 0 ends up containing only the negative-speed frames. The realized full-dataset distribution is therefore **0.098 / 0.402 / 0.250 / 0.250** rather than four equal quarters. The AI detected this and chose not to break the ties.

ii.
```python
        out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```
```python
        "speed_quartile_thresholds": speed_q.tolist(),
        "output_values": [..., ["0-25%", "25-50%", "50-75%", "75-100%"]],
```

iii. Step 7/10: "The speed 25th percentile is exactly zero in this sample. Tied zero values cannot be split without arbitrary/random tie breaking, hence classes 0/1 are 12%/38% while upper classes are exactly 25% each. This is expected and preserves deterministic value-based bins." Step 10: "Deterministic value bins cannot split tied zeros; no random tie-breaking was introduced."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one sample per imaging frame; it is truncated to the neural frame count and sliced with the same `[a, z)` window as the neural data.

ii.
```python
    speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    ...
        out[3] = np.clip(np.digitize(speed[a:z], speed_q), 0, 3)
```

iii. All behavior frame streams share the imaging-frame grid, so slicing with the neural window is sufficient; the per-trial length assertion guarantees it.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several specific defences: (1) behavior frame streams are truncated to the neural frame count `nfr` (behavior runs 0–2 frames longer in many sessions); (2) trial windows are clipped to `[0, nfr]`; (3) licks outside `[0, nfr)` are dropped; (4) the frame interval is computed from finite, positive `ft` differences only and range-checked; (5) the three `spks` components may have unequal neuron counts (third has 0–2 extra rows) and are simply concatenated, with the total asserted equal to the retinotopy length; (6) `TrialStim` placeholders (`stimulus_of_trial`) fall back to `WallName`; (7) NaN `RewardFr` on unrewarded trials is never touched because reward availability comes from `isRew`. Genuinely anomalous structure is treated as fatal (raise) rather than silently repaired: non-3-part `spks`, shape drift, non-finite speeds, conflicting duplicate views, `end <= start` windows, unknown stimulus names, region/neuron length mismatch. A final `validate_local` re-checks shapes, finiteness and class ranges for every trial.

ii.
```python
    pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
    speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
    lick_idx = lick_idx[(lick_idx >= 0) & (lick_idx < nfr)]
```
```python
    starts = np.clip(starts, 0, nfr); ends = np.clip(ends, 0, nfr)
    if np.any(ends <= starts):
        raise ValueError(...)
```
```python
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite retained running speeds")
```

iii. Step 4: "Reference code explicitly slices behavior masks/streams to neural `nfr`; conversion must do the same." Step 2: the third `spks` component's extra rows are "valid: raw retinotopy length equals the sum of all three components". The AI reports the dataset is otherwise clean: all trial-sized arrays match `ntrials`, no missing start/end/sound frames, and every trial window is at least 26 frames before the end of the neural stream.

## 12-a. What are the most time-consuming steps of the code?

i. (1) Reading the 434 GB of `spk/*_neural_data.npy` object pickles — unavoidable, un-memory-mappable, and ~2–40 s per session (≈1,100 s total); (2) serializing the 296.4 GB output pickle — 434 s; (3) the up-front pass that loads all 23 behavior files (≈6.6 GB) and the global speed-threshold scan. Total full run: 1,609.8 s. In `--sample` mode there is an extra full read of each selected neural file in `inspect_neural_shapes` (shapes only), doubling the sample's neural I/O.

ii.
```python
    obj = np.load(path, allow_pickle=True).item()   # per session, multi-GB
```
```python
    with open(args.outpicklefile, "wb") as f:
        pickle.dump(data, f, protocol=4)
```

iii. Step 6/9: "Native object NPY files cannot be memory-mapped and total 434 GB; eager loading all sessions would be wasteful … Optimized full conversion took 1,609.8 s; final serialization of the unavoidable 296.435 GB complete dataset took 434.1 s. The runtime exceeded 15 minutes because preserving all released traces creates a 296 GB target, not because of redundant source work."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining Python-level loops are: the per-trial loop in `convert_session`, which calls `np.concatenate([x[:, a:z] for x in parts])` once per trial (3 × 38,110 slice copies and 38,110 concatenations) instead of grouping trial columns once per session; the per-trial generator in `compute_speed_thresholds`; and the per-trial `merged_stimuli` loop that builds a Python set per trial across views. The AI did vectorize the parts that matter most: time axes via `np.arange`, discretization via `np.digitize`, and lick rasterization once per session instead of once per trial. A first implementation that concatenated the whole session before slicing was removed after it was measured to be too slow.

ii.
```python
    for i, (a, z) in enumerate(zip(starts, ends)):
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```
```python
    out[2] = np.clip(np.digitize(pos[a:z], [10.0, 20.0, 30.0]), 0, 3)
```

iii. Step 6: "Optimized to concatenate the three reference components only for retained trial slices, avoiding a redundant full-session copy, and to pre-rasterize licks once per session"; the representative conversion dropped from 27.8 s to 12.1 s with byte-identical output. The remaining per-trial copy is inherent to producing per-trial arrays and is small next to file I/O.

## 12-c. What processing does the code repeat multiple times?

i. `choose_behavior` — which re-runs six full-length `np.allclose` comparisons over every duplicate view — is called up to three times per session (once when building `nfr_for_speed`, once in `compute_speed_thresholds`, once in `convert_session`). `trial_bounds` is likewise computed twice per session (threshold pass and conversion pass). All 23 behavior files (~6.6 GB of views) are loaded up front and held in memory for the whole run even in `--sample` mode, where only two sessions are needed. In `--sample` mode `inspect_neural_shapes` reads each selected neural file in full just to record its shape, and `convert_session` then reads the same file again. The AI documents the removal of the redundant full-session neural copy but does not flag these remaining repeats.

ii.
```python
        nfr_for_speed = {sid: len(choose_behavior(all_views[sid])["ft"]) for sid in selected}
    speed_q, n_speed, speed_range = compute_speed_thresholds(selected, all_views, nfr_for_speed)
    ...
def convert_session(...):
    beh = choose_behavior(session_views)
```
```python
        # Explicit pre-scan is useful for sample diagnostics. Full mode avoids
        # reading all 434 GB twice; shapes are inferred during the conversion load.
        shapes, source_dtypes = inspect_neural_shapes(paths)
```

iii. The AI's rationale for the sample-mode double read is explicit in the code comment ("useful for sample diagnostics"), and it is disabled for full mode precisely to avoid reading 434 GB twice. The repeated `choose_behavior`/`trial_bounds` work is behavior-only and negligible next to neural I/O, which is presumably why it was left in place.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI's position is that there is none — "No invented filtering: retain all 89 sessions, all 38,110 valid trials, all concatenated Suite2p traces". In practice some work is not needed downstream: neural traces are stored as **float32** where **float16** carries the deconvolved values adequately (this alone doubles the artifact to 296.4 GB and forced the decoder onto CPU after CUDA ran out of memory); the ~586k `unassigned` ROIs, the ~0.3% of grey-space frames captured by `ceil(GrayFr)` rounding, and the 5,607-frame stalled trials are all carried through the pipeline into the pickle; `inspect_neural_shapes` (sample mode) computes shapes/dtypes that are only printed; and per-session descriptive metadata (`experiment_views`, `source_neural_dtypes`) is written but unused by the decoder.

ii.
```python
        nt = np.concatenate([x[:, a:z] for x in parts], axis=0).astype(np.float32, copy=False)
```
```python
        st["experiment_views"] = meta_views.get(sid, [])
        st["source_neural_dtypes"] = source_dtypes[sid]
```

iii. Step 5 Key Decision 9: "Use float32 for neural/input and compact integer outputs/indices." Step 7: "Full size is expected to be large (hundreds of GB) because all 4.69 million released traces and all retained trial frames are intentionally preserved." Step 11 records the consequence: "Initial CUDA setup completed all 89 session projections but exhausted GPU memory before optimization. The provided decoder automatically retried on CPU."
