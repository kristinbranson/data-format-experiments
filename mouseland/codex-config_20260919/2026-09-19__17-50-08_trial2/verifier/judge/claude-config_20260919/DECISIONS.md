# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads from the three directories under `/app/data`: `beh/` (behavior), `spk/` (deconvolved
calcium traces) and `retinotopy/` (visual area per neuron). Unlike the reference, it does **not**
drive the iteration from the master index `beh/Imaging_Exp_info.npy`; instead it globs every
`beh/Beh_*.npy` file, loads each once, and collapses the session keys onto "physical" recording
bases (`<mouse>_<date>_<block>`) by stripping the `_swap1`/`_swap2` suffix. `Imaging_Exp_info.npy`
is loaded separately, only to collect registry aliases per base and to resolve the training-day
input. The list of sessions actually converted is the set of `spk/*_neural_data.npy` file stems,
sorted. Before converting, the script asserts that the spike bases, behavior bases, and registry
bases are all the same 89-element set, and that a retinotopy file exists for each. Per session the
spikes are loaded plane-by-plane and the retinotopy `.npz` is read (the retinotopy filename drops
the block suffix). All 89 behavior dictionaries are held in memory for the whole run; spike arrays
are released per session.

ii.
```python
def behavior_catalog() -> ...:
    for path in sorted((DATA_ROOT / "beh").glob("Beh_*.npy")):
        loaded = np.load(path, allow_pickle=True).item()
        for key, beh in loaded.items():
            base = physical_base(key)
            if base in sessions:
                old = sessions[base]
                checks = (
                    int(old["ntrials"]) == int(beh["ntrials"]),
                    len(old["ft"]) == len(beh["ft"]),
                    np.array_equal(old["WallName"], beh["WallName"]),
                    np.array_equal(old["isRew"], beh["isRew"]),
                )
                if not all(checks):
                    raise ValueError(f"Conflicting behavior aliases for {base}")
                continue
            sessions[base] = beh
```
```python
registry = np.load(DATA_ROOT / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
for experiment_type, entries in registry.items():
    for entry in entries:
        base = f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"
```
```python
spk_bases = {p.name.removesuffix("_neural_data.npy") for p in (DATA_ROOT / "spk").glob("*_neural_data.npy")}
if spk_bases != set(behaviors) or spk_bases != set(aliases):
    raise ValueError(f"Catalog mismatch: spk={len(spk_bases)}, behavior={len(behaviors)}, registry={len(aliases)}")
```
```python
path = DATA_ROOT / "spk" / f"{base}_neural_data.npy"
neural_dict = np.load(path, allow_pickle=True).item()
planes = neural_dict["spks"]
...
retino = np.load(retinotopy_path(base), allow_pickle=True)
iarea = np.asarray(retino["iarea"])
```

iii. From CONVERSION_NOTES Step 4/5: "Registry contains 23 overlapping experiment labels ... Collapse
aliases by `<mouse>_<date>_<block>` and include each physical recording once; use `WallName`
directly so swap aliases are unnecessary." The AI notes the registry has 142 entries but only 89
physical recordings, matching the paper's "89 recordings in 19 mice", so it treats the spike-file
inventory as the authoritative session list and uses the three-way catalog equality as a sanity
check. It uses the reference loader's `np.load(..., allow_pickle=True).item()['spks']` convention
and plane concatenation order.

## 1-b. How are the data split into subjects (mice)?

i. The subject is the mouse-name prefix of the session base, i.e. everything before the first
underscore. `subjects` is the sorted unique set (19 mice) and `subject_idx` indexes into it in the
same order as the session list. This is numerically identical to the reference, which takes the
mouse from the registry field `mname`.

ii.
```python
subjects = sorted({base.split("_")[0] for base in bases})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx = np.asarray([subject_lookup[base.split("_")[0]] for base in bases], dtype=np.int16)
```

iii. CONVERSION_NOTES Step 2 enumerates the 19 mice (DR10 … VR2) with their session counts and
matches them to the paper's "89 recordings in 19 mice". The mouse name is part of the session id, so
no separate split is needed.

## 1-c. How are the data split into sessions?

i. A session is one physical recording: one mouse, one date, one block. Behavior keys that carry a
`_swap1`/`_swap2` suffix are alternate analysis views of the *same* recording, so the AI strips the
suffix and keeps the first alias encountered (verifying that `ntrials`, `len(ft)`, `WallName` and
`isRew` agree across aliases). Registry entries that repeat the same recording under several
experiment types are collapsed the same way. Result: 89 sessions, matching the spike-file count and
the paper. Sessions are converted in sorted-base order.

ii.
```python
def physical_base(session_key: str) -> str:
    """Remove the swap-analysis suffix, which is not a separate recording."""
    return re.sub(r"_swap[12]$", "", session_key)
```
```python
bases = sorted(spk_bases)
```

iii. CONVERSION_NOTES Step 10: "Behavior aliases could double-count 25,067 trials: resolved by
physical session base; raw/paper session count now matches 89." The AI also chose `WallName` over
`stim_id` for stimulus labels specifically so that the swap-specific aliases are not needed.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials that the behavior declares, indexed by the per-frame label
`ft_trInd`. A frame belongs to a trial if `ft_trInd == trial` **and** `ft_CorrSpc` (inside the 4 m
textured corridor) **and** `ft_move > 0` (the VR was moving, i.e. the animal was running). This adds
the movement mask on top of the reference conversion's `ft_trInd & ft_CorrSpc`. Trials are kept at
their own variable length; nothing is padded. The gray space is excluded, so position stays in
[0, 40) decimeters. Because interior stationary frames are removed, the retained samples of a trial
are not necessarily contiguous imaging frames.

ii.
```python
common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])
trial_stamp = np.asarray(beh["ft_trInd"][:common_nframes])
corridor = np.asarray(beh["ft_CorrSpc"][:common_nframes], dtype=bool)
movement = np.asarray(beh["ft_move"][:common_nframes]) > 0
valid = np.isfinite(trial_stamp) & corridor & movement
...
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Running-frame curation: Apply the paper/code mask
`ft_move>0` within `ft_CorrSpc`. This best satisfies the critical reference-matching constraint and
avoids stationary reward-consumption confounds." This mirrors `utils.py:Get_dprime_selective_neuron`
(`fr_valid = VRmove & isCorridor  # only use activity inside the texture area plus mouse is
running`) and the same mask in `Get_coding_direction` / `Get_sort_spk`. Step 4 records the
consequence quantitatively (821,579 running-texture frames versus 1,373,170 texture frames) and the
AI's mitigation: "Preserve actual elapsed time in temporal inputs using `ft` timestamps, so missing
stationary intervals remain explicit."

## 1-e. How are trials filtered based on quality controls?

i. Essentially none beyond the frame mask. A trial is dropped only if it has **zero** retained frames
after truncation and the texture/running mask; in the full run this removed 0 of 38,110 trials, so
every source trial is present in the output. A session with fewer than two usable trials raises an
exception (which would abort the run) rather than being skipped. There is no trial-length outlier
filter — the reference drops 382 trials longer than the 99th percentile of traversal length; the AI
instead relies on `ft_move > 0` to strip out the stationary frames of "parked animal" trials, which
brings the longest retained trial down to 178 samples. Residual consequence: the time inputs of
those trials still span the real stall, giving `time_since_trial_start` up to 1765 s and
`time_to_sound_cue` down to −1763 s.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
if frames.size == 0:
    excluded_trials.append(trial)
    continue
```
```python
if len(neural_trials) < 2:
    raise ValueError(f"{base} has fewer than two usable trials after curation")
```

iii. CONVERSION_NOTES Step 5 Key Decision 7: "Truncate all frame streams to common neural/behavior
length; omit only trials with no remaining curated frame. Fail loudly on non-finite neural/input/
output values, unknown stimulus prefixes, region-length mismatch, or sessions with fewer than two
included trials." Step 10 edge-case review: "The longest raw trial resumes after a 1,737-s
stationary interval with the same trial stamp and monotonic position. This is an authentic recorded
pause, not misalignment; following the paper, stationary samples are omitted but the trial is not
discarded."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<base>_neural_data.npy`, a list of one (neurons × frames) float32 array per
imaging plane. Planes are taken in file order so that the concatenated neuron index lines up with
`iarea` from `retinotopy/<mouse>_<date>_trans.npz`. Rather than concatenating and then subsetting,
the AI preallocates the output for the retained neurons and fills it plane by plane, which is
numerically identical to the reference's `np.concatenate(planes['spks'], 0)[keep]`.

ii.
```python
selected = np.empty((int(mapped.sum()), nframes), dtype=np.float32)
src0 = 0; dst0 = 0
for plane in planes:
    src1 = src0 + plane.shape[0]
    local_mask = mapped[src0:src1]
    count = int(local_mask.sum())
    selected[dst0 : dst0 + count] = plane[local_mask]
    source_ranges.append((src0, src1))
    src0 = src1
    dst0 += count
```
```python
retino = np.load(retinotopy_path(base), allow_pickle=True)
iarea = np.asarray(retino["iarea"])
mapped, region_codes = area_codes(iarea)
```

iii. CONVERSION_NOTES Step 1: "Neural files already contain plane-wise `spks`; the reference loader
concatenates these without computing delta-F/F. Thus no new dF/F computation is indicated." Step 6:
the preallocation is an explicit memory optimisation — "Naively concatenating all raw planes before
region filtering would allocate an additional complete session matrix." The script also validates
that all planes have the same frame count and that the plane neuron total equals `len(iarea)`.

## 2-b. How is the `neural` data processed?

i. Not processed at all. The supplied Suite2p non-negative deconvolved traces are used as-is: no
dF/F, no deconvolution, no z-scoring, no smoothing, no normalisation, no spatial interpolation. Per
trial, the retained neurons' columns for that trial's frames are copied out. Values are stored as
**float32** (the source dtype), which makes the full pickle 141 GiB (151.5 GB).

ii.
```python
neural = spk[:, frames].copy()
if not np.all(np.isfinite(neural)):
    raise ValueError(f"Non-finite neural values in {base} trial {trial}")
```
```python
assert neural.dtype == np.float32 and neural.ndim == 2
```

iii. CONVERSION_NOTES Step 3: "Suite2p performed motion correction, ROI detection, cell
classification, neuropil correction, and non-negative spike deconvolution (decay time 0.75 s). All
paper analyses used the provided deconvolved traces." Step 5 Key Decision 8: "Neural and inputs
float32; outputs int8 … Do not compress/quantize neural values, preserving exact supplied float32
data and decoder accuracy." (The human reference instead stores float16 to cut file size.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. The only neuron filter is anatomical: a neuron is kept if its retinotopic area code `iarea` maps
into one of the four coarse visual areas V1 (8), mHV (0, 1, 2, 9), lHV (5, 6), aHV (3, 4). Codes −1
(unknown/outside) and 7 (outside visual cortex) are dropped. This keeps 4,105,393 of 4,691,034 ROIs,
exactly the reference number. No activity, firing-rate, SNR, or selectivity threshold is applied, and
no neuron is selected using the decoded labels. `brain_region_idx` is filtered with the same mask so
it stays aligned with the neural rows.

ii.
```python
def area_codes(iarea: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the mapped-cell mask and zero-based grouped-region indices."""
    iarea = np.asarray(iarea)
    mapped = np.isin(iarea, [0, 1, 2, 3, 4, 5, 6, 8, 9])
    codes = np.full(iarea.shape, -1, dtype=np.int8)
    codes[iarea == 8] = 0
    codes[np.isin(iarea, [0, 1, 2, 9])] = 1
    codes[np.isin(iarea, [5, 6])] = 2
    codes[np.isin(iarea, [3, 4])] = 3
    if np.any(codes[mapped] < 0):
        raise AssertionError("Mapped area without grouped brain-region code")
    return mapped, codes[mapped]
```

iii. CONVERSION_NOTES Step 4: "`neu_area_ID` groups IDs into V1/mHV/lHV/aHV; density code excludes
−1 and 7 … Retain only mapped V1/mHV/lHV/aHV neurons (4,105,393); exclude −1/7 as outside/unknown
visual cortex." Step 5 Key Decision 4: "Keep all Suite2p-classified cells in mapped visual cortex;
do not select neurons based on the target labels, activity, d-prime, or firing rate, avoiding
leakage and matching the absence of a general quality threshold in the reference." The grouping
reproduces `utils.py:neu_area_ID` exactly.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is entry into the 0–4 m textured corridor (trial start). Each trial's array starts
at the first retained frame of that trial (the first frame that is inside the texture *and* moving)
and ends at the last such frame, so trials have variable length (mean 22.3, min 11, max 178 samples).
Nothing is cut to a common window and nothing is padded. Metadata declares
`temporal_alignment_event = "entry into the 0-4 m textured visual corridor"`, `off_start = 0.0`,
`off_end = None`. Because of the `ft_move` mask, the first sample is the first *moving* corridor
frame rather than the entry frame itself, and interior frames may be missing.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
```
```python
"temporal_alignment_event": "entry into the 0-4 m textured visual corridor",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 3: "Trials are naturally aligned to entry into each textured corridor
(`Trial_start_time`, `StartFr`, or the transition in `ft_trInd`/`ft_CorrSpc`). The requested four
1-m position classes imply restricting decoder timepoints to the 0–4 m textured corridor, not the
following 2-m gray space." Step 5 Key Decision 10 fixes the `(4, T)` input/output shape so every
column lines up with a neural column.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning, resampling, or spatial interpolation. Samples are the native two-photon imaging
frames at 3.17 Hz, so `time_bin_size` is 1000/3.17 = 315.46 ms; this is recorded in metadata along
with `native_frame_rate_hz`. The AI explicitly declined the reference's 60-bin/6-m spatial
interpolation because the decoder task asks for time-varying outputs on the neural grid. Caveat the
AI documents itself: since stationary frames are dropped, consecutive samples within a trial are not
always 315.46 ms apart, so the declared bin size is the frame interval rather than the true
inter-sample spacing; the time inputs are computed from real timestamps so the gaps remain visible.

ii.
```python
FS_HZ = 3.17
MS_PER_FRAME = 1000.0 / FS_HZ
```
```python
"time_bin_size": MS_PER_FRAME,
"native_frame_rate_hz": FS_HZ,
"time_sampling_note": (
    "Samples are native 3.17-Hz imaging frames; stationary frames are omitted per "
    "the reference analysis, and timestamp inputs preserve elapsed-time gaps."
),
```

iii. CONVERSION_NOTES Step 3 records "Calcium signal recording frame rate: fs = 3.17Hz" from the
reference notebook. Step 5 Key Decision 3: "Native imaging bins are nominally 315.46 ms. Selected
samples remain native frames but stationary intervals are omitted; actual timestamp-derived inputs
preserve elapsed timing. Metadata will explicitly distinguish native frame interval from curated
sampling gaps." Step 10 Check 6 contrasts this with the reference's 60-spatial-bin interpolation,
which is for the paper's spatial figures and not applicable here.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime` (the MATLAB datenum timestamp of the sound cue on each trial) and `ft` (the
datenum timestamp of every imaging frame). The reference instead interpolates the fractional frame
number `SoundFr` onto the `ft` axis; the two are numerically identical in this dataset (verified:
`np.interp(SoundFr, arange(n), ft)` reproduces `SoundTime` to sub-millisecond).

ii.
```python
ft = np.asarray(beh["ft"][:common_nframes], dtype=np.float64)
...
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. CONVERSION_NOTES Step 5 mapping table: "`SoundTime`, `ft` → `input[0]` time to sound cue …
Time-varying float32. Uses timestamps rather than fractional `SoundFr`." The AI preferred the raw
timestamp over the fractional frame index so no interpolation step is needed.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Cue timestamp minus frame timestamp, converted from days to seconds by ×86,400. The sign
convention is positive before the cue, zero at the cue, negative after it — matching the human
reference and documented in metadata. It is time-varying, one value per retained frame, stored
float32. No clipping or normalisation; the full-dataset range is [−1763.3, 723.5] s, the extremes
coming from the handful of trials with long stationary pauses.

ii.
```python
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
decoder_input = np.vstack([time_to_sound, ..., ...]).astype(np.float32)
```
```python
"time_to_sound_sign": "positive before cue, zero at cue, negative after cue",
```

iii. CONVERSION_NOTES Step 10 Check 7: "Independent formulas for cue time, day/stage, elapsed time,
and `isRew` passed `np.allclose`." Step 10 issues: "Long elapsed-time values looked suspicious in the
summary: direct raw inspection showed real continuous frame timestamps separated by stationary gaps
while `ft_trInd` and position remain coherent. Kept and documented."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frames` index array used to slice the neural columns for that
trial, so it has that trial's length and every column corresponds to the same imaging frame as the
neural column. Both `ft` and the neural array are first truncated to the common frame count.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
time_to_sound = (float(beh["SoundTime"][trial]) - ft[frames]) * DAY_TO_SECONDS
```

iii. CONVERSION_NOTES Step 10 Check 5: "Both paths truncate frame behavior to neural `nfr` … Trial
stamps come from `ft_trInd`; trial zero/last and common-length edges were checked. Task-specific
time inputs use original `ft`, `Trial_start_time`, and `SoundTime`, preserving raw stationary gaps."
Independent raw-data spot checks matched at 1e-6 tolerance.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the experiment registry `beh/Imaging_Exp_info.npy`, not from the session dates. For each
physical recording the AI gathers every registry alias, and uses the alias field `days` when any
alias has it, otherwise the minimum alias `sess#`. It does **not** order a mouse's sessions by date
and count them, which is what the human reference does.

ii.
```python
for base, entries in aliases.items():
    explicit_days = [e["days"] for e in entries if "days" in e]
    session_indices = [e["sess#"] for e in entries if "sess#" in e]
    if explicit_days:
        training_day[base] = float(min(explicit_days))
    elif session_indices:
        # Shared recordings may have alternate statistical labels (notably
        # swap1/swap2). The minimum is the physical stage/day index.
        training_day[base] = float(min(session_indices))
    else:
        raise ValueError(f"No day/session field for {base}")
```

iii. CONVERSION_NOTES Step 4: "registry uses `days` for later learned sessions and `sess#` elsewhere
… Use explicit `days` when present; otherwise the minimum `sess#` among aliases (avoids treating
swap label 2 or naive reuse as another physical day). Record both source fields and rule in
metadata." Metadata carries `"training_day_rule": "registry days if present, otherwise minimum alias
sess#"`.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The single scalar per session is broadcast across all frames of all trials of that session, cast
to float32. The resulting values are 0–15. In practice the rule collapses most sessions to the same
value: `sess#` is a stage index *within an experiment-type list*, not a day count, and taking the
minimum over aliases makes it non-monotonic in date. For example TX119's eight sessions in
chronological order receive 1, 1, 0, 1, 1, 10, 1, 1; TX108's seven receive 1, 0, 1, 1, 6, 1, 1. The
variable therefore neither increases with training nor uses a single unit across sessions (some
sessions carry a day count, others a session/stage index).

ii.
```python
np.full(frames.size, day, dtype=np.float64),
```
```python
"day_of_training": [0.0, 15.0]   # verification_full_out.txt
```

iii. CONVERSION_NOTES Step 5 mapping: "Registry `days` / `sess#` → `input[1]` day of training …
Continuous numeric session-level covariate; heterogeneous source convention disclosed in metadata."
The AI acknowledges the convention is heterogeneous but chose to record it in metadata rather than
derive a uniform per-mouse day index; it did not check monotonicity within a mouse.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time` (the datenum timestamp of corridor entry on each trial) and `ft` (frame
timestamps). Equivalent to the reference's `np.interp(StartFr, arange(n), ft)` (verified identical
to sub-millisecond on a sample session).

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. CONVERSION_NOTES Step 5 mapping: "`Trial_start_time`, `ft` → `input[2]` time since trial start …
Time-varying float32 and retains real gaps removed by running mask." Step 3 notes corridor entry is
available equivalently as `Trial_start_time`, `StartFr`, or the `ft_trInd`/`ft_CorrSpc` transition.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. Frame timestamp minus trial-start timestamp, ×86,400 to get seconds. Positive after trial start,
so it starts at ≈0 and increases. Time-varying, float32, no clipping or normalisation. Range over
the full dataset is [0.0, 1765.2] s; the AI verified it is monotonically increasing within every
trial and that the large values are genuine stationary pauses.

ii.
```python
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```

iii. CONVERSION_NOTES Step 10 Check 10: "Verified … 49 trials with elapsed duration >300 s;
monotonic elapsed time in every trial … The longest raw trial resumes after a 1,737-s stationary
interval with the same trial stamp and monotonic position. This is an authentic recorded pause, not
misalignment."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as the other time-varying streams: evaluated at the identical `frames` index used for the
neural slice, after common-length truncation, so column *t* of the input corresponds to column *t* of
the neural matrix.

ii.
```python
frames = np.flatnonzero(valid & (trial_stamp == trial))
neural = spk[:, frames].copy()
time_since_start = (ft[frames] - float(beh["Trial_start_time"][trial])) * DAY_TO_SECONDS
```
```python
assert decoder_input.dtype == np.float32 and decoder_input.shape == (4, neural.shape[1])
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Use 2-D `(4,T)` arrays for both input and output on
every trial. This avoids mixed dimensionality and ensures every timepoint aligns exactly with neural
columns." Verified by independent raw reloads in Step 10 Check 2.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`, which marks trials run in the rewarded corridor. The AI
explicitly rejected the session-level `Reward_Mode` string because unsupervised sessions carry an
"Active after cue" mode string while `isRew` is False for every trial.

ii.
```python
np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
```

iii. CONVERSION_NOTES Step 4: "Some unsupervised files say active mode but all `isRew=False`;
4,336/38,110 raw trials are rewarded corridors … Use per-trial `isRew`, never the session mode
string." Step 5 mapping adds: "Means 'rewarded corridor,' not whether water was actually delivered."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to 0/1 and broadcast across every frame of the trial, stored float32 as part of the `(4, T)`
input matrix. No other processing.

ii.
```python
decoder_input = np.vstack(
    [
        time_to_sound,
        np.full(frames.size, day, dtype=np.float64),
        time_since_start,
        np.full(frames.size, float(bool(beh["isRew"][trial])), dtype=np.float64),
    ]
).astype(np.float32)
```

iii. The instruction specifies "1 if in rewarded corridor, 0 if not, discrete, per-trial". The AI
broadcasts it over time only because all four inputs share one `(4, T)` matrix (Key Decision 10).

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the per-trial string `WallName`, which names the wall texture of that trial's corridor. The
AI deliberately did not use `TrialStim`/`stim_id`, because `stim_id` contains NaNs and has alternate
mappings in the swap sessions.

ii.
```python
category = stimulus_category(str(beh["WallName"][trial]))
```

iii. CONVERSION_NOTES Step 4: "`stim_id` maps exemplars but has NaNs and alternate swap mappings …
Derive physical category from alphabetic prefix of `WallName`." Step 10 Check 8: "Direct `WallName`
avoids ambiguous swap `stim_id`."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The leading alphabetic run of the name is extracted by regex and lower-cased, pooling exemplar
numbers and spatial-shuffle suffixes (`leaf1`, `leaf2`, `leaf1_swap1` → `leaf`; `wood1_swap2` →
`wood`). The four categories are `['circle', 'leaf', 'rock', 'wood']`, stored as the index into that
list; an unrecognised prefix raises. The per-trial code is broadcast across all frames of the trial
as row 0 of the `(4, T)` int8 output. Full-dataset distribution by timepoint:
circle 0.311, leaf 0.470, rock 0.085, wood 0.135.

ii.
```python
CATEGORY_VALUES = ["circle", "leaf", "rock", "wood"]
CATEGORY_TO_CODE = {name: i for i, name in enumerate(CATEGORY_VALUES)}

def stimulus_category(name: str) -> int:
    """Pool exemplar numbers and spatial swaps into their physical category."""
    match = re.match(r"([A-Za-z]+)", str(name))
    if match is None or match.group(1).lower() not in CATEGORY_TO_CODE:
        raise ValueError(f"Unknown stimulus category in WallName={name!r}")
    return CATEGORY_TO_CODE[match.group(1).lower()]
```
```python
decoder_output[0] = category
```

iii. CONVERSION_NOTES Step 4: "Paper describes circle/leaf/rock/brick categories … preserve native
`wood` label (paper's brick texture) and pool exemplar/swap variants into four categories."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the (fractional) imaging-frame number of every lick in the session. Not from
`LickTime`/`LickPos`.

ii.
```python
lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
```

iii. CONVERSION_NOTES Step 3: "Licking is event-like and natively supplied with timestamps and
imaging-frame indices, so it can be represented as a per-frame binary series without temporal
interpolation." Step 10 Check 8: "`LickFr.astype(int)` is exactly the cast used in reference
cue/first-lick helpers."

## 8-b. What processing is involved in computing `output` *Licking*?

i. A session-length int8 flag vector is built: the fractional lick frame numbers are truncated to
integers (so a lick is attributed to the frame it lands in), non-finite entries and frames outside
`[0, common_nframes)` are discarded, and the corresponding entries are set to 1. Multiple licks in
one frame still give 1. Values are binary, `['not_licking', 'licking']`. Full-dataset distribution
by retained timepoint: 0.963 / 0.037.

ii.
```python
lick_global = np.zeros(common_nframes, dtype=np.int8)
lick_float = np.asarray(beh["LickFr"], dtype=np.float64)
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
lick_global[lick_frames] = 1
```
```python
decoder_output[1] = lick_global[frames]
```

iii. CONVERSION_NOTES Step 5 mapping: "`LickFr` → `output[1]` licking. Reference-compatible
`LickFr.astype(int)` event frames, binary at selected frames … Multiple licks in a frame remain 1."
The bounds/finiteness guards are part of Key Decision 7 ("handle missing data … fail loudly").

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so the flag vector lives on the neural grid; the trial's
values are taken with the same `frames` index array as the neural columns, giving identical length
and per-column correspondence. Note that licks occurring in dropped (stationary or gray-space)
frames are not represented in any trial.

ii.
```python
neural = spk[:, frames].copy()
...
decoder_output[1] = lick_global[frames]
```

iii. CONVERSION_NOTES Step 10 Check 2: independent raw-data reconstruction of the lick mask for
first/middle/last retained frames of a middle trial in four sessions matched exactly. Step 7's
processing plots were checked for "exact binary lick events" against raw `LickFr`.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the within-corridor position at each imaging frame, in decimeters (0–40 across the
4 m texture, continuing to 60 through the 2 m gray space — but only texture frames are retained).

ii.
```python
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
```

iii. CONVERSION_NOTES Step 4: "`ft_Pos` spans [0,40) in `ft_CorrSpc` … Decoder trial is 0–4 m visual
corridor, aligned to its entry; exclude gray."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimeters is divided by 10 and floored to give metres, then clipped to 0–3, stored as
int8 row 2 of the output. No interpolation or smoothing; the reference's 60-bin spatial
interpolation is deliberately not used.

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
decoder_output[2] = position_class
```

iii. CONVERSION_NOTES Step 5 mapping: "`ft_Pos` → `output[2]` corridor position. floor(position
decimeters / 10), clipped 0–3 … Four equal 1-m bins spanning textured 0–4 m corridor." Step 4:
"Four position labels are [0,1), [1,2), [2,3), [3,4) m."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1-m bins with hard edges at 0/1/2/3/4 m, exactly as the decoder task
specifies ("4 equal-length, 1-m-long spatial bins"); the clip guards the boundary at 40 dm.
`output_values[2] = ['0-1 m', '1-2 m', '2-3 m', '3-4 m']`. Because the corridor is traversed at
roughly constant VR speed, the realised distribution is almost uniform:
[0.250, 0.249, 0.250, 0.252].

ii.
```python
position_class = np.clip(np.floor(position / 10.0), 0, 3).astype(np.int8)
```
```python
"output_values": [..., ["0-1 m", "1-2 m", "2-3 m", "3-4 m"], ...]
```

iii. CONVERSION_NOTES Step 9 consistency table confirms "four 1-m bins … [.250,.249,.250,.252]" and
Step 10 Check 6: "All position labels are 0–3."

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` is one value per imaging frame, already on the neural grid; the trial's values are taken
with the same `frames` index array used for the neural slice, after truncating `ft_Pos` to the common
frame count.

ii.
```python
position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
decoder_output[2] = position_class
...
assert decoder_output.dtype == np.int8 and decoder_output.shape == (4, neural.shape[1])
```

iii. CONVERSION_NOTES Step 7 processing-plot review: "raw 0–6 m cycles, retained samples only in the
rising 0–4 m texture segment, monotonically increasing four-bin position labels … No temporal offset
or discretization anomaly was found."

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the mouse already interpolated to imaging-frame times by
the authors, in native speed units.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
```

iii. CONVERSION_NOTES Step 12 speed audit: "The source variable is correctly `ft_RunSpeed`, which the
reference describes as running speed interpolated to imaging frames. Replacing it with VR motion
(`ft_move`) or including speed as an input would change the task/leak the answer and is not
justified."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw per-frame speeds of the retained frames are collected per trial and held aside (not
discretised yet). After **all** sessions are converted, every retained speed value in the dataset is
pooled into one array, three global quantile thresholds are computed, and each trial's speed row is
filled in a second pass. No smoothing, no per-session normalisation.

ii.
```python
def fill_speed_classes(outputs, speeds):
    """Compute one global quartile definition and fill output row 3."""
    all_speed = np.concatenate([trial for session in speeds for trial in session]).astype(np.float64)
    thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
    counts = np.zeros(4, dtype=np.int64)
    for out_session, speed_session in zip(outputs, speeds, strict=True):
        for out_trial, speed_trial in zip(out_session, speed_session, strict=True):
            labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
            out_trial[3] = labels
            counts += np.bincount(labels, minlength=4)
    return thresholds, counts
```
```python
decoder_output[3] = 0  # Filled after global speed thresholds are known.
```

iii. CONVERSION_NOTES Step 6: "computes global speed quartiles after collecting all small speed
vectors, avoiding a second neural-data pass" — i.e. the deferred fill exists so the 404 GiB of spike
files are read only once.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four dataset-wide quartile bins defined by `np.quantile(all_speed, [.25, .5, .75])` and assigned
with `np.searchsorted(..., side='right')`, labelled `['Q1','Q2','Q3','Q4']`. The thresholds
(12.42242, 25.352615, 40.854578) and the realised counts are written into metadata. Counts are
[205395, 205394, 205395, 205395] — exactly 25% each. Two differences from the human reference: the
split is global rather than per session, and because stationary frames were already removed by
`ft_move > 0` the zero-speed mass that would otherwise tie up the lowest bin is largely gone, so
plain value thresholds suffice instead of a rank-based split. In `--sample` mode the thresholds are
recomputed from only the two sample sessions, so sample and full runs do not use identical class
boundaries.

ii.
```python
thresholds = np.quantile(all_speed, [0.25, 0.50, 0.75])
labels = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
```
```python
"speed_quartile_thresholds": thresholds.astype(float).tolist(),
"speed_quartile_counts": speed_counts.astype(int).tolist(),
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Use global thresholds so class meanings are comparable
across recordings." Step 4: "All-frame Q1 is tied at zero; running-mask quartiles are
non-degenerate." Step 12: "Variation is ideal rather than degenerate: global counts are
[205395,205394,205395,205395], each 25.0%."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame; the trial's speeds are taken with the same `frames`
index array as the neural columns, after truncation to the common frame count, and the class labels
are written back into the matching columns of the same trial's output matrix.

ii.
```python
speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
...
out_trial[3] = np.searchsorted(thresholds, speed_trial, side="right").astype(np.int8)
```

iii. CONVERSION_NOTES Step 12 speed audit: "`np.searchsorted` of raw `ft_RunSpeed` with stored
thresholds matched every converted speed label exactly using `np.allclose` … Processing figures
overlay raw speed, thresholds, and converted classes on single trials; transitions are synchronized
with the neural columns and position/lick outputs."

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards:
- Behavior streams run 1–2 frames past the imaging in all 89 sessions. Every frame-aligned stream
  *and* the neural matrix are truncated to `common_nframes = min(neural frames, len(ft),
  len(ft_trInd), len(ft_Pos), len(ft_move), len(ft_CorrSpc), len(ft_RunSpeed))` — a two-sided
  minimum, slightly more defensive than the reference's `beh[...][:nfr]`.
- Licks with non-finite frame numbers or frame numbers outside `[0, common_nframes)` are dropped.
- Frames with non-finite `ft_trInd` are excluded from every trial.
- Trials with no surviving frame are skipped and their indices recorded in
  `metadata.session_info[...].excluded_trial_indices` (0 in the full run).
- Plane frame-count mismatches, retinotopy/neuron-count mismatches, unknown `WallName` prefixes,
  non-finite neural/input/speed values, a catalog mismatch between spk/behavior/registry, a missing
  retinotopy file, and a session with fewer than two usable trials all raise immediately rather than
  being silently absorbed. That last one would abort the whole conversion, where the reference logs
  and skips the session.
- Conflicting behavior aliases for the same physical recording raise.
- A final `validate_converted` pass re-checks dtypes, shapes, finiteness and label ranges before the
  pickle is written.

ii.
```python
common_nframes = min([neural_nframes] + [len(beh[k]) for k in frame_fields])
```
```python
lick_frames = lick_float[np.isfinite(lick_float)].astype(np.int64)
lick_frames = lick_frames[(lick_frames >= 0) & (lick_frames < common_nframes)]
```
```python
if frames.size == 0:
    excluded_trials.append(trial)
    continue
...
if len(neural_trials) < 2:
    raise ValueError(f"{base} has fewer than two usable trials after curation")
```
```python
if not np.all(np.isfinite(neural)):
    raise ValueError(f"Non-finite neural values in {base} trial {trial}")
```

iii. CONVERSION_NOTES Step 5 Key Decision 7 and Step 10 issues: "Every behavior frame stream exceeds
its neural stream by 1–2 frames: resolved with the reference code's common-length truncation. No
selected trial was lost." Step 2: "Alignment edge case confirmed: neural and behavior frame lengths
can differ by one frame … Reference code always truncates behavior frame arrays to neural `nfr`; the
converter must use their common length."

## 12-a. What are the most time-consuming steps of the code?

i. Reading the ~404 GiB of `spk/*_neural_data.npy` files. In the full run (742.95 s total), per-
session neural load was 2–12 s and accounted for roughly 60–70% of each session's conversion time
(~440 s in total). Second is writing the 141 GiB pickle: 128.85 s. Third, the per-trial slicing,
`.copy()`, and the two full finiteness scans over the retained neural data (~2 s/session, ~170 s
total). Behavior and retinotopy I/O is negligible.

ii.
```python
neural_dict = np.load(path, allow_pickle=True).item()
planes = neural_dict["spks"]
...
selected[dst0 : dst0 + count] = plane[local_mask]
```
```python
with args.outpicklefile.open("wb") as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. CONVERSION_NOTES Step 6: "Code speedups added: Preallocate only retained neurons and fill
plane-by-plane; process each neural file once; retain only small speed arrays until quartile
thresholds are known … Parallel session loading was intentionally avoided because sequential reads
are fast and parallel copies would substantially raise peak memory without clear throughput
benefit." Step 7 estimated ~13–14 min and the actual run took 12.4 min.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial loop rescans the whole session frame index once per trial:
`np.flatnonzero(valid & (trial_stamp == trial))` is O(n_frames) per trial, so O(ntrials × n_frames)
per session (~450 × ~23,000). A single pass — sorting frames by `trial_stamp` or using
`np.searchsorted` on the sorted trial labels — would give all trials' frame lists at once. This is
the same loop the human reference identifies and also leaves unvectorized, and it is negligible next
to the I/O. Also inside that loop, `np.asarray(beh['ft_Pos'][:common_nframes])` and
`np.asarray(beh['ft_RunSpeed'][:common_nframes], dtype=np.float32)` rebuild the full-session arrays
on every trial instead of once per session, and `spk[:, frames].copy()` copies an array that fancy
indexing has already copied. The AI's notes do not identify any of these; the inefficiencies it did
identify and fix were the plane concatenation and a second pass over the spike files.

ii.
```python
for trial in range(ntrials):
    frames = np.flatnonzero(valid & (trial_stamp == trial))
    ...
    position = np.asarray(beh["ft_Pos"][:common_nframes])[frames]
    speed = np.asarray(beh["ft_RunSpeed"][:common_nframes], dtype=np.float32)[frames]
```

iii. CONVERSION_NOTES Step 6 lists only "Naively concatenating all raw planes before region filtering
would allocate an additional complete session matrix. Computing exact global speed thresholds in a
separate pass would also reread roughly 400 GB of neural files unnecessarily." No statement is made
about the per-trial scan.

## 12-c. What processing does the code repeat multiple times?

i. Two genuine repeats:
1. **Finiteness of the neural data is scanned twice over the whole dataset** — once per trial inside
   `convert_session` (`np.all(np.isfinite(neural))`) and again for every trial in
   `validate_converted` before the pickle write. That is two full passes over ~38 billion float32
   values.
2. **Per-session behavior slices are rebuilt per trial** — `ft_Pos` and `ft_RunSpeed` are re-sliced
   and re-cast inside the trial loop (see 12-b), where `ft`, `lick_global`, `corridor`, `movement`
   are correctly hoisted out.

Speed labelling is a deliberate two-pass design (collect, then threshold) but the second pass touches
only the small speed arrays, not the neural data, so it is not wasteful. The human reference reports
no repeated processing.

ii.
```python
# in convert_session, per trial
if not np.all(np.isfinite(neural)):
    raise ValueError(f"Non-finite neural values in {base} trial {trial}")
```
```python
# in validate_converted, again for every trial
assert np.all(np.isfinite(neural)) and np.all(np.isfinite(decoder_input))
```

iii. The AI's stated rationale for `validate_converted` is Step 6: "performs strict finite/range/
shape validation before pickle writing". It does not note that the finiteness check duplicates the
per-trial check already performed during conversion.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor, mostly diagnostic:
- `load_selected_spikes` builds and returns `source_ranges`, which the caller discards (`spk,
  neural_nframes, _ = ...`).
- `registry_catalog` converts every registry entry to plain Python via `jsonable` and the full alias
  list for all 89 sessions is stored in `metadata.session_info[...].registry_aliases`; only the
  derived `training_day` scalar is used by the conversion, and the decoder uses none of it.
- `excluded_trial_indices`, `rewarded_trials_source`, `load_neural_seconds`, `conversion_seconds`
  and similar provenance fields are recorded for every session but never consumed downstream.
- The duplicate finiteness scan of 12-c.
- `spk[:, frames].copy()` performs a redundant second copy of every trial's neural block.
- In `--show-processing` runs, `plot_context` retains extra copies of raw neural, area and behavior
  slices for two sessions.
None of these affect correctness, and the provenance metadata is arguably useful documentation rather
than waste. The human reference reports none.

ii.
```python
spk, neural_nframes, _ = load_selected_spikes(base, mapped)   # source_ranges discarded
```
```python
info["registry_aliases"] = aliases[base]
```
```python
neural = spk[:, frames].copy()
```

iii. CONVERSION_NOTES Step 6 frames the extra metadata as a feature: the script "records detailed
per-session provenance and counts in metadata". No statement is made about the unused
`source_ranges` or the redundant `.copy()`.
