# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `data/beh/Imaging_Exp_info.npy` as the master index. It is a dict keyed by experiment type (23 types, 142 entries), each entry describing one recording (`mname`, `datexp`, `blk`, `sess#`, ...). `discover_sessions()` walks the index in dict order and keeps one descriptor per unique `mname_datexp_blk` recording id (89 unique recordings), remembering the experiment type it was *first* seen under and the matching `Beh_<exp_type>.npy` file. Three sources are then read per session: behavior from `beh/Beh_<exp_type>.npy` (keyed by the recording id, with an optional `_<stimtype>` suffix for the swap sessions), deconvolved traces from `spk/<rid>_neural_data.npy` (a dict with a list of per-plane `spks` arrays), and retinotopy area labels from `retinotopy/<mname>_<datexp>_trans.npz` (`iarea`). Behavior is *not* grouped by file: `load_behavior()` re-loads the whole `Beh_*.npy` pickle every time it is called, and it is called once per session in three separate passes (the visual-category scan, `compute_speed_edges`, and `convert_session`).

ii.
```python
def discover_sessions(data_root: Path) -> list[dict]:
    """Return one descriptor per physical recording, in repository order."""
    exp_info = np.load(
        data_root / "beh" / "Imaging_Exp_info.npy", allow_pickle=True
    ).item()
    sessions: OrderedDict[str, dict] = OrderedDict()
    for experiment_type, entries in exp_info.items():
        behavior_file = data_root / "beh" / f"Beh_{experiment_type}.npy"
        for entry in entries:
            rid = _recording_id(entry)
            if rid not in sessions:
                sessions[rid] = {...}
    return list(sessions.values())

def load_behavior(desc: dict) -> dict:
    records = np.load(desc["behavior_file"], allow_pickle=True).item()
    return records[_behavior_key(desc["entry"], records)]
```
```python
    beh = load_behavior(desc)
    spk_path = DATA_ROOT / "spk" / f"{desc['recording_id']}_neural_data.npy"
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    planes = spk_obj["spks"]
```
```python
    ret = np.load(
        DATA_ROOT / "retinotopy" /
        f"{desc['entry']['mname']}_{desc['entry']['datexp']}_trans.npz"
    )
    area_ids = np.asarray(ret["iarea"]).astype(np.int16, copy=False)
```

iii. From the trajectory (step 21): "I found 89 unique recordings represented 142 times in the experiment index because the same recording is reused for different paper analyses. The converter will deduplicate by recording ID, retain the paper's visual-cortex area masks and running-within-textured-corridor frame rule, and preserve all trials rather than duplicating recordings by figure panel." The AI first explored `code/utils.py`, `code/data_process_script.ipynb` and the raw arrays (steps 8–20) to establish the file layout and the `Beh_*` keying convention before writing the converter.

## 1-b. How are the data split into subjects (mice)?

i. The mouse identity is read straight from the index entry field `mname`; no derivation is needed. `subjects` is the sorted set of unique names (19 mice), and `subject_idx` is each session's index into that list, stored as `int16`. Sessions per subject range from 1 (TX140) to 8 (TX119, TX123).

ii.
```python
    subjects = sorted({str(x["entry"]["mname"]) for x in sessions})
    subject_to_id = {name: i for i, name in enumerate(subjects)}
    ...
        subject_idx.append(subject_to_id[str(desc["entry"]["mname"])])
    ...
        "subjects": subjects,
        "subject_idx": np.asarray(subject_idx, dtype=np.int16),
```

iii. No explicit justification is recorded; the AI treated `mname` as the subject id from its first inspection of `Imaging_Exp_info.npy` (step 11–12, where it printed the entry fields).

## 1-c. How are the data split into sessions?

i. A session is one physical recording = one (mouse, date, imaging block) triple, which also names the spike file. Because the index lists the same recording under several experiment types (142 listings for 89 recordings), duplicates are removed with an `OrderedDict` keyed by the recording id, keeping the first listing. The behavior key is the recording id, extended with the `stimtype` suffix when such a key exists in the behavior dict (swap sessions). The result is 89 sessions.

ii.
```python
def _recording_id(entry: dict) -> str:
    return f"{entry['mname']}_{entry['datexp']}_{entry['blk']}"

def _behavior_key(entry: dict, behavior: dict) -> str:
    """Resolve the optional swap-stimulus suffix used in test-3 tables."""
    base = _recording_id(entry)
    suffix = entry.get("stimtype")
    if suffix and f"{base}_{suffix}" in behavior:
        return f"{base}_{suffix}"
    if base not in behavior:
        raise KeyError(f"No behavior record for {base}")
    return base
```
```python
            if rid not in sessions:
                sessions[rid] = {
                    "recording_id": rid,
                    "experiment_type": experiment_type,
                    "entry": entry,
                    "behavior_file": behavior_file,
                }
```

iii. Trajectory step 21: the AI explicitly checked the multiplicity of index entries ("unique 89, multiplicity Counter(...)", step 12) and decided to "deduplicate by recording ID ... rather than duplicating recordings by figure panel", i.e. a recording reused in several figure panels is still one session.

## 1-d. How are the data split into trials?

i. Trials are the ones the behavior declares (`ntrials`, with `ft_trInd` labelling each imaging frame with its trial). A trial's frames are those that satisfy the paper's `fr_valid` mask *and* belong to that trial: inside the textured corridor (`ft_CorrSpc`) **and** with the virtual reality advancing (`ft_move > 0`). Frames in the 2 m grey space, and frames in which the animal was not running, are excluded. Trials therefore have variable length (median 21, max 178 bins) and are, in 1.8% of adjacent sample pairs, non-contiguous in real time because stationary frames have been cut out of the middle of the traversal.

ii.
```python
def retained_frame_mask(beh: dict, nframes: int) -> np.ndarray:
    """Paper's ``fr_valid = VRmove & isCorridor`` frame selection."""
    return (
        np.asarray(beh["ft_move"][:nframes]) > 0
    ) & np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
```
```python
    valid = retained_frame_mask(beh, nframes)
    trial_stamp = np.asarray(beh["ft_trInd"][:nframes])
    ...
    for trial in range(ntrials):
        frames = np.flatnonzero(valid & (trial_stamp == trial))
```

iii. Trajectory step 10: "The source makes one key distinction: its figure analyses retain only frames when the VR is advancing, but this decoder is explicitly trial-start/time aligned. I'm checking the stored frame timestamps and trial indices now so the conversion can remain time-based without accidentally turning position interpolation into 'time.'" Step 21: it decided to "retain the paper's ... running-within-textured-corridor frame rule". The metadata records `frame_filter: "ft_move > 0 and ft_CorrSpc, matching paper analysis code"`, which matches `utils.py:431-433` (`fr_valid = VRmove & isCorridor`) and the Methods sentence "We only considered timepoints during running for analysis".

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is emptiness: a trial with zero retained frames is dropped. In practice none was dropped, so all 38,110 source trials are kept in all 89 sessions. A session that would end up with fewer than two trials raises an error (never triggered). There is no filter on trial duration: 378 kept trials last longer than 75 s and 49 longer than 300 s (maximum 1,765 s — a mouse that stopped mid-corridor for 29 minutes). The `ft_move` mask keeps these trials short in *samples* (max 178 bins) but their elapsed-time inputs remain extreme.

ii.
```python
        frames = np.flatnonzero(valid & (trial_stamp == trial))
        if len(frames) == 0:
            dropped_empty += 1
            continue
```
```python
        if len(neural) < 2:
            raise ValueError(f"{desc['recording_id']} retained fewer than two trials")
```

iii. Trajectory step 21: "preserve all trials rather than duplicating recordings by figure panel"; step 46: "The converter is still preserving every nonempty source trial after the paper-defined frame mask; there have been no empty-trial drops so far." The implicit argument is that the paper's running mask already removes the stationary periods, so no separate outlier rule is needed. The AI never inspected the distribution of trial *durations* and never commented on the residual long-duration trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. From `spks` in `spk/<recording_id>_neural_data.npy` — a list of one (neurons × frames) deconvolved-fluorescence array per imaging plane — with the per-neuron visual area taken from `iarea` in `retinotopy/<mname>_<datexp>_trans.npz`. Planes are truncated to a common frame count and kept in plane order, which is the order the retinotopy labels index.

ii.
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    planes = spk_obj["spks"]
    nframes = min(a.shape[1] for a in planes)
    planes = [a[:, :nframes] for a in planes]

    plane_keep, region_idx = neuron_selection(desc, [a.shape[0] for a in planes])
```
```python
    expected = int(sum(plane_sizes))
    if len(area_ids) != expected:
        raise ValueError(
            f"{desc['recording_id']}: {len(area_ids)} retinotopy labels for "
            f"{expected} neurons"
        )
```

iii. Metadata: `source_neural_signal: "Suite2p non-negative deconvolved fluorescence (spks)"`. The module docstring states "deconvolved Suite2p `spks` are concatenated in plane order", following `code/utils.py`; the Methods say "All our analyses were based on deconvolved fluorescence traces".

## 2-b. How is the `neural` data processed?

i. No transformation at all: no dF/F, no deconvolution, no smoothing, no z-scoring, no normalisation. For each trial the selected neuron rows and retained frame columns are copied into a pre-allocated `(n_neurons, T)` `float32` array, one plane at a time, so that the row order is exactly the plane/retinotopy order. Trials are left at their natural length; nothing is padded. The resulting pickle is 151.5 GB (141 GiB).

ii.
```python
        # Allocate once, then copy each imaging plane directly into final row order.
        neural = np.empty((n_neurons, T), dtype=np.float32)
        row = 0
        for plane, keep in zip(planes, plane_keep):
            n = len(keep)
            neural[row:row + n] = plane[np.ix_(keep, frames)]
            row += n
```

iii. Trajectory step 27: "this is a large artifact because it intentionally preserves single-neuron, single-frame activity." The AI's stated aim was to hand the decoder the raw deconvolved traces without any lossy step; it kept the source `float32` dtype rather than down-casting.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. (1) Neurons: only neurons whose retinotopy `iarea` falls in the paper's four coarse visual areas are kept — V1 (`iarea == 8`), mHV (0, 1, 2, 9), lHV (5, 6), aHV (3, 4); `iarea == 7` and unassigned `-1` are dropped. This keeps 4,105,393 of 4,691,034 neurons (identical to the reference count). (2) Frames: the `ft_move > 0 & ft_CorrSpc` mask described in 1-d (about 31% of within-corridor frames are dropped). No other curation (no cell-classifier threshold, no SNR/activity filter) is applied; Suite2p's own cell classification is inherited from the source files.

ii.
```python
AREA_ID_TO_REGION = {
    8: 0,                         # V1
    0: 1, 1: 1, 2: 1, 9: 1,     # medial higher visual areas
    5: 2, 6: 2,                  # lateral higher visual areas
    3: 3, 4: 3,                  # anterior higher visual areas
}
REGION_NAMES = ["V1", "mHV", "lHV", "aHV"]
```
```python
    for size in plane_sizes:
        ids = area_ids[offset:offset + size]
        keep = np.isin(ids, np.fromiter(AREA_ID_TO_REGION, dtype=np.int16))
        local = np.flatnonzero(keep)
        plane_local_indices.append(local)
        region_chunks.append(
            np.fromiter((AREA_ID_TO_REGION[int(x)] for x in ids[local]), dtype=np.int16)
        )
        offset += size
```

iii. Comment in the code: "This is the exact grouping used by neu_area_ID() in the paper repository" (matches `utils.py:312-324`). Metadata: `neuron_filter: "Retinotopically assigned V1/mHV/lHV/aHV neurons; iarea -1 and 7 excluded"`. In step 17 the AI counted area labels across all retinotopy files to confirm the retained fraction before committing to the rule; step 38: "excluding only the retinotopy labels the paper treats as outside/unassigned visual cortex."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Each trial's array begins at the first retained frame inside that trial's textured corridor and ends at the last one; there is no common window, no truncation, and no padding, so trials have different lengths. `off_start` is declared 0.0 and `off_end` is `None`. Measured on a supervised session, the first retained frame falls a median 0.18 s after `Trial_start_time` — identical to the lag without the running mask, i.e. the running mask does not shift trial onsets (mice are running at corridor entry).

ii.
```python
        frames = np.flatnonzero(valid & (trial_stamp == trial))
        ...
        T = len(frames)
        neural = np.empty((n_neurons, T), dtype=np.float32)
```
```python
            "temporal_alignment_event": "entry into the 4 m textured corridor",
            "off_start": 0.0,
            "off_end": None,
```

iii. Trajectory step 10: the AI noted the decoder task "is explicitly trial-start/time aligned" and checked `StartFr`/`GrayFr`/`EndFr`/`Trial_start_time` against `ft_Pos`, `ft_trInd`, `ft_CorrSpc` (step 15) to confirm that the first corridor frame of a trial coincides with the stored trial-start timestamp.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: one column per two-photon imaging frame. The AI measured the true frame interval from the `ft` timestamps and stored it in the metadata (`observed_median_frame_interval_ms = 314.80`, IQR 311.0–318.8 ms, i.e. ~3.17 Hz), but set the declared `time_bin_size` to `1000.0 / 3.0 = 333.33 ms` on the grounds that the paper says "approximately 3 Hz". Because non-running frames are removed (1-d), successive bins of a trial are not always adjacent in real time.

ii.
```python
            # Imaging was nominally approximately 3 Hz in the paper. Exact frame
            # timestamps drive event variables; observed timing is also recorded.
            "time_bin_size": 1000.0 / 3.0,
```
```python
        dt = np.diff(np.asarray(beh["ft"][:nframes], dtype=np.float64)) * 86_400_000.0
        frame_dts_ms.append(dt[(dt > 100) & (dt < 1000)])
    ...
    stats = {
        "speed_quartile_edges_cm_s": edges.tolist(),
        "observed_median_frame_interval_ms": float(np.median(dt_all)),
        "observed_frame_interval_iqr_ms": np.quantile(dt_all, [0.25, 0.75]).tolist(),
    }
```

iii. Code comment: the nominal 3 Hz figure is taken from the paper, and the AI's justification is that "Exact frame timestamps drive event variables; observed timing is also recorded" — i.e. it regarded `time_bin_size` as descriptive only, because the time-valued inputs are computed from the real timestamps rather than from a nominal bin width.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundTime`, the MATLAB datenum timestamp of the sound cue on each trial, and `ft`, the timestamp of every imaging frame. (The reference instead interpolates the fractional frame index `SoundFr` onto the frame-time axis; both describe the same event, and `SoundTime` is the quantity `SoundFr` was derived from.) There are no NaN `SoundTime` values anywhere in the dataset.

ii.
```python
    frame_times = np.asarray(beh["ft"][:nframes], dtype=np.float64)
    ...
        to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. In step 15 the AI printed `(SoundTime - Trial_start_time) * 86400` alongside `SoundFr` and `ft` to verify that the datenum timestamps and the frame-index fields describe the same events, and then used the timestamps directly ("The source uses ... MATLAB datenums for timestamps. Outputs here use seconds", module docstring).

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. A per-frame difference, converted from MATLAB days to seconds by multiplying by 86,400: `SoundTime[trial] - ft[frame]`. The convention is "time *to* the cue", so the value is positive before the cue and negative after it. It is stored as `float32` and is fully time-varying. No clipping or normalisation is applied, so the range across the dataset is −1,763 s to +723 s (the extremes come from the long stalled trials kept in 1-e).

ii.
```python
        inputs = np.empty((4, T), dtype=np.float32)
        inputs[0] = to_cue_s
```
```python
        "input_names": [
            "time_to_sound_cue_s",
            ...
        ],
```

iii. Module docstring: "The source uses decimetres for VR position and MATLAB datenums for timestamps. Outputs here use seconds, centimetres/second, and zero-based class labels." No further justification of the sign convention is recorded; the name `time_to_sound_cue_s` states it.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same retained frame indices used for the neural columns of that trial (`frame_times[frames]`), so the input array has the same `T` as the neural array and shares its time base frame-by-frame.

ii.
```python
        elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
        to_cue_s = (float(beh["SoundTime"][trial]) - frame_times[frames]) * 86_400.0
```

iii. All streams in this dataset are stored on the imaging-frame grid (`ft`, `ft_Pos`, `ft_RunSpeed`, `ft_trInd`), so indexing everything with the same `frames` array is the alignment. The AI verified this frame grid in steps 15 and 18 before relying on it.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the `sess#` field of the `Imaging_Exp_info.npy` entry under which the recording was first listed, with a fallback rule when `sess#` is absent. The AI did **not** use the recording date (`datexp`) and did **not** use the `days` field that is present instead of `sess#` on 8 entries (values 6, 7, 8, 9, 9, 10, 13, 15 — the actual number of training days).

ii.
```python
def session_day(desc: dict) -> float:
    """Use the repository's session/day number; document the missing-value rule."""
    value = desc["entry"].get("sess#")
    if value is not None and np.isfinite(value):
        return float(value)
    # Several train2-after records have no sess# even though their stage is known.
    # Treating the two paper stages as 0/1 is consistent with train1 and avoids an
    # invented calendar-day estimate (recording dates are not training start dates).
    typ = desc["experiment_type"]
    return 1.0 if "after" in typ else 0.0
```

iii. The code comment is the justification: recording dates "are not training start dates", so the AI preferred the repository's own session counter to any date arithmetic, and preferred an explicit 0/1 stage label to "an invented calendar-day estimate". The metadata records the rule as `day_rule: "Imaging_Exp_info sess#; missing train2-after values inferred as stage 1"`. In step 16 the AI grepped the code and methods for `sess#|datexp|day|before_learning|after_learning` before settling on this.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The scalar is broadcast across all time bins of every trial of the session as input dimension 1, as `float32`. The resulting distribution over the 89 sessions is: 1.0 × 69 sessions, 0.0 × 13, 2.0 × 3, 3.0 × 1, and one session each at 6.0, 10.0 and 12.0 (range reported by the verifier: 0–12). So for most sessions the variable is a before/after-learning stage flag, while a handful of sessions carry a genuine session counter on a completely different scale. The 8 sessions whose entry carries `days` (6–15 real training days) are all assigned 1.0 by the fallback. `sess#` is also ambiguous for 12 recordings that are listed with conflicting `sess#` values under different experiment types; the value used is whichever listing came first in the index dict.

ii.
```python
    day = session_day(desc)
    ...
        inputs[1] = day
```
```python
        "input_names": [
            "time_to_sound_cue_s",
            "day_of_training",
            ...
        ],
```

iii. As in 4-a: the AI argued that the stage is "known" even when `sess#` is missing and that mapping "the two paper stages as 0/1 is consistent with train1". It recorded the imputation openly in `metadata['day_rule']` and in the per-session `session_info` (`source_session_number` vs `decoder_day_value`) so that a reader can see which sessions were imputed.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `Trial_start_time`, the MATLAB datenum timestamp of corridor entry for each trial, and `ft`, the per-frame timestamp. (The reference uses `StartFr` interpolated onto the frame-time axis; the two agree — `ft[StartFr[0]] ≈ Trial_start_time[0]` to within a fraction of a frame.)

ii.
```python
        elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
        ...
        inputs[2] = elapsed_s
```

iii. Step 15 of the trajectory shows the AI explicitly cross-checking `StartFr`, `GrayFr`, `EndFr` against `ft_Pos`/`ft_trInd`/`ft_CorrSpc` and printing `(SoundTime|Gray_space_time|Trial_end_time − Trial_start_time) × 86400` to confirm that the timestamp fields and the frame-index fields describe the same trial boundaries, after which it used the timestamps.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. A per-frame difference in seconds, `ft[frame] − Trial_start_time[trial]`, positive after entry. Stored as `float32`, time-varying, unclipped and unnormalised. Because no long-trial filter is applied (1-e), the range is 6×10⁻⁵ s to 1,765 s, with 378 trials exceeding 75 s.

ii.
```python
        inputs = np.empty((4, T), dtype=np.float32)
        inputs[0] = to_cue_s
        inputs[1] = day
        inputs[2] = elapsed_s
        inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. Module docstring: "Outputs here use seconds". No further discussion; the AI treated elapsed time as a direct read-out of the frame clock.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same as 3-c: evaluated at the identical `frames` indices as the neural columns, so it is bin-for-bin aligned and the first bin of each trial carries the true (small, median ≈0.18 s) offset from corridor entry rather than an assumed zero.

ii.
```python
        frames = np.flatnonzero(valid & (trial_stamp == trial))
        T = len(frames)
        ...
        elapsed_s = (frame_times[frames] - float(beh["Trial_start_time"][trial])) * 86_400.0
```

iii. Everything in the behavior file that starts with `ft_` is already on the imaging-frame grid, so one shared index array aligns all streams; the AI verified this correspondence in steps 15 and 18.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial boolean `isRew`, which flags trials run in the rewarded corridor. It is `False` for every trial of the unsupervised, naive and grating cohorts.

ii.
```python
        inputs[3] = float(bool(beh["isRew"][trial]))
```

iii. No separate justification recorded; in step 16 the AI tabulated reward types/`Reward_Mode` across experiment types while surveying the behavior fields, and then took `isRew` as the reward-availability flag.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. The boolean is cast to 0.0/1.0 and broadcast across all bins of the trial as input dimension 3 (`float32`). The verifier reports its range as [0.0, 1.0]. No cue-dependent gating (e.g. "reward only available after the sound cue") is applied — it is a per-trial corridor-identity flag, as the decoder task specifies.

ii.
```python
        inputs[3] = float(bool(beh["isRew"][trial]))
```
```python
        "input_names": [..., "reward_available"],
```

iii. The instructions define this input as "1 if in rewarded corridor, 0 if not, discrete, per-trial", which is exactly `isRew`; the AI implemented it literally.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From `WallName`, the per-trial name of the wall texture (e.g. `leaf1`, `circle2`, `wood1_swap2`). `TrialStim`, `WallType`, `stim_id` and `StimTrial` are not used.

ii.
```python
    categories = set()
    for desc in sessions:
        beh = load_behavior(desc)
        categories.update(visual_category(x) for x in beh["WallName"])
    category_names = sorted(categories)
    category_to_id = {name: i for i, name in enumerate(category_names)}
```
```python
        category = visual_category(beh["WallName"][trial])
        outputs[0] = category_to_id[category]
```

iii. Step 16 shows the AI counting wall names and categories across all experiment types ("walls/reward by exp first canonical unique ... walls=Counter()") before choosing `WallName` as the label source.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each `WallName` is collapsed to its leading alphabetic run via a regular expression, lower-cased: `circle1/2/3 → circle`, `leaf1/2/3`, `leaf1_swap1/2 → leaf`, `rock1/2 → rock`, `wood1/2/5`, `wood1_swap1/2 → wood`. Over the whole dataset the 15 distinct wall names collapse to exactly four categories, `['circle', 'leaf', 'rock', 'wood']` (counts: leaf 17,761; circle 11,619; wood 5,452; rock 3,278) — the same four categories and the same alphabetical order as the reference. The category list is discovered at run time by scanning every session's `WallName`, then the per-trial integer code is broadcast across the trial's bins as `int16`.

ii.
```python
def visual_category(name: str) -> str:
    """Collapse crop/version suffixes while retaining the source category."""
    match = re.match(r"[A-Za-z]+", str(name))
    if match is None:
        raise ValueError(f"Cannot derive visual category from {name!r}")
    return match.group(0).lower()
```
```python
        outputs = np.empty((4, T), dtype=np.int16)
        category = visual_category(beh["WallName"][trial])
        outputs[0] = category_to_id[category]
```

iii. Metadata: `visual_category_rule: "WallName crop/version suffixes collapsed to circle, leaf, rock, or wood"`. The decoder task asks for a stimulus *category* ("e.g. circle, leaf"), and the paper treats the different crops and spatial shuffles of a texture as the same category, so the crop index and `_swapN` suffix are discarded.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr`, the fractional imaging-frame index of every detected lick, together with `LickTrind`, the trial index of each lick. (Both were verified to be mutually consistent and 0-based.)

ii.
```python
    lick_frame = np.rint(np.asarray(beh["LickFr"], dtype=np.float64)).astype(np.int64)
    lick_trial = np.asarray(beh["LickTrind"], dtype=np.float64)
```

iii. Step 18 of the trajectory is a dedicated check of exactly this: the AI printed `LickFr`, `LickTime`, `LickTrind`, `LickPos` and compared `(LickTime[i] − ft[round(LickFr[i])]) * 86400`, `LickTrind[i]` vs `ft_trInd[round]` and `LickPos[i]` vs `ft_Pos[round]` to confirm the "fractional nearest correspondence" of lick frames to imaging frames.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Each lick's fractional frame index is rounded to the nearest imaging frame. A retained bin is labelled 1 if it is the nearest frame of at least one lick belonging to that trial, else 0; there is no smoothing or dilation. Values are stored as `int16`, and `output_values[1] = ['not_licking', 'licking']`. Across the dataset 3.47% of bins are labelled licking; on rewarded sessions the per-session rate is 7–23%, essentially the same as without the running mask (e.g. TX60: 15.5% vs 16.5%).

ii.
```python
        # Licks are assigned to their nearest original imaging frame. If that frame
        # was removed by the paper's running mask, the lick is removed as well.
        trial_licks = lick_frame[lick_trial == trial]
        outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. Metadata: `lick_alignment_rule: "nearest original imaging frame (rounded LickFr)"`. The code comment states the consequence of the running mask for licks explicitly. Rounding (rather than truncating) follows from the step-18 check that `LickTime` is closest to `ft[round(LickFr)]`.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is expressed in imaging frames, so the binary lick series lives on the same grid as `spks`; `np.isin(frames, trial_licks)` evaluates it on exactly the frames used for the neural columns, giving the same `T`. Licks that fall on frames removed by the running mask are dropped rather than reassigned to a neighbouring retained bin, and licks whose `LickTrind` disagrees with the trial of their rounded frame are likewise not carried over.

ii.
```python
        neural[row:row + n] = plane[np.ix_(keep, frames)]
        ...
        outputs[1] = np.isin(frames, trial_licks).astype(np.int16)
```

iii. As above — the AI documented the rule in the code comment and in `metadata['lick_alignment_rule']`, relying on the frame-index correspondence it had verified in step 18.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the virtual-reality position at each imaging frame, in decimetres (0–40 across the 4 m texture, continuing to 60 through the grey space; the maximum observed inside `ft_CorrSpc` is 39.99).

ii.
```python
    frame_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float64)
    ...
        # Source positions are decimetres; the requested bins are four 1 m bins.
        outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Metadata: `source_position_unit: "decimetres"`. In step 15 the AI printed `ft_Pos` at `StartFr`/`GrayFr`/`EndFr` together with `Corridor_Length`/`Texture_Length` to establish the unit and the 0–40 range of the textured section.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. A unit conversion (decimetres → metres) and an integer floor, giving a per-frame class label; nothing else (no interpolation onto a spatial grid, no smoothing). Stored as `int16` with `output_values[2] = ['0-1 m', '1-2 m', '2-3 m', '3-4 m']`.

ii.
```python
        outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```
```python
            ["0-1 m", "1-2 m", "2-3 m", "3-4 m"],
```

iii. Code comment: "Source positions are decimetres; the requested bins are four 1 m bins" — a direct implementation of the decoder-task requirement "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins".

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-length spatial thresholds at 1, 2 and 3 m (i.e. `floor(pos_dm / 10)`), clipped to [0, 3] so that a frame exactly at 40 dm cannot produce a fifth class. The class fractions come out almost exactly uniform (24.998%, 24.875%, 24.958%, 25.169%) — a consequence of the running mask, since the virtual corridor advances at constant speed whenever the mouse runs, whereas without the mask the stationary frames pile up unevenly (e.g. 0.257/0.301/0.210/0.232 in one session).

ii.
```python
        outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Step 27: the AI verified on its dry run that the conversion "gives balanced 1 m position bins" before launching the full run.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one entry per imaging frame, so it is indexed with the same `frames` array as the neural columns; same `T`, bin-for-bin.

ii.
```python
        neural[row:row + n] = plane[np.ix_(keep, frames)]
        ...
        outputs[2] = np.clip(np.floor(frame_pos[frames] / 10.0), 0, 3).astype(np.int16)
```

iii. Same frame-grid argument as 3-c/5-c.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed of the animal at each imaging frame, in cm s⁻¹ (values range roughly −4 to 70 cm s⁻¹ on retained frames).

ii.
```python
    frame_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float64)
    ...
        chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
```

iii. Metadata: `source_speed_unit: "centimetres per second"`. No further justification recorded; `ft_RunSpeed` is the only per-frame speed variable.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A dedicated first pass (`compute_speed_edges`) walks all 89 sessions, gathers `ft_RunSpeed` on exactly the frames that the retained-frame mask keeps, pools them across the whole dataset, and takes the 25/50/75% quantiles as three global bin edges (12.39, 25.33, 40.83 cm s⁻¹). Those single global edges are then applied to every session and trial. Note that this pass determines the number of frames as `len(beh['ft']) - 1` (to avoid opening the 405 GB of spike files) whereas the conversion pass uses the true `spks` frame count; on the session checked the two agree exactly. The same pass also records frame-interval statistics used only for metadata.

ii.
```python
def compute_speed_edges(sessions: list[dict]) -> tuple[np.ndarray, dict]:
    """Global quartiles over the exact samples that enter the decoder."""
    chunks = []
    frame_dts_ms = []
    for i, desc in enumerate(sessions):
        beh = load_behavior(desc)
        # The processed behavior clock has one terminal interpolation sample beyond
        # the Suite2p arrays (as seen throughout the repository). Avoid opening the
        # hundreds of GB of neural files merely to discard that one sample here.
        nframes = len(beh["ft"]) - 1
        valid = retained_frame_mask(beh, nframes)
        chunks.append(np.asarray(beh["ft_RunSpeed"][:nframes][valid], dtype=np.float32))
        ...
    speeds = np.concatenate(chunks)
    edges = np.quantile(speeds, [0.25, 0.50, 0.75])
```

iii. Trajectory step 21: "Speed quartiles will be computed globally over exactly those retained frames." Metadata: `speed_binning_rule: "global quartiles over retained decoder samples"`. The intent is that the four classes each hold 25% of the samples that the decoder actually sees, which is what the decoder task asks for; computing them over the retained frames only (rather than all frames) prevents discarded frames from moving the edges.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. With `np.digitize(speed, edges, right=False)`, giving classes 0–3 with `output_values[3] = ['Q1 (slowest)', 'Q2', 'Q3', 'Q4 (fastest)']`. Because the running mask removes almost all zero-speed frames (0.08% of retained frames are exactly 0, versus 25% without the mask), there is no large tie at zero and value-based thresholds do split the data close to evenly; the verifier reports the four classes at essentially 25% each dataset-wide. Per-session balance is not enforced — the edges are global, so a session with an unusually slow or fast mouse will not be quartered evenly.

ii.
```python
        outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```
```python
            ["Q1 (slowest)", "Q2", "Q3", "Q4 (fastest)"],
```

iii. As in 10-b. Step 27 reports the dry-run check: "produces the four global speed edges at 12.39, 25.33, and 40.83 cm/s"; step 68 confirms on the finished file that "Position bins and global speed bins are each essentially 25% of retained samples."

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one entry per imaging frame and is indexed with the same `frames` array as the neural columns, so it is bin-for-bin aligned and has the same `T`.

ii.
```python
        outputs[3] = np.digitize(frame_speed[frames], speed_edges, right=False).astype(np.int16)
```

iii. Same frame-grid argument as 3-c/5-c/9-d.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Four mechanisms. (1) The behavior clock runs one sample longer than the imaging arrays, so every behavior stream is truncated to the imaging frame count — `min(a.shape[1] for a in planes)` in the conversion, `len(beh['ft']) - 1` in the speed pass. (2) Licks whose rounded frame lies outside the retained frames (including beyond the last imaged frame) simply never match and are dropped. (3) A trial with zero retained frames is counted and skipped. (4) Structural inconsistencies are fatal rather than tolerated: a mismatch between the number of retinotopy labels and the number of neurons raises `ValueError`, a missing behavior key raises `KeyError`, and a session left with fewer than two trials raises `ValueError` and aborts the whole run. In the actual run none of these fired: 0 empty trials, 0 failed sessions, 89/89 converted. NaN handling is limited to `session_day`'s `np.isfinite` check; `SoundTime`/`Trial_start_time` contain no NaNs anywhere in the dataset, so no NaN ever reaches the inputs.

ii.
```python
    nframes = min(a.shape[1] for a in planes)
    planes = [a[:, :nframes] for a in planes]
```
```python
    if len(area_ids) != expected:
        raise ValueError(
            f"{desc['recording_id']}: {len(area_ids)} retinotopy labels for "
            f"{expected} neurons"
        )
```
```python
        if len(frames) == 0:
            dropped_empty += 1
            continue
```
```python
        if len(neural) < 2:
            raise ValueError(f"{desc['recording_id']} retained fewer than two trials")
```

iii. The code comment in `compute_speed_edges` states the one-extra-sample rule and why it is applied without opening the spike files. The strict-raise policy is implicitly justified by the AI's verification habit — it preferred a loud failure to a silently mis-ordered neuron list (steps 26–27 are a dry run of a single session specifically to exercise these checks before the 89-session run).

## 12-a. What are the most time-consuming steps of the code?

i. By far the dominant cost is reading the deconvolved traces: 405 GB of `spk/*.npy` pickles, one per session, fully materialised in memory (`np.load(..., allow_pickle=True).item()`), plus the `np.ix_` gather that copies the kept neurons/frames out of them. Second is writing the 151.5 GB output pickle. Third — and self-inflicted — is behavior I/O: `load_behavior()` re-reads and re-unpickles a whole `Beh_<type>.npy` file on each call, and it is called once per session in each of three passes (category scan, speed-edge pass, conversion). That is 3 × 23.5 GB ≈ 70 GB of pickle reads where 3.7 GB (15 distinct files, read once) would do. The wall-clock evidence in the trajectory is that the conversion ran from step 28 to step 61 — roughly 16 minutes of polling per ~20 sessions — before serialisation.

ii.
```python
    spk_obj = np.load(spk_path, allow_pickle=True).item()
    planes = spk_obj["spks"]
```
```python
            neural[row:row + n] = plane[np.ix_(keep, frames)]
```
```python
    with args.output.open("wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Step 24 shows the AI sizing the spike files (`du -h data/spk/*.npy | sort -h`) and step 25 shows it *removing* a `mmap_mode="r"` attempt in favour of not opening the spike files at all during the speed pass — "Avoid opening the hundreds of GB of neural files merely to discard that one sample here." So it did optimise the spike I/O consciously; it did not consider the repeated behavior I/O.

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three. (1) `for trial in range(ntrials): np.flatnonzero(valid & (trial_stamp == trial))` rescans the whole frame index once per trial — O(n_trials × n_frames) where one `np.argsort`/`np.searchsorted` grouping pass would do (the reference has the same pattern). (2) The per-neuron Python generator that maps `iarea` codes to region indices, `np.fromiter((AREA_ID_TO_REGION[int(x)] for x in ids[local]), ...)`, runs one Python-level dict lookup per neuron — about 4.1 million iterations over the dataset; a 11-element lookup array indexed by `ids[local]` would be a single vectorised op. (3) The plane-by-plane `np.ix_` copy inside the trial loop performs `n_planes × n_trials` fancy-index gathers; gathering the kept columns once per session and then slicing per trial would touch the data once. All three are negligible next to the 405 GB of I/O.

ii.
```python
    for trial in range(ntrials):
        frames = np.flatnonzero(valid & (trial_stamp == trial))
```
```python
        region_chunks.append(
            np.fromiter((AREA_ID_TO_REGION[int(x)] for x in ids[local]), dtype=np.int16)
        )
```
```python
        row = 0
        for plane, keep in zip(planes, plane_keep):
            n = len(keep)
            neural[row:row + n] = plane[np.ix_(keep, frames)]
            row += n
```

iii. Not discussed in the trajectory. The AI's only stated performance reasoning concerns I/O (steps 24–25) and memory ("Allocate once, then copy each imaging plane directly into final row order", a comment aimed at avoiding an intermediate concatenated array of all planes).

## 12-c. What processing does the code repeat multiple times?

i. (1) Behavior loading, as described in 12-a: each session's behavior file is unpickled in full three separate times (once to collect visual-category names, once for speed quartiles, once to convert), and sessions sharing a file do not share the read — 267 full-file unpicklings instead of 15. (2) `retained_frame_mask` is computed twice per session, once in the speed pass and once in the conversion (on a frame count derived differently in each). (3) The visual-category vocabulary is rebuilt from scratch by scanning every `WallName` of every session, although the four categories are fixed properties of the dataset. (4) `discover_sessions` is effectively re-derived information already available from the file listing.

ii.
```python
    categories = set()
    for desc in sessions:
        beh = load_behavior(desc)
        categories.update(visual_category(x) for x in beh["WallName"])
```
```python
    for i, desc in enumerate(sessions):
        beh = load_behavior(desc)
        ...
        valid = retained_frame_mask(beh, nframes)
```
```python
    beh = load_behavior(desc)          # third read of the same file, in convert_session
    ...
    valid = retained_frame_mask(beh, nframes)
```

iii. No justification is recorded; the three passes exist because the category vocabulary and the global speed edges must be known before any trial can be written, and the AI chose the simplest structure (a helper that loads a session's behavior on demand) rather than grouping sessions by behavior file.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The neural arrays are stored as `float32`, which makes the pickle 151.5 GB (141 GiB); the decoder immediately reduces the data to 100 PCs, and `float16` — as the source values are small non-negative deconvolved amplitudes — would have halved the artefact with no practical loss. A 151 GB pickle must be fully materialised in RAM by `train_decoder.py`, so the choice has a real downstream cost. (2) Outputs are `int16` where four classes need `int8`. (3) `compute_speed_edges` computes per-frame inter-frame intervals and their median/IQR for all 89 sessions purely to populate two descriptive metadata fields that nothing reads — and then the declared `time_bin_size` ignores them. (4) The full extra pass over every behavior file to discover four category names that are fixed. (5) Per-session bookkeeping (`n_source_neurons`, `n_empty_trials_dropped`, `source_session_number`, ...) is collected for all 89 sessions; useful for provenance but unused by the decoder.

ii.
```python
        neural = np.empty((n_neurons, T), dtype=np.float32)
```
```python
        outputs = np.empty((4, T), dtype=np.int16)
```
```python
        dt = np.diff(np.asarray(beh["ft"][:nframes], dtype=np.float64)) * 86_400_000.0
        frame_dts_ms.append(dt[(dt > 100) & (dt < 1000)])
    ...
        "observed_median_frame_interval_ms": float(np.median(dt_all)),
        "observed_frame_interval_iqr_ms": np.quantile(dt_all, [0.25, 0.75]).tolist(),
```

iii. Step 27: "this is a large artifact because it intentionally preserves single-neuron, single-frame activity" — the AI treated the size as a deliberate fidelity choice. Step 68: "a full 200-epoch run over the 141 GiB dataset is intentionally left to the grading run", i.e. it was aware of the size but did not revisit the dtype.
