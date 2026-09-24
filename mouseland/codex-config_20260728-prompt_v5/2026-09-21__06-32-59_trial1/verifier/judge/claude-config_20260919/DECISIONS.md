# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Three source directories under `/app/data` are used: `beh/` (behavior), `spk/` (deconvolved traces) and `retinotopy/` (visual-area label per neuron). `beh/Imaging_Exp_info.npy` is read only to build a `raw_key -> mouse name` lookup. The behavior itself is discovered by globbing every top-level `Beh_*.npy` file and loading **all** of them into memory up front; every key inside every file is parsed down to a "raw key" `<mouse>_<date>_<blk>` (dropping any `_swapN` stimulus suffix), and all behavior dictionaries that share a raw key are collected as multiple `BehaviorView`s of the same recording. Duplicated views are checked for agreement on the core trial/frame arrays, then a single canonical view is chosen. Spikes and retinotopy are loaded per session inside `process_session`, with up to 4 threads in `--full` mode.

ii.
```python
def load_exp_info() -> dict[str, list[dict[str, Any]]]:
    return np.load(EXP_INFO_PATH, allow_pickle=True).item()

def iter_behavior_files() -> list[Path]:
    files = []
    for path in sorted(BEH_DIR.glob("Beh_*.npy")):
        if path.name in {
            "Beh_no_pretrain.npy",
            "Beh_pretrain_on_grat_image.npy",
            "Beh_pretrain_on_nat_image.npy",
        }:
            continue
        files.append(path)
    return files
```
```python
    raw_to_views: dict[str, list[BehaviorView]] = defaultdict(list)
    for beh_path in iter_behavior_files():
        exp_type = beh_path.stem.removeprefix("Beh_")
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for full_key, beh in beh_all.items():
            raw_key = parse_raw_session_key(full_key)
            raw_to_views[raw_key].append(
                BehaviorView(exp_type=exp_type, file_name=beh_path.name, full_key=full_key, beh=beh)
            )
```
```python
def load_spike_planes(raw_key: str) -> list[np.ndarray]:
    path = SPK_DIR / f"{raw_key}_neural_data.npy"
    obj = np.load(path, allow_pickle=True).item()
    return obj["spks"]

def load_retinotopy(raw_key: str) -> np.ndarray:
    mouse_date = "_".join(raw_key.split("_")[:4])
    path = RETINO_DIR / f"{mouse_date}_trans.npz"
    return np.load(path, allow_pickle=True)["iarea"]
```

iii. From CONVERSION_NOTES Step 4/5: "142 experiment-session entries, 99 unique behavior keys, 89 raw neural session files" while the paper says "89 recordings in 19 mice", so the AI decided to "Treat the raw imaging recording as the non-duplicated neural session unit" and "Merge duplicates by raw session key when constructing the decoder dataset." The agent verified (Step 10, `cache/step10_sanity_checks.py`) that duplicated views carry identical core arrays. Result: 89 sessions, 19 subjects, 4,691,034 neurons, 38,110 trials.

---

## 1-b. How are the data split into subjects (mice)?

i. The mouse name comes from `mname` in `Imaging_Exp_info.npy`, keyed by the raw session key. `subjects` is the sorted unique set of mouse names over the selected sessions and `subject_idx` is each session's index into that list. 19 mice.

ii.
```python
    exp_info = load_exp_info()
    raw_to_subject: dict[str, str] = {}
    for records in exp_info.values():
        for rec in records:
            raw_key = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
            raw_to_subject[raw_key] = rec["mname"]
```
```python
    subjects = sorted({spec.subject for spec in selected_specs})
    subject_to_idx = {subject: i for i, subject in enumerate(subjects)}
    ...
    "subject_idx": np.array([subject_to_idx[spec.subject] for spec in selected_specs], dtype=np.int8),
```

iii. CONVERSION_NOTES Step 5 variable mapping: "Mouse name from session metadata -> `subjects`, `subject_idx` ... One subject per raw session." The index file already names the mouse, so nothing has to be inferred.

---

## 1-c. How are the data split into sessions?

i. A session is one unique raw imaging recording = `<mouse>_<date>_<blk>`. Behavior keys that carry a `_swap1`/`_swap2` stimulus suffix are stripped back to the raw key, so all "analysis views" of one recording collapse into one session. Before collapsing, the AI asserts the duplicate views agree on `WallName`, `SoundPos`, `StartFr`, `EndFr`, `isRew`, `SoundFr`, `RewardFr`, `ft_trInd`, `ft_CorrSpc`, `ft_Pos`, `ft_RunSpeed`; if they disagree it raises. The canonical view used downstream is the one with the most non-NaN `stim_id` entries. 89 sessions result.

ii.
```python
def parse_raw_session_key(full_key: str) -> str:
    parts = full_key.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected session key format: {full_key}")
    return "_".join(parts[:5])
```
```python
def validate_duplicate_behavior_views(raw_key: str, views: list[BehaviorView]) -> None:
    if len(views) < 2:
        return
    ref = views[0].beh
    keys_to_match = ["WallName", "SoundPos", "StartFr", "EndFr", "isRew", "SoundFr",
                     "RewardFr", "ft_trInd", "ft_CorrSpc", "ft_Pos", "ft_RunSpeed"]
    for view in views[1:]:
        for key in keys_to_match:
            if not arrays_match(np.asarray(ref[key]), np.asarray(view.beh[key])):
                raise ValueError(...)
```
```python
def choose_behavior_reference(views: list[BehaviorView]) -> dict[str, Any]:
    def score(view: BehaviorView) -> tuple[int, str, str]:
        non_nan = int(np.count_nonzero(~np.isnan(np.asarray(view.beh["stim_id"], dtype=float))))
        return (non_nan, view.file_name, view.full_key)
    return max(views, key=score).beh
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "Session unit = unique raw imaging recording (89 total): The paper and raw files agree on 89 recordings; repeated behavior entries across experiment files are reused analysis views of the same recording and should not become duplicate decoder sessions." Key Decision 2 adds that merging views "recovers a complete per-session label map without duplicating neural data."

---

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals the behavior declares. The frames of trial `k` are those imaging frames with `ft_trInd == k` **and** `ft_CorrSpc` (inside the 0–4 m textured corridor) **and** `ft_move > 0` (the animal/VR was moving). Trials are variable length; nothing is padded. Because the `ft_move` mask is applied *within* a trial, the retained frames of a trial are not necessarily contiguous — stationary frames in the middle of a traversal are deleted and the surrounding frames become adjacent bins.

ii.
```python
def get_trial_frame_indices(beh: dict[str, Any], nframes: int) -> list[np.ndarray]:
    ft_trind = np.asarray(beh["ft_trInd"][:nframes])
    ft_corr = np.asarray(beh["ft_CorrSpc"][:nframes], dtype=bool)
    ft_move = np.asarray(beh["ft_move"][:nframes]) > 0
    ntrials = int(beh["ntrials"])
    frame_sets: list[np.ndarray] = []
    for trial in range(ntrials):
        idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
        if idx.size == 0:
            frame_sets.append(idx)
            continue
        frame_sets.append(idx.astype(np.int64, copy=False))
    return frame_sets
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Trial frames will be selected with `(ft_trInd == trial) & ft_CorrSpc & (ft_move > 0)`: This matches the paper and reference code, which restrict imaging analyses to running frames inside the textured corridor, while still aligning the trial to corridor entry rather than to `StartFr`." Step 10 cites the reference `fr_valid = VRmove & isCorridor` in `utils.py:431`. Key Decision 6 adds that this "removes reward-stop periods that the Methods say were excluded from analysis." The agent also rejected slicing by `StartFr` after measuring that `StartFr` precedes the first usable frame by 1–4 frames.

---

## 1-e. How are trials filtered based on quality controls?

i. Only one trial-level filter: a trial with zero valid frames (no running corridor frames that were imaged) is skipped. There is no trial-duration outlier filter, no reward/behavior-based filter, and no per-trial neural-quality filter. A session that ends up with fewer than 2 usable trials raises a `ValueError` (which aborts the conversion rather than skipping the session). All 38,110 non-empty trials are kept.

ii.
```python
    for trial, frame_idx in enumerate(trial_frame_sets):
        if frame_idx.size == 0:
            continue
```
```python
    if len(neural_trials) < 2:
        raise ValueError(f"Session {spec.raw_key} has fewer than 2 valid trials after processing")
```

iii. Not argued explicitly in CONVERSION_NOTES beyond the frame mask. Implicitly, the `ft_move > 0` mask already removes long stationary stretches, so pathologically long trials never appear: the reported maximum trial length is 178 bins (mean T = 22.25). The `< 2 trials` guard exists because the target format requires "at least two trials within each session in order to evaluate the decoder performance."

---

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` inside `spk/<raw_key>_neural_data.npy`, which is a list of one (neurons × frames) array per imaging plane. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. The AI checks that all planes have the same frame count and that the total neuron count equals `len(iarea)`.

ii.
```python
    planes = load_spike_planes(spec.raw_key)
    nframes = int(planes[0].shape[1])
    if any(int(arr.shape[1]) != nframes for arr in planes):
        raise ValueError(f"Plane frame-count mismatch in session {spec.raw_key}")
    ...
    iarea = load_retinotopy(spec.raw_key)
    total_neurons = int(sum(arr.shape[0] for arr in planes))
    if total_neurons != int(iarea.shape[0]):
        raise ValueError(f"Neuron count mismatch in session {spec.raw_key}: ...")
```

iii. CONVERSION_NOTES Step 1/5: "`load_spk` ... concatenates `['spks']` across planes into one neuron-by-frame matrix"; "Use provided deconvolved traces directly; no `dF/F` recomputation."

---

## 2-b. How is the `neural` data processed?

i. No processing at all: no dF/F, no deconvolution, no normalization, no smoothing, no rebinning. For each trial the retained frame columns are taken from each plane and concatenated along the neuron axis, then cast to `float16`. Trials keep their own lengths.

ii.
```python
    neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
```

iii. CONVERSION_NOTES Step 3/4: "Analyses are based on deconvolved fluorescence traces from Suite2p spike deconvolution, not raw fluorescence and not recomputed `dF/F`" and "No `dF/F` recomputation is needed; use provided deconvolved activity matrices directly." Step 6 justifies `float16`: "`train_decoder.py` materializes its own `float32` buffers during training, so the lower on-disk precision cuts serialization cost without changing the decoder path." Slicing per trial per plane (instead of concatenating the whole session) was an explicit memory optimisation.

---

## 2-c. How is the `neural` data filtered based on quality controls?

i. **No neurons are dropped.** Every neuron in every session is kept (4,691,034 total). The retinotopy code is only used to *label* neurons: `iarea == 8 -> V1`, `{0,1,2,9} -> mHV`, `{5,6} -> lHV`, `{3,4} -> aHV`, and everything else (codes `-1`, `7`) -> a fifth region named `other` (585,641 neurons). There is no Suite2p re-curation, SNR, or firing-rate filter.

ii.
```python
BRAIN_REGIONS = ["V1", "mHV", "lHV", "aHV", "other"]

def build_brain_region_idx(iarea: np.ndarray) -> np.ndarray:
    idx = np.full(iarea.shape, 4, dtype=np.int8)
    idx[iarea == 8] = 0
    idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    idx[np.isin(iarea, [5, 6])] = 2
    idx[np.isin(iarea, [3, 4])] = 3
    return idx
```

iii. CONVERSION_NOTES Step 5 Key Decisions 8 and 9: "**All neurons will be retained**: The public reference code does not apply additional global neuron QC beyond Suite2p preprocessing; region labels are metadata, not a filter." and "**Brain-region mapping will follow the reference coarse groups with an additional `other` bucket**: The published helper only uses four named groups for some figures, but the raw data contains many neurons with codes `-1` and `7`; these should be retained and labeled explicitly." Step 3 curation notes: "No global public-code neuron QC beyond the upstream Suite2p pipeline is described."

---

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is corridor entry / trial start. Each trial's array starts at the first *running* corridor frame belonging to that trial and runs to the last one; no common window, no padding, no truncation. `metadata['off_start'] = 0.0`, `metadata['off_end'] = None`, `temporal_alignment_event = "first imaging frame assigned to the current corridor trial (corridor entry)"`. Because `ft_move > 0` is part of the mask, bin 0 is the first *moving* corridor frame, which spot-checks showed is 1–4 frames after `StartFr`.

ii.
```python
    for trial, frame_idx in enumerate(trial_frame_sets):
        if frame_idx.size == 0:
            continue
        ...
        neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0).astype(np.float16, copy=False)
```
```python
        "temporal_alignment_event": "first imaging frame assigned to the current corridor trial (corridor entry)",
        "off_start": 0.0,
        "off_end": None,
```

iii. CONVERSION_NOTES Step 10, Check 5: "`TX60_2021_06_07_1`: first corridor frame minus `StartFr` in `{1,2}`; first running-corridor frame minus `StartFr` ranged `1..4`, mean `1.08` ... Result: `StartFr` is not a reliable direct slicing boundary for the converted running-frame data; using `ft_trInd` plus frame masks avoids the off-by-one risk."

---

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One bin = one imaging frame. The nominal frame rate is 3.17 Hz, so `time_bin_size = 1000/3.17 = 315.46 ms`. **No** rebinning, resampling or position-interpolation is applied to the neural data. (Note that because non-moving frames are deleted inside trials, consecutive bins are not always 315.46 ms apart in wall-clock time even though the metadata declares a single bin size.)

ii.
```python
NOMINAL_FRAME_RATE_HZ = 3.17
TIME_BIN_MS = 1000.0 / NOMINAL_FRAME_RATE_HZ
...
        "time_bin_size": TIME_BIN_MS,
        "nominal_frame_rate_hz": NOMINAL_FRAME_RATE_HZ,
```

iii. CONVERSION_NOTES Step 3: "reference notebook states calcium frame rate `fs = 3.17 Hz` (~315.46 ms/frame)". Step 5 Key Decision 7: "Neural activity will remain on the original frame timebase after running-frame selection: The task is temporally aligned to corridor entry and requires time-varying cue, licking, and speed outputs; framewise traces are therefore more appropriate than position-interpolated spike maps for the decoder." Step 10(d) adds: "the target format requires neural activity by timepoint, not position-binned averages."

---

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. From `SoundPos` (the corridor **position**, in decimetres, at which the cue is played on each trial) and `ft_Pos` (corridor position at each imaging frame), together with a hard-coded constant virtual corridor speed of 6 dm/s (60 cm/s). It is **not** derived from `SoundFr` or from the frame timestamps `ft`.

ii.
```python
CORRIDOR_SPEED_DM_PER_S = 6.0
...
    sound_pos = np.asarray(beh["SoundPos"][: int(beh["ntrials"])], dtype=np.float32)
    ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
...
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
        cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. CONVERSION_NOTES Step 7/10: "A later review found that using raw `ft` timestamps after removing non-running frames preserved long idle pauses in the time inputs. This was corrected by converting corridor position to time using the paper's fixed virtual speed of 60 cm/s, bringing sample input ranges to the expected order of seconds." Trajectory step 391: "during included frames the corridor advances at a fixed 60 cm/s ... I'm switching the two time inputs to position-derived motion time so they stay trial-relative instead of inheriting long idle gaps from dropped frames." (Note: the Step 5 mapping table and Step 5 Key Decision 11 still say the time inputs use `ft` timestamps — those entries were never updated and contradict the final code.)

---

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `cue_sec = (SoundPos[trial] - max(ft_Pos, 0)) / 6.0`, in seconds, stored per timepoint as `float16`. Sign convention is positive before the cue and negative after, matching the "time *to* cue" wording. Full-data range is [-6.4, 6.5] s. No interpolation of `SoundFr` onto a time axis is performed; the real elapsed time between frames is ignored.

ii.
```python
        cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
        ...
        input_trial = np.vstack([
                cue_sec.astype(np.float16, copy=False),
                day_trial,
                t_sec.astype(np.float16, copy=False),
                reward_trial,
        ])
```

iii. CONVERSION_NOTES Step 9: "`time_to_sound_cue_s` range | cue uniformly distributed along corridor; at 60 cm/s this implies about +/- 5.8 s over 0-4 m ... [-6.4, 6.5] | Yes" — i.e. the agent validated the range against the paper's corridor geometry and VR speed rather than against actual cue times.

---

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated on exactly the same `frame_idx` array that selects the neural columns for that trial, so it has the same length and the same per-bin correspondence as the neural matrix.

ii.
```python
    for trial, frame_idx in enumerate(trial_frame_sets):
        ...
        neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)...
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
        cue_sec = (float(sound_pos[trial]) - trial_pos_dm) / CORRIDOR_SPEED_DM_PER_S
```

iii. CONVERSION_NOTES Step 5: "`beh['ft_trInd'] == trial`, `beh['ft_CorrSpc']`, and `beh['ft_move'] > 0` -> Trial time axis for `neural`, `input`, `output` ... this defines aligned timepoints." All streams share one frame index, so alignment is by construction.

---

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. From the calendar date embedded in the raw session key (`mname_YYYY_MM_DD_blk`, itself built from `datexp` and `blk` in `Imaging_Exp_info.npy`), relative to the earliest imaging date of the same mouse.

ii.
```python
def parse_date_and_block(raw_key: str) -> tuple[dt.date, int]:
    parts = raw_key.split("_")
    date = dt.date(int(parts[1]), int(parts[2]), int(parts[3]))
    block = int(parts[4])
    return date, block
...
    subject_first_date = {
        subject: min(date for date, _ in date_blocks)
        for subject, date_blocks in subject_dates.items()
    }
```

iii. CONVERSION_NOTES Step 5 Key Decision 12: "**Day-of-training input will be mouse-relative continuous session day**: Using days since first imaging session (with a tiny block offset if needed) respects within-mouse chronology and preserves the intended 'training day' context."

---

## 4-b. What processing is involved in computing `input` *Day of training*?

i. `training_day = (session_date - first_date_of_that_mouse).days + 0.01 * (block - 1)`. It is a per-trial scalar broadcast across all bins of the trial, stored as `float16`. Observed range across the dataset is [0, 92].

ii.
```python
            training_day = float((date - subject_first_date[subject]).days) + 0.01 * (block - 1)
```
```python
        day_trial = np.full(frame_idx.size, spec.training_day, dtype=np.float16)
```

iii. Same as 4-a. The `0.01 * (block - 1)` term is a tie-breaker so two blocks recorded on the same day are not identical. CONVERSION_NOTES Step 9 lists `training_day` range `[0.0, 92.0]` and flags it as consistent with the "multi-day learning/imaging schedule".

---

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. From `ft_Pos` alone (corridor position at each frame, decimetres) divided by the hard-coded 6 dm/s virtual corridor speed. It is **not** derived from `StartFr` or the frame timestamps `ft`.

ii.
```python
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
        t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. Same rationale as 3-a: CONVERSION_NOTES Step 10, "Inflated time inputs after frame exclusion: Using raw `ft` timestamps after dropping non-running frames produced unrealistic ranges (for example `time_since_trial_start_s` up to ~1765 s). Resolution: recompute `time_to_sound_cue_s` and `time_since_trial_start_s` from corridor position and fixed virtual speed."

---

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `t_sec = max(ft_Pos, 0) / 6.0`, per timepoint, `float16`. Range [0.0, 6.7] s (4 m corridor at 60 cm/s = 6.67 s). Because it is a pure rescaling of `ft_Pos`, and the decoder output `position_bin` is `clip(floor(ft_Pos/10), 0, 3)`, this input is an exact deterministic function of one of the decoder's targets. I verified this on `sample_data.pkl`: `clip(floor(input[2]*6/10), 0, 3)` reproduced `output[2]` with 0 mismatches out of 613 sampled timepoints.

ii.
```python
        t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```
```python
def position_to_bins(ft_pos_dm: np.ndarray) -> np.ndarray:
    pos_dm = np.asarray(ft_pos_dm, dtype=np.float32)
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. CONVERSION_NOTES Step 12: the agent did consider leakage but dismissed it on the wrong evidence — "train and validation accuracies for `position_bin` are nearly identical (`0.3654` vs `0.3650`), which argues against leakage or overfitting" — and concluded "there is no remaining evidence of a conversion bug in this variable." The functional identity between input[2] and output[2] was never checked.

---

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `frame_idx` as the neural columns of the trial, so same length and same per-bin correspondence.

ii.
```python
    for trial, frame_idx in enumerate(trial_frame_sets):
        neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)...
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
        t_sec = trial_pos_dm / CORRIDOR_SPEED_DM_PER_S
```

iii. As in 3-c: all streams share the one frame index, so alignment is by construction.

---

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. From the per-trial `isRew` flag in the canonical behavior view.

ii.
```python
    is_rew = np.asarray(beh["isRew"][: int(beh["ntrials"])], dtype=np.int8)
```

iii. CONVERSION_NOTES Step 5 mapping: "`beh['isRew'][trial]` -> `input[3]` = `reward_available` | Cast to {0,1} | Per-trial corridor identity, not actual reward delivery."

---

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to `{0,1}` and broadcast across the trial's bins as `float16`. Verified range [0.0, 1.0]; it is 0 for all trials in the unsupervised/naive cohorts.

ii.
```python
        reward_trial = np.full(frame_idx.size, float(is_rew[trial]), dtype=np.float16)
```

iii. CONVERSION_NOTES Step 4: "Use direct trial/frame variables from data (`isRew`, `SoundFr`, `RewardFr`, `Reward_Mode`) rather than assuming a single reward contingency."

---

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. From the per-trial `WallName`, but **remapped** through a per-session lookup built from `UniqWalls` and `stim_id` (merged over all behavior views of the recording). `stim_id` is an abstract experiment-role index (0–6) which the AI maps to fixed canonical names via `CANONICAL_STIM_BY_ID`. Wall names that appear in `WallName` but have no `stim_id` (e.g. `circle3`) are kept under their literal name if that name is already in the output vocabulary.

ii.
```python
CANONICAL_STIM_BY_ID = {0: "circle1", 1: "circle2", 2: "leaf1", 3: "leaf2",
                        4: "leaf3", 5: "leaf1_swap1", 6: "leaf1_swap2"}

OUTPUT_STIMULI = ["circle1", "circle2", "circle3", "leaf1", "leaf2", "leaf3",
                  "leaf1_swap1", "leaf1_swap2"]
```
```python
def build_wall_to_stimulus_map(raw_key: str, views: list[BehaviorView]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for view in views:
        uniq_walls = list(map(str, np.asarray(view.beh["UniqWalls"])))
        stim_ids = np.asarray(view.beh["stim_id"])
        for wall, stim_id in zip(uniq_walls, stim_ids):
            if np.isnan(stim_id):
                continue
            canonical = CANONICAL_STIM_BY_ID[int(stim_id)]
            ...
            label_map[wall] = canonical
    present_walls = set(map(str, np.unique(views[0].beh["WallName"])))
    missing = sorted(present_walls - set(label_map))
    for wall in list(missing):
        if wall in OUTPUT_STIMULI:
            label_map[wall] = wall
```

iii. Trajectory step 106: "whether 'visual stimulus category' should use literal wall names (`rock1`, `wood1`, etc.) or the abstract experiment categories ... the reference code appears to abstract rock/wood sessions into the same canonical category space." Step 110: "that lets `rock/wood` mice live in the same output label space as `circle/leaf` mice without inventing ad hoc rules." Step 115: "the same literal wall label can map to different abstract categories in different mice/sessions. The stable representation is a **session-specific** merged mapping from `stim_id`." CONVERSION_NOTES Key Decision 3 and 4 repeat this and add that `circle3` is kept as its own class.

---

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. Each trial's `WallName` string is looked up in the session's `wall_to_stimulus` map and turned into an index into the 8-element `OUTPUT_STIMULI` vocabulary, then broadcast across all bins of the trial as `uint8`. The concrete effect of the `stim_id` remapping (confirmed by reading the raw behavior files):
- `TX108_2023_03_25_1`: `UniqWalls = ['rock1','rock2','wood1','wood2']`, `stim_id = [0,1,2,3]` → rock1 is labelled **"circle1"**, wood1 **"leaf1"**.
- `TX109_2023_04_18_1`: `UniqWalls = ['circle1','circle2','leaf1','leaf2']`, `stim_id = [2,3,0,1]` → circle1 is labelled **"leaf1"** and leaf1 **"circle1"**.
So the same physical texture carries different labels in different sessions, and visually unrelated textures share a label. Full-data class fractions: circle1 0.317, circle2 0.060, circle3 0.007, leaf1 0.333, leaf2 0.169, leaf3 0.058, leaf1_swap1 0.027, leaf1_swap2 0.029.

ii.
```python
        literal_wall = str(wall_name[trial])
        canonical_wall = spec.wall_to_stimulus[literal_wall]
        stimulus_idx = OUTPUT_STIM_TO_IDX[canonical_wall]
        ...
        output_trial = np.vstack([
                np.full(frame_idx.size, stimulus_idx, dtype=np.uint8),
                ...
        ])
```

iii. CONVERSION_NOTES Key Decision 3: "**Visual stimulus output will use session-specific canonical categories, not literal wall names alone**: This preserves cross-mouse comparability when some mice use `rock/wood` stimuli and when some sessions reverse literal circle/leaf naming."

---

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. From `LickFr` (imaging-frame number of each lick) together with `LickTrind` (the trial each lick belongs to).

ii.
```python
    lick_frames = np.asarray(beh["LickFr"], dtype=int)
    lick_trials = np.asarray(beh["LickTrind"], dtype=int)
```

iii. CONVERSION_NOTES Step 5 mapping: "`beh['LickFr']`, `beh['LickTrind']` -> `output[1]` = `licking` | Binary vector over selected frames; 1 if >=1 lick on that frame."

---

## 8-b. What processing is involved in computing `output` *Licking*?

i. The fractional lick frame numbers are truncated to integers, restricted to the licks whose `LickTrind` equals the current trial, intersected with the trial's retained frame indices, and the corresponding bins are set to 1 (`uint8`); everything else is 0. Licks that fall on frames removed by the corridor/move mask are dropped. Overall lick fraction is 0.037.

ii.
```python
def binarize_licks(lick_frames, trial_lick_mask, frame_idx) -> np.ndarray:
    out = np.zeros(frame_idx.size, dtype=np.uint8)
    if frame_idx.size == 0:
        return out
    frames = np.asarray(lick_frames[trial_lick_mask], dtype=np.int64)
    if frames.size == 0:
        return out
    kept = np.intersect1d(frames, frame_idx, assume_unique=False)
    if kept.size == 0:
        return out
    offsets = np.searchsorted(frame_idx, kept)
    valid = (offsets >= 0) & (offsets < frame_idx.size) & (frame_idx[offsets] == kept)
    if np.any(valid):
        out[offsets[valid]] = 1
    return out
```

iii. CONVERSION_NOTES Step 10, sanity check 2: raw behavior/spike files were reloaded independently and the reconstructed licking output matched the converted arrays (`output=True` for `(session, trial) = (0,10)` and `(1,20)`).

---

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` indexes imaging frames, so licking already lives on the neural grid; the binary vector is indexed by the same `frame_idx` used for the neural columns, giving the same length.

ii.
```python
        lick_trial_mask = lick_trials == trial
        licking = binarize_licks(lick_frames, lick_trial_mask, frame_idx)
```

iii. All streams are cut with the same frame index, so alignment is structural.

---

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. From `ft_Pos`, the corridor position at each imaging frame, in decimetres (0–40 across the 4 m texture, on to 60 through the grey space).

ii.
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
    ...
        pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping: "`beh['ft_Pos']` -> `output[2]` = `position_bin` | Discretize corridor position into 4 bins: [0,1), [1,2), [2,3), [3,4] m."

---

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position in decimetres is divided by 10 and floored to give metres, then clipped into `[0, 3]`, stored as `uint8` per timepoint. No smoothing or interpolation. The frames retained are inside `ft_CorrSpc`, so position never exceeds 40 dm anyway; the clip only guards the ends.

ii.
```python
def position_to_bins(ft_pos_dm: np.ndarray) -> np.ndarray:
    pos_dm = np.asarray(ft_pos_dm, dtype=np.float32)
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. CONVERSION_NOTES Step 4: "These are consistent in decimeter units: 0-4 m texture + 4-6 m grey." Step 9 reports the resulting distribution [0.250, 0.249, 0.250, 0.252] as the expected near-uniform coverage of the 4 equal bins.

---

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four fixed, equal-length 1 m bins: `[0,1) m`, `[1,2) m`, `[2,3) m`, `[3,4] m`, i.e. thresholds at 10, 20 and 30 dm. The bins are geometric, not data-driven, exactly as the Decoder Task specified.

ii.
```python
        "output_values": [..., ["0-1m", "1-2m", "2-3m", "3-4m"], ...]
```
```python
    bins = np.floor(pos_dm / 10.0).astype(np.int16)
    return np.clip(bins, 0, 3)
```

iii. The Decoder Task specification: "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins, time varying." The near-uniform observed distribution is used as the sanity check.

---

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` gives one value per imaging frame; it is indexed with the same `frame_idx` as the neural columns, so it has the trial's length and per-bin correspondence.

ii.
```python
        pos_bins = position_to_bins(ft_pos[frame_idx]).astype(np.uint8, copy=False)
```

iii. Structural — one shared frame index for all streams.

---

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. From `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
    ...
        speed_trial = ft_speed[frame_idx].astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 mapping: "`beh['ft_RunSpeed']` -> `output[3]` = `speed_bin` | Global quartile binning across included frames."

---

## 10-b. What processing is involved in computing `output` *Running speed*?

i. The raw speeds of all retained frames are collected per trial during session processing, and speed binning is deferred to a second pass over the whole dataset. In that pass the speeds of **every session and trial** are concatenated into one array, the 0/25/50/75/100 % quantiles are computed once globally, and every trial's speed vector is digitized against the three interior edges. The placeholder row written during session processing is then overwritten.

ii.
```python
def apply_speed_bins(processed_sessions: list[dict[str, Any]]) -> np.ndarray:
    all_speeds = np.concatenate(
        [speed for session in processed_sessions for speed in session["speed_trials"]], axis=0)
    quantiles = np.quantile(all_speeds, [0.0, 0.25, 0.5, 0.75, 1.0])
    quantiles[0] = min(quantiles[0], np.min(all_speeds))
    quantiles[-1] = max(quantiles[-1], np.max(all_speeds))
    for session in processed_sessions:
        for trial_idx, speed_trial in enumerate(session["speed_trials"]):
            bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
            session["output"][trial_idx][3] = bins
    return quantiles.astype(np.float32)
```
```python
                np.zeros(frame_idx.size, dtype=np.uint8),  # placeholder for speed bins
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "**Running speed bins will be computed globally over included frames**: Global quartiles give consistent output classes across sessions, which is better suited to a cross-session decoder than per-session bins." Step 6: "Defer speed binning until after session conversion, using only the already collected per-trial speed arrays."

---

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Four data-driven quartile bins with global edges `[-19.174, 12.422, 25.353, 40.855, 163.487]`. Globally the classes hold exactly 25.0 % of the data each, satisfying the literal specification. Per-session, however, the classes are strongly unbalanced because the edges are global: e.g. session 0 has fractions `[0.956, 0.042, 0.002, 0.000]` and session 16 has `[0.128, 0.064, 0.122, 0.685]`. The bin names are written into `output_values` with their numeric edges.

ii.
```python
            [
                f"q1_[{speed_edges[0]:.3f},{speed_edges[1]:.3f})",
                f"q2_[{speed_edges[1]:.3f},{speed_edges[2]:.3f})",
                f"q3_[{speed_edges[2]:.3f},{speed_edges[3]:.3f})",
                f"q4_[{speed_edges[3]:.3f},{speed_edges[4]:.3f}]",
            ],
```

iii. Decoder Task: "Running speed discretized into 4 bins, each corresponding to 25% of the data." CONVERSION_NOTES Step 9 checks the achieved global distribution: "[0.250, 0.250, 0.250, 0.250] | Yes". The per-session degeneracy is not discussed.

---

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is per imaging frame and is indexed with the same `frame_idx` as the neural columns; the deferred binning writes back into `session["output"][trial_idx][3]`, i.e. the same trial array and the same bin ordering.

ii.
```python
        speed_trial = ft_speed[frame_idx].astype(np.float32, copy=False)
        ...
            session["output"][trial_idx][3] = bins
```

iii. Structural — one shared frame index; the second pass preserves per-trial ordering because `speed_trials` and `output` are appended in lockstep.

---

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards, mostly by assertion rather than repair:
- Every frame-level behavior stream is truncated to the number of imaged frames (`[:nframes]`), and every trial-level stream to `ntrials`.
- Plane frame-count mismatch, neuron/`iarea` count mismatch, conflicting stimulus mappings, disagreeing duplicate behavior views, unmapped wall names, and sessions with <2 usable trials all **raise**, aborting the conversion rather than skipping the offending session.
- Trials with zero valid frames are silently skipped.
- Licks outside the kept frames are dropped by the intersection with `frame_idx`; there is no separate `< nframes` clip.
- Negative `ft_Pos` is floored at 0 for the time inputs and clipped to bin 0 for position.
- In `--full` mode, an exception inside any worker propagates out of `future.result()` and kills the run.
- Minor latent fragility: `subject_idx` and `metadata` are built from `selected_specs` while `neural`/`input`/`output` come from `processed_sessions`; these are only aligned because no session is ever dropped silently.

ii.
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nframes], dtype=np.float32)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nframes], dtype=np.float32)
    wall_name = np.asarray(beh["WallName"][: int(beh["ntrials"])], dtype=object)
```
```python
    if total_neurons != int(iarea.shape[0]):
        raise ValueError(f"Neuron count mismatch in session {spec.raw_key}: ...")
```
```python
        trial_pos_dm = np.maximum(ft_pos[frame_idx], 0.0)
```

iii. CONVERSION_NOTES Step 10 Check 5 (edge cases) documents the `StartFr` off-by-one investigation and the decision to slice by `ft_trInd` instead. The dataset is described as clean; the full run reported no failures.

---

## 12-a. What are the most time-consuming steps of the code?

i. Reading and slicing the spike files (405 GB on disk) and pickling the 81 GiB output. The script prints per-session timing, and `--full` runs 4 threads over sessions; the full conversion took 5.16 min. Building the session specs is also non-trivial: all ~24 top-level `Beh_*.npy` files (~5 GB) are loaded up front and every view is kept alive for the whole run.

ii.
```python
    max_workers = 1 if args.sample else min(4, os.cpu_count() or 1)
    ...
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(process_session, spec, i) for i, spec in enumerate(selected_specs)]
```
```python
    print(f"[{session_idx:03d}] {spec.raw_key}: trials={len(neural_trials)} "
          f"neurons={total_neurons} frames/session={nframes} time={elapsed:.2f}s", flush=True)
```

iii. CONVERSION_NOTES Step 6/7: "Add session-level parallel processing for `--full` mode (up to 4 workers) because sessions are independent and the serial full-run estimate exceeded 15 minutes"; "Process spike planes directly and concatenate only per-trial slices, avoiding a full-session concatenated copy."

---

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i.
- `get_trial_frame_indices` scans the whole frame index once per trial (`np.flatnonzero((ft_trind == trial) & ...)`), O(ntrials × nframes); a single `np.argsort`/grouping pass would do.
- The per-trial loop slices each imaging plane separately, so fancy indexing is invoked `ntrials × nplanes` times (~38,110 × 3) instead of once per session.
- `binarize_licks` is called per trial and does a fresh `intersect1d` + `searchsorted`; a single per-session lick flag vector (as in the reference) would remove it entirely.
- `apply_speed_bins` loops over every trial in Python to write back the digitized bins.
- In `--sample` mode, `select_sample_specs` re-runs `get_trial_frame_indices` over *every* session and loads every retinotopy file just to score candidates.

ii.
```python
    for trial in range(ntrials):
        idx = np.flatnonzero((ft_trind == trial) & ft_corr & ft_move)
```
```python
        neural_trial = np.concatenate([plane[:, frame_idx] for plane in planes], axis=0)
```
```python
    for session in processed_sessions:
        for trial_idx, speed_trial in enumerate(session["speed_trials"]):
            bins = np.digitize(speed_trial, quantiles[1:-1], right=False).astype(np.int16)
```

iii. Not discussed as such in CONVERSION_NOTES; the agent's efficiency effort went to threading, `float16` storage and avoiding a whole-session spike concatenation, all of which dominate the I/O-bound runtime anyway.

---

## 12-c. What processing does the code repeat multiple times?

i.
- All behavior files are loaded and **all** views are retained for the entire run, even though only one canonical view per recording is used. 33 recordings appear in more than one file, so their trial/frame arrays are held several times over.
- `validate_duplicate_behavior_views` re-compares 11 full arrays for every duplicate pair, on top of the earlier exploratory verification.
- `build_wall_to_stimulus_map` iterates every view of a recording, re-deriving the same mapping repeatedly.
- In `--sample` mode `get_trial_frame_indices` and the retinotopy load are performed once for scoring and again for the real conversion.
- `int(beh["ntrials"])` and similar casts are recomputed inside loops.

ii.
```python
    for beh_path in iter_behavior_files():
        beh_all = np.load(beh_path, allow_pickle=True).item()
        for full_key, beh in beh_all.items():
            raw_to_views[raw_key].append(BehaviorView(..., beh=beh))
```
```python
        validate_duplicate_behavior_views(raw_key, views)
        ...
                behavior_views=views,
                behavior_ref=choose_behavior_reference(views),
                wall_to_stimulus=build_wall_to_stimulus_map(raw_key, views),
```
```python
def count_trials_with_corridor_licks(beh: dict[str, Any]) -> int:
    frame_sets = get_trial_frame_indices(beh, len(beh["ft"]))
```

iii. The duplicate validation is deliberate — CONVERSION_NOTES Step 5 planned sanity check: "Verify for duplicated raw sessions across behavior files that `WallName`, `SoundPos`, `StartFr`, `EndFr`, and `isRew` are identical." Holding all views is a side effect of that design, not an argued choice.

---

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- `speed_trials`, a `float32` copy of the raw running speed for every retained frame of every trial, is carried in each session dict for the second pass and then never written to the pickle.
- A `uint8` zero row is written into every output array as a speed placeholder and immediately overwritten by `apply_speed_bins`.
- `SessionSpec.behavior_views` keeps all raw behavior views alive after `behavior_ref` and `wall_to_stimulus` have been derived.
- `iter_behavior_files` filters three filenames that do not exist at the top level of `beh/` (they live in `beh/Unsupervised_pretraining_behavior/`), so the exclusion is dead code.
- `process_session` returns `reward_mode` and `nframes`; only `reward_mode` reaches metadata as a deduplicated list.
- `binarize_licks` recomputes a `valid` mask after `intersect1d` has already guaranteed membership.
- In `--sample` mode, the scoring machinery in `select_sample_specs` (lick counts, neuron counts, stimulus counts over all 89 sessions) is discarded after picking 2 sessions.

ii.
```python
        "speed_trials": speed_trials,
```
```python
                np.zeros(frame_idx.size, dtype=np.uint8),  # placeholder for speed bins
```
```python
    kept = np.intersect1d(frames, frame_idx, assume_unique=False)
    offsets = np.searchsorted(frame_idx, kept)
    valid = (offsets >= 0) & (offsets < frame_idx.size) & (frame_idx[offsets] == kept)
```

iii. CONVERSION_NOTES Step 6 frames the deferred speed binning as a speedup: "Defer speed binning until after session conversion, using only the already collected per-trial speed arrays." The retained `speed_trials` are the cost of the global-quantile decision (10-b); a per-session split would not have needed them.
