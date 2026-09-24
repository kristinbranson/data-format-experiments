# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads from three subfolders of `/app/data`: `beh/` (behavior), `spk/` (deconvolved calcium traces) and `retinotopy/` (per-neuron visual area). Unlike the reference, the master index `beh/Imaging_Exp_info.npy` is **not** used to enumerate recordings — it is loaded only to attach a set of experiment-type labels to each recording (`load_experiment_type_map`). Enumeration is done by globbing every top-level `beh/Beh_*.npy` file and iterating over the keys of each behavior dictionary. A recording ("base") is the first five underscore-separated fields of a behavior key, `<mouse>_<year>_<month>_<day>_<blk>`; behavior records are held in memory for every session (`SessionSpec.record`) in a first behavior-only pass, and then the spike file and retinotopy file are read once per session in the main conversion pass. The glob is non-recursive, so the behavior-only pretraining cohorts under `beh/Unsupervised_pretraining_behavior/` (which have no spike files) are excluded. Result: 99 unique behavior keys → 89 unique recordings → 89 sessions, 19 subjects.

ii.
```python
def load_experiment_type_map(root: Path) -> dict[str, set[str]]:
    exp_info = np.load(root / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()
    base_to_experiments: dict[str, set[str]] = {}
    for exp_type, records in exp_info.items():
        for rec in records:
            base = f"{rec['mname']}_{rec['datexp']}_{rec['blk']}"
            base_to_experiments.setdefault(base, set()).add(exp_type)
    return base_to_experiments


def select_representative_sessions(root: Path) -> list[SessionSpec]:
    beh_dir = root / "data" / "beh"
    base_to_experiments = load_experiment_type_map(root)
    selected: dict[str, SessionSpec] = {}

    for beh_path in sorted(beh_dir.glob("Beh_*.npy")):
        exp_type = beh_path.stem.replace("Beh_", "")
        beh_dict = np.load(beh_path, allow_pickle=True).item()
        for key, record in beh_dict.items():
            base = "_".join(key.split("_")[:5])
            ...
```
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)


def load_region_index(root: Path, base: str, nneurons: int) -> np.ndarray:
    subject, date, _ = parse_base(base)
    ret_path = root / "data" / "retinotopy" / f"{subject}_{date.strftime('%Y_%m_%d')}_trans.npz"
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=float)
```

iii. From CONVERSION_NOTES.md Step 2/Step 4: the AI observed that `Imaging_Exp_info.npy` holds 142 experiment-session records that collapse to 99 unique behavior keys and 89 unique neural recordings, and that "some sessions are reused across multiple experiment labels". It therefore decided to enumerate from the behavior dictionaries themselves and deduplicate to recording bases, which it checked against the paper's "We performed 89 recordings in 19 mice". Reference-code comparison (Step 10, Check 3) notes `load_spike_matrix` mirrors `code/utils.py::load_spk` (concatenate the stored `spks` planes) and `load_region_index` mirrors `load_retino` / `neu_area_ID`.

## 1-b. How are the data split into subjects?

i. The mouse name is parsed as the first underscore-separated field of the recording base (`parse_base`) and stored on every `SessionSpec`. `subjects` is the sorted list of unique mouse names, and `subject_idx` is each session's index into that list. Result: 19 subjects with the same sessions-per-subject counts as the reference (DR10:6, DR15:5, …, VR2:7).

ii.
```python
def parse_base(base: str) -> tuple[str, datetime, str]:
    parts = base.split("_")
    if len(parts) != 5:
        raise ValueError(f"Unexpected recording base format: {base}")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    blk = parts[4]
    return subject, date, blk
```
```python
subject_names = sorted({spec.subject for spec in specs})
subject_to_idx = {name: idx for idx, name in enumerate(subject_names)}
...
subject_idx.append(subject_to_idx[spec.subject])
```

iii. CONVERSION_NOTES.md Step 5 mapping table: "Session mouse name parsed from base id → `subjects`, `subject_idx`; unique mouse ids in deterministic sorted order; 19 subjects expected after deduplication." Planned sanity check: "Subject-count check: confirm the converted dataset contains 19 mice after deduplication" (reported as passing in Step 9/10).

## 1-c. How are the data split into sessions?

i. A session is one recording base `<mouse>_<date>_<blk>`. Behavior keys that share a base (e.g. `..._swap1`, `..._swap2`, or the same recording appearing in several `Beh_<exp_type>.npy` files) are collapsed to a single session; the retained behavior key is chosen by `session_key_score`, which prefers a non-swap key, then the key with the most finite `stim_id` entries, then the most unique `WallName`s, then alphabetical order. Sessions are sorted by `(subject, date, blk)`. This yields 89 sessions. The reference instead takes the first entry seen in `Imaging_Exp_info.npy` and keys the behavior with `session_id + '_' + stimtype`; both approaches keep exactly one behavior record per recording and both produce 89 sessions.

ii.
```python
def session_key_score(key: str, record: dict) -> tuple[int, int, int, str]:
    stim_id = np.asarray(record.get("stim_id", []), dtype=float)
    non_nan_stim = int(np.isfinite(stim_id).sum()) if stim_id.size else 0
    unique_walls = int(len(np.unique(record.get("WallName", []))))
    swap_penalty = 1 if "swap" in key else 0
    return (swap_penalty, -non_nan_stim, -unique_walls, key)
```
```python
            if base not in selected:
                selected[base] = SessionSpec(base=base, key=key, record=record,
                                             subject=subject, date=date, blk=blk)
            else:
                current = selected[base]
                if session_key_score(key, record) < session_key_score(current.key, current.record):
                    current.key = key
                    current.record = record
```

iii. Key Decisions 1 and 2 in CONVERSION_NOTES.md Step 5: "Use 89 unique recording bases as sessions: This matches the paper's '89 recordings in 19 mice' and avoids leakage from duplicated experiment labels and `swap1`/`swap2` aliases that share the same raw trials"; "Prefer a plain behavior key when multiple keys share one recording base … the plain key already contains the full `WallName` set, so it is the best canonical representative."

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` trials declared by the behavior file, and each frame is assigned to a trial by `ft_trInd`. Within a trial, the AI keeps **only** frames that are inside the textured corridor (`ft_CorrSpc`) **and** during running (`ft_move > 0`); frames with non-finite `ft_trInd` are dropped, and all behavior arrays are first truncated to `min(len)` over the frame-level arrays and later clipped to the number of imaged frames. Trials therefore have variable length and are *not* temporally contiguous — stationary frames inside a traversal are excised, which reduces mean trial length from 32.6 bins (reference) to 21.9 bins.

ii.
```python
def build_trial_frame_indices(record: dict, nfr: int) -> list[np.ndarray]:
    ft_tr = np.asarray(record["ft_trInd"][:nfr], dtype=float)
    valid = np.isfinite(ft_tr)
    valid_idx = np.flatnonzero(valid)
    trial_ids = ft_tr[valid].astype(int)
    keep = np.asarray(record["ft_CorrSpc"][:nfr], dtype=bool)[valid]
    keep &= np.asarray(record["ft_move"][:nfr], dtype=float)[valid] > 0

    ntrials = int(record["ntrials"])
    frame_indices: list[np.ndarray] = []
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
    return frame_indices
```

iii. Key Decisions 4 and 5 in Step 5: "Align trials using frame-level `ft_trInd` rather than `StartFr`/`EndFr`: this matches the reference code and avoids boundary ambiguities"; "Retain paper-style running corridor frames (`ft_CorrSpc` and `ft_move > 0`): this is the closest match to the published analyses, which consistently use running timepoints in the textured corridor." Step 3 records the paper statement "We only considered timepoints during running for analysis", and Step 1 notes that `get_interpPos_spk` uses `VRmove = beh['ft_move'][:nfr] > 0`. The trajectory (step 140) shows the AI explicitly framed this as "the biggest mapping tradeoff … whether to keep all trial frames or restrict to the running corridor frames the paper actually analyzes", and resolved it in favour of the running mask after checking that most trials stay usable.

## 1-e. How are trials filtered based on quality controls?

i. Three per-trial criteria, all applied to the *retained* (running-corridor) frames: at least 5 retained timepoints; retained duration (last retained frame time − `Trial_start_time`) ≤ 60 s; and no gap between consecutive retained frames > 10 s. Sessions with fewer than two surviving trials are dropped (none were). 2,217 trials were removed (470 for duration, 1,747 for gaps, the rest for too few frames), leaving 35,893 of 38,110 trials. The reference instead keeps any trial with ≥1 frame and length ≤ the 99th percentile of all trial lengths (382 trials dropped, 37,728 kept).

ii.
```python
MIN_TRIAL_TIMEPOINTS = 5
MAX_RETAINED_TRIAL_DURATION_S = 60.0
MAX_RETAINED_INTERFRAME_GAP_S = 10.0


def trial_passes_quality_filters(record, trial_idx, frame_idx, nfr) -> tuple[bool, str | None]:
    frame_idx = np.asarray(frame_idx, dtype=np.int64)
    frame_idx = frame_idx[frame_idx < nfr]
    if frame_idx.size < MIN_TRIAL_TIMEPOINTS:
        return False, "too_few_timepoints"

    frame_times = np.asarray(record["ft"][:nfr], dtype=float)[frame_idx]
    retained_duration_s = float((frame_times[-1] - float(record["Trial_start_time"][trial_idx])) * SEC_PER_DAY)
    if retained_duration_s > MAX_RETAINED_TRIAL_DURATION_S:
        return False, "retained_duration_gt_60s"

    if frame_idx.size > 1:
        max_gap_s = float(np.max(np.diff(frame_times)) * SEC_PER_DAY)
        if max_gap_s > MAX_RETAINED_INTERFRAME_GAP_S:
            return False, "interframe_gap_gt_10s"

    return True, None
```
```python
        if len(session_neural) < 2:
            removed_sessions.append((spec.base, len(session_neural), "fewer_than_two_valid_trials"))
```

iii. Step 5 Key Decision 6: "Filter bad/unusable trials only when required by decoder format. Trials with fewer than 5 retained neural timepoints will be excluded; sessions with fewer than 2 remaining trials will be excluded. This is a decoder-format curation step, not a paper curation step." The 60 s / 10 s filters were added during Step 10 after the AI traced extreme `time_to_sound_cue_s` / `time_since_trial_start_s` values (e.g. `TX88_2022_07_19_1` trial 391 spanning 1,765 s with only 36 retained running frames) to trials where the animal stalled: "a small number of raw trials keep the same `ft_trInd` for an exceptionally long wall-clock period, with only sparse later running frames still marked as the same trial. That is raw-data behavior, not a cross-session indexing mistake" (trajectory step 265). Step 10 records that no session fell below 2 trials (smallest kept session: 62 trials).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` in `spk/<base>_neural_data.npy`, a list of one (neurons × frames) array per imaging plane, concatenated along the neuron axis. The per-neuron visual area comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`. Both are identical to the reference sources.

ii.
```python
def load_spike_matrix(root: Path, base: str) -> np.ndarray:
    path = root / "data" / "spk" / f"{base}_neural_data.npy"
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
    return spk.astype(np.float32, copy=False)
```
```python
    ret = np.load(ret_path, allow_pickle=True)
    iarea = np.asarray(ret["iarea"], dtype=float)
    if iarea.shape[0] != nneurons:
        raise ValueError(
            f"Retinotopy neuron count mismatch for {base}: iarea={iarea.shape[0]}, spikes={nneurons}"
        )
```

iii. Step 4 discrepancy table: "`load_spk` loads stored `spks` directly; no dF/F computation anywhere in code … the stored `spks` arrays are the deconvolved fluorescence traces; do not compute dF/F", cross-checked against the paper's "All our analyses were based on deconvolved fluorescence traces."

## 2-b. How is the `neural` data processed?

i. No signal processing at all: no dF/F, no deconvolution, no z-scoring, no smoothing, no rebinning. The concatenated `spks` matrix is kept in `float32` and the columns for each trial's retained frames are sliced out. What *is* done to the neural array beyond slicing is a large **neuron subselection** (see 2-c), which reduces 4.69 M raw neurons to 304,548 exported neurons (mean 3,422/session versus 46,128 in the reference).

ii.
```python
        spk = load_spike_matrix(root, spec.base)
        nneurons, nfr = spk.shape
        region_idx = load_region_index(root, spec.base, nneurons)
        selected_neurons = select_decoder_neurons(spk, spec.record, region_idx)
        spk = spk[selected_neurons]
        region_idx = region_idx[selected_neurons]
```
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
```

iii. Step 5 Key Decision 3: "Keep the stored `spks` arrays directly, but export only a selective decoder subset. The paper and code both indicate these are already deconvolved fluorescence traces; no dF/F or additional normalization should be applied, but a compact neuron subset is necessary for tractable decoding." The trigger was file size: "The sample pickle came out at `3.84 GiB` for just two sessions, so the full all-neuron export is not viable" (trajectory step 179).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are combined inside `select_decoder_neurons`. (1) Area: `iarea` is mapped to V1 (8), mHV (0,1,2,9), lHV (5,6), aHV (3,4), with everything else labelled `other`; only the four visual areas are used for selection. (2) A **functional stimulus-selectivity filter** copied from the reference *coding-direction analysis*: for a chosen pair of reference stimuli, a d′ is computed per neuron over running corridor frames, neurons are restricted to those whose mean corridor response exceeds their grey-space response, and within each visual area only the top 5% and bottom 5% of d′ neurons are kept. Fallbacks exist (keep all neurons if no stimulus pair can be chosen; keep all visual-area neurons if either stimulus has <10 frames or fewer than 20 candidates in an area) but none of them fired in the full run — the exported data contains 0 `other` neurons and max 6,178 neurons/session. Net effect: ~93% of visual-area neurons are discarded. The reference applies only the area filter and keeps 4,105,393 of 4,691,034 neurons.

ii.
```python
def select_decoder_neurons(spk, record, region_idx) -> np.ndarray:
    """Select a compact, paper-grounded neuron subset for decoding.

    Uses the same 5% positive / 5% negative selectivity logic as the reference
    coding-direction analysis, computed on running frames in the textured corridor.
    """
    ...
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
    selected = np.zeros(spk.shape[0], dtype=bool)
    for area in range(4):
        candidates = corr_neu & (region_idx == area) & np.isfinite(dp)
        if candidates.sum() == 0:
            continue
        if candidates.sum() < 20:
            selected[candidates] = True
            continue
        hi, lo = np.nanpercentile(dp[candidates], [95, 5])
        selected |= candidates & ((dp >= hi) | (dp <= lo))
```
```python
    region_idx = np.full(nneurons, 4, dtype=np.int16)
    region_idx[iarea == 8] = 0
    region_idx[np.isin(iarea, [0, 1, 2, 9])] = 1
    region_idx[np.isin(iarea, [5, 6])] = 2
    region_idx[np.isin(iarea, [3, 4])] = 3
```

iii. Step 5 Key Decision 10: "Select neurons using the paper's 5% positive / 5% negative per-area logic: For each session, compute familiar-stimulus `d'` on running corridor frames, restrict to corridor-responsive neurons, and keep the top 5% positive and top 5% negative neurons within each of `V1`, `mHV`, `lHV`, `aHV`. This follows the reference coding-direction analysis and reduces the export size enough to make full decoding practical." Step 9's consistency table honestly labels the resulting neuron counts as "Partial: selected-subset export, not raw-count export". Step 12 concludes that the weak position/speed decoding "likely reflects the restricted paper-style stimulus-selective neuron subset exported for tractable full-dataset decoding, not a conversion error".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry (trial start). Each trial's neural matrix is the set of columns of `spks` at that trial's retained frames, in frame order, starting at the first retained running frame of the traversal. Trials keep their native, variable length; nothing is padded, truncated to a common window, or resampled. Metadata records `temporal_alignment_event = "corridor entry / trial start"`, `off_start = 0.0`, `off_end = None` — the same choices as the reference. The one difference from the reference is that non-running frames inside the traversal are removed, so bin *k* of a trial is not a fixed time after corridor entry.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```
```python
        "temporal_alignment_event": "corridor entry / trial start",
        "off_start": 0.0,
        "off_end": None,
```

iii. Step 5 Key Decision 9: "Use raw frame timestamps instead of resampling to a new clock." All streams (neural, inputs, outputs) are indexed by the identical `frame_idx` array for a trial, which the AI verified in Step 10 Check 2 by reconstructing three trials from the raw files with `np.allclose()`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One bin = one imaging frame. `time_bin_size` is reported as the median over sessions of each session's median inter-frame interval `diff(ft)`, giving 314.694 ms (the per-session medians span 314.392–315.367 ms). The reference hard-codes 1000/3.17 = 315.457 ms.

ii.
```python
        ft = np.asarray(record["ft"][:spec.estimated_nfr], dtype=float)
        dt_ms = np.diff(ft) * MS_PER_DAY
        spec.median_frame_dt_ms = float(np.median(dt_ms))
        frame_dt_medians.append(spec.median_frame_dt_ms)
```
```python
        "time_bin_size": float(np.median(frame_dt_medians)),
        "frame_bin_source": "native imaging frame timestamps; no temporal resampling",
```

iii. Step 4: "Code mostly works in original imaging frames or in position bins, not a common ms grid … median imaging frame interval is about 314.67 ms in a sampled session. Use raw frame timestamps to build a trial-start-aligned time grid; this preserves the reference streams while satisfying the decoder format requirement." Step 5 Key Decision 9 adds: "Median imaging frame spacing is very consistent across sessions (~314.7 ms), so keeping native frame bins is more faithful than interpolation to a synthetic time grid."

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. The trial-level timestamp `SoundTime` and the per-frame timestamp array `ft` (both MATLAB datenums, in days). The reference instead uses the cue *frame number* `SoundFr` interpolated onto the frame-time axis; `SoundTime` is the same event expressed directly as a timestamp.

ii.
```python
def session_trial_info(record: dict, trial: int, frame_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
```

iii. Step 5 mapping table: "`beh['SoundTime']` and retained frame times from `beh['ft']` → `input[0]` = `time_to_sound_cue_s` … `spk_2_cue` (same cue timing source), paper methods on cue timing." Planned sanity check: "verify `time_to_sound_cue_s` equals raw `SoundTime - ft[mask]`", reported as passing in Step 10 Check 2.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For every retained frame of a trial, `(SoundTime[trial] − ft[frame]) × 86400`, i.e. seconds, positive before the cue and negative after it — the same sign convention as the reference. Stored as `float32`, time-varying. No interpolation is needed because `SoundTime` is already a timestamp. Observed range over the full dataset: [−49.1, 45.7] s (reference: [−72.2, 73.4] s, wider because the reference keeps stalled frames).

ii.
```python
SEC_PER_DAY = 24.0 * 3600.0
...
    time_to_cue = (float(record["SoundTime"][trial]) - frame_times) * SEC_PER_DAY
    ...
    return time_to_cue.astype(np.float32), time_since_start.astype(np.float32)
```

iii. Step 5: "For each retained frame, compute `SoundTime - frame_time` in seconds; positive before cue, negative after cue. Time-varying continuous input." Step 10 records that after the stalled-trial filters the range became physically plausible.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frame_idx` array used to slice the neural columns of that trial, so it has the same length and the same bin-by-bin correspondence.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
            ...
            input_trial = np.vstack(
                [time_to_cue, training_day, time_since_start, reward_available]
            ).astype(np.float32, copy=False)
```

iii. All streams in this dataset share the imaging-frame grid; the AI's Step 10 Check 2 verified with `np.allclose()` that the exported inputs equal `SoundTime − ft[mask]` recomputed from the raw files for three sampled trials, and the `--show-processing` plots overlay inputs and outputs on the same retained-frame axis.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The recording date parsed out of the session base (`<mouse>_<Y>_<M>_<D>_<blk>`), relative to the earliest recording date for that mouse. No behavior variable is used.

ii.
```python
    specs = sorted(selected.values(), key=lambda s: (s.subject, s.date, int(s.blk)))
    first_date_by_subject: dict[str, datetime] = {}
    for spec in specs:
        first_date_by_subject.setdefault(spec.subject, spec.date)
        spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```

iii. Step 5 Key Decision 12: "Represent `training_day` as elapsed days since the mouse's first retained imaging session. The paper does not define this variable, but the decoder task requires it and the raw dates provide a reproducible continuous proxy."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. Integer calendar days elapsed since the subject's first recording, as a float, broadcast across every bin of every trial of the session. The count is computed over all 89 sessions before any sampling, so a `--sample` run and a full run agree. Observed range [0, 92] days. The reference instead uses the *ordinal index* of the recording within the mouse (0–7), so the two encodings are monotonically related but on very different scales.

ii.
```python
        spec.training_day = float((spec.date - first_date_by_subject[spec.subject]).days)
```
```python
            training_day = np.full(frame_idx.size, spec.training_day, dtype=np.float32)
```

iii. As above (Step 5 Key Decision 12) plus Step 7 Decision 7: "Store all four decoder inputs in a 2D time-varying array. Even the per-trial variables (`training_day`, `reward_available`) will be repeated across timepoints so every trial has a uniform `(4, T)` input shape." Step 9 lists the resulting `training_day` range [0.0, 92.0] as "Expected for derived variable".

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. The trial-level timestamp `Trial_start_time` and the per-frame timestamps `ft`. The reference uses the corridor-entry frame number `StartFr` interpolated onto the frame-time axis.

ii.
```python
    frame_times = np.asarray(record["ft"], dtype=float)[frame_idx]
    time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```

iii. Step 5 mapping table: "Retained frame times from `beh['ft']` and `beh['Trial_start_time']` → `input[2]` = `time_since_trial_start_s` … Raw timestamps; consistent with paper frame-time analyses."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft[frame] − Trial_start_time[trial]) × 86400` seconds for each retained frame, positive after trial start, stored as `float32`, time-varying. No interpolation. Observed range [0.0, 54.3] s; the 60 s trial filter bounds it. Because non-running frames are excised, the first value of a trial is the time of the first *running* corridor frame rather than of corridor entry itself.

ii.
```python
    time_since_start = (frame_times - float(record["Trial_start_time"][trial])) * SEC_PER_DAY
```
```python
        "max_retained_trial_duration_s": MAX_RETAINED_TRIAL_DURATION_S,
```

iii. Step 5: "`frame_time - Trial_start_time` in seconds for each retained frame. Time-varying continuous input." Step 10 documents that the pre-filter maximum was pathological (a trial spanning 1,765 s) and that the 60 s / 10 s filters were introduced specifically to bring this input back into a physically sensible range.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed on the same `frame_idx` as the neural columns, in the same order, giving one value per neural bin.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            time_to_cue, time_since_start = session_trial_info(spec.record, trial_idx, frame_idx)
```

iii. Same frame-grid argument as 3-c; verified in Step 10 Check 2 against raw `ft` and `Trial_start_time` with `np.allclose()`.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. The trial-level boolean array `isRew`, identical to the reference.

ii.
```python
            reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
```

iii. Step 5 mapping table: "`beh['isRew']` → `input[3]` = `reward_available`; trial-level binary repeated across timepoints; 1 only for rewarded-corridor task trials, 0 otherwise. In unsupervised / naive / grating sessions this is expected to be all 0."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to bool then to float 0.0/1.0 and broadcast across the trial's bins. No other processing. Full-dataset range [0, 1]; the per-session ranges show entire cohorts of sessions that are all-zero, as expected for the unsupervised/naive mice.

ii.
```python
            reward_available = np.full(frame_idx.size, float(bool(spec.record["isRew"][trial_idx])), dtype=np.float32)
            input_trial = np.vstack(
                [time_to_cue, training_day, time_since_start, reward_available]
            ).astype(np.float32, copy=False)
```

iii. Step 4 discrepancy table: "Unsupervised sessions have valid `SoundPos` but `RewPos` is all `NaN` and `isRew.sum()==0` in sampled sessions … Interpret unsupervised sessions as preserving corridor and cue structure without reward availability."

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. The trial-level `WallName` string. The category vocabulary is the sorted union of `UniqWalls` across all selected sessions, which yields 15 labels: `circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`. The reference uses the same `WallName` field but collapses these 15 names to 4 base textures (`circle`, `leaf`, `rock`, `wood`).

ii.
```python
def collect_visual_categories(specs: list[SessionSpec]) -> list[str]:
    categories = sorted({str(wall) for spec in specs for wall in spec.record["UniqWalls"]})
    return categories
```
```python
            stim_idx = np.full(
                frame_idx.size,
                visual_to_idx[str(spec.record["WallName"][trial_idx])],
                dtype=np.int16,
            )
```

iii. Step 5 mapping table: "Global categorical mapping over all unique stimulus names across retained sessions; repeated across timepoints. 15 global categories found in imaging data." The prompt the AI received specified the output as "Visual stimulus category. e.g. circle1, leaf2, etc., per-trial", i.e. it named the fine-grained wall identities as the example categories, which is what the AI implemented. The AI also notes elsewhere (Step 1) that `stim_id` is an alternative but is session-relative.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The wall name is looked up in the global 15-entry vocabulary and the index is broadcast across all bins of the trial, stored as `int16` in row 0 of the output matrix. `output_values[0]` is the list of 15 names. No grouping of crops or spatial-shuffle ("swap") variants is performed, so e.g. `leaf1`, `leaf1_swap1`, `leaf1_swap2`, `leaf2` and `leaf3` are five distinct classes. Full-dataset fractions: circle1 0.247, leaf1 0.260, leaf2 0.127, …, wood1_swap1 0.008. Many classes appear in only a handful of sessions (per-session output ranges show most sessions spanning only 3–7 of the 15 codes).

ii.
```python
    visual_to_idx = {name: idx for idx, name in enumerate(visual_categories)}
...
            output_trial = np.vstack([stim_idx, licking, position_bin, speed_bin]).astype(np.int16, copy=False)
...
        "output_values": [
            visual_categories,
            ["no_lick", "lick"],
            POSITION_OUTPUT_VALUES,
            RUNNING_SPEED_OUTPUT_VALUES,
        ],
```

iii. Step 5 Key Decision 8: "Store all four decoder outputs in a 2D time-varying array. The per-trial visual stimulus label will be repeated across retained timepoints to keep a uniform `(4, T)` output shape." The AI's own sanity check plan included "Stimulus-label check: verify the repeated categorical `visual_stimulus` output matches raw `WallName`", reported as passing.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr` (the imaging-frame number of every lick, fractional) together with `LickTrind` (the trial each lick belongs to). The reference uses `LickFr` alone.

ii.
```python
def build_lick_frame_lookup(record: dict, nfr: int) -> dict[int, np.ndarray]:
    lick_frames = np.asarray(record["LickFr"], dtype=float)
    lick_trial = np.asarray(record["LickTrind"], dtype=float)
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    lick_frames = lick_frames[valid].astype(int)
    lick_trial = lick_trial[valid].astype(int)
    valid = (lick_frames >= 0) & (lick_frames < nfr)
    lick_frames = lick_frames[valid]
    lick_trial = lick_trial[valid]
    lookup: dict[int, np.ndarray] = {}
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
    return lookup
```

iii. Step 5 mapping table: "`beh['LickFr']`, `beh['LickTrind']` → `output[1]` = `licking`; build a binary imaging-frame vector (1 if any lick occurs on that retained frame, else 0); `spk_2_firstLick`, `spk_2_cue`, `lickCount`."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Fractional lick frame numbers are truncated to integers (the frame the lick lands in), NaNs and frames outside `[0, nfr)` are dropped, and licks are grouped per trial. A retained frame is 1 if it appears in that trial's lick-frame set, else 0. Stored as `int16` in row 1. Because only running corridor frames are kept, the overall lick fraction is 0.038 (reference: 0.041), and sessions from unrewarded cohorts are entirely 0.

ii.
```python
            lick_frames = lick_lookup.get(trial_idx, np.empty(0, dtype=np.int32))
            licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
```

iii. Step 5: "Time-varying binary output aligned to neural frames." Planned sanity check: "Lick alignment check: for sampled trials, verify the converted licking vector matches the raw `LickFr` events projected onto the retained imaging frames" — Step 10 Check 2 reports exact agreement for three reconstructed trials.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already indexed in imaging frames, so the binary flag lives on the neural grid; it is sampled at the same `frame_idx` used for the neural columns and therefore has exactly the trial's length.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            ...
            licking = np.isin(frame_idx, lick_frames, assume_unique=False).astype(np.int16)
            output_trial = np.vstack([stim_idx, licking, position_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. Same frame-grid argument as the other streams; the `--show-processing` plots step-plot licking against the same retained-frame axis as the neural raster to demonstrate alignment.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the within-corridor position at each imaging frame, in decimeters (0–40 across the texture, continuing to 60 through the grey space). Identical to the reference.

ii.
```python
            position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Step 4 discrepancy table: "Raw data / code are in decimeters: 60 = 6 m total, 40 = 4 m texture, 20 = 2 m grey", reconciling `Corridor_Length=60` in the data with the paper's "4 m corridor + 2 m grey space".

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped into `[0, 39.999]` decimeters and integer-divided by 10, giving 4 bins of exactly 1 m, stored as `int16` in row 2 with `output_values` `["0-1m", "1-2m", "2-3m", "3-4m"]`. Because only `ft_CorrSpc` frames are kept, values never legitimately exceed 40 dm. Full-dataset distribution [0.249, 0.249, 0.250, 0.252], essentially identical to the reference [0.254, 0.242, 0.247, 0.257].

ii.
```python
def digitize_position(pos_decimeters: np.ndarray) -> np.ndarray:
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```
```python
POSITION_OUTPUT_VALUES = ["0-1m", "1-2m", "2-3m", "3-4m"]
```

iii. Step 5 mapping table: "Convert decimeter positions in the 0-40 textured corridor to four 1 m bins via edges `[0,10,20,30,40]`; paper corridor geometry + code's 60-position / 40-texture convention." Planned sanity check: "Position-bin check: verify position-bin transitions occur at raw `ft_Pos` thresholds of 10, 20 and 30 decimeters", reported as passing.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed 1 m (10 dm) edges at 0/10/20/30/40, i.e. four equal-length spatial bins exactly as the decoder task specifies. The clipping guards both ends (negative positions → bin 0, any residual ≥40 dm → bin 3).

ii.
```python
    pos = np.clip(pos_decimeters, 0.0, 39.999)
    return np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Directly from the Decoder Task specification ("Position in corridor discretized into 4 equal-length, 1-m-long spatial bins"); the resulting near-uniform occupancy (0.249–0.252) was used as the consistency check in Step 9.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, and it is indexed with the same `frame_idx` as the neural columns, so it is bin-for-bin aligned with the neural matrix.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
```

iii. Frame-grid argument as elsewhere; the processing plot draws raw `ft_Pos/10` against `time_since_trial_start` and the discretized bin on the same retained-frame axis, so a misalignment would be visible.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the per-imaging-frame running speed. Identical to the reference.

ii.
```python
        run_speed = np.asarray(record["ft_RunSpeed"][:spec.estimated_nfr], dtype=np.float32)
        for trial_idx, (keep_trial, frame_idx) in enumerate(zip(spec.kept_trial_mask, spec.trial_frame_indices)):
            if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
                all_speeds.append(run_speed[frame_idx])
```

iii. Step 5 mapping table: "`beh['ft_RunSpeed']` on retained frames → `output[3]` = `running_speed_bin`; paper running-speed interpolation; raw `ft_RunSpeed`."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A behavior-only pre-pass pools `ft_RunSpeed` over every retained frame of every retained trial of **all** sessions being converted and takes the 25/50/75% quantiles as global bin edges (13.83, 26.68, 41.84 in raw units). Each frame's speed is then assigned to a bin by `searchsorted(..., side="right")`, clipped to [0, 3], and stored as `int16` in row 3 with `output_values` `["q1","q2","q3","q4"]`. The edges are recorded in metadata. Because the retained frames exclude stationary (`ft_move == 0`) frames, the zero-speed mass that motivated the reference's rank-based split is largely absent. The reference instead computes a rank-based quartile split **per session**.

ii.
```python
    speed_values = np.concatenate(all_speeds).astype(np.float32, copy=False)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
    return speed_edges, np.asarray(frame_dt_medians, dtype=np.float32)
```
```python
def digitize_speed(speed: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```

iii. Step 5 Key Decision 13: "Use global quartiles for running-speed bins: The decoder task specifies 25% bins, so edges must be computed from the full retained dataset, not per session." Step 9 verifies the global distribution is exactly [0.250, 0.250, 0.250, 0.250]. The trajectory (step 310) notes the edges shifted materially once the stalled-trial filter was added, "because the stall-heavy trials are no longer contributing extreme low-speed retained frames, which is what I wanted to see."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three fixed global thresholds (the dataset-wide speed quartile boundaries) applied identically to every session, with ties pushed into the higher bin by `side="right"`. Globally the four bins each hold 25.0% of the bins, but *per session* the occupancy is very unequal — e.g. one session has 97.4% of its frames in bin 0 and 0% in bins 2–3, while others put 70% in bin 3 — because the global edges do not adapt to between-session differences in running speed.

ii.
```python
    bins = np.searchsorted(edges, speed, side="right")
    return np.clip(bins, 0, 3).astype(np.int16)
```
```python
        "running_speed_bin_edges": [float(x) for x in speed_edges],
```

iii. As in 10-b: the AI read "4 bins, each corresponding to 25% of the data" as a statement about the whole dataset and documented the edges in metadata so the discretization is reproducible. It did not discuss the per-session imbalance this creates.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is one value per imaging frame and is indexed with the same `frame_idx` as the neural columns, giving bin-for-bin alignment.

ii.
```python
            neural_trial = spk[:, frame_idx].astype(np.float32, copy=False)
            speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. Frame-grid argument as elsewhere; the processing plot overlays the session speed histogram with the quartile edges and step-plots the resulting bins on the retained-frame axis.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Several guards: (a) behavior frame arrays are truncated to the shortest frame-level array (`estimate_behavior_frame_count` over `ft, ft_trInd, ft_move, ft_CorrSpc, ft_Pos, ft_RunSpeed`) and then per-trial frame indices are clipped again to the true number of imaged frames (`frame_idx[frame_idx < nfr]`), which handles the documented 1–3 frame overhang of behavior over imaging; (b) frames with non-finite `ft_trInd` are dropped; (c) licks with NaN frame/trial or frames outside `[0, nfr)` are dropped; (d) trials with too few frames, implausible duration, or large gaps are dropped, and sessions with <2 trials would be dropped and logged; (e) positions are clipped into range; (f) a retinotopy/spike neuron-count mismatch raises `ValueError`. Removal reasons are recorded in `metadata['removed_trials']`/`['removed_sessions']`. There is no `try/except` around per-session processing, so any unexpected failure (including the `ValueError` above) would abort the whole run rather than skip the session, unlike the reference.

ii.
```python
def estimate_behavior_frame_count(record: dict) -> int:
    frame_keys = ["ft", "ft_trInd", "ft_move", "ft_CorrSpc", "ft_Pos", "ft_RunSpeed"]
    return min(int(np.asarray(record[key]).shape[0]) for key in frame_keys)
```
```python
        for trial_idx, frame_idx in enumerate(spec.trial_frame_indices):
            frame_idx = frame_idx[frame_idx < nfr]
            keep_trial, remove_reason = trial_passes_quality_filters(spec.record, trial_idx, frame_idx, nfr)
            if not keep_trial:
                if frame_idx.size > 0:
                    removed_trials.append((spec.base, int(trial_idx), int(frame_idx.size), remove_reason))
                continue
```
```python
    valid = np.isfinite(lick_frames) & np.isfinite(lick_trial)
    ...
    valid = (lick_frames >= 0) & (lick_frames < nfr)
```

iii. Step 2: "Raw behavior frame arrays are slightly longer than neural recordings; across 99 unique behavior sessions, `len(ft_trInd) - n_spike_frames` is always 1, 2, or 3. This matches the reference code pattern of truncating behavior arrays with `[:nfr]`." Step 4 resolution: "Use the reference-code convention: truncate behavior frame arrays to neural frame count." Step 10 Check 5 documents the stalled-trial edge cases and confirms no session fell below the 2-trial minimum.

## 12-a. What are the most time-consuming steps of the code?

i. Two dominate. (1) Reading and concatenating the per-session `spks` files (hundreds of GB of I/O in total) — the AI names this itself. (2) `select_decoder_neurons`, which materializes several large boolean-mask copies of the full (≈50,000 × 30,000 float32) spike matrix (`spk[:, stim1_mask]`, `spk[:, stim2_mask]`, `spk[:, gray_mask]`) and runs `nanmean`/`nanstd` over them before any neuron subsetting happens. There is also a behavior-only pre-pass that loads every `Beh_*.npy` file and keeps every session's behavior record resident in memory for the whole run. The measured cost was 5.5–10.6 s/session, 635 s total for 89 sessions.

ii.
```python
    spk_item = np.load(path, allow_pickle=True).item()
    spk = np.concatenate([plane for plane in spk_item["spks"]], axis=0)
```
```python
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
```

iii. Step 6: "The current implementation still has to load each neural recording file in full before selecting the compact neuron subset, so full conversion will still be I/O-heavy." Listed speedups: "Behavior-only preparation pass computes trial frame indices and speed quartiles before loading neural data; neural recordings are loaded one session at a time; retinotopy and lick-event processing are done once per session, not once per trial."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) `build_trial_frame_indices` rescans the whole frame index once per trial (`keep & (trial_ids == trial)`, O(ntrials × nframes)); a single `np.argsort`/`np.split` grouping pass would do. (2) `build_lick_frame_lookup` loops over unique trials and re-scans the lick arrays each time. (3) The main per-trial loop rebuilds `np.asarray(record["ft_Pos"])` and `np.asarray(record["ft_RunSpeed"])` on every trial and digitizes them per trial; both could be digitized once per session and then simply indexed (this is what the reference does). (4) The per-area loop in `select_decoder_neurons`. (5) The per-session loop itself is serial despite being embarrassingly parallel. None of these is large relative to the I/O and the d′ pass.

ii.
```python
    for trial in range(ntrials):
        trial_frames = valid_idx[keep & (trial_ids == trial)]
        frame_indices.append(trial_frames.astype(np.int32, copy=False))
```
```python
    for trial in np.unique(lick_trial):
        lookup[int(trial)] = np.unique(lick_frames[lick_trial == trial]).astype(np.int32)
```
```python
            position_bin = digitize_position(np.asarray(spec.record["ft_Pos"], dtype=np.float32)[frame_idx])
            speed_bin = digitize_speed(np.asarray(spec.record["ft_RunSpeed"], dtype=np.float32)[frame_idx], speed_edges)
```

iii. The AI's notes acknowledge the general principle ("Retinotopy and lick-event processing are done once per session, not once per trial") but do not identify these specific loops; it judged the remaining cost acceptable once the sample run projected ~15 min for the full conversion, which the actual 10.6 min run confirmed.

## 12-c. What processing does the code repeat multiple times?

i. (1) `spk[:, stim1_mask]` and `spk[:, stim2_mask]` are each materialized twice — once inside `dprime()` and again in the `corr_neu` expression — so two full large masked copies and two mean computations are redundant. (2) `trial_passes_quality_filters` is evaluated for every trial twice: once in `prepare_session_specs` (using the behavior-derived frame count) and again in the main loop (using the true neural frame count). (3) `np.asarray(record["ft_Pos"])`, `np.asarray(record["ft_RunSpeed"])` and `np.asarray(record["ft"])` are re-materialized on every trial. (4) `load_experiment_type_map` re-derives bases that `select_representative_sessions` derives again from the behavior keys. (5) `spec.kept_trial_mask` is computed in the pre-pass but the main loop re-derives the kept set from scratch and never reads it.

ii.
```python
    dp = dprime(spk[:, stim1_mask], spk[:, stim2_mask])
    corr_neu = (
        (np.mean(spk[:, stim1_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
        | (np.mean(spk[:, stim2_mask], axis=1) > np.mean(spk[:, gray_mask], axis=1))
    )
```
```python
            if keep_trial and trial_passes_quality_filters(record, trial_idx, frame_idx, spec.estimated_nfr)[0]:
```
```python
        spec.kept_trial_mask = np.array(
            [len(frame_idx) >= MIN_TRIAL_TIMEPOINTS for frame_idx in spec.trial_frame_indices],
            dtype=bool,
        )
```

iii. The AI does not document any of this. Its stated rationale for the duplicated filter evaluation is consistency: "I'm applying that in both the speed-bin estimation and the final export so the dataset statistics stay internally consistent" (trajectory step 275) — i.e. the repetition is deliberate for the quality filter, so that the quartile edges are measured on exactly the frames that are exported.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) `load_experiment_type_map` and the `experiment_types` / `source_behavior_keys` bookkeeping are written to metadata and never used by the decoder. (2) `session_key_score` computes `stim_id` finiteness and unique-wall counts for every behavior key just to pick a representative. (3) Per-session `median_frame_dt_ms` values are computed for all 89 sessions but only their median survives into metadata. (4) `spec.kept_trial_mask` is computed and never used in the export path. (5) `BRAIN_REGION_NAMES` reserves an `other` class that ends up with 0 neurons. (6) `spk.astype(np.float32, copy=False)` is a no-op since `spks` is already float32. (7) The `removed_trials` list is accumulated and truncated to 1,000 entries for metadata. (8) The whole `select_decoder_neurons` d′ computation is expensive and is used only to choose a neuron subset — and the subset it chooses is itself a deviation from the reference (see 2-c); the same size reduction could have been obtained for free by storing `float16` like the reference (halving the pickle without discarding any neuron). None of these is a correctness problem; the notes describe the metadata fields as deliberate provenance records.

ii.
```python
        "source_recording_bases": [spec.base for spec in specs],
        "source_behavior_keys": [spec.key for spec in specs],
        "source_experiment_types": {
            spec.base: sorted(spec.experiment_types) for spec in specs
        },
        "removed_sessions": removed_sessions,
        "removed_trials": removed_trials[:1000],
        "removed_trials_truncated": len(removed_trials) > 1000,
```
```python
BRAIN_REGION_NAMES = ["V1", "mHV", "lHV", "aHV", "other"]
```
```python
    return spk.astype(np.float32, copy=False)
```

iii. Step 5 Key Decision 11: "Represent brain regions with five labels: `V1`, `mHV`, `lHV`, `aHV`, `other`. The current selective export only keeps neurons from the first four groups, but the metadata schema still reserves `other` for completeness." The removal bookkeeping is justified in Step 6/10 as making the curation auditable: removal reasons and counts are reported so that "every removal" is documented.
