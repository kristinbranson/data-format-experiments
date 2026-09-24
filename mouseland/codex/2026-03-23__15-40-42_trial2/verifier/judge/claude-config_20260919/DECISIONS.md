# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI treats `data/beh/Imaging_Exp_info.npy` as the master index. It iterates over every experiment type and every entry, builds the session id `<mname>_<datexp>_<blk>`, and the behavior key (session id plus `_<stimtype>` when the entry has a `stimtype` field). Sessions are grouped by base session id and one canonical candidate is chosen per recording, giving 89 sessions / 19 mice / 38,110 trials. For each session it then loads three things: the behavior (`beh/Beh_<exp_type>.npy`, subscripted by the behavior key), the deconvolved traces via the reference helper `utils.load_spk()` on `spk/<session>_neural_data.npy`, and the retinotopy `retinotopy/<mouse>_<date>_trans.npz` for `iarea`. Behavior files are loaded from disk on every call to `load_behavior()`, so each multi-hundred-MB `Beh_*.npy` is re-read once per session it contains, plus once more in `collect_stimulus_values()` and once more in `compute_time_bin_ms()` (and again in `sample_candidate_score()` in `--sample` mode).

ii.
```python
def build_session_catalog() -> list[SessionCandidate]:
    exp_info = np.load(BEH_ROOT / "Imaging_Exp_info.npy", allow_pickle=True).item()
    grouped: dict[str, list[SessionCandidate]] = defaultdict(list)
    for exp_type, db_list in exp_info.items():
        for db in db_list:
            session_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            beh_key = session_id
            if "stimtype" in db:
                beh_key = f"{beh_key}_{db['stimtype']}"
            grouped[session_id].append(SessionCandidate(session_id=session_id, exp_type=exp_type,
                                                        beh_key=beh_key, db=db))
    return [choose_canonical(grouped[sid]) for sid in sorted(grouped, key=session_sort_key)]


def load_behavior(exp_type: str, beh_key: str) -> dict:
    beh = np.load(BEH_ROOT / f"Beh_{exp_type}.npy", allow_pickle=True).item()
    return beh[beh_key]
```
```python
    beh = load_behavior(candidate.exp_type, candidate.beh_key)
    spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
    ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
```

iii. CONVERSION_NOTES Step 1/2 records that `Imaging_Exp_info.npy` is the central index, that `load_spk()` is the reference loader that concatenates the per-plane `['spks']` arrays, and that `iarea` in the retinotopy files gives the visual area of each neuron; the AI states it deliberately reuses the reference loaders from `/app/code/utils.py`. It documents that the index has 142 entries over 23 experiment types but only 89 unique imaging recordings, and that only imaging sessions with matched neural and retinotopy files are used (excluding the behavior-only `example_bef_and_aft_learning_behavior.npy`).

## 1-b. How are the data split into subjects?

i. The subject is the `mname` field of the index entry, carried on every session record. `subjects` is the sorted set of unique mouse names in the processed catalog and `subject_idx` is each session's index into that list. Result: 19 subjects, 1–8 sessions each, matching the paper's "89 recordings in 19 mice".

ii.
```python
    subjects_all = sorted({cand.db["mname"] for cand in catalog})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects_all)}
```
```python
        "subject_idx": np.asarray([subject_to_idx[s.subject] for s in processed_sessions], dtype=np.int64),
```

iii. Step 4 consistency table: "`Imaging_Exp_info.npy` contains repeated mouse IDs across experiments / 19 unique `mname` values / paper says 19 mice → Consistent. Use `mname` as subject ID." No derivation is needed because the index names the mouse.

## 1-c. How are the data split into sessions?

i. A session is one mouse / one date / one block, `<mname>_<datexp>_<blk>`, which is also the name of the spike file. The same recording is listed under several experiment types (142 entries → 89 recordings), so entries are grouped by base session id and `choose_canonical()` picks one: entries **without** a `stimtype` field sort first, then by experiment type, then by behavior key. Sessions are ordered by (subject, date, block). For `test3` swap recordings, which only exist with `_swap1` / `_swap2` behavior keys, one of the two keys is used.

ii.
```python
def choose_canonical(candidates: list[SessionCandidate]) -> SessionCandidate:
    def key_fn(c: SessionCandidate) -> tuple[int, str, str]:
        has_stimtype = 1 if "stimtype" in c.db else 0
        return has_stimtype, c.exp_type, c.beh_key
    return sorted(candidates, key=key_fn)[0]
```

iii. Key Decision 1: "Deduplicate 142 behavior entries down to 89 unique neural sessions ... Conversion will use one canonical copy per base session ID, preferring a non-`stimtype` entry when present. For `test3` swap sessions, either suffix copy is sufficient because `WallName` contains the true trial label and the framewise data are duplicated." The AI verified in Step 2 that the two swap keys differ only in the masking of `stim_id`.

## 1-d. Are the data correctly split into trials?

i. Trials are the `ntrials` trials the behavior declares. For each trial the AI takes the half-open frame range `int(StartFr[trial])` to `int(GrayFr[trial])` and then keeps only the frames with `ft_move > 0` ("running-only"). Trials with zero running frames are skipped; nothing else is dropped, so all 38,110 trials survive. Trials keep their own length (mean T = 22.3 frames, min 12, max 179) with no padding.

Two properties of this window differ from what the notes claim. (1) `StartFr` is fractional (e.g. 12.85) and is truncated with `astype(int)`, so the window opens 1–2 frames *before* corridor entry; those leading frames belong to the previous trial's grey space. Measured on `LZ13_2024_05_15_1`: 466 of 10,055 kept frames (4.6%, ≈1.02 frames per trial) have `ft_CorrSpc == 0` and positions of 57.9–60.0 dm, i.e. 5.8–6.0 m. (2) No texture mask (`ft_CorrSpc`, or `ft_Pos < 40`) is ever applied, despite the notes saying the window "keeps exactly the 0–4 m texture interval". The running-only filter also removes ~11% of in-corridor frames (10,055 kept vs. 11,273 in-corridor frames for that session), so the frames of a trial are not temporally contiguous.

ii.
```python
    for trial_idx in range(int(beh["ntrials"])):
        frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
        frames = frames[ft_move[frames] > 0]
        if frames.size == 0:
            continue
```
```python
    start_fr = np.asarray(beh["StartFr"], dtype=int)
    gray_fr = np.asarray(beh["GrayFr"], dtype=int)
```

iii. Key Decision 2: "Use the texture segment only (`StartFr:GrayFr`): The decoder output requires exactly four 1 m position bins. Ending trials at grey-space entry also matches the paper's 0–4 m texture-area analyses." Key Decision 3: "Keep running frames only inside the texture segment: This matches the reference paper/code rule that analyses use running timepoints and removes extremely long paused trials. Actual elapsed time is still preserved through the continuous time inputs." Step 3 quotes the paper: "We only considered timepoints during running for analysis."

## 1-e. How are trials filtered based on quality controls?

i. The only trial-level filter is "at least one running frame between `StartFr` and `GrayFr`". No length-outlier filter, no per-session minimum trial count, no neuron-count gate. In practice zero trials are lost: the converted dataset has all 38,110 trials, exactly the raw count the AI measured in Step 2. Extremely long, mostly-stationary traversals are shortened rather than dropped, because their stationary frames fail the `ft_move > 0` test; the longest surviving trial is 179 frames (56 s).

ii.
```python
        frames = frames[ft_move[frames] > 0]
        if frames.size == 0:
            continue
```

iii. Key Decision 3 explicitly claims the running-only filter "removes extremely long paused trials", so no separate outlier rule was thought necessary. Step 9's consistency table reports "Trials (total) ... 38,110 / 38,110 / Yes" as evidence that no trial was lost in conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session_id>_neural_data.npy`, loaded through the reference helper `utils.load_spk()`, which concatenates the per-imaging-plane arrays into one neurons × frames matrix. The visual area of each neuron comes from `iarea` in `retinotopy/<mouse>_<date>_trans.npz`, mapped to `V1` (8), `mHV` (0,1,2,9), `lHV` (5,6), `aHV` (3,4), else `other`.

ii.
```python
    spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
    ret = np.load(RET_ROOT / f"{candidate.db['mname']}_{candidate.db['datexp']}_trans.npz", allow_pickle=True)
    _, region_idx_all = area_labels_from_iarea(ret["iarea"])
```
```python
def area_labels_from_iarea(iarea: np.ndarray) -> tuple[list[str], np.ndarray]:
    brain_regions = ["V1", "mHV", "lHV", "aHV", "other"]
    out = np.full(iarea.shape, 4, dtype=np.int64)
    out[iarea == 8] = 0
    out[np.isin(iarea, [0, 1, 2, 9])] = 1
    out[np.isin(iarea, [5, 6])] = 2
    out[np.isin(iarea, [3, 4])] = 3
    return brain_regions, out
```

iii. Step 1: "`load_spk(db, root='')` loads session neural data and concatenates the list in `['spks']` into a neuron x frame matrix"; "The reference code does **not** compute delta-F-over-F inside this repository ... the saved neural signal is already the processed neural activity used downstream." Step 5 mapping table: retinotopy `iarea` → `brain_region_idx` via `load_retino`, `neu_area_ID`.

## 2-b. How is the `neural` data processed?

i. No processing at all: the selected neurons' columns for a trial's retained frames are sliced out of `spks` and stored as `float32`. No dF/F, no deconvolution, no smoothing, no normalization, no padding; trials keep their native length.

ii.
```python
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
```
```python
            neural_session.append(neural_trial.astype(np.float32, copy=False))
```

iii. Step 1/Step 3: all paper analyses are based on deconvolved fluorescence traces, which is exactly what `spk/*_neural_data.npy` stores, so nothing further is needed. Step 12 notes the format verification reported no errors or warnings on values.

## 2-c. How is the `neural` data filtered based on quality controls?

i. This is the AI's largest departure. On top of requiring a retinotopic visual-area label (`iarea` in V1/mHV/lHV/aHV), the AI applies the paper's *analysis-time* selectivity criteria as a dataset curation filter:
- **stimulus-selective pool**: for each session pick a "reference pair" of walls (rewarded-like `stim_id == 2` vs non-rewarded-like `stim_id == 0`, with fallbacks); compute d′ between running corridor frames of the two walls; keep visual-area neurons with `|d′| >= 0.3`;
- **reward-prediction pool**: aHV neurons only, requiring cue-frame stimulus d′ > 0.3 **and** late-vs-early-cue d′ at or above the session's 95th percentile within aHV (trial means over running frames with `ft_Pos` in [5, 40] dm);
- **fallback**: if fewer than 64 neurons survive, keep the top-|d′| neurons (at least 64, up to 1% of candidates).

This keeps 357,328 of 4,691,034 neurons (7.6%), a mean of 4,015 per session instead of 52,708. Every surviving neuron is in a visual area, so the `other` region ends up with 0 neurons even though it is listed in `brain_regions`.

ii.
```python
    dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
    stim_selective = visual_mask & np.isfinite(dp) & (np.abs(dp) >= DP_THRESHOLD)
    ...
    reward_pred = (ahv_mask & np.isfinite(reward_dp) & np.isfinite(dp_sound)
                   & (dp_sound > DP_THRESHOLD) & (reward_dp >= reward_dp_thr))
    selected = stim_selective | reward_pred

    if selected.sum() < MIN_NEURONS_FALLBACK:
        candidate = visual_mask & np.isfinite(dp)
        ...
        nkeep = min(max(MIN_NEURONS_FALLBACK, int(0.01 * len(order))), len(order))
        selected = np.zeros(spk.shape[0], dtype=bool)
        selected[order[:nkeep]] = True
```

iii. Key Decision 4: "Curate neurons using paper-defined task relevance ... This follows the two main neuron-selection motifs in the paper and keeps the conversion computationally tractable." Step 3 cites the paper's "d′ ≥ 0.3 or d′ ≤ −0.3" and "top 5% selective neurons each". Step 10 iteration 1 tightened the reward-prediction rule to the aHV 95th-percentile form to "better match the reference code". Step 9 concedes the mismatch with the raw totals: "Partial: curated count intentionally differs from raw total".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The alignment event is corridor entry: each trial's frame window begins at `int(StartFr[trial])` and runs to `int(GrayFr[trial])`, and every input/output stream is sliced with the same `frames` index array, so all streams are aligned frame-for-frame with the neural matrix. Trials are variable length, nothing is padded, `off_start = 0.0`, `off_end = None`.

Two caveats found in the code: (1) the `int()` truncation of the fractional `StartFr` means the window starts ~1 frame before actual corridor entry (verified on `LZ13_2024_05_15_1`: the first 1–2 frames of each trial are at 5.8–6.0 m in the preceding grey space); (2) because non-running frames are dropped, consecutive columns of a trial are not consecutive in real time.

ii.
```python
        frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
        frames = frames[ft_move[frames] > 0]
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        ...
        licking = np.isin(frames, lick_frames_trial).astype(np.int64)
        pos = ft_pos[frames]
        raw_speed = ft_speed[frames].astype(np.float32)
```
```python
            "temporal_alignment_event": "corridor entry (trial start / StartFr)",
            "off_start": 0.0,
            "off_end": None,
            "trial_end_event": "entry into gray space (GrayFr)",
```

iii. The instructions specify alignment to trial start (corridor entry); the AI records this in metadata and in the README ("Temporal alignment: corridor entry (`StartFr`)"). Step 12 states "the processing plots and time-variable spot-checks show no temporal lag between retained neural frames and decoder targets", i.e. the AI checked stream-to-stream alignment but not the absolute position of the window relative to corridor entry.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One imaging frame is one bin. The bin size written to metadata is measured from the data: for each session the median positive `diff(ft)` converted to ms, then the median across sessions (≈315 ms, i.e. the 3.17 Hz imaging rate). The same nominal bin size is declared for all trials and sessions, although (because non-running frames are removed) successive bins within a trial are not always 315 ms apart in wall-clock time.

ii.
```python
def compute_time_bin_ms(catalog: list[SessionCandidate]) -> float:
    medians = []
    for cand in catalog:
        ft = np.asarray(load_behavior(cand.exp_type, cand.beh_key)["ft"], dtype=float)
        d = np.diff(ft)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            medians.append(float(np.median(d) * SECONDS_PER_DAY * 1000.0))
    return float(np.median(medians))
```
```python
            "time_bin_size": time_bin_ms,
```

iii. Step 1 notes the reference notebook states the imaging rate is 3.17 Hz; the AI chose to derive the value empirically from `ft` rather than hard-code it, and reports the sample/full statistics accordingly. No temporal rebinning is mentioned anywhere as required by the reference.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. As implemented: `SoundFr` (the imaging frame of the cue, truncated to int), the trial's retained `frames` array, and `frame_dt` — the session's median inter-frame interval in seconds derived from `ft`. The cue is located by `np.searchsorted(frames, SoundFr)`, i.e. as an index into the retained-frame list, not as a timestamp. Note that the Step 5 mapping table documents a different source: `beh['SoundTime']` and per-frame `beh['ft']`.

ii.
```python
    ft = np.asarray(beh["ft"][:nfr], dtype=float)
    dft = np.diff(ft)
    dft = dft[np.isfinite(dft) & (dft > 0)]
    frame_dt = float(np.median(dft) * SECONDS_PER_DAY)
    sound_fr = np.asarray(beh["SoundFr"], dtype=int)
    ...
        cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
```

iii. Step 5 mapping: "`beh['SoundTime']`, `beh['ft']`, `StartFr:GrayFr` running frames → `input[0]` = `time_to_sound_cue_sec`: for each retained frame, `(SoundTime[trial] - ft[frame]) * 86400`; can be positive before cue and negative after cue." Step 1 identifies `SoundFr`/`SoundDelPos` as the reference's cue anchors.

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. `t_to_cue = (cue_idx - retained_idx) * frame_dt`, where `retained_idx = arange(n_retained_frames)`. The sign convention matches the reference (positive before the cue, negative after). Because the clock is "retained-frame index × median frame interval" rather than the real frame timestamps, the value is (a) quantized to whole frames (the cue snaps to the first retained frame at or after `SoundFr`) and (b) blind to every frame that was dropped for not running. Measured on `LZ13_2024_05_15_1`, the equivalent error in the elapsed-time axis is a median of 0.012 s per trial but exceeds 1 s in 9.8% of trials and reaches 37.2 s in the worst trial. This contradicts the AI's own Key Decision 7.

ii.
```python
        retained_idx = np.arange(frames.size, dtype=np.float32)
        cue_idx = float(np.searchsorted(frames, sound_fr[trial_idx], side="left"))
        t_to_cue = ((cue_idx - retained_idx) * frame_dt).astype(np.float32)
```

iii. Key Decision 7: "Use actual frame times in seconds, not just frame index: Because non-running frames are dropped, elapsed time inputs must come from `ft` timestamps rather than assuming contiguous native-frame spacing." Key Decision 3 likewise asserts "Actual elapsed time is still preserved through the continuous time inputs." The Step 10 `np.allclose()` sanity check re-implements the same index-based formula, so it confirms self-consistency rather than agreement with `ft`.

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is built on exactly the same `frames` array used to slice the neural matrix, one value per neural column, and `finalize_io()` re-checks that the input array's time dimension equals the neural trial's T.

ii.
```python
            input_arr = np.vstack([input_raw["time_to_sound_cue_sec"], ...]).astype(np.float32)
            if input_arr.shape[1] != T or output_arr.shape[1] != T:
                raise ValueError(f"Time dimension mismatch in session {session.session_id}.")
```

iii. All streams in this dataset are indexed by imaging frame, so slicing every stream with the same frame index array guarantees alignment; the `--show-processing` plots overlay `time_to_cue`, `time_since_start`, `position_bin` and `licking` on the retained-frame axis to make this visible.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The `datexp` field of the index entry (the recording date) together with `mname`: for each mouse the earliest recording date is found and each session's value is the number of **calendar days** since that date. Values range 0–92.

ii.
```python
def compute_day_offsets(catalog: list[SessionCandidate]) -> dict[str, float]:
    by_subject: dict[str, list[datetime]] = defaultdict(list)
    session_dates: dict[str, datetime] = {}
    for cand in catalog:
        date = datetime.strptime(cand.db["datexp"], "%Y_%m_%d")
        by_subject[cand.db["mname"]].append(date)
        session_dates[cand.session_id] = date
    first_dates = {subject: min(dates) for subject, dates in by_subject.items()}
    return {cand.session_id: float((session_dates[cand.session_id] - first_dates[cand.db["mname"]]).days)
            for cand in catalog}
```

iii. Key Decision 8: "Use day-since-first-recording as the training-day proxy: The paper does not provide a complete per-session training-day label, while calendar date is available for every recording. The small subset of sessions with `db['days']` can be used later as a sanity check."

## 4-b. What processing is involved in computing `input` *Day of training*?

i. The per-session scalar is broadcast to every retained frame of every trial of that session as a `float32` row of the input matrix. The offsets are always computed over the **full** catalog (`full_catalog`), so `--sample` and `--full` agree.

ii.
```python
    full_catalog = build_session_catalog()
    day_offsets = compute_day_offsets(full_catalog)
```
```python
        day = np.full(frames.shape, session_day, dtype=np.float32)
```

iii. Key Decision 6: "Represent all decoder inputs as time-varying arrays: Even trial-constant variables (`day_of_training`, `reward_available`) will be repeated across timepoints so the dataset is uniform and easy for `train_decoder.py`."

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. As implemented: only the position of each retained frame within the trial (`arange(frames.size)`) and the session's median frame interval `frame_dt` (itself derived from `ft`). `StartFr` enters only through the definition of the window. The Step 5 mapping documents a different source: `(ft[frame] - ft[StartFr[trial]]) * 86400`.

ii.
```python
        retained_idx = np.arange(frames.size, dtype=np.float32)
        t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. Step 5 mapping: "`beh['ft']`, `StartFr` → `input[2]` = `time_since_trial_start_sec`: `(ft[frame] - ft[StartFr[trial]]) * 86400` for each retained frame. Continuous time-varying input in seconds."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `t_since = retained_frame_index * frame_dt`, so it always starts at exactly 0.0 on the first retained frame and increases by a constant 315 ms per bin. Two consequences: the zero point is the truncated `int(StartFr)` frame, which is typically 1 frame before corridor entry (in the grey space at 5.8–6.0 m); and pauses are invisible, so the variable is really "running time elapsed", not elapsed time. Per-trial discrepancy versus true `ft`-based elapsed time on `LZ13_2024_05_15_1`: median 0.012 s, 90th percentile 0.96 s, max 37.2 s; 9.8% of trials off by more than 1 s. The reported range is [0.0, 56.0] s vs. the reference's [0.0, 74.8] s.

ii.
```python
        t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. Same as 3-b: Key Decision 7 says timestamps must come from `ft` because non-running frames are dropped; the implementation does not do this, and the Step 10 spot-check reproduces the implemented formula rather than the documented one.

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same `frames` window as the neural data, one value per neural column, shape-checked in `finalize_io()`.

ii.
```python
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        retained_idx = np.arange(frames.size, dtype=np.float32)
        t_since = (retained_idx * frame_dt).astype(np.float32)
```

iii. Every stream is sliced with the same frame index array, which the AI verified with `np.allclose()` spot-checks on sessions 0, 34 and 88 and with the `--show-processing` overlay plots.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `beh['isRew']`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
    is_rew = np.asarray(beh["isRew"]).astype(np.float32)
    ...
        reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Step 5 mapping: "`beh['isRew']` → `input[3]` = `reward_available`: trialwise 0/1 repeated across retained frames. Uses rewarded-corridor identity even in unsupervised sessions, matching reference use of `isRew`."

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and broadcast across the trial's frames. As expected, it is identically 0 for the unsupervised/naive sessions and 0/1 for the task sessions (the per-session ranges in the verification log show `[0,0]` for roughly two-thirds of sessions).

ii.
```python
        reward = np.full(frames.shape, is_rew[trial_idx], dtype=np.float32)
```

iii. Key Decision 6 (all inputs represented as time-varying arrays). No further rationale is given because no processing is required.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `beh['WallName']`, the texture name of each trial. `TrialStim`/`stim_id` are explicitly avoided because `stim_id` is NaN-masked in swap sessions.

ii.
```python
    wall_name = np.asarray(beh["WallName"]).astype(str)
    ...
        stim_val = np.full(frames.shape, stimulus_to_idx[wall_name[trial_idx]], dtype=np.int64)
```

iii. Key Decision 5: "Use exact `WallName` strings as stimulus labels: This avoids ambiguity from `stim_id` NaNs in swap sessions and preserves non-leaf/circle texture pairs (`rock*`, `wood*`)."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The label vocabulary is the sorted set of **all distinct wall-name strings** in the dataset, collected in a preliminary pass over every behavior file. That yields **15** categories (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`) rather than the four base textures. The per-trial index is broadcast across the trial's frames. Because the vocabulary is global, any one session contains only 2–4 of the 15 labels, and chance level for the decoder is reported as 1/15 = 0.067.

ii.
```python
def collect_stimulus_values(catalog: list[SessionCandidate]) -> list[str]:
    names = set()
    for cand in catalog:
        beh = load_behavior(cand.exp_type, cand.beh_key)
        names.update(map(str, np.asarray(beh["WallName"]).tolist()))
    return sorted(names)
```
```python
    stimulus_to_idx = {name: idx for idx, name in enumerate(stimulus_values)}
```

iii. Key Decision 5 (above) plus the Step 4 resolution on swaps: "Code keeps `leaf1_swap1` and `leaf1_swap2` as separate IDs/files ... Resolve by preserving raw swap subtype identity in converted trial labels; pooling, if needed, should happen only in downstream statistics."

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `beh['LickFr']` (the fractional imaging-frame number of each lick) together with `beh['LickTrind']` (the trial each lick belongs to), which is used to restrict the licks considered for a trial.

ii.
```python
    lick_fr = np.asarray(beh["LickFr"], dtype=float)
    lick_tr = np.asarray(beh["LickTrind"], dtype=float)
    ...
        lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
        lick_frames_trial = lick_frames_trial.astype(int)
```

iii. Step 5 mapping: "`beh['LickFr']`, `beh['LickTrind']` → `output[1]` = `licking`: binary per retained frame: 1 if one or more licks occur on that imaging frame, else 0."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are truncated to integers (the frame the lick falls in), NaNs are discarded, and the output is `np.isin(frames, lick_frames_trial)` — 1 if at least one lick of that trial lands on the retained frame, else 0. Overall 3.7% of bins are licks (reference: 4.1%), and 62 of 89 sessions have no licks at all, consistent with the unsupervised/naive animals. Because only running frames are kept, licks emitted while the animal is stationary are lost: on the task session `VR2_2021_04_06_1`, only 53% of in-corridor lick frames are running frames.

ii.
```python
        licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```
```python
        "output_values": [ ..., ["no_lick", "lick"], ... ]
```

iii. Step 5 mapping (binary per retained frame). The running-only restriction is inherited from Key Decision 3; the AI's Step 9 table calls the resulting distribution "Sparse lick events after running-only texture-frame filtering" and marks it consistent.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` is already expressed in imaging frames, so the binary flag is built directly on the trial's `frames` array — the same index array used for the neural columns — and has that trial's length.

ii.
```python
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        licking = np.isin(frames, lick_frames_trial).astype(np.int64)
```

iii. Frame-indexed streams need no interpolation; the AI's `np.allclose()` spot-checks included the licking row for three (session, trial) pairs.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `beh['ft_Pos']`, the within-corridor position of each imaging frame in decimeters (0–40 across the texture, 40–60 through the grey space), truncated to the number of imaged frames and sampled at the trial's retained frames.

ii.
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
    ...
        pos = ft_pos[frames]
```

iii. Step 4 resolved the units: "`Corridor_Length=60`, `Texture_Length=40`, `Gray_Space_length=20` ... Consistent after unit conversion: raw data are in decimeters over a 6 m total corridor."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. `floor(ft_Pos / 10)` clipped to [0, 3] — identical in form to the reference's `clip(ft_Pos // 10, 0, 3)`. Stored as int64 and broadcast nowhere (it is genuinely time-varying). The resulting distribution is near-uniform (0.249 / 0.248 / 0.249 / 0.254).

ii.
```python
        pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```
```python
            ["0-1m", "1-2m", "2-3m", "3-4m"],
```

iii. Step 5 mapping: "Discretize `0–40 dm` into 4 equal bins: `[0,10), [10,20), [20,30), [30,40]` ... Exactly matches the requested 4 one-meter bins", citing the paper's 0–4 m texture area.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed 1-m (10-dm) thresholds at 10, 20 and 30 dm, giving the four equal-length bins the task specifies. The `clip(..., 0, 3)` also silently assigns any frame beyond 40 dm to bin 3. Because the trial window opens before corridor entry (see 1-d), the grey-space frames at 57.9–60.0 dm that leak into each trial are labelled "3-4 m": 4.6% of kept frames in the session I checked (≈1 frame per trial, always the first).

ii.
```python
        pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. The thresholds come straight from the Decoder Task specification ("4 equal-length, 1-m-long spatial bins"); the AI's assumption that only texture frames reach this code is stated in Key Decision 2 ("`GrayFr` end keeps exactly the 0–4 m texture interval").

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, so it is sampled at the trial's `frames` array — the same indices as the neural columns — giving one label per neural bin.

ii.
```python
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        pos = ft_pos[frames]
        pos_bin = np.clip(np.floor(pos / 10.0).astype(np.int64), 0, 3)
```

iii. Frame-indexed alignment; the `--show-processing` plot overlays `position_bin` with the other per-frame streams for an example trial, and the raw-data `np.allclose()` checks cover the position row.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `beh['ft_RunSpeed']`, the per-frame running speed, truncated to the imaged frames and sampled at the trial's retained frames.

ii.
```python
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
    ...
        raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Step 5 mapping: "`beh['ft_RunSpeed']` on retained frames → `output[3]` = `running_speed_bin`."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. Raw speeds of every retained frame of every processed session are accumulated, and a single **global** set of quartile edges is computed with `np.quantile(..., [0, .25, .5, .75, 1])` after all sessions have been processed; ties/degenerate edges are nudged with `np.nextafter`. Each frame's bin is then `searchsorted(edges[1:-1], speed, side='right')`. The edges for the full run were [-19.17, 12.51, 25.51, 41.21, 161.08]. Because only running frames are kept, the stationary population that would otherwise pile up at zero is largely excluded before the quantiles are taken.

ii.
```python
    all_speed_concat = np.concatenate(all_speed_values).astype(np.float32)
    speed_edges = np.quantile(all_speed_concat, [0.0, 0.25, 0.5, 0.75, 1.0]).astype(np.float32)
    for i in range(1, len(speed_edges)):
        if speed_edges[i] <= speed_edges[i - 1]:
            speed_edges[i] = np.nextafter(speed_edges[i - 1], np.float32(np.inf))
```
```python
            speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```

iii. Step 5 mapping: "Compute global quartile edges over all retained running frames in all sessions; assign bin 0–3 ... Quartiles computed on the included running-only texture frames", justified by the Decoder Task requirement of "4 bins, each corresponding to 25% of the data".

## 10-c. How is `output` *Running speed* thresholded into categories?

i. Three thresholds — the 25th, 50th and 75th percentiles of the pooled retained-frame speeds — applied with `side='right'`. Globally this is exactly 25.0% per bin, and the bin names carry the numeric edges (`"-19.17-12.51"`, …). Per session, however, the distribution is far from uniform: the verification log shows sessions with 95.8% of frames in bin 0 and 0.0% in bin 3, and others with 69% in bin 3.

ii.
```python
    thresholds = speed_edges[1:-1]
    speed_bin = np.searchsorted(thresholds, output_raw["running_speed_raw"], side="right").astype(np.int64)
```
```python
def make_speed_bin_names(speed_edges: np.ndarray) -> list[str]:
    names = []
    for i in range(4):
        names.append(f"{speed_edges[i]:.2f}-{speed_edges[i + 1]:.2f}")
    return names
```

iii. Step 9 consistency table: "`running_speed_bin` distribution ... `0.250/0.250/0.250/0.250` by construction — Yes." The AI treats "25% of the data" as a dataset-level requirement and stores the edges in metadata (`speed_bin_edges`) so the mapping is reproducible.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. One speed per imaging frame, sampled at the trial's `frames` array (the same indices as the neural columns); binning is applied elementwise afterwards, so alignment is unchanged.

ii.
```python
        neural = spk_sel[:, frames].astype(np.float32, copy=False)
        raw_speed = ft_speed[frames].astype(np.float32)
```

iii. Frame-indexed alignment, as for all other streams; the `--show-processing` plot draws the raw speed trace of an example trial with the quartile edges overlaid so the discretization can be checked visually.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled: (a) the behavior can run past the imaging, so every framewise stream is truncated to `nfr = spk.shape[1]` (`ft`, `ft_move`, `ft_Pos`, `ft_RunSpeed`, `ft_WallID`, `ft_CorrSpc`); (b) NaN lick frames are filtered with `np.isfinite`; (c) trials with no running frames are skipped; (d) sessions where no finite aHV reward d′ exists get an infinite threshold so that no neuron is selected by that rule; (e) `get_reference_pair()` has a chain of fallbacks (stim_id → rewarded/non-rewarded wall names → "…1" walls → any two walls) for sessions where the canonical pair cannot be identified; (f) the ≥64-neuron fallback in the selection. Not handled: the per-trial window `arange(int(StartFr), int(GrayFr))` is never clipped to `nfr`, so a trial whose grey-space entry fell after the last imaged frame would raise an `IndexError`; and there is no try/except around session processing, so one bad session would abort the whole run. Neither case occurs in this dataset (max `GrayFr` < `nfr` in every file I checked).

ii.
```python
    nfr = spk_sel.shape[1]
    ft = np.asarray(beh["ft"][:nfr], dtype=float)
    ft_move = np.asarray(beh["ft_move"][:nfr], dtype=float)
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=float)
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=float)
```
```python
        lick_frames_trial = lick_fr[(lick_tr == trial_idx) & np.isfinite(lick_fr)]
```
```python
    reward_dp_thr = np.inf
    ahv_mask = region_idx_all == 3
    if np.isfinite(reward_dp[ahv_mask]).any():
        reward_dp_thr = float(np.nanpercentile(reward_dp[ahv_mask], 95))
```

iii. Step 10 Check 5: "Sessions with no finite aHV reward `d′` values are now handled explicitly by assigning an infinite reward threshold, preventing accidental over-selection. The sample-selection guard from earlier steps still prevents degenerate all-zero reward/licking sample runs." The `[:nfr]` truncation mirrors the reference code's `beh[...][:nfr]` idiom noted in Step 1.

## 12-a. What are the most time-consuming steps of the code?

i. Reading the 405 GB of `spk/*_neural_data.npy` files dominates: the full conversion took 501 s for 89 sessions, and the per-session times printed in `conversion_full_out.txt` (2.1–12.7 s) track spike-file size almost exactly (~2.1 s/GB). Secondary costs, all of which are consequences of the neuron-selection step and are absent from the reference: the `d′` computations, which form full copies `spk[:, stim1_fr]` and `spk[:, stim2_fr]` of a ~50,000 × 20,000 matrix and run `nanmean`/`nanstd` over them; and the per-reward-trial `nanmean` loop over all neurons. On top of that, `Beh_*.npy` files (6.6 GB total) are read from disk three times over: once in `collect_stimulus_values()`, once in `compute_time_bin_ms()`, and once per session in `process_session()`.

ii.
```python
    spk = utils.load_spk(candidate.db, root=str(SPK_ROOT))
```
```python
    dp = safe_dprime(spk[:, stim1_fr], spk[:, stim2_fr])
```
```python
            for trial_idx in np.flatnonzero(valid_reward_trials):
                frames = np.arange(start_fr[trial_idx], gray_fr_trial[trial_idx], dtype=int)
                frames = frames[running[frames] & (ft_pos[frames] >= 5.0) & (ft_pos[frames] <= TEXTURE_LENGTH_DM)]
                if frames.size:
                    trial_means[:, trial_idx] = np.nanmean(spk[:, frames], axis=1)
```

iii. Step 6: "Session processing loads each large spike file only once, computes the neuron mask, then immediately drops the full matrix after subsetting." Step 7 estimated ~2.10 s/GB → ~870 s for the full run, within the 15-minute budget; the actual run took 501 s. Step 6 also lists as *unfixed* inefficiencies: "Full-mode metadata currently recomputes frame-interval medians by reloading behavior files" and "Stimulus vocabulary collection reloads behavior dictionaries once more after catalog creation."

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI did not document any loop-vectorization opportunities. Present in the code: (1) the per-trial loop in `build_trial_arrays()`, which rebuilds `arange(StartFr, GrayFr)` and re-filters per trial instead of grouping frames by `ft_trInd` in one pass (this is the same loop the reference identifies, and it is negligible against I/O); (2) the per-reward-trial `nanmean` loop in `compute_neuron_selection()`, which could be a single `np.add.reduceat`/matrix product over a frame→trial indicator; (3) `sample_candidate_score()`, which loops over every trial of every session (and loads every behavior file) purely to pick two sample sessions; (4) the `for i in range(1, len(speed_edges))` tie-fixing loop (trivial). Nothing else iterates over frames or neurons in Python.

ii.
```python
    for trial_idx in range(int(beh["ntrials"])):
        frames = np.arange(start_fr[trial_idx], gray_fr[trial_idx], dtype=int)
        frames = frames[ft_move[frames] > 0]
```
```python
    lick_trials = 0
    for trial_idx, (s, g) in enumerate(zip(start, gray)):
        frames = np.arange(s, g, dtype=int)
        frames = frames[move[frames]]
        ...
```

iii. Step 6 lists the speed-ups that were implemented ("`--sample` processes the two smallest neural recordings", "Session processing loads each large spike file only once", "Speed quantiles are computed from concatenated retained running frames only"), i.e. the AI's efficiency effort went into I/O and memory rather than loop vectorization, on the (correct) view that I/O dominates.

## 12-c. What processing does the code repeat multiple times?

i. Behavior files are the main repetition: `load_behavior()` re-reads the whole `Beh_<exp_type>.npy` from disk every time it is called, and it is called once per session in `collect_stimulus_values()`, once per session in `compute_time_bin_ms()`, and once per session in `process_session()` — three full passes over 6.6 GB, where the reference reads each behavior file exactly once for the group of sessions it holds. Within that, the median frame interval is computed twice (globally in `compute_time_bin_ms()` and again per session inside `build_trial_arrays()`), the `SoundDelPos`-derived cue positions are computed twice per session (in `compute_neuron_selection()` and in `process_session()`), the trial frame windows are rebuilt in `compute_neuron_selection()` and again in `build_trial_arrays()`, and `neural_trial.astype(np.float32, copy=False)` is applied a second time in `finalize_io()`.

ii.
```python
    stimulus_values = collect_stimulus_values(full_catalog)   # one full pass over all beh files
    time_bin_ms = compute_time_bin_ms(full_catalog)           # a second full pass
    ...
        session = process_session(cand, ...)                  # a third read of the same beh file
```
```python
    cue_positions = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
```

iii. Step 6 "Code inefficiencies identified" names two of these explicitly (the metadata frame-interval pass and the stimulus-vocabulary pass) but they were left in the final script; the AI judged the cost acceptable given the 501 s total runtime.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI documents none. In the code: `build_trial_arrays()` returns a concatenated `speed_values` array that `process_session()` receives and never uses (main re-concatenates the same values from `output_raw`); `cue_positions_dm` and the whole `processing_info` dict are computed for every session even when `--show-processing` is off, and only the first two sessions ever plot them; `trial_summary` statistics are computed per session but only `mean_trial_length` is printed; `compute_time_bin_ms()` re-reads every behavior file to produce one scalar that could be taken from any session (or from the reference's 3.17 Hz); in `--sample` mode `sample_candidate_score()` scans every trial of all 89 sessions to pick two. The largest item is arguably the selectivity machinery itself: the `d′`, `dp_sound` and `reward_dp` arrays are computed over the full neuron set of every session and, apart from the boolean mask, are discarded — and the mask itself removes 92% of the neurons the downstream decoder could have used.

ii.
```python
    neural_trials, input_trials, output_trials, speed_values, trial_summary = build_trial_arrays(...)
    # speed_values is never referenced again in process_session()
```
```python
    cue_positions = np.mod(np.asarray(beh["SoundDelPos"], dtype=float), float(beh["Corridor_Length"]))
    cue_positions = cue_positions[np.isfinite(cue_positions)]
    proc_info = {**selection_info, **trial_summary, "cue_positions_dm": cue_positions,
                 "example_trial": example_trial}
```

iii. No rationale is given, because the AI did not flag any processing as unnecessary; Step 12 concludes "No additional conversion issues were found after the Step 10 curation fix."
