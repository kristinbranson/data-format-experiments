# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The master index `data/beh/Imaging_Exp_info.npy` is read first and flattened: every entry of every experiment type is keyed by the recording id `<mname>_<datexp>_<blk>`, giving 142 references that collapse to 89 unique recordings. For each unique recording the AI picks one "canonical" behavior entry among its duplicate references (see 1-c), then per session it loads three files: the behavior dictionary `Beh_<exp_type>.npy` (indexed by the session key, with `_<stimtype>` appended for swap sessions), the deconvolved traces `data/spk/<rec_id>_neural_data.npy` (a dict whose `spks` value is a list of per-plane chunks), and the retinotopy `data/retinotopy/<mouse>_<date>_trans.npz` (`iarea`). The behavior files are re-opened with `np.load` every time they are needed — there is no caching — so the 6.6 GB of `Beh_*.npy` files are read roughly five times over (duplicate validation, sample selection, global metadata, category collection, `time_bin_size`, and the conversion loop itself).

ii.
```python
def load_exp_info() -> dict:
    return np.load(ROOT / "data" / "beh" / "Imaging_Exp_info.npy", allow_pickle=True).item()

def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
    for exp_type, dbs in exp_info.items():
        for db in dbs:
            rec_id = f"{db['mname']}_{db['datexp']}_{db['blk']}"
            per_rec[rec_id].append((exp_type, db))
```
```python
        beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
        spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
        spk_chunks = list(spk_obj["spks"])
        ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
        iarea = np.asarray(ret["iarea"])
```

iii. From CONVERSION_NOTES Step 1/2/4: these are exactly the sources the reference `utils.load_spk`, `utils.load_exp_beh` and `utils.load_retino` use. The AI documents that `Imaging_Exp_info.npy` is iterated by experiment group in the reference code and that "the same recording can appear in multiple groups", resolving it to "89 unique recordings" to agree with the paper's "We performed 89 recordings in 19 mice".

## 1-b. How are the data split into subjects?

i. The subject is the `mname` field of the index entry, carried on each `SessionRef`. Sessions are sorted by `(subject, date, blk)` and `subjects` is built in order of first appearance, with `subject_idx` an int array of each session's index into that list. Result: 19 subjects, 89 sessions, session counts per subject identical to the expert reference.

ii.
```python
    sessions.sort(key=lambda s: (s.subject, parse_date(s.date_str), int(s.blk)))
```
```python
    for sess in sessions:
        if sess.subject not in subject_to_idx:
            subject_to_idx[sess.subject] = len(subjects)
            subjects.append(sess.subject)
...
        "subject_idx": np.array([subject_to_idx[s.subject] for s in sessions], dtype=np.int64),
```

iii. "Use `<mouse>_<date>_<blk>` for neural session identity"; the mouse name is already in the index so no split has to be inferred (CONVERSION_NOTES Step 4/5).

## 1-c. How are the data split into sessions?

i. A session is one unique recording id `<mname>_<datexp>_<blk>`. Because the same recording is listed under several experiment types, the AI groups all references per recording and then chooses one canonical behavior entry: the candidate with the most non-NaN `stim_id` entries, tie-broken by number of unique `WallName` values then by key. Before choosing, it *validates* that all duplicate references agree on `ntrials`, `WallName`, `isRew`, `SoundPos` and `ft_trInd`, raising if they disagree. 142 references → 89 sessions.

ii.
```python
        score = (
            int(np.sum(~np.isnan(np.asarray(beh.get("stim_id", []), dtype=float)))),
            len(np.unique(as_str_array(beh["WallName"]))),
            key,
        )
        candidates.append((score, exp_type, key, beh))
...
        same = (
            int(beh["ntrials"]) == int(base_beh["ntrials"])
            and np.array_equal(as_str_array(beh["WallName"]), as_str_array(base_beh["WallName"]))
            ...
        )
        if not same:
            raise ValueError(...)
    _, exp_type, key, _ = max(candidates)
```

iii. Step 4 discrepancy table: "142 experiment entries but only 89 unique recording IDs … Treat unique recording ID as the true session unit for conversion. Experiment-group duplicates are analysis references, not distinct neural recordings." Step 5 key decision 2: duplicates were checked to have matching `WallName`, `isRew`, `SoundPos` and trial counts, so `WallName` is the reliable stimulus label.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals the behavior declares, and each frame is assigned to a trial by `ft_trInd`. The frames kept for a trial are those that are inside the texture corridor **and** where the animal is running: `finite(ft_trInd) & ft_CorrSpc & (ft_move > 0)`. Grey-space frames are excluded, as in the reference, but so are all stationary frames — roughly 29% of the corridor frames of a session (e.g. 8,882 of 12,423 kept in `TX83_2022_08_17_1`). Trials therefore keep a variable number of bins (mean T = 22.3, min 11, max 178) that are **not contiguous in real time**: a trial in which the mouse stopped still appears, but its stationary bins are deleted, leaving a trial whose `time_since_trial_start` jumps across a gap (up to 1,765 s in one trial).

ii.
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)
    ...
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
    ft_move = np.asarray(beh["ft_move"], dtype=float) > 0
    valid = finite_trial & ft_corr & ft_move
    masks = []
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
        if len(frame_idx) == 0:
            raise ValueError(f"Trial {trial} has no retained running corridor frames")
        masks.append(frame_idx)
    return masks
```

iii. Step 5 key decisions 4 and 5: "Use only running corridor frames: This matches the paper statement 'We only considered timepoints during running for analysis' and the reference-code masks built from `ft_CorrSpc` and `ft_move > 0`"; and "Trial extent for export = corridor portion only, not gray space … Gray-space frames are excluded from exported trials."

## 1-e. How are trials filtered based on quality controls?

i. No trial-level quality control is applied. All 38,110 trials of all 89 sessions are exported. The only trial-level condition is structural: a trial with zero retained running-corridor frames raises `ValueError` and aborts the whole conversion rather than being skipped (no trial triggered it). No session is dropped and no trial-length outlier rule exists; the consequence is visible in the exported inputs, whose ranges reach `time_since_trial_start` = 1,765 s and `time_to_sound_cue` = −1,763 s (the expert reference, which drops trials above the 99th-percentile traversal length, spans ±75 s).

ii.
```python
        if len(frame_idx) == 0:
            raise ValueError(f"Trial {trial} has no retained running corridor frames")
        masks.append(frame_idx)
```
```python
        for trial_idx, frame_idx in enumerate(trial_masks):   # every trial is exported
```

iii. CONVERSION_NOTES Step 3 states "No explicit global trial-rejection rule is described in the provided paper/methods excerpts beyond restricting analyses to running timepoints", so no filter was added. In the Step 9 consistency table the extreme input ranges are listed and explained away as "Consistent with variable trial durations and running-only frame retention" and "Consistent with running-only frame retention and long rewarded trials". The Step 10 edge-case scan records only that "Every converted trial is non-empty (`min T = 11`, `max T = 178`)".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `data/spk/<rec_id>_neural_data.npy` — a list of per-imaging-plane neuron × frame arrays — together with `iarea` from `data/retinotopy/<mouse>_<date>_trans.npz`, which is mapped to the four grouped visual areas by the reference helper `utils.neu_area_ID`.

ii.
```python
        spk_obj = np.load(ROOT / "data" / "spk" / f"{sess.rec_id}_neural_data.npy", allow_pickle=True).item()
        spk_chunks = list(spk_obj["spks"])
        ret = np.load(ROOT / "data" / "retinotopy" / f"{sess.subject}_{sess.date_str}_trans.npz", allow_pickle=True)
        iarea = np.asarray(ret["iarea"])
```
```python
    spk = np.concatenate(spk_chunks, axis=0)
    areas = utils.neu_area_ID(iarea)
```

iii. Step 4: "The saved `spks` arrays are the deconvolved activity traces from Suite2p, not raw fluorescence; no additional dF/F computation is needed", matching the paper's "All our analyses were based on deconvolved fluorescence traces". Region labels come from the reference's own `neu_area_ID` grouping (`V1`, `mHV`, `lHV`, `aHV`).

## 2-b. How is the `neural` data processed?

i. The traces themselves are not transformed: no dF/F, no deconvolution, no smoothing, no z-scoring, no rebinning. For each trial the selected neurons' columns at the retained frames are taken and cast to `float16`. Trials keep their own length; nothing is padded. (The chunked layout is preserved: rows are selected inside each plane chunk and the per-trial pieces are concatenated.)

ii.
```python
            trial_chunks = []
            for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
                if len(local_rows) == 0:
                    continue
                trial_chunks.append(chunk[local_rows][:, frame_idx])
            neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
```

iii. Step 6: "Neural arrays are currently stored as `float16` to control output size while preserving continuous-valued activity." Step 4/10: the `spks` are already the analysis-ready deconvolved traces used by every reference analysis, so no further processing is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. This is the AI's largest departure. Instead of keeping all neurons of the four visual areas, it exports a small, *selectivity-defined* subset per session, and only from two of the four areas:

- **mHV stimulus-selective neurons.** The two "familiar" wall textures of the session are identified (`get_familiar_pair`, base names ending in `1`, preferring the rewarded stimulus). On even-indexed trials only (`ft_trInd % 2 == 0`, called "odd trials" in the reference code) and on running corridor frames, a d′ between the two familiar textures is computed per neuron. Neurons must also be "corridor-responsive" (`corr_neu`: mean activity on either stimulus greater than on grey frames). Among the mHV neurons satisfying `corr_neu`, the top 5% and bottom 5% by d′ are kept.
- **aHV reward-prediction neurons.** Only in sessions with rewards: aHV activity is interpolated into 60 position bins with the reference's `utils.get_interpPos_spk`, mean activity over bins 5:40 is compared between late-cue and early-cue trials, and neurons with `d' >= 0.3` and non-negative stimulus d′ are kept.
- **Fallbacks.** If fewer than 128 mHV neurons survive, the top 128 `corr_neu` mHV neurons by |d′| are taken; if nothing at all survives, the top 128 mHV neurons by |d′| are taken regardless of `corr_neu`.

`brain_regions` is therefore only `['mHV', 'aHV']`; V1 and lHV are absent. 102,541 of 4,691,034 neurons (2.2%) are exported, 1,152 per session on average versus 46,128 in the expert reference.

ii.
```python
    stim_dp = dprime(spk[:, stim_pos_fr], spk[:, stim_neg_fr])
    corr_neu = (spk[:, stim_pos_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1)) | (
        spk[:, stim_neg_fr].mean(axis=1) > spk[:, gray_train].mean(axis=1))

    mhv_mask = percentile_union_mask(stim_dp, corr_neu & areas["mHV"], 95, 5)

    if int(np.sum(mhv_mask)) < 128:
        mhv_candidates = np.where(corr_neu & areas["mHV"])[0]
        ...
                reward_dp = dprime(mean_corr[:, late], mean_corr[:, early])
                local_keep = (reward_dp >= 0.3) & (stim_dp[ahv_idx] >= 0)
                ahv_mask[ahv_idx[local_keep]] = True
    keep_mask = mhv_mask | ahv_mask
```
```python
        "brain_regions": ["mHV", "aHV"],
```

iii. Step 6: "Initial mapping plan was revised during implementation because the provided decoder concatenates all session data in memory. Exporting all visual-area neurons would be intractable. The implemented compromise uses a reference-style neuron subset" with the same d′ logic as `Get_coding_direction` and `Get_dprime_rewPred_neuron`. The trajectory (steps 112–123) shows the reasoning: raw data is ~412 GB, a naive export would be 76–152 GB, memmap-backed pickles were tried and abandoned, and the AI then decided "a more defensible tractable option than arbitrary downsampling: reuse the paper's own neuron-selection logic". Step 9/10 record the mismatch explicitly as "the only deliberate mismatch … needed to keep `train_decoder.py` tractable".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry: a trial's columns are the retained frames of that trial in increasing frame order, starting at the first running frame inside the corridor and ending at the last one. Trials are variable length, nothing is cut to a common window and nothing is padded. Metadata declares `temporal_alignment_event = 'corridor entry (trial start)'`, `off_start = 0.0`, `off_end = None`. Because non-running frames are removed, the first exported bin is the first *running* frame after entry rather than entry itself, and interior bins are not uniformly spaced.

ii.
```python
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
```
```python
        for trial_idx, frame_idx in enumerate(trial_masks):
            ...
            neural_trial = np.concatenate(trial_chunks, axis=0).astype(np.float16, copy=False)
            current_ft = ft[frame_idx]
            time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Step 5 decision 3: "Neural activity will stay frame-aligned rather than position-interpolated in the exported dataset: The requested decoder outputs include frame-resolved licking and running speed, so exported trials should preserve native behavioral alignment." Step 10(c): "Exported trials are aligned to corridor entry (`Trial_start_time`) … This preserves the native frame grid while matching the reference running-frame restriction."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. One column per imaging frame. `time_bin_size` is computed empirically as the median over sessions of the median inter-frame interval of `ft`, giving 314.69 ms (≈3.17 Hz, the reference notebook's frame rate). Note that, with non-running frames deleted, 314.69 ms is the width of a bin but not the spacing between consecutive exported bins.

ii.
```python
            "time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
```

iii. Step 10(d): "Neural data stay on the native imaging frame bins (median `314.69 ms`), which is consistent with the reference frame-based deconvolved traces." Step 1 notes "The notebook explicitly states imaging frame rate `fs = 3.17 Hz`", which the measured value confirms.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundTime` (the MATLAB datenum timestamp of the cue on each trial) and `ft` (the timestamp of every imaging frame). The reference instead interpolates the fractional cue frame `SoundFr` onto the frame-time axis; the two are numerically identical in the data (checked: max |difference| = 0.0 s in `TX83_2022_08_17_1`).

ii.
```python
        ft = np.asarray(beh["ft"], dtype=float)
        sound_time = np.asarray(beh["SoundTime"], dtype=float)
```

iii. Step 5 mapping table: "Behavior key fields `ft`, `SoundTime` → `input[0]` … Paper/methods cue timing + frame-aligned behavior structure". Step 10(e): "All four decoder inputs come directly from raw behavioral fields (`SoundTime`, `Trial_start_time`, `ft`, `isRew`)".

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. For each retained frame of a trial, `(SoundTime[trial] − ft[frame]) × 86400`, i.e. seconds, positive before the cue and negative after — the same sign convention as the reference. Stored as `float32`, time-varying. No clipping or normalisation; values inherit the extreme ranges of unfiltered stopped trials (global range −1763.3 to 723.5 s).

ii.
```python
            current_ft = ft[frame_idx]
            time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. Step 5: "For each retained frame, compute signed seconds to cue … `time_to_sound_cue`; positive before cue, negative after cue." Step 5 decision 7: "Use actual behavioral timestamps for continuous time covariates: `ft` and `Trial_start_time` / `SoundTime` provide trial-aligned times in days; converting to seconds preserves real elapsed time even when non-running frames are removed."

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same `frame_idx` used to slice the neural columns of that trial, so it has the trial's length and shares the neural time base bin-for-bin.

ii.
```python
            trial_chunks.append(chunk[local_rows][:, frame_idx])
            ...
            current_ft = ft[frame_idx]
            time_to_cue = ((sound_time[trial_idx] - current_ft) * 86400.0).astype(np.float32)
```

iii. Step 10(f)/(c): all streams are built "on the same retained frame indices used for neural export"; every behavioral stream in this dataset is already on the imaging-frame grid.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The recording date `datexp` in the index entry (parsed by `parse_date`), grouped by mouse.

ii.
```python
def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y_%m_%d")
```
```python
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
```

iii. Step 5 mapping table: "Session date / mouse chronology → `input[1]` … Session metadata in `Imaging_Exp_info.npy`"; a per-trial continuous covariate required by the decoder task but not present in the raw behavior.

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse, the earliest imaging date is found and each session gets `(date − first_date).days + 1` — **calendar** days elapsed, 1-based. Range across the dataset is 1 to 93. The value is broadcast as a constant over all bins of every trial of the session. (The expert reference instead counts recorded sessions, giving 0–7.)

ii.
```python
    day_map: dict[str, dict[str, float]] = defaultdict(dict)
    for subject, sess_list in by_subject.items():
        first_date = min(parse_date(s.date_str) for s in sess_list)
        for sess in sess_list:
            day_map[subject][sess.rec_id] = float((parse_date(sess.date_str) - first_date).days + 1)
```
```python
        day_value = np.float32(day_map[sess.subject][sess.rec_id])
        ...
            day_of_training = np.full(len(frame_idx), day_value, dtype=np.float32)
```

iii. Step 5 mapping table: "Compute per-session day as calendar-day offset from each mouse's first unique imaging recording; repeat over trial frames." Step 5 decision 11: per-trial constants are broadcast over time so every trial has a consistent `(n_features, n_timepoints)` shape.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `Trial_start_time` (the datenum of corridor entry for each trial) and `ft`. Verified to equal the reference's `np.interp(StartFr, arange(n), frame_time)` exactly.

ii.
```python
        trial_start = np.asarray(beh["Trial_start_time"], dtype=float)
```

iii. Step 5 mapping table: "Behavior field `ft` + `Trial_start_time` → `input[2]` … Trial alignment fields documented in notebook / behavior structure."

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(ft[frame] − Trial_start_time[trial]) × 86400`, in seconds, positive after entry, stored as `float32` and time-varying. It starts at the time of the first *running* frame of the trial (slightly greater than zero), is monotonically increasing within a trial, and — because stationary frames are deleted and no long-trial filter exists — can jump discontinuously and reach 1,765 s.

ii.
```python
            time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
```

iii. Step 5 decision 7 (real elapsed time is preserved even when non-running frames are removed); Step 10 edge-case scan: "All checked `time_since_trial_start` traces are monotonic within trial."

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Computed at the same `frame_idx` as the neural columns, so it is bin-for-bin aligned and has the trial's length.

ii.
```python
            current_ft = ft[frame_idx]
            time_since_start = ((current_ft - trial_start[trial_idx]) * 86400.0).astype(np.float32)
            trial_input = np.vstack([time_to_cue, day_of_training, time_since_start, reward_available]).astype(np.float32, copy=False)
```

iii. Same as 3-c: every stream is indexed by the same retained-frame array.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial flag marking trials run in the rewarded corridor.

ii.
```python
        is_rew = np.asarray(beh["isRew"], dtype=float)
```

iii. Step 5 mapping table: "`reward_available`; must come from raw per-trial value because many sessions are unrewarded or partially rewarded." Step 4 checks that the pooled `isRew` fraction (~0.114) is low because the unsupervised/naive/grating cohorts are entirely unrewarded.

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. None beyond a cast to float and broadcast over the trial's bins as a constant 0/1 channel.

ii.
```python
            reward_available = np.full(len(frame_idx), is_rew[trial_idx], dtype=np.float32)
```

iii. Step 5 decision 11: per-trial constants are repeated across frames so all input rows share the time axis.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the string naming the wall texture of each trial. `stim_id` and `TrialStim` are deliberately not used.

ii.
```python
        wall_name = as_str_array(beh["WallName"])
```

iii. Step 5 decision 8: "Use `WallName` string labels for stimulus decoding targets: This preserves all naturalistic / grating / swap categories present in the raw trials and avoids ambiguity from NaNs in `stim_id`."

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The set of distinct `WallName` strings over all converted sessions is collected and sorted, giving **15** classes (`circle1, circle2, circle3, leaf1, leaf1_swap1, leaf1_swap2, leaf2, leaf3, rock1, rock2, wood1, wood1_swap1, wood1_swap2, wood2, wood5`). Each trial's name is mapped to its index and broadcast over the trial's bins. No collapsing onto the four base textures (circle / leaf / rock / wood) is done, so crops of the same texture (`leaf1`, `leaf2`, `leaf3`) and spatial shuffles (`leaf1` vs `leaf1_swap1`) are separate categories. Global fractions: `[0.252, 0.049, 0.010, 0.264, 0.019, 0.019, 0.128, 0.040, 0.071, 0.013, 0.065, 0.008, 0.010, 0.038, 0.014]`; each individual session contains only 2–6 of the 15.

ii.
```python
    categories = sorted({str(v) for sess in sessions
                         for v in np.asarray(load_beh(...)["WallName"]).tolist()})
    category_to_idx = {name: idx for idx, name in enumerate(categories)}
...
            stim_code = np.full(len(frame_idx), category_to_idx[str(wall_name[trial_idx])], dtype=np.int16)
```

iii. Step 5 decision 8 (above) and the Step 9 table, which records "15 category strings present across raw sessions → 15 category strings present" as a match. The AI also notes in Step 3 that `leaf2`, `circle2`, `leaf3` and the shuffled `leaf1` variants are the paper's *introduced* stimuli, i.e. it was aware they are variants of the same base textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr` (the fractional imaging-frame number of every lick) together with `LickTrind` (the trial each lick belongs to), used to restrict licks to the current trial.

ii.
```python
        lick_fr = np.asarray(beh["LickFr"], dtype=float).astype(int)
        lick_tr = np.asarray(beh["LickTrind"], dtype=float).astype(int)
```

iii. Step 5 mapping table: "Behavior fields `LickFr`, `LickTrind` → `output[1]` … Binary frame vector: 1 if one or more licks fall in retained frame, else 0."

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are truncated to integers (`astype(int)`), the licks of the current trial are selected via `LickTrind`, and a retained frame is labelled 1 if any of them falls in it, else 0. Values `['no_lick', 'lick']`; global fraction 0.963 / 0.037 (expert reference 0.959 / 0.041). Because only running frames are retained, licks emitted while the animal is stationary are dropped together with their frames — about 42% of all in-corridor licks in the session checked (`TX109_2023_04_18_1`). A further ~0.7% of licks are lost because their `LickTrind` disagrees with `ft_trInd` at the truncated frame.

ii.
```python
            lick_trial_frames = lick_fr[lick_tr == trial_idx]
            licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
```

iii. Step 3 notes the paper's licking measure "explicitly counts licks occurring inside the corridor before the sound cue"; the AI keeps a plain per-frame binary indicator as the decoder task requires ("Licking, binary, time-varying").

## 8-c. How is `output` *Licking* aligned with the neural data?

i. `LickFr` already indexes imaging frames, so after truncation the indicator lives on the neural grid; it is evaluated by membership test against the same `frame_idx` used for the neural columns, hence bin-for-bin aligned and the trial's length.

ii.
```python
            licking = np.isin(frame_idx, lick_trial_frames).astype(np.int16)
            trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. Step 10(f): all outputs are built "on the same retained frame indices used for neural export". The `--show-processing` plot overlays lick frames on the retained-frame position trace to demonstrate this visually.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the within-corridor position at each imaging frame, in decimetres (0–40 across the 4 m texture, on to 60 through the grey space).

ii.
```python
        ft_pos = np.asarray(beh["ft_Pos"], dtype=float)
```

iii. Step 4: "Data are stored in decimeter units: 60 dm total = 6 m, with 40 dm texture corridor + 20 dm gray space. This is fully consistent with paper + code."

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position at the retained frames is clipped to `[0, 39.999]` dm, integer-divided by 10 dm, and clipped again to `[0, 3]`, producing an `int16` class index broadcast per frame. Since only `ft_CorrSpc` frames are retained, positions are already inside 0–40 dm, so the clipping is defensive only. Global occupancy `[0.250, 0.249, 0.250, 0.252]` (reference `[0.254, 0.242, 0.247, 0.257]`).

ii.
```python
            pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
            pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Step 5 decision 9: "Discretize position with paper-consistent meter bins: Raw positions are in decimeters; use 4 bins across the 40 dm texture corridor to match the requested 4 equal 1 m bins."

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Four equal 1 m bins with edges at 0 / 10 / 20 / 30 / 40 dm, named `['0-1m', '1-2m', '2-3m', '3-4m']` — exactly the "4 equal-length, 1-m-long spatial bins" the decoder task asks for. The thresholds are fixed (not data-driven), so the bins are physically identical across sessions.

ii.
```python
        "output_values": [ categories, ["no_lick", "lick"], ["0-1m", "1-2m", "2-3m", "3-4m"], ["q1", "q2", "q3", "q4"] ],
```
```python
            pos_bin = np.clip((pos // 10.0).astype(np.int16), 0, 3)
```

iii. Same as 9-b; the AI's processing plot draws horizontal lines at 10/20/30 dm over the position trace of a sample trial to show the discretisation is correct.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame, and it is indexed with the same `frame_idx` as the neural columns, so it is bin-for-bin aligned and of the trial's length.

ii.
```python
            pos = np.clip(ft_pos[frame_idx], 0.0, 39.999)
```

iii. As in 8-c: every stream is read out on the retained-frame index array.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the running speed at each imaging frame.

ii.
```python
        ft_speed = np.asarray(beh["ft_RunSpeed"], dtype=float)
```

iii. Step 5 mapping table: "Behavior field `ft_RunSpeed` → `output[3]` … `running_speed_bin`; quartiles should each contain ~25% of retained samples."

## 10-b. What processing is involved in computing `output` *Running speed*?

i. A first, behavior-only pass over **all** sessions collects `ft_RunSpeed` at every retained (running, in-corridor) frame and computes three **global** quantile edges at 0.25 / 0.5 / 0.75. Those same three edges are then applied to every session. Because stationary frames were already removed by the `ft_move` mask, the zero-speed tie that would otherwise make quantile edges degenerate is almost absent (0.01% of retained frames are exactly 0 in the session checked, versus 20% of all corridor frames). Global occupancy is `[0.250, 0.250, 0.250, 0.250]`. Per session, however, occupancy is far from uniform (e.g. `DR10_2022_07_12_1`: `[0.956, 0.042, 0.002, 0.000]`).

ii.
```python
    speed_values = np.concatenate(speed_values).astype(np.float32)
    speed_edges = np.quantile(speed_values, [0.25, 0.5, 0.75]).astype(np.float32)
```
```python
def speed_to_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges, right=False), 0, 3).astype(np.int16)
```

iii. Step 5 decision 10: "Discretize running speed with global quartiles over retained samples: The user explicitly requests 25% data bins, so quartiles will be computed after all filtering decisions are applied." Step 6: "Global speed-bin estimation uses a lightweight first pass over behavior only; neural files are not touched until conversion stage."

## 10-c. How is `output` *Running speed* thresholded into categories?

i. `np.digitize(speed, edges, right=False)` against the three global quartile edges, clipped to `[0, 3]`, labelled `['q1','q2','q3','q4']`. Edges are values (thresholds), not ranks, and are shared by all sessions, so a given bin means the same physical speed everywhere; the edges are reported in metadata as `speed_bin_edges`.

ii.
```python
            speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
```
```python
            "speed_bin_edges": [float(x) for x in speed_edges.tolist()],
```

iii. Step 9 consistency table lists the resulting distribution `[0.250, 0.250, 0.250, 0.250]` as matching the requested "25% of the data" specification; the `--show-processing` plot overlays the edges on the histogram of retained speeds.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` has one value per imaging frame and is indexed with the same `frame_idx` as the neural columns, so it is bin-for-bin aligned and of the trial's length.

ii.
```python
            speed_bin = speed_to_bin(ft_speed[frame_idx], speed_edges)
            trial_output = np.vstack([stim_code, licking, pos_bin, speed_bin]).astype(np.int16, copy=False)
```

iii. As in 8-c/9-d.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is partial and inconsistent:
- **Behavior longer than imaging.** Every session's behavior arrays are 1–2 frames longer than `spks` (checked in six sessions, e.g. `TX83_2022_08_17_1`: 20,429 behavior frames vs 20,428 imaging frames). The AI truncates behavior to `nfr = spk.shape[1]` **only inside the neuron-selection routine**; `compute_trial_masks` and all input/output construction use the full-length behavior arrays, so a frame index beyond the last imaged frame would raise `IndexError` and abort the run. It did not happen — no trailing frame was a running-corridor frame — but nothing in the code prevents it. The expert reference cuts every stream with `[:nfr]` as the reference notebook does.
- **Empty trials.** A trial with no retained frames aborts the conversion with `ValueError` rather than being skipped.
- **Non-finite trial indices.** `ft_trInd` NaNs are handled explicitly and excluded.
- **Duplicate session references.** Validated for equality and rejected with `ValueError` if they disagree.
- **Missing familiar pair / non-monotonic cumulative position / too few selective neurons.** Handled with fallbacks in the neuron-selection path (top-128 by |d′|, skipping the aHV branch).
- No handling for NaN `SoundTime`/`Trial_start_time` (none occur in this dataset).

ii.
```python
    nfr = spk.shape[1]
    ft_wall = as_str_array(beh["ft_WallID"][:nfr])          # truncated here ...
```
```python
def compute_trial_masks(beh: dict) -> list[np.ndarray]:
    ft_trial = np.asarray(beh["ft_trInd"], dtype=float)      # ... but not here
    ...
    ft_corr = np.asarray(beh["ft_CorrSpc"]).astype(bool)
```
```python
    finite_trial = np.isfinite(ft_trial)
    ft_trial_int[finite_trial] = ft_trial[finite_trial].astype(np.int64)
```

iii. CONVERSION_NOTES Step 10 edge-case scan claims: "Every converted trial is non-empty", "No off-by-one issues were found at trial starts/ends in the checked raw-vs-converted comparisons." The mismatch between behavior length and imaging length is not discussed anywhere in the notes, and Step 3 records no explicit trial-rejection rule in the paper.

## 12-a. What are the most time-consuming steps of the code?

i. Measured: 1,013.7 s (16.9 min) for 89 sessions, 3.9–46.9 s per session. The dominant costs are (1) reading the ~405 GB of `spk/*.npy` files, one full session at a time — unavoidable and shared with the reference; (2) the neuron-selection pass, which concatenates the whole session into one array (`np.concatenate(spk_chunks)`, several GB, held alongside the chunks) and then computes d′ and corridor/grey means over the full neuron × frame matrix, plus `utils.get_interpPos_spk` position interpolation for aHV neurons in rewarded sessions; (3) the per-trial neural slice, which re-materialises the selected rows over *all* frames of every plane chunk once per trial (see 12-b); and (4) repeatedly re-reading the 6.6 GB of `Beh_*.npy` files — `load_beh` is called with no caching in duplicate validation, sample selection, `build_global_metadata`, the category scan, the `time_bin_size` expression and the conversion loop, roughly five full passes plus one per duplicate reference.

ii.
```python
    spk = np.concatenate(spk_chunks, axis=0)     # whole session materialised again
    ...
    interp_spk = utils.get_interpPos_spk(spk[ahv_idx][:, move_idx], poscum_move, int(beh["ntrials"]), n_bins=60, lengths=float(beh["Corridor_Length"]))
```
```python
        print(f"[{sess_idx + 1}/{len(sessions)}] {sess.rec_id}: "
              f"{kept_stats['n_selected']} neurons, {len(session_neural)} trials, "
              f"{total_tp} retained timepoints, {elapsed:.1f}s")
```

iii. Step 6 identifies only one bottleneck — "Full-session interpolation for reward-prediction selection can be expensive if applied to all neurons; implementation restricts this step to aHV neurons only" — and lists three speedups: one recording in memory at a time, "trial construction concatenates only the selected neuron subset", and a behavior-only first pass for speed quartiles. Step 9 accepts the 16.9 min runtime as "above the nominal 15 min guideline but … within the Step 7 estimate band".

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. Three, in decreasing importance:
- **The per-trial chunk slice.** `chunk[local_rows]` is advanced indexing, so each trial copies the selected rows across *all* frames of the chunk (e.g. ~1,000 neurons × 20,000 frames ≈ 40 MB per chunk) before selecting the ~25 frames it needs, and this is repeated for every one of the session's ~430 trials. Hoisting `sel = chunk[local_rows]` out of the trial loop — or slicing the session once with the concatenated `kept_idx` — would remove essentially all of this work.
- **`compute_trial_masks`** evaluates `valid & (ft_trial_int == trial)` once per trial, scanning the whole frame index `ntrials` times instead of grouping frames by trial in a single pass (`np.argsort`/`np.split`). The reference has the same pattern.
- **The per-trial input/output construction**, which builds four `np.full` vectors and two `np.vstack`s per trial; these could be built once per session and split.

ii.
```python
        for trial_idx, frame_idx in enumerate(trial_masks):
            trial_chunks = []
            for chunk, local_rows in zip(spk_chunks, chunk_local_rows):
                if len(local_rows) == 0:
                    continue
                trial_chunks.append(chunk[local_rows][:, frame_idx])   # full-width copy, every trial
```
```python
    for trial in range(int(beh["ntrials"])):
        frame_idx = np.flatnonzero(valid & (ft_trial_int == trial))
```

iii. No justification is given: CONVERSION_NOTES Step 6 asserts the opposite — "Trial construction concatenates only the selected neuron subset, not the full recording" — and neither loop is listed among the inefficiencies identified.

## 12-c. What processing does the code repeat multiple times?

i.
- `load_beh(...)` re-reads whole `Beh_<exp_type>.npy` files (300–500 MB each, 6.6 GB total) every time behavior is needed: once per duplicate reference in `choose_canonical_behavior` (142 references), once per session in `build_global_metadata`, once per session in the `categories` comprehension, once per session again inside the `time_bin_size` expression, once per session in the conversion loop, plus once per candidate session in the `--sample` branch. Nothing is memoised.
- `compute_trial_masks(beh)` is computed twice per session (once in `build_global_metadata` for the speed quantiles, once in `convert_dataset`).
- `np.median(np.diff(ft))` is computed per session in `build_global_metadata` into the local `dts` list, then computed again for all sessions inside the `time_bin_size` expression.
- The per-trial neural slice repeats the full-width row selection for every trial (12-b).
- `as_str_array(beh["WallName"])` is recomputed in `choose_canonical_behavior`, `compute_selected_neurons`, `convert_dataset` and the category scan.

ii.
```python
def load_beh(exp_type: str) -> dict:
    return np.load(ROOT / "data" / "beh" / f"Beh_{exp_type}.npy", allow_pickle=True).item()
```
```python
    for sess in sessions:
        beh = load_beh(sess.canonical_exp_type)[sess.canonical_beh_key]
        ft = np.asarray(beh["ft"], dtype=float)
        dts.append(np.median(np.diff(ft)) * 86400.0)
        trial_masks = compute_trial_masks(beh)
```
```python
            "time_bin_size": float(np.median([np.median(np.diff(np.asarray(load_beh(s.canonical_exp_type)[s.canonical_beh_key]['ft'], dtype=float))) * 86400.0 for s in sessions]) * 1000.0),
```

iii. Not discussed. The only related claim is Step 6's "Behavior-only first pass for speed quartiles and metadata: Avoids touching neural files until conversion stage", which describes the extra behavior passes as a speedup rather than as repeated work.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- `build_global_metadata` builds `dts` (per-session frame intervals) and `category_values` (the set of wall names) and returns neither; both are recomputed elsewhere.
- The duplicate-reference validation in `choose_canonical_behavior` loads every duplicate behavior file and compares five arrays purely as a consistency assertion; the chosen entry's fields are identical by construction of that same check.
- `region_names` is built as an array of Python strings and then immediately converted to an integer index array.
- `SessionRef.refs` (the tuple of experiment types / stimtypes) is stored on every session and never used.
- `kept_stats["stim_pos"] / ["stim_neg"]` and `n_mhv_selected / n_ahv_selected` are only used for printing/plots.
- The `--show-processing` path recomputes `np.concatenate(trial_masks)` and re-reads behavior arrays for the plots (only when the flag is passed).
- Everything computed for neuron selection (d′, `corr_neu`, the 60-bin aHV interpolation) is discarded after producing `kept_idx` — it is not itself exported, though it does determine the export.

ii.
```python
    dts = []
    speed_values = []
    category_values = set()
    for sess in sessions:
        ...
        dts.append(np.median(np.diff(ft)) * 86400.0)
        category_values.update(str(v) for v in np.asarray(beh["WallName"]).tolist())
    ...
    return subjects, subject_to_idx, day_map, speed_edges      # dts, category_values dropped
```
```python
    region_names = np.empty(np.sum(keep_mask), dtype=object)
    region_names[:] = "mHV"
    region_names[np.isin(kept_indices, np.where(ahv_mask)[0])] = "aHV"
    region_idx = np.array([0 if name == "mHV" else 1 for name in region_names], dtype=np.int64)
```

iii. Not identified in CONVERSION_NOTES; Step 6 lists only the reward-prediction interpolation as a potential inefficiency and states it was restricted to aHV neurons.
