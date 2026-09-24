# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the master index `beh/Imaging_Exp_info.npy` at all. Instead it enumerates every `beh/Beh_*.npy` file in sorted order, loads each one fully, and keys every behavior record by a "base session key" (`mname_yyyy_mm_dd_blk`) obtained by stripping any `_swapN` suffix. All 23 behavior files (~6.6 GB) are merged into one in-memory dict `behaviors` that is held for the whole run. The resulting 89 base keys were explicitly checked against the 89 spike files and found to match exactly. Neural data are then loaded per session from `spk/<base_key>_neural_data.npy` (`['spks']`, a list of one array per imaging plane) and the visual-area labels from `retinotopy/<mouse>_<yyyy_mm_dd>_trans.npz` (`iarea`). A neuron-count consistency check between spikes and retinotopy raises if they disagree.

ii.
```python
def merge_behavior_records() -> dict[str, dict]:
    merged = {}
    beh_files = sorted(
        fname for fname in os.listdir(BEH_ROOT) if fname.startswith("Beh_") and fname.endswith(".npy")
    )
    for fname in beh_files:
        beh = np.load(os.path.join(BEH_ROOT, fname), allow_pickle=True).item()
        for key, record in beh.items():
            base_key = parse_base_session_key(key)
            if base_key not in merged:
                merged[base_key] = {k: v for k, v in record.items()}
                continue
            ...
```
```python
spk_path = os.path.join(SPK_ROOT, f"{base_key}_neural_data.npy")
retino_path = os.path.join(RETINO_ROOT, f"{subject}_{base_key.split('_', 1)[1][:10]}_trans.npz")

spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
nneu_total = sum(arr.shape[0] for arr in spk_planes)
if nneu_total != iarea.shape[0]:
    raise RuntimeError(...)
```

iii. From the trajectory (step 38, 61): "Some recordings are reused across multiple figure-analysis files, so I'm checking whether those are genuinely different trial sets or just alternate annotations of the same session. That determines whether the converted dataset should deduplicate recordings or treat them as separate sessions." … "I've confirmed the duplicated entries are the same recordings with alternate stimulus annotations." The AI verified (step 77–78) that the 89 base keys derived from the behavior files are exactly the 89 spike files, so it treated the behavior directory itself as the session index rather than `Imaging_Exp_info.npy`.

## 1-b. How are the data split into subjects?

i. The subject is the first underscore-separated token of the base session key. `subjects` is the sorted unique list (19 mice) and `subject_idx` is each session's index into it, stored as `np.int16`. Session order is the sorted base-key order, so sessions of a mouse are contiguous and in date order.

ii.
```python
def parse_subject_and_date(base_key: str) -> tuple[str, datetime]:
    parts = base_key.split("_")
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
    return subject, date
```
```python
subjects = sorted(per_subject_dates)
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. No explicit justification in the trajectory; the mouse name is part of the session key/file name, so no derivation is needed. The result (19 subjects, same sessions-per-subject counts) was confirmed by the format verifier.

## 1-c. How are the data split into sessions?

i. A session is one mouse / date / block, i.e. one base key `mname_yyyy_mm_dd_blk`. Behavior entries that differ only by a `_swapN` stimulus-annotation suffix are collapsed onto a single recording; the first record encountered (alphabetical file order) supplies all fields and only `stim_id` is merged across the duplicates by filling NaNs. Every session is processed with all its sessions kept: 89 sessions total, all present in the output.

ii.
```python
def parse_base_session_key(key: str) -> str:
    parts = key.split("_")
    if parts[-1].startswith("swap"):
        return "_".join(parts[:-1])
    return key
```
```python
            # The duplicated session entries differ only in stimulus annotation
            # fields used for distinct figure analyses. Merge the stimulus IDs so
            # each physical recording is represented once with the richest mapping.
            stim_old = np.asarray(merged[base_key]["stim_id"], dtype=float)
            stim_new = np.asarray(record["stim_id"], dtype=float)
            if stim_old.shape == stim_new.shape:
                fill = np.isnan(stim_old) & ~np.isnan(stim_new)
                stim_old[fill] = stim_new[fill]
                merged[base_key]["stim_id"] = stim_old
```

iii. Step 61: "I've confirmed the duplicated entries are the same recordings with alternate stimulus annotations, mainly in the newer naive/grating cohorts. I'm now collapsing those to one recording apiece." The AI directly compared duplicated entries field-by-field (step 39) and found `ntrials`, `WallName`, `isRew`, `StartFr`, `SoundFr` identical, with only `stim_id`/`TrialStim` differing.

## 1-d. How are the data split into trials?

i. Trials are the `ntrials` traversals declared by the behavior, and a frame belongs to trial `tr` if `ft_trInd == tr`. On top of the corridor mask `ft_CorrSpc`, the AI adds a **running mask** `ft_move > 0`, so only frames where the animal was inside the 4 m texture *and* the VR was moving are kept. Trials with no surviving frame are silently skipped. Trials are variable length and are never padded or truncated to a common window.

ii.
```python
    trial_idx = np.asarray(beh["ft_trInd"][:nfr])
    move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
...
    for tr in range(int(beh["ntrials"])):
        mask = (trial_idx == tr) & move_corr_mask
        if not np.any(mask):
            continue
```

iii. Step 16: "they use deconvolved traces, restrict many figure analyses to frames where the corridor is visible and the VR is moving". Step 31: "The remaining gap is whether the conversion should keep all frames per trial or only the running periods the paper analyzed… so that choice is tied to the authors' own pipeline rather than guesswork." The reference `utils.py` indeed contains `VRmove = beh['ft_move'][:nfr]>0; fr_valid = VRmove & isCorridor  # only use activity inside the texture area plus mouse is running`, which the AI adopted verbatim as its frame-selection rule.

## 1-e. How are trials filtered based on quality controls?

i. Essentially none. The only trial-level exclusion is "no frames survive the corridor+running mask". There is no trial-length outlier rule and no exclusion of trials where the mouse stopped for a long time. Session-level, the AI raises `RuntimeError` (aborting the whole conversion) if a session ends up with fewer than two trials or with no selected neurons; in practice neither triggered, so all 89 sessions and all 38,110 trials are kept.

ii.
```python
        mask = (trial_idx == tr) & move_corr_mask
        if not np.any(mask):
            continue
```
```python
        if len(neural_trials) < 2:
            raise RuntimeError(f"Session {base_key} has fewer than two valid trials after filtering.")
```

iii. Step 133: "The timing ranges are still wide, but after checking the construction they can arise legitimately from trials where mice paused for long periods and only the running frames were retained, which is consistent with the paper's running-only analysis rule. The decisive check now is decoder training." So the AI saw the extreme trials, attributed them to legitimate pauses, and deliberately kept them on the grounds that the running-frame mask already removes the stationary bins.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `spks` from `spk/<session>_neural_data.npy` (Suite2p deconvolved traces, one neurons × frames array per imaging plane), plus `iarea` from `retinotopy/<mouse>_<date>_trans.npz` for the area assignment. Unlike the reference the planes are **not** concatenated up front; selection is done plane-by-plane and only the selected rows are concatenated.

ii.
```python
spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
iarea = np.asarray(np.load(retino_path, allow_pickle=True)["iarea"], dtype=int)
```
```python
    for plane_idx, plane in enumerate(spk_planes):
        ...
        plane_areas = iarea[offset : offset + plane.shape[0]]
```

iii. Step 16: the analysis code "use[s] deconvolved traces … and use[s] retinotopy to group neurons into `V1`, `mHV`, `lHV`, and `aHV`". The metadata records "Used deconvolved Suite2p traces."

## 2-b. How is the `neural` data processed?

i. The traces themselves are not transformed — no dF/F, no z-scoring, no smoothing, no spatial interpolation. For each trial the selected neurons' columns at the kept (corridor + running) frames are sliced out and cast to `float16`.

ii.
```python
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. Metadata: "Used deconvolved Suite2p traces. Kept only frames where the animal was running and located within the 4 m corridor, matching the paper's running-only analyses." The AI noted (step 71) that "these raw recordings are large enough that an unfiltered trial-by-trial export could be unmanageable", motivating the `float16` cast and the neuron subselection described in 2-c.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Three stacked filters. (1) Neurons whose `iarea` is not in V1 `[8]`, mHV `[0,1,2,9]`, lHV `[5,6]` or aHV `[3,4]` are dropped. (2) A "corridor-responsive" criterion: a neuron is kept only if its mean activity on running corridor frames exceeds its mean activity on running grey-space frames. (3) A hard cap: within each of the four areas the surviving neurons are ranked by their variance over corridor frames and only the top `MAX_NEURONS_PER_AREA = 128` are kept, i.e. exactly **512 neurons per session, 45,568 in total**, with an artificially equal 11,392 neurons in each of the four areas. For the smallest session this keeps 512 of 17,363 area-assigned neurons (10,857 of which pass the responsiveness test); the reference keeps every area-assigned neuron (4.1 M in total, mean 46,128 per session).

ii.
```python
# To keep the exported dataset trainable, cap the number of retained neurons per
# broad visual area after applying the paper-consistent running/corridor filter.
MAX_NEURONS_PER_AREA = 128
```
```python
        corr_mean = plane[:, corr_mask].mean(axis=1)
        gray_mean = plane[:, gray_mask].mean(axis=1)
        corr_var = plane[:, corr_mask].var(axis=1)
        responsive = corr_mean > gray_mean
```
```python
        # Highest variance first, then plane/row for deterministic tiebreaks.
        order = np.lexsort((row_arr, plane_arr, -score_arr))
        top = order[:MAX_NEURONS_PER_AREA]
```

iii. Step 71: "these raw recordings are large enough that an unfiltered trial-by-trial export could be unmanageable. I'm checking dataset scale now before I lock in the final representation, so the converter matches the paper but still produces a tractable `converted_data.pkl` and decoder run." Step 79/87: "retain a deterministic, paper-motivated subset of retinotopically assigned corridor-responsive neurons per session so the dataset stays trainable… cap neurons deterministically per broad visual area so the final pickle is large but still trainable by the provided decoder." The responsiveness rule mirrors `corr_neu = (spk[:, stim1_fr].mean(1) > spk[:, grey_fr].mean(1)) | ...` from the paper's sequence-sorting analysis; the 128-per-area cap and variance ranking are the AI's own additions for tractability.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to corridor entry / trial start: each trial array begins at the first *retained* frame of that traversal and runs to the last. Trials keep their own lengths (T from 11 to 178 bins, mean 21.6); nothing is padded or cut to a common window. Metadata declares `temporal_alignment_event = 'corridor entry (trial start)'`, `off_start = 0.0`, `off_end = None`. Because non-running frames are dropped, the first retained bin is on median 0.18 s after `StartFr`, but can be as late as 56.8 s after corridor entry when the animal entered and stood still.

ii.
```python
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
```
```python
        time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```
```python
            "temporal_alignment_event": "corridor entry (trial start)",
            "off_start": 0.0,
            "off_end": None,
```

iii. Implicit in step 79 ("keep corridor-aligned trial segments"). The explicit `time_since_trial_start` input is what carries the alignment information to the decoder, since trials have different lengths.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling: one bin = one imaging frame. `time_bin_size` is computed empirically as the median of `np.diff(ft)` (converted from MATLAB datenum days to ms) across all sessions, giving 314.69 ms (≈3.18 Hz); the per-session value `dt_seconds` is used for the timing inputs. Note that, because non-running frames are removed inside trials, consecutive bins of a trial are *not* uniformly spaced: 5.3 % of consecutive bin pairs are non-adjacent frames, 44.8 % of trials contain at least one internal gap longer than 1 s, and the largest gap is ~1,271 frames (~6.7 min).

ii.
```python
def compute_time_bin_size_ms(behaviors: dict[str, dict]) -> float:
    dts = []
    for beh in behaviors.values():
        ft = np.asarray(beh["ft"])
        if ft.size > 1:
            dts.append(np.median(np.diff(ft)) * MS_PER_DAY)
    return float(np.median(dts))
```
```python
        dt_seconds = float(np.median(np.diff(np.asarray(beh["ft"], dtype=np.float64))) * SECONDS_PER_DAY)
```

iii. Step 125–127: the AI switched the timing inputs from absolute behavior timestamps to frame-index arithmetic, "which is the correct alignment space for the neural data". The imaging frame is treated as the native resolution of the dataset, so no rebinning is applied.

## 3-a. What variables in the raw data is `input` *Time to sound cue* derived from?

i. `SoundFr[tr]` (the fractional imaging-frame number of the cue in each trial), the integer frame indices of the retained frames, and the per-session frame interval `dt_seconds` derived from `ft`.

ii.
```python
    frame_numbers = np.arange(nfr, dtype=np.float32)
...
        selected_frames = frame_numbers[mask]
        time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. Step 125: "some `time_to_sound` and `time_since_start` values are implausibly large, which points to mixing absolute behavior timestamps with imaging-frame alignment. I'm switching those inputs to frame-index timing (`StartFr`/`SoundFr`), which is the correct alignment space for the neural data."

## 3-b. What processing is involved in computing `input` *Time to sound cue*?

i. Signed seconds to the cue: `(SoundFr[tr] - frame_index) * dt_seconds`, positive before the cue and negative after, stored as `float32`, one value per retained bin. The reference instead interpolates the fractional cue frame onto the real `ft` timestamp axis; the AI approximates that with a constant median frame interval. Observed range is [-1762 s, +723 s] (vs. [-72 s, +73 s] for the reference), the extremes coming from the long stopped trials that were not filtered out.

ii.
```python
        time_to_sound = ((float(beh["SoundFr"][tr]) - selected_frames) * dt_seconds).astype(np.float32)
```

iii. Same as 3-a; the AI treated frame index × frame period as the alignment-consistent way to express time for imaging data. Step 133 argues the wide range is legitimate ("trials where mice paused for long periods and only the running frames were retained").

## 3-c. How is the `input` *Time to sound cue* aligned with the neural data?

i. It is evaluated at exactly the same frame indices (`mask`) used to slice the neural columns of that trial, so it has the same length as the trial's neural array and is bin-for-bin aligned.

ii.
```python
        selected_frames = frame_numbers[mask]
        ...
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
        input_trials.append(
            np.vstack([time_to_sound, day_arr, time_since_start, reward_available]).astype(np.float32)
        )
```

iii. Everything in this dataset lives on the imaging-frame grid (`ft_*` streams and `*Fr` event markers), so using the same frame mask guarantees alignment.

## 4-a. What variables in the raw data is `input` *Day of training* derived from?

i. The date embedded in the session key (`mname_YYYY_MM_DD_blk`), parsed with `datetime.strptime`. No behavior field (e.g. `days` in `Imaging_Exp_info`) is used.

ii.
```python
    subject = parts[0]
    date = datetime.strptime("_".join(parts[1:4]), "%Y_%m_%d")
```

iii. Metadata: "Calendar days since the first imaging session for each subject, used because a consistent explicit training-day count was not available for every recording." (Only some `Imaging_Exp_info` experiment types carry a `days` field, which is what the AI is referring to.)

## 4-b. What processing is involved in computing `input` *Day of training*?

i. For each mouse the earliest session date is found, and each session's value is the number of **calendar days** between that first date and the session date, as a float broadcast across all bins of every trial of the session. Range 0–92 days, mean 21.3. The reference instead uses the ordinal index of the recording day (0–7).

ii.
```python
    first_dates = {subject: min(dates) for subject, dates in per_subject_dates.items()}

    day_since_first = {}
    for key in session_keys:
        subject, date = parse_subject_and_date(key)
        day_since_first[key] = float((date - first_dates[subject]).days)
```
```python
        day_arr = np.full(mask.sum(), day_value, dtype=np.float32)
```

iii. As quoted in 4-a: the source metadata does not give an explicit training-day count for every recording, so elapsed calendar time since the mouse's first imaging session is used as a continuous proxy.

## 5-a. What variables in the raw data is `input` *Time since trial start* derived from?

i. `StartFr[tr]` (fractional corridor-entry frame), the retained frame indices, and `dt_seconds` from `ft`.

ii.
```python
        time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. Same rationale as 3-a (step 125): `StartFr` is expressed in the imaging-frame space, which is where the neural data live.

## 5-b. What processing is involved in computing `input` *Time since trial start*?

i. `(frame_index - StartFr[tr]) * dt_seconds`, seconds, positive after entry, `float32`, one value per bin. Observed range [0, 1763.9] s with median trial end at 7.1 s; the 1 % of trials longer than 75 s are the stopped-animal traversals that the reference removes.

ii.
```python
        time_since_start = ((selected_frames - float(beh["StartFr"][tr])) * dt_seconds).astype(np.float32)
```

iii. Step 133: wide ranges were checked and accepted as "trials where mice paused for long periods and only the running frames were retained".

## 5-c. How is the `input` *Time since trial start* aligned with the neural data?

i. Same frame mask as the neural slice for that trial, therefore identical length and bin-for-bin correspondence.

ii.
```python
        mask = (trial_idx == tr) & move_corr_mask
        selected_frames = frame_numbers[mask]
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. Every stream is indexed by imaging frame, so one mask aligns all of them.

## 6-a. What variables in the raw data is `input` *Reward availability* derived from?

i. `isRew`, the per-trial boolean flag marking trials run in the rewarded corridor.

ii.
```python
        reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. No explicit discussion in the trajectory; `isRew` is the direct encoding of the requested variable. The AI inspected `isRew` across experiment types early on (step 34).

## 6-b. What processing is involved in computing `input` *Reward availability*?

i. Cast to `bool` then to `float` (0.0/1.0) and broadcast across every bin of the trial. Overall 12.25 % of bins are rewarded-corridor bins.

ii.
```python
        reward_available = np.full(mask.sum(), float(bool(beh["isRew"][tr])), dtype=np.float32)
```

iii. None needed — the field is already the requested binary per-trial variable.

## 7-a. What variables in the raw data is `output` *Visual stimulus category* derived from?

i. `WallName`, the per-trial name of the corridor wall texture (e.g. `leaf1`, `leaf1_swap2`, `wood5`). `stim_id` is merged across duplicate session entries but is never used for the output label.

ii.
```python
        wall_name = str(beh["WallName"][tr])
        stim_category = category_to_idx[wall_to_category(wall_name)]
```

iii. Step 39 established that `WallName` is identical across the duplicated `_swapN` behavior entries while `stim_id`/`TrialStim` are masked differently, so `WallName` is the reliable per-trial stimulus label.

## 7-b. What processing is involved in computing `output` *Visual stimulus category*?

i. The leading alphabetic run of the wall name is extracted by regex and lower-cased, giving the texture family; anything outside `{circle, leaf, rock, wood}` raises. The category list is built by scanning every trial of every session and sorting, which yields exactly `['circle','leaf','rock','wood']`; the label index is broadcast over the trial's bins. Resulting fractions (0.311 / 0.470 / 0.085 / 0.135) closely match the reference (0.312 / 0.481 / 0.082 / 0.125).

ii.
```python
def wall_to_category(name: str) -> str:
    match = re.match(r"([A-Za-z]+)", str(name))
    if match is None:
        raise ValueError(f"Could not parse stimulus family from wall name {name!r}")
    category = match.group(1).lower()
    if category not in {"circle", "leaf", "rock", "wood"}:
        raise ValueError(f"Unexpected stimulus category {category!r} from wall name {name!r}")
    return category
```
```python
        visual_category = np.full(mask.sum(), stim_category, dtype=np.int16)
```

iii. Step 138: the converter "maps wall names to stimulus families (`circle`, `leaf`, `rock`, `wood`)". Crops and spatial shuffles (`leaf1`, `leaf2`, `leaf1_swap1`) are treated as the same visual category, matching how the paper groups textures.

## 8-a. What variables in the raw data is `output` *Licking* derived from?

i. `LickFr`, the (fractional) imaging-frame number of every lick in the session.

ii.
```python
    lick_binary = np.zeros(nfr, dtype=np.int8)
    lick_fr = np.asarray(beh["LickFr"], dtype=int)
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
    if lick_fr.size:
        lick_binary[np.unique(lick_fr)] = 1
```

iii. No explicit discussion; `LickFr` is already on the imaging-frame grid, which is the grid the output must be on.

## 8-b. What processing is involved in computing `output` *Licking*?

i. Lick frame numbers are truncated to int, clipped to the imaged range `[0, nfr)`, and a session-length binary vector is set to 1 at those frames; a bin is 1 if at least one lick fell in it. Values are then sliced per trial and stored as `int16`. Overall lick fraction 3.69 % (reference: 4.14 %; the difference comes from licks that fall on stationary frames, which this pipeline discards).

ii.
```python
        lick = lick_binary[mask].astype(np.int16)
```
```python
        "output_values": [..., ["no_lick", "lick"], ...]
```

iii. None stated beyond the implicit "represent a time of behavior as a binary time series" requirement in the instructions.

## 8-c. How is `output` *Licking* aligned with the neural data?

i. The binary lick vector is built on the full imaging-frame axis and then indexed by exactly the same `mask` used for the neural columns, so it is bin-for-bin aligned with the trial's neural array. Licks occurring on corridor frames where the animal was not running are dropped along with those frames.

ii.
```python
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
        output_trials.append(np.vstack([visual_category, lick, pos_bin, speed_bin]).astype(np.int16))
```

iii. All streams share the imaging-frame index, so a single mask aligns them.

## 9-a. What variables in the raw data is `output` *Position in corridor* derived from?

i. `ft_Pos`, the per-imaging-frame VR position in decimeters (0–40 across the texture, up to 60 through the grey space), truncated to the number of imaged frames.

ii.
```python
    ft_pos = np.asarray(beh["ft_Pos"][:nfr], dtype=np.float32)
```

iii. Step 24 inspected `ft_Pos`/`VRpos`; the paper's own interpolation code uses `ft_PosCum` on the same frame grid with 1-decimeter bins over a 6 m corridor, confirming the units.

## 9-b. What processing is involved in computing `output` *Position in corridor*?

i. Position is clipped to `[0, 39.999]` dm, divided by 10 and floored, giving an integer 0–3; stored as `int16` per bin with value names `['0-1m','1-2m','2-3m','3-4m']`. Resulting occupancy is essentially uniform (0.250 / 0.249 / 0.250 / 0.252), matching the reference.

ii.
```python
        pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. Directly follows the instruction "Position in corridor discretized into 4 equal-length, 1-m-long spatial bins"; only corridor (texture) frames are kept, so positions ≥ 40 dm should not occur and the clip is only a guard.

## 9-c. How is `output` *Position in corridor* thresholded into categories?

i. Fixed, equal-length 1 m (10 dm) thresholds at 10/20/30 dm — not data-driven quantiles.

ii.
```python
        pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```
```python
            ["0-1m", "1-2m", "2-3m", "3-4m"],
```

iii. The instruction specifies "4 equal-length, 1-m-long spatial bins", which is exactly what the division by 10 dm produces.

## 9-d. How is `output` *Position in corridor* aligned with the neural data?

i. `ft_Pos` has one value per imaging frame; it is indexed with the same trial `mask` as the neural columns, giving identical length and bin correspondence.

ii.
```python
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
        pos_bin = np.clip(np.floor(np.clip(ft_pos[mask], 0.0, 39.999) / 10.0), 0, 3).astype(np.int16)
```

iii. Shared imaging-frame grid.

## 10-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ft_RunSpeed`, the per-imaging-frame running speed (cm/s), together with `ft_CorrSpc` and `ft_move` which define the frames the quartile edges are estimated on.

ii.
```python
def compute_speed_edges(behaviors: dict[str, dict]) -> np.ndarray:
    speeds = []
    for beh in behaviors.values():
        mask = np.asarray(beh["ft_CorrSpc"], dtype=bool) & (np.asarray(beh["ft_move"]) > 0)
        if np.any(mask):
            speeds.append(np.asarray(beh["ft_RunSpeed"])[mask])
    all_speeds = np.concatenate(speeds)
    return np.quantile(all_speeds, [0.25, 0.5, 0.75]).astype(np.float32)
```

iii. No separate justification; `ft_RunSpeed` is the direct measurement, and the mask reuses the same corridor+running definition used everywhere else in the converter.

## 10-b. What processing is involved in computing `output` *Running speed*?

i. One **global** set of quartile edges is estimated once, over the running corridor frames of all 89 sessions pooled (12.39, 25.33, 40.83 cm/s), and every bin is assigned by `np.digitize` with `np.clip` to 0–3. Because non-running (≈ zero-speed) frames were already excluded from both the edge estimation and the exported data, the zero-speed tie problem the reference had to solve with a rank split does not arise, and the exported bins come out essentially exactly 25 % each (0.2493 / 0.2502 / 0.2501 / 0.2504). The reference instead computes rank-based quartiles per session.

ii.
```python
        speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
        speed_bin = np.clip(speed_bin, 0, 3)
```
```python
def describe_speed_bins(edges: np.ndarray) -> list[str]:
    return [
        f"<= {edges[0]:.2f} cm/s", f"{edges[0]:.2f}-{edges[1]:.2f} cm/s",
        f"{edges[1]:.2f}-{edges[2]:.2f} cm/s", f"> {edges[2]:.2f} cm/s",
    ]
```

iii. Metadata records the numeric edges under `running_speed_bin_edges_cm_s`; step 138 describes them as "running-speed quartiles". The instruction asks for "4 bins, each corresponding to 25% of the data", which global quantiles satisfy on the pooled dataset.

## 10-c. How is `output` *Running speed* thresholded into categories?

i. By the three pooled-dataset quartile thresholds (12.39 / 25.33 / 40.83 cm/s), applied identically to every session and trial; the bin labels in `output_values` spell out those thresholds.

ii.
```python
    speed_edges = compute_speed_edges(behaviors)
...
        speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
```

iii. As above: thresholds derived from the data rather than fixed, so each bin holds a quarter of the pooled frames.

## 10-d. How is `output` *Running speed* aligned with the neural data?

i. `ft_RunSpeed` is sampled per imaging frame and indexed with the same trial `mask` as the neural columns, so it is bin-for-bin aligned.

ii.
```python
    ft_speed = np.asarray(beh["ft_RunSpeed"][:nfr], dtype=np.float32)
...
        speed_bin = np.digitize(ft_speed[mask], speed_edges, right=False).astype(np.int16)
        neural_trials.append(selected_spk[:, mask].astype(np.float16))
```

iii. Shared imaging-frame grid.

## 11. How are minor mistakes in the data, e.g. missing data, handled?

i. (a) Behavior streams run longer than the imaging, so every per-frame stream is truncated to `nfr = selected_spk.shape[1]`. (b) Lick frames outside `[0, nfr)` are dropped. (c) `ft_trInd` is NaN outside trials, and `(trial_idx == tr)` is simply False there, so those frames never enter a trial. (d) Trials with no surviving frame are skipped. (e) Position is clipped to the valid 0–40 dm range. (f) Hard failures: a spikes/retinotopy neuron-count mismatch, a session with no selected neurons, or a session with fewer than two trials each raise and abort the whole run rather than skipping that session. Note that `compute_speed_edges` and `compute_time_bin_size_ms` operate on the untruncated behavior arrays, so a few frames past the end of imaging contribute to the global quartile edges and bin size.

ii.
```python
    nfr = selected_spk.shape[1]
    trial_idx = np.asarray(beh["ft_trInd"][:nfr])
    move_corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```
```python
    lick_fr = lick_fr[(lick_fr >= 0) & (lick_fr < nfr)]
```
```python
        if nneu_total != iarea.shape[0]:
            raise RuntimeError(
                f"Neuron count mismatch for {base_key}: spikes={nneu_total}, retinotopy={iarea.shape[0]}"
            )
```

iii. The truncation mirrors the paper's own `nfr = spk.shape[1]; beh[...][:nfr]` idiom that the AI read in `utils.py` (steps 14–15). The raise-on-anomaly choices are fail-fast guards; none of them triggered on this dataset.

## 12-a. What are the most time-consuming steps of the code?

i. Two steps dominate. (1) Reading the 405 GB of spike files, one 1.5–10 GB `.npy` per session — unavoidable I/O, the same cost the reference pays. (2) `merge_behavior_records`, which loads all 23 behavior files (6.6 GB) up front and keeps every field of every session resident for the entire run; in the logged run this startup phase took roughly 5 minutes, comparable to the ~2 minutes of the session loop (which benefited from a warm page cache). Within each session, `select_neurons_for_session` computes a mean and a variance over corridor frames for *every* neuron of every plane before discarding 99 % of them.

ii.
```python
        spk_planes = np.load(spk_path, allow_pickle=True).item()["spks"]
```
```python
        beh = np.load(os.path.join(BEH_ROOT, fname), allow_pickle=True).item()
        for key, record in beh.items():
            ...
            merged[base_key] = {k: v for k, v in record.items()}
```
```python
        corr_mean = plane[:, corr_mask].mean(axis=1)
        gray_mean = plane[:, gray_mask].mean(axis=1)
        corr_var = plane[:, corr_mask].var(axis=1)
```

iii. Step 71–72: the AI measured the data volume (`du -sh` → 405 G / 6.6 G / 170 M) before choosing its representation, and monitored throughput during the run ("roughly 8-9 sessions every 30 seconds", step 115). Step 104: it added flushed progress output specifically "so I can see exactly where the time is going".

## 12-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The per-trial loop rebuilds `(trial_idx == tr) & move_corr_mask` for every trial, a full scan of the frame axis per trial — O(ntrials × nframes) where one `np.argsort`/grouping pass would do. (2) The nested plane × region loops in `select_neurons_for_session` build Python lists of candidate indices and then a `sorted(zip(...))` over the chosen rows, plus a `defaultdict` of Python ints per selected neuron. (3) `collect_subject_info` re-parses every session key twice. None of these matter next to the spike-file I/O.

ii.
```python
    for tr in range(int(beh["ntrials"])):
        mask = (trial_idx == tr) & move_corr_mask
```
```python
        chosen = sorted(zip(plane_arr[top], row_arr[top]), key=lambda x: (x[0], x[1]))
        for plane_idx, row_idx in chosen:
            selected_rows_by_plane[int(plane_idx)].append(int(row_idx))
            selected_region_idx_by_plane[int(plane_idx)].append(region_to_idx[region])
```

iii. Not discussed in the trajectory; the AI's stated efficiency concern was memory/size of the export, not CPU time.

## 12-c. What processing does the code repeat multiple times?

i. The behavior data are traversed in full four separate times before any session is processed: once to merge, once in `compute_speed_edges`, once in `compute_time_bin_size_ms`, and once to enumerate the stimulus categories (`wall_to_category` is therefore applied to all ~38 k trial wall names twice). The corridor+running mask `ft_CorrSpc & (ft_move > 0)` is recomputed three times per session — once per plane inside `select_neurons_for_session` and again in `build_trial_data`. `parse_subject_and_date` is called repeatedly for the same key.

ii.
```python
    behaviors = merge_behavior_records()
    subjects, subject_to_idx, day_since_first = collect_subject_info(session_keys)
    speed_edges = compute_speed_edges(behaviors)
    time_bin_size = compute_time_bin_size_ms(behaviors)
    all_categories = sorted({wall_to_category(wall_name)
                             for beh in behaviors.values()
                             for wall_name in np.asarray(beh["WallName"]).tolist()})
```
```python
        corr_mask = np.asarray(beh["ft_CorrSpc"][:nfr], dtype=bool) & (np.asarray(beh["ft_move"][:nfr]) > 0)
```

iii. Not discussed. These repeats are cheap relative to the spike I/O, and holding all behavior in memory is what makes the repeated passes possible at all.

## 12-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `stim_id` NaN-filling merge across duplicate session entries is computed for every duplicated session and then never read — the stimulus label comes from `WallName`. (2) `compute_time_bin_size_ms` computes a global median bin size used only as a metadata string; the timing inputs use a separately computed per-session `dt_seconds`. (3) `corr_var` is computed for every neuron of every plane purely to rank candidates, and the scores are then thrown away. (4) Means over grey-space frames are computed for all neurons, including the ~99 % that are discarded by the cap anyway. (5) `merge_behavior_records` copies and retains every behavior field (including the large `VRpos`/`VRposTime`/`ft_*` arrays of all 89 sessions) although only ~12 fields per session are used. (6) `describe_speed_bins` string formatting and the `session_keys` metadata list are cosmetic.

ii.
```python
            stim_old = np.asarray(merged[base_key]["stim_id"], dtype=float)
            stim_new = np.asarray(record["stim_id"], dtype=float)
            if stim_old.shape == stim_new.shape:
                fill = np.isnan(stim_old) & ~np.isnan(stim_new)
                stim_old[fill] = stim_new[fill]
                merged[base_key]["stim_id"] = stim_old
```
```python
        corr_var = plane[:, corr_mask].var(axis=1)
```

iii. Step 61 motivates the `stim_id` merge ("collapsing those to one recording apiece … with the richest mapping"), i.e. it was intended as a safeguard for stimulus labelling that the final implementation did not end up needing. The AI did free per-session arrays explicitly (`del spk_planes, ...; gc.collect()`) to keep peak memory bounded.
