# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every NWB file under `data/` is discovered with a single recursive glob over the pattern `sub-*/sub-*_behavior+ophys.nwb`, sorted for reproducible ordering. Each file is one session. Files are read directly with `h5py` (raw HDF5 group paths) rather than `pynwb`, in one pass; there is no separate survey pass. Per session the AI reads the behavior group `processing/behavior/BehavioralTimeSeries` (position, speed, lick, environment, reward_zone, `trial number`, trial_start, teleport, position timestamps, Reward timestamps), the ophys group `processing/ophys` (`Deconvolved/plane<N>/data`, `ImageSegmentation/PlaneSegmentation/iscell`, `planeIdx`), and the top-level `identifier` (used for the scene/reward-zone metadata). All 152 files are processed; nothing is subsampled in `--full` mode. `--sample` takes `files[:2]`.

ii.
```python
DATA_ROOT = Path("data")

def list_nwb_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if sample:
        return files[:2]
    return files
```
```python
def process_session(path: Path) -> tuple[dict, list[dict]]:
    session_id = path.stem.replace("_behavior+ophys", "")
    with h5py.File(path, "r") as f:
        identifier = decode_bytes(f["identifier"][()])
        scene = identifier.split("/")[-1]
        scene_info = parse_scene(scene)

        subject = path.parent.name.replace("sub-", "")
        session_name = path.stem

        behavior_group = f["processing/behavior/BehavioralTimeSeries"]
        ophys_group = f["processing/ophys"]

        position = behavior_group["position/data"][()].astype(np.float32)
        speed = behavior_group["speed/data"][()].astype(np.float32)
        lick = behavior_group["lick/data"][()].astype(np.float32)
        environment = behavior_group["environment/data"][()].astype(np.float32)
        reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
        trial_number = behavior_group["trial number/data"][()].astype(np.float32)
        trial_start = behavior_group["trial_start/data"][()].astype(np.float32)
        teleport = behavior_group["teleport/data"][()].astype(np.float32)
        timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
        reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
```
```python
    for idx, path in enumerate(nwb_files, start=1):
        session_start = time.time()
        session, examples = process_session(path)
```

iii. From CONVERSION_NOTES Step 2: "`data/` contains one NWB file per subject-session plus `dandiset.yaml`. File layout is `data/sub-<mouse>/sub-<mouse>_ses-<NN>_behavior+ophys.nwb`. There are 152 NWB files total across 11 subjects." Step 4 cross-checks this against the paper: "Consistent: `10 mice * 14 + m11 * 12 = 152`." Step 6 justifies the reader choice: "Reads NWB directly with `h5py` for speed" and "Uses `h5py` instead of higher-level NWB objects for the main conversion path." Key Decision 12: "Sort sessions by file path (`subject`, then `session`): This gives stable ordering for `subjects`, `subject_idx`, and reproducible outputs."

## 1-b. How are the data split into subjects?

i. Subject identity is parsed from the parent directory name of each NWB file (`sub-m11` → `m11`). The unique subject set is built from the sessions that survived processing, sorted alphabetically, and `subject_idx` indexes each session into that list. 11 subjects result (`m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7`, in sorted-string order).

ii.
```python
subject = path.parent.name.replace("sub-", "")
```
```python
def build_dataset(processed_sessions: list[dict]) -> dict:
    subjects = sorted({sess["subject"] for sess in processed_sessions})
    subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_to_idx[sess["subject"]] for sess in processed_sessions],
            dtype=np.int64,
        ),
```

iii. CONVERSION_NOTES Step 2: "Subject IDs: `m3, m4, m7, m11, m12, m13, m14, m15, m17, m18, m19`", "Subjects | 11". Step 3 ties this to the paper: "11 switch-task mice in the main dataset — 'counterbalanced across mice (n = 11 mice)'". Step 4: "Released `data/` contains the 11 switch-task mice only." Key Decision 12 covers the sorted/deterministic ordering. Step 5 planned sanity check: "Subject/session consistency check: confirm subject list, session ordering, and `subject_idx` match file paths exactly."

## 1-c. How are the data split into sessions?

i. One NWB file = one session. No merging across days or across mice; no cross-session neural alignment is attempted. Sessions are ordered by sorted file path (subject directory, then `ses-NN`). A session is dropped entirely only if fewer than 2 trials survive filtering (this never triggered — all 152 sessions are retained).

ii.
```python
files = sorted(DATA_ROOT.glob("sub-*/sub-*_behavior+ophys.nwb"))
```
```python
        session_name = path.stem            # e.g. "sub-m11_ses-03_behavior+ophys"
```
```python
        session, examples = process_session(path)
        if len(session["neural"]) < 2:
            print(
                f"[{idx}/{len(nwb_files)}] Skipping {path.name}: "
                f"only {len(session['neural'])} valid trials after filtering."
            )
            continue

        processed_sessions.append(session)
```

iii. CONVERSION_NOTES Step 2: "one NWB file per subject-session"; Step 4 "Session count | One imaging session/day; `m11` started on day 3 | 152 sessions total | 14 task days ... | Consistent". Step 9 reports 152 sessions in the converted data. The `< 2 trials` guard is motivated by the target-format requirement quoted in the instructions ("There needs to be at least two trials within each session in order to evaluate the decoder performance").

## 1-d. How are the data split into trials?

i. Trials are the frame intervals `[trial_start, teleport)`. `find_complete_trial_bounds` walks the indices where `trial_start > 0` and, for each, takes the *next* index where `teleport > 0` that is strictly greater than the start, consuming teleports greedily so each teleport is used once. Only "complete" trials — those with both a start marker and a following teleport marker — are emitted; a trailing start with no teleport is discarded. The teleport sample itself is excluded from the trial (half-open interval), so inter-trial teleport periods are never part of any trial.

ii.
```python
def find_complete_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    teleports = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    teleport_idx = 0

    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        if teleport_idx >= len(teleports):
            break
        stop = teleports[teleport_idx]
        if stop > start:
            bounds.append((int(start), int(stop)))
        teleport_idx += 1

    return bounds
```
```python
    trial_bounds = find_complete_trial_bounds(trial_start, teleport)
    ...
    for start, stop in trial_bounds:
        ...
        trial_slice = slice(start, stop)
```

iii. CONVERSION_NOTES Step 1: "`dff` keeps fluorescence only from `trial_start_inds` to `teleport_inds` with slices `start-1:stop-1`; teleport samples are excluded unless explicitly requested otherwise" and "`get_timeseries_data` ... loops over trial epochs defined by `trial_start_inds` and `teleport_inds`". Step 4: "Reference code uses `trial_start_inds` and `teleport_inds` | NWB stores continuous `trial_start` and `teleport` vectors, no trials table | Use `trial_start > 0` and `teleport > 0` to define trial epochs; these match one-for-one in all files." Key Decisions 4 and 5: "Segment trials from `trial_start` to `teleport` and exclude teleport periods: This matches the reference code's trial-epoch logic and the target requirement to align to trial start"; "Keep only complete trials with explicit start and end markers: This avoids fabricating incomplete boundaries; for example the clipped first `m11` day-3 trial will be excluded." Step 10 edge-case check: "`m11` day 3 has 81 raw trial numbers but only 80 complete `trial_start`/`teleport` trial epochs; converter correctly keeps the 80 complete trials."

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level filters, applied in this order:
1. **Too-short trials**: `stop - start < MIN_TRIAL_FRAMES` (5 frames) → dropped.
2. **Lick-sensor failure**: a trial is dropped if more than 35% of its in-trial frames have cumulative lick count `> 2` (`LICK_ERROR_FRACTION = 0.35`). This drops 69 of 12,216 complete trials (0.56%).
3. **Too few valid frames after per-frame masking**: after intersecting the per-frame validity masks (finite position/speed/lick/environment/timestamp, `position > -100`, finite neural), a trial needs ≥ 5 surviving frames or it is dropped.

Additionally, a *session* is dropped if fewer than 2 trials survive (never triggered). Final: 12,147 kept trials.

ii.
```python
MIN_TRIAL_FRAMES = 5
LICK_ERROR_FRACTION = 0.35

def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
```
```python
    for start, stop in trial_bounds:
        if stop - start < MIN_TRIAL_FRAMES:
            dropped_missing += 1
            continue
        ...
        trial_meta.append({..., "lick_error": has_lick_sensor_error(lick[start:stop])})

    for meta in trial_meta:
        if meta["lick_error"]:
            dropped_lick += 1
            continue
        ...
        frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
        if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
            dropped_missing += 1
            continue
```

iii. CONVERSION_NOTES Step 1 notes the reference-code rule: "marks a trial's lick samples as invalid if more than 35% of samples exceed cumulative lick count 2". Step 3 records the paper/code discrepancy: "Methods text says bad lick trials were detected when `>30%` of frame samples had cumulative lick count `>2`; `glmUtils.get_timeseries_data` in the code uses a threshold of `>35%`", and Step 4 resolves it: "Prefer the released code threshold (`35%`) when reproducing continuous-sample processing, and document the text/code mismatch." Key Decision 10: "Drop lick-artifact trials entirely rather than leaving NaNs: The validator forbids NaNs, and the reference code already treats these trials as invalid for licking analyses." Step 9 cross-checks the count: "Kept trials total (`12147`) reflects dropping lick-artifact trials using the code-consistent `>35%` threshold" against the paper's "81 out of 12,376 trials removed across 11 switch mice" / "~0.65% of all imaged trials".

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrices are taken directly from the NWB `processing/ophys/Deconvolved/plane<N>/data` arrays (shape `(T, n_ROI)`), subset to `iscell[:,0] == 1` ROIs and transposed to `(n_neurons, n_timepoints)`. The raw `Fluorescence` (F) and `Neuropil` (Fneu) arrays are read *nowhere* in the conversion script — no dF/F is computed. For two-plane sessions (`m17`, `m18`) the per-plane `Deconvolved` matrices are stitched back into pooled-ROI column order using `planeIdx`.

ii.
```python
        iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
        accepted_mask = np.asarray(iscell[:, 0]) == 1
        accepted_idx = np.flatnonzero(accepted_mask)
        plane_idx_all = ophys_group["ImageSegmentation/PlaneSegmentation/planeIdx"][()].astype(np.int16)
        plane_idx = plane_idx_all[accepted_idx]

        deconv_shape_t = None
        deconv = None
        planes = sorted(int(x) for x in np.unique(plane_idx_all))
        for plane in planes:
            plane_roi_idx = np.flatnonzero(plane_idx_all == plane)
            accepted_total_idx = plane_roi_idx[accepted_mask[plane_roi_idx]]
            accepted_local_idx = np.flatnonzero(accepted_mask[plane_roi_idx])
            plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
            plane_data = plane_data.astype(np.float32, copy=False)

            if deconv_shape_t is None:
                deconv_shape_t = plane_data.shape[0]
                deconv = np.empty((deconv_shape_t, accepted_idx.size), dtype=np.float32)

            dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
            deconv[:, dest_cols] = plane_data
```
```python
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "**Use NWB deconvolved traces directly**: The paper's RR decoder uses deconvolved calcium events, and the NWB `Deconvolved` data are the released equivalent of reference `sess.timeseries['events']`." Key Decision 3: "**Do not recompute dF/F from fluorescence**: The release already provides frame-aligned deconvolved traces; recomputation would add unnecessary divergence from the released data product." Step 4 states the same assumption: "Map NWB `Deconvolved` to reference `events`". Step 10 records the multi-plane fix: "multi-plane NWB sessions store `Deconvolved/plane0` and `Deconvolved/plane1` separately, while `iscell` indexes pooled ROIs. Resolution: ... reconstruct the accepted-cell deconvolved matrix by concatenating per-plane series into pooled ROI order before trial slicing."

## 2-b. How is the `neural` data processed?

i. Essentially no processing beyond selection, stitching and slicing. The stored `Deconvolved` values are cast to `float32`, restricted to accepted ROIs, sliced to the trial window, masked with the per-frame validity mask, and transposed. There is no neuropil subtraction (`F - 0.7*Fneu`), no per-trial maximin baseline (Gaussian σ=15, 300-sample min then max filter), no `(F - baseline)/|baseline|` normalization, no 2-sample Gaussian smoothing, and no OASIS deconvolution at `tau=0.7` with the per-plane frame rate — i.e. none of the paper's `preprocessing.dff(..., deconvolve=True)` pipeline is reproduced. No normalization, z-scoring or rescaling is applied either, so the values delivered to the decoder are on the raw-fluorescence scale (observed range 0 to ~1.5e4).

ii.
```python
            plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
            plane_data = plane_data.astype(np.float32, copy=False)
            ...
            dest_cols = np.searchsorted(accepted_idx, accepted_total_idx)
            deconv[:, dest_cols] = plane_data
```
```python
        frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
        ...
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
        ...
        neural_trials.append(neural_trial)
```

iii. CONVERSION_NOTES Key Decision 3: "Do not recompute dF/F from fluorescence: The release already provides frame-aligned deconvolved traces; recomputation would add unnecessary divergence from the released data product." Step 4 "Final understanding": "The released NWB files are processed, frame-aligned session-level data equivalent in content to what the repo stores inside `sess`. Conversion should therefore avoid re-deriving alignment from raw files and instead preserve the shared frame-aligned time base already present in NWB." Step 10 Check 5 claims consistency with the reference: "loading/curation matches `iscell`-filtered `sess` content". (The AI did document the paper's actual dF/F recipe in Step 3 — "dF/F baseline computed independently within each trial using a maximin procedure with a 20 s sliding window ... deconvolution performed with OASIS" — but gave no reason for not applying it beyond the assumption that `Deconvolved` already is that signal.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single neuron filter: `iscell[:,0] == 1` (suite2p manual curation), giving 138,678 cells across 152 sessions (155–2341 per session). The paper's additional exclusion of putative interneurons (cells whose dF/F correlates with running speed at Pearson *r* > 0.5) is **not** applied. There is also a per-frame neural validity mask (`np.all(np.isfinite(deconv), axis=1)`) that would drop non-finite frames, but it is a no-op on this data.

ii.
```python
        iscell = ophys_group["ImageSegmentation/PlaneSegmentation/iscell"][()]
        accepted_mask = np.asarray(iscell[:, 0]) == 1
        accepted_idx = np.flatnonzero(accepted_mask)
```
```python
    valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```
```python
        "brain_regions": ["CA1"],
        "brain_region_idx": [sess["brain_region_idx"] for sess in processed_sessions],
```
(there is no `is_putative_interneuron` / speed-correlation code anywhere in `convert_data.py`)

iii. CONVERSION_NOTES Key Decision 2: "**Filter neurons with `iscell[:,0] == 1` only**: This is the shared, explicit curated-cell mask available in NWB and matches the baseline neuron curation used before downstream analyses." Step 4 explicitly noticed the issue but deferred it: "Code uses curated `iscell`; later analyses may also exclude putative interneurons | `iscell[:,0]==1` gives 155-2341 accepted cells/session; only three `m18` sessions exceed 2172 | Methods report 155-2172 putative pyramidal neurons/session | Accept `iscell[:,0]==1` as the primary shared-data curation. The slight excess occurs only in pooled multi-plane `m18` sessions and likely reflects release-version differences or later exclusions not baked into NWB." And: "If interneuron exclusion is needed, use the paper/dayData threshold `0.5`, not the helper's default `0.3`" — a conditional that was never acted on.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start requires no extra work: the neural matrix is sliced with exactly the same frame indices `[trial_start, teleport)` used for every behavioral stream, so frame 0 of each trial's neural matrix is the trial-start frame. `metadata['temporal_alignment_event'] = 'trial_start'`, `off_start = 0.0`, `off_end = None`. No pre-trial window is included and no interpolation/resampling is done.

ii.
```python
        trial_slice = slice(start, stop)
        position_trial = position[trial_slice][frame_mask]
        speed_trial = speed[trial_slice][frame_mask]
        lick_trial = lick[trial_slice][frame_mask]
        time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```
```python
            "temporal_alignment_event": "trial_start",
            "off_start": 0.0,
            "off_end": None,
            ...
            "trial_definition": "frames from trial_start inclusive to teleport exclusive",
```

iii. CONVERSION_NOTES Step 2: "the behavior series length matches the deconvolved series length, so the aligned sample grid needs to be taken from timestamps/array length". Step 4 final understanding: "Conversion should therefore avoid re-deriving alignment from raw files and instead preserve the shared frame-aligned time base already present in NWB." Key Decision 4: segmentation from `trial_start` to `teleport` "matches ... the target requirement to align to trial start." Step 10 verified this with raw spot-checks: "Raw-to-converted neural sanity checks with `np.allclose()`: passed on 5 sessions spanning single-plane and multi-plane data."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging/behavior frame grid is kept — no rebinning, no resampling, no smoothing, no downsampling. `metadata['time_bin_size']` is computed as the median across sessions of the per-session median inter-timestamp interval, in ms: 64.48 ms (~15.5 Hz). The NWB ophys `rate` attribute is deliberately not used, because on the 28 two-plane sessions it reports the 31.0 Hz scanner rate rather than the ~15.5 Hz per-plane rate.

ii.
```python
        "time_bin_size_ms": float(np.median(np.diff(timestamps)) * 1000.0),
```
```python
    median_bin_ms = float(
        np.median([sess["summary"]["time_bin_size_ms"] for sess in processed_sessions])
    )
    ...
            "time_bin_size": median_bin_ms,
```

iii. CONVERSION_NOTES Step 2: "Behavior timestamps have spacing about `0.0645 s` (~15.5 Hz). Some multiplane files report a `Deconvolved` rate attribute of `31.015625`, but the behavior series length matches the deconvolved series length, so the aligned sample grid needs to be taken from timestamps/array length rather than trusting the ophys `rate` attribute blindly." Step 4: "Use behavior timestamps / shared sample length as the aligned time base (~15.5 Hz effective). Do not trust the NWB ophys `rate` attribute blindly in multi-plane files." Key Decision 7: "Use actual timestamps for time-from-start and nominal frame size for metadata: Behavior timestamps are the reliable aligned time base, especially in multi-plane sessions where the ophys `rate` attribute can be misleading."

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From `processing/behavior/BehavioralTimeSeries/position/timestamps` (float64 seconds). All behavioral series in these files share one timestamp vector, so the choice of `position` as the carrier is arbitrary but equivalent.

ii.
```python
        timestamps = behavior_group["position/timestamps"][()].astype(np.float64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "behavior timestamps within each trial → `input[0]` (`time_from_trial_start_sec`) | `timestamps[start:stop] - timestamps[start]` | frame-aligned continuous sampling in `glmUtils.get_timeseries_data` | Time-varying continuous input." Key Decision 7: "Behavior timestamps are the reliable aligned time base".

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The trial-start frame's timestamp is subtracted from the trial's timestamps, then cast to float32. The reference point is `timestamps[start]` — the trial-start frame itself — so the first element is exactly 0.0 (and remains the correct offset even if the frame mask were to drop the first frame). Range across the full dataset: [0.0, 216.5] s.

ii.
```python
        time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
        ...
        input_trial = np.vstack(
            [
                time_trial.astype(np.float32),
                ...
            ]
        )
```

iii. CONVERSION_NOTES Step 5 mapping: "`timestamps[start:stop] - timestamps[start]`". Step 5 planned sanity check: "Input time-base check: confirm converted `time_from_trial_start_sec` equals raw behavior timestamps minus trial-start timestamp for several trials using `np.allclose()`", reported passing in Step 10 ("Raw-to-converted input sanity checks with `np.allclose()`: passed for `time_from_trial_start_sec` on the same sessions"). Step 10 also sanity-checks the outlier: "Longest kept trial is session `sub-m4_ses-04`, trial 39, `T=3359`, `216.5 s`; raw position still spans only `0.18-449.14 cm` with no teleport samples, so this is a genuine slow/paused track traversal rather than a segmentation bug."

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: the same `trial_slice` and the same `frame_mask` are applied to the timestamp vector and to the neural matrix, so element *t* of the time vector and column *t* of the neural matrix are the same imaging frame. No interpolation or offset correction is applied. The AI relies on the NWB file already having behavior aligned to imaging frames (both arrays have the same length, apart from 10 sessions where the neural array has one extra trailing frame that is never inside a trial).

ii.
```python
        frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
        ...
        time_trial = timestamps[trial_slice][frame_mask] - timestamps[start]
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 2: "the behavior series length matches the deconvolved series length". Step 4: "The released NWB files are processed, frame-aligned session-level data equivalent in content to what the repo stores inside `sess`." Step 10: "The raw-check script hit a benign one-frame mismatch between one plane's deconvolved matrix and the behavior array length. The converter never used that extra trailing frame because trial bounds come from behavior."

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. From the `environment` behavior time series (`processing/behavior/BehavioralTimeSeries/environment/data`).

ii.
```python
        environment = behavior_group["environment/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping: "`processing/behavior/BehavioralTimeSeries/environment/data` → `input[1]` (`environment`) | Take per-trial modal value in `{0,1}` and repeat across timepoints | `behavior.get_trial_types` (`morph`) | Binary per-trial input, repeated across time." Step 1 notes from the reference README: "`morph` is binary environment identity: `0=Env1`, `1=Env2`."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, non-finite values and negative sentinel values (`-1`, which occurs before VR/imaging sync) are removed, the median of the remainder is taken, and the result is thresholded at 0.5 to give a 0/1 scalar. That scalar is then broadcast across every timepoint of the trial as input row 1. (Despite the notes saying "modal", the implementation uses the median; on a variable that is constant within a trial the two agree.)

ii.
```python
def env_to_binary(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid environment values in trial.")
    return int(np.round(np.median(values)) > 0.5)
```
```python
        env_bin = env_to_binary(environment[start:stop])
        ...
        input_trial = np.vstack(
            [
                time_trial.astype(np.float32),
                np.full(time_trial.shape, meta["environment"], dtype=np.float32),
                ...
            ]
        )
```

iii. CONVERSION_NOTES Step 2: "Observed sentinel values match the code documentation: before valid synchronization, `environment` and `trial number` can be `-1`" — motivating the `values >= 0` guard. Key Decision 6: "**Represent all inputs and outputs as 2D `(d, T)` arrays**: Although some labels are per-trial, repeating them across timepoints makes dimensions uniform and matches `train_decoder.py` well." Step 5 mapping cites `behavior.get_trial_types` (`morph`) as the reference function.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the stored `trial number` behavior time series (`processing/behavior/BehavioralTimeSeries/trial number/data`), not from a loop counter. Each trial gets the modal value of that series over its own frame window.

ii.
```python
        trial_number = behavior_group["trial number/data"][()].astype(np.float32)
        ...
        trial_num = modal_trial_number(trial_number[start:stop])
```

iii. CONVERSION_NOTES Step 5 mapping: "`processing/behavior/BehavioralTimeSeries/trial number/data` or trial index → `input[2]` (`trial_number`) | Use per-trial integer index (0-based within session), repeated across timepoints | `glmUtils.get_timeseries_data` (`trial_ids`) | Prefer complete-trial order used in segmentation." Step 4 flags the count discrepancy it resolves: "12216 trials from `trial_start`, 12217 from unique valid `trial number` because one session (`m11` day 3) has a clipped first start marker". Step 10 edge-case check: "`m11` day 3 has 81 raw trial numbers but only 80 complete `trial_start`/`teleport` trial epochs; converter correctly keeps the 80 complete trials and starts at trial number 0."

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Non-finite and negative (`-1` sentinel) samples are removed, remaining values are rounded to integers, and the most frequent value (`np.bincount(...).argmax()`) is taken as the trial's number. That integer is broadcast across the trial as input row 2, and is also the key used for the switch-trial rule and for the previous-trial-outcome lookup. Because the stored trial number is used rather than a post-filtering counter, the numbering stays anchored to the true session trial index even where lick-artifact trials have been dropped. Observed range 0–99.

ii.
```python
def modal_trial_number(values: np.ndarray) -> int:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    values = np.rint(values).astype(np.int64)
    values = values[values >= 0]
    if values.size == 0:
        raise ValueError("No valid trial numbers in trial.")
    counts = np.bincount(values)
    return int(np.argmax(counts))
```
```python
                np.full(time_trial.shape, trial_num, dtype=np.float32),
```

iii. CONVERSION_NOTES Step 2: "before valid synchronization, `environment` and `trial number` can be `-1`" (motivating the sentinel filter). Key Decision 6 covers broadcasting a per-trial label across timepoints. Step 5 mapping: "Prefer complete-trial order used in segmentation."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. Indirectly, from the same source as `output` *Reward outcome*: the `Reward` time series' own timestamps (`processing/behavior/BehavioralTimeSeries/Reward/timestamps`) combined with the `reward_zone` behavior series. A per-trial outcome is computed for every complete trial (including those later dropped for lick-sensor error) and stored in a dict keyed by trial number; the previous trial's entry is then looked up.

ii.
```python
        reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
        reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```
```python
        reward_outcome = reward_outcome_for_trial(
            reward_timestamps=reward_timestamps,
            t_start=float(timestamps[start]),
            t_stop=float(timestamps[stop]),
            reward_zone_segment=reward_zone[start:stop],
        )
        reward_by_trial_number[trial_num] = reward_outcome
```

iii. CONVERSION_NOTES Step 5 mapping: "prior trial reward outcome → `input[3]` (`previous_trial_rewarded`) | Compute from previous complete trial, `0` for first kept trial in a session; repeat across timepoints | derived from `behavior.get_trial_types` logic | Binary per-trial input, repeated across time."

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `reward_by_trial_number.get(trial_num - 1, 0)` — the outcome of the trial whose stored trial number is one less. Because the dict is populated over all complete trials before the emit loop runs, a trial that follows a lick-dropped trial still gets that dropped trial's true outcome rather than skipping to an earlier trial. If no previous trial exists (first trial of the session, or the previous trial number is absent), the value defaults to 0 = omitted. The scalar is broadcast across the trial's timepoints as input row 3.

ii.
```python
    for meta in trial_meta:
        ...
        trial_num = meta["trial_number"]
        prev_outcome = reward_by_trial_number.get(trial_num - 1, 0)
        ...
        input_trial = np.vstack(
            [
                time_trial.astype(np.float32),
                np.full(time_trial.shape, meta["environment"], dtype=np.float32),
                np.full(time_trial.shape, trial_num, dtype=np.float32),
                np.full(time_trial.shape, prev_outcome, dtype=np.float32),
            ]
        )
```

iii. CONVERSION_NOTES Key Decision 11: "**Set first-trial previous outcome to 0**: There is no previous within-session trial, so `0` is the least assumption-laden sentinel compatible with the requested binary variable." Key Decision 6 covers the per-timepoint broadcast. The instruction spec is "Previous trial outcome (binary, omitted = 0, rewarded = 1, per trial)".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From two sources: (1) the `position` behavior time series, and (2) the trial's reward-zone label, which is derived from the session's `identifier` scene string (e.g. `Env1_LocationC`, `Env1_LocationC_to_A`, `Env1_C_to_Env2_B`) parsed by regex, combined with the trial number and a hard-coded switch at trial 30. The label is mapped to fixed coordinates A = 80–130 cm, B = 200–250 cm, C = 320–370 cm. The `reward_zone` behavior series is *not* used to locate the zone (it is only used as a gate for reward outcome).

ii.
```python
ZONE_TO_CODE = {"A": 0, "B": 1, "C": 2}
ZONE_TO_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
SWITCH_TRIAL = 30

SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")

def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        assert scene_info.after_zone is not None
        return scene_info.after_zone
    return scene_info.before_zone
```
```python
        identifier = decode_bytes(f["identifier"][()])
        scene = identifier.split("/")[-1]
        scene_info = parse_scene(scene)
        ...
        zone_label = zone_for_trial(scene_info, trial_num)
        zone_coords = ZONE_TO_COORDS_CM[zone_label]
```

iii. CONVERSION_NOTES Step 1: "`get_reward_zones` | `code/src/reward_relative/behavior.py` | Maps scene names to reward zone coordinates and labels (`A`, `B`, `C`), including switch days with a default change at trial 30 unless `sess.change_reward_trial` overrides it." Key Decision 8: "**Use paper/code reward-zone semantics rather than inferring from sparse `reward_zone` events alone**: The `reward_zone` series marks zone entry events, not the full zone extent, so zone A/B/C must come from session condition metadata." Step 3 records the coordinates from the paper: "Reward zones | A: 80-130 cm, B: 200-250 cm, C: 320-370 cm | 'zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm'" and "Reward switch point | after 30 trials on each switch day | 'Each switch occurred after 30 trials'". Step 4: "Code-style scene parsing still works because switch names contain `A_to/B_to/C_to` patterns and the final character identifies the destination zone."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Per timepoint, the signed distance to the nearest edge of the trial's reward zone: `position - zone_start` when before the zone (negative), exactly `0` when inside `[zone_start, zone_end]`, and `position - zone_end` when past it (positive). Computed vectorized over the trial's masked position vector, then discretized (see 7-c).

ii.
```python
def signed_distance_to_zone(position_cm: np.ndarray, zone_start: float, zone_end: float) -> np.ndarray:
    distance = np.zeros_like(position_cm, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```
```python
        zone_start, zone_end = meta["zone_coords"]
        distance_trial = signed_distance_to_zone(position_trial, zone_start, zone_end)
```

iii. CONVERSION_NOTES Step 5 mapping: "`position` + trial-specific reward-zone start/end → `output[0]` (`distance_to_reward_zone`) | Signed nearest distance to zone: `<start => pos-start`, `inside => 0`, `>end => pos-end`, then discretize to 7 bins | reward-zone semantics from `behavior.get_reward_zones`; relative-position logic from `glmUtils.get_timeseries_data`". The decoder spec says "Distance to any location in the reward zone", which the AI reads as distance-to-nearest-point-in-zone (hence 0 inside).

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven categories assigned by explicit boolean masks rather than `np.digitize`, with a sentinel `-1` initialization and a raise if any element is left unassigned: `<-50` → 0; `[-50,-10)` → 1; `[-10,0)` → 2; `==0` → 3; `(0,10]` → 4; `(10,50]` → 5; `>50` → 6. The "exactly 0" class is the in-zone class. Resulting full-dataset distribution: [0.251, 0.102, 0.073, 0.238, 0.021, 0.072, 0.243].

ii.
```python
def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, -1, dtype=np.int16)
    out[distance_cm < -50] = 0
    out[(distance_cm >= -50) & (distance_cm < -10)] = 1
    out[(distance_cm >= -10) & (distance_cm < 0)] = 2
    out[distance_cm == 0] = 3
    out[(distance_cm > 0) & (distance_cm <= 10)] = 4
    out[(distance_cm > 10) & (distance_cm <= 50)] = 5
    out[distance_cm > 50] = 6
    if np.any(out < 0):
        raise ValueError("Failed to discretize reward-zone distance.")
    return out
```
```python
        output_trial = np.vstack(
            [
                discretize_distance(distance_trial),
                ...
```
```python
            ["lt_-50", "-50_to_-10", "-10_to_0", "0", "0_to_10", "10_to_50", "gt_50"],
```

iii. CONVERSION_NOTES Step 5 mapping: "Bins: `<-50`, `[-50,-10]`, `[-10,0)`, `0`, `(0,10]`, `(10,50]`, `>50` cm" — taken verbatim from the Decoder Task spec. Step 7: "discretized outputs transition at plausible positions/times with no visible off-by-one shift". The explicit `raise` is the AI's own guard that no sample falls outside the seven classes.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices as the neural data: `position[trial_slice][frame_mask]`, i.e. identical `trial_slice` and identical `frame_mask` as `deconv[trial_slice][frame_mask]`. No shift, interpolation or resampling.

ii.
```python
        trial_slice = slice(start, stop)
        position_trial = position[trial_slice][frame_mask]
        ...
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. CONVERSION_NOTES Step 4: "The released NWB files are processed, frame-aligned session-level data"; Step 10 Check 4: "Raw-to-converted output sanity checks with `np.allclose()`: passed for discretized outputs on the same sessions [m11, m17, m18, m3, m4]." Step 7 processing-plot review: "reward-zone shading aligns with the plateau / crossing region in position; reward-event markers appear within the active reward zone on rewarded trials ... no visible off-by-one shift."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavior time series (`processing/behavior/BehavioralTimeSeries/position/data`), in cm along the 450 cm VR corridor.

ii.
```python
        position = behavior_group["position/data"][()].astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 mapping: "`position` → `output[1]` (`absolute_position_bin`) | Discretize absolute track position on `[0,450]` into 5 equal bins | task description in paper; `glmUtils` position handling | Time-varying categorical output." Step 3: "450 cm linear VR track in two environments".

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. The raw position is clipped to `[0, nextafter(450, 0)]` (i.e. `[0, 450)`) so that the handful of samples marginally outside the track (observed −1.4 to +450.8 cm within trials) land in the end bins, then divided by 90 and floored; a final `bins[bins > 4] = 4` guard catches anything above. No smoothing or unwrapping.

ii.
```python
def discretize_absolute_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
    return bins
```
```python
        output_trial = np.vstack(
            [
                discretize_distance(distance_trial),
                discretize_absolute_position(position_trial),
                ...
```

iii. CONVERSION_NOTES Step 5 mapping: "Discretize absolute track position on `[0,450]` into 5 equal bins", taken from the Decoder Task spec ("Discretized into 5 equal-sized bins spanning the 450 cm track"). Step 3 gives the 450 cm track length from the Methods. The clip is the AI's handling of samples marginally off-track (see also Step 2: "`position` can be `-500`", which the per-frame `position > -100` mask removes at the frame level).

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track: 0 = `<90`, 1 = `90–180`, 2 = `180–270`, 3 = `270–360`, 4 = `≥360`. Implemented as `floor(clip(position, 0, 450⁻)/90)`. Full-dataset distribution: [0.212, 0.177, 0.231, 0.226, 0.154].

ii.
```python
    clipped = np.clip(position_cm, 0.0, np.nextafter(450.0, 0.0))
    bins = np.floor(clipped / 90.0).astype(np.int16)
    bins[bins > 4] = 4
```
```python
            ["bin0", "bin1", "bin2", "bin3", "bin4"],
```

iii. Directly from the Decoder Task spec's bin table (`0: < 90 cm` … `4: > 360 cm`), recorded in CONVERSION_NOTES Step 5 as "5 equal bins" on `[0,450]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical frame indexing to the neural data — `position[trial_slice][frame_mask]` against `deconv[trial_slice][frame_mask]`. No extra alignment step.

ii.
```python
        position_trial = position[trial_slice][frame_mask]
        ...
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. Same justification as 7-d: CONVERSION_NOTES Step 4 ("frame-aligned session-level data"), Step 7 plot review ("position increases monotonically from trial start to near track end ... no visible off-by-one shift"), Step 10 raw-vs-converted `np.allclose()` spot checks on 5 sessions.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavior time series (`processing/behavior/BehavioralTimeSeries/lick/data`), which stores a per-frame cumulative lick count rather than a binary flag.

ii.
```python
        lick = behavior_group["lick/data"][()].astype(np.float32)
        ...
        lick_trial = lick[trial_slice][frame_mask]
```

iii. CONVERSION_NOTES Step 1: "clips licks to binary with `licks[licks > 1] = 1`" (reference `glmUtils.get_timeseries_data`). Step 2: "NWB stores cumulative lick counts per frame". Step 5 mapping: "`lick` → `output[3]` (`lick`) | Convert cumulative lick count per frame to binary (`>0 => 1`), after trial-level lick-error filtering | `glmUtils.get_timeseries_data` | Time-varying binary output."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarized with `(lick_trial > 0).astype(np.int16)`. Additionally, trials whose lick channel looks broken are removed wholesale beforehand (>35% of in-trial frames with cumulative count >2; 69 trials, 0.56%), rather than being masked with NaN. Resulting full-dataset distribution: [0.777 no, 0.223 yes].

ii.
```python
                (lick_trial > 0).astype(np.int16),
```
```python
def has_lick_sensor_error(lick_segment: np.ndarray) -> bool:
    if lick_segment.size == 0:
        return True
    return bool(np.mean(lick_segment > 2) > LICK_ERROR_FRACTION)
...
    for meta in trial_meta:
        if meta["lick_error"]:
            dropped_lick += 1
            continue
```

iii. CONVERSION_NOTES Key Decision 10: "**Drop lick-artifact trials entirely rather than leaving NaNs**: The validator forbids NaNs, and the reference code already treats these trials as invalid for licking analyses." Step 4: "Prefer the released code threshold (`35%`) when reproducing continuous-sample processing, and document the text/code mismatch [Methods says 30%]." Step 3: "Lick artifact trial fraction | ~0.65% of imaged trials removed from licking analyses."

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `trial_slice` and `frame_mask` as the neural data; no shift or resampling.

ii.
```python
        lick_trial = lick[trial_slice][frame_mask]
        ...
        neural_trial = deconv[trial_slice][frame_mask].T.astype(np.float32, copy=False)
```

iii. Same justification as 7-d/8-d: NWB behavior is already frame-aligned to imaging (CONVERSION_NOTES Step 2/Step 4); verified by the Step 7 plot review ("lick raster and speed traces line up with trial progression") and the Step 10 `np.allclose()` raw spot checks.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session-level NWB `identifier` string (its final path component is the VR scene name) plus the trial number. Regexes handle three scene grammars: fixed (`Env1_LocationC`), within-environment switch (`Env1_LocationC_to_A`) and cross-environment switch (`Env1_C_to_Env2_B`). The `reward_zone` behavior series is not used for the label.

ii.
```python
SCENE_SINGLE_RE = re.compile(r"^(Env[12])_Location([ABC])$")
SCENE_SWITCH_RE = re.compile(r"^(Env[12])_Location([ABC])_to_([ABC])$")
SCENE_CROSS_ENV_RE = re.compile(r"^(Env[12])_([ABC])_to_(Env[12])_([ABC])$")

def parse_scene(scene: str) -> SceneInfo:
    match = SCENE_SINGLE_RE.match(scene)
    if match:
        env, zone = match.groups()
        return SceneInfo(before_env=env, before_zone=zone, after_env=None, after_zone=None)

    match = SCENE_SWITCH_RE.match(scene)
    if match:
        env, before_zone, after_zone = match.groups()
        return SceneInfo(before_env=env, before_zone=before_zone,
                         after_env=env, after_zone=after_zone)

    match = SCENE_CROSS_ENV_RE.match(scene)
    if match:
        before_env, before_zone, after_env, after_zone = match.groups()
        return SceneInfo(before_env=before_env, before_zone=before_zone,
                         after_env=after_env, after_zone=after_zone)

    raise ValueError(f"Unsupported scene format: {scene}")
```
```python
        identifier = decode_bytes(f["identifier"][()])
        scene = identifier.split("/")[-1]
        scene_info = parse_scene(scene)
```

iii. CONVERSION_NOTES Step 2: "`identifier` strings include the original scene name (for example `Env1_LocationC`), which is needed to infer reward-zone labels consistently with the reference code." Key Decision 8: "Use paper/code reward-zone semantics rather than inferring from sparse `reward_zone` events alone: The `reward_zone` series marks zone entry events, not the full zone extent, so zone A/B/C must come from session condition metadata." Step 4: "Code infers A/B/C from scene names via string matching ... Code-style scene parsing still works because switch names contain `A_to/B_to/C_to` patterns and the final character identifies the destination zone."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. `zone_for_trial` picks the pre-switch zone for trials with trial number `< 30` and the post-switch zone for trial number `>= 30` on switch sessions (fixed sessions always use the single zone). The label is mapped to `A→0, B→1, C→2` and broadcast across all timepoints of the trial as output row 4. The switch point is hard-coded at 30 (no per-session override is read). Full-dataset distribution: [0.332, 0.336, 0.332].

ii.
```python
SWITCH_TRIAL = 30

def zone_for_trial(scene_info: SceneInfo, trial_number: int) -> str:
    if scene_info.has_switch and trial_number >= SWITCH_TRIAL:
        assert scene_info.after_zone is not None
        return scene_info.after_zone
    return scene_info.before_zone
```
```python
        zone_label = zone_for_trial(scene_info, trial_num)
        ...
                np.full(time_trial.shape, ZONE_TO_CODE[meta["zone_label"]], dtype=np.int16),
```
```python
            ["A", "B", "C"],
```

iii. CONVERSION_NOTES Step 3: "Reward switch point | after 30 trials on each switch day | 'Each switch occurred after 30 trials'". Step 1: "`get_reward_zones` ... including switch days with a default change at trial 30 unless `sess.change_reward_trial` overrides it." Step 5 mapping: "For switch sessions, use pre-switch zone for trials `<30`, post-switch zone for trials `>=30`." Step 9 validates the resulting balance: "`reward_zone_location` distribution | balanced A/B/C across counterbalanced task | [0.332, 0.336, 0.332] | Yes."

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` time series' own timestamps (`processing/behavior/BehavioralTimeSeries/Reward/timestamps`), combined with a gate on the `reward_zone` behavior series (the trial must also contain a reward-zone entry).

ii.
```python
        reward_timestamps = behavior_group["Reward/timestamps"][()].astype(np.float64)
        reward_zone = behavior_group["reward_zone/data"][()].astype(np.float32)
```
```python
def reward_outcome_for_trial(
    reward_timestamps: np.ndarray,
    t_start: float,
    t_stop: float,
    reward_zone_segment: np.ndarray,
) -> int:
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```

iii. CONVERSION_NOTES Step 5 mapping: "reward delivery within trial → `output[5]` (`reward_outcome`) | Trial is rewarded if any reward delivery occurs within the trial and the trial has an `rzone` entry event; encode `0/1`, repeat across timepoints | `behavior.get_trial_types` | Per-trial categorical output, repeated across time." Step 2 describes `Reward` as "(sparse reward deliveries with timestamps)".

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. The trial's time window `[timestamps[start], timestamps[stop])` is used to bracket the sorted reward timestamp array with two `np.searchsorted` calls; a non-empty bracket means a reward was delivered. That is AND-ed with "the `reward_zone` series is non-zero somewhere in the trial". The resulting 0/1 is broadcast across all the trial's timepoints as output row 5, and is also cached per trial number for the previous-trial-outcome input. Full-dataset distribution: [0.158 no, 0.842 yes].

ii.
```python
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
    has_reward = right > left
    has_rzone_entry = np.any(reward_zone_segment > 0)
    return int(has_reward and has_rzone_entry)
```
```python
        reward_outcome = reward_outcome_for_trial(
            reward_timestamps=reward_timestamps,
            t_start=float(timestamps[start]),
            t_stop=float(timestamps[stop]),
            reward_zone_segment=reward_zone[start:stop],
        )
        reward_by_trial_number[trial_num] = reward_outcome
        ...
                np.full(time_trial.shape, meta["reward_outcome"], dtype=np.int16),
```

iii. CONVERSION_NOTES Step 3: "Reward rate | ~85% rewarded, ~15% omitted | 'Reward was randomly omitted on approximately 15% of trials'". Step 9: "Omission fraction (`15.8%`) matches the paper's stated `~15%` omission design closely." Step 12 re-verified against raw NWB: "Direct raw reward counts within those trial windows were: trial `0`: `1` reward event; trial `1`: `0`; trial `2`: `1`; trial `6`: `0` ... This confirms that the weak `reward_outcome` decoder result is not caused by a labeling or alignment bug."

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several defensive mechanisms:
- **Sentinel/NaN frames**: a per-frame validity mask requires finite `position`, `speed`, `lick`, `environment`, `timestamps`, finite neural values in every neuron, and `position > -100` (excluding the `-500` teleport sentinel). Invalid frames are dropped from the trial (empirically a no-op — 0 invalid frames inside trials on the sessions checked).
- **Sentinel `-1` in `environment` / `trial number`**: filtered out before taking the median/mode, so the per-trial label is never corrupted by pre-sync samples.
- **Incomplete trials**: a `trial_start` with no following `teleport` is dropped rather than extrapolated.
- **Degenerate/short trials**: `< 5` frames before or after masking → dropped, counted in `n_trials_dropped_missing`.
- **Broken lick sensor**: whole trial dropped (see 1-e / 9-b).
- **Neural/behavior length mismatch**: not handled explicitly; the code relies on trial bounds coming from behavior so the extra trailing neural frame (present in 10 of 152 sessions) is never indexed. There is no crop/warn as in the reference.
- **Multi-plane ROI bookkeeping**: per-plane `Deconvolved` matrices are reassembled into pooled-ROI order via `planeIdx` + `searchsorted`.
- **Empty/unusable session**: skipped with a message if fewer than 2 trials survive.
- **Hard failures**: `parse_scene`, `env_to_binary`, `modal_trial_number`, `discretize_distance`, `discretize_speed` all raise rather than silently emitting a wrong value.

ii.
```python
    valid_behavior_frames = (
        np.isfinite(position)
        & np.isfinite(speed)
        & np.isfinite(lick)
        & np.isfinite(environment)
        & np.isfinite(timestamps)
        & (position > -100.0)
    )
    valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```
```python
        frame_mask = valid_behavior_frames[start:stop] & valid_neural_frames[start:stop]
        if np.count_nonzero(frame_mask) < MIN_TRIAL_FRAMES:
            dropped_missing += 1
            continue
```
```python
    if deconv is None:
        raise ValueError(f"No accepted deconvolved traces found in {path}")
```
```python
        if len(session["neural"]) < 2:
            print(f"[{idx}/{len(nwb_files)}] Skipping {path.name}: "
                  f"only {len(session['neural'])} valid trials after filtering.")
            continue
```
```python
        "n_trials_dropped_lick": int(dropped_lick),
        "n_trials_dropped_missing": int(dropped_missing),
```

iii. CONVERSION_NOTES Step 2: "Observed sentinel values match the code documentation: before valid synchronization, `environment` and `trial number` can be `-1`; `position` can be `-500`." Key Decision 5: "Keep only complete trials with explicit start and end markers: This avoids fabricating incomplete boundaries." Key Decision 10 (drop lick trials rather than NaN, "the validator forbids NaNs"). Step 10: "The raw-check script hit a benign one-frame mismatch between one plane's deconvolved matrix and the behavior array length. The converter never used that extra trailing frame because trial bounds come from behavior." Step 10 issue log: the multi-plane pooled-ROI bug "Resolution: updated `convert_data.py` to reconstruct the accepted-cell deconvolved matrix by concatenating per-plane series into pooled ROI order before trial slicing."

## 13-a. What are the most time-consuming steps of the code?

i. Almost all wall-clock time is HDF5 I/O: reading `Deconvolved/plane<N>/data` with fancy column indexing (`[:, accepted_local_idx]`, which forces h5py to read the full `(T, n_ROI)` chunk set), plus reading the ten behavior arrays and the per-session `np.all(np.isfinite(deconv), axis=1)` pass over the whole neural matrix. After that, pickling the ~9 GB `converted_data.pkl`. Per-trial arithmetic is negligible. Measured: 3.36 min for all 152 sessions (~1.3 s/session), which the AI considered acceptable against the 15-minute budget. Note this is fast largely *because* the paper's dF/F + OASIS pipeline is skipped.

ii.
```python
            plane_data = ophys_group[f"Deconvolved/plane{plane}/data"][:, accepted_local_idx]
```
```python
    valid_neural_frames = np.all(np.isfinite(deconv), axis=1)
```
```python
    with args.outpicklefile.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```
```python
        elapsed = time.time() - session_start
        total_elapsed = time.time() - start_time
        mean_per_session = total_elapsed / len(processed_sessions)
        eta = mean_per_session * (len(nwb_files) - idx)
        print(f"[{idx}/{len(nwb_files)}] {path.name}: ... {elapsed:.2f}s (ETA {eta / 60:.1f} min)")
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Full-session deconvolved matrices are still loaded into memory one session at a time." Step 6 speedups: "Uses `h5py` instead of higher-level NWB objects for the main conversion path. Processes sessions sequentially and writes only the final pickle, avoiding intermediate I/O." Step 7: "Sample conversion | ~0.42 s/session on first 2 sessions | ~64 s for 152 sessions, plus pickle write overhead." Step 9: "Full conversion runtime: `3.36 min` for all 152 sessions."

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Three Python loops remain, all cheap relative to I/O:
- `find_complete_trial_bounds` walks `starts` with an inner `while` over `teleports`; this is a two-pointer merge that could be done with `np.searchsorted(teleports, starts, side='right')` in one vectorized call.
- The two per-trial loops (`for start, stop in trial_bounds` building `trial_meta`, then `for meta in trial_meta` building arrays). The discretizations (`discretize_distance`, `discretize_absolute_position`, `discretize_speed`, lick binarization) are pure element-wise functions that could be applied once to the whole session array and then sliced per trial; the per-trial `env_to_binary` / `modal_trial_number` reductions could be done with segment reductions.
- The per-plane loop in `process_session` is over at most 2 planes, so irrelevant.
- `build_trial_plot` loops over example trials and over reward times to draw `axvline`s (plot-only path).

ii.
```python
    for start in starts:
        while teleport_idx < len(teleports) and teleports[teleport_idx] <= start:
            teleport_idx += 1
        ...
```
```python
    for start, stop in trial_bounds:
        ...
        trial_num = modal_trial_number(trial_number[start:stop])
        env_bin = env_to_binary(environment[start:stop])
        ...
    for meta in trial_meta:
        ...
        output_trial = np.vstack([
            discretize_distance(distance_trial),
            discretize_absolute_position(position_trial),
            discretize_speed(speed_trial),
            (lick_trial > 0).astype(np.int16),
            ...
        ])
```

iii. CONVERSION_NOTES does not discuss loop vectorization explicitly beyond Step 6 ("Write efficient code" was satisfied by the h5py path and the 3.36 min runtime). Step 6: "Filters trials before building output arrays to avoid wasted allocations for dropped trials" is the one loop-level optimization the AI documents. The per-trial loop is the natural structure given variable-length trials and the ragged output format required by the target spec.

## 13-c. What processing does the code repeat multiple times?

i.
- **Two passes over the trial list**: the first loop computes `trial_num`, `env_bin`, `zone_label`, `reward_outcome` and `lick_error` for every complete trial; the second loop re-walks the same list to emit arrays. The split is needed only so that `reward_by_trial_number` is fully populated before previous-outcome lookups, but everything else is recomputed-adjacent work carried in a dict.
- **`reward_zone[start:stop]` is scanned twice per trial** in effect — once inside `reward_outcome_for_trial` and once in the slicing that produces it.
- **`np.searchsorted(reward_timestamps, ...)` is called twice per trial** (2 × n_trials calls) where one vectorized `np.searchsorted(reward_timestamps, all_boundaries)` would suffice.
- **`np.searchsorted(accepted_idx, accepted_total_idx)`** recomputes column destinations per plane.
- **Reward times are re-filtered** for the plotting examples (`(reward_timestamps >= ...) & (reward_timestamps < ...)`) after already having been bracketed for the outcome.
- **Timing/ETA statistics** are recomputed every session (trivial).

Notably, the code does *not* repeat the expensive step: unlike the reference, there is no separate survey pass, so each NWB file is opened and read exactly once.

ii.
```python
    for start, stop in trial_bounds:
        ...
        trial_meta.append({...})
    ...
    for meta in trial_meta:
        if meta["lick_error"]:
            ...
```
```python
    left = np.searchsorted(reward_timestamps, t_start, side="left")
    right = np.searchsorted(reward_timestamps, t_stop, side="left")
```
```python
            trial_reward_times = reward_timestamps[
                (reward_timestamps >= timestamps[start]) & (reward_timestamps < timestamps[stop])
            ] - timestamps[start]
```

iii. CONVERSION_NOTES Step 6: "Processes sessions sequentially and writes only the final pickle, avoiding intermediate I/O" and "Filters trials before building output arrays to avoid wasted allocations for dropped trials" — i.e. the two-pass structure is presented as an optimization (skip work for dropped trials), not as redundancy. The AI does not otherwise enumerate repeated processing.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- **`kept_examples` are built for every session regardless of `--show-processing`**: up to 3 trials' position, speed, lick, full neural matrix and outputs are copied into a dict per session and carried in `session["examples"]` all the way into `processed_sessions`. Only two sessions ever get plotted, and `session["examples"]` is never read by `build_dataset`. This is pure memory/copy overhead on a 9 GB dataset.
- **`plane_idx` / `n_planes`** are computed only to populate a summary field.
- **`reward_zone`** is read in full (a whole session array) but used only for the boolean `np.any(... > 0)` gate.
- **`trial_numbers_kept`** in the session summary re-derives trial numbers from the already-built input arrays (`int(np.rint(x[0, 0]))`), duplicating `meta["trial_number"]`.
- **`build_trial_plot`'s `output_values` parameter** is accepted and never used; `main` passes `output_values=[]`.
- **`session_id` in `process_session`** is computed and never used (`session_name` is used instead).
- **`dataclass` `SceneInfo.after_env`** is parsed but never consulted — only `after_zone` matters for the label.
- Conversely, the script skips the paper's dF/F + deconvolution entirely, so it does *not* waste time on processing that the decoder discards.

ii.
```python
        if len(kept_examples) < 3:
            trial_reward_times = reward_timestamps[...]
            kept_examples.append({
                "trial_number": trial_num, ..., "neural": neural_trial, "output": output_trial, ...
            })
```
```python
    return (
        {..., "summary": session_summary, "examples": kept_examples},
        kept_examples,
    )
```
```python
        "n_planes": int(np.unique(plane_idx).size),
        ...
        "trial_numbers_kept": [int(np.rint(x[0, 0])) for x in input_trials] if input_trials else [],
```
```python
def build_trial_plot(session_id, trial_examples, output_names, output_values) -> None:
    # output_values is never referenced in the body
```
```python
            build_trial_plot(
                session["summary"]["session_name"],
                examples,
                output_names=["dist_to_zone", "abs_pos", "speed", "lick", "zone", "reward"],
                output_values=[],
            )
```

iii. CONVERSION_NOTES does not identify any of these. Step 6 lists only "Full-session deconvolved matrices are still loaded into memory one session at a time" and "Reward-zone labels are parsed from scene metadata rather than cached lookup tables" as inefficiencies, and Step 13 notes only file organization. The unconditional example collection appears to be an oversight rather than a documented decision.
