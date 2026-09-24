# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script discovers every session file with a single glob over the data root, `/app/data/sub-*/sub-*_behavior+ophys.nwb`, sorted for determinism. This finds all 152 NWB files (11 subject directories; 12 sessions for `m11`, 14 for the other 10 mice). Each file is one session and is opened once with `pynwb.NWBHDF5IO(..., load_namespaces=True)`; behavior streams are read from `processing/behavior/BehavioralTimeSeries` and imaging from `processing/ophys`. `--sample` truncates the file list to the first two sessions; `--full` (default) processes all of them. Every trial found by the `trial_start`/`teleport` pulse pairing is kept (12,216 trials total).

ii.
```python
def main() -> None:
    args = parse_args()
    data_root = Path("/app/data")
    files = sorted(data_root.glob("sub-*/sub-*_behavior+ophys.nwb"))
    if not files:
        raise FileNotFoundError("No NWB session files found under /app/data")

    if args.sample:
        files = files[:2]
```
```python
def convert_session(file_path: Path) -> ...:
    with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
        nwb = io.read()
        meta = parse_identifier(nwb.identifier, file_path)

        beh = nwb.processing["behavior"]["BehavioralTimeSeries"]
        frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
        position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
        ...
        deconv, plane_per_cell, n_planes = load_curated_deconvolved(nwb)
```

iii. From CONVERSION_NOTES Step 2 and Step 5: the data are "a DANDI-style NWB dataset with one folder per subject and one NWB file per session", filenames follow `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`, and the conversion plan states "include all 152 imaging sessions in the NWB dataset … no fixed-condition mice are present in `/app/data`, so the dataset already matches the 11-mouse switch cohort". The count was cross-checked against the paper (11 switch mice, 14 task days, m11 imaging starting on day 3 → 152 sessions) and against a direct raw-data audit (Step 9/10 consistency tables).

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the parent directory name of each NWB file (`sub-m11` → `m11`); this is also cross-stored in `session_info` together with the original internal animal name (`GCAMP…`) parsed from `nwb.identifier`. Subjects are appended to `data['subjects']` in first-encounter order and `subject_idx` records, for each session, the index into that list. 11 subjects result.

ii.
```python
def parse_identifier(identifier: str, file_path: Path) -> SessionMeta:
    parts = identifier.strip("/").split("/")
    subject_original = parts[-3] if len(parts) >= 3 else file_path.parent.name.replace("sub-", "GCAMP")
    ...
    subject = file_path.parent.name.replace("sub-", "")
```
```python
        if meta.subject not in subject_to_idx:
            subject_to_idx[meta.subject] = len(subjects)
            subjects.append(meta.subject)
        subject_idx.append(subject_to_idx[meta.subject])
```

iii. Step 2 notes that "`/app/data` contains … one folder per subject", that `subject.subject_id` inside the NWB matches the folder name, and that the `identifier` string "encodes the original animal ID, recording date, and scene string". Step 9 verifies 11 subjects with 12/14 sessions each, matching the paper's n = 11 switch mice.

## 1-c. How are the data split into sessions?

i. One NWB file = one session. The session number is parsed from the file name (`_ses-(\d+)_`) into `session_id`, and the session's date and scene come from the `identifier` path string. Sessions are emitted in sorted-filename order, so sessions of one subject stay contiguous and in day order. No cross-session cell alignment (the paper's ROI tracking across days) is attempted.

ii.
```python
    session_match = re.search(r"_ses-(\d+)_", file_path.name)
    session_id = session_match.group(1) if session_match else "unknown"
    ...
    return SessionMeta(subject=subject, subject_original=subject_original,
                       session_name=file_path.stem.replace("_behavior+ophys", ""),
                       session_id=session_id, date=date, scene=scene, ...)
```

iii. Step 2: "Each NWB filename follows `sub-<mouse>_ses-<NN>_behavior+ophys.nwb`" and `session_id` is stored in NWB metadata. Step 4 reconciles this with the reference `sessions_dict.py` (one entry per animal per experiment day) and concludes "Counts are consistent once m11's delayed imaging start is accounted for" (152 sessions).

## 1-d. How are the data split into trials?

i. Trials are defined by the behavioral pulse streams: each rising sample of `trial_start` opens a trial and the next `teleport > 0` sample after it closes the trial. The slice used is `[start, stop)` — trial-start sample inclusive, teleport sample exclusive, i.e. the lap on the track, excluding the inter-trial teleport/tunnel period. The stored `trial number` stream is used only as a label, never to define boundaries. Sessions are required to yield at least two trials.

ii.
```python
def pair_trial_bounds(trial_start: np.ndarray, teleport: np.ndarray) -> list[tuple[int, int]]:
    starts = np.flatnonzero(trial_start > 0)
    stops = np.flatnonzero(teleport > 0)
    bounds: list[tuple[int, int]] = []
    stop_idx = 0
    for start in starts:
        while stop_idx < len(stops) and stops[stop_idx] <= start:
            stop_idx += 1
        if stop_idx >= len(stops):
            break
        stop = int(stops[stop_idx])
        bounds.append((int(start), stop))
        stop_idx += 1
    return bounds
```
```python
    trial_bounds = pair_trial_bounds(trial_start, teleport)
    if len(trial_bounds) < 2:
        raise ValueError(f"{meta.session_name}: expected at least 2 complete trials, found {len(trial_bounds)}")
    ...
    for trial_idx, (start, stop) in enumerate(trial_bounds):
        neural_trial = np.nan_to_num(deconv[start:stop, :].T, ...)
```

iii. Step 4/Step 5: "Reference code uses `sess.trial_start_inds` and `sess.teleport_inds` arrays … Reconstruct trial starts/stops from pulse indices; use `trial_start`/`teleport` pulses as authoritative." Key decision 3: "Trial window = start index inclusive, teleport index exclusive. This reproduces the reference code's effective `start-1:stop-1` slicing semantics after translating to 0-based NWB indices." The AI also documented the one anomalous session (`sub-m11_ses-03`, 81 `trial number` segments but 80 pulse-defined trials) and argued the pulse definition avoids counting the trailing tunnel-only segment.

## 1-e. How are trials filtered based on quality controls?

i. Essentially no quality-based trial rejection is applied. The only trials that can be dropped are degenerate ones: `stop <= start`, or a trial containing no valid (`environment >= 0`) sample. There is no minimum-duration rule and no maximum-duration rule; the notes explicitly justify keeping very long laps. A trial whose stop exceeds the neural array raises an error rather than being dropped, and sessions with fewer than two usable trials raise. In practice nothing is filtered: all 12,216 pulse-defined trials survive.

ii.
```python
    for trial_idx, (start, stop) in enumerate(trial_bounds):
        if stop <= start:
            continue
        if stop > deconv.shape[0]:
            raise ValueError(f"{meta.session_name}: trial stop exceeds neural data length")

        env_trial = environment[start:stop]
        env_valid = env_trial[env_trial >= 0]
        if env_valid.size == 0:
            continue
```
```python
    if len(neural_trials) < 2:
        raise ValueError(f"{meta.session_name}: fewer than 2 usable trials after conversion")
```

iii. Step 10 edge-case review: "Long trials: some sessions contain very long completed laps (up to `216.5 s`, `T=3359`). These are preserved because they are present in the aligned raw data and do not violate format checks." The pre-sync sentinel samples (`environment = -1`, `position = -500`) are excluded implicitly because they fall outside any trial window ("exclude pre-sync sentinel samples … by working only within extracted trial windows"). The paper describes no trial-level exclusion other than lick-sensor trials (which the AI handles inside the lick output rather than by dropping the trial).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural matrix is the NWB-stored `processing/ophys/Deconvolved` ROI response series (suite2p's own deconvolved trace, `spks`), restricted to curated ROIs. For multi-plane sessions the per-plane series are concatenated along the ROI axis in plane order. The `Fluorescence` (F) and `Neuropil` (Fneu) series are read by the script's plotting path only in the sense that they are never read at all — they are not used. The AI thus treats the stored `Deconvolved` array as the paper's "events".

ii.
```python
def load_curated_deconvolved(nwb) -> tuple[np.ndarray, np.ndarray, int]:
    seg = nwb.processing["ophys"]["ImageSegmentation"].plane_segmentations["PlaneSegmentation"]
    iscell = np.asarray(seg["iscell"].data[:] ...)
    plane_idx = np.asarray(seg["planeIdx"].data[:] ...).astype(int)
    keep = iscell[:, 0] == 1

    deconv_mod = nwb.processing["ophys"]["Deconvolved"]
    plane_names = sorted(deconv_mod.roi_response_series.keys(), key=get_plane_number)
    for plane_name in plane_names:
        rs = deconv_mod.roi_response_series[plane_name]
        plane_keep_mask = keep & (plane_idx == plane_num)
        plane_keep_local = plane_keep_mask[plane_idx == plane_num]
        plane_data = np.asarray(rs.data[:, plane_keep_local], dtype=np.float32)
        arrays.append(plane_data)
    deconv = np.concatenate(arrays, axis=1)
```

iii. Step 5 Key Decision 1: "**Neural signal = deconvolved activity, not raw fluorescence or dF/F**: The reference decoder notebook explicitly decodes from `events`, and the paper describes deconvolution as the activity-rate representation used in downstream analyses." Metadata records `"neural_signal": "Deconvolved calcium events from NWB processing/ophys/Deconvolved."` The AI never checked whether the stored `Deconvolved` array equals the paper's `sess.timeseries['events']`, which the reference code builds itself from F and Fneu via `preprocessing.dff(..., deconvolve=True)`.

## 2-b. How is the `neural` data processed?

i. No signal processing is applied. The stored deconvolved traces for curated cells are sliced per trial, transposed to `(n_neurons, n_timepoints)`, passed through `np.nan_to_num`, and cast to `float16` for storage. There is no neuropil subtraction (`F − 0.7·Fneu`), no per-trial maximin baseline over a 20 s window, no `(F − baseline)/|baseline|` dF/F normalization, no 2-sample Gaussian smoothing, and no OASIS deconvolution at `tau = 0.7` with the per-plane frame rate. Multi-plane sessions concatenate cells across planes. A single trailing neural frame is trimmed where the deconvolved array is exactly one sample longer than the behavior stream (10 multi-plane sessions).

ii.
```python
        frame_delta = deconv.shape[0] - frame_times.shape[0]
        if frame_delta == 1:
            deconv = deconv[: frame_times.shape[0], :]
        elif frame_delta != 0:
            raise ValueError(...)
```
```python
        neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0,
                                     neginf=0.0).astype(np.float16, copy=False)
```

iii. Step 3 correctly records the paper's pipeline ("dF/F baseline is computed independently within each trial using a maximin procedure with a 20 s sliding window … Deconvolved activity is extracted with OASIS and is the main 'activity rate' used in analyses"), but Step 5 concludes that the NWB `Deconvolved` field already *is* that signal, so no recomputation is needed. Step 6 justifies `float16` purely as a storage optimization ("~2x smaller neural payload"), and Step 10 explains that the initially failing neural sanity check was "expected `float16` quantization from the deliberate storage-size optimization, not a neuron-order or alignment bug".

## 2-c. How is the `neural` data filtered based on quality controls?

i. One filter only: suite2p/manual curation, `iscell[:, 0] == 1`. Uncurated ROIs are never read from disk (the mask is applied at the h5 read). The paper's additional exclusion of putative interneurons (cells whose dF/F correlates with running speed at r > 0.5) is explicitly considered and then not applied. Result: 138,678 cells (mean 912.4/session, range 155–2341).

ii.
```python
    iscell = np.asarray(seg["iscell"].data[:] if hasattr(seg["iscell"], "data") else seg["iscell"][:])
    keep = iscell[:, 0] == 1
    ...
        plane_keep_mask = keep & (plane_idx == plane_num)
        plane_keep_local = plane_keep_mask[plane_idx == plane_num]
        plane_data = np.asarray(rs.data[:, plane_keep_local], dtype=np.float32)
```
```python
            "cell_curation": "Suite2p/manual curated cells only (iscell[:,0] == 1).",
```

iii. Step 4 identified the discrepancy and the correct threshold ("Methods state additional putative interneurons were excluded by speed correlation `>0.5` … If applying post-manual interneuron exclusion … use `0.5`, not the helper default"). Step 5 Key Decision 11 deferred the choice, and Step 10 settled it: "I did not apply the later speed-correlation interneuron exclusion because the decoder task here is session-wide and not restricted to place-cell subsets; the converted neuron counts and decoder verification remained internally consistent." Step 4 also flags that curated counts (max 2341) slightly exceed the paper's stated 155–2172 range, only for the pooled two-plane mouse m18.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start needs no extra work: the neural array is on the same frame grid as the behavior streams, so the trial slice `deconv[start:stop]` starts exactly at the `trial_start` pulse sample. `off_start` is recorded as `0.0` (no pre-trial window) and `off_end` as `None` (variable trial length). Sample-count equality between the neural and behavior arrays is enforced, with the single-frame trim described in 2-b.

ii.
```python
        neural_trial = np.nan_to_num(deconv[start:stop, :].T, ...)
        ...
        trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
```
```python
            "temporal_alignment_event": "trial start",
            "off_start": 0.0,
            "off_end": None,
```

iii. Step 4: "NWB behavior streams already have one sample per imaging frame and shared timestamps … Treat NWB behavioral streams as already aligned products of the original synchronization step; no extra interpolation is needed." Step 5 Key Decision 2: "Trial alignment event = trial start: This is required by the user task and is compatible with the paper's lap structure and the reference code's use of `trial_start_inds` / `teleport_inds`."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. No rebinning or resampling. Data are kept at the native imaging frame rate, 15.5078125 Hz → 64.4836 ms per bin. The bin size written to metadata is the median behavior-frame interval of the first session, and every session is hard-checked against the expected 1/15.5078125 s interval (raising if it deviates by more than 1e-9 s), which guarantees a single common bin size across all 152 sessions. Multi-plane sessions are treated as one ~15.5 Hz sample per plane row, deliberately ignoring the `rate` attribute (31.015625 Hz) on the deconvolved series.

ii.
```python
EXPECTED_FRAME_DT_S = 1.0 / 15.5078125
...
        frame_dt_s = float(np.median(np.diff(frame_times)))
        if not np.isclose(frame_dt_s, EXPECTED_FRAME_DT_S, atol=1e-9, rtol=0):
            raise ValueError(f"{meta.session_name}: unexpected frame dt {frame_dt_s} s; ...")
...
        if time_bin_ms is None:
            time_bin_ms = float(stats["frame_dt_s"]) * 1000.0
```

iii. Step 4: "Use behavior timestamps / frame count as the true decoder time base. Ignore the misleading deconvolved `rate` metadata in multi-plane exports and treat each sample row as one ~15.5 Hz frame-aligned observation per plane." Step 3 records the paper's statement that "All behavioral and neural time series were sampled at ~15.5 Hz, the imaging frame rate", so keeping the native grid preserves all temporal information the paper used.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the `timestamps` attached to the `position` behavioral time series (all behavior streams in these files share one timestamp vector at the imaging frame rate). The same vector is used to rasterize reward events and to derive the time bin size.

ii.
```python
        frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
```

iii. Step 5 variable-mapping table maps "`behavior.position.timestamps` and trial start time → `input[0]`, `time_from_trial_start_s = timestamps - trial_start_timestamp` per frame". Step 4 established that all behavior streams are already synchronized to the imaging frames, so any stream's timestamps would give the same values.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the trial's first frame time from the frame times within the trial, and cast to float32. The first sample of every trial is therefore exactly 0 s. No smoothing, clipping, or normalization.

ii.
```python
        trial_time_s = (frame_times[start:stop] - frame_times[start]).astype(np.float32, copy=False)
        ...
        input_trial = np.vstack([
            trial_time_s,
            ...
        ]).astype(np.float32, copy=False)
```

iii. Straightforward implementation of the instruction "Time from start of trial in seconds (continuous, time-varying)". Step 10 sanity check: "`sub-m11_ses-03`, trial 0 `time_from_trial_start_s` matched raw frame timestamps relative to the trial-start frame with `np.allclose(..., atol=1e-6) == True`." Verified range in the full data: [0.0, 216.5] s.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: both are sliced with the same `[start, stop)` index range on a shared frame grid, so timepoint *k* of `input` is timepoint *k* of `neural`. The only alignment work is the pre-check that the neural array and the behavior timestamp vector have the same length (with the one-frame trim for the 10 multi-plane sessions that have one extra neural sample), plus the per-session assertion on the frame interval.

ii.
```python
        frame_delta = deconv.shape[0] - frame_times.shape[0]
        if frame_delta == 1:
            deconv = deconv[: frame_times.shape[0], :]
        elif frame_delta != 0:
            raise ValueError(f"{meta.session_name}: neural frames ({deconv.shape[0]}) do not match "
                             f"behavior frames ({frame_times.shape[0]})")
```

iii. Step 10: "One-frame neural/behavior mismatch: 10 multi-plane sessions had deconvolved arrays longer than the aligned behavior streams by exactly 1 frame. Fix implemented … trim the final neural frame only when `deconv_len == behavior_len + 1`; otherwise raise an error." Step 4 argues no interpolation is needed because the NWB streams are already the aligned product of `vr_align_to_2P`.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behavioral time series (0 = ENV1, 1 = ENV2 once synchronized; −1 is a pre-sync sentinel).

ii.
```python
        environment = np.asarray(beh.time_series["environment"].data[:], dtype=np.float32)
        ...
        env_trial = environment[start:stop]
        env_valid = env_trial[env_trial >= 0]
```

iii. Step 4: "Code uses morph / env logic (`Env1 -> 0`, `Env2 -> 1`); NWB `environment` valid values are `0` and `1` after sync, `-1` pre-sync … Use `environment` directly as binary ENV1/ENV2 after removing invalid pre-sync samples." Step 5 Key Decision 7: environment is taken "from the aligned `environment` stream, not from scene-name prefixes alone … necessary for cross-environment switch sessions, where ENV changes mid-session after trial 30."

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Within each trial, sentinel samples (`environment < 0`) are discarded, the integer mode of the remaining values is taken, and that single value is broadcast across all timepoints of the trial as a constant row of the input matrix. (The environment stream is in fact constant within every trial in this dataset and never negative inside a trial, so the mode/sentinel logic is a no-op defensive step.)

ii.
```python
def mode_int(values: np.ndarray) -> int:
    uniq, counts = np.unique(values.astype(int), return_counts=True)
    return int(uniq[np.argmax(counts)])
```
```python
        env_valid = env_trial[env_trial >= 0]
        if env_valid.size == 0:
            continue
        env_code = float(mode_int(env_valid))
        ...
        input_trial = np.vstack([
            trial_time_s,
            np.full(trial_time_s.shape, env_code, dtype=np.float32),
            ...
        ])
```

iii. Step 5 Key Decision 4: "Per-trial constants will be repeated across time: To keep `input` and `output` arrays uniform as `(n_channels, n_timepoints)`, trial-wise variables … will be broadcast across all frames in the trial." The mode is the AI's defensive way of collapsing a stream it expects to be constant while ignoring pre-sync sentinels. Step 10 spot-check: "`sub-m11_ses-03`, trial 40 `environment_type` matched the raw aligned environment stream over the whole trial with `np.allclose(..., atol=0) == True`."

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the stored `trial number` behavioral stream, read at the trial's first sample, with a fallback to the loop index if that value is negative. The trial *boundaries* still come from the pulse streams; `trial number` is used purely as a label.

ii.
```python
        trial_number = np.asarray(beh.time_series["trial number"].data[:], dtype=np.float32)
        ...
        trial_num_native = int(round(float(trial_number[start])))
        if trial_num_native < 0:
            trial_num_native = trial_idx
```

iii. Step 5 mapping table: "Use the pulse-paired trial order / native 0-indexed completed-trial number, repeat across frames … Keep 0-indexing to match reference code and switch-at-trial-30 logic; boundaries come from pulses, not trial-number transitions, because trial numbers extend into tunnel periods." Key Decision 5 makes the same point about `sub-m11_ses-03`'s extra trailing segment.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond rounding to an integer and broadcasting the per-trial constant across all timepoints of the trial as input row 2. Values are 0-indexed within a session and run 0–99.

ii.
```python
        input_trial = np.vstack([
            trial_time_s,
            np.full(trial_time_s.shape, env_code, dtype=np.float32),
            np.full(trial_time_s.shape, float(trial_num_native), dtype=np.float32),
            np.full(trial_time_s.shape, float(prev_reward_outcomes[trial_idx]), dtype=np.float32),
        ]).astype(np.float32, copy=False)
```

iii. Same as 5-a: per-trial constants are broadcast (Key Decision 4), and the 0-indexed convention is kept to line up with the reference code's `trial_ids` and its switch-after-30-trials logic.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the `Reward` event series' `timestamps` (reward deliveries are stored as events with their own timestamps rather than as a per-frame vector). These are rasterized onto the shared behavior frame grid to produce a binary `reward_frames` vector, which is then reduced to a per-trial outcome and shifted by one trial.

ii.
```python
def rasterize_reward_events(frame_times: np.ndarray, reward_times: np.ndarray) -> np.ndarray:
    reward_frames = np.zeros(frame_times.shape[0], dtype=np.int8)
    if reward_times.size == 0:
        return reward_frames
    reward_indices = np.searchsorted(frame_times, reward_times)
    reward_indices = np.clip(reward_indices, 0, frame_times.shape[0] - 1)
    nearest = frame_times[reward_indices]
    if not np.allclose(nearest, reward_times, atol=1e-9, rtol=0):
        diffs = np.abs(nearest - reward_times)
        raise ValueError(f"Reward timestamps do not align to frame timestamps. Max diff={diffs.max()}")
    reward_frames[reward_indices] = 1
    return reward_frames
```

iii. Step 4: "NWB stores reward deliveries as event series `Reward` with timestamps, not a lowercase per-frame reward vector … Rasterize `Reward.timestamps` onto the behavior/imaging frame timestamps. In inspected data the reward timestamps exactly match frame timestamps (`max_abs_time_diff = 0`)." The code enforces that exactness with a hard error (atol = 1e-9) rather than a tolerance of half a bin.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Per-trial reward outcome is `any(reward_frames[start:stop] > 0)`; the previous-trial outcome vector is that array shifted forward by one, with 0 for the first trial of a session. The value is broadcast across all timepoints of the trial as input row 3. Note the shift is over the *pulse-defined trial list*, not over the kept-trial list (immaterial here, since no trials are dropped).

ii.
```python
def trial_reward_outcomes(bounds: list[tuple[int, int]], reward_frames: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(bounds), dtype=np.int8)
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
    return outcomes
```
```python
    reward_outcomes = trial_reward_outcomes(trial_bounds, reward_frames)
    prev_reward_outcomes = np.zeros_like(reward_outcomes)
    if reward_outcomes.size > 1:
        prev_reward_outcomes[1:] = reward_outcomes[:-1]
```

iii. Step 5 mapping table: "Reward outcome of previous trial, derived from `Reward` event timestamps grouped by trial → `input[3]`; binary per trial, repeated across frames. First trial has no history; set previous outcome to `0` and document as a boundary convention." Step 10 sanity check: "`sub-m11_ses-03`, trial 41 `previous_trial_outcome` matched the reward outcome of trial 40 with `np.allclose(..., atol=0) == True`."

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the `position` behavioral stream (cm on the 450 cm track) together with the *active reward-zone interval* for that trial. The interval is not read from the data: it is looked up from a hard-coded dictionary of the paper's zone coordinates (A: 80–130, B: 200–250, C: 320–370 cm) indexed by the per-trial zone label, which is itself inferred from the session's scene string (see 10-a). The `reward_zone` behavioral time series is never read.

ii.
```python
REWARD_ZONE_COORDS_CM = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}
```
```python
        rz_label = reward_labels[trial_idx]
        rz_start, rz_stop = REWARD_ZONE_COORDS_CM[rz_label]
        rz_dist = signed_distance_to_interval(pos_trial, rz_start, rz_stop)
```

iii. Step 3 records the coordinates from the Methods ("zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm"), and Step 4 explains that "NWB stores no explicit `A/B/C` label; only `identifier` contains scene name", so the label is reconstructed from the scene using the reference code's `behavior.get_reward_zones` logic. Key Decision 9: "Distance-to-reward-zone will be defined relative to the active interval, not just the reward-zone start: This best matches the user's wording 'distance to any location in the reward zone,' making all in-zone positions class `3`."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest point of the zone interval: negative before the zone (`position − zone_start`), exactly 0 anywhere inside the zone, positive after the zone (`position − zone_stop`). Computed per trial on the raw position samples (after `nan_to_num`), then discretized (7-c). No smoothing and no circular wrapping (unlike the reference GLM code's circular reward-relative coordinate, which is not what the instructions ask for).

ii.
```python
def signed_distance_to_interval(position_cm: np.ndarray, start_cm: float, stop_cm: float) -> np.ndarray:
    out = np.zeros_like(position_cm, dtype=np.float32)
    below = position_cm < start_cm
    above = position_cm > stop_cm
    out[below] = position_cm[below] - start_cm
    out[above] = position_cm[above] - stop_cm
    return out
```

iii. Key Decision 9 (above) plus the metadata string `"reward_distance_bin_rule": "signed distance to nearest point in active reward zone interval; bins: <-50, [-50,-10), [-10,0), 0, (0,10], (10,50], >50"`. Step 10 sanity check: "`sub-m11_ses-03`, trial 40 `distance_to_reward_zone` matched bins recomputed directly from raw position and the active reward-zone interval with `np.allclose(..., atol=0) == True`."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit boolean masks: 0 for `d < −50`, 1 for `−50 ≤ d < −10`, 2 for `−10 ≤ d < 0`, 3 for `d == 0` (in zone), 4 for `0 < d ≤ 10`, 5 for `10 < d ≤ 50`, 6 for `d > 50`. Note the sign asymmetry at the ±10 and ±50 edges (left-closed below zero, right-closed above zero), which the instructions leave ambiguous.

ii.
```python
def discretize_reward_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.full(distance_cm.shape, 6, dtype=np.int8)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. Step 5 "Output bin conventions" lists exactly this rule and ties it to the Decoder Task specification; the class-3 ("in zone") bin follows from Key Decision 9. Resulting full-data distribution `[0.253, 0.102, 0.074, 0.237, 0.021, 0.072, 0.242]` was reported and judged sensible in Step 9.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same-index slicing: `position[start:stop]` uses exactly the index range used for `deconv[start:stop]`, on the common frame grid, so no resampling or shifting is involved.

ii.
```python
        pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        ...
        output_trial = np.vstack([
            discretize_reward_distance(rz_dist),
            ...
        ])
```

iii. Step 4's conclusion that the NWB streams are already frame-aligned, enforced by the neural/behavior length check in 3-c. Step 7 plot review: "reward events align to positions within the active reward zone; trial windows cover the track traversal and exclude teleport segments."

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behavioral time series (cm along the 450 cm virtual track), the same array used for the reward-zone distance.

ii.
```python
        position = np.asarray(beh.time_series["position"].data[:], dtype=np.float32)
        ...
        pos_trial = np.nan_to_num(position[start:stop], ...)
```

iii. Step 2 lists `position` among the per-frame behavior variables and notes the pre-sync sentinel value −500, which lies outside all trial windows. Step 5 maps `behavior.position → output[1]`.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice, `nan_to_num`, and discretization into 5 bins. Raw cm values are used directly; no re-centering, no wrapping, no exclusion of samples slightly outside [0, 450].

ii.
```python
        pos_trial = np.nan_to_num(position[start:stop], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        ...
        output_trial = np.vstack([
            discretize_reward_distance(rz_dist),
            discretize_position(pos_trial),
            ...
        ])
```

iii. Step 5 mapping: "`behavior.position` → `output[1]`: Discretize absolute position into 5 bins across 450 cm track". The AI's rationale for using the raw stream is that the NWB position is already the aligned VR position in cm used throughout the reference analyses.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins spanning the 450 cm track, with open ends: 0 = `< 90`, 1 = `[90, 180)`, 2 = `[180, 270)`, 3 = `[270, 360)`, 4 = `≥ 360`. Samples marginally outside the track therefore fall into the first/last bins rather than forming extra classes.

ii.
```python
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    out = np.full(position_cm.shape, 4, dtype=np.int8)
    out[position_cm < 90.0] = 0
    out[(position_cm >= 90.0) & (position_cm < 180.0)] = 1
    out[(position_cm >= 180.0) & (position_cm < 270.0)] = 2
    out[(position_cm >= 270.0) & (position_cm < 360.0)] = 3
    out[position_cm >= 360.0] = 4
    return out
```
```python
            "position_bin_rule": "[0,90), [90,180), [180,270), [270,360), [360,inf)",
```

iii. Step 5: "absolute-position bins will be half-open `[0,90)`, `[90,180)`, `[180,270)`, `[270,360)`, `[360, inf)`", matching the Decoder Task's "5 equal-sized bins spanning the 450 cm track" and the paper's 450 cm track length. Resulting distribution `[0.211, 0.178, 0.231, 0.227, 0.154]`.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical index slicing to the neural data (`[start, stop)` on the shared frame grid); no additional alignment step.

ii.
```python
        neural_trial = np.nan_to_num(deconv[start:stop, :].T, ...)
        pos_trial = np.nan_to_num(position[start:stop], ...)
```

iii. Same justification as 7-d: Step 4 established the streams are already synchronized at the imaging frame rate, and the length check plus frame-interval assertion guard it.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behavioral time series, which stores a cumulative lick count per imaging frame (values can exceed 1).

ii.
```python
        lick_counts = np.asarray(beh.time_series["lick"].data[:], dtype=np.float32)
        ...
        lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
```

iii. Step 5 mapping: "`behavior.lick` → `output[3]`: Convert to binary lick/no-lick per frame after sensor-error handling and clipping … Native stream is cumulative lick count per frame; convert any positive count to `1`."

## 9-b. What processing is involved in computing `output` *Lick*?

i. Two steps. (1) Lick-sensor artifact rejection copied from the reference code: if more than 35% of the frames in a trial have a cumulative lick count > 2, the whole trial's lick trace is set to zero and the trial is counted in `bad_lick_trials`. (2) Binarization: any remaining positive count → 1, else 0. The reference code sets such trials to NaN (i.e. excludes them); because the output must be a categorical label, the AI encodes them as "no lick" instead. Across the full dataset 69 of 12,216 trials (0.56%) trip the rule, closely matching the paper's report of 81/12,376 trials (~0.65%).

ii.
```python
LICK_SENSOR_COUNT_THRESHOLD = 2.0
LICK_SENSOR_BAD_FRAC = 0.35

def correct_lick_trial(lick_counts: np.ndarray) -> tuple[np.ndarray, bool]:
    corrected = lick_counts.copy()
    bad_trial = float(np.mean(corrected > LICK_SENSOR_COUNT_THRESHOLD)) > LICK_SENSOR_BAD_FRAC
    if bad_trial:
        corrected[:] = 0.0
    corrected = (corrected > 0).astype(np.int8)
    return corrected, bad_trial
```
```python
            "lick_sensor_rule": {
                "count_threshold": LICK_SENSOR_COUNT_THRESHOLD,
                "bad_fraction_threshold": LICK_SENSOR_BAD_FRAC,
                "bad_trial_replacement": "binary lick set to 0 for the full trial",
            },
```

iii. Step 1 recorded that the reference `glmUtils.get_timeseries_data` performs "sensor-error cleanup" before thresholding licks to binary, and Key Decision 12 states "Lick sensor errors will be handled using the reference heuristic: Trials/samples with obvious cumulative-sensor failure should not be silently treated as real lick bursts." The thresholds (count > 2, fraction > 0.35) are exactly the reference code's (`if sum(licks[start:stop] > 2)/len(licks[start-1:stop-1]) > 0.35: licks[start:stop] = np.nan`) and match the Methods' description (">30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2"). Step 12 verified two flagged trials (`sub-m11_ses-06` trial 60, `sub-m12_ses-10` trial 73) against raw data.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start, stop)` slice as the neural data; the artifact rule operates within that same window, so no shift is introduced.

ii.
```python
        lick_binary, bad_lick = correct_lick_trial(lick_counts[start:stop])
        ...
        output_trial = np.vstack([
            discretize_reward_distance(rz_dist),
            discretize_position(pos_trial),
            discretize_speed(speed_trial),
            lick_binary,
            ...
        ])
```

iii. Same justification as 7-d/8-d — a single shared frame grid for all behavior streams and the neural array.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. Not from a behavioral time series at all: from the session's *scene string*, parsed out of `nwb.identifier` (e.g. `/data/InVivoDA/GCAMP3/03_10_2022/Env1_LocationC_to_A` → `Env1_LocationC_to_A`), combined with a hard-coded switch trial of 30 for switch sessions. This is a direct port of the reference code's `behavior.get_reward_zones(sess, rz_dict=None, change_trial=30)`, which likewise maps scene names to zone labels. The NWB `reward_zone` time series is not used.

ii.
```python
SCENE_SWITCH_TRIAL = 30

def scene_to_reward_labels(scene: str, ntrials: int, change_trial: int = SCENE_SWITCH_TRIAL) -> list[str]:
    if scene in {"Env1_LocationA", "Env2_LocationA", "Env3_LocationA"}:
        return ["A"] * ntrials
    ...
    first_n = min(change_trial, ntrials)
    second_n = max(0, ntrials - change_trial)
    if "A_to" in scene and scene.endswith("B"):
        return ["A"] * first_n + ["B"] * second_n
    ...
    if "Training" in scene:
        return ["A"] * ntrials
    raise ValueError(f"Unsupported scene string for reward-zone inference: {scene}")
```
```python
    meta = parse_identifier(nwb.identifier, file_path)   # meta.scene comes from nwb.identifier
    reward_labels = scene_to_reward_labels(meta.scene, len(trial_bounds))
    reward_codes = np.array([reward_label_to_code(label) for label in reward_labels], dtype=np.int8)
```

iii. Step 4: "`behavior.get_reward_zones` uses internal coordinate keys `X/Y/Z` but returns labels `A/B/C` based on scene string … Parse scene from `nwb.identifier`, then use the reference code's scene-to-label logic." Key Decision 6: "The NWB export omits categorical reward-zone labels, so I will reconstruct them from the `identifier` scene string using the same scene logic as `behavior.get_reward_zones`." Step 3 records the paper's "Each switch occurred after 30 trials", which is also the reference function's default `change_trial`.

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The per-trial label is mapped A→0, B→1, C→2 and broadcast across all timepoints of the trial as output row 4. For switch sessions the first 30 trials get the pre-switch zone and the remainder the post-switch zone; for environment-switch scenes (`Env1_B_to_Env2_C` etc.) the same first-letter/last-letter rule applies. The resulting class distribution is `[0.329, 0.337, 0.335]`.

ii.
```python
def reward_label_to_code(label: str) -> int:
    return {"A": 0, "B": 1, "C": 2}[label]
```
```python
        output_trial = np.vstack([
            ...
            np.full(trial_time_s.shape, reward_codes[trial_idx], dtype=np.int8),
            np.full(trial_time_s.shape, reward_outcomes[trial_idx], dtype=np.int8),
        ]).astype(np.int16, copy=False)
```

iii. Key Decision 4 (broadcast per-trial constants) plus Step 10's statistics check: "Reward-zone occupancy: converted per-frame zone fractions `[0.329, 0.337, 0.335]`, confirming the scene-derived A/B/C mapping is balanced as expected from counterbalancing across mice." Step 10's spot check compared converted labels with the scene-derived expectation (`trial 0 = B`, `trial 40 = A`) rather than with the raw `reward_zone` stream.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. From the `Reward` event series timestamps, rasterized onto the frame grid (the same `reward_frames` vector used for the previous-trial input, 6-a). Reward amounts and the `autoreward` flag are not used, so auto-delivered rewards count as rewarded.

ii.
```python
        reward_times = np.asarray(beh.time_series["Reward"].timestamps[:], dtype=np.float64)
        reward_frames = rasterize_reward_events(frame_times, reward_times)
```

iii. Step 5 mapping: "Reward event presence per trial from `Reward` timestamps → `output[5]`: Binary per trial (`any reward event in trial`), repeat across frames. Rewarded includes auto-rewarded trials." Step 4 established that the reward event timestamps coincide exactly with frame timestamps.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. `any(reward_frames[start:stop] > 0)` per trial → 0/1, broadcast across all timepoints of the trial as output row 5. The resulting omission fraction is 0.157, consistent with the paper's ~15% random omission rate.

ii.
```python
def trial_reward_outcomes(bounds: list[tuple[int, int]], reward_frames: np.ndarray) -> np.ndarray:
    outcomes = np.zeros(len(bounds), dtype=np.int8)
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
    return outcomes
```

iii. Step 10: "Reward rate: paper/methods about 15% omitted, converted `reward_outcome=no` fraction `0.157`, consistent." Step 12 additionally verified an omission trial (`sub-m11_ses-03`, trial 1) and a rewarded trial against the raw event list, and argued the near-chance decoding of this variable follows from the per-trial constant target required by the instructions rather than from a conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases, mixing tolerant handling with hard failures:
- **Neural/behavior length mismatch**: exactly one extra neural frame is trimmed (occurs in 10 multi-plane sessions); any other mismatch raises `ValueError` and aborts the conversion.
- **Reward timestamps**: must coincide with a frame timestamp to within 1e-9 s, otherwise raise.
- **Frame interval**: must equal 1/15.5078125 s to within 1e-9 s, otherwise raise.
- **Pre-sync sentinels** (`environment = -1`, `position = -500`, `trial number = -1`): excluded implicitly because they lie outside trial windows; additionally, environment samples < 0 are dropped before taking the mode, a trial with no valid environment sample is skipped, and a negative `trial number` falls back to the loop index.
- **Non-finite values** in neural, position and speed are replaced with 0 by `np.nan_to_num` (a no-op in practice — the stored `Deconvolved` arrays contain no NaNs).
- **Degenerate trials/sessions**: `stop <= start` trials are skipped; a session yielding fewer than 2 usable trials raises.
- **Lick sensor failures**: handled by the artifact rule in 9-b.

ii.
```python
        frame_delta = deconv.shape[0] - frame_times.shape[0]
        if frame_delta == 1:
            deconv = deconv[: frame_times.shape[0], :]
        elif frame_delta != 0:
            raise ValueError(...)
        frame_dt_s = float(np.median(np.diff(frame_times)))
        if not np.isclose(frame_dt_s, EXPECTED_FRAME_DT_S, atol=1e-9, rtol=0):
            raise ValueError(...)
```
```python
        env_valid = env_trial[env_trial >= 0]
        if env_valid.size == 0:
            continue
        trial_num_native = int(round(float(trial_number[start])))
        if trial_num_native < 0:
            trial_num_native = trial_idx
        ...
        neural_trial = np.nan_to_num(deconv[start:stop, :].T, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. Step 10 "Check 5: edge cases" documents the one-frame mismatch and its fix ("trim the final neural frame only when `deconv_len == behavior_len + 1`; otherwise raise an error"), the `sub-m11_ses-03` extra `trial number` segment, and the decision to keep very long laps. Step 5 documents the sentinel strategy ("exclude pre-sync sentinel samples … by working only within extracted trial windows"). The hard-failure style is deliberate: the AI prefers to abort rather than silently accept an unexpected misalignment.

## 13-a. What are the most time-consuming steps of the code?

i. The dominant cost is NWB file I/O — opening each of the 152 files and reading the curated columns of the deconvolved array (0.28–0.93 s per session; 102 s summed over the full run, per `conversion_full_out.txt`). Secondary costs are (a) serializing the 4.6 GB pickle at the end, which is a substantial fraction of total wall-clock, and (b) in `--show-processing` mode, re-opening the first two NWB files and rendering an 8-panel matplotlib figure per session. The per-trial Python loop is negligible by comparison. Because the script skips the paper's dF/F + OASIS pipeline, the usual bottleneck of this conversion (deconvolution) is absent entirely.

ii.
```python
    t0 = time.perf_counter()
    for session_idx, file_path in enumerate(files):
        session_t0 = time.perf_counter()
        neural_trials, input_trials, output_trials, plane_per_cell, meta, stats = convert_session(file_path)
        ...
        session_elapsed = time.perf_counter() - session_t0
        print(f"[{session_idx + 1:03d}/{len(files):03d}] {meta.session_name}: "
              f"{len(neural_trials)} trials, {neural_trials[0].shape[0]} neurons, {session_elapsed:.2f}s")
```

iii. Step 6/Step 7: "Re-reading full sessions for plots would be expensive on the full dataset"; benchmarks are given for a small single-plane session (~0.44 s) and a large two-plane session (~0.93 s), with the full-run estimate "likely a few minutes; comfortably below 15 min" — which the actual run confirmed.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Four Python loops remain, all cheap but all vectorizable:
- `pair_trial_bounds` walks the start pulses with an inner `while` over stop pulses; with one teleport pulse per trial this is just `np.searchsorted(stops, starts, side='right')`.
- `trial_reward_outcomes` loops over trials to test `np.any` on a slice; this could be done with a cumulative sum of `reward_frames` differenced at the trial boundaries.
- The main per-trial loop recomputes `np.full(...)` broadcasts and calls `mode_int` (an `np.unique`) per trial; the discretizations (`discretize_reward_distance`, `discretize_position`, `discretize_speed`) and the lick rule could be applied once to the whole-session arrays and then split, since the zone interval is the only per-trial quantity involved.
- The plotting branch calls `pair_trial_bounds` and `scene_to_reward_labels` four separate times in one call expression.
None of this is a real bottleneck given that I/O dominates; the per-trial loop is also the natural structure for variable-length trials.

ii.
```python
    for start in starts:
        while stop_idx < len(stops) and stops[stop_idx] <= start:
            stop_idx += 1
```
```python
    for idx, (start, stop) in enumerate(bounds):
        outcomes[idx] = int(np.any(reward_frames[start:stop] > 0))
```
```python
                trial_bounds=pair_trial_bounds(trial_start, teleport),
                reward_zone_labels=scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport))),
                reward_zone_codes=np.array([reward_label_to_code(label) for label in scene_to_reward_labels(meta.scene, len(pair_trial_bounds(trial_start, teleport)))], dtype=np.int8),
```

iii. The notes do not enumerate these loops. Step 6 only lists the speed-ups adopted ("Trial segmentation uses pulse streams directly, avoiding extra per-frame grouping logic", "Only curated cells are read from NWB response series", "Neural data saved as `float16`"), on the grounds that the full conversion already runs far under the 15-minute budget.

## 13-c. What processing does the code repeat multiple times?

i. The conversion itself is a single pass over the data — there is no separate survey/statistics pass, because reward-zone identity comes from the scene string rather than from a data-driven segmentation. The repetition that does exist is confined to `--show-processing`: for each of the first two sessions the NWB file is opened a second time and `position`, `speed`, `lick`, `trial_start`, `teleport` and the reward rasterization are all recomputed, the trial bounds and scene labels are recomputed up to four times, and the per-trial lick correction is re-run over the whole session. `pair_trial_bounds` output is also thrown away and rebuilt rather than returned from `convert_session`.

ii.
```python
        if show_processing and session_idx < 2:
            full_lick = np.concatenate([trial[3] for trial in output_trials]).astype(np.int8)   # computed, never used
            with NWBHDF5IO(str(file_path), "r", load_namespaces=True) as io:
                nwb = io.read()
                beh = nwb.processing["behavior"]["BehavioralTimeSeries"]
                frame_times = np.asarray(beh.time_series["position"].timestamps[:], dtype=np.float64)
                ...
                reward_frames = rasterize_reward_events(frame_times, reward_times)
            corrected_full_lick = np.zeros_like(lick_counts, dtype=np.int8)
            for start, stop in pair_trial_bounds(trial_start, teleport):
                corrected_full_lick[start:stop], _ = correct_lick_trial(lick_counts[start:stop])
```

iii. Step 6 lists "Re-reading full sessions for plots would be expensive on the full dataset" as a known inefficiency and mitigates it by limiting plots to the first two sessions ("Sample/full plots are limited to at most two sessions"), which keeps the cost bounded rather than eliminating it.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A few small items:
- `load_curated_deconvolved` builds a per-cell plane-index array (`plane_per_cell`), which `convert_session` returns and `build_dataset` then ignores — `brain_region_idx` is filled with zeros by `session_brain_region_idx`, so plane identity never reaches the output (it survives only as `n_planes` in `session_info`).
- `np.nan_to_num` is applied to every neural, position and speed trial although the stored arrays contain no non-finite values.
- In the plotting branch, `full_lick` and `trial_reward` are computed and never used, and trial bounds/labels are rebuilt several times (see 13-c).
- `stats` accumulates per-session extras (`reward_zone_codes` for every trial, `trial_numbers_kept`, `mean_trial_len`, `bad_lick_trials`) that are stored in `metadata['session_info']` but unused by the decoder.
- Neural data is written as `float16` and then converted back to float by the decoder; this halves file size but discards precision that the downstream model would otherwise use.
None of these materially affect runtime.

ii.
```python
def session_brain_region_idx(nneurons: int) -> np.ndarray:
    return np.zeros(nneurons, dtype=np.int16)
...
        neural_trials, input_trials, output_trials, plane_per_cell, meta, stats = convert_session(file_path)
        ...
        plane_all.append(session_brain_region_idx(neural_trials[0].shape[0]))   # plane_per_cell unused
```
```python
    trial_reward = output_trials[0][5, 0:1]        # never used
    reward_outcomes = np.array([trial[5, 0] for trial in output_trials], dtype=int)
```

iii. Key Decision 10 explains why plane identity is not a brain region: "Brain region will be encoded as `CA1` for all cells: Deep/superficial plane identity is plane metadata, not a distinct named brain region in the sense of the target schema", with the metadata note "All neurons are from hippocampal CA1; multi-plane identity is stored in session_info." The `float16` choice is justified in Step 6 as a storage optimization and re-examined in Step 10, where a failing neural sanity check was traced to float16 quantization and the check was relaxed to a float16 round-trip comparison. The remaining dead computations are not discussed in the notes.
