# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file with a sorted glob over `data/sub-*/sub-*_behavior+ophys.nwb` (152 files found, all processed in the full run) and reads each one **directly with `h5py`** rather than `pynwb`, for speed. For each file it reads one ophys group (`processing/ophys/Deconvolved/plane0`) and the eleven behavioural time series under `processing/behavior/BehavioralTimeSeries` (`position`, `speed`, `lick`, `environment`, `reward_zone`, `scanning`, `trial number`, `trial_start`, `teleport`, plus `Reward` timestamps). Neural and behavioural streams are cropped to a common length `t_common = min(n_neural_frames, n_behaviour_samples)`. Trials are then cut out of these session-long arrays. No subject, session or trial is excluded at the loading stage.

ii.
```python
def list_session_files() -> list[str]:
    files = sorted(glob.glob(os.path.join("data", "sub-*", "sub-*_behavior+ophys.nwb")))
    if not files:
        raise FileNotFoundError("No NWB files found under data/sub-*/")
    return files
```
```python
def load_session_arrays(path: str) -> SessionArrays:
    with h5py.File(path, "r") as f:
        subject = decode_if_bytes(f["general/subject/subject_id"][()])
        session_id = decode_if_bytes(f["general/session_id"][()])
        session_label = f"{subject}_ses-{session_id}"

        beh = f["processing/behavior/BehavioralTimeSeries"]
        neural_group = f["processing/ophys/Deconvolved/plane0"]
        seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
        ...
        t_neural = neural.shape[0]
        t_behavior = beh["position/data"].shape[0]
        t_common = min(t_neural, t_behavior)
```

iii. From CONVERSION_NOTES Step 2/Step 6: the archive is "one NWB file per subject-session plus `dandiset.yaml`", giving `152` files over 11 subject directories, which the AI cross-checked against the paper ("11 switch-task mice", "14 imaging days", "m11 imaging started on day 3" → 11×14−2 = 152). Direct `h5py` reads were chosen "instead of heavier NWB object materialization" as a documented speed-up (full conversion 95.11 s). The verification log confirms all 152 sessions, 11 subjects, 12,147 trials were written.

## 1-b. How are the data split into subjects?

i. Subject identity is read from inside each NWB file (`general/subject/subject_id`), not from the directory name. Subjects are added to `data['subjects']` in order of first appearance while iterating the sorted file list, and `subject_idx` records the index for each session. This yields the 11 mice `m11, m12, m13, m14, m15, m17, m18, m19, m3, m4, m7` (glob/string sort order), matching the reference solution's subject list exactly.

ii.
```python
if arrays.subject not in subject_to_idx:
    subject_to_idx[arrays.subject] = len(all_subjects)
    all_subjects.append(arrays.subject)
...
data["subject_idx"].append(subject_to_idx[arrays.subject])
...
data["subjects"] = all_subjects
data["subject_idx"] = np.asarray(data["subject_idx"], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2 lists the 11 `sub-*` directories and confirms `11` subjects against the paper's "switch task (n = 11 mice)". Taking the id from the file metadata rather than the path is the more authoritative source and was verified to reproduce the directory-derived names.

## 1-c. How are the data split into sessions?

i. One session = one NWB file. Sessions are processed in sorted-path order, so sessions are grouped by subject and ordered by session number within subject. No pooling or cross-day alignment of neurons is attempted; each session keeps its own neuron set and its own `brain_region_idx`.

ii.
```python
for session_idx, path in enumerate(files):
    arrays = load_session_arrays(path)
    neural_trials, input_trials, output_trials, brain_region_idx, session_info = convert_session(...)
    data["neural"].append(neural_trials)
    data["input"].append(input_trials)
    data["output"].append(output_trials)
    data["brain_region_idx"].append(brain_region_idx)
```

iii. CONVERSION_NOTES Step 2/Step 4: "`data/` contains one NWB file per subject-session", `152` files = "11 mice x 14 days minus m11 day1-2", consistent with the paper's 14 imaging days and m11 starting on day 3. Session counts per subject in `verification_full_out.txt` (m11: 12, all others: 14) match the file inventory.

## 1-d. Are the data correctly split into trials?

i. Trials are reconstructed from the two binary behaviour markers: a trial starts at each sample where `trial_start > 0.5` and ends at the following `teleport > 0.5` sample, which is **included** in the trial (`stop = teleport_idx + 1`). Starts and teleports are paired positionally after truncating both to the shorter list, and any pair with `stop < start` is silently skipped; a session with fewer than 2 resulting trials raises. This produces 12,216 raw trials (mean 80.4/session), the same trial set the expert reference produces.

ii.
```python
def build_trial_slices(trial_start_signal, teleport_signal):
    starts = np.flatnonzero(trial_start_signal > 0.5)
    teleports = np.flatnonzero(teleport_signal > 0.5)
    if starts.size == 0 or teleports.size == 0:
        raise RuntimeError("Missing trial_start or teleport markers.")
    n = min(starts.size, teleports.size)
    slices = []
    for start, stop in zip(starts[:n], teleports[:n]):
        if stop < start:
            continue
        slices.append((int(start), int(stop) + 1))
    if len(slices) < 2:
        raise RuntimeError("Need at least two trials in a session.")
    return slices
```

iii. CONVERSION_NOTES Step 4/Step 5: "Code uses `trial_start_inds` and `teleport_inds` arrays in `sess`… Reconstruct trial start/end indices from `trial_start` and `teleport` signals in NWB", chosen over the stored `trial number` because it "matches the reference trial definition more closely than relying only on trial-number changes". Step 10 records a per-session check that `n_trials_raw − n_trials_lick_artifact_removed == n_trials_kept`.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level quality filter is applied: a trial is dropped if more than 35% of its frames have a cumulative lick count `> 2`, which is the erroneous-lick-detection rule from the paper's `glmUtils.get_timeseries_data`. This removed 69 of 12,216 trials (0.56%), close to the paper's reported 81/12,376 (0.65%). No minimum-trial-length filter and no other trial exclusion is applied (the shortest converted trial is 63 bins, so a 50-sample floor would have removed nothing).

ii.
```python
LICK_ARTIFACT_FRACTION = 0.35  # Matches glmUtils.get_timeseries_data in the reference code.
...
lick_trial = arrays.lick[start:stop_exclusive]
if lick_trial.size and np.mean(lick_trial > 2) > LICK_ARTIFACT_FRACTION:
    lick_artifact[trial_id] = True
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    if lick_artifact[trial_id]:
        continue
```

iii. CONVERSION_NOTES Step 3: "Licking analysis removes trials with erroneous lick detection when >30% of imaging frames in the trial have cumulative lick count `> 2`… Reported lick-artifact removal: `81 / 12,376`". The threshold constant `0.35` is taken verbatim from the reference code path (`sum(licks[start:stop] > 2)/len(...) > 0.35`), which the trajectory shows the AI read. Step 9 documents the resulting 0.56% removal rate as a consistency check against the paper.

## 2-a. What variables in the raw data is the final `neural` data derived from?

i. `neural` is the archive's stored suite2p deconvolution, `processing/ophys/Deconvolved/plane0/data`, restricted to ROIs listed in `Deconvolved/plane0/rois` whose `iscell[:,0] > 0.5`. The raw `Fluorescence` and `Neuropil` series are never read. For the two-plane mice (m17, m18; 28 sessions) **only `plane0` is read — the `plane1` response series is not loaded at all** (e.g. m17 ses-01 has 236 curated plane-0 cells and 355 curated plane-1 cells; only the 236 are kept). Total converted neurons: 118,493 (expert reference: 138,298).

ii.
```python
neural_group = f["processing/ophys/Deconvolved/plane0"]
seg = f["processing/ophys/ImageSegmentation/PlaneSegmentation"]
...
neural = neural_group["data"][()].astype(np.float32, copy=False)
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
roi_ids = roi_ids[iscell]
```

iii. CONVERSION_NOTES Step 4: "Use NWB `processing/ophys/Deconvolved/plane0/data` as the paper-equivalent neural signal", because "`glmUtils.get_timeseries_data` uses `sess.timeseries['events']` after dF/F + OASIS" and "the paper decoder uses deconvolved calcium activity". For the two-plane mice it records the (mistaken) conclusion: "The archive exposes only the plane-0 response series for these files. Conversion will use the provided response matrix and document this archive-level limitation." Step 5 also notes the ROI-region-aware `iscell` indexing was adopted to avoid counting plane-1 ROIs, which it treated as absent from the data rather than as a second series to load.

## 2-b. How is the `neural` data processed?

i. Essentially no processing: the stored deconvolved traces are cast to `float32`, transposed to (neurons, time), sliced per trial, and — for the 28 two-plane sessions — summed in adjacent pairs of frames. There is **no neuropil subtraction, no per-trial maximin dF/F baseline, no 2-sample Gaussian smoothing and no OASIS deconvolution at `tau = 0.7`**, i.e. none of the `preprocessing.dff` pipeline the paper's Methods describe and the reference code implements.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
...
neural_trials.append(neural_trial.astype(np.float32, copy=False))
```

iii. CONVERSION_NOTES Step 1 concludes "**Yes, dF/F must be computed** to match the reference processing. The decoder-relevant neural stream is most likely the deconvolved `events`". Step 4 then resolves the discrepancy the other way: "NWB stores `Fluorescence`, `Neuropil`, and `Deconvolved`; no explicit dF/F timeseries → Use NWB `Deconvolved/plane0/data` as the paper-equivalent neural signal. Raw `Fluorescence`/`Neuropil` remain available for sanity checks." Step 10(b) restates this as "exact dF/F-speed-correlation interneuron exclusion was not re-run because the archive exposes deconvolved responses but not the exact reference dF/F stream".

## 2-c. How is the `neural` data filtered based on quality controls?

i. A single filter: suite2p's manual curation flag, `iscell[:,0] > 0.5`, applied to the ROI ids referenced by the plane-0 response matrix. The paper's second neuron filter — dropping putative interneurons whose dF/F correlates with running speed at `r > 0.5` — is **not** applied. (Separately, plane-1 cells are lost for m17/m18; see 2-a.) Resulting counts: 118,493 cells, 155–1780 per session, mean 779.6.

ii.
```python
roi_ids = neural_group["rois"][()].astype(np.int64, copy=False)
iscell = seg["iscell"][()][roi_ids, 0] > 0.5
neural = neural[:, iscell]
```

iii. CONVERSION_NOTES Step 3 documents both rules ("Manual suite2p curation removed ROIs… Additional putative interneurons were excluded if Pearson correlation between dF/F and running speed exceeded `0.5`… `0.42 ± 0.85%` of cells"). Step 5 planned to implement the interneuron filter "if feasible from the NWB fluorescence streams and reference `dff` logic". Step 10 then drops it: "documented as an archive-level limitation. The shared NWB files provide the deconvolved response stream used for decoding, but not the exact reference dF/F timeseries needed to reproduce that curation step bit-for-bit. Given that converted neuron counts already fall within the paper's reported range, this difference was not forced with a surrogate filter."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to trial start needs no extra work: neural and behavioural samples share one frame index, and each trial's arrays simply start at the `trial_start` frame (bin 0 = alignment event) and run to the teleport frame inclusive. `metadata['temporal_alignment_event'] = 'trial_start'`, `off_start = 0.0`, `off_end = None`. No pre-trial baseline window is included.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    neural_trial = arrays.neural[start:stop_exclusive].T...
    position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
    ...
    time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```
```python
"temporal_alignment_event": "trial_start",
"off_start": 0.0,
"off_end": None,
```

iii. CONVERSION_NOTES Step 5: "Align all trials to the first frame of the trial", relying on the fact (Step 1/Step 4) that `append_session_data` produces frame-aligned behaviour and imaging, so the same index slices both streams. Step 10's raw-vs-converted spot checks (m11 ses-03 trial 0; m17 ses-08 trial 0; m11 ses-08 trial 30) were used to confirm indexing.

## 2-e. How is the `neural` data temporally binned/resampled?

i. The AI declares one common bin of `64.483 ms` (15.5078125 Hz) for the whole dataset. Sessions whose `Deconvolved` series carries `rate = 15.5078125` are left untouched; sessions whose series carries `rate = 31.015625` (m17, m18 — 28 sessions) are **rebinned by a factor of 2**: neural frames summed in pairs, position/speed averaged in pairs, lick OR-ed in pairs. Any other rate raises.

In fact the stored `rate` for those files is the scanner rate for the two interleaved planes; each plane's series already has one sample per behavioural sample (m17 ses-01: 22,634 neural frames and 22,634 behaviour samples, behaviour `dt = 0.06448 s = 15.5 Hz`). So the factor-2 rebin halves the true resolution of those 28 sessions to ~7.75 Hz (128.97 ms bins) while the metadata still reports 64.483 ms.

ii.
```python
if math.isclose(arrays.rate_hz, TARGET_RATE_HZ):
    factor = 1
elif math.isclose(arrays.rate_hz, 2.0 * TARGET_RATE_HZ):
    factor = 2
else:
    raise RuntimeError(f"Unexpected sampling rate {arrays.rate_hz} in {arrays.session_label}")
```
```python
def rebin_2d_sum_time_last(x, factor):
    ...
    full = x[:, : n_full * factor].reshape(x.shape[0], n_full, factor).sum(axis=2)
```
```python
"time_bin_size": 1000.0 * TARGET_DT_S,
"common_sampling_rate_hz": TARGET_RATE_HZ,
```

iii. CONVERSION_NOTES Step 5: "**Standardize all sessions to a common `15.5078125 Hz` bin size**: This is the dominant dataset rate and the paper's effective per-plane sampling rate", motivated by the format requirement that "Time bins should be the same size for all trials and sessions". Step 3 records the correct fact ("m17 and m18 were imaged in two planes interleaved at ~31 Hz, yielding ~15.5 Hz per plane") but the conversion treats the stored 31 Hz attribute as the sampling rate of the plane-0 array. Step 10's sanity check for a 31 Hz session compared the converted trial against a *manually pair-summed* raw slice, so it validated the rebinning against itself rather than against the behaviour timestamps.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. Not from the stored `timestamps` at all: it is synthesised from the bin index times the nominal bin width (`TARGET_DT_S = 1/15.5078125 s`). The behavioural `timestamps` array is loaded (and used for mapping reward events) but is not used to build this input.

ii.
```python
TARGET_RATE_HZ = 15.5078125
TARGET_DT_S = 1.0 / TARGET_RATE_HZ
...
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
```

iii. CONVERSION_NOTES Step 5 maps "Reconstructed trial-relative elapsed time → `input[0]`: Seconds since trial start for each bin". The implicit justification is that sampling is uniform at the common rate; independent checks confirm this — across all 152 files the maximum inter-sample interval equals the nominal `0.0644836 s`, so index × dt reproduces the stored timestamps exactly for single-plane sessions.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. The vector is `[0, dt, 2·dt, …]` for each trial, with `dt` fixed at 64.483 ms, so every trial begins at exactly 0.0 s. Because the 28 two-plane sessions were rebinned by 2 (see 2-e) but still use `dt = 64.483 ms`, their elapsed times are understated by a factor of 2 — e.g. m17 ses-01 has a true mean trial duration of 13.3 s and maximum 30.4 s, but the converted input reports a maximum of 15.2 s for that session.

ii.
```python
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
...
input_trial = np.stack([time_from_start, env_series, trial_series, prev_reward_series], axis=0).astype(np.float32, copy=False)
```

iii. Step 5 ("Seconds since trial start for each bin") and Step 7/Step 9 report the per-session ranges (`[0.0, 20.8]` in the sample, `[0.0, 216.6]` overall) as sanity checks; the halved range for m17/m18 sessions was not flagged. Step 10's input spot check was performed on `m11_ses-08`, a single-plane session, so it could not surface the 2× error.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction: `t_bins` is taken from the neural trial's own second dimension, so the time vector always has exactly the same length as the neural matrix and starts at the same bin. Neural and behaviour are pre-cropped to a common length at load time, so the index correspondence holds throughout.

ii.
```python
t_bins = neural_trial.shape[1]
time_from_start = (np.arange(t_bins, dtype=np.float32) * TARGET_DT_S)
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```
```python
t_common = min(t_neural, t_behavior)
```

iii. CONVERSION_NOTES Step 4: neural and behaviour are frame-aligned in the archive; Step 10 notes "handled 10 sessions with a 1-sample neural/behavior length mismatch by cropping to the common minimum length before trial parsing". Deriving every per-trial array's length from the neural matrix guarantees the format checker's dimension requirements are met.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The `environment` behaviour time series (values −1/0/1, where −1 marks samples outside valid trial/imaging periods).

ii.
```python
environment=beh["environment/data"][()][:t_common].astype(np.float32, copy=False),
...
env = arrays.environment[start:stop_exclusive]
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
```

iii. CONVERSION_NOTES Step 2 documents the observed coding ("`environment`: values `-1, 0, 1` (`-1` appears outside valid trial/imaging periods)") and Step 5 maps it to `input[1]` with "Map `0 -> ENV1`, `1 -> ENV2`", matching the paper's two switch-task environments.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Per trial, the modal value over the trial's non-negative samples is taken and broadcast as a constant across all time bins of that trial (and it is a decoder input, not an output). No other transformation.

ii.
```python
def mode_int(values: np.ndarray, default: int = 0) -> int:
    if values.size == 0:
        return default
    values = values.astype(np.int64, copy=False)
    values = values[values >= 0]
    if values.size == 0:
        return default
    return int(np.bincount(values).argmax())
...
env_series = np.full((t_bins,), float(trial_env[trial_id]), dtype=np.float32)
```

iii. Step 5: "Per trial, take modal valid environment value (`0` or `1`) and repeat across bins", with the modal/valid-only rule chosen to be robust to the `-1` sentinel found in Step 2. (Independent check: within converted trials the environment is in fact constant and never −1, so this is equivalent to copying the per-sample values.) The full-run per-session ranges in `verification_full_out.txt` show blocks of 0-only and 1-only sessions with a single switch session per block, as the task design predicts.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. From the position of the trial in the reconstructed trial list — i.e. the enumeration index of `trial_slices`, which is built from `trial_start`/`teleport`. The stored `trial number` series is not used for this input (it is used only to attach reward events to trials).

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    ...
    trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5: "Per trial, use within-session trial index and repeat across bins… Keep native 0-indexing to match code conventions (`glmUtils.get_timeseries_data` creates 0-indexed `trial_ids`)". The converted range is `[0, 99]`, consistent with the 80–100 trials/session the paper targets.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond broadcasting the 0-based index across the trial's time bins. Note that the index refers to the *raw* trial position, so a trial dropped by the lick-artifact filter leaves a gap in the numbering rather than renumbering the survivors — which keeps the value a faithful measure of within-session progress.

ii.
```python
trial_series = np.full((t_bins,), float(trial_id), dtype=np.float32)
```

iii. Step 5 key decision: "Keep within-session trial number 0-indexed: This matches raw NWB values and `glmUtils.get_timeseries_data`."

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the sparse `Reward` event series. Each reward timestamp is mapped to the nearest behaviour frame with `np.searchsorted`, the `trial number` value at that frame gives the trial the reward belongs to, and the resulting set of rewarded trial ids is shifted by one trial.

ii.
```python
def reward_frames_for_session(arrays):
    if arrays.reward_times.size == 0:
        return np.empty((0,), dtype=np.int64)
    idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
    return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)
...
reward_frame_idx = reward_frames_for_session(arrays)
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
trial_rewarded = np.array([1 if trial_id in rewarded_trial_ids else 0
                           for trial_id in range(len(trial_slices))], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 2 notes "`Reward`: sparse event series with separate timestamps (not frame-length)", so the timestamps must be searched into the behaviour clock; Step 5 maps "Trial reward outcome shifted by one trial → `input[3]`" and notes this is a "User-requested decoder input, not a paper decoder variable".

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. `prev[t] = rewarded[t-1]`, with `prev[0] = 0` for the first trial of the session, broadcast constant across the trial's bins. The shift is over raw trial indices, so a trial removed by the lick filter still counts as the predecessor of the trial that followed it.

ii.
```python
trial_prev_rewarded = np.concatenate([[0], trial_rewarded[:-1]]).astype(np.float32, copy=False)
...
prev_reward_series = np.full((t_bins,), float(trial_prev_rewarded[trial_id]), dtype=np.float32)
```

iii. Step 5: "Previous trial rewarded (`1`) vs omitted (`0`), repeated across current-trial bins; first trial gets `0`", which is exactly the coding the instructions specify. The rewarded/omitted base rate (84.1%/15.9%) was checked against the paper's "~15% omission".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From `position` (clipped to `[0, 450]` cm) together with the trial's inferred reward-zone identity. Zone bounds are hard-coded from the paper: A `[80, 130]`, B `[200, 250]`, C `[320, 370]` cm — identical to the reference's `reward_zone_dict`. The zone identity itself comes from the `reward_zone` behaviour series (see 10-a).

ii.
```python
ZONE_BOUNDS = {
    0: (80.0, 130.0),   # A
    1: (200.0, 250.0),  # B
    2: (320.0, 370.0),  # C
}
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
zone_idx = int(trial_zone[trial_id])
dist = compute_distance_to_zone(position, zone_idx)
```

iii. CONVERSION_NOTES Step 1/Step 3: "Reward-zone labels come from `behavior.get_reward_zones`: `A` = zone `X` = `[80, 130]`, `B` = `Y` = `[200, 250]`, `C` = `Z` = `[320, 370]`", cross-checked against the Methods quote "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm".

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed distance to the nearest edge of the active zone, per time bin: `position − zone_start` when before the zone (negative), `position − zone_end` when past it (positive), and exactly 0 anywhere inside the zone. Computed on the (rebinned, clipped) per-trial position vector.

ii.
```python
def compute_distance_to_zone(position_cm: np.ndarray, zone_idx: int) -> np.ndarray:
    zone_start, zone_end = ZONE_BOUNDS[zone_idx]
    distance = np.zeros(position_cm.shape, dtype=np.float32)
    before = position_cm < zone_start
    after = position_cm > zone_end
    distance[before] = position_cm[before] - zone_start
    distance[after] = position_cm[after] - zone_end
    return distance
```

iii. Step 5: "Signed distance to nearest point in active reward zone: negative before zone, `0` inside zone, positive after zone… For zone `[start, end]`: `pos < start -> pos-start`; `start <= pos <= end -> 0`; `pos > end -> pos-end`", which is the literal reading of "distance to any location in the reward zone" in the decoder spec.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit boolean masks rather than `np.digitize`: `<−50 → 0`, `[−50,−10) → 1`, `[−10,0) → 2`, `==0 → 3`, `(0,10] → 4`, `(10,50] → 5`, `>50 → 6`. (Edge conventions at ±10 and ±50 are inclusive-on-the-inner-side, the mirror image of the reference's `np.digitize` edges; only samples landing exactly on ±10/±50 differ.) Resulting distribution `[0.251, 0.102, 0.074, 0.238, 0.021, 0.071, 0.243]` is within ~0.002 of the expert reference at every class.

ii.
```python
def discretize_distance(distance_cm: np.ndarray) -> np.ndarray:
    out = np.zeros(distance_cm.shape, dtype=np.int64)
    out[distance_cm < -50.0] = 0
    out[(distance_cm >= -50.0) & (distance_cm < -10.0)] = 1
    out[(distance_cm >= -10.0) & (distance_cm < 0.0)] = 2
    out[distance_cm == 0.0] = 3
    out[(distance_cm > 0.0) & (distance_cm <= 10.0)] = 4
    out[(distance_cm > 10.0) & (distance_cm <= 50.0)] = 5
    out[distance_cm > 50.0] = 6
    return out
```

iii. Step 5: the bins are taken directly from the Decoder Task specification ("0: < −50 cm … 6: > +50 cm"); the separate `== 0.0` class exists because "inside the zone" is exactly 0 by construction of `compute_distance_to_zone`. Step 7/Step 9 report the class fractions as a sanity check.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same trial slice, same bin grid: position is taken with the identical `[start, stop)` index range as the neural matrix and, when `factor == 2`, is rebinned with the same factor (mean over the pair) so the two streams stay bin-for-bin aligned; the output array's length is the neural array's `t_bins`.

ii.
```python
neural_trial = arrays.neural[start:stop_exclusive].T.astype(np.float32, copy=False)
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    neural_trial = rebin_2d_sum_time_last(neural_trial, factor)
    position = rebin_1d_mean(position, factor)
```

iii. Step 4: neural and behaviour are frame-aligned in the archive, so a shared index range is sufficient; Step 10's output spot check on `m11_ses-08` trial 30 reconstructed the distance bins from raw NWB and reported `np.allclose(...) == True`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The `position` behaviour time series (cm along the VR corridor), clipped to `[0, 450]`.

ii.
```python
position=beh["position/data"][()][:t_common].astype(np.float32, copy=False),
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
```

iii. Step 5 maps `position/data` directly to `output[1]`; Step 3 records the paper's "450 cm linear track", which sets the clip bound and the bin width.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. Clip to `[0, 450 − 1e−6]`, mean-rebin by 2 on the two-plane sessions, then integer-divide by 90 cm and cap at bin 4. Values slightly outside the track (position reaches −50 at the teleport frame and ~452 at the end of a lap) are absorbed into the first/last bin by the clip.

ii.
```python
TRACK_LENGTH_CM = 450.0
def discretize_position(position_cm: np.ndarray) -> np.ndarray:
    clipped = np.clip(position_cm, 0.0, TRACK_LENGTH_CM - 1e-6)
    return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. Step 5: "Discretize absolute position on the 450 cm corridor into 5 equal bins… Decoder spec requires 5 bins, not the paper's 45 bins." The resulting distribution `[0.212, 0.175, 0.230, 0.226, 0.156]` is within 0.003 of the expert reference in every bin.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five equal 90 cm bins over the 450 cm track (`0: <90, 1: 90–180, 2: 180–270, 3: 270–360, 4: ≥360`), implemented as `floor(clip(pos)/90)` capped at 4 — equivalent to the specified edges with open ends.

ii.
```python
return np.minimum((clipped / (TRACK_LENGTH_CM / 5.0)).astype(np.int64), 4)
```

iii. Directly from the Decoder Task spec ("Discretized into 5 equal-sized bins spanning the 450 cm track"); the clip guarantees no sixth class is created by out-of-range samples.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same per-trial index range and same rebinning factor as the neural data; array length is set by `t_bins` from the neural matrix.

ii.
```python
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
if factor == 2:
    position = rebin_1d_mean(position, factor)
pos_bin = discretize_position(position)
```

iii. Same frame-alignment argument as 7-d, verified by the Step 10 raw-vs-converted spot check and by the `--show-processing` plots ("No obvious temporal misalignment or discretization artifacts").

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The `lick` behaviour time series (a per-frame lick count that can exceed 1).

ii.
```python
lick=beh["lick/data"][()][:t_common].astype(np.float32, copy=False),
...
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
```

iii. Step 5 maps `lick/data` to `output[3]`: "Convert to binary per bin (`lick > 0`) after lick-artifact handling", following the Methods' "licking is converted to binary counts".

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarised at `> 0`. On the two-plane sessions the binary series is then rebinned with a logical OR over each pair of frames (a bin is 1 if either constituent frame had a lick), which slightly raises the lick rate in those sessions relative to the rest. Whole trials with pathological lick counts are dropped beforehand (1-e).

ii.
```python
lick_binary = (arrays.lick[start:stop_exclusive] > 0).astype(np.int64, copy=False)
if factor == 2:
    lick_binary = rebin_1d_any(lick_binary, factor)
```
```python
def rebin_1d_any(x, factor):
    ...
    out.append(np.any(x[: n_full * factor].reshape(n_full, factor) > 0, axis=1))
```

iii. Step 5: "For rebinned 31 Hz sessions, use logical OR within each 15.5 Hz bin", chosen so that brief licks are not averaged away. The converted lick distribution `[0.767, 0.233]` matches the expert reference `[0.770, 0.230]`.

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same trial slice and same rebin factor as the neural data, so lick bins are the neural bins.

ii.
```python
output_trial = np.stack([dist_bin, pos_bin, speed_bin, lick_binary, zone_bin, reward_bin], axis=0).astype(np.int64, copy=False)
```

iii. Frame alignment as in 7-d/8-d; Step 12 explicitly re-examined lick alignment ("Reviewed sample/full processing plots: lick transients are temporally aligned to trial time and position") when its decoding accuracy came in below the 1.5× chance heuristic.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the `reward_zone` behaviour series combined with `position`: for each trial the median position over samples with `reward_zone > 0` is matched to the nearest of the three zone centres (105/225/345 cm). If a trial never has `reward_zone > 0`, the position at the trial's reward event is used as a fallback; if there is still nothing, the label is filled from the block majority. Rewards/`trial number` are used only for that fallback.

ii.
```python
ZONE_CENTERS = np.array([(lo + hi) / 2.0 for lo, hi in ZONE_BOUNDS.values()], dtype=np.float32)
...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    mask = reward_zone_signal[start:stop_exclusive] > 0
    if np.any(mask):
        zone_pos = float(np.nanmedian(positions[start:stop_exclusive][mask]))
    elif trial_id in reward_pos_by_trial:
        zone_pos = reward_pos_by_trial[trial_id]
    else:
        zone_pos = np.nan
    if np.isfinite(zone_pos):
        observed_position[trial_id] = zone_pos
        observed[trial_id] = int(np.argmin(np.abs(ZONE_CENTERS - zone_pos)))
```

iii. Step 5: "Infer from reward-zone-active positions / reward events and fill omission trials within stable blocks", because "Missing zone labels in the NWB export are concentrated on omission trials; the task structure constrains the active zone to one contiguous block per condition."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Unobserved trials are filled using the paper's task structure: the session is split at trial index 30 (`SWITCH_TRIAL_INDEX`), and each half is filled with the majority observed label of that half. If the session contains two environments, both halves are additionally *forced* to their majority label. More than two distinct zones in a session, or none at all, raises an error. The label is emitted as a constant time series per trial, coded `A=0, B=1, C=2`.

ii.
```python
SWITCH_TRIAL_INDEX = 30
...
if unique_observed.size == 1:
    filled[filled < 0] = int(unique_observed[0])
    return filled, observed_position
if unique_observed.size > 2:
    raise RuntimeError(f"Observed more than two reward zones within one session: {unique_observed.tolist()}")
split = min(SWITCH_TRIAL_INDEX, n_trials)
pre_majority = trial_majority_zone(observed, 0, split, fallback=int(unique_observed[0]))
post_fallback = int(unique_observed[1] if unique_observed.size > 1 else pre_majority)
post_majority = trial_majority_zone(observed, split, n_trials, fallback=post_fallback)
filled[:split][filled[:split] < 0] = pre_majority
filled[split:][filled[split:] < 0] = post_majority
if unique_env.size > 1 and split < n_trials:
    filled[:split] = pre_majority
    filled[split:] = post_majority
```

iii. Step 3/Step 5: "Reward switch timing: switch after `30` trials" ("Each switch occurred after 30 trials"), and "switch sessions should change condition near trial 30" was adopted as a sanity anchor. The converted zone distribution `[0.328, 0.338, 0.334]` matches the expert reference `[0.329, 0.337, 0.334]`. (Independent check: the inferred labels agree with the directly observed label on all 10,394 trials where `reward_zone` is ever active, and in all 77 two-zone sessions the observed switch straddles trial 30, so the hard-coded split never contradicts the data here.)

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The `Reward` event timestamps, mapped to behaviour frames via `searchsorted` and then to a trial via the `trial number` value at that frame.

ii.
```python
idx = np.searchsorted(arrays.timestamps, arrays.reward_times, side="left")
return np.clip(idx, 0, arrays.timestamps.shape[0] - 1).astype(np.int64, copy=False)
...
rewarded_trial_ids = set(int(t) for t in arrays.trial_number[reward_frame_idx] if t >= 0)
```

iii. Step 5: "Rewarded if any `Reward` event timestamp falls within the trial", implemented through the `trial number` channel rather than through the trial index ranges. Step 2 established that `Reward` has its own timestamps and is not frame-length.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Binary per trial (`1` if the trial received any reward event, else `0`), broadcast across all of the trial's time bins. No use of reward amount (the AI checked amounts are uniform) and no special handling of automatic rewards.

ii.
```python
trial_rewarded = np.array([1 if trial_id in rewarded_trial_ids else 0
                           for trial_id in range(len(trial_slices))], dtype=np.int64)
...
reward_bin = np.full((t_bins,), int(trial_rewarded[trial_id]), dtype=np.int64)
```

iii. Step 5/Step 9: the resulting omission fraction (15.9%) was checked against the raw-data fraction (15.35%) and the paper's "Reward was randomly omitted on approximately 15% of trials". (Independent check: this labelling is identical to the time-window method on all 12,216 trials.)

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Neural/behaviour length mismatch** (10 sessions off by one sample): both streams are cropped to `min(t_neural, t_behavior)` before any trial parsing.
- **`-1` sentinels** in `environment`/`trial number`: excluded before taking the per-trial mode, and reward events landing on a `trial number < 0` frame are ignored (never happens in practice).
- **Out-of-range behaviour**: position clipped to `[0, 450]`, speed clipped at 0.
- **Missing `reward_zone` activation** (1,822 of 12,216 trials): falls back to the reward-event position, then to the pre/post-trial-30 block majority; a session with no observable zone at all raises.
- **Unpaired trial markers**: starts and teleports are truncated to the shorter list and `stop < start` pairs are skipped; sessions ending with <2 usable trials raise.
- **Lick-sensor artifacts**: whole trials dropped (1-e).
Not handled: there is no minimum-trial-length filter (none is needed — the shortest trial is 63 bins), and the teleport frame itself is included in each trial, so in ~12% of trials the final sample's position has already jumped to −50 cm (clipped to 0) before the trial is cut.

ii.
```python
t_neural = neural.shape[0]
t_behavior = beh["position/data"].shape[0]
t_common = min(t_neural, t_behavior)
```
```python
env = env[env >= 0]
trial_env[trial_id] = mode_int(env, default=0)
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
speed = np.clip(arrays.speed[start:stop_exclusive], 0.0, None)
```
```python
if len(neural_trials) < 2:
    raise RuntimeError(f"{arrays.session_label}: fewer than 2 valid trials after filtering.")
```

iii. Step 2 found the `-1` sentinels and the 10 off-by-one sessions ("most sessions have equal behavior/neural frame counts; 10 sessions are off by exactly 1 sample… so edge handling will be required"); Step 10's edge-case checks record "handled 10 sessions with a 1-sample neural/behavior length mismatch by cropping to the common minimum length before trial parsing" and "confirmed `reward_zone_location` and `reward_outcome` are constant within each converted trial".

## 13-a. What are the most time-consuming steps of the code?

i. The dominant cost is file I/O: reading each session's `Deconvolved` matrix and eleven behaviour arrays out of HDF5 (0.4–1.4 s/session), plus pickling the 7.3 GB output. Everything else (trial slicing, discretisation, zone inference) is milliseconds. The whole 152-session conversion runs in 95.11 s in a single pass, with no OASIS/dF/F computation to pay for. In `--sample` mode there is an extra cost: `infer_sample_files` opens all 152 files just to pick 2.

ii.
```python
session_t0 = time.perf_counter()
arrays = load_session_arrays(path)
...
print(f"    kept {len(neural_trials)} trials, mean trial bins={mean_t:.2f}, "
      f"session conversion time={elapsed:.2f}s")
...
print(f"Processed {len(files)} sessions in {total_elapsed:.2f}s")
```

iii. CONVERSION_NOTES Step 6: "Full-session NWB reads currently materialize the session response matrix in memory once per session" and "Processing plots intentionally add overhead and should stay disabled for full conversion"; Step 7 estimated "`1.22-1.35 s/session` → `~3.5 min` for 152 sessions", which the actual 95 s run beat.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops, all per-trial and all cheap relative to I/O: (a) the pre-pass loop computing per-trial environment mode and the lick-artifact flag; (b) the main per-trial conversion loop; (c) the per-trial loop inside `infer_trial_zones`; (d) the loop over reward frames building `reward_pos_by_trial`. (a) and (c) could be merged into (b) to avoid three passes over the trial list, and the reward-frame loop is a one-line `np.unique`/`searchsorted` operation. Because trials have variable length, the main loop would need padding or `np.add.reduceat`-style segment operations to vectorise fully. The file loop itself is serial and could have been parallelised across processes.

ii.
```python
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):
    env = arrays.environment[start:stop_exclusive]
    ...
for trial_id, (start, stop_exclusive) in enumerate(trial_slices):   # second pass
    if lick_artifact[trial_id]:
        continue
```
```python
for frame_idx in reward_frame_idx:
    ...
    reward_pos_by_trial.setdefault(trial_id, float(positions[frame_idx]))
```

iii. Step 6 claims "Used vectorized trial construction and simple factor-2 rebinning for 31 Hz sessions"; the per-trial arithmetic is indeed vectorised within each trial (the rebinning uses `reshape`+`sum/mean/any`, and all discretisation is mask-based), and the notes treat the remaining per-trial iteration as acceptable since the run is I/O-bound at 95 s total.

## 13-c. What processing does the code repeat multiple times?

i. Repeats within the conversion: `np.clip(arrays.position, 0, 450)` is recomputed for the zone inference, again inside the main trial loop, and twice more in the `--show-processing` branch; the trial list is walked three times (env/lick pre-pass, zone inference, main loop); the example trial's position/speed/lick are re-sliced and re-rebinned for plotting even though the same arrays were just built; and in `--sample` mode `infer_sample_files` opens and reads every NWB file before the real conversion reads two of them again. Unlike the expert reference, however, there is **no** separate survey pass — the full run reads each NWB file exactly once.

ii.
```python
trial_zone, observed_zone_position = infer_trial_zones(
    positions=np.clip(arrays.position, 0.0, TRACK_LENGTH_CM), ...)
...
position = np.clip(arrays.position[start:stop_exclusive], 0.0, TRACK_LENGTH_CM)
...
example_position=np.clip(arrays.position[trial_slices[example_idx][0] : trial_slices[example_idx][1]], 0.0, TRACK_LENGTH_CM)
if factor == 1 else rebin_1d_mean(np.clip(...), factor),
```

iii. Step 6 lists the speed-ups adopted ("Direct `h5py` loading instead of full NWB object graph", "ROI-region-aware cell filtering before downstream work") and notes plots are kept off for the full run; the repeated clips/passes are not called out, presumably because they are negligible next to HDF5 reads.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Small amounts of dead work: the `scanning` time series is read and stored on `SessionArrays` but never used; `roi_ids` is subset and carried but never used after filtering; `trial_rewarded` is passed into `infer_trial_zones` and never read there; `observed_zone_position` is computed for every trial and only written into metadata; the `reward_pos_by_trial` fallback is built for every session although `reward_zone` is active in 85% of trials and the block-majority fill covers the rest. None of these measurably affect the 95 s runtime. Conversely, the expensive processing the paper *does* require (dF/F + OASIS) is skipped entirely, so there is no wasted heavy computation.

ii.
```python
scanning=beh["scanning/data"][()][:t_common].astype(np.float32, copy=False),
...
roi_ids = roi_ids[iscell]
```
```python
def infer_trial_zones(..., trial_rewarded: np.ndarray, ...):   # never referenced in the body
```
```python
"observed_zone_positions_cm": observed_zone_position.tolist(),
```

iii. Not discussed in CONVERSION_NOTES. The extra channels appear to have been loaded during exploration (Step 2 inventories `scanning` and `autoreward` and finds them uninformative — "scanning is all 1s") and were left in the loader after the mapping in Step 5 dropped them.
