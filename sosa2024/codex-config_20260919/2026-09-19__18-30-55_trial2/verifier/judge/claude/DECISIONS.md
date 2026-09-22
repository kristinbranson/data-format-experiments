# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI enumerates every NWB file with a single glob over `/app/data/sub-*/*_behavior+ophys.nwb`, sorts them naturally by `(subject number, session number)` parsed from the filename, and processes them one at a time. It does **not** use `pynwb`; it opens each file directly with `h5py` and reads only the datasets it needs (`general/subject/subject_id`, `general/session_id`, `identifier` (scene), `processing/behavior/BehavioralTimeSeries/*`, `processing/ophys/Fluorescence/plane0/data`, `processing/ophys/Neuropil/plane0/data`, `processing/ophys/ImageSegmentation/PlaneSegmentation/{iscell,planeIdx}`). `--sample` takes the first two files. All 152 files (11 subjects, 12,216 raw paired trials) are loaded in the full run. Each session is read in exactly one pass (there is no separate survey pass). Note: within each file only the `plane0` RoiResponseSeries is read (see 2-a).

ii.
```python
DATA_ROOT = Path("/app/data")

def discover_files(sample: bool) -> list[str]:
    files = sorted(
        glob.glob(str(DATA_ROOT / "sub-*" / "*_behavior+ophys.nwb")),
        key=_session_sort_key,
    )
    if not files:
        raise FileNotFoundError(f"No NWB files found below {DATA_ROOT}")
    return files[:2] if sample else files
```
```python
    with h5py.File(path, "r") as nwb:
        subject = _decode(nwb["general/subject/subject_id"])
        day = int(_decode(nwb["general/session_id"]))
        scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
        session_id = f"{subject}_ses-{day:02d}"

        behavior = nwb["processing/behavior/BehavioralTimeSeries"]
        neural_group = nwb["processing/ophys"]
```
```python
def build_dataset(files: list[str], show_processing: bool) -> dict:
    converted = []
    for i, path in enumerate(files):
        print(f"Converting session {i + 1}/{len(files)}: {path}", flush=True)
        converted.append(convert_session(path, show_processing and i < 2))
```

iii. From CONVERSION_NOTES Step 2/Step 10: "There are 152 NWB 2.8.0 files organized as `data/sub-m<id>/sub-m<id>_ses-<day>_behavior+ophys.nwb`: one file per subject/day session." Direct h5py reading was chosen because it "avoids depending on unpublished session pickles" and lets the converter read only the needed datasets ("avoid loading unused NWB `Deconvolved` and image datasets"), which was part of the stated speed-up strategy. An independent re-scan of all NWBs (not through the converter) reproduced 152 files, 11 subjects, and 12,216 paired trials.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from inside each NWB file (`general/subject/subject_id`), not from the directory name. After all sessions are converted, the unique subject ids are sorted numerically to form `subjects`, and `subject_idx` maps each session to its subject. The file-sort key also parses `sub-m(\d+)` from the filename so sessions are grouped contiguously by subject.

ii.
```python
subject = _decode(nwb["general/subject/subject_id"])
```
```python
def _session_sort_key(path: str) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", os.path.basename(path))
    if match is None:
        raise ValueError(f"Unexpected NWB filename: {path}")
    return int(match.group(1)), int(match.group(2))
```
```python
    subjects = sorted({x["subject"] for x in converted}, key=lambda s: int(s[1:]))
    subject_lookup = {subject: i for i, subject in enumerate(subjects)}
    ...
        "subjects": subjects,
        "subject_idx": np.asarray(
            [subject_lookup[x["subject"]] for x in converted], dtype=int64
        ),
```

iii. CONVERSION_NOTES Step 2 records 11 subjects (`m3, m4, m7, m11–m15, m17–m19`), matching the paper's "counterbalanced across mice (n = 11 mice)". Reading the id from the file metadata rather than the path is the more authoritative source; the numeric sort ("`int(s[1:])`") is used so that `m3` precedes `m11` rather than sorting lexically.

## 1-c. How are the data split into sessions?

i. One session = one NWB file = one experiment day. The day index comes from `general/session_id` inside the file, and the session label is `f"{subject}_ses-{day:02d}"`. Sessions are emitted in `(subject, day)` order. No cross-session neuron alignment (multi-day ROI matching) is attempted; each session's neurons are independent.

ii.
```python
day = int(_decode(nwb["general/session_id"]))
session_id = f"{subject}_ses-{day:02d}"
```
```python
info = {
    "session_id": session_id,
    "source_file": os.path.relpath(path, "/app"),
    "subject": subject,
    "session_day": day,
    "scene": scene,
    ...
}
```

iii. CONVERSION_NOTES Step 2: "one file per subject/day session… Subjects m3, m4, m7, m12–m15, m17–m19 have sessions 1–14; m11 has sessions 3–14", cross-checked against the paper's "total of 14 days" and "imaging started on day 3" for m11. The day number is also required downstream by the scene/switch logic, so it is read from the file rather than inferred.

## 1-d. How are the data split into trials?

i. A trial is the on-track lap from the `trial_start` flag row up to (but excluding) the `teleport` flag row, i.e. NWB rows `[start, teleport)`. Before trial extraction, every dense stream and the neural matrix are truncated to their common minimum length. The code asserts that the number of `trial_start` flags equals the number of `teleport` flags and that every teleport follows its start; it also asserts that all dense behavior series share identical timestamps and that the sampling interval is exactly `1/15.5078125` s. The teleport/ITI period is excluded.

ii.
```python
        common_length = min(
            fluorescence_ds.shape[0],
            *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
        )
        ...
        timestamps = np.asarray(behavior["position/timestamps"][:common_length])
        if not all(
            np.allclose(
                behavior[name]["timestamps"][:common_length], timestamps,
                rtol=0.0, atol=1e-9,
            )
            for name in dense_names
        ):
            raise ValueError(f"Dense behavior timestamps differ in {session_id}")
        if not np.allclose(np.diff(timestamps), 1.0 / FRAME_RATE_HZ, atol=1e-9):
            raise ValueError(f"Nonuniform timestamps in {session_id}")

        starts = np.flatnonzero(dense["trial_start"] > 0)
        teleports = np.flatnonzero(dense["teleport"] > 0)
        if len(starts) != len(teleports) or not np.all(teleports > starts):
            raise ValueError(f"Unpaired or reversed trial bounds in {session_id}")
```
```python
        for i, (start, stop) in enumerate(zip(starts, teleports)):
            fluorescence = np.asarray(
                fluorescence_ds[start:stop, :], dtype=np.float32
            )[:, roi_columns].T
```

iii. CONVERSION_NOTES Step 2/Step 4: "Trial starts and teleports are paired exactly in all sessions. Slicing the dense NWB arrays as `[start_flag_index:teleport_flag_index]` retains the on-track samples (approximately 0 through 450 cm) and excludes the teleport sample/ITI; this is equivalent to the reference object's stored 1-based indices sliced as `[start-1:stop-1]`." The paper describes trials as laps from track entry through the 450 cm track, with the variable-length gray teleport zone after the lap, so the teleport period is deliberately not part of a trial. `off_start = 0.0`, `off_end = None` because lap duration varies.

## 1-e. How are trials filtered based on quality controls?

i. One trial-level quality control is applied: a trial is dropped if more than 30% of its frames have a cumulative lick count > 2 (the paper's lick-sensor-damage criterion). This removed exactly 81 of 12,216 trials, leaving 12,135. OASIS deconvolution is skipped for these trials (they are still used for the dF/F–speed correlation that drives the interneuron filter). There is no minimum-trial-length filter, and no speed-based sample removal. The code raises an error if a session ends up with fewer than two retained trials (no session did).

ii.
```python
LICK_ERROR_FRACTION = 0.30
...
        lick_bad = np.array([
            np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
            for start, stop in zip(starts, teleports)
        ])
```
```python
            dff, events, processing_trace = compute_dff_and_events(
                fluorescence, neuropil, compute_events=not lick_bad[i]
            )
            speed = np.asarray(dense["speed"][start:stop], dtype=np.float32)
            _update_corr_sums(corr_sums, dff, speed)

            if lick_bad[i]:
                continue
```
```python
        if len(kept_events) < 2:
            raise ValueError(f"Fewer than two retained trials in {session_id}")
```

iii. CONVERSION_NOTES Step 4/Step 5: the methods state that trials with erroneous lick detection ("~0.65% of all imaged trials, n = 81 out of 12,376") are set to NaN and removed from licking analysis, detected by ">30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2". The repository snapshot (`glmUtils.get_timeseries_data`) uses 0.35; the AI notes that 0.35 yields 69 trials while 0.30 reproduces the published 81 exactly, and therefore "Published methods take precedence". Because lick is a required per-frame output and NaNs are not allowed by the target format, the AI drops the whole trial rather than emitting NaN or a fabricated lick label. It also explicitly declines to drop speed <2 cm/s samples ("the requested decoder explicitly requires classifying speed `<2 cm/s`, and all outputs must remain on a regular trial-start time axis").

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is recomputed from the raw suite2p traces `processing/ophys/Fluorescence/plane0/data` and `processing/ophys/Neuropil/plane0/data`, with ROI curation from `ImageSegmentation/PlaneSegmentation/{iscell, planeIdx}`. The NWB `Deconvolved` series is deliberately **not** used. Only the `plane0` RoiResponseSeries is read: the AI asserted that for the two-plane subjects (m17, m18) plane-1 response data do not exist in the NWB files, and therefore restricted curation to `iscell[planeIdx == 0]`.

ii.
```python
        neural_group = nwb["processing/ophys"]
        fluorescence_ds = neural_group["Fluorescence/plane0/data"]
        neuropil_ds = neural_group["Neuropil/plane0/data"]
        ...
        segmentation = neural_group["ImageSegmentation/PlaneSegmentation"]
        plane_index = np.asarray(segmentation["planeIdx"][:])
        iscell = np.asarray(segmentation["iscell"][:, 0]) > 0
        plane0_iscell = iscell[plane_index == 0]
        if fluorescence_ds.shape[1] != len(plane0_iscell):
            raise ValueError(
                f"plane0 response/segmentation mismatch in {session_id}: "
                f"{fluorescence_ds.shape[1]} vs {len(plane0_iscell)}"
            )
        roi_columns = np.flatnonzero(plane0_iscell)
```
```python
            "source_limitation": (
                "m17/m18 segmentation contains two planes, but NWB RoiResponseSeries data are "
                "available only for plane0; unavailable plane1 cells are not included."
            ),
```

iii. CONVERSION_NOTES Step 4: "NWB `Deconvolved` has large Suite2P-scale values and correlates only median r≈0.50 with a reference-style recomputation in 20 spot-checked cells… A spot check gave recomputed event range 0–0.81 versus NWB series 0–4,466, confirming they are different processing stages. Do not use NWB `Deconvolved` directly." For the plane restriction: "PlaneSegmentation contains two planes for m17/m18, but all response matrices are named/stored as `plane0` and have columns only for `planeIdx==0`; plane1 response data are absent… It is impossible to recover plane1 activity. Use curated plane0 cells only and disclose this source limitation."

## 2-b. How is the `neural` data processed?

i. Per trial and per curated ROI, a copy of the paper's `preprocessing.dff` is applied: subtract `0.7 × Fneu`, add back `0.7 ×` the trial-mean neuropil, compute a maximin baseline (Gaussian smoothing with sigma 15 samples along time only, then a 300-sample running minimum followed by a 300-sample running maximum — the Methods' 20 s window), form `(F − baseline)/|baseline|`, smooth with a 2-sample Gaussian, then deconvolve with suite2p's OASIS (`dcnv.oasis`, batch 2000, `tau = 0.7`, rate 15.5078125 Hz). Non-finite OASIS output is replaced by 0. Events are stored as float32, neuron × time. Baselines are always restricted to the lap; the reference repo's `keep_teleports` per-mouse/day table is not applied.

ii.
```python
NEUROPIL_COEF = 0.7
BASELINE_SMOOTH_SIGMA = 15.0
BASELINE_WINDOW_SAMPLES = 300
DFF_SMOOTH_SIGMA = 2.0
OASIS_TAU_SECONDS = 0.7

def compute_dff_and_events(fluorescence, neuropil, compute_events=True):
    """Reference-style processing for one trial, arrays neuron x time."""
    corrected = fluorescence - NEUROPIL_COEF * neuropil
    corrected += NEUROPIL_COEF * np.mean(neuropil, axis=1, keepdims=True)

    # The reference TwoPUtils call uses a 2-D sigma [0, 15], smoothing time only.
    baseline_seed = gaussian_filter(
        corrected, sigma=(0.0, BASELINE_SMOOTH_SIGMA), mode="reflect"
    )
    baseline = minimum_filter1d(
        baseline_seed, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect"
    )
    baseline = maximum_filter1d(
        baseline, BASELINE_WINDOW_SAMPLES, axis=1, mode="reflect"
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        dff_unsmoothed = (corrected - baseline) / np.abs(baseline)
    dff = gaussian_filter1d(dff_unsmoothed, DFF_SMOOTH_SIGMA, axis=1)

    events = None
    if compute_events:
        events = dcnv.oasis(
            np.asarray(dff, dtype=np.float32), 2000, OASIS_TAU_SECONDS, FRAME_RATE_HZ,
        )
        events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
        events = np.asarray(events, dtype=np.float32)
```

iii. CONVERSION_NOTES Step 3/Step 10: "dF/F: subtract 0.7× neuropil (per reference code), calculate a maximin baseline separately within each trial using a 20 s/300-sample window, `(F-baseline)/abs(baseline)`, then Gaussian-smooth with 2-sample (~0.129 s) s.d. Deconvolve with OASIS/canonical GCaMP kernel." The Step 10 reference-code comparison table lists the reference `preprocessing.dff` operations next to the converter's and concludes "Same; raw recomputation is used because NWB `Deconvolved` is not the paper's final trial-wise processing". `tau = 0.7` and the 15.5078125 Hz per-plane rate are taken from the repo's suite2p ops and from the behavior timestamps respectively. The notes never mention `teleport_metadata.py` / laser blanking.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) suite2p manual curation: only `iscell[planeIdx == 0, 0] > 0` ROIs are read. (2) Putative interneurons: the Pearson correlation between each cell's dF/F and the animal's running speed is accumulated across all trials of the session (using a numerically stable batch-Welford merge) and cells with `r > 0.5` are removed. 118,493 curated plane-0 cells → 118,169 retained (324 removed, 0.27%). No place-cell / spatial-information selection is applied. The code errors out if the interneuron filter would remove every neuron.

ii.
```python
INTERNEURON_SPEED_R = 0.50
...
        roi_columns = np.flatnonzero(plane0_iscell)
        n_curated = len(roi_columns)
...
            speed = np.asarray(dense["speed"][start:stop], dtype=np.float32)
            _update_corr_sums(corr_sums, dff, speed)
...
        speed_corr = _finalize_corr(corr_sums)
        is_interneuron = np.isfinite(speed_corr) & (speed_corr > INTERNEURON_SPEED_R)
        neuron_keep = ~is_interneuron
        if np.count_nonzero(neuron_keep) == 0:
            raise ValueError(f"Interneuron filter removed every neuron in {session_id}")
        kept_events = [np.asarray(x[neuron_keep], dtype=np.float32) for x in kept_events]
```

iii. CONVERSION_NOTES Step 3: "Manual Suite2P curation removed multi-soma/dendritic ROIs, ROIs without clear transients, overexpression, and continuously fluctuating putative interneurons (`iscell`). The paper then excluded additional putative interneurons whose dF/F correlated >0.5 with running speed (0.42 ± 0.85% of cells across mice/days). Place-cell and remapping-class restrictions are analysis-specific scientific selections and are not a generic quality filter." The realized 0.273% exclusion is compared to the paper's 0.42 ± 0.85%. During development the AI found its first (raw-sum) Pearson accumulator produced |r| > 1 and replaced it with a centered Welford merge unit-tested against `np.corrcoef`.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. No extra alignment is needed: the neural rows and the dense behavior rows are the same synchronized frames, so slicing rows `[trial_start, teleport)` already aligns t = 0 to the trial-start event. The code enforces this by truncating neural and behavior to a common length, by asserting all dense timestamps are identical, and by checking per trial that the neural, input and output matrices have the same number of columns. Metadata records the alignment event and `off_start = 0.0`.

ii.
```python
        common_length = min(
            fluorescence_ds.shape[0],
            *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
        )
```
```python
            fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
            ...
            T = stop - start
            ...
            if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
                raise AssertionError(f"Trial shape mismatch in {session_id} trial {i}")
```
```python
            "temporal_alignment_event": "entry into the 450 cm track (trial_start)",
            "off_start": 0.0,
            "off_end": None,
```

iii. CONVERSION_NOTES Step 4: "rows of every dense NWB behavior series and plane0 neural series are already synchronized at 64.4836 ms. Valid trials are paired `trial_start`→`teleport` rows." Step 10 edge-case audit: "ten NWBs with extra trailing rows are safely truncated to the common synchronized length". `off_end` is None because lap duration varies.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The native imaging-frame resolution is kept: 15.5078125 Hz, i.e. `time_bin_size = 64.4836 ms`. No rebinning, resampling or interpolation is performed. For the two-plane m17/m18 files the stored NWB `rate` attribute is 31.015625 Hz, but the AI uses the behavior timestamps (which step by exactly 1/15.5078125 s in every file) and treats the stored rate as the scanner/volume rate. The same rate is also used as the OASIS kernel rate. A hard check rejects any session whose timestamp spacing differs.

ii.
```python
FRAME_RATE_HZ = 15.5078125
TIME_BIN_MS = 1000.0 / FRAME_RATE_HZ
...
        if not np.allclose(np.diff(timestamps), 1.0 / FRAME_RATE_HZ, atol=1e-9):
            raise ValueError(f"Nonuniform timestamps in {session_id}")
```
```python
            "time_bin_size": TIME_BIN_MS,
            "sampling_rate_hz": FRAME_RATE_HZ,
```
```python
            if not np.allclose(x[0], np.arange(x.shape[1]) / FRAME_RATE_HZ, atol=2e-4):
                raise AssertionError(f"Time axis differs in session {s}")
```

iii. CONVERSION_NOTES Step 2/Step 4: "Dense behavior timestamps have an invariant 0.064483627204 s interval (15.5078125 Hz) in every file. This is the authoritative sampling interval… the attribute is the volume/scan rate before division by `n_planes` and must not be used as the time-bin rate." Step 5: "Keep native 64.483627 ms bins for every session… No temporal resampling is needed." The paper's "~15.5 Hz per plane" is cited as confirmation. Spatial (10 cm) binning used in the paper's own analyses is explicitly rejected because the requested outputs are time-varying.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. From the dense behavior timestamps. The AI reads `behavior['position']['timestamps']` (truncated to the common length) and verifies that every other dense series carries bitwise-comparable timestamps (`atol=1e-9`), so the choice of series is immaterial.

ii.
```python
        timestamps = np.asarray(behavior["position/timestamps"][:common_length])
        if not all(
            np.allclose(
                behavior[name]["timestamps"][:common_length], timestamps,
                rtol=0.0, atol=1e-9,
            )
            for name in dense_names
        ):
            raise ValueError(f"Dense behavior timestamps differ in {session_id}")
```

iii. CONVERSION_NOTES Step 2: "Dense float64 series are `autoreward`, `environment`, `lick`, `position`, `reward_zone`, `scanning`, `speed`, `teleport`, `trial number`, and `trial_start`, each with explicit timestamps." Rather than assume they agree, the converter asserts it, which is also what makes the "one shared time index" argument used elsewhere valid.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtraction of the trial's first timestamp, so every trial's time axis starts at exactly 0 and increments by 64.4836 ms. Stored as float32 row 0 of `input`.

ii.
```python
            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start],
                np.full(T, env_values[0]),
                np.full(T, raw_trial),
                np.full(T, outcomes[i - 1] if i > 0 else 0),
            )).astype(np.float32)
```
```python
            if not np.allclose(x[0], np.arange(x.shape[1]) / FRAME_RATE_HZ, atol=2e-4):
                raise AssertionError(f"Time axis differs in session {s}")
```

iii. CONVERSION_NOTES Step 5 variable map: "`timestamp[s:e] - timestamp[s]`, float32 seconds… Time-varying, begins at exactly 0." The `validate_converted` check exists to guarantee the row really is a regular ramp at the declared bin size (a consistency check between the declared `time_bin_size` and the emitted input).

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. By construction — the timestamps and the neural rows are the same frame indices, sliced with the same `[start, stop)`. Alignment is enforced by (a) the common-length truncation, (b) the all-timestamps-identical assertion, (c) the uniform-spacing assertion, and (d) the per-trial shape assertion that `events.shape[1] == T == input_data.shape[1]`.

ii.
```python
            fluorescence = np.asarray(fluorescence_ds[start:stop, :], dtype=np.float32)[:, roi_columns].T
            ...
            T = stop - start
            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start], ...
            )).astype(np.float32)
            ...
            if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
                raise AssertionError(f"Trial shape mismatch in {session_id} trial {i}")
```

iii. CONVERSION_NOTES Step 4: neural and behavior streams "are already synchronized at 64.4836 ms" — the paper synchronized Unity VR frames to imaging frames with TTLs, so the NWB rows are one-to-one. No interpolation is therefore applied.

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The dense `environment` behavior time series (`processing/behavior/BehavioralTimeSeries/environment/data`), truncated to the common length and sliced per trial.

ii.
```python
        dense_names = [
            "position", "speed", "lick", "environment", "trial number",
            "reward_zone", "trial_start", "teleport",
        ]
        dense = {
            name: np.asarray(behavior[name]["data"][:common_length])
            for name in dense_names
        }
...
            env_values = np.unique(dense["environment"][start:stop])
```

iii. CONVERSION_NOTES Step 2: "environment −1/0/1 (−1 invalid/pre-sync)… Within paired trial spans, environment is always binary 0/1." Step 5 maps it to `input[1]` with 0 = ENV1, 1 = ENV2, referencing the reference repo's `behavior.get_trial_types`, which likewise reads the per-trial unique morph/environment value.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. Essentially none: the code takes the unique value within the trial, raises if it is not a single value in {0, 1}, and then broadcasts that scalar across all timepoints of the trial (per-trial value represented as a constant time series).

ii.
```python
            env_values = np.unique(dense["environment"][start:stop])
            if len(env_values) != 1 or env_values[0] not in (0, 1):
                raise ValueError(
                    f"Environment must be a constant 0/1 in {session_id} trial {i}: "
                    f"{env_values}"
                )
            ...
            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start],
                np.full(T, env_values[0]),
                ...
            )).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "All signals are two-dimensional time series… Per-trial variables are repeated in time so static and dynamic variables coexist consistently." The strict validity check turns the "environment is constant within a trial" assumption into a hard, per-trial verified fact rather than an assumption.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The dense `trial number` behavior time series, sampled at the trial's start row and rounded to an integer. The converter additionally asserts that the resulting per-trial numbers are unique within a session. It does *not* use a loop counter, and it does *not* renumber after bad-lick trials are dropped.

ii.
```python
        raw_trial_numbers = np.rint(dense["trial number"][starts]).astype(int)
        if len(np.unique(raw_trial_numbers)) != len(raw_trial_numbers):
            raise ValueError(f"Repeated raw trial numbers in {session_id}")
...
            raw_trial = int(raw_trial_numbers[i])
```

iii. CONVERSION_NOTES Step 5 variable map: "Read at start row and repeat original zero-based value across frames… Continuous per-trial input; do not renumber after exclusions." Step 10 edge-case audit: "zone changes occur at raw trial 30 even if a prior trial is dropped; all session IDs and raw trial numbers are unique." Preserving the raw number also keeps the switch-at-trial-30 rule (used for the reward zone) in the experiment's own chronology.

## 5-b. What processing is involved in computing `input` *Trial number*?

i. Rounding to the nearest integer and broadcasting the per-trial scalar across the trial's timepoints as row 2 of `input` (float32). No renumbering, no normalization.

ii.
```python
            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start],
                np.full(T, env_values[0]),
                np.full(T, raw_trial),
                np.full(T, outcomes[i - 1] if i > 0 else 0),
            )).astype(np.float32)
```

iii. Same as 5-a: the input is described in the notes as `input_names[2] = "trial number (zero-based)"`, and the verification log confirms the range is [0, 99] with per-session maxima matching each session's raw trial count.

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the per-trial reward outcome array computed for **all raw trials** (before bad-lick removal), which itself comes from the sparse `Reward` event timestamps plus the dense `reward_zone` stream. Reward times are located within `[t_start, t_teleport)` by `np.searchsorted` on the behavior timestamps.

ii.
```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
...
        outcomes = _reward_outcomes(
            timestamps, starts, teleports,
            np.asarray(behavior["Reward/timestamps"][:]),
            dense["reward_zone"],
        )
```

iii. CONVERSION_NOTES Step 4/Step 5: "A trial is rewarded iff it has a sparse reward timestamp within `[start, teleport)` and a zone-entry flag, matching `get_trial_types`; ignore the 3 out-of-trial events." This mirrors the reference repo's `behavior.get_trial_types`, which computes `np.any(reward>0) and np.any(rzone>0)`.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. For raw trial `i > 0`, take `outcomes[i-1]`; for the first raw trial of the session, use 0. The value is broadcast across the trial's timepoints as row 3 of `input`. Because outcomes are computed for all raw trials before curation, a dropped bad-lick trial still counts as the "previous" trial — the chronology is the experiment's, not the retained-trial list's.

ii.
```python
            input_data = np.vstack((
                timestamps[start:stop] - timestamps[start],
                np.full(T, env_values[0]),
                np.full(T, raw_trial),
                np.full(T, outcomes[i - 1] if i > 0 else 0),
            )).astype(np.float32)
```

iii. CONVERSION_NOTES Step 5 Key Decision 10: "Outcomes are calculated for every raw trial before bad-lick removal so previous outcome preserves experimental chronology." Step 10: "the first trial's previous outcome is zero; after a bad-lick trial, 'previous' still means the preceding raw trial rather than preceding retained trial." The instructions define the input as binary (omitted = 0, rewarded = 1).

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. From the dense `position` series together with the trial's reward-zone bounds. The zone bounds are not inferred from the data: they come from the canonical A/B/C coordinates (A 80–130, B 200–250, C 320–370 cm) selected by the zone label, and the label is parsed from the session's scene name in the NWB `identifier`, with the second label taking effect from raw trial 30 onward on switch sessions.

ii.
```python
ZONE_BOUNDS = {
    "A": (80.0, 130.0),
    "B": (200.0, 250.0),
    "C": (320.0, 370.0),
}

def parse_scene_zones(scene: str) -> list[str]:
    """Return chronological A/B/C zone labels encoded in a scene name."""
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    if len(labels) not in (1, 2):
        raise ValueError(f"Could not parse one or two reward zones from {scene!r}")
    return labels

def zone_for_trial(labels: list[str], raw_trial_number: int) -> str:
    if len(labels) == 1 or raw_trial_number < 30:
        return labels[0]
    return labels[1]
```
```python
            position = np.asarray(dense["position"][start:stop], dtype=np.float32)
            ...
            zone_label = zone_for_trial(labels, raw_trial)
            zone_start, zone_end = ZONE_BOUNDS[zone_label]
```

iii. CONVERSION_NOTES Step 1/Step 4: this reproduces the reference repo's `behavior.get_reward_zones`, which sets the zone from the scene ID and switches at `change_trial = 30`, with `rz_dict` X/Y/Z = A/B/C at [80,130]/[200,250]/[320,370]. The paper states "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" and "Each switch occurred after 30 trials". The AI validated the parsed labels against the raw data: "Validated against 10,394 raw reward-zone entries: 0 nearest-zone mismatches."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed point-to-interval distance: `distance = position − clip(position, zone_start, zone_end)`. This is negative before the zone, exactly 0 anywhere inside the 50 cm zone, and positive after it. The continuous value is then discretized (see 7-c).

ii.
```python
            distance = position - np.clip(position, zone_start, zone_end)
            ...
            output_data = np.vstack((
                distance_classes(distance),
                ...
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "'Distance to any location in the reward zone' is the signed point-to-interval distance, zero throughout the 50 cm zone. This directly creates the specified zero class instead of measuring only from zone start." The instructions list a distinct class for exactly 0 cm, which only makes sense under the point-to-interval reading.

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes assigned with explicit inequalities rather than `np.digitize`: 0 `< −50`; 1 `[−50, −10)`; 2 `[−10, 0)`; 3 `== 0`; 4 `(0, 10]`; 5 `(10, 50]`; 6 `> 50`. Stored as int8. `validate_converted` re-checks the range 0–6.

ii.
```python
def distance_classes(distance: np.ndarray) -> np.ndarray:
    """Discretize signed point-to-zone distance exactly as specified."""
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50.0] = 0
    out[(distance >= -50.0) & (distance < -10.0)] = 1
    out[(distance >= -10.0) & (distance < 0.0)] = 2
    out[distance == 0.0] = 3
    out[(distance > 0.0) & (distance <= 10.0)] = 4
    out[(distance > 10.0) & (distance <= 50.0)] = 5
    out[distance > 50.0] = 6
    return out
```

iii. CONVERSION_NOTES Step 5 Key Decision 9: "Use explicit inequalities for every output… Signed distance is class 0 `<−50`; 1 `[-50,−10)`; 2 `[-10,0)`; 3 `==0`; 4 `(0,10]`; 5 `(10,50]`; 6 `>50`. This obeys the task's strict final-bin `>` wording and its positive-distance 'to +10/+50' wording." Step 10 Iteration 2 records that `np.digitize` was deliberately replaced after the AI found it would put exactly-360 cm and exactly-40 cm/s into the strict `>` classes, and that boundary unit tests were added.

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Same frame indices — position is sliced with the identical `[start, stop)` used for the neural matrix, and the per-trial shape assertion guarantees identical T. No shifting or interpolation.

ii.
```python
            position = np.asarray(dense["position"][start:stop], dtype=np.float32)
            ...
            if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
                raise AssertionError(f"Trial shape mismatch in {session_id} trial {i}")
```

iii. Follows from the shared-frame synchronization established in 2-d/3-c; the Step 7 processing plots overlay position, distance and the discretized class on the same time axis as the neural trace to make the alignment visible ("the zone-C zero-distance plateau matches 320–370 cm… No temporal shift or discretization anomaly is visible").

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The dense `position` behavior series (cm along the 450 cm corridor), truncated to the common length and sliced per trial. Since the teleport row is excluded, the values are on-track.

ii.
```python
        dense = {
            name: np.asarray(behavior[name]["data"][:common_length])
            for name in dense_names
        }
...
            position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. CONVERSION_NOTES Step 5: "Positions are on-track because teleport is excluded." Step 2 notes the full-recording range is −500 to 452.47 cm, with −500 pre-sync and −50 the teleport jitter, both outside paired trial spans.

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and the discretization; the raw cm values are used directly with no clipping, smoothing or re-referencing.

ii.
```python
            output_data = np.vstack((
                distance_classes(distance),
                position_classes(position),
                ...
            )).astype(np.int8)
```

iii. CONVERSION_NOTES Step 5 variable map for `output[1]`: "Explicit classes `<90`, `[90,180)`, `[180,270)`, `[270,360]`, `>360`… Positions are on-track because teleport is excluded." No transform is needed because the track is already in cm from 0.

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes of 90 cm each over the 450 cm track, assigned with explicit inequalities: 0 `< 90`; 1 `[90, 180)`; 2 `[180, 270)`; 3 `[270, 360]`; 4 `> 360`. The first and last classes are open so that the handful of samples slightly outside [0, 450] fall into the end classes. int8; range re-checked in `validate_converted`.

ii.
```python
def position_classes(position: np.ndarray) -> np.ndarray:
    """Use explicit inequalities, including 360 cm in class 3 as specified."""
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90.0] = 0
    out[(position >= 90.0) & (position < 180.0)] = 1
    out[(position >= 180.0) & (position < 270.0)] = 2
    out[(position >= 270.0) & (position <= 360.0)] = 3
    out[position > 360.0] = 4
    return out
```
```python
            ["< 90 cm", "90 to < 180 cm", "180 to < 270 cm", "270 to 360 cm", "> 360 cm"],
```

iii. CONVERSION_NOTES Step 5 Key Decision 9 and Step 10 Iteration 2: the instructions write class 3 as "270 to 360 cm" and class 4 as "> 360 cm", so exactly 360 is placed in class 3, which `np.digitize` would not do. "Independent scanning found zero raw samples exactly at any position/speed edge, so distributions did not change." The realized distribution is [0.212, 0.177, 0.231, 0.226, 0.154].

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Same frame indices as the neural data (`dense["position"][start:stop]`), identical T enforced per trial. No separate alignment step.

ii.
```python
            position = np.asarray(dense["position"][start:stop], dtype=np.float32)
```

iii. As in 2-d/7-d: dense behavior and neural rows are the same synchronized imaging frames, asserted by the shared-timestamp and common-length checks; the `--show-processing` plot panel 5 shows position starting near 0 cm at t = 0 and rising to ~450 cm.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The dense `lick` behavior series, which holds a cumulative lick count per imaging frame (values 0–8 in this dataset).

ii.
```python
            lick = np.asarray(dense["lick"][start:stop])
```

iii. CONVERSION_NOTES Step 2: "lick cumulative count 0–8/frame". Step 5 maps "dense cumulative `lick`" to `output[3]` via `glmUtils.get_timeseries_data` and the paper's licking methods.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarization `(lick > 0)` → int8, at every frame. The paper's sensor-error criterion is applied upstream as a whole-trial exclusion (see 1-e) rather than by NaN-ing lick within retained trials, so no retained frame has an invalid lick label.

ii.
```python
            output_data = np.vstack((
                distance_classes(distance),
                position_classes(position),
                speed_classes(speed),
                (lick > 0).astype(np.int8),
                np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8),
                np.full(T, outcomes[i], dtype=np.int8),
            )).astype(np.int8)
```

iii. CONVERSION_NOTES Step 3/Step 5: the reference `glmUtils` does `licks[licks > 1] = 1` after NaN-ing bad trials; the instructions require a binary 0/1 output. Step 5 Key Decision 7: "The target requires a valid lick label at every frame and forbids NaN; dropping entire corrupt trials is safer than fabricating lick outputs." Realized distribution [0.777, 0.223].

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same frame indices (`dense["lick"][start:stop]`), same T as the neural matrix; no alignment work needed because the lick counts are already resampled onto imaging frames in the NWB.

ii.
```python
            lick = np.asarray(dense["lick"][start:stop])
```

iii. CONVERSION_NOTES Step 3: "Lick methods explicitly refer to '0.0645 s imaging frame samples'", i.e. the stored lick stream is already at the imaging frame rate and shares the verified dense timestamps.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. From the session's scene name (NWB `identifier`, e.g. `Env1_LocationB_to_A`) plus the raw trial number: the chronological A/B/C labels are parsed from the scene, and on switch sessions the second label applies from raw trial 30 onward. The dense `reward_zone` stream is used only for validation (and for the reward-outcome rule), not to derive the label.

ii.
```python
        scene = _decode(nwb["identifier"]).rstrip("/").split("/")[-1]
        ...
        labels = parse_scene_zones(scene)
        ...
            zone_label = zone_for_trial(labels, raw_trial)
```
```python
def parse_scene_zones(scene: str) -> list[str]:
    labels = re.findall(r"(?:Location)?([ABC])(?=_to|$)", scene)
    if len(labels) not in (1, 2):
        raise ValueError(f"Could not parse one or two reward zones from {scene!r}")
    return labels
```

iii. CONVERSION_NOTES Step 1 lists `behavior.get_reward_zones` ("Map scene and switch trial to per-trial zone A/B/C coordinates. Canonical zones are A/X=[80,130], B/Y=[200,250], C/Z=[320,370] cm") and `define_trial_subsets` ("default switch trial 30"). Step 4: "Parse labels from scene and assign the second label from trial index 30 onward. Validated against 10,394 raw reward-zone entries: 0 nearest-zone mismatches."

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. The label is mapped to a class index via `{"A": 0, "B": 1, "C": 2}` and broadcast across the trial's timepoints as row 4 of `output` (int8).

ii.
```python
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
...
                np.full(T, ZONE_TO_CLASS[zone_label], dtype=np.int8),
```
```python
            ["A", "B", "C"],
```

iii. The instructions specify per-trial reward zone location 0 = A, 1 = B, 2 = C; Key Decision 1 requires per-trial variables to be repeated across time. The realized class distribution is [0.332, 0.336, 0.333], which the AI notes is the expected near-balance given the counterbalanced design.

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward` event time series (`Reward/timestamps`) combined with the dense `reward_zone` stream, evaluated over `[t_trial_start, t_teleport)`.

ii.
```python
        outcomes = _reward_outcomes(
            timestamps, starts, teleports,
            np.asarray(behavior["Reward/timestamps"][:]),
            dense["reward_zone"],
        )
```

iii. CONVERSION_NOTES Step 2: "`Reward` is a sparse float64 event series with delivery timestamps and 0.004 mL values… 10,345 sparse reward events, of which 3 lie outside paired on-track spans." Step 4 states the rule is chosen to match the reference `behavior.get_trial_types`.

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. For each raw trial, `searchsorted` on the sorted reward timestamps gives the count of reward events falling in the half-open window `[timestamps[start], timestamps[stop])`; the trial is rewarded (1) iff that count is > 0 **and** the dense `reward_zone` flag is positive somewhere in the trial. The per-trial value is broadcast across the trial as row 5 of `output` (int8). The three reward events outside any paired trial window are ignored. Outcomes are computed once for all raw trials and reused for the previous-outcome input.

ii.
```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
...
                np.full(T, outcomes[i], dtype=np.int8),
```

iii. CONVERSION_NOTES Step 4: "A trial is rewarded iff it has a sparse reward timestamp within `[start, teleport)` and a zone-entry flag, matching `get_trial_types`; ignore the 3 out-of-trial events. Result is 84.66% rewarded", consistent with the paper's "reward was randomly omitted on approximately 15% of trials". Step 12 records an explicit investigation (raw-value check on three trials, an alignment overlay plot, class-balance check) after reward outcome decoded at only 1.16× chance, concluding no conversion error.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. A mix of silent repair, documented exclusion, and hard failure:
- **Neural/behavior length mismatch** (10 files with one extra trailing neural sample): all streams are truncated to their common minimum before anything else.
- **Reward events outside any trial** (3 events): ignored, because the trial window is half-open `[start, teleport)`.
- **Lick-sensor damage** (81 trials): whole trial dropped (see 1-e).
- **Non-finite dF/F or OASIS output** (zero/negative baselines): replaced by 0 with `np.nan_to_num`, then the retained data are re-checked for finiteness and non-negativity.
- **Structural anomalies** are *not* tolerated: unpaired/reversed trial flags, disagreeing dense timestamps, non-uniform sampling, a non-constant or non-binary environment within a trial, repeated raw trial numbers, a segmentation/response column mismatch, a session left with < 2 trials, or an interneuron filter that removes every cell all raise and abort the conversion.
- There is **no** minimum-trial-length filter (the shortest trial in the data is 96 frames).

ii.
```python
        common_length = min(
            fluorescence_ds.shape[0],
            *(behavior[name]["data"].shape[0] for name in behavior if name != "Reward"),
        )
```
```python
        events = np.nan_to_num(events, nan=0.0, posinf=0.0, neginf=0.0)
```
```python
        if len(kept_events) < 2:
            raise ValueError(f"Fewer than two retained trials in {session_id}")
        if any(not np.all(np.isfinite(x)) for x in kept_events + kept_inputs):
            raise ValueError(f"Nonfinite converted data in {session_id}")
        if any(np.min(x) < -1e-6 for x in kept_events):
            raise ValueError(f"Negative OASIS activity in {session_id}")
```
```python
        if len(starts) != len(teleports) or not np.all(teleports > starts):
            raise ValueError(f"Unpaired or reversed trial bounds in {session_id}")
```

iii. CONVERSION_NOTES Step 10 edge-case audit: "verified paired and ordered start/teleport flags; ten NWBs with extra trailing rows are safely truncated to the common synchronized length; three sparse reward events outside paired trial windows are ignored; the first trial's previous outcome is zero… all retained trials have finite equal-length neural/input/output arrays; all sessions exceed the two-trial requirement." The zero-substitution for non-finite events is justified by analogy to the paper's Fig. 3 decoder, which "replaces remaining neural NaNs by zero"; metadata records `nonfinite_events_replaced_with_zero: True`. The philosophy is stated as: repair only what is documented and understood, and fail loudly on anything structural.

## 13-a. What are the most time-consuming steps of the code?

i. From the printed per-session timings (1.0–2.9 s/session; 241.82 s total for 152 sessions plus 8.85 s to pickle), the cost is dominated by, in order: (1) reading the per-trial `Fluorescence`/`Neuropil` row blocks out of HDF5 — all ROI columns are read and then subset, so roughly twice the needed bytes move; (2) the dF/F filter chain (a 2-D Gaussian plus 300-sample minimum and maximum filters over neurons × time, per trial); (3) OASIS deconvolution per trial; (4) writing the 7.5 GiB pickle. The AI instrumented this with `time.perf_counter()` per session and in aggregate.

ii.
```python
def convert_session(path: str, show_processing: bool) -> dict:
    started = time.perf_counter()
    ...
        elapsed = time.perf_counter() - started
    ...
    print(
        f"[{session_id}] {len(starts)} raw -> {len(kept_events)} trials; "
        f"{n_curated} curated -> {np.count_nonzero(neuron_keep)} neurons; "
        f"speed-r [{np.nanmin(speed_corr):.3f}, {np.nanmax(speed_corr):.3f}]; "
        f"{elapsed:.2f} s",
        flush=True,
    )
```
```python
    print(
        f"Wrote {output} ({output.stat().st_size / 2**30:.3f} GiB) "
        f"in {time.perf_counter() - dump_started:.2f} s",
        flush=True,
    )
```

iii. CONVERSION_NOTES Step 6: "NWB datasets are contiguous rather than chunked. Repeated fancy-column reads can trigger expensive HDF5 selection overhead, while loading full multi-thousand-ROI sessions creates high peak memory. OASIS must operate trial-by-trial to match the reference." Step 7 estimated ~258 s for the full run from the 2-session sample; the actual run took 241.82 s, well under the 15-minute budget, so no further optimization was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Candidates that remain un-vectorized:
- `_reward_outcomes` loops over trials doing two scalar `searchsorted` calls and a slice reduction per trial; both could be done in one vectorized `searchsorted` over all trial boundaries plus `np.add.reduceat` / `np.maximum.reduceat` on `reward_zone`.
- The `lick_bad` list comprehension does a per-trial `np.mean` over a slice; also expressible as a single `np.add.reduceat`.
- The main per-trial loop does the HDF5 read, dF/F, OASIS, and all input/output construction one trial at a time. The purely behavioral parts (distance, position/speed/lick classes, the per-trial constants) could be computed once on the whole-session arrays and then split, as the reference does for most streams.
- `distance_classes` / `position_classes` / `speed_classes` each build 4–7 boolean masks sequentially instead of a single `np.digitize`/`np.searchsorted` pass.
- `_update_corr_sums` is already vectorized over neurons (that was a deliberate rewrite), and OASIS/dF/F genuinely must stay per trial because the reference baselines and deconvolves within a trial.

ii.
```python
def _reward_outcomes(timestamps, starts, teleports, reward_times, reward_zone):
    outcomes = np.zeros(len(starts), dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, teleports)):
        lo = np.searchsorted(reward_times, timestamps[start], side="left")
        hi = np.searchsorted(reward_times, timestamps[stop], side="left")
        outcomes[i] = int(hi > lo and np.any(reward_zone[start:stop] > 0))
    return outcomes
```
```python
        lick_bad = np.array([
            np.mean(dense["lick"][start:stop] > 2) > LICK_ERROR_FRACTION
            for start, stop in zip(starts, teleports)
        ])
```

iii. The AI does not discuss these specific loops; Step 6 only records that "OASIS must operate trial-by-trial to match the reference" and that correlation accumulation was vectorized ("accumulate interneuron correlations from sufficient statistics"). The remaining loops run over ~80 trials per session and are negligible next to the I/O and filtering, which is consistent with the measured 1–3 s/session.

## 13-c. What processing does the code repeat multiple times?

i. Repetition is modest. The main one is HDF5 access: each session's `Fluorescence` and `Neuropil` datasets are opened once but read in ~80 separate row-block requests (one per trial), and each read pulls **all** ROI columns before subsetting to the ~45% that pass `iscell`, so roughly 2× the needed fluorescence bytes are transferred. Secondary repetitions: `validate_converted` re-walks every trial after conversion and recomputes the time-axis comparison that the per-trial assertion already covered; per-trial shape assertions duplicate checks that `validate_converted` repeats globally; `speed` is sliced for the correlation accumulator and used again for the speed output. Notably, the code makes **one** pass over the dataset — there is no separate survey/statistics pass.

ii.
```python
            # Reading the short contiguous row interval first is substantially
            # faster for contiguous NWB datasets than HDF5 fancy column reads.
            fluorescence = np.asarray(
                fluorescence_ds[start:stop, :], dtype=np.float32
            )[:, roi_columns].T
```
```python
            if events.shape[1] != T or input_data.shape != (4, T) or output_data.shape != (6, T):
                raise AssertionError(f"Trial shape mismatch in {session_id} trial {i}")
...
def validate_converted(data: dict) -> None:
    ...
            if not (n.shape[1] == x.shape[1] == y.shape[1]):
                raise AssertionError(f"Time dimensions differ in session {s}")
            if not np.allclose(x[0], np.arange(x.shape[1]) / FRAME_RATE_HZ, atol=2e-4):
                raise AssertionError(f"Time axis differs in session {s}")
```

iii. CONVERSION_NOTES Step 6 explains the read-all-columns choice as a deliberate trade: "Read each short contiguous trial row block once and subset curated columns in memory" was listed as a *speed-up*, because HDF5 fancy column selection on contiguous datasets is slower than reading the block and slicing in NumPy. The duplicated validation is presented as intentional defence in depth ("runs internal shape/range/time-axis checks before pickling").

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. A small amount:
- dF/F is computed for the 81 bad-lick trials even though those trials are dropped. This is not strictly wasted — the dF/F is fed to the speed-correlation accumulator — but the OASIS step for them is correctly skipped, and one could equally have excluded them from the correlation entirely.
- dF/F is computed for every curated cell including the 324 later removed as putative interneurons; unavoidable, since the filter is defined on dF/F.
- `compute_dff_and_events` always assembles the `trace` dict (`corrected`, `baseline`, `dff_unsmoothed`, `dff`) even when `--show-processing` is off; these are references, not copies, but they keep the intermediate arrays alive for the whole trial iteration.
- Diagnostic-only statistics are computed per session for metadata: `dff_speed_correlation_range` and `dff_speed_correlation_percentiles`.
- `validate_converted` recomputes a full `np.arange(T)/rate` comparison for all 12,135 trials after conversion.
- Roughly 55% of the fluorescence/neuropil bytes read from disk are discarded immediately by the `roi_columns` subset.
- The `--show-processing` path retains copies of one example neuron's full trace set; only relevant in that mode.

ii.
```python
            dff, events, processing_trace = compute_dff_and_events(
                fluorescence, neuropil, compute_events=not lick_bad[i]
            )
            speed = np.asarray(dense["speed"][start:stop], dtype=np.float32)
            _update_corr_sums(corr_sums, dff, speed)

            if lick_bad[i]:
                continue
```
```python
    trace = {
        "corrected": corrected,
        "baseline": baseline,
        "dff_unsmoothed": dff_unsmoothed,
        "dff": dff,
    }
    return dff, events, trace
```
```python
            "dff_speed_correlation_percentiles": np.nanpercentile(
                speed_corr, [1, 50, 99]
            ).tolist(),
```

iii. The AI's own framing (Step 6) is that it removed the main waste it identified: "avoid loading unused NWB `Deconvolved` and image datasets; process a single session at a time; … skip OASIS on trials already rejected for lick corruption; keep arrays float32". It does not flag the remaining items above. Given the total runtime of ~4 minutes for 152 sessions, none of them is material.
