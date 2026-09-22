# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI walks `/app/data` two levels deep: every immediate sub-directory is treated as a subject and every dated sub-directory inside it as a session (`discover_sessions`). A session is only accepted if all seven files it needs exist (`suite2p/plane0/{F,Fneu,iscell,ops}.npy` and `move_deve/{motion_energy_glob,tstamps,interframe_int}.npy`); otherwise the session is silently skipped, and if nothing at all is found a `RuntimeError` is raised. Per session it loads:
- `ops.npy` (only the scalar acquisition/processing parameters it needs: `fs`, `nframes`, `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`, `nchannels`, `nplanes`, and a count of `badframes`), then deletes the ops dict;
- `F.npy` and `Fneu.npy` with `mmap_mode="r"`, read in chunks of 64 neurons;
- `motion_energy_glob.npy`, `tstamps.npy`, `interframe_int.npy`.

`spks.npy`, `stat.npy`, the FOV images in `ops`, and the manual-tracking ground-truth CSVs are deliberately not loaded. Sessions are iterated in sorted subject/date order. The result is 41 sessions from 6 subjects, 20,445 session-neuron instances, 1,090 trials.

ii.
```python
def discover_sessions() -> list[tuple[str, Path]]:
    """Return all dated session directories in subject/date order."""
    found: list[tuple[str, Path]] = []
    for subject_dir in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
            plane = session_dir / "suite2p" / "plane0"
            behavior = session_dir / "move_deve"
            required = (
                plane / "F.npy", plane / "Fneu.npy", plane / "iscell.npy", plane / "ops.npy",
                behavior / "motion_energy_glob.npy", behavior / "tstamps.npy",
                behavior / "interframe_int.npy",
            )
            if all(path.exists() for path in required):
                found.append((subject_dir.name, session_dir))
    if not found:
        raise RuntimeError(f"No sessions found under {DATA_DIR}")
    return found
```
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
...
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
```

iii. From CONVERSION_NOTES Step 2/Step 5: the data README defines the `subject/session/{suite2p,move_deve}` hierarchy, so directory walking is the natural enumeration. `ops` is loaded because "actual Suite2p `ops['fs']` is authoritative" and because the paper says analyses used "default Suite2p parameters" — the AI wanted the real `neucoeff=0.7`, `win_baseline=60`, `sig_baseline=10` rather than the reference GUI helper's accidental `neucoeff=0.0` default. `spks.npy` is explicitly rejected: "Paper decoder explicitly used baseline-corrected fluorescence, so spks is not mixed with it." `stat`/FOV images are rejected as "not decoder variables". Memory-mapping and chunking are justified as bounding peak memory for the 746×54,000 arrays.

## 1-b. How are the data split into subjects?

i. One subject per top-level directory under `/app/data`. Subject IDs are the directory names, collected from the selected sessions and sorted, giving `['jm031','jm032','jm038','jm039','jm040','jm046']`. `subject_idx` is the index of the owning subject for each session, as an int64 array.

ii.
```python
subjects = sorted({subject for subject, _ in selected})
subject_lookup = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[subject])
...
"subjects": subjects,
"subject_idx": np.asarray(subject_idx, dtype=np.int64),
```

iii. The data README states "For each subject there is a folder corresponding to the subject id" and that the folders are named in alphabetically increasing order matching mice A–F in the paper. The AI cross-checked this against the paper's "6 mice imaged daily for a minimum of 6 consecutive days" and recorded 6 subjects with 7/7/7/7/6/7 sessions.

## 1-c. How are the data split into sessions?

i. One session per dated sub-directory of a subject (e.g. `jm031/2023-10-18_a`), sorted by name so sessions are in chronological order. 41 sessions total. No session is merged, split, or excluded; the `_a` suffix is stripped only for the `date` field of `session_info`.

ii.
```python
for session_dir in sorted(p for p in subject_dir.iterdir() if p.is_dir()):
```
```python
info = {
    "session_id": f"{subject}/{session_dir.name}",
    "subject": subject,
    "date": session_dir.name.removesuffix("_a"),
    ...
}
```

iii. Per the data README, "Each subject folder contains a number of session folders, each corresponding to one recording day" and "the '_a' in the end of the folder name can be ignored". CONVERSION_NOTES Step 4 records the decision explicitly: "each dated recording is one decoder session". The AI also flagged and resolved a paper/data discrepancy here: methods say "each session lasted 20 minutes" but 27 of 41 recordings contain 54,000 frames (30 min). It chose to keep the full recordings — "Preserve every supplied complete minute … Treat paper statement as a summary/protocol discrepancy, not a reason to truncate valid data."

## 1-d. How are the data split into trials?

i. There is no native trial structure (continuous spontaneous recording), so trials are defined as consecutive, non-overlapping 60-second blocks of each session, per the Decoder Task instruction. At 30 Hz with 10-frame averaging this is 1,800 raw frames = 180 time bins per trial. The number of frames per trial is derived from each session's own `ops['fs']`, and the code *requires* the session to divide exactly into whole trials (and into whole 10-frame bins), raising a `ValueError` otherwise — it never discards a remainder. This yields 20 trials for the 36,000-frame sessions and 30 for the 54,000-frame sessions, 1,090 trials total. Trials are materialised as contiguous copies, and at least 2 trials per session is asserted.

ii.
```python
frames_per_trial = int(TRIAL_SECONDS * ops["fs"])          # 60 * 30 = 1800
bins_per_trial = frames_per_trial // RAW_FRAMES_PER_BIN     # 180
...
if n_frames % frames_per_trial:
    raise ValueError(f"Session does not divide into complete 60-s trials: {session_dir}")
```
```python
def split_trials(array: np.ndarray, bins_per_trial: int) -> list[np.ndarray]:
    if array.ndim != 2 or array.shape[1] % bins_per_trial:
        raise ValueError(f"Cannot split array of shape {array.shape}")
    return [array[:, start : start + bins_per_trial].copy()
            for start in range(0, array.shape[1], bins_per_trial)]
```

iii. CONVERSION_NOTES Step 5 Key Decision 1: "Partition the full stream into consecutive, nonoverlapping 60-s blocks. At 30 Hz and 10-frame averaging this is 180 timepoints/trial; all source recordings divide evenly, yielding 1,090 trials with no truncation." The AI verified divisibility empirically before relying on it (Step 10 check 12: "Raw sessions are exactly divisible by 1,800 frames/trial and 10 frames/bin") and noted the requested 60-s trials override the paper's 2-minute cross-validation blocks.

## 1-e. How are trials filtered based on quality controls?

i. No trials are filtered out. Every complete 60-second block of every session is retained (1,090/1,090). The only trial-level gates are structural assertions that abort the whole conversion rather than drop data: equal trial counts across neural/input/output, ≥2 trials per session, exact per-trial shapes `(n_neurons,180)`/`(1,180)`/`(1,180)`, finiteness, and correct dtypes.

ii.
```python
if not (len(neural_trials) == len(input_trials) == len(output_trials) >= 2):
    raise AssertionError(f"Trial count mismatch in {session_dir}")
for neural, decoder_input, output in zip(neural_trials, input_trials, output_trials):
    if not (neural.shape == (n_neurons, bins_per_trial)
            and decoder_input.shape == (1, bins_per_trial)
            and output.shape == (1, bins_per_trial)):
        raise AssertionError(f"Trial shape mismatch in {session_dir}")
```

iii. CONVERSION_NOTES Step 3 ("Trial curation rules"): "No behavioral trials or rejected-trial rule is described. Recordings are continuous spontaneous behavior… Paper cross-validation used consecutive 2-minute blocks but did not discard behavioral blocks." Step 5 Key Decision 6: "Keep all trials; there are no trial rejection rules, all arrays are finite, missing behavior is recoverable, and the lone Suite2p bad frame is not prescribed for downstream deletion."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `suite2p/plane0/F.npy` (raw fluorescence of the Track2p-tracked ROIs) and `suite2p/plane0/Fneu.npy` (neuropil), plus the scalar processing parameters read out of `suite2p/plane0/ops.npy` (`fs`, `nframes`, `neucoeff`, `baseline`, `win_baseline`, `sig_baseline`, `prctile_baseline`). `spks.npy` (deconvolved) is explicitly *not* used. `F`/`Fneu` shapes are cross-checked against each other and against `ops['nframes']`.

ii.
```python
fluorescence = np.load(plane / "F.npy", mmap_mode="r")
neuropil = np.load(plane / "Fneu.npy", mmap_mode="r")
if fluorescence.shape != neuropil.shape:
    raise ValueError(f"F/Fneu shape mismatch in {session_dir}")
n_neurons, n_frames = fluorescence.shape
if n_frames != ops["nframes"]:
    raise ValueError(f"ops nframes={ops['nframes']} but F has {n_frames} in {session_dir}")
```
```python
result = {"fs": float(ops["fs"]), "nframes": int(ops["nframes"]),
          "neucoeff": float(ops.get("neucoeff", 0.7)),
          "baseline": str(ops.get("baseline", "maximin")),
          "win_baseline": float(ops.get("win_baseline", 60.0)),
          "sig_baseline": float(ops.get("sig_baseline", 10.0)), ...}
```

iii. CONVERSION_NOTES Step 4: the methods say "We used baseline corrected fluorescence traces as our dF/F (using the default Suite2p parameters) for all subsequent analyses", so the AI selected `F`/`Fneu` and the Suite2p baseline pipeline over `spks`. Step 5 variable-mapping table: "`spks.npy` … not mapped. Available deconvolved activity but not selected — Paper decoder explicitly used baseline-corrected fluorescence."

## 2-b. How is the `neural` data processed?

i. Three steps, reproducing `DataManagement.F_processing` from the reference repo `/app/code/track2p/gui/data_management.py`:
1. Neuropil subtraction `Fc = F − neucoeff·Fneu` with `neucoeff` taken from each session's ops (= 0.7 everywhere), in float32.
2. Maximin baseline: Gaussian smooth along time only (`sigma=(0, sig_baseline=10)`), then `minimum_filter1d` then `maximum_filter1d` with window `int(win_baseline·fs) = 1800` frames, and subtract: `processed = Fc − Flow`. The `constant` / `constant_prctile` / else branches of the reference function are reproduced too, but are dead code for this dataset (ops always says `maximin`).
3. Non-overlapping mean of 10 consecutive frames → 3 Hz, stored float32.

No division by baseline (no true ΔF/F), no z-scoring, no smoothing beyond the above. Neurons are processed in chunks of 64 to bound memory; the filters act along the time axis only so chunking is exact. A finiteness check runs over the result.

ii.
```python
corrected = raw_f - neucoeff * raw_fneu

if baseline_mode == "maximin":
    flow = gaussian_filter(corrected, sigma=(0.0, sigma))
    flow = minimum_filter1d(flow, size=baseline_window, axis=1)
    flow = maximum_filter1d(flow, size=baseline_window, axis=1)
elif baseline_mode == "constant":
    smoothed = gaussian_filter(corrected, sigma=(0.0, sigma))
    flow = np.full_like(corrected, np.amin(smoothed))
elif baseline_mode == "constant_prctile":
    flow = np.percentile(corrected, ops["prctile_baseline"], axis=1, keepdims=True).astype(np.float32)
else:
    flow = np.zeros((corrected.shape[0], 1), dtype=np.float32)

processed = corrected - flow
binned[start:stop] = processed.reshape(stop - start, n_bins, RAW_FRAMES_PER_BIN).mean(axis=2)
```

iii. CONVERSION_NOTES Step 4 and Step 5 Key Decision 2: "Compute `Fc=F-0.7*Fneu`, then reference maximin baseline and `Fc-Flow`; use ops values. This matches paper and Suite2p settings more closely than the GUI helper's accidental zero-neuropil default. Do not divide by Flow because neither reference implementation nor Suite2p preprocessing does so." Step 10 records the terminology caveat: "Reference terminology calls baseline-subtracted activity dF/F although code does not divide by baseline. Converter follows executable reference logic (`Fc-Flow`)." Step 10 check 2 re-derives the pipeline independently from the raw files for selected neurons and requires `np.allclose`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron is filtered out by the conversion script; all rows of `F.npy` are kept (20,445 session-neuron instances, 2,998 unique tracked cells). The AI's stated reason is that the distributed arrays are *already* curated: it verified that `iscell[:,0] == 1` for every row of every session and that every classifier probability exceeds 0.5 (observed range 0.50025–0.99841), i.e. the paper's `>0.5` threshold and Track2p's complete-longitudinal-track requirement have already been applied upstream. `iscell.npy` is therefore only required to *exist* (as a session-validity gate in `discover_sessions`); its contents are never read by the script. The provenance is recorded in metadata. `ops['badframes']` (one true frame in 2,214,000 across the whole dataset) is counted and reported but not excluded.

ii.
```python
required = (plane / "F.npy", plane / "Fneu.npy", plane / "iscell.npy", plane / "ops.npy", ...)
```
```python
"badframes": int(np.count_nonzero(ops.get("badframes", []))),
...
"suite2p_badframes": ops["badframes"],
```
```python
"neuron_curation": (
    "distributed Track2p complete tracks; source iscell probability >0.5"
),
```

iii. CONVERSION_NOTES Step 2: "The supplied `iscell` arrays are already matched/filtered: first column is one for every row, and all probabilities exceed 0.5 … Reapplying `>0.5` changes nothing." Step 5 Key Decision 6: "Keep every provided neuron because exports are already `iscell>0.5` and complete Track2p tracks." Step 10 check 6 compares this to the reference `load_stat_ds_plane` (`iscell[:,1] > 0.5`) and `DataManagement.import_files` (rows present on all days) and concludes "Converter correctly does not filter again." For the bad frame, Step 4: "Retain it: F traces exist, paper does not prescribe deletion, and dropping one frame would break synchronized complete-minute blocks. Its 10-frame average limits influence."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no stimulus or behavioural event to align to — the recording is continuous. Trials are contiguous 60-second blocks measured from the first imaging frame of the session, so each trial is aligned to the start of its own 60-s block. The AI encodes this as `temporal_alignment_event = "start of each non-overlapping 60-second block"` with `off_start = 0.0` and `off_end = 60.0` seconds, and records `input_time_reference` separately to make clear that the input time value itself remains absolute from session start. Neural, input and output are all sliced from the same bin grid with the same `split_trials` call, so alignment between streams is exact by construction.

ii.
```python
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
output_trials = split_trials(output_binned, bins_per_trial)
```
```python
"temporal_alignment_event": "start of each non-overlapping 60-second block",
"off_start": 0.0,
"off_end": 60.0,
"input_time_reference": "absolute elapsed time from session start at bin centers",
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "alignment event is each 60-s block start, `off_start=0.0`, `off_end=60.0`. Time input remains absolute from the recording start." Step 10 check 12 verifies there is no off-by-one at block boundaries: "Concatenating every converted trial exactly reconstructs independently processed session streams, proving neither duplicated nor dropped boundary bins."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Source data are 30 Hz (verified from `ops['fs']` in every session). The conversion rebins **once**: a non-overlapping mean of 10 consecutive frames, giving 3 Hz, i.e. a time bin of 1000·10/30 = 333.333 ms and exactly 180 bins per 60-second trial. The same 10-frame averaging is applied to the neural traces and to the motion-energy trace, and it is applied *before* motion energy is discretized. No other resampling, interpolation onto a new grid, or smoothing across bins occurs (the Gaussian filter inside the baseline estimator only shapes the subtracted baseline, not the output signal). `metadata['time_bin_size']` is 333.333 ms.

ii.
```python
RAW_FRAMES_PER_BIN = 10
...
n_bins = n_frames // RAW_FRAMES_PER_BIN
binned[start:stop] = processed.reshape(stop - start, n_bins, RAW_FRAMES_PER_BIN).mean(axis=2)
...
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
```
```python
"time_bin_size": 1000.0 * RAW_FRAMES_PER_BIN / 30.0,
"temporal_averaging_frames": RAW_FRAMES_PER_BIN,
"source_sampling_rate_hz": 30.0,
```

iii. Directly from the methods: "For all decoding analysis we slightly denoised the dF/F as well as the behaviour traces by averaging in bins of 10 consecutive timestamps." CONVERSION_NOTES Step 5 Key Decision 3: "Repair sparse behavior gaps at the 30-Hz frame level, baseline-process neural data at full rate, then average both streams over identical groups of 10 frames. Only after averaging is motion discretized … avoids averaging arbitrary category numbers."

## 3-a. What variables in the raw data is `input` *Time from start of experiment* derived from?

i. No raw data array. Time is computed from the imaging-frame index and the sampling rate `ops['fs']` (30 Hz in every session). The behavioural `tstamps.npy` clock is deliberately *not* used for this, only for locating dropped camera frames. The single input is named `session_elapsed_time_s`.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
```
```python
"input_names": ["session_elapsed_time_s"],
```

iii. CONVERSION_NOTES Step 4: "Use imaging `ops['fs']=30` for elapsed seconds. Use behavior timestamps only to locate missing frame indices; avoids uncertain stored-unit scaling and minor camera-clock jitter." (The AI found the camera timestamp median increment to be ~3.359e-5 in unknown stored units, so it judged that clock unreliable as an absolute time base.)

## 3-b. What processing is involved in computing `input` *Time from start of experiment*?

i. The elapsed time assigned to bin *k* is the **centre** of the 10 frames that were averaged: `(10k + 4.5)/fs` seconds. So the first bin is 0.15 s and the last bin of a 30-min session is 1799.8167 s. The series is absolute and continuous across the whole session — it does **not** reset at each artificial 60-s trial boundary, so trial *t* of a session carries values in `[60t + 0.15, 60(t+1) − 0.1833]`. It is stored as float32, shape `(1, 180)` per trial. No normalisation, centring or scaling is applied.

ii.
```python
frame_centers_s = (
    np.arange(n_bins, dtype=np.float64) * RAW_FRAMES_PER_BIN
    + (RAW_FRAMES_PER_BIN - 1) / 2.0
) / ops["fs"]
input_binned = frame_centers_s[np.newaxis, :].astype(np.float32)
...
input_trials = split_trials(input_binned, bins_per_trial)
```

iii. CONVERSION_NOTES Step 5 mapping table: "Absolute elapsed time at the center of each 10-frame bin: mean of frame times … Does not reset at trial boundaries because requested input is from session start." Step 10 check 9: "The paper decoder did not use elapsed time as a covariate, but the requested decoder input explicitly requires it. Absolute bin-center time (rather than resetting at each artificial trial) implements 'from the beginning of the session' precisely." Step 10 check 3 independently regenerates all bin-centre times from raw frame counts for all 41 sessions and requires `np.allclose`; check 12 verifies `first = 0.15 s`, `last = duration − 5.5/30 s`.

## 3-c. How is the `input` *Time from start of experiment* aligned with the neural data?

i. By construction: the time vector is built on exactly the same bin grid as the binned neural matrix (`n_bins` from the same session), and both are passed through the same `split_trials` with the same `bins_per_trial`, so bin *k* of the input is the same 10 raw frames as bin *k* of the neural matrix. Shape equality (`neural.shape[1] == decoder_input.shape[1]`) is asserted per trial in `convert_session` and again in `validate_converted`.

ii.
```python
neural_trials = split_trials(neural_binned, bins_per_trial)
input_trials = split_trials(input_binned, bins_per_trial)
...
if not (neural.ndim == decoder_input.ndim == output.ndim == 2
        and neural.shape[1] == decoder_input.shape[1] == output.shape[1]):
    raise AssertionError("Trial time dimensions differ")
```

iii. CONVERSION_NOTES Step 10 check 3 and check 12: concatenating the converted input trials must exactly reproduce an independently constructed, uninterrupted session-long time series — "This checks every trial boundary because concatenation must equal the uninterrupted source-derived series." Passed for all 41 sessions at `rtol=1e-7, atol=1e-5`.

## 4-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `move_deve/motion_energy_glob.npy` — the pre-computed per-video-frame global motion energy (sum of squared pixel differences between consecutive frames). `move_deve/tstamps.npy` supplies camera timestamps used to locate dropped frames, and `move_deve/interframe_int.npy` is loaded purely as a consistency check (the AI asserts it equals `np.diff(tstamps)`). No video is re-processed.

ii.
```python
raw_motion = np.load(behavior_dir / "motion_energy_glob.npy")
timestamps = np.load(behavior_dir / "tstamps.npy")
intervals = np.load(behavior_dir / "interframe_int.npy")
if raw_motion.ndim != 1 or timestamps.shape != raw_motion.shape:
    raise ValueError(f"Invalid motion/timestamp shapes in {session_dir}")
if intervals.shape != (raw_motion.size - 1,):
    raise ValueError(f"Invalid interframe interval shape in {session_dir}")
if not np.allclose(intervals, np.diff(timestamps), rtol=1e-10, atol=1e-12):
    raise ValueError(f"interframe_int != diff(tstamps) in {session_dir}")
```

iii. CONVERSION_NOTES Step 3: "Motion energy is a whole-frame scalar: square consecutive-frame pixel differences and sum across pixels. Supplied motion energy is already processed, so this operation must not be repeated." Step 5 mapping table lists `interframe_int.npy` as "validation only … Redundant with timestamps."

## 4-b. What processing is involved in computing `output` *Motion energy*?

i. Four steps:
1. **Dropped-frame repair.** If the motion array is shorter than the neural array, each interframe interval is divided by that session's median interval and rounded to an integer number of frame steps; the cumulative sum gives the imaging-frame index of each observed video sample. The code then requires that the inferred number of missing frames equals the length deficit exactly *and* that the last observed sample lands on frame `n_frames−1`; otherwise it aborts. Missing samples are filled with `np.interp` (linear) at the inferred positions, and it asserts that no observed value was altered. If lengths already match, frames are paired directly with no repair. 276 frames were repaired across 9 sessions (of 1,962,000).
2. **Temporal averaging.** Non-overlapping mean of 10 frames, identical to the neural stream.
3. **Discretization.** Within-session quintiles (see 4-c).
4. Stored int64, shape `(1, 180)` per trial.

ii.
```python
median_interval = float(np.median(intervals))
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
observed_idx = np.concatenate((np.array([0], dtype=np.int64),
                               np.cumsum(frame_steps, dtype=np.int64)))
deficit = n_frames - raw_motion.size
inferred = int(np.sum(frame_steps - 1))
if inferred != deficit or observed_idx[-1] != n_frames - 1:
    raise ValueError(f"Timestamp gaps infer {inferred} missing frames, expected {deficit}, ...")
missing_mask[:] = True
missing_mask[observed_idx] = False
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)
if not np.allclose(repaired[observed_idx], raw_float):
    raise AssertionError(f"Motion repair altered observed values in {session_dir}")
```
```python
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "round each camera interval relative to its median to map observed values to imaging-frame indices, require the inferred missing count to equal the length deficit, then linearly interpolate. For equal-length streams, retain direct frame pairing even if timestamps jitter; this follows README's explicit length criterion." Step 2 justifies the length-based criterion empirically: "Three jm046 sessions have timestamp pauses but no length deficit, so length—not timestamp jitter alone—is the reliable missing-frame criterion described by the data README." Step 4 Key Decision 3 justifies averaging before discretizing.

## 4-c. How is `output` *Motion energy* thresholded into categories?

i. Five equal-percentile bins computed **independently for each session**, on the 10-frame-averaged motion trace over the whole session (all trials pooled), i.e. before trial splitting. Edges are the 20/40/60/80 % quantiles from `np.quantile` (linear interpolation); labels are assigned with `np.searchsorted(..., side="right")`, giving classes 0–4 with 0 = lowest motion. The code asserts the four edges are distinct and that each class receives ≈20 % of bins (tolerance = 1 bin); `validate_converted` re-checks per session that all five labels are present and that fractions are exactly 0.2. Value names are descriptive strings ("lowest motion (0-20%)" … "highest motion (80-100%)").

ii.
```python
quantile_edges = np.quantile(motion_binned, [0.2, 0.4, 0.6, 0.8])
if np.unique(quantile_edges).size != 4:
    raise ValueError(f"Non-distinct motion quintile thresholds in {session_dir}")
motion_classes = np.searchsorted(quantile_edges, motion_binned, side="right").astype(np.int64)
fractions = np.bincount(motion_classes, minlength=5) / motion_classes.size
if not np.allclose(fractions, 0.2, atol=1.0 / motion_classes.size):
    raise AssertionError(f"Unexpected quintile fractions {fractions} in {session_dir}")
```
```python
"output_names": ["motion_energy_quintile"],
"output_values": [["lowest motion (0-20%)", "low motion (20-40%)", "middle motion (40-60%)",
                   "high motion (60-80%)", "highest motion (80-100%)"]],
"motion_discretization": "five equal-percentile bins selected independently over each full session",
```

iii. Directly required by the Decoder Task ("Motion energy, discretized into five equal-percentile bins, selected per session"). CONVERSION_NOTES Step 5 Key Decision 5: "Select thresholds independently for each full session, as required. Use NumPy linear quantiles and right-sided boundary assignment. Tests show distinct thresholds and exactly `[0.2,0.2,0.2,0.2,0.2]` class fractions in every session." Step 12 addresses the leakage question: "session quintile edges are computed on full continuous sessions before splitting. They are unsupervised/reference preprocessing and the task explicitly requires bins 'selected per session'."

## 4-d. How is `output` *Motion energy* aligned with the neural data?

i. Camera and microscope are frame-synchronous — the paper states the microscope acquisition triggered every camera frame at 30 Hz — so video frame *i* corresponds to imaging frame *i*. The only thing that breaks this is dropped camera frames, which make the motion array shorter than `F`. The AI restores the one-to-one mapping by placing each observed motion sample at its inferred imaging-frame index (from rounded timestamp gaps) and interpolating the holes, checking that the reconstruction has exactly `n_frames` samples, that the last observed sample sits at the last imaging frame, and that no observed value changed. It also refuses to proceed if the motion array is *longer* than the neural array. After repair, motion is binned on the identical 10-frame grid and split with the identical `split_trials` call as the neural data, so alignment is exact by construction.

ii.
```python
if raw_motion.size > n_frames:
    raise ValueError(f"More behavior frames than neural frames in {session_dir}")
...
repaired_motion, missing_mask, raw_motion, observed_idx = repair_motion(session_dir, n_frames)
motion_binned = repaired_motion.reshape(n_bins, RAW_FRAMES_PER_BIN).mean(axis=1)
...
output_trials = split_trials(output_binned, bins_per_trial)
```

iii. CONVERSION_NOTES Step 3: "the microscope triggered each camera frame, providing direct framewise synchronization." Step 4: "Insert missing samples at timestamp-identified indices and linearly interpolate only those points. This maintains frame alignment, fixed trial lengths, and all neural data; record counts and test reconstructed non-missing values exactly." Step 10 check 4 independently reloads every raw motion/timestamp file, reconstructs the positions outside the conversion code, and requires every converted label and threshold to match; Step 12 adds a non-model alignment check: population-mean activity correlates positively with motion class in 36/41 sessions (median r = 0.124).

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is split between *repair* and *hard failure*:
- **Dropped camera frames (the one real defect, 276 frames in 9 sessions)**: repaired by timestamp-inferred insertion + linear interpolation, with three guards (inferred count == deficit, last observed index == `n_frames−1`, observed values unchanged). Counts are recorded per session in `metadata['session_info']['missing_motion_frames']`.
- **Suite2p `badframes`** (exactly one flagged frame in the whole dataset, jm032/2023-10-24): counted and reported in metadata but deliberately *not* excluded.
- **Everything else is a hard error, not a silent fix**: `F`/`Fneu` shape mismatch, `F` length ≠ `ops['nframes']`, frame count not divisible by 10, session not divisible into whole 60-s trials, `nplanes ≠ 1` or `fs ≤ 0`, motion longer than neural, `interframe_int ≠ diff(tstamps)`, non-finite processed neural values, degenerate quintile edges, wrong class fractions, dtype/shape/finiteness violations in `validate_converted` (run before the pickle is written).
- **Sessions missing any of the seven required files are skipped silently** by `discover_sessions` (no such session exists in this dataset).
- No remainder frames are ever dropped, because no session has any.

ii.
```python
if not np.isfinite(binned).all():
    raise ValueError(f"Non-finite processed neural values in {session_dir}")
```
```python
if inferred != deficit or observed_idx[-1] != n_frames - 1:
    raise ValueError(...)
if not np.allclose(repaired[observed_idx], raw_float):
    raise AssertionError(f"Motion repair altered observed values in {session_dir}")
```
```python
def validate_converted(data: dict[str, Any]) -> None:
    """Fail early on structural or statistical conversion mistakes."""
    ...
    if not np.array_equal(np.unique(labels_concat), np.arange(5)):
        raise AssertionError(f"Session {session} does not contain labels 0..4")
    if not np.allclose(np.bincount(labels_concat, minlength=5) / labels_concat.size, 0.2):
        raise AssertionError(f"Session {session} quintiles are imbalanced")
```

iii. The data README says missing camera frames "can be obtained by looking at 'tstamps.npy' or 'interframe_int.npy' and treated as missing values for motion energy or they can be interpolated over"; the AI chose interpolation because "This maintains frame alignment, fixed trial lengths, and all neural data" (Step 4). The fail-fast philosophy is stated in Step 6 ("strict source-shape/parameter validation … pre-save validation") and Step 5 Key Decision 6 ("all arrays are finite, missing behavior is recoverable"). Retaining the bad frame is justified in Step 4: "paper does not prescribe deletion, and dropping one frame would break synchronized complete-minute blocks."

## 6-a. What are the most time-consuming steps of the code?

i. The full conversion takes 44 s for 41 sessions (0.29–1.81 s per session). The dominant cost is the per-session neural preprocessing — the Gaussian + 60-s `minimum_filter1d`/`maximum_filter1d` sweep over every neuron × frame (up to 746 × 54,000) — together with the memory-mapped read of `F.npy`/`Fneu.npy`. The AI instrumented this explicitly: `convert_session` times `baseline_correct_and_bin` separately and stores it as `neural_processing_seconds` in each session's metadata, and the per-session wall time is printed as the run progresses. Per-session times scale with neurons × frames (0.3 s for 221×36,000 vs 1.8 s for 746×54,000), confirming the filtering is the bottleneck. Pickling 395 MiB takes 0.30 s. Plot rendering (`--show-processing`, ≤2 sessions) is a noticeable but optional cost.

ii.
```python
t_neural = time.perf_counter()
neural_binned, neural_diag = baseline_correct_and_bin(session_dir, ops, collect_diagnostics)
neural_seconds = time.perf_counter() - t_neural
...
"neural_processing_seconds": neural_seconds,
```
```python
print(f"[{index + 1:02d}/{len(selected):02d}] {session_id}: "
      f"{len(neural)} trials, {neural[0].shape[0]} neurons, "
      f"{info['missing_motion_frames']} missing motion frames, {elapsed:.2f} s", flush=True)
...
print(f"Saved {args.outpicklefile} (... MiB) in {save_seconds:.2f} s")
print(f"Total conversion time: {total_seconds:.2f} s", flush=True)
```

iii. CONVERSION_NOTES Step 7 extrapolates from the sample run using a "Neural-element-scaled estimate (full/sample ratio 21.34x) — 3.81 s / 48.24M elements → ~81 s conservative total", i.e. the AI explicitly modelled runtime as proportional to the neural array size, and concluded "it is far below 15 minutes, so no further optimization is required before full conversion." The actual 44 s beat that estimate.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. Very little is left un-vectorized. The remaining Python loops are:
- the loop over sessions (unavoidable — each session is a separate set of files, and it is I/O plus SciPy-bound anyway);
- the 64-neuron chunk loop in `baseline_correct_and_bin`, which is a *deliberate* de-vectorization to bound peak memory; processing all 746 neurons in one call would be somewhat faster but would hold several ~160 MB float32 intermediates at once;
- the list comprehension in `split_trials` and the per-trial shape/dtype checks in `convert_session`/`validate_converted`, which build/inspect the required list-of-arrays output and cannot be collapsed further given the target format.

Notably, the dropped-frame repair — the one loop the human reference identified as vectorizable (`np.insert` one frame at a time) — is already fully vectorized here via `np.rint`/`np.cumsum`/`np.interp`. The per-chunk work (neuropil subtraction, three SciPy 1-D filters, reshape-mean) is array-level across all neurons in the chunk.

ii.
```python
for start in range(0, n_neurons, NEURAL_CHUNK_SIZE):   # deliberate memory-bounding chunk loop
    stop = min(start + NEURAL_CHUNK_SIZE, n_neurons)
    ...
    binned[start:stop] = processed.reshape(stop - start, n_bins, RAW_FRAMES_PER_BIN).mean(axis=2)
```
```python
frame_steps = np.maximum(np.rint(intervals / median_interval).astype(np.int64), 1)
observed_idx = np.concatenate((np.array([0], dtype=np.int64), np.cumsum(frame_steps, dtype=np.int64)))
repaired = np.interp(np.arange(n_frames), observed_idx, raw_float)   # vectorized repair
```

iii. CONVERSION_NOTES Step 6: "Code inefficiencies identified: Loading/processing a full 746 x 54,000 fluorescence array with several intermediate copies could require >600 MB per session… Code speedups added: Neurons are processed in vectorized 64-neuron chunks; SciPy's O(n) 1-D filters operate across each chunk; F/Fneu use memory mapping; only the 10x smaller temporally averaged result is retained; trial slices are copied into compact contiguous arrays." The AI traded a small amount of speed for bounded memory, and documented that trade.

## 6-c. What processing does the code repeat multiple times?

i. Almost nothing substantive is recomputed. The small repetitions are:
- **Duplicated structural validation.** `convert_session` already asserts per-trial shapes and equal trial counts for each session; `validate_converted` then walks every session and every one of the 1,090 trials again to re-check shapes, dtypes, finiteness and label distributions, and recomputes the per-session class histogram that `convert_session` already computed as `fractions`.
- **`np.diff(timestamps)`** is recomputed only to compare it with the stored `interframe_int.npy`, which by definition already holds that difference.
- **`ops.npy` is opened separately** from `F.npy`/`Fneu.npy`, so the session directory is touched twice.
- Each raw file is otherwise read exactly once per session, and the baseline/bin/quantile computations are each performed once.

ii.
```python
fractions = np.bincount(motion_classes, minlength=5) / motion_classes.size   # in convert_session
if not np.allclose(fractions, 0.2, atol=1.0 / motion_classes.size): ...
```
```python
# again, over every trial of every session, in validate_converted
labels_concat = np.concatenate(labels)
if not np.allclose(np.bincount(labels_concat, minlength=5) / labels_concat.size, 0.2):
    raise AssertionError(f"Session {session} quintiles are imbalanced")
```
```python
if not np.allclose(intervals, np.diff(timestamps), rtol=1e-10, atol=1e-12):
    raise ValueError(f"interframe_int != diff(tstamps) in {session_dir}")
```

iii. These repeats are intentional correctness checks rather than oversights: CONVERSION_NOTES Step 6 describes "strict source-shape/parameter validation … and pre-save validation" as separate layers, and Step 5's sanity-check list includes a final structural pass over the assembled dictionary. The `diff(tstamps)` recomputation is described in the Step 5 mapping table as "validation only — Confirm it equals `diff(tstamps)`". All are O(n) on already-loaded arrays and invisible in the 44 s runtime.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items are produced but never consumed by the decoder:
- **`iscell.npy` is required to exist but never read**, and `ops['nchannels']`, `ops['prctile_baseline']` and `ops['badframes']` are parsed although the first two are unused for `maximin` data and the bad-frame count only lands in metadata.
- **Dead baseline branches**: the `constant`, `constant_prctile` and fallback branches of `baseline_correct_and_bin` can never execute for this dataset (every `ops['baseline']` is `maximin`). They were copied from the reference function for fidelity.
- **Audit-only metadata**: `motion_quintile_edges`, `motion_class_fractions`, `suite2p_badframes`, `source_motion_frames`, `neural_processing_seconds` and the whole 41-entry `session_info` list are stored for provenance and never used by `train_decoder.py`.
- **`interframe_int.npy`** is loaded purely to cross-check `diff(tstamps)`.
- **Diagnostics arrays** (`raw_f`, `raw_fneu`, `neuropil_corrected`, `baseline`, `observed_idx`, `missing_mask`, …) are gathered and a 6-panel figure rendered — but only under `--show-processing`, and only for the first two sessions, so they cost nothing in the normal full run.

None of this is expensive; the largest is the `session_info` list, which is kilobytes against a 395 MiB pickle.

ii.
```python
required = (..., plane / "iscell.npy", ...)   # existence gate only; contents never read
```
```python
elif baseline_mode == "constant":       # unreachable for this dataset
    ...
elif baseline_mode == "constant_prctile":
    ...
```
```python
info = {..., "suite2p_badframes": ops["badframes"],
        "motion_quintile_edges": [float(x) for x in quantile_edges],
        "motion_class_fractions": [float(x) for x in fractions],
        "neural_processing_seconds": neural_seconds}
```
```python
if collect_diagnostics and start == 0:
    diagnostics = {"raw_f": raw_f[:nplot_neurons, :nplot_frames].copy(), ...}
```

iii. CONVERSION_NOTES Step 5 mapping table justifies keeping provenance without bulk: "`metadata['session_info']` … Per-session dictionaries with ID, subject/date, source/raw sizes, duration/trial count, neuron count, missing-frame count, quintile edges — Allows audit without embedding unused large arrays", and explicitly excludes the heavy unused objects: "ROI `stat`, `iscell`, Suite2p images/offsets, FOV → metadata summary only … do not copy large imaging objects". Step 6 notes that diagnostics are limited to "at most two sessions".
