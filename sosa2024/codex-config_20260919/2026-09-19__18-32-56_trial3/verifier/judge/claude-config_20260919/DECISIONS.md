# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Every `.nwb` file under `/app/data/sub-*/` is discovered with a glob and sorted by numeric mouse ID then numeric session ID (152 files, 11 mice). Files are opened directly as HDF5 with `h5py` (not `pynwb`), reading only the datasets needed: `processing/behavior/BehavioralTimeSeries/*` (dense per-frame streams plus the sparse `Reward` series), `processing/ophys/Deconvolved/plane*/data`, `processing/ophys/Fluorescence|Neuropil/plane*/data` (for QC only), `processing/ophys/ImageSegmentation/PlaneSegmentation/iscell`, and the metadata fields `general/subject/subject_id`, `general/session_id`, `identifier`. Each session is processed one at a time and appended to per-session lists. `--sample` restricts processing to two hand-picked sessions (m12 ses-10, m18 ses-11).

ii.
```python
DATA_ROOT = Path("/app/data")

def natural_file_key(path: Path) -> tuple[int, int]:
    match = re.search(r"sub-m(\d+)_ses-(\d+)", path.name)
    ...
    return int(match.group(1)), int(match.group(2))

def discover_files(sample: bool) -> list[Path]:
    files = sorted(DATA_ROOT.glob("sub-*/*.nwb"), key=natural_file_key)
    if not files:
        raise FileNotFoundError(f"No NWB files found under {DATA_ROOT}")
    if not sample:
        return files
    wanted = {(12, 10), (18, 11)}
    ...
```
```python
with h5py.File(path, "r") as nwb:
    behavior = nwb[BEHAVIOR_ROOT]          # "processing/behavior/BehavioralTimeSeries"
    subject = decode_text(nwb["general/subject/subject_id"])
    session_id = decode_text(nwb["general/session_id"])
    scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
    timestamps = behavior["position/timestamps"][:]
    position = behavior["position/data"][:]
    ...
```

iii. From CONVERSION_NOTES Step 2/4: the DANDI dandiset (001361, version 0.251124.0550) declares 152 assets and 11 subjects; the local tree has one NWB per mouse/day, 14 days for 10 mice and 12 for m11 (imaging began on day 3 for m11, as stated in the Methods). The agent notes that the reference repository has no NWB loader at all (it loads pre-NWB `dill` `sess` pickles), so "NWB loading details must therefore come from the file hierarchy… while preserving the reference processing semantics"; reading the HDF5 groups directly is described as "a format adaptation, not a processing change" and allows lazy, per-cell slab reads instead of materializing whole arrays.

## 1-b. How are the data split into subjects?

i. Subjects are the distinct mice parsed from the file names (`sub-m<N>`), sorted numerically, giving `['m3','m4','m7','m11','m12','m13','m14','m15','m17','m18','m19']`. Each session's subject index is looked up from the NWB's own `general/subject/subject_id` string, so the file-name parse and the in-file metadata must agree.

ii.
```python
subjects = sorted(
    {f"m{natural_file_key(path)[0]}" for path in files},
    key=lambda value: int(value[1:]),
)
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
"subjects": subjects,
"subject_idx": np.asarray(
    [subject_lookup[item["subject"]] for item in converted_sessions], dtype=np.int16
),
```

iii. Step 2/4 of CONVERSION_NOTES: 11 mice are found, matching the paper's "n = 11 mice" switch cohort (the 3 fixed-condition mice are described in the paper but absent from the released data). Using the in-file `subject_id` for the index is an implicit consistency check between the directory/file naming and the NWB metadata.

## 1-c. How are the data split into sessions?

i. One session = one NWB file = one mouse-day. All 152 are converted; no session is dropped (a session is only rejected if fewer than two trials survive, which never happens). Sessions are ordered by mouse then day.

ii.
```python
for index, path in enumerate(files):
    print(f"[{index + 1}/{len(files)}] {path.relative_to(DATA_ROOT)}", flush=True)
    converted, info = convert_session(path, show_processing and index < 2)
    converted_sessions.append(converted)
    session_info.append(info)
...
if len(neural_trials) < 2:
    raise ValueError(f"{session_tag} retains fewer than two trials")
```

iii. Key decision 1 in Step 5: "Include all 152 imaged NWB sessions. The decoder task is not limited to switch days, and every available session has the requested variables and ≥2 valid trials." Session identity/scene/day are recorded per session in `metadata['session_info']`. Cross-day cell tracking (the paper's ROI aligner) is deliberately not used, so neurons are not matched across sessions.

## 1-d. How are the data split into trials?

i. A trial is the half-open frame interval `[trial_start frame, teleport frame)`: starts are the frames where the binary `trial_start` stream is non-zero, ends are the frames where the binary `teleport` stream is non-zero. The teleport frame itself is excluded (its position value is an interpolation artifact), and the inter-trial teleport/jitter period is excluded. The code asserts equal numbers of starts and stops and that the intervals are ordered and non-overlapping.

ii.
```python
def trial_bounds(group: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    starts = np.flatnonzero(group["trial_start/data"][:] > 0)
    stops = np.flatnonzero(group["teleport/data"][:] > 0)
    if starts.size != stops.size or starts.size < 2:
        raise ValueError(f"Invalid trial boundaries: {starts.size} starts, {stops.size} stops")
    if np.any(starts >= stops) or np.any(starts[1:] <= stops[:-1]):
        raise ValueError("Trial starts/stops are not ordered non-overlapping intervals")
    return starts, stops
...
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    trial_position = position[start:stop]
```

iii. Step 4/5: "Trials are exactly delimited by nonzero `trial_start` and `teleport` samples… Trials extend from track entry/start to teleport/end and exclude intertrial teleport periods." The agent explicitly considered and rejected the reference repo's legacy `start-1:stop-1` slicing: those indices came from the pre-NWB one-based pipeline, whereas the NWB stores explicit event frames, so shifting would place time 0 one frame before the actual start (on a frame with negative teleport-zone position). The legacy shift is reproduced *only* inside the interneuron-QC dF/F computation, to keep that filter numerically comparable to the reference. This yields 12,216 source trials.

## 1-e. How are trials filtered based on quality controls?

i. Exactly one trial-level exclusion: trials in which more than 30% of frames have a cumulative lick count > 2 are dropped entirely (neural, input and output). This is the paper's lick-sensor-fault criterion; it removes 81 of 12,216 trials, leaving 12,135. There is no minimum-trial-length filter, no speed filter, and no exclusion of omission trials. The code additionally raises (rather than filters) if a trial contains a non-scanning frame, more than one environment value, or more than one trial ID.

ii.
```python
LICK_FAULT_FRACTION = 0.30
...
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)],
    dtype=bool,
)
...
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    if bad_trials[raw_trial]:
        continue
    if not np.all(scanning[start:stop] == 1):
        raise ValueError(f"Non-scanning sample inside {session_tag} trial {raw_trial}")
    env_values = np.unique(environment[start:stop])
    trial_values = np.unique(trial_number[start:stop])
    if env_values.size != 1 or env_values[0] not in (0, 1):
        raise ValueError(...)
```

iii. Methods: "A very small number of trials with erroneous lick detection… were removed from subsequent licking analysis… (~0.65% of all imaged trials, n = 81 out of 12,376 trials)… detected by >30% of the 0.0645 s imaging frame samples in the trial containing a cumulative lick count >2." The agent noticed that the repository's GLM helper uses a stale 0.35 threshold (which flags 69 trials) and chose the paper's 0.30 (which flags exactly 81, reproducing the paper's count). It removes the whole trial rather than NaN-ing the lick trace because "the target format has no per-output missing mask, and lick is required for every trial" (Step 4/Step 5 decision 6). The paper's speed > 2 cm/s sample restriction is deliberately *not* applied because the requested output specification defines an explicit `< 2 cm/s` class.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The exported neural matrices are the stored `processing/ophys/Deconvolved/plane*/data` arrays (frames × ROIs), subset to retained cells and concatenated across planes in global ROI order. The raw `Fluorescence` and `Neuropil` arrays are read, but only to recompute dF/F for the interneuron QC mask; they are not used to build the exported neural signal. No dF/F or deconvolution is recomputed for the output.

ii.
```python
def load_deconvolved(nwb, keep_global, n_behavior) -> np.ndarray:
    chunks, ids_all = [], []
    deconvolved = nwb[f"{OPHYS_ROOT}/Deconvolved"]
    for plane_name in sorted(deconvolved):
        group = deconvolved[plane_name]
        global_ids = group["rois"][:].astype(np.int64)
        local = np.flatnonzero(keep_global[global_ids])
        if local.size:
            chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
            ids_all.append(global_ids[local])
    activity = np.concatenate(chunks, axis=1)
    ids = np.concatenate(ids_all)
    order = np.argsort(ids)
    if not np.array_equal(ids[order], np.flatnonzero(keep_global)):
        raise ValueError("Plane ROI references do not match the segmentation table")
    return activity[:, order]
...
neural = activity[start:stop].T.copy()
```

iii. Step 1 note: "dF/F does need to be computed in the raw preprocessing pipeline, but the reference repository's shared processed products already contain both `dff` and deconvolved `events`; the manuscript's position decoder and place-cell analyses use `events`. If the provided NWB files expose these processed event traces, they should be loaded rather than recomputing dF/F from fluorescence." Step 3: "The provided `Deconvolved` stream is the resulting manuscript neural representation." Step 5 decision 4: "Use author-provided OASIS-deconvolved calcium events, the stream used by their decoder. Recomputing deconvolution would add numerical differences without benefit. Recompute dF/F only transiently to apply the paper's r>0.5 interneuron mask."

## 2-b. How is the `neural` data processed?

i. Essentially no processing. The stored deconvolved values are cast to float32, truncated to the behavior length (10 two-plane sessions have one extra terminal neural frame), restricted to retained cells, pooled across planes in global ROI order, sliced per trial and transposed to (n_neurons, n_timepoints). There is no neuropil subtraction, no per-trial maximin baseline, no dF/F normalization, no Gaussian smoothing, no OASIS deconvolution, and no per-cell scaling/normalization of the exported values. The paper's dF/F pipeline is implemented in the script, but only inside `compute_interneuron_mask`, and its output is discarded after the speed correlations are computed.

ii.
```python
# only place the paper's dF/F pipeline appears; result used only for the QC correlation
corrected = f - 0.7 * f_neu + 0.7 * f_neu.mean(axis=1, keepdims=True)
baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
with np.errstate(divide="ignore", invalid="ignore"):
    dff = (corrected - baseline) / np.abs(baseline)
dff = ndimage.gaussian_filter1d(dff, 2, axis=1)
```
```python
# exported neural signal: stored deconvolved values, unmodified
chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))
...
neural = activity[start:stop].T.copy()
```
and in the metadata:
```python
"neural_signal": "author-produced OASIS-deconvolved calcium events",
```

iii. Step 5 decision 4 and the Step 10 comparison: "`convert_data.py::convert_session` therefore reads the NWB datasets directly while selecting the same synchronized behavior and OASIS `Deconvolved` event streams. This is a format adaptation, not a processing change." The agent's premise is that the NWB `Deconvolved` array already *is* the manuscript's `events`, so recomputation would only introduce numerical noise. (Note: this premise is not verified anywhere in CONVERSION_NOTES — no comparison between the stored stream and a recomputation was performed.)

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters, both the paper's. (1) Suite2p manual curation: only ROIs with `iscell[:,0] == 1` are kept. (2) Putative interneurons are excluded: for every curated cell the paper's dF/F is recomputed from `Fluorescence`/`Neuropil` over the concatenated trial windows and Pearson-correlated with running speed; cells with r > 0.5 are dropped. Across the full dataset this removes 409 of 138,678 curated cells (0.29%), leaving 138,269 neurons (154–2,323 per session). Place-cell selection and the speed > 2 cm/s sample mask are deliberately not applied.

ii.
```python
INTERNEURON_R_THRESHOLD = 0.5
...
segmentation = nwb[f"{OPHYS_ROOT}/ImageSegmentation/PlaneSegmentation"]
iscell = segmentation["iscell"][:, 0].astype(bool)
keep, speed_correlations = compute_interneuron_mask(nwb, iscell, speed, starts, stops)
n_interneurons = int(np.sum(iscell & ~keep))
activity = load_deconvolved(nwb, keep, n_behavior)
```
```python
qc_segments = [(max(0, int(a) - 1), max(0, int(b) - 1)) for a, b in zip(starts, stops)]
...
for start, stop in qc_segments:          # streaming sufficient statistics over trial windows
    ...
    sum_xy += (dff * y[None, :]).sum(axis=1)
numerator = sum_xy - sum_x * sum_y / n_samples
denominator = np.sqrt((sum_xx - np.square(sum_x) / n_samples)
                      * (sum_yy - sum_y * sum_y / n_samples))
corr = numerator / denominator
keep[ids[corr > INTERNEURON_R_THRESHOLD]] = False
```

iii. Step 3/4: `iscell` is the Suite2p manual curation the Methods describe; the paper additionally excludes "putative interneurons" with dF/F–speed Pearson r > 0.5 (0.42 ± 0.85% of cells). The NWB contains no saved dF/F or interneuron mask, so the agent decided to "recreate the exact paper dF/F preprocessing only to compute this mask". It documents that the parameters match the reference `preprocessing.dff` (0.7 neuropil coefficient, Gaussian σ 15, 300-frame min then max filter, σ 2 smoothing) and the reference `spatial.is_putative_interneuron` threshold of 0.5, and that the legacy `start-1:stop-1` windows are reproduced for this computation only. Place cells are not selected because "requested outputs include speed/lick/outcome as well as position, so non-place neurons can carry relevant information".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to trial start, and because the deconvolved rows are already on the same imaging-frame grid as the behavior, alignment requires nothing beyond slicing the same `[start:stop)` frame indices out of every stream. The first sample of every trial is the `trial_start` frame, so `input[0][0] == 0` by construction; this is asserted in the validator. `metadata['temporal_alignment_event']` = "first imaging frame with trial_start > 0 (entry to 0 cm track)", `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
...
neural = activity[start:stop].T.copy()
...
if inp[0, 0] != 0 or np.any(np.diff(inp[0]) <= 0):
    raise AssertionError(f"Time alignment failure in session {s}")
```

iii. Step 4/5: neural rows and dense behavior samples share the imaging grid (the NWB distributes VR data already synchronized to imaging frames by the authors' `vr_align_to_2P`), so "no temporal resampling is needed". The one-frame-longer neural arrays in 10 two-plane sessions are trimmed to the behavior length; no trial reaches those frames.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 64.4836272 ms per bin (15.5078125 Hz), the native per-plane imaging rate. No rebinning, resampling or interpolation is performed. The constant is hard-coded and checked against every session's timestamps to a tolerance of 1e-10 s; any non-uniform session would abort the conversion. For the two-plane m17/m18 sessions the agent notes that the advertised 31.015625 Hz ophys rate is the interleaved scanner rate and that the stored rows are already per-plane at 15.5 Hz.

ii.
```python
DT_SECONDS = 1.0 / 15.5078125
...
if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
    raise ValueError(f"Nonuniform or unexpected timestamps in {path}")
...
"time_bin_size": DT_SECONDS * 1000.0,
"sampling_rate_hz": 1.0 / DT_SECONDS,
```

iii. Methods: "sampling rate of ~15.5 Hz per plane"; "0.0645 s imaging frame samples". Step 4: "Use aligned timestamps, 64.4836 ms/bin. The 31-Hz metadata is interleaved acquisition, not per-plane row spacing." Keeping the native grid preserves all samples and keeps neural and behavior streams trivially aligned.

## 3-a. What variables in the raw data is `input` *Time from start of trial in seconds* derived from?

i. The dense behavior timestamps, read from `position/timestamps` (the code verifies that all ten dense streams have identical length, and the `DT_SECONDS` check makes the grid uniform, so the choice of which stream's timestamps to read is immaterial).

ii.
```python
timestamps = behavior["position/timestamps"][:]
...
dense_lengths = {
    name: behavior[f"{name}/data"].shape[0]
    for name in ["position", "speed", "lick", "environment", "trial number",
                 "reward_zone", "scanning", "trial_start", "teleport"]
}
if set(dense_lengths.values()) != {n_behavior}:
    raise ValueError(f"Dense behavior length mismatch in {path}: {dense_lengths}")
```

iii. Step 2: "All dense behavioral variables within a file have identical timestamps and lengths. Median frame interval is exactly 0.0644836272 s (15.5078125 Hz) in every session." Timestamps are the authors' VR-to-imaging synchronized clock.

## 3-b. What processing is involved in computing `input` *Time from start of trial in seconds*?

i. Subtract the timestamp of the trial's first frame, cast to float32. Nothing else; the result runs 0 to (T−1)·0.0645 s and is stored as a time-varying row.

ii.
```python
relative_time = (timestamps[start:stop] - timestamps[start]).astype(np.float32)
decoder_input = np.vstack([
    relative_time,
    np.full(T, env_values[0], dtype=np.float32),
    np.full(T, trial_values[0], dtype=np.float32),
    np.full(T, previous_outcomes[raw_trial], dtype=np.float32),
]).astype(np.float32, copy=False)
```

iii. Required by the instruction "Temporally align based on start of the trial"; the validator asserts `input[0][0] == 0` and strict monotonicity in every trial of every session.

## 3-c. How is the `input` *Time from start of trial in seconds* aligned with the neural data?

i. No alignment step is needed: the same `[start:stop)` frame indices index the neural array and the timestamps, and the dense-length check plus the uniform-Δt check guarantee the two grids coincide. Trial length T is identical for neural, input and output, and this is re-asserted per trial and again in `validate_converted`.

ii.
```python
if neural.shape[1] != T or decoder_input.shape != (4, T) or decoder_output.shape != (6, T):
    raise AssertionError(f"Shape mismatch in {session_tag} trial {raw_trial}")
...
for neural, inp, out in zip(data["neural"][s], data["input"][s], data["output"][s]):
    T = neural.shape[1]
    if neural.shape[0] != n_neurons or inp.shape != (4, T) or out.shape != (6, T):
        raise AssertionError(f"Converted shape mismatch in session {s}")
```

iii. Step 4: the NWB distributes behavior already resampled onto the imaging-frame grid, so "no temporal resampling is needed"; Step 10 check 6 reports the alignment was verified directly (time starts at 0, track-entry position non-negative, teleport sample absent).

## 4-a. What variables in the raw data is `input` *Environment type* derived from?

i. The dense `environment` behavior stream (0 = ENV1, 1 = ENV2), the same variable the reference code calls `morph`.

ii.
```python
environment = behavior["environment/data"][:]
...
env_values = np.unique(environment[start:stop])
if env_values.size != 1 or env_values[0] not in (0, 1):
    raise ValueError(f"Invalid environment in {session_tag} trial {raw_trial}: {env_values}")
```

iii. Step 2/5: "`environment` (0=Env1, 1=Env2; -1 before synchronization)"; the mapping is taken from the reference `behavior.env_morph_dict` (`Env1: 0`, `Env2: 1`) and from `get_trial_types`, which reads `morph` as the unique value over each trial interval.

## 4-b. What processing is involved in computing `input` *Environment type*?

i. None beyond validating that exactly one value in {0, 1} occurs within each trial and broadcasting that value across the trial's timepoints as a float32 row.

ii.
```python
np.full(T, env_values[0], dtype=np.float32),
```

iii. The variable is constant within a trial by construction (the environment only changes between trials, at trial 30 of the 11 cross-environment days); Step 2 verified this for every trial of every session, so taking the unique value doubles as a consistency check.

## 5-a. What variables in the raw data is `input` *Trial number* derived from?

i. The dense `trial number` behavior stream from the NWB — the session-native zero-based lap index — rather than an enumeration counter. Its uniqueness within each `[trial_start, teleport)` window is asserted.

ii.
```python
trial_number = behavior["trial number/data"][:]
...
trial_values = np.unique(trial_number[start:stop])
if trial_values.size != 1:
    raise ValueError(f"Multiple trial IDs in {session_tag} trial {raw_trial}: {trial_values}")
```

iii. Step 5 mapping table: "use native zero-based experimental lap ID; broadcast… preserve original numbering across any excluded lick-fault trial". Using the stored lap ID means a trial dropped by lick QC leaves a gap rather than renumbering subsequent trials, which keeps trial number consistent with the previous-outcome variable (which also refers to the raw preceding trial).

## 5-b. What processing is involved in computing `input` *Trial number*?

i. None beyond the uniqueness check and broadcasting the single value across all timepoints of the trial as float32. Values run 0…(n_trials−1) within each session (max 99).

ii.
```python
np.full(T, trial_values[0], dtype=np.float32),
```

iii. The stored lap index is already the quantity requested ("Trial number (continuous, per trial)"); the verification log confirms the per-session range is [0, 79] to [0, 99].

## 6-a. What variables in the raw data is `input` *Previous trial outcome* derived from?

i. From the current session's per-trial reward outcomes, which are computed from the sparse `Reward/timestamps` series together with the dense `reward_zone` stream — that is, a trial counts as rewarded if a reward delivery timestamp falls inside the trial's time window **and** the reward-zone signal is active somewhere in that trial. The previous-trial outcome is the outcome of the preceding *raw* trial (even if that trial was later excluded by lick QC).

ii.
```python
def reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops):
    """Match behavior.get_trial_types: delivery plus zone evidence per trial."""
    out = np.zeros(starts.size, dtype=np.int8)
    for i, (start, stop) in enumerate(zip(starts, stops)):
        delivered = np.any((reward_timestamps >= timestamps[start])
                           & (reward_timestamps < timestamps[stop]))
        zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
        out[i] = int(delivered and zone_evidence)
    return out
...
outcomes = reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops)
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
```

iii. Step 5 mapping: "any delivery in previous `[start,teleport)` and zone signal… uses actual preceding experimental trial even if it is excluded for lick QC", explicitly modelled on the reference `behavior.get_trial_types`, which computes `np.any(tmp_reward > 0) and np.any(tmp_rzone > 0)`. Step 2 reports that every delivered reward co-occurs with a zone entry, and that 52 trials have zone entry without delivery (behavioral lapses), which correctly remain outcome 0.

## 6-b. What processing is involved in computing `input` *Previous trial outcome*?

i. Shift the outcome vector by one trial and prepend 0 for the first trial of each session; broadcast the resulting 0/1 value across all timepoints of the trial as float32. Comparison against reward timestamps is done in seconds against the trial's first and last frame times, rather than by index matching.

ii.
```python
previous_outcomes = np.r_[0, outcomes[:-1]].astype(np.int8)
...
np.full(T, previous_outcomes[raw_trial], dtype=np.float32),
```

iii. Step 5 decision 11: "Prior outcome refers to the preceding raw experimental trial, not preceding retained trial. First-trial history is 0 because the requested binary has no missing category." The encoding is documented in the metadata as "previous reward outcome 0=omitted-or-unavailable/1=rewarded".

## 7-a. What variables in the raw data is `output` *Distance to reward zone* derived from?

i. The dense `position` stream plus the trial's reward-zone identity. Zone identity is **not** inferred from the data: it is parsed from the session protocol encoded at the end of the NWB root `identifier` (e.g. `/data/InVivoDA/GCAMP12/02_03_2023/Env2_LocationA_to_C`), with the zone changing after trial 30 on switch sessions. Zone coordinates are the paper's A = 80–130, B = 200–250, C = 320–370 cm.

ii.
```python
ZONE_BOUNDS = {"A": (80.0, 130.0), "B": (200.0, 250.0), "C": (320.0, 370.0)}

def scene_zone_labels(scene: str, n_trials: int) -> np.ndarray:
    """Return A/B/C for each trial from the scene protocol name."""
    labels = re.findall(r"(?:Location)?([ABC])", scene)
    if len(labels) == 1:
        return np.full(n_trials, labels[0], dtype="<U1")
    if len(labels) == 2:
        if n_trials <= 30:
            raise ValueError(f"Switch scene {scene} has only {n_trials} trials")
        out = np.full(n_trials, labels[1], dtype="<U1")
        out[:30] = labels[0]
        return out
    raise ValueError(f"Could not parse reward-zone protocol from scene {scene!r}: {labels}")
...
scene = decode_text(nwb["identifier"]).rsplit("/", 1)[-1]
labels = scene_zone_labels(scene, starts.size)
```

iii. Step 1/4: this mirrors the reference `behavior.get_reward_zones`, which sets the zone from `sess.scene` with `change_trial=30` and the dictionary `X=[80,130], Y=[200,250], Z=[320,370]` (mapped `A→X, B→Y, C→Z`). Methods: "zone A, 80–130 cm; zone B, 200–250 cm; zone C, 320–370 cm" and "Each switch occurred after 30 trials." Step 2 corroborates the switch trial empirically: "All 11 cross-environment sessions change environment exactly at zero-based trial index 30."

## 7-b. What processing is involved in computing `output` *Distance to reward zone*?

i. Signed linear distance to the nearest edge of the 50-cm zone: `position − zone_start` when before the zone (negative), exactly 0 anywhere inside the zone (inclusive of both edges), `position − zone_end` when past it (positive). Computed vectorized per trial, then discretized.

ii.
```python
zone_start, zone_stop = ZONE_BOUNDS[label]
signed_distance = np.where(
    trial_position < zone_start,
    trial_position - zone_start,
    np.where(trial_position > zone_stop, trial_position - zone_stop, 0.0),
)
```

iii. Step 5 decision 10: "'distance to any location in the reward zone' is zero everywhere inside the 50-cm interval, negative to its near/start edge before entry, and positive to its far/end edge after exit. This uniquely gives the requested negative/zero/positive semantics." Step 1/3 note that this differs from the paper's own circular reward-relative coordinate (centered on the zone start), and record that the requested definition takes precedence: "The requested output differs explicitly… so Step 5 must define inside-zone distance as exactly 0."

## 7-c. How is `output` *Distance to reward zone* thresholded into categories?

i. Seven classes via explicit boolean masks: `< −50 → 0`, `[−50, −10) → 1`, `[−10, 0) → 2`, `== 0 → 3`, `(0, 10] → 4`, `(10, 50] → 5`, `> 50 → 6`. Every boundary is unit-tested at values just below, exactly at, and just above the threshold before any conversion runs.

ii.
```python
def discretize_distance(distance: np.ndarray) -> np.ndarray:
    out = np.empty(distance.shape, dtype=np.int8)
    out[distance < -50] = 0
    out[(distance >= -50) & (distance < -10)] = 1
    out[(distance >= -10) & (distance < 0)] = 2
    out[distance == 0] = 3
    out[(distance > 0) & (distance <= 10)] = 4
    out[(distance > 10) & (distance <= 50)] = 5
    out[distance > 50] = 6
    return out
...
def verify_discretization_edges() -> None:
    distance = np.array([-50 - eps, -50, -10 - eps, -10, -eps, 0, eps, 10, 10 + eps, 50, 50 + eps])
    expected_distance = np.array([0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6])
    if not np.array_equal(discretize_distance(distance), expected_distance):
        raise AssertionError("Distance discretization edge test failed")
```

iii. Step 5 decision 9: "Exact-boundary conventions follow every explicit strict inequality in the request; continuous data make unresolved endpoints negligible." Class 3 is reserved for exactly 0, i.e. being inside the zone, matching the requested "3: 0 cm" category. The resulting full-dataset distribution is [.251, .102, .073, .238, .021, .072, .243].

## 7-d. How is `output` *Distance to reward zone* aligned with the neural data?

i. Position is sliced with the same `[start:stop)` frame indices as the neural array, so the distance row is sample-for-sample aligned with the neural matrix; no shifting or interpolation.

ii.
```python
trial_position = position[start:stop]
...
decoder_output = np.vstack([
    discretize_distance(signed_distance),
    ...
])
neural = activity[start:stop].T.copy()
```

iii. Same rationale as 3-c: behavior is distributed already synchronized to imaging frames, the per-stream lengths are checked, and shape equality of neural/input/output is asserted per trial and in `validate_converted`.

## 8-a. What variables in the raw data is `output` *Absolute position* derived from?

i. The dense `position` behavior stream (cm along the 450 cm virtual track), used as-is.

ii.
```python
position = behavior["position/data"][:]
...
trial_position = position[start:stop]
```

iii. Step 2: "`position` (cm)… Track position progresses from approximately 0 to 450 cm; presynchronization/teleport samples include -500/-50 cm and are excluded by trial boundaries."

## 8-b. What processing is involved in computing `output` *Absolute position*?

i. None beyond the per-trial slice and discretization into the five requested 90-cm bins. Slightly out-of-range samples near the track ends are absorbed by the open first/last bins.

ii.
```python
decoder_output = np.vstack([
    discretize_distance(signed_distance),
    discretize_position(trial_position),
    ...
])
```

iii. The Methods define a 450 cm linear track, so five equal bins are 90 cm each; the conversion needs no unit change because the stream is already in cm. Resulting distribution: [.212, .177, .231, .226, .154].

## 8-c. How is `output` *Absolute position* thresholded into categories?

i. Five classes: `< 90 → 0`, `[90, 180) → 1`, `[180, 270) → 2`, `[270, 360] → 3`, `> 360 → 4`, again with explicit masks and an edge unit test.

ii.
```python
def discretize_position(position: np.ndarray) -> np.ndarray:
    out = np.empty(position.shape, dtype=np.int8)
    out[position < 90] = 0
    out[(position >= 90) & (position < 180)] = 1
    out[(position >= 180) & (position < 270)] = 2
    out[(position >= 270) & (position <= 360)] = 3
    out[position > 360] = 4
    return out
...
position = np.array([90 - eps, 90, 180 - eps, 180, 270, 360, 360 + eps])
expected_position = np.array([0, 1, 1, 2, 3, 3, 4])
```

iii. Step 5 decision 9 lists the intended intervals verbatim, matching the requested "0: < 90 cm … 4: > 360 cm"; the open end bins mean the handful of samples fractionally outside [0, 450] land in the terminal classes rather than producing invalid labels.

## 8-d. How is `output` *Absolute position* aligned with the neural data?

i. Identical `[start:stop)` indexing as the neural data; no further alignment.

ii.
```python
trial_position = position[start:stop]
neural = activity[start:stop].T.copy()
```

iii. Same rationale as 3-c/7-d; the processing plots additionally show continuous position with the discretized class overlaid on the trial's own time axis to demonstrate that there is no temporal offset.

## 9-a. What variables in the raw data is `output` *Lick* derived from?

i. The dense `lick` stream ("lick detection by capacitive sensor, cumulative per imaging frame").

ii.
```python
lick = behavior["lick/data"][:]
...
trial_lick = lick[start:stop]
```

iii. Step 2: the lick stream is the per-frame cumulative lick count; the reference GLM helper uses the same stream, clipping counts to 1.

## 9-b. What processing is involved in computing `output` *Lick*?

i. Binarize: any frame with count > 0 becomes 1. Trials flagged by the paper's lick-sensor-fault criterion were already removed (see 1-e), so no NaN masking is needed inside retained trials.

ii.
```python
(trial_lick > 0).astype(np.int8),
```
```python
"output_discretization": {..., "lick": "lick count >0"},
```

iii. Step 3/5: "The paper binarizes valid lick counts per frame"; the reference `glmUtils.get_timeseries_data` does `licks[licks > 1] = 1` on integer counts, which is identical to thresholding at > 0. Full-dataset distribution [.777, .223].

## 9-c. How is `output` *Lick* aligned with the neural data?

i. Same `[start:stop)` frame indices as the neural data — the lick counts are already accumulated per imaging frame by the authors' synchronization, so no resampling or event-time matching is required.

ii.
```python
trial_lick = lick[start:stop]
```

iii. Step 3 notes that the authors' VR-to-imaging alignment accumulates event counts during downsampling "so lick/reward events are not lost"; hence per-frame counts can be used directly.

## 10-a. What variables in the raw data is `output` *Reward zone location* derived from?

i. The session protocol string at the end of the NWB `identifier` (e.g. `Env1_LocationB_to_C`, `Env1_B_to_Env2_C`, `Env2_LocationA`), with the change occurring after trial 30 on switch sessions — the same source as 7-a. The dense `reward_zone` stream is *not* used to infer identity (it is used only as corroborating evidence for reward outcome).

ii.
```python
labels = re.findall(r"(?:Location)?([ABC])", scene)
...
out = np.full(n_trials, labels[1], dtype="<U1")
out[:30] = labels[0]
```

iii. See 7-a: this reproduces the reference `behavior.get_reward_zones` (scene-driven, `change_trial=30`), rather than inferring zone identity from behavior. Step 2 reports the resulting label counts as A: 4,186, B: 4,010, C: 4,020 over source trials (A/B/C ≈ 1/3 each after QC: .332/.336/.333 of timepoints).

## 10-b. What processing is involved in computing `output` *Reward zone location*?

i. Map the parsed letter to a class index (A→0, B→1, C→2) and broadcast it across all timepoints of the trial as an int8 row, so the per-trial variable can live in the same `(6, T)` array as the time-varying outputs.

ii.
```python
ZONE_TO_CLASS = {"A": 0, "B": 1, "C": 2}
...
np.full(T, ZONE_TO_CLASS[label], dtype=np.int8),
```

iii. Step 5 decision 8: per-trial values are broadcast because "one ndarray cannot mix a `(T,)` row with scalars… It also matches `train_decoder.py`'s timepoint-wise target convention." `output_values[4]` documents the classes as "A (80-130 cm)", "B (200-250 cm)", "C (320-370 cm)".

## 11-a. What variables in the raw data is `output` *Reward outcome* derived from?

i. The sparse `Reward/timestamps` series (reward deliveries, which carry their own timestamps) combined with the dense `reward_zone` stream, evaluated within each trial's `[trial_start, teleport)` time window — the same `reward_outcomes()` used for the previous-trial input.

ii.
```python
reward_timestamps = behavior["Reward/timestamps"][:]
...
delivered = np.any((reward_timestamps >= timestamps[start])
                   & (reward_timestamps < timestamps[stop]))
zone_evidence = np.any(reward_zone_signal[start:stop] > 0)
out[i] = int(delivered and zone_evidence)
```

iii. Explicitly modelled on the reference `behavior.get_trial_types`, which requires both `np.any(tmp_reward > 0)` and `np.any(tmp_rzone > 0)`. Step 2: "Sparse reward timestamps produce 10,342 rewarded and 1,874 unrewarded trials… Three sparse reward events occur outside analyzed start-to-teleport intervals and are ignored."

## 11-b. What processing is involved in computing `output` *Reward outcome*?

i. Per-trial binary (1 = rewarded, 0 = omitted/lapsed), broadcast across all timepoints as int8. Reward times are compared directly in seconds against the trial's bounding timestamps rather than being snapped to frame indices.

ii.
```python
outcomes = reward_outcomes(timestamps, reward_timestamps, reward_zone_signal, starts, stops)
...
np.full(T, outcomes[raw_trial], dtype=np.int8),
```

iii. Step 9: the converted trial-level reward rate is 10,271/12,135 = 84.64%, matching the Methods' "reward was randomly omitted on approximately 15% of trials". Step 12 notes the consequence for decoding: because the per-trial label is broadcast from trial start, most pre-delivery samples cannot contain evidence of the outcome, so near-chance balanced accuracy on this output is expected and was not "fixed" by redefining the label.

## 12. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is mostly defensive validation that aborts on anything unexpected, plus three specific accommodations:
- **Extra terminal neural frame** (10 two-plane sessions have one more ophys row than behavior rows): neural data is read only up to the behavior length; no trial reaches that frame.
- **Faulty lick sensor** (81 trials): whole trials excluded (see 1-e), with the excluded raw indices recorded in `metadata['session_info']`.
- **Teleport-frame position artifact**: the teleport frame is excluded from the trial, and the processing plot deliberately marks the last *included* frame instead.
Everything else raises: mismatched dense stream lengths, non-uniform timestamps, unequal/overlapping trial bounds, non-scanning samples inside a trial, multiple environment or trial IDs per trial, non-finite dF/F, speed, neural or input values, plane ROI IDs inconsistent with the segmentation table, sessions with fewer than two retained trials, and any output class outside its valid range.

ii.
```python
chunks.append(group["data"][:n_behavior, local].astype(np.float32, copy=False))   # trims extra frame
...
if set(dense_lengths.values()) != {n_behavior}:
    raise ValueError(f"Dense behavior length mismatch in {path}: {dense_lengths}")
if not np.allclose(np.diff(timestamps), DT_SECONDS, rtol=0, atol=1e-10):
    raise ValueError(f"Nonuniform or unexpected timestamps in {path}")
...
if not np.all(np.isfinite(dff)) or not np.all(np.isfinite(y)):
    raise ValueError(f"Non-finite dF/F or speed in {plane_name}")
...
if not np.all(np.isfinite(neural)) or not np.all(np.isfinite(decoder_input)):
    raise ValueError(f"Non-finite converted values in {session_tag} trial {raw_trial}")
if len(neural_trials) < 2:
    raise ValueError(f"{session_tag} retains fewer than two trials")
```

iii. Step 4/5: the one-frame surplus is attributed to the documented "scan-stop one-frame correction" in the reference mock aligner and is trimmed because "no trial reaches it"; the teleport sample is excluded because "the position value on the teleport event sample is… an unreliable interpolation between track end and tunnel". The agent's stated philosophy for everything else is to "fail explicitly rather than silently creating a different nan policy", so that a data anomaly cannot pass silently into the converted file. CONVERSION_NOTES Step 10 reports an edge-case audit (two-plane ROI ordering, first-trial history default, raw-trial history across exclusions, variable trial lengths, early-ended sessions, missing m11 days 1–2, switch at index 30, sparse reward timestamps, positions at trial endpoints) that found no off-by-one errors.

## 13-a. What are the most time-consuming steps of the code?

i. Per the agent's own accounting and the run log (383.4 s conversion + 7.8 s serialization for 152 sessions, 1.1–5.0 s per session), the dominant cost is the interneuron QC: it reads the full `Fluorescence` and `Neuropil` arrays for every curated cell in trial-sized slabs and runs Gaussian/minimum/maximum filters on them. Next are the HDF5 read of the `Deconvolved` rows for retained cells, and writing the 8.86 GiB pickle. The exported neural pipeline itself is cheap because nothing is recomputed.

ii.
```python
for block_start in range(0, local_cells.size, block_size):     # 128 cells at a time
    ...
    for start, stop in qc_segments:                            # ~80 trial slabs per block
        f = f_data[start:stop, local].T.astype(np.float64, copy=False)
        f_neu = n_data[start:stop, local].T.astype(np.float64, copy=False)
        ...
        baseline = ndimage.gaussian_filter1d(corrected, 15, axis=1)
        baseline = ndimage.minimum_filter1d(baseline, 300, axis=1)
        baseline = ndimage.maximum_filter1d(baseline, 300, axis=1)
```
```python
elapsed = time.perf_counter() - started
per_session = time.perf_counter() - session_started
remaining = per_session * (len(files) - index - 1)
print(f"  elapsed {elapsed:.1f}s; local remaining estimate {remaining:.1f}s", flush=True)
```

iii. Step 6: "The exact paper interneuron filter requires reading raw fluorescence and neuropil and applying per-trial image filters before deconvolved events are loaded. The final pickle necessarily copies each variable-length neural trial because the target format is a list of independent arrays." Step 7 estimated ~442 s for the full run from the sample timings; the measured 391 s came in under that, so no further optimization was pursued.

## 13-b. What loops in the code could have been vectorized to improve efficiency?

i. Remaining Python-level loops: (1) the per-trial conversion loop in `convert_session`; (2) the per-trial loop inside `reward_outcomes`, whose delivery test could be done for all trials at once with `np.searchsorted` on the reward timestamps; (3) the `bad_trials` list comprehension, which re-scans each lick window separately; (4) the nested plane/block/trial-segment loop in `compute_interneuron_mask`, which issues one fancy-index HDF5 read per (block, trial) pair — the per-trial filtering could instead be run once per block over a contiguous read. The heavy numeric work is already vectorized: dF/F, the correlation sufficient statistics, all discretizers, and all broadcasts are array operations.

ii.
```python
for i, (start, stop) in enumerate(zip(starts, stops)):     # reward_outcomes: per-trial python loop
    delivered = np.any((reward_timestamps >= timestamps[start])
                       & (reward_timestamps < timestamps[stop]))
```
```python
bad_trials = np.array(
    [np.mean(lick[a:b] > 2) > LICK_FAULT_FRACTION for a, b in zip(starts, stops)], dtype=bool)
```
```python
for raw_trial, (start, stop) in enumerate(zip(starts, stops)):
    ...
    decoder_output = np.vstack([...])
```

iii. Step 6: "Interneuron QC is vectorized over 128-cell blocks, uses sufficient statistics rather than retaining dF/F… Inputs/outputs are NumPy-vectorized, no resampling is performed." The per-trial loop is inherent to the target format, whose trials are variable-length independent arrays; vectorizing it would require padding or ragged bookkeeping for no real gain relative to the I/O cost.

## 13-c. What processing does the code repeat multiple times?

i. Little is repeated at the session level: each NWB is opened exactly once, and there is no separate survey pass. What is repeated is (1) the HDF5 slab read of `Fluorescence`/`Neuropil` for the same trial windows once per 128-cell block; (2) the per-trial neuropil mean, baseline filtering and smoothing for the QC, which duplicate work that would be shared if the exported signal were derived from the same dF/F; (3) in `--show-processing` mode, the signed distance and per-trial slices are recomputed inside `make_processing_plot` for the plotted trial; and (4) `trial_number`/`environment`/`scanning` are re-scanned per trial as validation.

ii.
```python
for block_start in range(0, local_cells.size, block_size):
    for start, stop in qc_segments:
        f = f_data[start:stop, local].T...       # same windows re-read for every block
```
```python
signed_distance = np.where(                      # recomputed in the plotting function
    pos < zone_start, pos - zone_start,
    np.where(pos > zone_stop, pos - zone_stop, 0.0),
)
```

iii. The block loop is a deliberate memory/speed trade-off ("bounds memory while matching reference filters", Step 7); the plot-side recomputation only runs for at most two sessions and is done from class-independent raw inputs on purpose, "to reconstruct continuous values for visual validation" independently of the converted arrays.

## 13-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest item is the full dF/F reconstruction in `compute_interneuron_mask`: the paper's complete preprocessing is run on every curated cell and then thrown away, retaining only one Pearson r per cell — and only 409 of 138,678 cells (0.29%) are actually removed by it. Because the exported neural signal is the stored deconvolved stream, none of that dF/F is reused. Smaller items: `speed_correlations` is a full per-ROI array of which only `np.nanmax` is stored; `scanning` and `trial number` are read solely to raise on impossible conditions; `verify_discretization_edges()` runs on every invocation; `plane_frames`/`scene`/`source_file` bookkeeping is written into metadata; and per-trial constant outputs (zone, outcome) plus per-trial constant inputs (environment, trial number, previous outcome) are materialized as full-length T rows, which multiplies their storage but is required by the target format.

ii.
```python
dff = ndimage.gaussian_filter1d(dff, 2, axis=1)   # used only for sum_x/sum_xy, then discarded
...
correlations[ids] = corr.astype(np.float32)
...
"max_dff_speed_correlation": float(np.nanmax(speed_correlations)),
```
```python
np.full(T, ZONE_TO_CLASS[label], dtype=np.int8),
np.full(T, outcomes[raw_trial], dtype=np.int8),
```

iii. Step 5 decision 8 justifies the broadcasting ("one ndarray cannot mix a `(T,)` row with scalars… matches `train_decoder.py`'s timepoint-wise target convention"), and Step 5 decision 14 lists variables that are read but intentionally unused (`autoreward`, ROI pixel masks, background images, `scanning` as an integrity check only). The dF/F work is justified as the only way to reproduce the paper's interneuron exclusion, since the NWB stores neither dF/F nor an interneuron mask.
