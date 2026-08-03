# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reconstructs the analyzed session list by parsing the authors' `load*_ALMVideo.m` scripts, intersecting those entries with `.mat` files found under `data/*/`. It then loads each selected session with either an HDF5 reader or `scipy.io.loadmat`, and loads motion energy from a separate `motionEnergy_<subject>_<date>.mat` file.

ii.
```python
def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        ...
        if {"subject", "date", "probes"} <= current.keys():
            key = (current["subject"], current["date"])
            if key in data_files:
                path = data_files[key]
                specs.append(SessionSpec(...))
```
```python
def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```
```python
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. In Step 4 and Step 5 of `CONVERSION_NOTES.md`, the AI says it used the intersection of the reference loader-script session lists and available data files so the converted dataset would match the paper's analyzed cohort rather than the broader raw archive.

## 1-b. How are the data split into subjects?

i. Each session carries `spec.subject`, obtained from the loader-script entry / filename, and subjects are accumulated while building the final dataset. `subject_idx` points from each session to that running subject list.

ii.
```python
SessionSpec(
    subject=current["subject"],
    date=current["date"],
    probes=current["probes"],
    session_path=path,
    folder=path.parent.name,
)
```
```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The notes justify this by saying filenames / loader scripts are the reliable source of subject identity across sessions.

## 1-c. How are the data split into sessions?

i. One parsed `SessionSpec` corresponds to one session. Each selected session file becomes one element in `neural`, `input`, and `output` after `convert_one_session`.

ii.
```python
for spec in session_specs:
    session = convert_one_session(spec, show_processing=show_processing, outdir=outdir)
    if session is None:
        continue

    neural.append(session["neural"])
    inputs.append(session["input"])
    outputs.append(session["output"])
```

iii. The AI's notes state that it retained the 44 sessions found in both the reference loader scripts and the provided data, treating fixed-delay and randomized-delay sessions uniformly.

## 1-d. How are the data split into trials?

i. Trials are treated as row-wise/per-index entries of the behavioral arrays. A boolean mask over per-trial fields selects valid trial indices, and each kept raw trial index becomes one converted trial.

ii.
```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )
```
```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
...
for local_idx, trial_idx in enumerate(selected_trials):
    ...
    output_trials.append(output_arr)
```

iii. The notes frame the behavioral arrays in `obj.bp` as the trial-defining source, with one array entry per trial.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out photostimulation trials, early-lick trials, and all non-hit/non-miss trials. It also drops trials whose trial number exceeds the last neural unit trial index. Unlike the reference solution, this removes ignore trials entirely.

ii.
```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )
```
```python
covered_trial_max = [
    int(np.nanmax(unit["trial"]))
    for unit in raw["units"]
    if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size
]
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. In Step 5, the AI explicitly decided to exclude stimulation, early-lick, and ignore/no-response trials, and in Step 9/10 it justified the final trial-number cutoff as a fix for sessions where behavior outlasted neural recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from per-unit `quality`, `trialtm`, and `trial` loaded from `obj.clu`, together with `bp.ev.goCue` for alignment.

ii.
```python
units.append(
    {
        "quality": mat_to_str(getattr(unit, "quality", "")),
        "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
        "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
    }
)
```
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The notes identify `obj.clu{probe}` spike times plus `goCue` as the reference neural source variables.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the AI aligns spike times to `goCue`, bins them at 5 ms, converts counts to Hz by adding `1/DT` per spike, and smooths the binned rate with its own `my_smooth` function implementing a causal half-Gaussian-like kernel with reflected padding.

ii.
```python
DT = 1.0 / 200.0
SMOOTH = 15
```
```python
def compute_unit_trial_matrix(unit: dict, go_cue: np.ndarray, trial_to_pos: np.ndarray, n_sel: int, time_edges: np.ndarray) -> np.ndarray:
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    ...
    if np.any(keep):
        bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
        np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat
```

iii. Step 6 says the AI intentionally reimplemented `mySmooth.m` and considered that closer to the reference than a simpler symmetric Gaussian.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units whose lower-cased quality label is not in `{"garbage", "gabrga", "noisy", "real?"}` and whose mean firing rate over the aligned window exceeds 1 Hz. It also skips sessions with fewer than 10 surviving units.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}
```
```python
for unit in raw["units"]:
    if not good_quality(unit["quality"]):
        continue
    unit_mat = compute_unit_trial_matrix(...)
    if float(unit_mat.mean()) <= LOW_FR_HZ:
        continue
    kept_units += 1
```
```python
if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
    return None
```

iii. Step 5 and Step 10 justify this as "all non-garbage manually curated units" plus a 1 Hz firing-rate cutoff from the paper, and the notes also mention the paper's session-level `>=10` unit inclusion rule.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned by subtracting the raw trial's `goCue` time from each spike's within-trial time.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The notes explicitly say `goCue` is the universal alignment event because that matches both the instructions and the default reference alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses a fixed 5 ms grid from `-2.5` s to `+2.5` s around `goCue`, yielding 1000 bins. It does not apply any further temporal rebinning.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
...
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. Step 5 says this was chosen to match the default reference pipeline (`dt = 1/200`, `[-2.5, 2.5] s`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a stored array; it is constructed as the session-wide time grid corresponding to alignment on `bp.ev.goCue`.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes describe this as storing the aligned neural time base itself as the decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI simply computes bin centers from the fixed global `[-2.5, 2.5]` window with 5 ms spacing and repeats that same vector for every trial.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. No additional justification beyond "use the shared go-cue-centered neural time base" appears in the notes.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time_vec` that defines the neural bins, so it is directly aligned with the neural data.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
unit_mat = compute_unit_trial_matrix(unit, raw["events"]["goCue"], trial_to_pos, selected_trials.size, time_edges)
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes state that all converted streams share the same go-cue-centered time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In code, the AI derives lick direction from the instructed-side flags `R`/`L` only. It does not use `hit`/`miss` to invert missed trials into the opposite lick direction.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. Step 5 says the AI kept only hit/miss trials so that direction would be "well defined," implying it treated instructed side as sufficient.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI converts each kept trial to a constant binary label over time: right `1` if `R==1`, otherwise left `0`.

ii.
```python
output_arr = np.vstack(
    [
        np.full(time_vec.size, lick_direction, dtype=np.int64),
        ...
    ]
)
```

iii. The notes justify constant time-series outputs as a way to put all outputs on the same `(n_output, n_timepoints)` grid.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `autowater`.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. Step 5 says `autowater=1` corresponds to WC, so `1 - autowater` gives DR/WC in the requested coding.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI remaps `autowater` to a constant binary label per trial: WC `0`, DR `1`.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
...
np.full(time_vec.size, context, dtype=np.int64),
```

iii. The notes say this coding follows the decoder prompt's required WC/DR label order.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Because the AI drops ignore trials beforehand, outcome is effectively derived from `hit` alone, with non-hit kept trials becoming incorrect.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. Step 5 explicitly says ignore/no-response trials are excluded rather than labeled.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI creates a constant binary time series per trial: correct `1` if `hit==1`, otherwise incorrect `0`.

ii.
```python
np.full(time_vec.size, outcome, dtype=np.int64),
```

iii. The notes justify this by aligning with the prompt's requested incorrect/correct coding after dropping ignores.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived only from the side-view DLC feature named `"tongue"` in `obj.traj[0]`, plus frame times and video-alignment metadata.

ii.
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. Step 5 says the AI deliberately used the side-view `tongue` marker to avoid "inventing an unreferenced cross-camera combination."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates x/y tongue position from video frame times onto the 5 ms neural time axis, computes speed as the magnitude of `np.gradient` of those interpolated coordinates, and then keeps only bins where both x and y are finite.

ii.
```python
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```
```python
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
...
return np.sqrt(xvel**2 + yvel**2)
```
```python
tongue_valid = np.isfinite(tongue_pos[0]) & np.isfinite(tongue_pos[1])
tongue_speed[~tongue_valid] = np.nan
```

iii. The notes justify this as using "reference-style interpolation to neural time base" while avoiding a cross-camera combination for tongue position.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session median over finite tongue-speed values from kept trials, then binarizes each bin as `>= median` vs `< median`. Non-finite bins are forced to `0`.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)
```
```python
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. Step 7 says an earlier all-ones bug was fixed by computing the percentile on visible frames only and assigning non-visible frames to bin `0`.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimates a session video offset from bitcode timing, subtracts that and `goCue`, interpolates tongue positions onto the neural `time_vec`, and then computes speed on that shared time base.

ii.
```python
def find_video_offset(raw: dict) -> float:
    bitstart = np.asarray(raw["bitcode_bitstart"], dtype=np.float64).reshape(-1)
    fs = float(raw["fs"])
    ...
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])
```
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes explicitly compare this to `findVideoOffset`, `findPosition`, and `findVelocity`, arguing that interpolation onto the neural time base is the intended alignment path.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from both bottom-view DLC paw features, `"top_paw"` and `"bottom_paw"`, in `obj.traj[1]`.

ii.
```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. Step 5 says the AI averaged top- and bottom-paw speed magnitudes to capture overall paw movement.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature, the AI interpolates x/y position onto the neural time axis, computes gradient-based speed, applies nearest-neighbor filling inside `feature_xy`/`feature_speed` for non-tongue features, then averages the two paw speeds per time bin.

ii.
```python
if "tongue" not in feature_name:
    xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
    ypos[:, trix] = nearest_fill_1d(ypos[:, trix])
```
```python
if "tongue" not in feature_name:
    xv = xv - basederiv[0]
    yv = yv - basederiv[1]
    xv = nearest_fill_1d(xv)
    yv = nearest_fill_1d(yv)
```
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_count = np.sum(np.isfinite(paw_stack), axis=0)
paw_sum = np.nansum(paw_stack, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), where=np.maximum(paw_count, 1) > 0)
```

iii. The notes justify this as using bottom-view paw markers described in the paper while producing a single overall paw-movement signal.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI computes the per-session 50th percentile of the averaged paw-speed values and binarizes each time bin relative to that threshold. Any remaining `NaN` values are converted to `0` before thresholding.

ii.
```python
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
...
paw_thr = summarize_threshold(paw_sel)
...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. Step 5 says the per-session median split was chosen to follow the decoder instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The paw streams use the same video-offset correction and interpolation-to-`time_vec` procedure as tongue position, then paw speed is computed directly on that neural time axis.

ii.
```python
vidshift = find_video_offset(raw)
...
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes say the AI intentionally aligned video-derived variables by interpolating them to the same time base as neural activity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<subject>_<date>.mat` file for each selected session.

ii.
```python
def load_motion_energy(path: Path) -> tuple[list[np.ndarray], float]:
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    ...
    data = [np.asarray(unwrap_motion_energy_container(v), dtype=np.float64).reshape(-1) for v in flat]
    ...
    return data, thresh
```

iii. The notes explain that motion energy lives in separate files for ephys sessions and that the loader had to handle multiple MATLAB layouts.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI aligns each trial's motion-energy trace to `goCue` using side-camera frame times and the session video offset, interpolates it onto the neural `time_vec`, nearest-fills missing values, and then applies a per-session median split.

ii.
```python
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
out[:, trix] = nearest_fill_1d(out[:, trix])
```

iii. Step 10 says the AI considered this the same alignment logic as the reference motion-energy loader, with the decoder-specific change being the median split.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The AI computes a per-session 50th percentile threshold and binarizes each aligned time bin as below-vs-at/above median.

ii.
```python
motion = np.nan_to_num(motion, nan=0.0)
...
motion_thr = summarize_threshold(motion_sel)
...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. The notes repeatedly justify the 50th-percentile split as required by the decoder task, even though the paper used a manual threshold for movement analyses.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI uses side-camera frame times, subtracts a session video offset and per-trial `goCue`, interpolates the motion-energy trace onto the neural time vector, and then discretizes on that shared axis.

ii.
```python
side_trials = raw["traj"][0]
vidshift = find_video_offset(raw)
...
old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. The notes say motion energy was aligned with the same video-offset logic used for the other video-derived variables.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or malformed video data by interpolation/filling rather than by preserving missingness as a third category. If frame times are missing, it synthesizes them at 400 Hz; for non-tongue features it nearest-fills missing positions and velocities; for motion energy it nearest-fills aligned traces; and for tongue/motion/paw output labels it turns missing values into class `0`.

ii.
```python
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
```
```python
if "tongue" not in feature_name:
    xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
    ypos[:, trix] = nearest_fill_1d(ypos[:, trix])
```
```python
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
out[:, trix] = nearest_fill_1d(out[:, trix])
```
```python
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. The notes justify some of this as following `findPosition`/`findVelocity` nearest-filling logic, and Step 7 specifically says non-visible tongue bins were assigned to `0` to avoid degenerate verifier behavior.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies session loading, especially mixed-format MATLAB/HDF5 reading, as the main runtime bottleneck. The rest of the processing is described as lightweight by comparison.

ii.
```python
def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```
```python
log(f"SESSION {spec.session_id}: kept {session_out['n_trials']} trials, "
    f"{session_out['n_units']} units, format={raw['format']}, "
    f"time={time.time() - t0:.2f}s")
```

iii. Step 6 and Step 7 explicitly say full recursive HDF5 decoding was too slow and that targeted field loading was added because file I/O dominated runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several per-trial loops in place for video interpolation and output assembly, but it explicitly vectorizes spike accumulation with `np.add.at` instead of a more nested spike/bin loop. The remaining trial loops are mostly in `feature_xy`, `feature_speed`, `aligned_motion_energy`, and the final per-trial output assembly.

ii.
```python
for trix, trial in enumerate(trials):
    ...
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```
```python
for i in range(n_trials):
    tsinterp = np.column_stack([xpos[:, i], ypos[:, i]])
    ...
```
```python
np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
```

iii. Step 6 says the AI intentionally vectorized the neural spike accumulation path and avoided fully recursive HDF5 object loading because those were the main speed wins.

## 11-c. What processing does the code repeat multiple times?

i. The code recomputes the session video offset multiple times per session by calling `find_video_offset(raw)` separately inside `feature_xy` for tongue and each paw feature, and again inside `aligned_motion_energy`. It also recreates the same `time_vec` copy for every trial's input.

ii.
```python
def feature_xy(...):
    ...
    vidshift = find_video_offset(raw)
```
```python
def aligned_motion_energy(...):
    ...
    vidshift = find_video_offset(raw)
```
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes claim the code avoids redundant work overall, but the implementation still repeats video-offset estimation across multiple feature-processing functions.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads several fields that are not used later (`sample`, `delay`, `reward`, `L`, `no`, `motion_thresh`), and it computes/stores per-session continuous tongue/paw/motion arrays in `session_out["continuous"]` even though `build_dataset` discards them from the final saved dataset unless plotting is inspected during conversion.

ii.
```python
"events": {
    "bitStart": ...,
    "sample": ...,
    "delay": ...,
    "goCue": ...,
    "reward": ...,
},
"L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
"no": np.asarray(bp.no, dtype=np.float64).reshape(-1),
"motion_thresh": motion_thresh,
```
```python
"continuous": {
    "tongue": tongue_sel.astype(np.float32),
    "paw": paw_sel.astype(np.float32),
    "motion": motion_sel.astype(np.float32),
    "tongue_thr": tongue_thr,
    "paw_thr": paw_thr,
    "motion_thr": motion_thr,
},
```

iii. No explicit justification for these extra fields appears beyond debugging/plotting support in Step 6 and Step 7.
