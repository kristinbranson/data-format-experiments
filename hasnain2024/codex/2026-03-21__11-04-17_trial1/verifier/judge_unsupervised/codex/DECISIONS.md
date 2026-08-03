# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The script does not load every raw file indiscriminately. It first reconstructs the reference ephys session list by parsing the MATLAB loader scripts, matches those `(subject, date)` pairs against `data/*/data_structure_*.mat`, and then loads each matched session with either an HDF5 reader or a legacy MATLAB reader. For each retained session it also loads the paired `motionEnergy_<subject>_<date>.mat` file.

ii. ```python
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

def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. In `CONVERSION_NOTES.md`, the agent says it should use the “intersection of reference-code session lists and available data files,” which gave 44 ephys sessions, and that Python needed dual loading because the raw files mix MATLAB v7.3/HDF5 and older `.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the parsed session specs and then deduplicated while building the final dataset. Each session keeps its original subject string, and `subject_idx` points from each session to the corresponding entry in `subjects`.

ii. ```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))

data = {
    ...
    "subjects": subject_names,
    "subject_idx": np.asarray(subject_index, dtype=np.int64),
    ...
}
```

iii. The notes say the loader scripts define the analyzed sessions by animal/date, and that the converted dataset should preserve the 14 unique neural subjects present in the retained session list.

## 1-c. How are the data split into sessions?

i. Each `(subject, date)` pair parsed from a reference `load*_ALMVideo.m` script becomes one session. The converted dataset keeps one session entry per parsed spec, in the same order that the Python script iterates over `session_specs`.

ii. ```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probes: tuple[int, ...]
    session_path: Path
    folder: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"

for spec in session_specs:
    session = convert_one_session(spec, show_processing=show_processing, outdir=outdir)
    if session is None:
        continue
    neural.append(session["neural"])
```

iii. The agent justified this by saying the reference loader scripts, not the raw directory listing alone, define the analyzed session list, and that this is the cleanest way to match the paper/code selection.

## 1-d. How are the data split into trials?

i. Trials are indexed from the raw behavioral arrays (`R`, `L`, `hit`, `miss`, etc.). After forming a valid-trial mask, the script keeps the selected raw trial indices, maps raw trial numbers to positions within the kept subset, and uses each unit’s `trial` field to place spikes into the correct per-trial matrix.

ii. ```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)

trial_to_pos = np.full(raw["R"].size, -1, dtype=np.int64)
trial_to_pos[selected_trials] = np.arange(selected_trials.size, dtype=np.int64)

aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
trial_pos = trial_to_pos[unit["trial"] - 1]
```

iii. The notes repeatedly describe the target as “one `(n_neurons, n_timepoints)` matrix per trial” and say the converter should build that from the raw `obj.bp` trial structure and `obj.clu(...).trial` membership, matching the reference pipeline’s trial-centric organization.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are non-stimulation, non-early, hit-or-miss trials with a defined left/right instruction. The script also drops otherwise-valid behavioral trials whose raw trial index exceeds the last trial with neural coverage, and skips sessions with fewer than 2 kept trials.

ii. ```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )

if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. The notes justify this as excluding stimulation, early-lick, and ignore/no-response trials to stay close to the paper/code, while preserving misses because `outcome` must distinguish incorrect from correct. The notes also document a later fix for sessions with trailing behavioral trials but no neural coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the selected probe’s clustered spike data: unit `quality`, `trialtm`, and `trial`, together with `bp.ev.goCue` for alignment. In the HDF5 path these are read from `obj/clu` and `obj/bp/ev/goCue`; in the legacy MATLAB path they are read from `obj.clu` and `obj.bp.ev.goCue`.

ii. ```python
units.append(
    {
        "quality": h5_read_string(f, probe["quality"][i, 0]),
        "trialtm": np.asarray(h5_read_numeric(f, probe["trialtm"][i, 0]), dtype=np.float64).reshape(-1),
        "trial": np.asarray(h5_read_numeric(f, probe["trial"][i, 0]), dtype=np.int64).reshape(-1),
    }
)

"events": {
    "goCue": np.asarray(f["obj/bp/ev/goCue"][()], dtype=np.float64).reshape(-1),
},
```

iii. The notes say the dataset is electrophysiology, not imaging, and that the core raw neural inputs are the sorted spikes in `obj.clu` plus the behavioral event times needed for alignment.

## 2-b. How is the `neural` data processed?

i. For each kept unit, the script subtracts each spike’s trial-specific go-cue time, bins the aligned spikes into 5 ms bins across `[-2.5, 2.5] s`, converts counts to firing rate by dividing by `DT`, and applies a causal Gaussian smoother with reflected padding.

ii. ```python
def compute_unit_trial_matrix(unit: dict, go_cue: np.ndarray, trial_to_pos: np.ndarray, n_sel: int, time_edges: np.ndarray) -> np.ndarray:
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    ...
    if np.any(keep):
        bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
        np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat
```

iii. The notes explicitly say the converter reimplements the reference `alignSpikes` plus `getSeq` path in Python, including the “causal Gaussian kernel behavior in `mySmooth.m`.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script keeps only units on the probe(s) named in the reference loader scripts, rejects units whose quality labels are `garbage`, `gabrga`, `noisy`, or `real?`, then drops units whose mean firing rate across the kept trial matrices is `<= 1 Hz`. Entire sessions are skipped if fewer than 10 units remain.

ii. ```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

for unit in raw["units"]:
    if not good_quality(unit["quality"]):
        continue
    unit_mat = compute_unit_trial_matrix(...)
    if float(unit_mat.mean()) <= LOW_FR_HZ:
        continue
    ...

if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
    return None
```

iii. The notes say the agent resolved a code/paper mismatch in favor of “all non-garbage manually curated units on the selected probe” plus a 1 Hz FR threshold, because the methods text says units with firing rates exceeding 1 Hz were used for most analyses, and sessions should have at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial’s `bp.ev.goCue` time before binning. This makes time 0 correspond to go-cue onset for every trial.

ii. ```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
...
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The notes say this follows both the user instruction and the reference default `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 5 ms bins (`DT = 1/200` s). Spikes are binned directly onto this grid; there is no later rebinning step.

ii. ```python
DT = 1.0 / 200.0
...
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The notes say the agent chose the reference default analysis bin (`dt = 1/200`) rather than the separate 10 ms tutorial example.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The stored decoder input is not computed from a separate raw signal. It is the synthetic aligned time axis created from the chosen analysis window and bin size, with 0 implicitly defined by the same go-cue alignment used for the neural data.

ii. ```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes describe this as storing the reference `obj.time` axis itself as the decoder input `time_from_go_cue_s`.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The processing is just constructing the bin-center vector for the `[-2.5, 2.5] s` analysis window and repeating that same `(1, T)` array for every kept trial in the session.

ii. ```python
time_vec = time_edges[:-1] + DT / 2.0
...
for local_idx, trial_idx in enumerate(selected_trials):
    ...
    input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes justify this as using the exact aligned neural time base so the decoder input and neural data live on the same clock.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `time_vec` used for neural spike binning is copied into each trial’s decoder input, so the input is exactly aligned to the neural bin centers.

ii. ```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
unit_mat = compute_unit_trial_matrix(..., time_edges)
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes say one planned sanity check was to verify that all trials share the same time vector and that `input[0]` matches the neural bin centers exactly.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `lick_direction` is derived from the raw per-trial behavioral flags `R` and `L`.

ii. ```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. The notes map this output directly to `bp.R` and `bp.L`, with left = 0 and right = 1.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After trial filtering, the script encodes each kept trial as right = 1 and left = 0, then repeats that scalar as a constant time series across the full trial window.

ii. ```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
...
np.full(time_vec.size, lick_direction, dtype=np.int64),
```

iii. The notes say trial-level categorical outputs were deliberately represented as constant time series so that every output shares the same `(n_output, n_timepoints)` layout.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `behavioral_context` is derived from the raw per-trial `autowater` flag.

ii. ```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. The notes say the mapping is `WC = 0, DR = 1` via `1 - autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script interprets `autowater == 1` as the water-cued context and `autowater == 0` as the delayed-response context, then stores the resulting label as a constant trial-long time series.

ii. ```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
...
np.full(time_vec.size, context, dtype=np.int64),
```

iii. The notes explicitly justify this with “Stored `autowater=1` corresponds to WC in the reference code/paper.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `outcome` is derived from the raw per-trial `hit` and `miss` flags.

ii. ```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. The notes map this output to `bp.hit` and `bp.miss`, with miss = 0 and hit = 1.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Only hit/miss trials are retained, then each kept trial is encoded as correct = 1 if `hit == 1`, otherwise incorrect = 0, and repeated across the trial window as a constant time series.

ii. ```python
valid = session_valid_trial_mask(raw)
...
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
...
np.full(time_vec.size, outcome, dtype=np.int64),
```

iii. The notes say ignores/no-response trials were excluded rather than inventing a third outcome label, because the decoder task requested a binary incorrect/correct output.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The tongue-velocity output is derived from side-view DeepLabCut tongue coordinates (`traj` feature `tongue`), plus each trial’s frame times, the video/neural offset, and the go-cue times used to align the trajectory to the neural clock.

ii. ```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
...
old_t = frame_times - vidshift - align_times[trix]
ts = np.asarray(trial["ts"][:, :2, feat_idx], dtype=np.float64)
```

iii. The notes justify this as using “the canonical `tongue` marker from the side view to avoid mixing camera coordinate systems.”

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script interpolates side-view tongue x/y position onto the neural time grid, does not nearest-fill missing tongue positions, computes x/y velocity by `np.gradient`, converts tongue NaNs to zero inside `feature_speed`, takes speed magnitude, then re-masks time points where tongue position was invalid before thresholding.

ii. ```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
tongue_valid = np.isfinite(tongue_pos[0]) & np.isfinite(tongue_pos[1])
tongue_speed = tongue_speed.astype(np.float64, copy=False)
tongue_speed[~tongue_valid] = np.nan
```

iii. The notes say this was changed after a sample run exposed a degenerate all-ones tongue bin; the documented fix was to compute the tongue percentile only on visible frames and assign non-visible frames to bin 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script computes a per-session 50th-percentile threshold from the selected trials’ finite tongue-speed samples only, then binarizes each time point as 1 if `>= threshold` and 0 otherwise; non-visible tongue frames are forced to 0.

ii. ```python
tongue_sel = tongue_speed[:, selected_trials]
tongue_thr = summarize_threshold(tongue_sel)
...
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. The notes explicitly document this decision as a post-debugging change: “computing the tongue percentile on visible frames only and assigning non-visible frames to bin `0`.”

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue trajectories are aligned by subtracting both the session video offset and the trial’s go-cue time from the video frame times, then linearly interpolating onto the same `time_vec` used for the neural bins.

ii. ```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes say the converter follows the same “video offset correction + interpolation to the neural time base” logic as the reference `findPosition` function.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The paw-velocity output is derived from bottom-view DeepLabCut coordinates for `top_paw` and `bottom_paw`, together with each trial’s frame times, the video/neural offset, and go-cue alignment.

ii. ```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. The notes justify this by saying paws are tracked only in the bottom view in the paper/methods, so the decoder’s scalar paw variable should come from those bottom-view paw features.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each of `top_paw` and `bottom_paw`, the script interpolates x/y position onto the neural time base, nearest-fills missing values, computes x/y velocities with baseline subtraction, converts each paw to speed magnitude, then averages the available top- and bottom-paw speeds at each time point.

ii. ```python
paw_speeds = []
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
paw_stack = np.stack(paw_speeds, axis=0)
paw_count = np.sum(np.isfinite(paw_stack), axis=0)
paw_sum = np.nansum(paw_stack, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), where=np.maximum(paw_count, 1) > 0)
```

iii. The notes explicitly say the decision was to “compute top- and bottom-paw speed magnitudes from x/y velocity pairs, [and] average them per timepoint.”

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The script computes a per-session median threshold from all kept paw-speed samples and sets the bin to 1 when the time point is at or above that threshold.

ii. ```python
paw_sel = paw_speed[:, selected_trials]
paw_thr = summarize_threshold(paw_sel)
...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. The notes say this follows the decoder-task instruction to use a per-session 50th-percentile split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned exactly like the other video features: frame times are shifted by the video offset and the trial’s go cue, then interpolated onto the neural time axis before velocity is computed.

ii. ```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes say paw velocity should use the same reference-style interpolation/video-offset logic as `findPosition` and `findVelocity`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motion_energy` is derived from the separate per-session `motionEnergy_<subject>_<date>.mat` files, specifically `me.data`, together with side-camera frame times, the video/neural offset, and the go-cue event times.

ii. ```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
...
me_trials = raw["motion_energy"]
side_trials = raw["traj"][0]
vidshift = find_video_offset(raw)
```

iii. The notes describe this as matching the reference `loadMotionEnergy` path, except that the final categorical output uses the decoder-task median split rather than the paper’s manual movement threshold.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script loads the per-trial motion-energy traces, aligns them to the go cue with the same video-offset correction used for the kinematics, interpolates them onto the neural time grid, and nearest-fills missing values.

ii. ```python
for trix, me in enumerate(me_trials):
    ...
    if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
        old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
    else:
        old_t = frame_times - vidshift - align_times[trix]
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
    out[:, trix] = nearest_fill_1d(out[:, trix])
```

iii. The notes say this mirrors the reference interpolation/fill behavior in `loadMotionEnergy`, including the fallback when frame times are missing.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. After alignment, the script computes a per-session median threshold across the kept trials’ motion-energy samples and binarizes each time point relative to that threshold.

ii. ```python
motion_sel = motion[:, selected_trials]
motion_thr = summarize_threshold(motion_sel)
...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. The notes say this is a deliberate override of the paper’s manual `moveThresh`, because the user instructions explicitly required a 50th-percentile discretization for decoder output.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and the trial’s go-cue time from the motion-energy time base, then interpolating onto the same `time_vec` used for neural binning.

ii. ```python
old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. The notes repeatedly say that motion energy and kinematics are not kept at native 400 Hz; they are aligned and interpolated onto the shared neural time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script contains several repair/fallback paths: dual MATLAB readers for mixed file formats; recursive unwrapping for nested `motionEnergy` structs; fallback frame times when `frameTimes` are absent; default video offset `0.5` s if bitcode timing is unavailable; nearest-value filling for most missing kinematic/motion traces; and zero-filling if an entire vector is non-finite. It also drops trials whose behavioral metadata extend past the available neural coverage.

ii. ```python
def nearest_fill_1d(x: np.ndarray) -> np.ndarray:
    ...
    if not mask.any():
        return np.zeros_like(x)
    ...

if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

def find_video_offset(raw: dict) -> float:
    ...
    if bitstart.size == 0 or not np.isfinite(fs) or fs <= 0:
        return 0.5
```

iii. The notes explicitly document fixes for nested `motionEnergy` structs, empty HDF5 probe slots, trailing behavioral trials with no neural coverage, and the tongue-tracking edge case.

## 11-a. What are the most time-consuming steps of the code?

i. The expensive parts are session loading, per-unit spike binning/smoothing across many trials, and per-trial video/motion interpolation plus velocity computation. The code structure makes that visible: every session loops over all retained units and all selected video trials.

ii. ```python
for unit in raw["units"]:
    ...
    unit_mat = compute_unit_trial_matrix(...)
    ...

for trix, trial in enumerate(trials):
    ...
    xpos[:, trix] = interp_to_taxis(...)
    ypos[:, trix] = interp_to_taxis(...)
```

iii. In Step 6 the notes explicitly call out full recursive HDF5 decoding and naïve per-spike/per-trial histogram loops as the main expected bottlenecks, then list targeted HDF5 loading and `np.add.at` as the key speedups.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious remaining scalar loops are the per-trial loops in `feature_xy`, `feature_speed`, and `aligned_motion_energy`, plus the top-level loop over units in `convert_one_session`. These are only partly vectorized now; the spike accumulation inside `compute_unit_trial_matrix` is vectorized, but the surrounding iteration still scales linearly with units/features/trials.

ii. ```python
for trix, trial in enumerate(trials):
    ...

for i in range(n_trials):
    ...
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])

for unit in raw["units"]:
    ...
```

iii. The notes explicitly say “per-spike/per-trial histogram loops would likely be a bottleneck if implemented naively,” which shows the agent was thinking in these vectorization terms.

## 11-c. What processing does the code repeat multiple times?

i. The same alignment/interpolation logic is repeated for tongue, top paw, bottom paw, and motion energy. The same fixed time vector is also copied into every trial, and each unit is smoothed independently after separate binning even though the kernel is unchanged.

ii. ```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
...
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
...
motion = aligned_motion_energy(raw, raw["events"]["goCue"], time_vec)
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes describe the converter as applying the same reference-style alignment machinery to multiple data streams; that necessarily repeats similar interpolation and thresholding work per feature.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The largest discarded intermediate is the `continuous` block stored in each `session_out`: continuous tongue, paw, and motion traces plus thresholds are computed and kept temporarily, but `build_dataset()` never writes them into the final pickle. They are only used for optional plotting and debugging. The script also keeps session-local `format` and `source_file` fields that are collapsed into metadata rather than used directly by the decoder.

ii. ```python
session_out = {
    ...
    "continuous": {
        "tongue": tongue_sel.astype(np.float32),
        "paw": paw_sel.astype(np.float32),
        "motion": motion_sel.astype(np.float32),
        "tongue_thr": tongue_thr,
        "paw_thr": paw_thr,
        "motion_thr": motion_thr,
    },
}

data = {
    "neural": neural,
    "input": inputs,
    "output": outputs,
    ...
}
```

iii. The notes say the processing plots and sanity checks were part of validation/debugging, which explains why these continuous intermediates exist even though the final decoder dataset only keeps the discretized outputs.
