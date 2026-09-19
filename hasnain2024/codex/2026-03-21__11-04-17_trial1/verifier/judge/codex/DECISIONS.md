# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads session files by first globbing `data_structure_*.mat` under `data/*/`, then parsing the authors' `load*_ALMVideo.m` scripts to recover the reference session list and probe selections. Each retained session is loaded with either an HDF5 reader or a `scipy.io.loadmat` reader, and the paired `motionEnergy_<subject>_<date>.mat` file is loaded separately.

ii.
```python
def find_data_files(data_dir: Path) -> dict[tuple[str, str], Path]:
    out = {}
    for path in sorted(data_dir.glob("*/*.mat")):
        if not path.name.startswith("data_structure_"):
            continue
        parts = path.stem.split("_")
        out[(parts[2], parts[3])] = path
    return out

def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        ...
                if key in data_files:
                    path = data_files[key]
                    specs.append(SessionSpec(...))
```
```python
def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. The notes say the converter should use the intersection of the reference loader scripts and the available data files so that it matches the paper's analyzed session list instead of the broader raw archive, while still supporting both MATLAB file formats and standalone motion-energy files.

## 1-b. How are the data split into subjects?

i. Each session carries a `subject` from the parsed loader script and filename, and the final dataset builds `subjects` in first-seen session order with `subject_idx` pointing from each session to that list.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    ...

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"
```
```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
...
"subjects": subject_names,
"subject_idx": np.asarray(subject_index, dtype=np.int64),
```

iii. The notes justify using the loader/file naming because probe metadata are inconsistent across sessions, while the loader scripts and filenames reliably encode the animal identity.

## 1-c. How are the data split into sessions?

i. One parsed `SessionSpec` becomes one session. The script treats each `data_structure_<subject>_<date>.mat` file chosen by the loader scripts as a session, processes it once, and appends one entry to `neural`, `input`, and `output`.

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

iii. Step 4 and Step 5 of the notes explicitly say to use the 44-session intersection of the reference ALM loader scripts and the available data files, treating fixed-delay and randomized-delay sessions uniformly.

## 1-d. How are the data split into trials?

i. Trials are split by indexing parallel per-trial behavioral arrays and by using each unit's 1-based `trial` field. A trial is any index passing the session validity mask; spike counts are assigned to that trial through a `trial_to_pos` lookup built from the selected trial indices.

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
selected_trials = np.flatnonzero(valid)
...
trial_to_pos = np.full(raw["R"].size, -1, dtype=np.int64)
trial_to_pos[selected_trials] = np.arange(selected_trials.size, dtype=np.int64)
```
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
trial_pos = trial_to_pos[unit["trial"] - 1]
```

iii. The notes do not give a separate theory of trial boundaries beyond using the Bpod trial arrays and unit trial indices. The implementation follows that assumption directly instead of using `bp.Ntrials` truncation helpers.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are non-stimulation, non-early, hit-or-miss trials with a defined left/right instruction. After that, the code additionally drops trials whose index exceeds the last neural `unit.trial` seen in the kept-quality units. Sessions with fewer than two retained trials are skipped entirely.

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

if selected_trials.size < 2:
    ...
```

iii. The notes justify excluding stimulation, early-lick, and ignore/no-response trials as the cleanest way to match the reference analyses, and later justify the extra neural-coverage cutoff as a fix for late behavioral trials that had no spikes after the recording stopped.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu` entries on the selected probe(s), specifically each unit's `trialtm`, `trial`, and `quality`, together with `bp.ev.goCue` for alignment.

ii.
```python
units.append(
    {
        "quality": ...,
        "trialtm": ...,
        "trial": ...,
    }
)
```
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. Step 5 in the notes says neural output should come from the code-selected ALM probe plus its spike times, filtered by cluster quality, aligned to `bp.ev.goCue`, and then binned/smoothed.

## 2-b. How is the `neural` data processed?

i. For each retained unit, spike times are aligned to go cue, placed into a fixed 5 ms grid over `[-2.5, 2.5]` s, converted to rates by adding `1/DT`, and smoothed with a causal Gaussian-like kernel using reflected padding. No z-scoring or baseline subtraction is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
SMOOTH = 15
```
```python
if np.any(keep):
    bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
    np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
```

iii. The notes say the script should reimplement the reference spike binning and `mySmooth.m` behavior in Python, including the causal kernel and reflected padding, on the default `goCue`-aligned 5 ms time base.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps units unless their quality string is `garbage`, `gabrga`, `noisy`, or `real?`, and then drops units whose mean rate over the processed window is `<= 1 Hz`. It also skips any entire session with fewer than 10 surviving units.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}
```
```python
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
...
if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units after quality/FR filtering")
    return None
```

iii. The notes justify the `> 1 Hz` cutoff from the paper's methods and justify using all non-garbage manually curated units on the selected probe. They also cite the paper's session inclusion rule of at least 10 units, which is why sessions below that threshold are skipped.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting its own trial's `goCue` time from `trialtm`, so the neural data are expressed directly in seconds from go-cue onset.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The notes explicitly choose `goCue` as the universal alignment event because it matches both the decoder instructions and the default reference pipeline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use a 5 ms grid (`DT = 1/200`) from `-2.5` to `2.5` s. Spikes are binned directly onto that grid; there is no later temporal rebinning.

ii.
```python
DT = 1.0 / 200.0
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The notes justify this as the default reference pipeline setting rather than the 10 ms tutorial example.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read as a raw variable. It is constructed from the fixed `goCue`-centered analysis window and bin size, so it is effectively derived from the chosen alignment event and the synthetic bin centers.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes describe `obj.time` or the aligned bin centers as the intended decoder input because the user requested time from the go cue rather than an original behavioral covariate.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The only processing is building a fixed 1000-bin time vector from the chosen window and repeating that same vector for every retained trial.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes treat this as a direct representation of the reference trial time base, with no extra transformations beyond defining the common time axis.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the exact same `time_vec` as the neural bins, so each input column corresponds to the same 5 ms interval used when counting spikes.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
...
unit_mat = compute_unit_trial_matrix(unit, raw["events"]["goCue"], trial_to_pos, selected_trials.size, time_edges)
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes explicitly say the decoder input should be the same aligned time base used by the neural representation.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. In the implemented code, lick direction is derived only from the instructed side fields `R` and `L`. The code does not use `hit` or `miss` when assigning left versus right.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. The notes justify this by saying `bp.R` and `bp.L` define lick direction once trials are restricted to non-early hit/miss trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The code creates a constant time series per trial with two classes only: `1` for right-instructed trials and `0` otherwise. It does not reverse miss trials to the opposite lick direction and does not create a no-lick class.

ii.
```python
output_arr = np.vstack(
    [
        np.full(time_vec.size, lick_direction, dtype=np.int64),
        ...
    ]
)
```
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. The Step 5 mapping in the notes says to use `bp.R`/`bp.L` and encode the trial label as a constant time series over the whole window.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag.

ii.
```python
"autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1),
...
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. The notes explicitly map `bp.autowater` to context and interpret autowater trials as WC and the rest as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code relabels each retained trial as WC (`0`) if `autowater==1`, else DR (`1`), and repeats that category across all time bins in the trial.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
...
np.full(time_vec.size, context, dtype=np.int64),
```
```python
"output_values": [
    ["left", "right"],
    ["WC", "DR"],
    ...
]
```

iii. The notes say the prompt requires WC `0` and DR `1`, and they chose constant per-trial time series so all outputs share one shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The implemented output is derived effectively from `hit` alone after the trial filter has already removed ignore trials and retained only hit/miss trials.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. The notes say outcome comes from `bp.hit` and `bp.miss`, but also say ignore/no-response trials should be excluded instead of represented as a third class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code encodes outcome as a constant binary time series: `1` for hits and `0` otherwise. Because the earlier trial filter excludes ignore trials, there is no ignore category in the saved output.

ii.
```python
np.full(time_vec.size, outcome, dtype=np.int64),
```
```python
"output_values": [
    ["left", "right"],
    ["WC", "DR"],
    ["incorrect", "correct"],
    ...
]
```

iii. The notes justify this as avoiding an invented ignore label and keeping only non-early hit/miss trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The implemented tongue-velocity signal uses only the side-camera `tongue` feature from `obj.traj[0]`, together with frame times, bitcode-based video offset, and `goCue` alignment.

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
```
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. The notes explicitly say they chose the side-view `tongue` marker to avoid mixing camera coordinate systems.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates side-view tongue x/y positions onto the neural time base, computes velocity with `np.gradient` on that interpolated grid, converts tongue NaNs to zeros inside `feature_speed`, then restores invalid bins to `NaN` using a finite-position mask. No per-run smoothing or two-camera normalization/averaging is applied.

ii.
```python
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```
```python
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
...
xv = np.nan_to_num(xv, nan=0.0)
yv = np.nan_to_num(yv, nan=0.0)
```
```python
tongue_valid = np.isfinite(tongue_pos[0]) & np.isfinite(tongue_pos[1])
tongue_speed[~tongue_valid] = np.nan
```

iii. The notes say tongue should come from a direct reference-style kinematic channel, then be median-split for the decoder. They later note a debugging change: invalid tongue periods were excluded from percentile estimation and then assigned to class `0`.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session median is computed from finite tongue values in the selected trials, and each finite bin is labeled `1` if it is at or above the threshold and `0` otherwise. Non-finite bins are also written as `0`; there is no separate `not visible` class.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)
...
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```
```python
"output_values": [
    ...,
    ["below_session_median", "at_or_above_session_median"],
    ...
]
```

iii. The notes explicitly justify a session-median split from the decoder task and state that non-visible tongue bins were assigned to `0` after an earlier verifier issue.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code estimates one session-wide video offset from bitcode timing, subtracts that offset and the trial's `goCue` from each frame time, and interpolates x/y positions onto the same `time_vec` used by the neural bins.

ii.
```python
def find_video_offset(raw: dict) -> float:
    ...
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])
```
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes say video outputs should follow the reference video-offset logic and be aligned to the neural time base on the same go-cue-centered window.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-view `top_paw` and `bottom_paw` tracked features in `obj.traj[1]`, not from just one paw.

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
```
```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. The notes explicitly justify averaging top- and bottom-paw speed magnitudes to represent overall paw movement from the bottom view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Each paw's x/y position is interpolated to the neural time base, velocity is estimated by `np.gradient`, a baseline derivative is subtracted, NaNs are nearest-filled, and the two paw speed magnitudes are averaged pointwise. No explicit per-run smoothing is applied.

ii.
```python
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
...
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

iii. The notes justify this as using the paper's bottom-view paw tracking while capturing overall paw movement by averaging the two tracked paws.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. A per-session median of the selected paw-speed values is computed, and bins are encoded as binary `0/1` depending on whether they fall below or at/above that threshold. Missing bins are converted to numeric values before thresholding and therefore do not receive a dedicated `not visible` label.

ii.
```python
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
...
paw_thr = summarize_threshold(paw_sel)
...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. The notes justify the median split from the decoder task and do not preserve a third visibility class for paw velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Bottom-view frame times are corrected by the session video offset and by each trial's `goCue`, then paw positions are interpolated onto the shared neural `time_vec`.

ii.
```python
paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes state that video variables should be put on the same aligned neural time base using the reference-style bitcode offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<subject>_<date>.mat` file associated with each session.

ii.
```python
def load_motion_energy(path: Path) -> tuple[list[np.ndarray], float]:
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    ...
    data = [np.asarray(unwrap_motion_energy_container(v), dtype=np.float64).reshape(-1) for v in flat]
```
```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. The notes justify this because the standalone motion-energy files are available for all retained ephys sessions and occur in several structural variants that need explicit unwrapping.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code interpolates each per-frame motion-energy trace onto the neural time base using side-camera frame times and video-offset correction, then nearest-fills missing values and later median-thresholds the selected-trial values.

ii.
```python
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
out[:, trix] = nearest_fill_1d(out[:, trix])
```

iii. The notes describe this generally as "reference-style interpolation to neural time base with video offset correction," even though motion energy is already a frame-level scalar.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. A per-session median is computed over the selected-trial motion-energy values, and bins are labeled `1` for values at or above the threshold and `0` otherwise. Missing values are converted to `0` before thresholding, so there is no separate `no video` class.

ii.
```python
motion = aligned_motion_energy(raw, raw["events"]["goCue"], time_vec)
motion = np.nan_to_num(motion, nan=0.0)
...
motion_thr = summarize_threshold(motion_sel)
...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```
```python
"output_values": [
    ...,
    ["below_session_median", "at_or_above_session_median"],
]
```

iii. The notes justify only the median split required by the decoder task; they do not preserve a third missing-video category.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting a session-wide video offset and the trial's `goCue` from side-camera frame times, then interpolating onto the same `time_vec` as the neural data.

ii.
```python
vidshift = find_video_offset(raw)
...
old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. The notes explicitly state that motion energy should use the same video-offset logic and neural time base as the other video-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handles missing or malformed video data by substituting defaults rather than leaving a separate missing-data class. Missing frame times are replaced with synthetic `1/400` s frame times; non-tongue positions are nearest-filled after interpolation; motion energy NaNs are nearest-filled and then zero-filled; paw NaNs are zero-filled; and tongue invalid bins are finally mapped to output class `0`. For neural coverage issues, late behavioral trials beyond the last spike trial are dropped.

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
...
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
motion = np.nan_to_num(motion, nan=0.0)
```

iii. The notes justify these fixes pragmatically: they mention handling several raw-file variants, repairing sessions with trailing behavioral-only trials, and intentionally assigning non-visible tongue bins to class `0` after a validation issue.

## 11-a. What are the most time-consuming steps of the code?

i. The notes say the expensive parts were HDF5 loading and, in a naive version, per-spike/per-trial accumulation. The final code tries to reduce both by targeted field loading and vectorized `np.add.at`, but session loading still dominates the main conversion path.

ii.
```python
def load_session_hdf5(spec: SessionSpec) -> dict:
    with h5py.File(spec.session_path, "r") as f:
        ...
```
```python
np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
```

iii. Step 6 and Step 7 of the notes explicitly identify full recursive HDF5 decoding as too slow and describe targeted field loading and vectorized spike accumulation as the main speedups.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining scalar loops are over units, trials, and features: loading unit structs, interpolating positions one trial at a time, computing speeds one trial at a time, and appending one unit row into each trial's neural list inside nested loops. Spike binning itself was already vectorized within a unit.

ii.
```python
for trix, trial in enumerate(trials):
    ...
    xpos[:, trix] = interp_to_taxis(...)
```
```python
for i in range(n_trials):
    ...
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
```
```python
for unit in raw["units"]:
    ...
    for trix in range(selected_trials.size):
        neural_trials[trix].append(unit_mat[trix])
```

iii. The notes explicitly contrast the implemented vectorized spike accumulation with the slower nested histogram approach they considered earlier, but they leave the trialwise video loops in place.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats some session-wide computations. `find_video_offset(raw)` is called separately by each `feature_xy` invocation and again by `aligned_motion_energy`; `time_edges` and `time_vec` are rebuilt in every session; and each unit matrix is iterated trial-by-trial again just to append rows into per-trial lists.

ii.
```python
def feature_xy(...):
    vidshift = find_video_offset(raw)
    ...
```
```python
def aligned_motion_energy(...):
    vidshift = find_video_offset(raw)
    ...
```
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The notes emphasize avoiding repeated full-session decoding, but they do not call out these smaller repeated computations; they seem to have accepted them as a practical tradeoff.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script does some extra work that does not end up in the saved dataset. It loads several raw fields that are never used downstream (`L`, `no`, `sample`, `delay`, `reward`, `motion_thresh`, `folder`), builds a `continuous` block only for plotting/threshold diagnostics and then drops it when assembling the final dataset, and optionally renders figures that are not part of `converted_data.pkl`.

ii.
```python
"L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
"no": np.asarray(bp.no, dtype=np.float64).reshape(-1),
"events": {
    "bitStart": ...,
    "sample": ...,
    "delay": ...,
    "goCue": ...,
    "reward": ...,
},
...
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
```python
data = {
    "neural": neural,
    "input": inputs,
    "output": outputs,
    ...
}
```

iii. The notes justify some of this as support for processing plots, threshold inspection, and debugging raw-file variants, not because the extra data are needed in the final pickle.
