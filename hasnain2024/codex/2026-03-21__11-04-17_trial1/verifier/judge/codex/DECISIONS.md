# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent does not scan every raw file indiscriminately. It first enumerates `data_structure_*.mat` files, then parses the reference `load*_ALMVideo.m` scripts to recover the session list and probe choices that the paper/code used, and only those sessions are processed. Each selected session is loaded through either an HDF5 reader or an old-MATLAB fallback loader.

ii. ```python
def find_data_files(data_dir: Path) -> dict[tuple[str, str], Path]:
    for path in sorted(data_dir.glob("*/*.mat")):
        if path.name.startswith("data_structure_"):
            out[(parts[2], parts[3])] = path

def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        ...
        specs.append(SessionSpec(..., session_path=path, ...))

def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. In Step 4/5 of `CONVERSION_NOTES.md`, the agent says the raw archive is broader than the analyzed dataset and resolves this by taking the intersection of the loader-script session list with available data, yielding 44 ALM ephys sessions.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `subject` field parsed from the reference loader scripts and filenames. The final dataset stores a unique `subjects` list and a per-session `subject_idx`.

ii. ```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str

if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The notes explicitly say animal IDs come from loader scripts / filenames and should become `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` corresponds to one `(subject, date)` session. The converter iterates session-by-session, converts one session at a time, and appends the results to the top-level lists.

ii. ```python
for spec in session_specs:
    session = convert_one_session(spec, ...)
    if session is None:
        continue
    neural.append(session["neural"])
    inputs.append(session["input"])
    outputs.append(session["output"])
```

iii. The notes describe session selection as the 44 reference-selected ALM sessions and emphasize per-session processing for bounded memory use.

## 1-d. How are the data split into trials?

i. Trials are defined from behavioral trial indices. A session-level valid-trial mask is built from behavioral fields, converted to `selected_trials`, and every kept trial becomes one `(neurons, time)` neural matrix, one input array, and one output array.

ii. ```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
...
trial_to_pos = np.full(raw["R"].size, -1, dtype=np.int64)
trial_to_pos[selected_trials] = np.arange(selected_trials.size, dtype=np.int64)
...
for local_idx, trial_idx in enumerate(selected_trials):
    neural_arr = np.stack(neural_trials[local_idx], axis=0)
    final_neural.append(neural_arr)
```

iii. Step 5 says the agent would keep non-stim, non-early hit/miss trials and represent each retained trial separately in the decoder format.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding stimulation trials, early-lick trials, ignore/no-response trials, and trials without a left/right label. After that, the code also drops behavioral trials that extend past the last neural trial index and skips sessions with fewer than two valid trials.

ii. ```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )

selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
if selected_trials.size < 2:
    return None
```

iii. The notes repeatedly justify this as matching the paper/code choice to omit stim, early, and ignore trials, plus an extra edge-case fix for sessions with behavioral trials lacking neural coverage.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj.clu` units on the selected ALM probe(s), using each unit’s `quality`, `trialtm`, and `trial` fields, together with behavioral `bp.ev.goCue` times.

ii. ```python
units.append(
    {
        "quality": ...,
        "trialtm": np.asarray(...),
        "trial": np.asarray(...),
    }
)
...
unit_mat = compute_unit_trial_matrix(unit, raw["events"]["goCue"], ...)
```

iii. Step 1 and Step 5 in the notes identify `obj.clu{probe}` spike fields as the reference neural source and map them directly to the converted `neural` output.

## 2-b. How is the `neural` data processed?

i. For each unit, the code aligns spike times to go cue, bins spikes into a fixed `[-2.5, 2.5]` window with 5 ms bins, converts counts to rate by dividing by `DT`, and smooths the binned trace with a reflected, causal Gaussian-like kernel meant to mimic MATLAB `mySmooth`.

ii. ```python
DT = 1.0 / 200.0
SMOOTH = 15
BCTYPE = "reflect"

aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
```

iii. The notes say the agent was intentionally reimplementing `alignSpikes`, `getSeq`, and `mySmooth.m` in Python and preferred the default 5 ms pipeline over the 10 ms tutorial example.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code keeps only units whose `quality` label is not one of `garbage`, `gabrga`, `noisy`, or `real?`, then removes units with mean firing rate `<= 1 Hz`, and finally skips any session with fewer than 10 remaining units.

ii. ```python
def good_quality(label: str) -> bool:
    return label not in {"garbage", "gabrga", "noisy", "real?"}

if not good_quality(unit["quality"]):
    continue
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
...
if kept_units < 10:
    return None
```

iii. Step 4/5 of the notes justify this as “all non-garbage manually curated units” plus a 1 Hz FR cutoff, with the 10-unit session threshold coming from the methods summary.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The agent aligns spikes to go cue by subtracting the per-trial `bp.ev.goCue` time from each spike’s within-trial time before binning.

ii. ```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. The notes say go cue is the default alignment in the reference code and also the explicit alignment event requested by the task.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins (`DT = 1/200 s`) on a single shared trial time axis. No later temporal rebinning is applied.

ii. ```python
DT = 1.0 / 200.0
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The notes explicitly resolve a 5 ms vs 10 ms ambiguity in favor of the default reference pipeline (`getDefaultParams.m`).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read as a stored raw variable. It is synthesized from the fixed analysis window (`TMIN`, `TMAX`, `DT`) and the decision to align all streams to `bp.ev.goCue`.

ii. ```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
time_vec = time_edges[:-1] + DT / 2.0
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The notes map this to the reference `obj.time` / aligned bin centers rather than to a raw stored field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code builds a uniform vector of bin centers over `[-2.5, 2.5]` seconds and repeats that same `1 x n_timepoints` array for every kept trial in the session.

ii. ```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. In the notes, the agent says this matches the reference trial time base `obj.time` after go-cue alignment.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input and neural data share the exact same `time_edges` / `time_vec`. Neural traces are binned on those edges, and the input stores the corresponding bin centers.

ii. ```python
unit_mat = compute_unit_trial_matrix(..., time_edges)
time_vec = time_edges[:-1] + DT / 2.0
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. Step 5 and Step 10 both say the time vector was meant to be the same aligned axis used for neural binning.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw behavioral flags `bp.R` and `bp.L`.

ii. ```python
"R": np.asarray(bp.R, dtype=np.float64).reshape(-1),
"L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
...
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. The notes map `bp.R` / `bp.L` directly to left/right choice labels.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After trial filtering guarantees a valid choice label, the code encodes right as `1`, left as `0`, and expands that per-trial label into a constant time series over the full aligned window.

ii. ```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
np.full(time_vec.size, lick_direction, dtype=np.int64)
```

iii. The notes justify constant time-series encoding as a way to keep all outputs on the same `(n_output, n_timepoints)` grid.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii. ```python
"autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1),
...
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. The notes say the reference code/paper treat `autowater=1` as water-cue (`WC`) and therefore `DR` is `1 - autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater == 1` to `WC = 0` and `autowater == 0` to `DR = 1`, then repeats that categorical label across all time bins of the trial.

ii. ```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
np.full(time_vec.size, context, dtype=np.int64)
```

iii. Step 5 explicitly records this mapping and explains it as the paper-consistent interpretation of `autowater`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` and `bp.miss`.

ii. ```python
"hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
"miss": np.asarray(bp.miss, dtype=np.float64).reshape(-1),
...
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. The notes say ignore trials are excluded rather than given a separate outcome, leaving miss=`0` and hit=`1`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code converts hit to `1`, miss to `0`, and stores the per-trial outcome as a constant time series.

ii. ```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
np.full(time_vec.size, outcome, dtype=np.int64)
```

iii. The notes justify this as the simplest decoder-compatible representation once ignore trials have already been removed.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view DLC `tongue` feature in `obj.traj`, using the first two position coordinates and each trial’s `frameTimes`, with alignment corrected by the video offset and go-cue times.

ii. ```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
```

iii. Step 5 says the agent intentionally chose the canonical side-view `tongue` marker rather than combining multiple tongue markers or views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates tongue x/y position onto the neural time base, leaves tongue gaps unfilled, computes x/y derivatives with `np.gradient`, converts tongue NaNs to zero velocity, takes speed magnitude, then later restricts threshold computation to visible frames by setting invalid positions back to `NaN`.

ii. ```python
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
...
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
...
xv = np.nan_to_num(xv, nan=0.0)
yv = np.nan_to_num(yv, nan=0.0)
return np.sqrt(xvel**2 + yvel**2)
```

iii. The notes cite `findPosition` / `findVelocity` as the intended model, and the trajectory records a later fix: threshold the tongue only over visible frames because converting missing periods to zero made the bin degenerate.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes a per-session 50th percentile over all finite tongue-speed samples from kept trials, then binarizes each timepoint as below-median `0` or at/above-median `1`. Non-visible frames are forced to `0`.

ii. ```python
tongue_thr = summarize_threshold(tongue_sel)
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64)
```

iii. The notes say this follows the decoder-task specification, not the paper’s manual movement threshold, and document the sample-run bug fix that changed how tongue invalid frames were handled.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position is aligned by shifting each trial’s video timestamps by the video offset and subtracting the trial’s `goCue`, then interpolating onto the neural `time_vec`.

ii. ```python
vidshift = find_video_offset(raw)
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
```

iii. The notes explicitly say video streams should be corrected with the reference bitcode offset and resampled to the same time axis as the neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-view DLC features `top_paw` and `bottom_paw` in `obj.traj`, using their x/y coordinates and video timing.

ii. ```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```

iii. Step 5 says paws are tracked in the bottom view only, so the agent averaged the two paw markers to form one decoder output.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw marker, the code interpolates positions onto the neural time base, fills missing positions by nearest-neighbor interpolation, computes derivatives with `np.gradient`, subtracts a per-trial baseline derivative, converts to speed magnitude, and averages the two paw speeds per timepoint.

ii. ```python
if "tongue" not in feature_name:
    xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
    ypos[:, trix] = nearest_fill_1d(ypos[:, trix])
...
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
xv = xv - basederiv[0]
yv = yv - basederiv[1]
...
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
```

iii. The notes cite reference nearest-fill rules for non-tongue features and describe the average of top- and bottom-paw speeds as a pragmatic single-variable choice.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code computes the per-session median of all kept paw-speed samples and binarizes each sample relative to that threshold.

ii. ```python
paw_thr = summarize_threshold(paw_sel)
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64)
```

iii. The notes say this median split is required by the decoder instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned exactly like tongue trajectories: video `frameTimes` are shifted by the bitcode-derived video offset and by trial `goCue`, then interpolated onto the neural time axis.

ii. ```python
paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```

iii. The notes say all video-derived outputs should share the same aligned axis as neural activity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<subject>_<date>.mat` file, specifically the stored `me.data` traces and optional `me.moveThresh`.

ii. ```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. The notes identify `loadMotionEnergy` as the relevant reference loader and state that ephys sessions store motion energy separately.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads the precomputed per-trial motion-energy traces, unwraps multiple MATLAB storage variants, aligns them to go cue using frame times and video offset, interpolates them onto the neural time base, and fills missing samples by nearest-neighbor interpolation.

ii. ```python
def load_motion_energy(path: Path) -> tuple[list[np.ndarray], float]:
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    ...
    data = [np.asarray(...).reshape(-1) for v in flat]

motion = aligned_motion_energy(raw, raw["events"]["goCue"], time_vec)
motion = np.nan_to_num(motion, nan=0.0)
```

iii. The notes say the paper’s raw frame-difference computation was already collapsed into `me.data` and that the converter should mirror the reference loader rather than recompute motion energy from video.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The code uses a per-session 50th percentile over kept motion-energy samples and binarizes each timepoint relative to that threshold.

ii. ```python
motion_thr = summarize_threshold(motion_sel)
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64)
```

iii. Step 5 explicitly notes that this follows the decoder task even though the paper sometimes used a manually chosen movement threshold instead.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to go cue by shifting the motion-energy frame times by the video offset and trial `goCue`, then interpolating onto `time_vec`. If frame times are missing, the code falls back to a synthetic 400 Hz grid with a `-0.5 s` offset.

ii. ```python
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    old_t = (np.arange(me.size, dtype=np.float64) + 1.0) / 400.0 - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. The notes say the intended rule was to follow the reference video-offset logic and interpolate onto the neural axis; later notes document extra handling for motion-energy file-format variants.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several data irregularities explicitly: HDF5-vs-old-MAT session files, empty HDF5 probe slots, nested or alternate motion-energy containers, missing `frameTimes`, missing kinematic samples, absent non-tongue tracking frames, trials beyond neural coverage, and sessions with too few trials or units.

ii. ```python
return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
...
if not isinstance(probe, h5py.Group):
    raise IndexError(...)
...
if frame_times is None or frame_times.size == 0 ...:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0
...
x[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
...
selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. Step 9/10 in the notes document the concrete issues the agent found and fixed: nested `JEB15` motion-energy structs, randomized-delay motion-energy files without `moveThresh`, an empty raw probe slot in `JEB6_2021-04-18`, and late behavioral trials without neural coverage in two `JEB24` sessions.

## 11-a. What are the most time-consuming steps of the code?

i. The agent identifies loading large HDF5 sessions, per-unit spike accumulation across trials/time bins, and session conversion as the main runtime costs.

ii. ```python
with h5py.File(spec.session_path, "r") as f:
    ...
for unit in raw["units"]:
    unit_mat = compute_unit_trial_matrix(...)
...
for spec in session_specs:
    session = convert_one_session(spec, ...)
```

iii. Step 6/7 of the notes explicitly call out full recursive HDF5 loading and per-spike/per-trial histogramming as the expected bottlenecks.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The biggest remaining non-vectorized loops are the loop over units, the loop over trials within `feature_xy`, the loop over trials within `feature_speed`, and the per-trial assembly loop that stacks outputs. The spike binning itself was already partly vectorized with `np.add.at`.

ii. ```python
for unit in raw["units"]:
    ...
for trix, trial in enumerate(trials):
    ...
for i in range(n_trials):
    ...
for local_idx, trial_idx in enumerate(selected_trials):
    ...
```

iii. The notes say the agent already optimized away a nested histogram loop, implying these remaining loops are the obvious vectorization targets.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats very similar video-alignment and interpolation logic separately for tongue, each paw marker, and motion energy. It also recomputes per-trial aligned feature arrays session-by-session instead of sharing intermediate aligned streams.

ii. ```python
tongue_pos = feature_xy(raw, 0, "tongue", ...)
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, ...)
motion = aligned_motion_energy(raw, raw["events"]["goCue"], time_vec)
```

iii. The notes describe this as deliberate targeted loading, but the repeated interpolation path is still evident in the code.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The converter stores a `continuous` block containing the continuous tongue, paw, and motion traces and threshold values only for plotting/debugging, but those arrays are not part of the final exported dataset. The script also contains plotting code used only when `--show-processing` is enabled.

ii. ```python
session_out = {
    ...
    "continuous": {
        "tongue": tongue_sel.astype(np.float32),
        "paw": paw_sel.astype(np.float32),
        "motion": motion_sel.astype(np.float32),
        ...
    },
}
...
if show_processing:
    plot_processing(...)
```

iii. The notes say these pieces were added for validation and sample-review plots rather than for the final decoder data structure.
