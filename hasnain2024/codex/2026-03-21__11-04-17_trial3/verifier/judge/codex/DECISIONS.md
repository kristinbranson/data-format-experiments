# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not glob the data folders. It parses the reference MATLAB loader scripts in `/app/code/DataLoadingScripts/Recording and video/` to recover the session list, per-session probe selection, task type, and source folder. Each parsed entry becomes a `SessionSpec`, then `process_session()` loads the corresponding `data_structure_<subject>_<date>.mat` file with `pymatreader.read_mat`. Motion energy is loaded separately from `motionEnergy_<subject>_<date>.mat` when present, with a fallback to embedded `obj["me"]`.

ii. 
```python
def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    ...
    for raw_line in path.read_text().splitlines():
        ...
        if "datapth = fullfile" in line and current_date is not None and current_probe is not None:
            sessions.append(
                SessionSpec(
                    subject=subject,
                    date=current_date,
                    probe=current_probe,
                    folder=folder,
                    task=task,
                )
            )
```
```python
def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions
```
```python
obj = read_mat(spec.data_path)["obj"]
...
if spec.motion_energy_path.exists():
    raw_me = read_mat(spec.motion_energy_path).get("me")
```

iii. In `CONVERSION_NOTES.md`, the AI says it deliberately recovered the analyzed sessions from the authors' loader files so session inclusion would stay code-matched, and later notes that it fixed dual-probe parsing for `JEB15`. It also chose `pymatreader` so one loader could handle both MATLAB formats.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are carried as the `subject` field of each `SessionSpec`, which is parsed from the MATLAB loader filename such as `loadJEB24_ALMVideo.m`. During assembly, each session is assigned the index of its subject in first-seen order.

ii. 
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
...
subject = subject_match.group(1)
```
```python
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. The notes state that the released files do not store subject identifiers consistently internally, so the agent preferred the loader/file naming convention as the canonical subject source.

## 1-c. How are the data split into sessions?

i. One parsed loader entry becomes one session. A session is identified by `<subject>_<date>`, points at exactly one `data_structure_*.mat` file in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`, and can list one or more selected probes.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probe: Tuple[int, ...]
    folder: str
    task: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"
```
```python
def process_session(
    spec: SessionSpec,
    show_processing: bool,
) -> Optional[Tuple[dict, List[np.ndarray], List[np.ndarray], List[np.ndarray], np.ndarray, int, np.ndarray]]:
    obj = read_mat(spec.data_path)["obj"]
```

iii. `CONVERSION_NOTES.md` says the AI followed the explicit ephys session lists from the loader files rather than treating the two task folders as separate datasets or scanning every raw file.

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial arrays in `obj["bp"]`. The AI builds a Boolean `trial_mask` over the raw trial axis, converts it to kept trial indices with `np.flatnonzero`, and then indexes all trial-level variables and per-neuron spike times by those trial indices.

ii. 
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
```
```python
trial_mask = build_trial_mask(obj)
keep_trials = np.flatnonzero(trial_mask)
...
R = to_vector(bp["R"], float).astype(int)
lick_direction = R[keep_trials]
```

iii. The notes describe trial selection as being driven directly by `obj.bp` fields and say this keeps output definitions simple because all downstream per-trial labels are read from those same trial-indexed arrays.

## 1-e. How are trials filtered based on quality controls?

i. The AI drops photostimulation trials (`stim.enable`), early-lick trials (`early`), and no-response trials (`no`). After neuron selection, it also trims any remaining behavior-valid trials whose raw trial number exceeds the last trial in which any selected neuron spikes.

ii. 
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
```
```python
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
if max_trial_with_spikes > 0:
    neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
    refined_trial_mask = trial_mask & neural_coverage_mask
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI says it excluded `no` trials to keep choice and outcome “well-defined” and later says the neural-coverage trim was added after validation exposed all-zero neural trials in two `JEB24` sessions.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from each selected probe's `trial` and `trialtm` spike annotations in `obj["clu"]`, together with `obj["bp"]["ev"]["goCue"]` for alignment. `quality` is also read for neuron selection.

ii. 
```python
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
...
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
```
```python
qualities = [normalize_string(q) for q in probe["quality"]]
quality_mask = quality_keep_mask(qualities)
```

iii. The notes say the released code path is ephys, not calcium imaging, so spikes should come directly from the curated cluster tables and be aligned to the go cue.

## 2-b. How is the `neural` data processed?

i. For each kept neuron, the AI counts spikes into 5 ms bins across the `[-2.5, 2.5)` window, divides by `DT` to convert to rate, and applies a one-sided “causal” Gaussian-like smoothing kernel of length `SMOOTH=15` with reflective padding. It does not z-score or baseline-subtract before storing the trial matrices.

ii. 
```python
def binned_neuron_trials(
    probe: dict,
    neuron_index: int,
    align_times: np.ndarray,
    keep_trials_0based: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    ...
    counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
    np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
    rates = counts / DT
    return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```
```python
def causal_gaussian_smooth(x: np.ndarray, n: int, bctype: str = BCTYPE) -> np.ndarray:
    ...
    kern = gaussian_window(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()
```

iii. The AI explicitly states in `CONVERSION_NOTES.md` and `README.md` that it believed the reference used “causal Gaussian smoothing,” and that it was reproducing `getSeq` with `reflect` boundary handling.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only selected probes, removes clusters with labels `garbage`, `gabrga`, `noisy`, or `real?`, computes each remaining neuron's mean firing rate over the go-cue window on kept trials, and retains neurons with `mean_fr > 1 Hz`. It also drops any session with fewer than `10` kept units.

ii. 
```python
def quality_keep_mask(qualities: Sequence[str]) -> np.ndarray:
    cleaned = [normalize_string(q).lower() for q in qualities]
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)
```
```python
mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
if mean_fr > LOW_FR:
    kept_indices.append(int(neuron_index))
```
```python
if len(selected_neurons) < MIN_UNITS_PER_SESSION:
    log(f"Skipping {spec.session_id}: only {len(selected_neurons)} units after filtering")
    return None
```

iii. Step 5 of `CONVERSION_NOTES.md` says the AI followed the released-code quality filter plus the paper's `>1 Hz` criterion, and also added the “at least 10 units” session rule from the methods text.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the go-cue time of the spike's own trial from `trialtm`, then binning the resulting time-from-go-cue values.

ii. 
```python
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The notes repeatedly say the decoder task required go-cue alignment and that the AI was matching `alignSpikes` by expressing each spike relative to `bp.ev.goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use a fixed 5 ms grid (`DT = 0.005`) over `[-2.5, 2.5)`, yielding 1000 time bins. Spikes are binned directly onto this grid; no later temporal rebinning is applied.

ii. 
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
```
```python
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. The AI's notes say it chose 5 ms because the paper's decoder analyses explicitly use 5 ms bins and because that same grid can be shared across neural and behavioral streams.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a raw field. It is constructed from the chosen go-cue-aligned analysis window and bin size, using the common bin centers implied by `TMIN`, `TMAX`, and `DT`.

ii. 
```python
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)

def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0
```

iii. In `CONVERSION_NOTES.md`, the AI describes this input as the common aligned time axis required by the decoder format rather than a raw recorded variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI creates the bin edges, converts them to bin centers, and reuses the same `(1, 1000)` time vector for every trial in every session.

ii. 
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
...
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The notes say the input was intentionally minimal: one continuous time-from-go-cue channel that exactly matches the neural grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same analysis grid the neural spikes are binned onto: neural uses `time_edges`, while the decoder input uses the corresponding `time_centers`.

ii. 
```python
rates = counts / DT
return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The AI's notes explicitly say the input is the common aligned time axis derived from the same binning used for neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the per-trial `obj["bp"]["R"]` indicator only. Because it removes `no` trials earlier, it treats the instructed side as the lick-direction label on the kept trials.

ii. 
```python
R = to_vector(bp["R"], float).astype(int)
...
lick_direction = R[keep_trials]
```

iii. In Step 5 of `CONVERSION_NOTES.md`, the AI justified excluding `no` trials so “choice/outcome are well-defined,” and then treated the remaining left/right task label as sufficient for lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is no extra inference beyond selecting the `R` flag on kept trials and broadcasting it across time. `R=0` becomes left and `R=1` becomes right.

ii. 
```python
lick_direction = R[keep_trials]
...
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
```

iii. The notes frame this as a simplification enabled by dropping no-response trials, so the AI did not reconstruct choice from hit/miss plus instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `obj["bp"]["autowater"]` field.

ii. 
```python
autowater = to_vector(bp["autowater"], float).astype(int)
```

iii. The notes say the AI used `autowater` as the raw indicator for water-cued versus delayed-response context, matching the reference task structure.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI remaps `autowater` to the decoder's requested coding, using `1 - autowater` so WC becomes `0` and DR becomes `1`.

ii. 
```python
context = 1 - autowater[keep_trials]
```

iii. Step 5 in `CONVERSION_NOTES.md` explicitly records the decision to encode WC=`0`, DR=`1` even though raw `autowater` has the opposite polarity.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived primarily from `obj["bp"]["hit"]`, with `obj["bp"]["miss"]` used only as a consistency check after filtering.

ii. 
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
...
outcome = hit[keep_trials]
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(f"{spec.session_id}: hit/miss are not mutually exclusive on kept trials")
```

iii. Because the AI had already removed `no` trials, the notes say it treated outcome as a binary miss-versus-hit target rather than preserving an ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is converted to a binary label by taking `hit` directly on the kept trials: `0` for incorrect and `1` for correct. It is then repeated across all time bins within each trial.

ii. 
```python
outcome = hit[keep_trials]
...
np.full(time_centers.size, outcome[tr], dtype=np.int64)
```

iii. Step 5 of `CONVERSION_NOTES.md` says the AI chose to “represent only miss versus hit” and to exclude ignore/no-response trials instead of adding a third outcome class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from multiple tracked tongue-related features in `obj["traj"]` across both cameras, plus frame times, go-cue times, and the video/behavior clock offset from `obj["sglx"]["bitcode"]["bitstart"]` and `obj["bp"]["ev"]["bitStart"]`.

ii. 
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
```
```python
vidshift = compute_video_offset(obj)
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
```

iii. The notes say the decoder task wanted a single tongue-velocity output, so the AI chose to aggregate “the feature groups needed by the decoder task” rather than preserve the reference multi-feature representation.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each selected tongue feature, the AI interpolates x/y coordinates from frame times directly onto the common 5 ms time grid, computes x and y velocities with `np.gradient`, converts them to speed, and averages speeds across all available tongue-related features and both views. Missing values are converted to zeros at the aggregate stage. The code does not use the original DLC likelihood values, run-wise frame-domain smoothing, or per-view normalization before combining views.

ii. 
```python
interp = interp1d(
    old_time,
    xy,
    axis=0,
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)
xy_aligned = interp(taxis)
```
```python
def feature_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> Tuple[np.ndarray, np.ndarray]:
    ...
    xv = np.gradient(tsinterp[:, 0]).astype(np.float32)
    yv = np.gradient(tsinterp[:, 1]).astype(np.float32)
    ...
```
```python
stacked = np.stack(speed_components, axis=2)
...
agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. Step 5 of `CONVERSION_NOTES.md` justifies this as a required collapse from the paper's richer kinematic feature tensor to one tongue-speed trace suitable for the decoder outputs.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes a per-session 50th percentile value, but the actual discretization function does not compare each sample to that value. Instead, it sorts the entire flattened session trace and assigns the top half of samples to class `1` and the bottom half to class `0`, forcing an exact 50/50 split. There is no extra “not visible” category.

ii. 
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
...
tongue_bin = discretize_trace(tongue_speed, tongue_thr)
```
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. The notes say the AI was following the task's “per-session median” instruction and intentionally kept the movement outputs binary rather than adding a visibility class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimates a session-wide video offset, subtracts that and the trial's go-cue time from each frame-time vector, and then interpolates tracked positions directly onto the same 5 ms time grid used for the neural data.

ii. 
```python
def compute_video_offset(obj: dict) -> float:
    ...
    return float(vid_file_offset - bit_start)
```
```python
old_time = frame_times - vidshift - float(align_times[trial])
...
taxis = time_centers + ADVANCE_MOVEMENT
xy_aligned = interp(taxis)
```

iii. `CONVERSION_NOTES.md` says the AI intended to reproduce the reference video-offset correction and common go-cue-aligned time axis, but to represent the resulting kinematics directly on the neural grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-camera paw features in `obj["traj"]`, specifically both `top_paw` and `bottom_paw`, along with frame times, go-cue times, and the computed video offset.

ii. 
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
```
```python
paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
```

iii. The notes say the AI wanted one paw-speed output for the decoder and therefore aggregated the relevant paw-tracking features rather than using the paper's original feature set.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw positions are linearly interpolated onto the 5 ms grid, missing non-tongue positions are nearest-filled, x/y velocities are computed by `np.gradient` on the interpolated trajectory, a per-trial baseline is subtracted from those gradients, speeds are formed with Euclidean magnitude, and the available paw-feature speeds are averaged together.

ii. 
```python
if "tongue" not in feat_name:
    xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
    ypos[:, trial] = fill_nearest_1d(ypos[:, trial])
```
```python
if "tongue" not in feat_name:
    xv = xv - base[0]
    yv = yv - base[1]
    xv = fill_nearest_1d(xv)
    yv = fill_nearest_1d(yv)
```
```python
speed = np.sqrt(np.square(xvel) + np.square(yvel))
```

iii. The AI's notes describe this as “reference video interpolation and per-feature velocity calculation” followed by aggregation to the single paw-speed output required by the task.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Like tongue velocity, paw velocity is turned into a binary output by computing a session median but then assigning class `1` to the top half of flattened samples and class `0` to the bottom half. There is no separate missing/visibility category.

ii. 
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
...
paw_bin = discretize_trace(paw_speed, paw_thr)
```
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. The notes say the AI followed the decoder task's request for two median-split paw-velocity bins.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity uses the same session-wide video offset and trial-wise go-cue subtraction as the tongue features, then interpolates onto the same 5 ms common grid as the neural data.

ii. 
```python
old_time = frame_times - vidshift - float(align_times[trial])
...
taxis = time_centers + ADVANCE_MOVEMENT
```

iii. The notes say all behavioral traces were meant to share the common go-cue-aligned time axis with neural data.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the `me` structure in `motionEnergy_<session>.mat` when that sidecar file exists, otherwise from embedded `obj["me"]`. The usable signal is `me["data"]`, recursively unwrapped until it becomes a per-trial list of frame-wise motion-energy traces.

ii. 
```python
def load_motion_energy_raw(obj: dict, spec: SessionSpec) -> Optional[dict]:
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        ...
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}
    if "me" in obj:
        raw_me = obj["me"]
        ...
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": np.nan}
```
```python
def unwrap_embedded_motion_energy(raw_me) -> List[np.ndarray]:
    data = raw_me
    ...
    while isinstance(data, dict) and "data" in data and id(data) not in seen:
        ...
```

iii. The notes say the AI added recursive unwrapping because some files, especially `JEB15`, store nested `me.data` wrappers, and it included the embedded `obj.me` fallback to be robust to missing sidecars.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each frame-wise motion-energy trace onto the common 5 ms time grid using side-camera frame times corrected by video offset and go-cue time. It nearest-fills edge `NaN`s and converts all remaining missing values to zero before discretization.

ii. 
```python
interp = interp1d(
    old_time,
    me_trial,
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)
aligned[:, trial] = interp(taxis).astype(np.float32)
aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
```
```python
return np.nan_to_num(aligned, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```

iii. The notes state that movement outputs should share the same aligned time axis as neural data and that motion energy should then be binarized by the task-mandated session median rather than the paper's manual threshold.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded in the same way as tongue and paw velocity: the AI computes a session 50th percentile value but actually assigns class `1` to the upper half of flattened samples and class `0` to the lower half, yielding an exactly balanced binary split.

ii. 
```python
me_thr = percentile_threshold(motion_energy, 50.0)
...
me_bin = discretize_trace(motion_energy, me_thr)
```
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)
```

iii. The notes justify this with the decoder task's request for per-session median binning and the desire to keep motion energy binary.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by using side-camera frame times, subtracting the session video offset and per-trial go-cue time, and interpolating onto the same 5 ms common grid used by the neural data.

ii. 
```python
view = obj["traj"][0]
...
frame_times = get_frame_times(trial_view, ts.shape[0])
...
old_time = frame_times - vidshift - float(align_times[trial])
...
aligned[:, trial] = interp(taxis).astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says the AI treated motion energy like the other camera-derived streams, with explicit video-offset correction and a shared go-cue-aligned time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills or falls back rather than preserving missingness. Missing or invalid frame times are replaced by synthetic `1/400 s` spacing. Missing non-tongue positions and velocities are nearest-filled. Missing tongue velocities are zero-filled after aggregation. Missing motion-energy bins are nearest-filled and then zero-filled. If the video offset cannot be computed, the code falls back to `0.5 s`. If a motion-energy sidecar is absent and `obj["me"]` is also absent, the code returns an all-zero motion-energy trace.

ii. 
```python
def get_frame_times(trial_view: dict, n_frames: int) -> np.ndarray:
    frame_times = trial_view.get("frameTimes")
    if frame_times is None:
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
    ...
    if arr.size != n_frames or not np.isfinite(arr).any():
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0
```
```python
def compute_video_offset(obj: dict) -> float:
    ...
    return 0.5
```
```python
if "tongue" not in feat_name:
    xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
    ypos[:, trial] = fill_nearest_1d(ypos[:, trial])
...
agg = np.nan_to_num(agg, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
```
```python
if raw_me is None:
    return np.zeros((time_centers.size, n_trials), dtype=np.float32)
```

iii. The notes mention several of these choices as pragmatic fixes for messy files and describe nearest-filling and fallback defaults as ways to keep the decoder-format outputs fully finite and validation-clean.

## 11-a. What are the most time-consuming steps of the code?

i. The AI identifies large MATLAB-file loading as the main cost and also calls out kinematic interpolation and per-neuron spike binning as the dominant processing costs once a session is in memory.

ii. 
```python
obj = read_mat(spec.data_path)["obj"]
...
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
...
rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
```

iii. In Step 6 and Step 7 of `CONVERSION_NOTES.md`, the AI explicitly says session loading dominates runtime and that kinematic interpolation and spike binning are the other expensive stages.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI leaves several explicit Python loops in place: trial loops in `aligned_position()` and `feature_velocity()`, feature loops in `aggregate_speed()`, neuron loops in `select_neurons()` and `process_session()`, and a per-trial copy loop when writing each neuron's rates into `neural_trials`. Its notes suggest it considered frame-count irregularity a reason not to fully vectorize the camera-side computations.

ii. 
```python
for trial in range(n_trials):
    trial_view = get_view_trial(view, trial)
    ...
    xy_aligned = interp(taxis)
```
```python
for out_idx, selected in enumerate(selected_neurons):
    probe = probes[selected.probe_num - 1]
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
    ...
    for tr in range(n_trials):
        neural_trials[tr][out_idx, :] = rates[:, tr]
```

iii. The notes' efficiency section argues that the AI already added the important speedups it cared about, especially parsing loader files once and using a two-pass neural path so it would not build trial matrices for low-rate neurons.

## 11-c. What processing does the code repeat multiple times?

i. The AI intentionally uses a two-pass neural path: first it scans candidate neurons to compute mean firing rates for selection, then it bins the kept neurons again to build the final trial matrices. If neural-coverage trimming changes the kept trials, it reruns neuron selection a second time under the refined mask.

ii. 
```python
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
...
if not np.array_equal(refined_trial_mask, trial_mask):
    ...
    selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
```
```python
for neuron_index in np.flatnonzero(quality_mask):
    mean_fr = neuron_mean_fr(probe, int(neuron_index), align_times, trial_mask)
```
```python
for out_idx, selected in enumerate(selected_neurons):
    ...
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)
```

iii. Step 6 in `CONVERSION_NOTES.md` explicitly says this repeated work was a deliberate two-pass speed/memory tradeoff: estimate firing rates first, then materialize full trial matrices only for kept neurons.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes several intermediates that are not used for the final labels themselves: the scalar percentile thresholds are stored and plotted but not used by `discretize_trace()`, which instead rank-splits the flattened traces; `moveThresh` is loaded from motion-energy files but never used; and the code aggregates many tongue/paw features down to a single scalar trace, discarding the per-feature information after computing it.

ii. 
```python
tongue_thr = percentile_threshold(tongue_speed, 50.0)
paw_thr = percentile_threshold(paw_speed, 50.0)
me_thr = percentile_threshold(motion_energy, 50.0)
...
tongue_bin = discretize_trace(tongue_speed, tongue_thr)
```
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
```
```python
return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": move_thresh}
```

iii. The notes emphasize decoder-oriented scalar movement outputs and plotting/metadata summaries, which explains why these extra values were computed even though some are not consumed by the final discrete labels.
