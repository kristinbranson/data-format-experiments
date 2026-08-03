# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the reference MATLAB loader scripts (`load*_ALMVideo.m`) in `code/DataLoadingScripts/Recording and video/` to recover the exact list of sessions and probe selections. It then intersects this list with the `.mat` data files actually present in `data/`. Data files are loaded one at a time using either `h5py` (for MATLAB v7.3/HDF5 files) or `scipy.io.loadmat` (for older `.mat` files). Each session's `.mat` file contains the `obj` structure with behavioral, neural, video/DLC, and event data. Motion energy is loaded from separate `motionEnergy_*.mat` files.

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
        # ... parses anm, date, probe from MATLAB loader scripts ...
        key = (current["subject"], current["date"])
        if key in data_files:
            # ... creates SessionSpec ...

def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. The AI justified this by noting that the reference code's loader scripts define the analyzed session list and probe selections, and that intersecting with available data files yields the 44-session dataset described in the paper (25 fixed-delay + 19 randomized-delay).

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is extracted from the session file names and loader script parameters (`anm` variable). Each `SessionSpec` stores a `subject` field. Unique subjects are collected in a list, and `subject_idx` maps each session to its subject.

ii.
```python
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The AI identified 14 unique neural subjects across the available data, consistent with 9 fixed-delay + 4 randomized-delay mice (with some overlap).

## 1-c. How are the data split into sessions?

i. Each `.mat` file corresponds to one recording session. The AI processes sessions one at a time via `convert_one_session()`, yielding lists of trials per session. Sessions are identified by `subject_date` (e.g., `EKH1_2021-08-07`).

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

iii. The AI used the reference loader scripts to determine which sessions to include, yielding 44 sessions.

## 1-d. How are the data split into trials?

i. Trials are indexed within each session using the behavioral data arrays (e.g., `bp.R`, `bp.hit`, etc.), which have one entry per trial. Selected trials (after filtering) are indexed into the session's neural and behavioral arrays.

ii.
```python
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
# ... also limits to trials with neural coverage ...
if selected_trials.size < 2:
    log(f"SKIP {spec.session_id}: ...")
    return None
```

iii. The AI noted that each session file has `bp.Ntrials` trials, and the conversion selects a subset based on quality filters.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only those that are: (1) not stimulation trials (`stim_enable == 0`), (2) not early lick trials (`early == 0`), (3) either hit or miss (`hit == 1 | miss == 1`), and (4) have a defined lick direction (`R == 1 | L == 1`). Additionally, trials whose index exceeds the maximum neural trial index are excluded.

ii.
```python
def session_valid_trial_mask(raw: dict) -> np.ndarray:
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )

# Also in convert_one_session:
covered_trial_max = [
    int(np.nanmax(unit["trial"]))
    for unit in raw["units"]
    if good_quality(unit["quality"]) and np.asarray(unit["trial"]).size
]
if covered_trial_max:
    max_neural_trial = min(raw["R"].size, max(covered_trial_max))
    selected_trials = selected_trials[selected_trials + 1 <= max_neural_trial]
```

iii. The AI justified this by referencing the default conditions in `getDefaultParams.m` which require `~stim.enable`, `~early`, and specific hit/miss+direction conditions. The neural coverage filter handles edge cases where behavioral trials continue after neural recording ends.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}` which contains spike times per unit. Specifically, for each unit: `trialtm` (within-trial spike times), `trial` (trial index for each spike), and `quality` (manual curation label).

ii.
```python
# In load_session_mat:
for unit in np.asarray(probe).reshape(-1):
    units.append({
        "quality": mat_to_str(getattr(unit, "quality", "")),
        "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
        "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
    })
```

iii. The AI correctly identified that the reference code operates on `obj.clu{probe}(cluster).trialtm` for spike times relative to trial start, and `trial` for the trial assignment of each spike.

## 2-b. How is the `neural` data processed?

i. For each unit, spikes are aligned to the go cue event, binned into 5 ms bins over a [-2.5, 2.5] s window (1000 bins), converted to firing rates by dividing by `dt`, and smoothed with a causal Gaussian kernel (window size 15, reflected boundary condition).

ii.
```python
TMIN = -2.5; TMAX = 2.5; DT = 1.0 / 200.0; SMOOTH = 15; BCTYPE = "reflect"

def compute_unit_trial_matrix(unit, go_cue, trial_to_pos, n_sel, time_edges):
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    # ... select valid spikes ...
    bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
    np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat
```

iii. The AI matched the reference `getSeq.m` processing: `edges = tmin:dt:tmax`, `N = histc(trialtm_aligned, edges)`, `N./dt`, then `mySmooth(N./dt, smooth, bctype)`. The AI uses `bctype='reflect'`; the default in `processData.m` is `'none'`, but the Figure 8 decoding analysis script explicitly sets `bctype='reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) units with quality labels `garbage`, `gabrga`, `noisy`, or `real?` are excluded; (2) units with mean firing rate <= 1 Hz are excluded. Sessions with fewer than 10 remaining units are skipped entirely.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

LOW_FR_HZ = 1.0
# In convert_one_session:
if not good_quality(unit["quality"]):
    continue
# ...
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
# ...
if kept_units < 10:
    log(f"SKIP {spec.session_id}: only {kept_units} units ...")
    return None
```

iii. The AI justified the quality filter by matching `findClusters.m` with `quality = {'all'}`. For the FR threshold, the AI chose 1 Hz based on the methods text ("units with firing rates exceeding 1 Hz") rather than the code default of 0.5 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset by subtracting `bp.ev.goCue(trial)` from each spike's `trialtm`. The aligned spike times are then binned relative to the go cue.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. The AI matched the reference `alignSpikes.m` function which computes `trialtm_aligned` by subtracting the alignment event time from each spike's trial time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (1/200 s), matching `getDefaultParams.m`. No additional rebinning is applied. The time axis spans [-2.5, 2.5] s = 1000 bins.

ii.
```python
DT = 1.0 / 200.0  # 5 ms
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. The AI noted that the default `getDefaultParams.m` uses `dt = 1/200`, and chose this over the 10 ms bin used in the tutorial `WorkingWithDataObjs.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself: the bin centers of the neural time axis, relative to the go cue alignment event.

ii.
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. This follows directly from the instructions requesting "Time from go cue onset in seconds" as a continuous, time-varying decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time vector is constructed as bin centers from the edges array: `edges[:-1] + dt/2`. This is the same for all trials and sessions.

ii.
```python
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. Matches the reference code's `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1)`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is identical to the neural time axis since both are defined from the same bin edges. Each trial gets the same time vector `time_vec` as a `(1, n_timepoints)` array.

ii.
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. By construction, the time axis is shared between neural and input data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` and `bp.L` (per-trial indicators of right and left lick direction).

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. The AI noted that `R=1` means right lick and `L=1` means left lick, matching the reference code's `findTrials` conditions.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Right lick (`R==1`) maps to 1, left lick maps to 0. The per-trial value is broadcast as a constant time series across all time bins.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
np.full(time_vec.size, lick_direction, dtype=np.int64),
```

iii. Matches the instruction specification: left = 0, right = 1.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater` (per-trial indicator of whether autowater/water-contingent mode was active).

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. The AI identified that `autowater=1` corresponds to WC (water contingent) trials and `autowater=0` to DR (delayed response) trials from the reference code conditions.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. WC (`autowater==1`) maps to 0, DR (`autowater==0`) maps to 1. The per-trial value is broadcast as a constant time series.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
np.full(time_vec.size, context, dtype=np.int64),
```

iii. Matches the instruction specification: WC = 0, DR = 1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` and `bp.miss` (per-trial indicators of correct and incorrect outcomes).

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. Only hit and miss trials are included (early and ignore trials are already filtered out).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit (`hit==1`) maps to 1 (correct), miss maps to 0 (incorrect). The per-trial value is broadcast as a constant time series.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
np.full(time_vec.size, outcome, dtype=np.int64),
```

iii. Matches the instruction specification: incorrect = 0, correct = 1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from the side-view (view index 0) DLC `tongue` feature trajectories (`obj.traj{1}(trial).ts(:,1:2,tongue_feat_idx)`) and the associated `frameTimes`.

ii.
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. The AI used the canonical `tongue` marker from the side view, matching the reference code's `findPosition(taxis, obj, nTrials, 1, 'tongue', alignEvent)`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The tongue x,y positions are extracted from the DLC trajectories, aligned using the video offset and go cue times, interpolated to the neural time axis. Speed is computed as `sqrt(gradient(x)^2 + gradient(y)^2)`. For tongue features, NaN positions are NOT filled with nearest values (unlike other features), and NaN velocities are set to zero.

ii.
```python
def feature_speed(xpos, ypos, feature_name):
    # ...
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    if "tongue" not in feature_name:
        xv = xv - basederiv[0]
        yv = yv - basederiv[1]
        xv = nearest_fill_1d(xv)
        yv = nearest_fill_1d(yv)
    else:
        xv = np.nan_to_num(xv, nan=0.0)
        yv = np.nan_to_num(yv, nan=0.0)
    return np.sqrt(xvel**2 + yvel**2)
```

iii. The AI followed the reference `findVelocity.m` logic: gradient for velocity, no baseline subtraction for tongue, NaN-to-zero for tongue velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The continuous tongue speed is binarized using the per-session 50th percentile (median) threshold computed over all selected trials' valid (finite) time points. Values >= threshold map to 1, values < threshold map to 0. Non-finite values map to 0.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)
# ...
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. Matches the instruction specification: 50th percentile threshold, 0 for below, 1 for at/above.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The tongue position trajectories are interpolated onto the neural time axis using the same video offset correction and go cue alignment as the neural data. The velocity is then computed on this aligned time axis.

ii.
```python
def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    vidshift = find_video_offset(raw)
    # ...
    old_t = frame_times - vidshift - align_times[trix]
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. Matches the reference `findPosition.m`: `interp1(frameTimes-vidshift-alignTimes(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom-view (view index 1) DLC `top_paw` and `bottom_paw` feature trajectories.

ii.
```python
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
```

iii. The AI used both paw features from the bottom view, consistent with the reference code's DLC feature list.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature (top_paw, bottom_paw): positions are extracted, NaN-filled with nearest values (non-tongue behavior), aligned and interpolated to the neural time axis. Velocity is computed via `gradient()`, with baseline drift subtracted (median of `diff`). The two paw speeds are averaged. NaN values are replaced with 0.

ii.
```python
paw_stack = np.stack(paw_speeds, axis=0)
paw_count = np.sum(np.isfinite(paw_stack), axis=0)
paw_sum = np.nansum(paw_stack, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The AI computed speed as `sqrt(xvel^2 + yvel^2)` for each paw then averaged them, capturing overall paw movement.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile threshold over all selected trials. Values >= threshold map to 1, values < threshold map to 0.

ii.
```python
paw_thr = summarize_threshold(paw_sel)
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. Matches the instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: paw positions are interpolated onto the neural time axis using video offset correction and go cue alignment.

ii.
```python
paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```

iii. Uses the same `findPosition`-style interpolation as all other video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from per-trial motion energy traces stored in separate `motionEnergy_*.mat` files (`me.data`).

ii.
```python
me_path = spec.session_path.parent / f"motionEnergy_{spec.subject}_{spec.date}.mat"
motion_energy, motion_thresh = load_motion_energy(me_path)
```

iii. Matches the reference `loadMotionEnergy.m` which loads the motion energy file for each session.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces are interpolated onto the neural time axis using the side-camera frame times, video offset, and go cue alignment. NaN values are filled with nearest values.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    # ...
    old_t = frame_times - vidshift - align_times[trix]
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
    out[:, trix] = nearest_fill_1d(out[:, trix])
```

iii. Matches the reference `loadMotionEnergy.m`: `interp1(frameTimes-vidshift-alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing('nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold over all selected trials. Values >= threshold map to 1, values < threshold map to 0.

ii.
```python
motion_thr = summarize_threshold(motion_sel)
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. Matches the instruction specification. The AI noted this differs from the paper's manual movement threshold but follows the decoder task instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned to the neural time axis via interpolation using side-camera frame times, video offset, and go cue times.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. Matches the reference `loadMotionEnergy.m` alignment approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Missing/NaN video data: trials with `NdroppedFrames == NaN` are skipped for DLC features.
- Missing `frameTimes`: synthesized as `(1:nframes)/400` (matching reference code fallback).
- Tongue NaN positions: left as NaN (not filled), and velocities set to 0.
- Non-tongue NaN positions: filled with nearest values.
- Motion energy NaN values: filled with nearest values.
- Sessions with <10 units after filtering: skipped.
- Sessions with <2 valid trials: skipped.
- Trials exceeding neural coverage: excluded.
- Nested motion energy struct variants: recursively unwrapped.
- Empty HDF5 probe slots: handled by indexing raw slots directly.

ii.
```python
# Missing frameTimes fallback:
if frame_times is None or ...:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Tongue NaN handling:
if "tongue" not in feature_name:
    xpos[:, trix] = nearest_fill_1d(xpos[:, trix])
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Steps 9 and 10, and verified fixes against the specific sessions where they occurred (JEB6, JEB15, JEB23, JEB24).

## 11-a. What are the most time-consuming steps of the code?

i. The AI identified file loading (especially HDF5 sessions) and per-unit spike binning/smoothing as the main bottlenecks. The AI noted full conversion took ~135 seconds for 44 sessions.

ii.
```python
# Timing output:
log(f"SESSION {spec.session_id}: ... time={time.time() - t0:.2f}s")
log(f"TOTAL TIME {time.time() - t0:.2f}s")
```

iii. The AI documented timing in CONVERSION_NOTES.md Step 7 (4 s/session average for sample).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-column smoothing loop in `my_smooth` (line 91-92) iterates over neurons/trials. The per-trial feature extraction loops in `feature_xy` and `feature_speed` iterate over trials. The per-unit neural computation loop iterates over units.

ii.
```python
# Smoothing loop:
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")

# Feature extraction loop:
for trix, trial in enumerate(trials):
    # ... per-trial interpolation ...
```

iii. The AI noted it vectorized spike accumulation with `np.add.at` but left the smoothing and interpolation loops as-is.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed multiple times: once for tongue, once for each paw feature, and once for motion energy, all within the same session. The smoothing function `my_smooth` is called once per unit (per trial matrix), applying the same kernel construction each time.

ii.
```python
# find_video_offset called in feature_xy and aligned_motion_energy:
vidshift = find_video_offset(raw)  # called 4 times per session
```

iii. The AI did not explicitly note this redundancy but the code structure makes it evident.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes continuous velocity values for tongue, paw, and motion energy (`session_out["continuous"]`) which are stored in the session output dict but not included in the final pickle file. These are only used for plotting. The code also processes all trials for video features (not just selected trials) before subsetting to `selected_trials`.

ii.
```python
# Continuous data stored but not saved:
"continuous": {
    "tongue": tongue_sel.astype(np.float32),
    "paw": paw_sel.astype(np.float32),
    "motion": motion_sel.astype(np.float32),
    ...
}
# Video computed for all trials then subsetted:
tongue_sel = tongue_speed[:, selected_trials]
```

iii. The continuous data is used for plotting in `--show-processing` mode but not included in the final output dictionary.
