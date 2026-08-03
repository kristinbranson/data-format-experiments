# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI dynamically parses the authors' MATLAB loader scripts (`load*_ALMVideo.m`) to discover the 44 analyzed sessions and their probe assignments, rather than hard-coding them. It then loads each session file using either `scipy.io.loadmat` (for older MAT files) or `h5py` (for v7.3/HDF5 files). Motion energy is loaded separately from `motionEnergy_*.mat` files.

ii.
```python
def parse_reference_session_specs(code_dir, data_dir):
    data_files = find_data_files(data_dir)
    specs = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        ...
        if {"subject", "date", "probes"} <= current.keys():
            key = (current["subject"], current["date"])
            if key in data_files:
                specs.append(SessionSpec(...))
    return specs

def load_session(spec):
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. From CONVERSION_NOTES.md: "Parses the reference ALM session loader scripts to recover the analyzed session list and probe selections." The AI justified this as "the cleanest way to match the paper's analyzed dataset rather than the broader raw archive."

## 1-b. How are the data split into subjects?

i. The subject is extracted from the session filename (e.g., `EKH1` from `EKH1_2021-08-07`). Unique subjects are accumulated in order of first appearance.

ii.
```python
SessionSpec(subject=current["subject"], ...)
# In build_dataset:
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. The AI noted that the subject ID is always in the filename, consistent with the authors' loader scripts.

## 1-c. How are the data split into sessions?

i. One session corresponds to one `data_structure_*.mat` file. Sessions from both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` folders are treated uniformly. The result is 44 sessions (25 fixed-delay + 19 randomized-delay).

ii.
```python
def find_data_files(data_dir):
    out = {}
    for path in sorted(data_dir.glob("*/*.mat")):
        if not path.name.startswith("data_structure_"):
            continue
        parts = path.stem.split("_")
        out[(parts[2], parts[3])] = path
    return out
```

iii. From CONVERSION_NOTES.md: "Use the intersection of reference-code session lists and available data files. This yields 44 code-selected, paper-consistent ephys sessions."

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial arrays in `obj.bp` (behavioral protocol). Each element in arrays like `hit`, `miss`, `R`, `L`, `early`, `autowater` corresponds to one trial. Trial indices are 0-based in the converted representation.

ii.
```python
# In load_session_mat:
"R": np.asarray(bp.R, dtype=np.float64).reshape(-1),
"hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
...
# Spike times carry trial indices:
"trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
```

iii. The Bpod trial table defines the trials directly; each spike carries its trial index so no trial boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) early-lick trials removed (`early == 0`), (2) photostimulation trials removed (`stim_enable == 0`), (3) ignore/no-response trials removed (only `hit == 1` or `miss == 1` kept), and (4) trials must have a defined side (`R == 1` or `L == 1`). Additionally, trials past the last neural recording are dropped. This results in 11,955 trials from the original dataset.

ii.
```python
def session_valid_trial_mask(raw):
    return (
        (raw["stim_enable"] == 0)
        & (raw["early"] == 0)
        & ((raw["hit"] == 1) | (raw["miss"] == 1))
        & ((raw["R"] == 1) | (raw["L"] == 1))
    )
```

iii. From CONVERSION_NOTES.md: "Exclude stimulation, early-lick, and ignore/no-response trials: Reference analyses consistently use ~stim.enable, ~early, and usually hit/miss conditions; ignore trials are omitted in the paper."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` spike-sorted clusters, specifically the `trialtm` (spike time relative to trial start), `trial` (trial index), and `quality` (manual curation label) fields. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
# In load_session_mat:
units.append({
    "quality": mat_to_str(getattr(unit, "quality", "")),
    "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
    "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
})
```

iii. The AI correctly identified these as the spike timing fields used by the reference code.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue (`trialtm - goCue[trial-1]`), binned into 5 ms bins on a [-2.5, 2.5] s grid, converted to firing rates (by adding 1/dt per spike), and smoothed with a causal half-Gaussian kernel matching the MATLAB `mySmooth.m` implementation. The kernel is `gausswin(15)` with the first 7 elements zeroed out, normalized, and applied via convolution with reflected boundary padding.

ii.
```python
def compute_unit_trial_matrix(unit, go_cue, trial_to_pos, n_sel, time_edges):
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    ...
    bins = np.floor((aligned[keep] - time_edges[0]) / DT).astype(np.int64)
    np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat

def my_smooth(x, n, bctype="reflect"):
    ...
    kern = matlab_gausswin(n)
    kern[:n // 2] = 0.0
    kern /= kern.sum()
    for j in range(x_filt.shape[1]):
        out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```

iii. From CONVERSION_NOTES.md: "Reimplements the reference spike binning and smoothing path in Python, including the causal Gaussian kernel behavior in mySmooth.m."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels `garbage`, `gabrga`, `noisy`, or `real?` are excluded (matching `findClusters.m`), and (2) units with mean firing rate <= 1 Hz are dropped. Sessions with fewer than 10 units after filtering are skipped entirely.

ii.
```python
def good_quality(label):
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

# In convert_one_session:
for unit in raw["units"]:
    if not good_quality(unit["quality"]):
        continue
    unit_mat = compute_unit_trial_matrix(...)
    if float(unit_mat.mean()) <= LOW_FR_HZ:
        continue
    kept_units += 1
```

iii. From CONVERSION_NOTES.md: "Use all non-garbage manually curated units on the selected probe, then apply a 1 Hz FR threshold: This matches the paper's 'all units >1 Hz' rule."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's `trialtm` is subtracted by the `goCue` time of its trial, putting spike times in seconds relative to go cue onset. This is consistent with the reference `alignSpikes.m`.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. From CONVERSION_NOTES.md: "Aligns all streams to goCue on the same [-2.5, 2.5] s window with 5 ms bins."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Spikes are binned into 5 ms bins (dt = 1/200) on a [-2.5, 2.5] s window around the go cue, yielding 1000 time bins per trial. No rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
...
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. From CONVERSION_NOTES.md: "Use a fixed window of [-2.5, 2.5] s around goCue with a 5 ms bin (dt = 1/200): This matches the default reference processing pipeline."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is purely derived from the time bin grid definition, not from any raw data variable. It is the vector of bin centers spanning [-2.4975, 2.4975] s in 5 ms steps.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
...
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. N/A - this is a constructed time axis.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing - the time axis is directly constructed from the bin edge definitions.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the neural binning grid itself. The same `time_vec` defines the bin centers for both the spike histogram edges and the decoder input.

ii.
```python
# Same time_edges used for both:
unit_mat = compute_unit_trial_matrix(unit, ..., time_edges)
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction solely from `bp.R` (whether the right port was the correct/instructed port). If `R == 1`, lick direction is coded as 1 (right); otherwise 0 (left).

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. From CONVERSION_NOTES.md Step 5: "Lick direction: left 0, right 1; encode as a constant time series over the trial window." The AI treats R as the direction indicator without considering whether the trial was a hit or miss.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct mapping: `R == 1` becomes right (1), else left (0). This encodes the **instructed** direction, not the actual lick direction. For miss trials, the actual lick was the opposite of the instructed direction. Two classes only (no "no lick" class since ignore trials were excluded).

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
output_arr = np.vstack([
    np.full(time_vec.size, lick_direction, dtype=np.int64),
    ...
])
```

iii. The AI justified this as following the decoder task specification of "left = 0, right = 1."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`. Autowater trials correspond to the WC (water-cued) context; non-autowater trials are DR (delayed-response).

ii.
```python
"autowater": np.asarray(bp.autowater, dtype=np.float64).reshape(-1),
...
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. From CONVERSION_NOTES.md: "Stored autowater=1 corresponds to WC in the reference code/paper."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct relabelling: `autowater == 0` becomes DR (1), `autowater == 1` becomes WC (0). This matches the requested coding of WC = 0, DR = 1.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. Codes follow the instruction's WC = 0, DR = 1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The per-trial field `bp.hit`. Since ignore trials are excluded, outcome is binary: hit (correct) or miss (incorrect).

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. Since only hit|miss trials are retained (ignore trials filtered out), the outcome is fully determined by the hit flag.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct mapping: `hit == 1` becomes correct (1), otherwise (i.e., miss) becomes incorrect (0). Two classes only, no "ignore" class since those trials were removed during filtering.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
output_arr = np.vstack([
    ...
    np.full(time_vec.size, outcome, dtype=np.int64),
    ...
])
```

iii. The AI justified dropping ignore trials based on the paper: "ignore trials are omitted in the paper."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DeepLabCut tracking for the `tongue` feature from the side camera (view 0) only. The `top_tongue` feature from the bottom camera is not used. The raw fields are `traj[0].ts` (x,y,likelihood), `traj[0].frameTimes`, and the bitcode fields for video offset correction.

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. From CONVERSION_NOTES.md Step 5: "Define tongue velocity from the side-view tongue marker speed magnitude: This uses a direct reference-processed kinematic channel without inventing an unreferenced cross-camera combination."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Raw x,y positions are loaded and the side camera's video offset is subtracted along with the trial's go cue time. (2) Positions are linearly interpolated from frame times to the neural time axis (1000 bins), with NaN positions filtered out before interpolation. (3) Speed is computed as `sqrt(gradient(x)^2 + gradient(y)^2)` on the interpolated (5 ms) time grid. (4) NaN velocities (where tongue was not tracked) are replaced with zero. (5) Speed is thresholded at the session median.

ii.
```python
def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    ...
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)
    # No nearest_fill for tongue

def feature_speed(xpos, ypos, feature_name):
    ...
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    # For tongue: replace NaN with zero
    xv = np.nan_to_num(xv, nan=0.0)
    yv = np.nan_to_num(yv, nan=0.0)
    return np.sqrt(xvel**2 + yvel**2)
```

iii. The AI chose to interpolate positions to the neural time axis before differentiating, following the reference code's `findPosition` / `findVelocity` approach of operating on interpolated positions. NaN tongue velocity is set to zero.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Session median is computed over all finite values of tongue speed across kept trials. Speed >= threshold becomes 1, below becomes 0. No "not visible" class; NaN tongue values were already set to zero before thresholding.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)  # 50th percentile of finite values
...
np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,
).astype(np.int64),
```

iii. Two classes as specified in the decoder task instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The camera's video offset is computed from bitcode synchronization (`sglx.bitcode.bitstart / fs - bp.ev.bitStart`), using the mode of each. Frame times are corrected by subtracting the offset and the trial's go cue time. The positions are then linearly interpolated onto the neural time axis.

ii.
```python
def find_video_offset(raw):
    ...
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])

def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    ...
    old_t = frame_times - vidshift - align_times[trix]
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. This follows the reference `findVideoOffset.m` and `findPosition.m` logic.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Both `top_paw` and `bottom_paw` from the bottom camera (view 1). The speeds from both are averaged.

ii.
```python
needed_by_view = [["tongue"], ["top_paw", "bottom_paw"]]
...
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
paw_stack = np.stack(paw_speeds, axis=0)
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
```

iii. From CONVERSION_NOTES.md Step 5: "Define paw velocity from the average of top- and bottom-paw speed magnitudes in the bottom view: This captures overall paw movement."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw (top and bottom): (1) positions are interpolated to the neural time axis with nearest-fill for NaN gaps, (2) a baseline drift is subtracted (nanmedian of diff), (3) speed is computed as `sqrt(gradient(x)^2 + gradient(y)^2)`, (4) NaN values are filled with nearest. Then the two paw speeds are averaged, and NaN is replaced with zero.

ii.
```python
def feature_speed(xpos, ypos, feature_name):
    ...
    if "tongue" not in feature_name:
        # Subtract baseline drift
        basederiv = np.nanmedian(deriv, axis=0)
        xv = xv - basederiv[0]
        yv = yv - basederiv[1]
        xv = nearest_fill_1d(xv)
        yv = nearest_fill_1d(yv)
    ...

# Average the two paw speeds:
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The AI applied baseline drift subtraction for non-tongue features, following its interpretation of the reference code.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Session median is computed, then speed >= threshold becomes 1, below becomes 0. No "not visible" class; NaN was already replaced with zero.

ii.
```python
paw_thr = summarize_threshold(paw_sel)
...
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
```

iii. Two classes as specified in the decoder task instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, go cue subtraction, then linear interpolation to the neural time axis.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. Same alignment pipeline as all camera-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The separate `motionEnergy_*.mat` files. These contain per-trial motion energy traces (one value per camera frame).

ii.
```python
def load_motion_energy(path):
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    ...
    data = [np.asarray(unwrap_motion_energy_container(v), ...).reshape(-1) for v in flat]
    return data, thresh
```

iii. From CONVERSION_NOTES.md: "Motion energy for ephys sessions is stored separately as motionEnergy_*.mat files."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The raw motion energy trace (one value per camera frame) is linearly interpolated onto the neural time axis using the side camera's frame times (corrected for video offset and go cue alignment). NaN values are then filled using nearest-neighbor interpolation.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    ...
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
    out[:, trix] = nearest_fill_1d(out[:, trix])
    return out
```

iii. From CONVERSION_NOTES.md: "Reference-style interpolation to neural time base with video offset correction."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median of the (nearest-filled) motion energy across kept trials. Values >= threshold become 1, below become 0. No "not visible" class since NaN was filled.

ii.
```python
motion_thr = summarize_threshold(motion_sel)
...
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
```

iii. Two classes as specified in the decoder task instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset correction as other camera streams. Motion energy uses the side camera's frame times, corrected by subtracting the video offset and the trial's go cue time, then linearly interpolated to the neural time axis.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    ...
    old_t = frame_times - vidshift - align_times[trix]
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
```

iii. Same alignment pipeline used for all video-derived streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Multiple strategies: (1) Missing frame times (None or all-NaN): the trial is skipped in `feature_xy` (resulting in all-NaN). (2) NaN DLC positions (low-likelihood frames): filtered out before interpolation in `interp_to_taxis`, so values are interpolated across gaps. (3) For non-tongue features: NaN positions and velocities are filled with nearest-neighbor interpolation. (4) For tongue: NaN velocities are replaced with zero. (5) Motion energy NaN values are filled with nearest-neighbor interpolation, then any remaining NaN replaced with zero. (6) Trials past the last neural recording are dropped. (7) Sessions with fewer than 10 units or fewer than 2 valid trials are skipped entirely.

ii.
```python
def interp_to_taxis(old_t, values, new_t):
    mask = np.isfinite(old_t) & np.isfinite(values)
    if mask.sum() < 2:
        return np.full_like(new_t, np.nan)
    return np.interp(new_t, old_t[mask], values[mask], left=np.nan, right=np.nan)

# Tongue NaN velocity → 0:
xv = np.nan_to_num(xv, nan=0.0)
# Non-tongue NaN → nearest fill:
xv = nearest_fill_1d(xv)
# Motion energy NaN → 0:
motion = np.nan_to_num(motion, nan=0.0)
```

iii. From CONVERSION_NOTES.md: "Missing position values filled with nearest values for all features except tongue" (following the paper's description).

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files dominates runtime. The full conversion runs in ~135 seconds for 44 sessions (~3-4 seconds per session), with file I/O being the bottleneck.

ii.
```python
def load_session(spec):
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. From CONVERSION_NOTES.md: "Full recursive HDF5-to-Python loading was too slow and memory-heavy for the large session files." The AI implemented targeted field loading instead.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop in `convert_one_session` (iterating over units to build neural matrices) and the per-trial loop to construct output arrays could theoretically be vectorized. The per-trial velocity computation in `feature_xy` and `feature_speed` could also be vectorized but is complicated by variable frame counts per trial.

ii.
```python
# Per-unit loop:
for unit in raw["units"]:
    if not good_quality(unit["quality"]):
        continue
    unit_mat = compute_unit_trial_matrix(...)

# Per-trial output construction:
for local_idx, trial_idx in enumerate(selected_trials):
    neural_arr = np.stack(neural_trials[local_idx], axis=0)
    ...
```

iii. The AI used vectorized `np.add.at` for spike accumulation within each unit, avoiding the innermost loop, but retained the outer per-unit and per-trial loops.

## 11-c. What processing does the code repeat multiple times?

i. The video offset `find_video_offset` is called once per feature extraction call rather than being cached per session. In `convert_one_session`, it is called 3 times: once for tongue, once for top_paw, once for bottom_paw (via `feature_xy`), plus once for motion energy (via `aligned_motion_energy`). Each call recomputes the same offset from the same bitcode data.

ii.
```python
# Called in feature_xy:
vidshift = find_video_offset(raw)
# Called again in aligned_motion_energy:
vidshift = find_video_offset(raw)
```

iii. The AI did not explicitly document this redundancy. The computation is cheap, so it doesn't significantly affect runtime.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `continuous` dictionary in session output (containing raw tongue/paw/motion arrays and thresholds) is computed and stored in the session result but is NOT included in the final `build_dataset` output. It is only used for processing plots. Also, `nearest_fill_1d` and `nearest_fill_2d` helper functions are defined but `nearest_fill_2d` is never called. The `bottom_paw` feature is loaded and processed but the reference approach only uses `top_paw`.

ii.
```python
session_out = {
    ...
    "continuous": {  # Only used for plotting, not in final output
        "tongue": tongue_sel.astype(np.float32),
        "paw": paw_sel.astype(np.float32),
        "motion": motion_sel.astype(np.float32),
        ...
    },
}
```

iii. The AI retained the continuous values for diagnostic plotting during the `--show-processing` mode.
