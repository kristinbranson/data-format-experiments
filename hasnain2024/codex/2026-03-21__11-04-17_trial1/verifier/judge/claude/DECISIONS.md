# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the authors' MATLAB loading scripts (`load*_ALMVideo.m`) to discover which sessions and probes to use, then loads each session's `.mat` file. It handles both HDF5 (v7.3) and older MATLAB formats via separate `load_session_hdf5` and `load_session_mat` functions. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`.

ii.
```python
def parse_reference_session_specs(code_dir: Path, data_dir: Path) -> list[SessionSpec]:
    data_files = find_data_files(data_dir)
    specs: list[SessionSpec] = []
    loader_dir = code_dir / "DataLoadingScripts" / "Recording and video"
    for loader in sorted(loader_dir.glob("load*_ALMVideo.m")):
        # ... parses anm, date, probe from each loader script
        if key in data_files:
            specs.append(SessionSpec(...))
    return specs

def load_session(spec: SessionSpec) -> dict:
    return load_session_hdf5(spec) if is_hdf5_mat(spec.session_path) else load_session_mat(spec)
```

iii. From CONVERSION_NOTES.md: "Parses the same loader scripts to recover the exact analyzed session list and probe selection, then loads either v7.3/HDF5 or old-format .mat sessions." The AI dynamically parses the reference loading scripts rather than hard-coding session names.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the session filenames and the parsed loader scripts. The `SessionSpec` dataclass carries the `subject` field. At assembly, unique subject names are collected in encounter order, and each session is mapped to its subject index.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probes: tuple[int, ...]
    ...

# In build_dataset:
if session["subject"] not in subject_names:
    subject_names.append(session["subject"])
subject_index.append(subject_names.index(session["subject"]))
```

iii. From CONVERSION_NOTES.md: "14 unique subjects across retained sessions."

## 1-c. How are the data split into sessions?

i. Each session is one `.mat` file identified by the `<subject>_<date>` key. The AI discovers sessions from both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` folders by matching against the reference loader scripts. Each session becomes one element in the output lists. The result is 44 sessions.

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
```

iii. From CONVERSION_NOTES.md: "Use the intersection of reference-code session lists and available data files. This yields 44 code-selected, paper-consistent ephys sessions."

## 1-d. How are the data split into trials?

i. Trials are identified from the behavioral fields in `obj.bp`. Each entry in the behavioral arrays (hit, miss, R, L, etc.) corresponds to one trial. Trial indices are 0-based in the converted code. The AI reads all behavioral fields as flat arrays and indexes them by trial position.

ii.
```python
# In load_session_mat:
"R": np.asarray(bp.R, dtype=np.float64).reshape(-1),
"hit": np.asarray(bp.hit, dtype=np.float64).reshape(-1),
# ...

# In convert_one_session:
valid = session_valid_trial_mask(raw)
selected_trials = np.flatnonzero(valid)
```

iii. The Bpod table defines one entry per trial, consistent with the reference code.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied: (1) photostimulation trials removed (`stim_enable == 0`), (2) early-lick trials removed (`early == 0`), (3) **only hit and miss trials kept** (`hit == 1 | miss == 1`), dropping ignore trials, (4) only trials where `R == 1 | L == 1`, (5) trials past the last neural recording are dropped. This yields 11,955 trials from the original dataset.

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

iii. From CONVERSION_NOTES.md Step 5: "Exclude stimulation, early-lick, and ignore/no-response trials: Reference analyses consistently use ~stim.enable, ~early, and usually hit/miss conditions; ignore trials are omitted in the paper."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu{probe}`. Each unit carries `trial` (which trial each spike belongs to), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
# In load_session_mat:
units.append({
    "quality": mat_to_str(getattr(unit, "quality", "")),
    "trialtm": np.asarray(getattr(unit, "trialtm"), dtype=np.float64).reshape(-1),
    "trial": np.asarray(getattr(unit, "trial"), dtype=np.int64).reshape(-1),
})
```

iii. Same variables as the reference code's processing pipeline.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue[trial-1]` from `trialtm`. Aligned spikes are binned into 5 ms bins spanning -2.5 to +2.5 s using `np.floor` and `np.add.at` to accumulate spike counts, then divided by `DT` to get firing rates in Hz. The rates are then smoothed using a **causal** Gaussian kernel: a `gausswin(15)` with the first half zeroed out, matching the reference MATLAB `mySmooth.m` with `reflect` boundary conditions.

ii.
```python
def compute_unit_trial_matrix(unit, go_cue, trial_to_pos, n_sel, time_edges):
    aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
    # ... bin assignment with np.floor, accumulation with np.add.at
    mat = np.zeros((n_sel, time_edges.size - 1), dtype=np.float64)
    np.add.at(mat, (trial_pos[keep], bins), 1.0 / DT)
    mat = my_smooth(mat.T, SMOOTH, BCTYPE).T
    return mat

def my_smooth(x, n, bctype="reflect"):
    kern = matlab_gausswin(n)
    kern[:n//2] = 0.0   # causal: zero out first half
    kern /= kern.sum()
    # convolve with reflect padding
```

iii. From CONVERSION_NOTES.md: "Reimplements the reference spike binning and smoothing path in Python, including the causal Gaussian kernel behavior in mySmooth.m."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, units with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (note: `poor` is NOT excluded). Then units with mean firing rate <= 1 Hz are dropped. Sessions with fewer than 10 surviving units are skipped entirely.

ii.
```python
def good_quality(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

# In convert_one_session:
if not good_quality(unit["quality"]):
    continue
if float(unit_mat.mean()) <= LOW_FR_HZ:
    continue
if kept_units < 10:
    return None
```

iii. From CONVERSION_NOTES.md: "findClusters(..., {'all'}) excludes garbage, gabrga, noisy, real?; removeLowFRClusters uses mean FR > lowFR."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting `goCue[trial-1]` from each spike's `trialtm`. The aligned spike times are then binned into the fixed time grid.

ii.
```python
aligned = unit["trialtm"] - go_cue[unit["trial"] - 1]
```

iii. This matches the reference `alignSpikes.m` logic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins (DT = 1/200), spanning -2.5 to +2.5 s from go cue, giving 1000 time bins. No rebinning is applied; spikes are directly counted into this grid.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. Matches the reference `getDefaultParams.m` settings: `params.dt = 1/200`, `params.tmin = -2.5`, `params.tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin centers of the time grid, defined by the conversion parameters. It is not derived from any raw data variable.

ii.
```python
input_trials.append(time_vec[None, :].astype(np.float32))
```

iii. The time axis is constructed from the binning parameters matching the paper's alignment window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — the input is the predefined time axis `time_vec = time_edges[:-1] + DT/2`, the bin centers in seconds from the go cue.

ii.
```python
time_vec = time_edges[:-1] + DT / 2.0
```

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the neural binning grid's bin centers. Both neural data and the input share the same time axis by construction.

ii.
```python
# Neural binning uses time_edges; input uses the corresponding bin centers
time_edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time_vec = time_edges[:-1] + DT / 2.0
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction directly from `bp.R` (the instructed right-side flag). If `R == 1`, direction is right (1); otherwise left (0). It does NOT use `hit`/`miss` to determine the actual lick direction.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
```

iii. From CONVERSION_NOTES.md Step 5: "Lick direction: left 0, right 1; encode as a constant time series."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A direct mapping from `R` to direction: R=1 → right (1), else left (0). Only two classes are defined (left, right). There is no "no lick" class because ignore trials were already filtered out. However, this encodes the **instructed** side, not the **actual** lick direction. On miss trials, the animal licked the opposite port from the instructed side, so this label is incorrect for miss trials.

ii.
```python
lick_direction = np.int64(1 if raw["R"][trial_idx] == 1 else 0)
# output_values: ["left", "right"]
```

iii. The AI's CONVERSION_NOTES.md states "Keep only non-early hit/miss trials so direction is well defined" but does not acknowledge that miss trials require flipping the direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`. When `autowater == 1`, the trial is WC (water-cued); when `autowater == 0`, it is DR (delayed-response).

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
```

iii. Consistent with the reference code and paper.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: `autowater == 0` → DR (1), `autowater == 1` → WC (0). Two classes: WC=0, DR=1.

ii.
```python
context = np.int64(1 if raw["autowater"][trial_idx] == 0 else 0)
# output_values: ["WC", "DR"]
```

iii. Matches the reference mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` only. Hit trials get outcome=1 (correct), miss trials get outcome=0 (incorrect).

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
```

iii. Since ignore trials were filtered out in trial selection, only hit and miss remain.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: hit=1 → correct (1), else incorrect (0). Only two classes are defined (incorrect, correct). There is no "ignore" class because ignore trials were dropped during trial filtering.

ii.
```python
outcome = np.int64(1 if raw["hit"][trial_idx] == 1 else 0)
# output_values: ["incorrect", "correct"]
```

iii. From CONVERSION_NOTES.md: "Exclude stimulation, early-lick, and ignore/no-response trials."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking in `obj.traj`, specifically the `tongue` feature from the side camera (view index 0). Only x and y coordinates are extracted and used. Frame times and the bitcode-derived video offset are used for temporal alignment.

ii.
```python
tongue_pos = feature_xy(raw, 0, "tongue", raw["events"]["goCue"], time_vec)
tongue_speed = feature_speed(*tongue_pos, "tongue")
```

iii. From CONVERSION_NOTES.md Step 5: "Define tongue velocity from the side-view tongue marker speed magnitude."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) DLC x/y positions are **interpolated** onto the neural time axis using `np.interp`. (2) Velocity is computed as the magnitude of `np.gradient` of x and y. (3) For tongue specifically, NaN velocity values are replaced with 0 (not nearest-filled). (4) Timepoints where both x and y positions are NaN are marked as invalid. (5) The continuous speed is thresholded at the session 50th percentile.

ii.
```python
def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    # interpolates x, y onto time_vec
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_to_taxis(old_t, ts[:, 1], taxis)

def feature_speed(xpos, ypos, feature_name):
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    # For tongue: nan_to_num(xv, nan=0.0)
    return np.sqrt(xvel**2 + yvel**2)
```

iii. The AI follows the reference code's `findPosition.m` (interpolation to neural time axis) and `findVelocity.m` (gradient of position) pipeline.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two classes only: 0 (below session 50th percentile) and 1 (at or above). Timepoints where the tongue is not visible (NaN speed) are assigned class 0 ("below_session_median") rather than a separate "not visible" class.

ii.
```python
tongue_thr = summarize_threshold(tongue_sel)  # 50th percentile of finite values

np.where(
    np.isfinite(tongue_sel[:, local_idx]),
    tongue_sel[:, local_idx] >= tongue_thr,
    0,   # not visible → class 0
).astype(np.int64),
# output_values: ["below_session_median", "at_or_above_session_median"]
```

iii. The CONVERSION_NOTES.md does not discuss the lack of a "not visible" class.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from bitcode synchronization: `robust_mode(bitstart) / fs - robust_mode(bp.ev.bitStart)`. Frame times are corrected by subtracting this offset and the trial's go cue time, then x/y positions are interpolated onto the neural time axis via `np.interp`.

ii.
```python
def find_video_offset(raw):
    return robust_mode(bitstart) / fs - robust_mode(raw["events"]["bitStart"])

def feature_xy(raw, view_idx, feature_name, align_times, time_vec):
    old_t = frame_times - vidshift - align_times[trix]
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)
```

iii. This replicates the reference `findVideoOffset.m` and `findPosition.m` pipeline.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from the bottom camera (view index 1), using **both** `top_paw` and `bottom_paw` features. The speeds from both paws are averaged.

ii.
```python
paw_speeds = []
for paw_name in ("top_paw", "bottom_paw"):
    paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
    paw_speeds.append(feature_speed(*paw_pos, paw_name))
paw_stack = np.stack(paw_speeds, axis=0)
```

iii. From CONVERSION_NOTES.md Step 5: "Compute top- and bottom-paw speed magnitudes from x/y velocity pairs, average them per timepoint."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: (1) interpolate x/y positions to neural time axis, (2) compute speed as magnitude of gradient. For paw (non-tongue features), a baseline drift subtraction is applied (median of `np.diff` is subtracted from velocity), and NaN values are nearest-filled. The two paw speeds are averaged where both are available. NaN paw speeds are filled with 0.0 before thresholding.

ii.
```python
# In feature_speed, for non-tongue:
basederiv = np.nanmedian(deriv, axis=0)
xv = xv - basederiv[0]
yv = yv - basederiv[1]
xv = nearest_fill_1d(xv)
yv = nearest_fill_1d(yv)

# Averaging two paws:
paw_speed = np.divide(paw_sum, np.maximum(paw_count, 1), ...)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The baseline subtraction and nearest fill match the reference `findVelocity.m` behavior for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two classes only: 0 (below session 50th percentile) and 1 (at or above). NaN values were already filled with 0.0, so no "not visible" class exists.

ii.
```python
paw_thr = summarize_threshold(paw_sel)
(paw_sel[:, local_idx] >= paw_thr).astype(np.int64),
# output_values: ["below_session_median", "at_or_above_session_median"]
```

iii. No discussion of missing "not visible" class in CONVERSION_NOTES.md.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, alignment to go cue, interpolation onto neural time axis.

ii.
```python
paw_pos = feature_xy(raw, 1, paw_name, raw["events"]["goCue"], time_vec)
```

iii. Same pipeline as all video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motionEnergy_<subject>_<date>.mat` files, loaded via `scipy.io.loadmat`. The motion energy trace has one value per camera frame per trial.

ii.
```python
def load_motion_energy(path):
    me = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    # handles nested struct unwrapping
```

iii. Same source files as the reference code's `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is interpolated from frame times onto the neural time axis using `np.interp`, then nearest-filled to remove NaNs. The continuous values are thresholded at the session 50th percentile.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
    out[:, trix] = nearest_fill_1d(out[:, trix])

# Then:
motion = np.nan_to_num(motion, nan=0.0)
```

iii. From CONVERSION_NOTES.md: "Loads per-trial motion energy, aligns to chosen event using video offset, interpolates onto obj.time, fills missing samples."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two classes only: 0 (below session 50th percentile) and 1 (at or above). NaN values were nearest-filled and then zero-filled, so no "no video" class exists.

ii.
```python
motion_thr = summarize_threshold(motion_sel)
(motion_sel[:, local_idx] >= motion_thr).astype(np.int64),
# output_values: ["below_session_median", "at_or_above_session_median"]
```

iii. No discussion of missing "no video" class in CONVERSION_NOTES.md.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset correction as other video features. Frame times from the side camera are used. Motion energy values are interpolated onto the neural time axis, then nearest-filled.

ii.
```python
def aligned_motion_energy(raw, align_times, time_vec):
    old_t = frame_times - vidshift - align_times[trix]
    out[:, trix] = interp_to_taxis(old_t, me, time_vec)
    out[:, trix] = nearest_fill_1d(out[:, trix])
```

iii. This follows the reference `loadMotionEnergy.m` interpolation approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Missing frame times: if `frame_times` is None or all NaN, a synthetic time axis is created assuming 400 Hz (`(np.arange(ts.shape[0]) + 1.0) / 400.0`). (2) For tongue, NaN speeds are set to 0 and assigned class 0. (3) For non-tongue features, NaN positions and velocities are nearest-filled. (4) Motion energy NaNs are nearest-filled then zero-filled. (5) Sessions with fewer than 10 units or fewer than 2 valid trials are skipped. (6) Trials past the last neural recording are dropped.

ii.
```python
# Synthetic frame times fallback:
if frame_times is None or frame_times.size == 0 or np.all(~np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Tongue NaN → 0:
xv = np.nan_to_num(xv, nan=0.0)

# Non-tongue nearest fill:
xv = nearest_fill_1d(xv)

# Motion energy fill:
out[:, trix] = nearest_fill_1d(out[:, trix])
motion = np.nan_to_num(motion, nan=0.0)
```

iii. From CONVERSION_NOTES.md: "Trials with missing video are kept rather than dropped, since their neural, behavioural data are unaffected."

## 11-a. What are the most time-consuming steps of the code?

i. Loading the `.mat` files dominates runtime, especially HDF5 files which require targeted field extraction. The full conversion runs in approximately 135 seconds for 44 sessions.

ii.
```python
raw = load_session(spec)
# load_session_hdf5 or load_session_mat
```

iii. From CONVERSION_NOTES.md: "Targeted HDF5 field loading instead of recursive object decoding keeps large v7.3 session load times to a few seconds."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial feature extraction loop in `feature_xy` iterates over every trial to interpolate positions. The per-unit neural matrix computation loops over all units. These could potentially be vectorized but each trial has different frame counts, making rectangular operations difficult.

ii.
```python
# Per-trial interpolation loop:
for trix, trial in enumerate(trials):
    xpos[:, trix] = interp_to_taxis(old_t, ts[:, 0], taxis)

# Per-unit accumulation loop:
for unit in raw["units"]:
    unit_mat = compute_unit_trial_matrix(...)
```

iii. The spike accumulation within each unit IS vectorized using `np.add.at`, avoiding per-trial histogram loops.

## 11-c. What processing does the code repeat multiple times?

i. The video offset `find_video_offset(raw)` is recomputed for each call to `feature_xy` and `aligned_motion_energy` — once for tongue, once for top_paw, once for bottom_paw, and once for motion energy (4 times per session). This could be computed once and passed in.

ii.
```python
# In feature_xy:
vidshift = find_video_offset(raw)  # called each time feature_xy is called

# In aligned_motion_energy:
vidshift = find_video_offset(raw)  # called again
```

iii. Not documented as an issue in CONVERSION_NOTES.md.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `bottom_paw` feature is extracted and processed even though the reference only uses `top_paw` — the averaged result differs from what the reference expects. (2) The full `obj` is loaded including many fields never used (e.g., `events.sample`, `events.delay`, `events.reward`, `no`, `L`). (3) The `continuous` output dictionary with raw float speeds and thresholds is computed and stored in the session result but not included in the final pickle output (only used for plotting).

ii.
```python
# Unused loaded fields:
"L": np.asarray(bp.L, dtype=np.float64).reshape(-1),
"no": np.asarray(bp.no, dtype=np.float64).reshape(-1),
"events": {"sample": ..., "delay": ..., "reward": ...},

# Continuous data computed but not in final output:
"continuous": {"tongue": ..., "paw": ..., "motion": ...}
```

iii. Not discussed in CONVERSION_NOTES.md.
