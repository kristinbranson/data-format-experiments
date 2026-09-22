# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Each session is one MATLAB file `data_structure_<anm>_<date>.mat` located in either `Ephys_Behavior` or `RandomizedDelay_Ephys_Behavior`. The 44 session names and probes are hard-coded in `FIXED_SPECS` and `RANDOMIZED_SPECS` lists, transcribed from the authors' loading scripts. Each file is loaded by `load_any_mat`, which checks if the file is HDF5 (using `mat73.loadmat`) or MATLAB v5 (using `scipy.io.loadmat`). Motion energy is loaded from separate `motionEnergy_<anm>_<date>.mat` files via `load_motion_energy`.

ii. Session list definition:
```python
FIXED_SPECS = [
    SessionSpec("EKH1", "2021-08-07", "fixed", (2,)),
    ...
    SessionSpec("JGR3", "2021-11-18", "fixed", (1,)),
]
RANDOMIZED_SPECS = [
    SessionSpec("JEB11", "2022-05-10", "randomized", (1,)),
    ...
    SessionSpec("JEB24", "2023-11-03", "randomized", (1,)),
]
ALL_SPECS = FIXED_SPECS + RANDOMIZED_SPECS
```

Loading:
```python
def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))
```

iii. The AI's CONVERSION_NOTES.md documents that the session list was derived from the reference code's `Recording and video/load*_ALMVideo.m` files, including the probe selection for each session. The AI justified using a hard-coded list (vs globbing) to avoid including sessions the paper excluded.

## 1-b. How are the data split into subjects?

i. The subject (mouse) ID is taken from the `SessionSpec.subject` field, which is set when the session list is defined. Subjects are accumulated in encounter order during `build_dataset`, and `subject_idx` maps each session to its subject index.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    cohort: str
    probes: tuple[int, ...]
```

In `build_dataset`:
```python
if spec.subject not in subject_lookup:
    subject_lookup[spec.subject] = len(subjects)
    subjects.append(spec.subject)
subject_idx.append(subject_lookup[spec.subject])
```

iii. The subject is encoded in the session spec directly. The AI notes that 14 unique subjects are present across the 44 sessions.

## 1-c. How are the data split into sessions?

i. One session = one `SessionSpec` entry in `ALL_SPECS`. Each session corresponds to one `.mat` file. Fixed-delay and randomized-delay sessions are treated uniformly. The result is 44 sessions (25 fixed-delay, 19 randomized-delay), each becoming one element of `neural`, `input`, and `output`.

ii.
```python
for sess_idx, spec in enumerate(session_specs, start=1):
    session_data, _ = process_session(spec, show_processing=...)
    neural.append(session_data["neural"])
    decoder_input.append(session_data["input"])
    decoder_output.append(session_data["output"])
```

iii. The AI follows the reference code session lists exactly, confirmed in CONVERSION_NOTES Step 4.

## 1-d. How are the data split into trials?

i. Trials are defined by `bp.Ntrials` in each session file. All per-trial fields are indexed by trial number. A valid mask is built to select which trials to keep after filtering.

ii.
```python
ntrials = int(float(bp["Ntrials"]))
go_cue = get_bp_array(bp["ev"], "goCue", ntrials, dtype=np.float64)
valid_mask = build_valid_mask(bp, ntrials, go_cue)
valid_idx = np.flatnonzero(valid_mask)
```

iii. The trial count comes from the Bpod table directly.

## 1-e. How are trials filtered based on quality controls?

i. Four filters are applied:
1. Early-lick trials (`bp.early`) are excluded.
2. Photostimulation trials (`bp.stim.enable`) are excluded.
3. Trials must be one of hit/miss/no (i.e., `hit | miss | no` must be true).
4. Trials must have a finite go cue time.
5. Trials past the last recorded neural trial are excluded.

ii.
```python
def build_valid_mask(bp, ntrials, go_cue):
    early = get_bp_array(bp, "early", ntrials, dtype=bool)
    stim = get_stim_enable(bp, ntrials)
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)
    no = get_bp_array(bp, "no", ntrials, dtype=bool)
    outcome = hit | miss | no
    return (~early) & (~stim) & outcome & np.isfinite(go_cue)
```

```python
last_neural_trial = max_recorded_trial(obj, spec.probes)
if last_neural_trial > 0 and last_neural_trial < ntrials:
    valid_mask &= (np.arange(1, ntrials + 1) <= last_neural_trial)
```

iii. The AI documents in CONVERSION_NOTES that early-lick and photostim exclusion follows the paper. The recording-coverage filter was added after discovering all-zero neural trials in two JEB24 sessions. The `outcome` and finite `goCue` requirements are additional safety checks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The spike-sorted clusters from `obj.clu`, specifically each cluster's `trial` (1-based trial assignment), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times `bp.ev.goCue` are also used for alignment.

ii.
```python
trial_ids = as_1d_numeric(probe["trial"][clu_idx], dtype=np.float64)
trial_times = as_1d_numeric(probe["trialtm"][clu_idx], dtype=np.float64)
rates = bin_cluster_rates(trial_ids, trial_times, go_cue, valid_mask)
```

iii. These are the standard spike-sorted cluster fields in the data structure.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5ms bins from -2.5s to +2.5s, converted to firing rates (Hz), then smoothed with a **causal** Gaussian kernel matching the reference `mySmooth(..., 15, 'reflect')`. The smoothing kernel is a `gausswin(15)` with the left half zeroed out, making it causal. Prepend-style reflect boundary handling is used.

ii.
```python
def my_smooth(x, n, bctype="none"):
    ...
    if bctype.lower() == "reflect":
        x_filt = np.concatenate([x[:n, :], x], axis=0)
        trim = n
    ...
    kern = matlab_gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0
    kern /= kern.sum()
    out = np.empty_like(x_filt, dtype=np.float64)
    for col in range(x_filt.shape[1]):
        out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
    out = out[trim:, :]
    ...
```

```python
rates = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The AI states this is a "Python port of the reference MATLAB `mySmooth`" and describes it as a causal Gaussian kernel with reflect boundary handling, matching the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering:
1. Cluster quality labels are checked; clusters with labels in `{'garbage', 'gabrga', 'noisy', 'real?'}` are excluded.
2. Low firing rate filter: units whose mean smoothed firing rate across 4 condition PSTHs (DR-right-hit, DR-left-hit, WC-right-hit, WC-left-hit) is <= 1 Hz are removed.
3. Sessions with fewer than 10 remaining units are excluded (though none were excluded in practice).

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
def cluster_good_mask(qualities):
    labels = [str(q).strip().lower() if q is not None else "" for q in qualities]
    return np.array([label not in QUALITY_EXCLUDE for label in labels], dtype=bool)
```

Low-FR filter using condition PSTHs:
```python
psth_stack = []
for cond_pos in condition_positions:
    if cond_pos.size == 0:
        psth_stack.append(np.zeros(TIME_CENTERS.size, dtype=np.float32))
    else:
        psth_stack.append(rates[:, cond_pos].mean(axis=1))
mean_fr = float(np.mean(np.stack(psth_stack, axis=1)))
if mean_fr > LOW_FR_HZ:
    kept_rates.append(rates)
```

iii. The AI's CONVERSION_NOTES documents that the quality exclusion follows `findClusters.m` which excludes `garbage`, `gabrga`, `noisy`, `real?`. The 1 Hz threshold matches the paper's statement that "all units with firing rates exceeding 1 Hz were included."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike time, then binning the aligned times into 5ms bins.

ii.
```python
aligned = trial_times - go_cue[original_idx]
keep = (
    (kept_pos >= 0)
    & np.isfinite(aligned)
    & (aligned >= TMIN)
    & (aligned < TMAX)
)
...
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This follows the reference `alignSpikes.m` which subtracts per-trial event times from spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5ms bins (DT = 1/200), spanning -2.5s to +2.5s from the go cue, yielding 1000 time bins. No rebinning is applied; this is the native binning resolution.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

iii. Matches the reference `getDefaultParams.m` setting of `dt = 1/200`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is defined by the time bin grid itself. It is the center of each 5ms bin from -2.5s to +2.5s around the go cue.

ii.
```python
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
...
session_input = [TIME_CENTERS[None, :].astype(np.float32).copy() for _ in range(valid_idx.size)]
```

iii. The time axis is determined by the binning parameters, not by any raw data variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing. The time axis is defined by the binning grid parameters.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input IS the neural binning grid (bin centers), so it is inherently aligned with the neural data by construction.

ii.
```python
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The actual lick event times: `bp.ev.lickL` (left lick times) and `bp.ev.lickR` (right lick times), along with `bp.ev.goCue`.

ii.
```python
lick_l = ensure_event_list(bp["ev"].get("lickL", []), ntrials)
lick_r = ensure_event_list(bp["ev"].get("lickR", []), ntrials)
lick_direction = first_post_go_lick_direction(lick_l, lick_r, go_cue)[valid_idx]
```

iii. The AI's CONVERSION_NOTES (Step 5) states: "Define lick direction from actual behavior, not instructed side. First post-go-cue lick side is the cleanest behavioral output and naturally gives `none` for ignore trials."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the code finds the first left and first right lick event within a 3-second response window after the go cue. If a left lick comes first, direction is 0 (left); if right comes first, direction is 1 (right); if no lick occurs, direction is 2 (none).

ii.
```python
RESPONSE_WINDOW_S = 3.0

def first_post_go_lick_direction(lick_l, lick_r, go_cue):
    direction = np.full(go_cue.shape, 2, dtype=np.int8)
    for trix in range(go_cue.size):
        left = lick_l[trix]
        right = lick_r[trix]
        left = left[(left >= go_cue[trix]) & (left < go_cue[trix] + RESPONSE_WINDOW_S)]
        right = right[(right >= go_cue[trix]) & (right < go_cue[trix] + RESPONSE_WINDOW_S)]
        left_first = left[0] if left.size else np.inf
        right_first = right[0] if right.size else np.inf
        if not np.isfinite(left_first) and not np.isfinite(right_first):
            direction[trix] = 2
        elif left_first <= right_first:
            direction[trix] = 0
        else:
            direction[trix] = 1
    return direction
```

iii. The AI chose to use actual lick events rather than deriving direction from the combination of instructed side and hit/miss outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater` — a per-trial flag indicating whether the trial is a water-cued (WC) context trial.

ii.
```python
autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)[valid_idx]
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. Autowater trials map to WC (0), all others to DR (1).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater=True -> WC (0), autowater=False -> DR (1).

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. Straightforward relabeling.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial flags: `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit = get_bp_array(bp, "hit", ntrials, dtype=bool)[valid_idx]
miss = get_bp_array(bp, "miss", ntrials, dtype=bool)[valid_idx]
no = get_bp_array(bp, "no", ntrials, dtype=bool)[valid_idx]
outcome = np.full(valid_idx.size, 2, dtype=np.int8)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. Uses all three flags directly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A relabeling: miss -> incorrect (0), hit -> correct (1), no -> ignore (2).

ii. Same as 6-a code snippet.

iii. The encoding matches the task specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The DLC tracking in `obj.traj[0]` (side camera view), specifically the tracked feature named `"tongue"`. The code extracts `ts` (x, y, likelihood) and `frameTimes` from the side camera view, along with the video offset computed from `sglx.bitcode.bitstart` / `sglx.fs` and `bp.ev.bitStart`.

ii.
```python
tongue_x, tongue_y, tongue_visible = extract_feature_traces(obj, "tongue", 0, go_cue)
```

iii. The AI uses only the side camera view (view index 0) for tongue tracking, unlike the reference which uses both side and bottom camera views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Steps:
1. Extract tongue x,y positions from the side camera's DLC tracking for each trial.
2. Interpolate x,y positions to the bin time centers using `np.interp`, after aligning frame times to the go cue via the video offset.
3. For tongue features, missing values in velocity are set to zero (not nearest-filled).
4. Compute velocity as `np.gradient` of x and y positions, then speed as `sqrt(vx^2 + vy^2)`.
5. Threshold by per-session 50th percentile of visible samples: 0 (below), 1 (above), 2 (not visible).

ii.
```python
def extract_feature_traces(obj, feat_name, view_idx, go_cue):
    ...
    interp_xy = interp_feature(aligned_times, feat_xy, taxis)
    ...

def extract_velocity(xpos, ypos, feat_name, visible=None):
    ...
    xv = np.gradient(trial_xy[:, 0])
    yv = np.gradient(trial_xy[:, 1])
    if "tongue" not in feat_name.lower():
        xv = xv - basederiv[0]
        yv = yv - basederiv[0]
        xv = nearest_fill(xv)
        yv = nearest_fill(yv)
    else:
        xv[~np.isfinite(xv)] = 0.0
        yv[~np.isfinite(yv)] = 0.0
    ...

tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
tongue_cat, tongue_thresh = compute_speed_categories(tongue_speed, tongue_visible)
```

iii. The AI's CONVERSION_NOTES documents that the tongue-specific handling (setting missing velocities to zero) matches the reference code's `findVelocity.m`. However, the AI only uses a single camera view rather than both.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile (median) of visible speed values is computed. Values >= threshold get class 1, below get class 0, not visible gets class 2.

ii.
```python
def compute_speed_categories(speed, visible):
    out = np.full(speed.shape, 2, dtype=np.int8)
    valid_speed = speed[visible & np.isfinite(speed)]
    threshold = float(np.nanpercentile(valid_speed, 50.0))
    out[visible] = (speed[visible] >= threshold).astype(np.int8)
    return out, threshold
```

iii. Matches the task specification for 50th percentile thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from `sglx.bitcode.bitstart/fs` minus `bp.ev.bitStart`, using **median** (not mode). Frame times are corrected by this offset and the trial's go cue, then the positions are **interpolated** to the neural time bin centers using `np.interp`.

ii.
```python
def get_vidshift(obj):
    ...
    bit_file_offset = as_1d_numeric(bitcode.get("bitstart"), dtype=np.float64)
    fs_val = float(np.asarray(fs).reshape(-1)[0])
    return float(np.nanmedian(bit_file_offset) / fs_val - np.nanmedian(bit_start))

def extract_feature_traces(obj, feat_name, view_idx, go_cue):
    ...
    aligned_times = frame_times - vidshift - go_cue[trix]
    interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. The AI uses interpolation to the bin centers rather than binning (averaging) frame values within each bin.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from `obj.traj[1]` (bottom camera view). The code processes **both** `top_paw` and `bottom_paw` features and averages them where both are visible.

ii.
```python
paw_features = []
paw_visible_masks = []
for feat_name in ("top_paw", "bottom_paw"):
    try:
        paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
    except KeyError:
        continue
    ...
    paw_features.append(np.sqrt(paw_vx**2 + paw_vy**2))
    paw_visible_masks.append(paw_visible)
if paw_features:
    paw_stack = np.stack(paw_features, axis=2)
    ...
```

iii. The AI chose to use both paw features and average them, rather than using only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature:
1. Extract x,y from DLC tracking in the bottom camera view.
2. Interpolate to bin time centers.
3. Apply nearest-fill for non-tongue features (fills NaN gaps).
4. Compute velocity via `np.gradient`, subtract median baseline derivative.
5. Speed = sqrt(vx^2 + vy^2).
6. Average speeds from top_paw and bottom_paw where both are visible.
7. Threshold by per-session 50th percentile.

ii.
```python
if "tongue" not in feat_name.lower():
    xpos[:, trix] = nearest_fill(xpos[:, trix])
    ypos[:, trix] = nearest_fill(ypos[:, trix])
...
if "tongue" not in feat_name.lower():
    xv = xv - basederiv[0]
    yv = yv - basederiv[0]
    xv = nearest_fill(xv)
    yv = nearest_fill(yv)
```

iii. The AI applies nearest-fill to paw positions and velocities, following the reference `findVelocity.m` convention where non-tongue features are nearest-filled.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile on visible samples. 0 = below, 1 = above, 2 = not visible.

ii. Uses the same `compute_speed_categories` function.

iii. Matches task specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: video offset computed via median, positions interpolated to bin centers.

ii. Same `extract_feature_traces` and `get_vidshift` functions.

iii. Uses interpolation rather than binning.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<anm>_<date>.mat` files, loaded via `load_motion_energy`. The motion energy trace is one value per camera frame per trial.

ii.
```python
def load_motion_energy(spec):
    payload = load_any_mat(spec.motion_path)
    me = payload.get("me")
    if isinstance(me, dict):
        return me
    ...

def align_motion_energy(obj, spec, go_cue):
    me = load_motion_energy(spec)
    me_data = me["data"]
    if isinstance(me_data, dict) and "data" in me_data:
        me_data = me_data["data"]
    ...
```

iii. The AI handles multiple container layouts (bare cell array, dict with data, doubly-nested dict).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-frame motion energy trace is interpolated to the bin time centers using `np.interp`, after aligning frame times via the video offset and go cue. Then thresholded by per-session 50th percentile.

ii.
```python
aligned[:, trix] = np.interp(
    TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix],
    trace[valid],
    left=np.nan,
    right=np.nan,
)
...
motion_cat, motion_thresh = compute_speed_categories(motion, motion_visible)
```

iii. The AI uses interpolation rather than binning (mean within each bin).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile. 0 = below, 1 = above, 2 = no video.

ii. Same `compute_speed_categories` function.

iii. Matches task specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (`obj.traj[0].frameTimes`), corrected by the video offset (computed via median) and aligned to the go cue. The values are then interpolated to the neural bin centers.

ii.
```python
frame_times = view0.get("frameTimes", [None] * ntrials)[trix]
...
aligned[:, trix] = np.interp(
    TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix],
    trace[valid],
    left=np.nan,
    right=np.nan,
)
```

iii. Same interpolation approach as the DLC features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases:
- **Missing frame times**: When `frameTimes` is NaN or missing, a synthetic frame time series at 400 Hz is generated as a fallback.
- **Untracked frames**: For tongue features, missing velocity values are set to zero. For non-tongue features (paw), nearest-fill interpolation is used on both positions and velocities.
- **Not visible bins**: Bins where the feature is not visible get class 2 in the output.
- **Trials past recording**: Excluded via the `max_recorded_trial` filter.
- **Missing go cue**: Trials with non-finite go cue are excluded.

ii.
```python
if frame_times.size == 1 and np.isnan(frame_times[0]):
    frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0

if "tongue" not in feat_name.lower():
    xpos[:, trix] = nearest_fill(xpos[:, trix])
    ypos[:, trix] = nearest_fill(ypos[:, trix])
```

iii. The AI's CONVERSION_NOTES documents the fallback frame time generation and the paw visibility mask preservation fix.

## 11-a. What are the most time-consuming steps of the code?

i. File loading dominates the runtime. The full conversion takes about 215 seconds for 44 sessions. Loading both the data structure file and the motion energy file accounts for most of each session's processing time.

ii.
```python
obj = normalize_obj(load_any_mat(spec.data_path))
```

iii. The AI notes per-session times of 3-8 seconds, with most of that being I/O.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain:
- Per-trial loop in `extract_feature_traces` for interpolating DLC positions
- Per-trial loop in `align_motion_energy` for interpolating motion energy
- Per-trial loop in `first_post_go_lick_direction`
- Per-column loop in `my_smooth` for convolution
- Per-cluster loop in `process_probe`

The spike binning is vectorized with `np.bincount`. The trial-level DLC interpolation loops could potentially be vectorized if all trials had the same number of frames, but they don't.

ii.
```python
for trix in range(ntrials):
    ...
    interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. The AI notes that vectorized spike binning was implemented as a speedup.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`get_vidshift`) is called multiple times per session - once in `extract_feature_traces` for each feature, and once in `align_motion_energy`. Each call recomputes the same value. Frame time alignment is also repeated for each feature extraction.

ii.
```python
vidshift = get_vidshift(obj)  # called in extract_feature_traces
...
vidshift = get_vidshift(obj)  # called again in align_motion_energy
```

iii. The video offset is a session constant but is recomputed each time it's needed rather than cached.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items:
- `build_condition_positions` constructs condition indices (DR-right-hit, DR-left-hit, WC-right-hit, WC-left-hit) used only for the low-FR filter PSTH computation. This is extra work compared to a simple overall mean rate filter.
- `normalize_obj` recursively normalizes the entire data structure, including fields never used (e.g., `meta`, `ex`, waveform data).
- The `plot_payload` dictionary is always constructed even when `show_processing` is False.
- The `nearest_fill` function is called on paw positions and velocities, adding processing that the reference doesn't do (the reference only bins raw frame-level speeds).

ii.
```python
condition_positions = build_condition_positions(bp, valid_mask)
...
plot_payload = {
    "session_id": spec.session_id,
    ...
}
```

iii. The condition-PSTH approach for firing rate filtering is arguably closer to the reference MATLAB `removeLowFRClusters` behavior, but it's more complex than needed.
