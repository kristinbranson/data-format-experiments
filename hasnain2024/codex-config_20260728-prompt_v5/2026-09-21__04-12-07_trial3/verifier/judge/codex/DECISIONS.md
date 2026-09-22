# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded the included sessions as `SessionSpec` objects in two lists (`FIXED_SPECS` and `RANDOMIZED_SPECS`), then loaded each session by opening `spec.data_path`. It chose the MATLAB reader by file format: HDF5/v7.3 files through `mat73`, other `.mat` files through `scipy.io.loadmat`, then normalized the result into a shared Python dict layout.

ii. 
```python
FIXED_SPECS = [
    SessionSpec("EKH1", "2021-08-07", "fixed", (2,)),
    ...
]
RANDOMIZED_SPECS = [
    SessionSpec("JEB11", "2022-05-10", "randomized", (1,)),
    ...
]
ALL_SPECS = FIXED_SPECS + RANDOMIZED_SPECS

def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))

obj = normalize_obj(load_any_mat(spec.data_path))
```

iii. In `CONVERSION_NOTES.md` Step 6, the agent says it "hard-coded the reference ALM session/probe manifest" and added mixed MATLAB-loader support because the dataset contains both v7.3/HDF5 and v5 `.mat` files.

## 1-b. How are the data split into subjects?

i. Subjects are split by the `subject` field in each hard-coded `SessionSpec`, which matches the animal ID prefix in the session name. During assembly, the code builds `subjects` in first-seen order and stores a parallel `subject_idx`.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    cohort: str
    probes: tuple[int, ...]

if spec.subject not in subject_lookup:
    subject_lookup[spec.subject] = len(subjects)
    subjects.append(spec.subject)
subject_idx.append(subject_lookup[spec.subject])
```

iii. `CONVERSION_NOTES.md` Step 5 says animal/session identity should come from filename/session metadata and be mapped into `subjects` and `subject_idx`.

## 1-c. How are the data split into sessions?

i. One `SessionSpec` is one session. Each becomes one entry in `neural`, `input`, and `output`. Fixed-delay sessions are loaded from `Ephys_Behavior`; randomized-delay sessions are loaded from `RandomizedDelay_Ephys_Behavior`.

ii.
```python
@property
def data_dir(self) -> Path:
    return EPHYS_DIR if self.cohort == "fixed" else RAND_DIR

for sess_idx, spec in enumerate(session_specs, start=1):
    session_data, _ = process_session(spec, ...)
    neural.append(session_data["neural"])
    decoder_input.append(session_data["input"])
    decoder_output.append(session_data["output"])
```

iii. In Step 6 notes, the agent explicitly says it implemented 25 fixed-delay and 19 randomized-delay sessions from the reference manifest.

## 1-d. How are the data split into trials?

i. Trials are indexed directly by `bp["Ntrials"]` and trial-numbered arrays in `bp.ev`, `traj`, and `clu`. The code builds a boolean `valid_mask`, converts it to `valid_idx`, then slices all per-trial signals with that index list.

ii.
```python
ntrials = int(float(bp["Ntrials"]))
valid_mask = build_valid_mask(bp, ntrials, go_cue)
valid_idx = np.flatnonzero(valid_mask)

session_input = [TIME_CENTERS[None, :].astype(np.float32).copy()
                 for _ in range(valid_idx.size)]
```

iii. The notes describe this as using the native trial structure in `obj.bp`, `obj.traj`, and `obj.clu` rather than reconstructing trial boundaries.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding `early` trials, photostimulation trials (`stim.enable`), trials without a defined outcome flag (`hit`, `miss`, or `no`), trials with non-finite `goCue`, and trials beyond the last neural trial covered by the selected probe(s).

ii.
```python
def build_valid_mask(bp: dict[str, Any], ntrials: int, go_cue: np.ndarray) -> np.ndarray:
    early = get_bp_array(bp, "early", ntrials, dtype=bool)
    stim = get_stim_enable(bp, ntrials)
    hit = get_bp_array(bp, "hit", ntrials, dtype=bool)
    miss = get_bp_array(bp, "miss", ntrials, dtype=bool)
    no = get_bp_array(bp, "no", ntrials, dtype=bool)
    outcome = hit | miss | no
    return (~early) & (~stim) & outcome & np.isfinite(go_cue)

last_neural_trial = max_recorded_trial(obj, spec.probes)
if last_neural_trial > 0 and last_neural_trial < ntrials:
    valid_mask &= (np.arange(1, ntrials + 1) <= last_neural_trial)
```

iii. Step 5 notes say the plan was to exclude `early` and `stim.enable` trials, keep `hit`/`miss`/`no` trials for the outcome target, and later Step 9 notes justify the `last_neural_trial` cutoff as a fix for tail trials with all-zero neural activity.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The final neural data come from `obj["clu"][probe_idx]["trial"]` and `obj["clu"][probe_idx]["trialtm"]`, with `bp["ev"]["goCue"]` used for alignment and cluster `quality` labels used for QC.

ii.
```python
trial_ids = as_1d_numeric(probe["trial"][clu_idx], dtype=np.float64)
trial_times = as_1d_numeric(probe["trialtm"][clu_idx], dtype=np.float64)
rates = bin_cluster_rates(trial_ids, trial_times, go_cue, valid_mask)
```

iii. The Step 5 mapping table says the neural source is ALM probe spike times aligned to `bp.ev.goCue` after quality and rate filtering.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue, restricted to `[-2.5, 2.5)` seconds, binned at 5 ms, converted to firing rate by dividing by `DT`, and smoothed with a Python port of the reference `mySmooth(..., 15, 'reflect')` causal Gaussian kernel.

ii.
```python
aligned = trial_times - go_cue[original_idx]
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
counts = counts_flat.reshape(TIME_CENTERS.size, n_valid).astype(np.float64)
rates = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The code header and Step 6 notes both say the intent was to match the reference 5 ms go-cue-aligned neural pipeline and the causal Gaussian smoothing used by MATLAB `mySmooth`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code excludes clusters whose `quality` lower-cases to one of `{"garbage", "gabrga", "noisy", "real?"}`. It then keeps only units whose mean firing rate, computed as the mean of four condition-averaged PSTHs, exceeds 1 Hz. It also aborts a session if fewer than 10 units remain.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
MIN_UNITS_PER_SESSION = 10

def cluster_good_mask(qualities: list[Any]) -> np.ndarray:
    labels = [str(q).strip().lower() if q is not None else "" for q in qualities]
    return np.array([label not in QUALITY_EXCLUDE for label in labels], dtype=bool)

mean_fr = float(np.mean(np.stack(psth_stack, axis=1)))
if mean_fr > LOW_FR_HZ:
    kept_rates.append(rates)
```

iii. Step 6 notes say the agent intended to "exclude cluster qualities `garbage`, `gabrga`, `noisy`, `real?`", "remove units with mean FR `<= 1 Hz`", and "exclude sessions with fewer than 10 remaining units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that spike's trial go-cue time, i.e. `trialtm - goCue[trial]`.

ii.
```python
original_idx = trial_ids.astype(np.int64) - 1
aligned = trial_times - go_cue[original_idx]
```

iii. The code header and notes repeatedly describe go-cue alignment as a direct match to the reference `alignSpikes` logic.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use a fixed 5 ms bin size (`DT = 1/200`) across a `-2.5` to `+2.5` s window, yielding 1000 bins. There is no later rebinning.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 1.0 / 200.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
```

iii. Step 4 notes explicitly resolve the paper/code discrepancy in favor of 5 ms bins.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw variable. The input is a synthetic shared time axis (`TIME_CENTERS`) defined relative to go cue using the chosen `TMIN`, `TMAX`, and `DT`.

ii.
```python
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
session_input = [TIME_CENTERS[None, :].astype(np.float32).copy()
                 for _ in range(valid_idx.size)]
```

iii. Step 5 notes say the shared time grid relative to go cue should be the sole decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code simply constructs bin centers from the chosen window and bin width, then repeats that `1 x T` array for every trial.

ii.
```python
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
session_input = [TIME_CENTERS[None, :].astype(np.float32).copy()
                 for _ in range(valid_idx.size)]
```

iii. The Step 10 sanity checks say the expected input was the shared bin-center axis `[-2.4975, ..., 2.4975]`.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same time grid used for neural binning, so input bin `k` corresponds to neural bin `k`.

ii.
```python
TIME_CENTERS = (EDGES[:-1] + EDGES[1:]) / 2.0
rates = my_smooth(counts / DT, SMOOTH, "reflect")
session_input = [TIME_CENTERS[None, :].astype(np.float32).copy()
                 for _ in range(valid_idx.size)]
```

iii. Step 10 notes explicitly say the input check compared the converted input to the shared neural bin-center axis and matched with `np.allclose`.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The agent derives lick direction from the actual post-go-cue lick event times `bp.ev.lickL` and `bp.ev.lickR`, together with `goCue`, rather than from instructed side plus hit/miss.

ii.
```python
lick_l = ensure_event_list(bp["ev"].get("lickL", []), ntrials)
lick_r = ensure_event_list(bp["ev"].get("lickR", []), ntrials)
lick_direction = first_post_go_lick_direction(lick_l, lick_r, go_cue)[valid_idx]
```

iii. Step 5 notes justify this as using the "actual first post-go-cue lick side" because it is the cleanest behavioral output and naturally yields `none` for ignore trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the code looks at left and right licks in the 3 s response window after go cue. If neither occurs, the class is `2` (`none`); otherwise it assigns `0` for whichever side licked first if left wins, `1` if right wins.

ii.
```python
left = left[(left >= go_cue[trix]) & (left < go_cue[trix] + RESPONSE_WINDOW_S)]
right = right[(right >= go_cue[trix]) & (right < go_cue[trix] + RESPONSE_WINDOW_S)]
left_first = left[0] if left.size else np.inf
right_first = right[0] if right.size else np.inf
...
elif left_first <= right_first:
    direction[trix] = 0
else:
    direction[trix] = 1
```

iii. The justification in Step 5 is again that actual first lick is a more direct behavioral readout than inferring lick direction from task side and outcome.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp["autowater"]`.

ii.
```python
autowater = get_bp_array(bp, "autowater", ntrials, dtype=bool)[valid_idx]
```

iii. Step 5 notes map `bp.autowater` directly to the behavioral-context output.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater == True` to `0` (`WC`) and everything else to `1` (`DR`).

ii.
```python
context = np.where(autowater, 0, 1).astype(np.int8)
```

iii. The notes say DR-only sessions should remain valid with a constant DR context label.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes from the per-trial boolean fields `bp["hit"]`, `bp["miss"]`, and `bp["no"]`.

ii.
```python
hit = get_bp_array(bp, "hit", ntrials, dtype=bool)[valid_idx]
miss = get_bp_array(bp, "miss", ntrials, dtype=bool)[valid_idx]
no = get_bp_array(bp, "no", ntrials, dtype=bool)[valid_idx]
```

iii. Step 5 notes say the converter should keep `hit`, `miss`, and `no` trials so outcome can include correct / incorrect / ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is encoded as `0` for miss/incorrect, `1` for hit/correct, and `2` for no-response/ignore.

ii.
```python
outcome = np.full(valid_idx.size, 2, dtype=np.int8)
outcome[miss] = 0
outcome[hit] = 1
outcome[no] = 2
```

iii. The notes say early trials were excluded entirely, while ignore trials were kept as an explicit class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The code derives tongue velocity from the side-camera trajectory only: `obj["traj"][0]["ts"]`, `featNames`, and `frameTimes` for the `"tongue"` feature. It also uses `sglx.bitcode.bitstart`, `sglx.fs`, `bp.ev.bitStart`, and `bp.ev.goCue` for video/behavior alignment.

ii.
```python
tongue_x, tongue_y, tongue_visible = extract_feature_traces(obj, "tongue", 0, go_cue)

feat_idx = find_feat_index(view, feat_name)
vidshift = get_vidshift(obj)
ts = np.asarray(view["ts"][trix], dtype=np.float64)
frame_times = view.get("frameTimes", [None] * ntrials)[trix]
aligned_times = frame_times - vidshift - go_cue[trix]
```

iii. The notes justify preserving explicit tongue missingness, but do not give a detailed written defense for choosing only the side-view tongue feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code linearly interpolates tongue x/y coordinates to the 5 ms neural time grid, marks bins visible when both interpolated coordinates are finite, computes `np.gradient` separately on x and y, replaces non-finite tongue gradients with zero, and defines speed as `sqrt(vx^2 + vy^2)`. It does not smooth x/y and does not use the bottom-view tongue.

ii.
```python
interp_xy = interp_feature(aligned_times, feat_xy, taxis)
visible[:, trix] = np.isfinite(interp_xy[:, 0]) & np.isfinite(interp_xy[:, 1])

xv = np.gradient(trial_xy[:, 0])
yv = np.gradient(trial_xy[:, 1])
...
xv[~np.isfinite(xv)] = 0.0
yv[~np.isfinite(yv)] = 0.0

tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. In Step 5/6 notes, the agent says it wanted "reference interpolation/velocity logic" but also to preserve explicit `not visible` states rather than silently imputing them into low/high movement classes.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Visible tongue-speed samples are median-split within each session. Invisible samples become class `2`.

ii.
```python
def compute_speed_categories(speed: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, float]:
    out = np.full(speed.shape, 2, dtype=np.int8)
    valid_speed = speed[visible & np.isfinite(speed)]
    threshold = float(np.nanpercentile(valid_speed, 50.0))
    out[visible] = (speed[visible] >= threshold).astype(np.int8)
    return out, threshold
```

iii. The Step 5 key decisions say movement outputs should be thresholded by per-session medians on valid samples and use class `2` for not visible / no video.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The side-camera frame times are aligned by subtracting a video-behavior offset (`vidshift`) and then subtracting per-trial `goCue`. The tongue positions are then interpolated directly onto the neural 5 ms time grid `TIME_CENTERS`.

ii.
```python
def get_vidshift(obj: dict[str, Any]) -> float:
    bit_file_offset = as_1d_numeric(bitcode.get("bitstart"), dtype=np.float64)
    fs_val = float(np.asarray(fs).reshape(-1)[0])
    return float(np.nanmedian(bit_file_offset) / fs_val - np.nanmedian(bit_start))

aligned_times = frame_times - vidshift - go_cue[trix]
interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. Step 10 notes say the motion-energy sanity check used the same `vidshift` formula and common neural time axis, and the Step 5 plan explicitly chose go-cue alignment for all outputs.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two tracked paw features in the bottom camera, `"top_paw"` and `"bottom_paw"`, both taken from `obj["traj"][1]`.

ii.
```python
for feat_name in ("top_paw", "bottom_paw"):
    try:
        paw_x, paw_y, paw_visible = extract_feature_traces(obj, feat_name, 1, go_cue)
```

iii. The Step 7 notes mention a "two paw markers" averaging path and a visibility bug that had to be fixed, which shows the agent intentionally used both paw traces rather than one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw marker, the code interpolates x/y to the 5 ms grid, nearest-fills missing non-tongue positions, computes `np.gradient`, subtracts the median x-difference baseline from both x and y gradients, nearest-fills the resulting velocities, converts each marker to speed, and averages speeds across the two paw markers wherever at least one marker is visible.

ii.
```python
if "tongue" not in feat_name.lower():
    xpos[:, trix] = nearest_fill(xpos[:, trix])
    ypos[:, trix] = nearest_fill(ypos[:, trix])

xv = np.gradient(trial_xy[:, 0])
yv = np.gradient(trial_xy[:, 1])
if "tongue" not in feat_name.lower():
    xv = xv - basederiv[0]
    yv = yv - basederiv[0]
    xv = nearest_fill(xv)
    yv = nearest_fill(yv)

paw_weighted = np.where(paw_vis_stack, paw_stack, 0.0)
paw_counts = paw_vis_stack.sum(axis=2)
paw_speed = np.divide(paw_weighted.sum(axis=2), paw_counts, ...)
```

iii. Step 5/6 notes say the agent wanted to use reference interpolation/velocity logic while preserving an explicit not-visible class before thresholding.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Visible paw-speed samples are median-split within each session; invisible bins become class `2`.

ii.
```python
paw_cat, paw_thresh = compute_speed_categories(paw_speed, paw_visible)
```

iii. Step 5 says all movement outputs should use per-session median thresholds on valid samples.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The paw coordinates use the same video offset calculation as the tongue and are interpolated onto the same neural `TIME_CENTERS` grid after subtracting `goCue`.

ii.
```python
vidshift = get_vidshift(obj)
aligned_times = frame_times - vidshift - go_cue[trix]
interp_xy = interp_feature(aligned_times, feat_xy, taxis)
```

iii. The justification is the same as for tongue and motion energy: the Step 5 plan chose one shared go-cue-centered time axis for neural and video outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from the per-session `motionEnergy_<subject>_<date>.mat` file, specifically its `me["data"]` trace, with side-camera frame times from `obj["traj"][0]["frameTimes"]` and the same `vidshift` / `goCue` alignment variables.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> dict[str, Any] | None:
    payload = load_any_mat(spec.motion_path)
    me = payload.get("me")
    ...
    return {"data": me.tolist(), "moveThresh": np.nan}

frame_times = view0.get("frameTimes", [None] * ntrials)[trix] if view0 else None
aligned[:, trix] = np.interp(
    TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix],
    trace[valid],
    left=np.nan,
    right=np.nan,
)
```

iii. Step 5 notes explicitly map the motion-energy output to `motionEnergy_*.mat` when available, with fallback to explicit missing/no-video handling.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code unwraps several possible `me` container layouts, takes one scalar trace per trial, aligns it with `vidshift` and `goCue`, and linearly interpolates it onto the neural 5 ms grid. It does not smooth or differentiate the trace before thresholding.

ii.
```python
me_data = me["data"]
if isinstance(me_data, dict) and "data" in me_data:
    me_data = me_data["data"]
...
aligned[:, trix] = np.interp(
    TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix],
    trace[valid],
    left=np.nan,
    right=np.nan,
)
```

iii. The notes justify this by referencing the original MATLAB motion-energy loader, which aligns/interpolates motion energy to the shared time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Valid motion-energy samples are median-split per session; bins with no valid aligned motion-energy sample become class `2` (`no_video`).

ii.
```python
motion_visible = np.isfinite(motion)
motion_cat, motion_thresh = compute_speed_categories(motion, motion_visible)
```

iii. Step 5 says all continuous movement outputs should use per-session medians and explicit missingness classes.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy traces are aligned with the same `vidshift` and `goCue` correction as other video-derived variables, then interpolated directly onto the neural `TIME_CENTERS` grid using side-camera frame times.

ii.
```python
vidshift = get_vidshift(obj)
frame_times = view0.get("frameTimes", [None] * ntrials)[trix] if view0 else None
aligned[:, trix] = np.interp(
    TIME_CENTERS,
    frame_times[valid] - vidshift - go_cue[trix],
    trace[valid],
    left=np.nan,
    right=np.nan,
)
```

iii. The notes describe this as aligning/interpolating motion energy to the common neural time base.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles missing data by mixing explicit missingness classes with imputation. If frame times are missing or NaN, it fabricates a 400 Hz frame-time vector. For non-tongue features, it nearest-fills missing positions and velocities. For tongue features, missing gradients are forced to zero. Motion energy with too few valid samples remains NaN and later becomes class `2`. Missing/untracked movement bins are ultimately encoded as class `2`.

ii.
```python
if frame_times is None:
    frame_times = np.arange(1, feat_xy.shape[0] + 1, dtype=np.float64) / 400.0
...
if "tongue" not in feat_name.lower():
    xpos[:, trix] = nearest_fill(xpos[:, trix])
    ypos[:, trix] = nearest_fill(ypos[:, trix])
...
xv[~np.isfinite(xv)] = 0.0
yv[~np.isfinite(yv)] = 0.0
...
out = np.full(speed.shape, 2, dtype=np.int8)
```

iii. The Step 5/6 notes justify preserving explicit `not visible` / `no video` outputs, while the Step 1 notes cite the reference rule that tongue missing values are handled differently from non-tongue features.

## 11-a. What are the most time-consuming steps of the code?

i. The agent's documented view is that session loading dominates runtime, with spike binning and movement interpolation as secondary costs. The code was written to keep per-session processing moderate by vectorizing spike binning.

ii.
```python
def load_any_mat(path: Path) -> dict[str, Any]:
    if h5py.is_hdf5(path):
        return mat73.loadmat(str(path))
    return convert_mat_struct(loadmat(path, squeeze_me=True, struct_as_record=False))

counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
```

iii. Step 6 says a naive cluster-by-trial histogram loop would have been too slow; Step 7/9 runtime notes say file loading was the main remaining cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code already vectorizes spike binning with `np.bincount`, but it still loops over trials in `extract_feature_traces`, `extract_velocity`, and output assembly, and loops over neurons in `build_session_neural`. The agent's notes present the avoided nested spike/trial loops as the main vectorization win.

ii.
```python
for trix in range(ntrials):
    ...

for trix in range(valid_idx.size):
    per_trial = np.vstack([...])

counts_flat = np.bincount(flat_idx, minlength=TIME_CENTERS.size * n_valid)
```

iii. Step 6 notes say the intended speedup was to avoid Python-level nested spike/trial loops and repeated re-binning of spikes.

## 11-c. What processing does the code repeat multiple times?

i. The notes claim the code tries to avoid repeated movement interpolation and spike re-binning, but in the implementation some work is repeated: `get_vidshift(obj)` is recomputed inside each `extract_feature_traces` call and inside `align_motion_energy`, feature interpolation happens separately for tongue, top paw, and bottom paw, and `build_session_neural` copies each neuron's trial data again into per-trial matrices.

ii.
```python
def extract_feature_traces(...):
    vidshift = get_vidshift(obj)
    for trix in range(ntrials):
        ...

def align_motion_energy(...):
    vidshift = get_vidshift(obj)
    for trix in range(ntrials):
        ...

for neuron_idx, rates in enumerate(kept_rates):
    trial_mat[neuron_idx, :] = rates[:, trix]
```

iii. Step 6 notes explicitly say the intent was to do "single-pass motion / kinematic interpolation per feature" and to reuse rate matrices for filtering, but the code still repeats some per-feature/per-trial work.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads and normalizes the entire `obj` tree even though only part of it is used, builds condition-position tables only to compute the firing-rate filter, and always constructs a large `plot_payload` summary dict even when `show_processing` is false. The notes mostly acknowledge extra loading of unused raw fields.

ii.
```python
obj = normalize_obj(load_any_mat(spec.data_path))
condition_positions = build_condition_positions(bp, valid_mask)

plot_payload = {
    "session_id": spec.session_id,
    ...
    "motion_cat": motion_cat.T,
}
if show_processing:
    plot_processing(plot_payload)
```

iii. Step 6/11 notes frame the main unnecessary work as broad loading/normalization of unused raw fields and extra plotting support for inspection.
