# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the authors' manifest files (`load*_ALMVideo.m`) to discover sessions programmatically using regex, rather than hard-coding sessions. Each session's `.mat` file is loaded with separate loaders for v7.3 HDF5 (`load_session_v73`) and v5 (`load_session_v5`), determined by `h5py.is_hdf5()`. Motion energy is loaded from separate `motionEnergy_*.mat` files via `load_motion_energy()`.

ii.
```python
def parse_manifest_sessions() -> list[SessionSpec]:
    sessions: list[SessionSpec] = []
    for manifest in sorted(MANIFEST_ROOT.glob("load*_ALMVideo.m")):
        subject = manifest.stem.replace("load", "").replace("_ALMVideo", "")
        lines = [line for line in manifest.read_text().splitlines() if not line.lstrip().startswith("%")]
        text = "\n".join(lines)
        dates = re.findall(r"meta\(end\)\.date = '([^']+)';", text)
        probes = re.findall(r"meta\(end\)\.probe = ([^;]+);", text)
        ...

def load_raw_session(path: Path) -> dict:
    if h5py.is_hdf5(path):
        return load_session_v73(path)
    return load_session_v5(path)
```

iii. From CONVERSION_NOTES Step 1 and Step 5: The AI identified that the authors' `load*_ALMVideo.m` files define the curated session lists and chose to parse them programmatically rather than hard-coding, to match the paper/code selection logic.

## 1-b. How are the data split into subjects?

i. Subject identity is extracted from the session filename (the part before the underscore). At assembly, `subjects` is the sorted set of unique subject names and `subject_idx` maps each session to its subject.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    ...

# In parse_manifest_sessions:
subject = manifest.stem.replace("load", "").replace("_ALMVideo", "")

# In build_dataset:
subjects = sorted({sess["subject"] for sess in processed_sessions})
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
```

iii. From CONVERSION_NOTES Step 4: Subject identity comes from the manifest filenames and session naming convention.

## 1-c. How are the data split into sessions?

i. Each session is one entry from `parse_manifest_sessions()`, identified by `<subject>_<date>`. The function searches both `Ephys_Behavior` and `RandomizedDelay_Ephys_Behavior` directories. This yields 44 sessions (25 fixed-delay + 19 randomized-delay).

ii.
```python
for cohort in ("Ephys_Behavior", "RandomizedDelay_Ephys_Behavior"):
    data_path = DATA_ROOT / cohort / f"data_structure_{subject}_{date}.mat"
    if data_path.exists():
        sessions.append(SessionSpec(...))
        break
```

iii. From CONVERSION_NOTES Step 5, Key Decision 2: "Final neural-session set is 44 unique sessions: 25 fixed-delay ephys sessions from Ephys_Behavior plus 19 randomized-delay sessions from RandomizedDelay_Ephys_Behavior."

## 1-d. How are the data split into trials?

i. Trials come from the behavioral struct `obj.bp`, with `Ntrials` defining the total count. Each trial is indexed by its position in the behavioral arrays. Spike data carries trial indices (`clu.trial`), so no trial boundary reconstruction is needed.

ii.
```python
ntrials = int(np.asarray(bp["Ntrials"][()], dtype=float).reshape(-1)[0])
# ... all per-trial arrays indexed by trial number
```

iii. From CONVERSION_NOTES Step 2: "Native session files contain `obj.bp` with trial labels and event timing fields."

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied. First, early-lick trials are excluded (using `bp.early`). Second, trials requiring a finite go cue time are kept. Additionally, after neural binning, trials whose entire neural matrix is zero are removed (to handle sessions where behavioral trials extend past the recording). Photostimulation trials (`bp.stim.enable`) are NOT excluded.

ii.
```python
candidate_trials = np.flatnonzero(~raw["bp"]["early"] & np.isfinite(raw["bp"]["ev"]["goCue"]))
# ...
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
if nonzero_trial_mask.size and not np.all(nonzero_trial_mask):
    candidate_trials = candidate_trials[nonzero_trial_mask]
    neural_trials = [trial for trial, keep in zip(neural_trials, nonzero_trial_mask.tolist()) if keep]
```

iii. From CONVERSION_NOTES Step 5, Key Decision 4: "Exclude early-lick trials but retain ignore trials: Early trials are omitted throughout the paper's analyses..." The metadata field says `trial_exclusion: early_lick_only`. Photostim exclusion is not mentioned.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu` (spike-sorted clusters), specifically `trial` (1-based trial index for each spike), `trialtm` (spike time relative to trial start), and `quality` (manual curation label). The go cue times `bp.ev.goCue` provide alignment.

ii.
```python
# In load_session_v73:
tm = np.asarray(h5_deref(f, h5_cell_ref(probe_target["tm"], unit_idx))[()], dtype=float).reshape(-1)
trialtm = np.asarray(h5_deref(f, h5_cell_ref(probe_target["trialtm"], unit_idx))[()], dtype=float).reshape(-1)
trial = np.asarray(h5_deref(f, h5_cell_ref(probe_target["trial"], unit_idx))[()], dtype=float).reshape(-1).astype(int)
quality = h5_decode_string_dataset(...)

# In bin_session_neural:
align_times = raw["bp"]["ev"]["goCue"]
aligned = unit["trialtm"] - align_times[trial_idx]
```

iii. From CONVERSION_NOTES Step 1: "Session files contain a MATLAB obj struct with at least: bp (behavior/trials), clu (sorted spikes by probe)."

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue by subtracting `goCue[trial]` from `trialtm`. Spikes are binned into 10 ms bins spanning [-2.5, 2.5] s (500 bins total), converted to firing rate (counts/DT), then smoothed with a **causal** reflected Gaussian kernel with window size 15. The causal kernel zeros out the first half of a `gausswin(15)` window before normalizing.

ii.
```python
DT = 1 / 100  # 10 ms
SMOOTH_N = 15

def smooth_causal_reflect(x, n, bctype="reflect"):
    std = ((n - 1) / 2) / 2.5
    kern = gaussian(n, std=std)
    kern[: len(kern) // 2] = 0.0  # causal: zero out first half
    kern = kern / kern.sum()
    ...

# In bin_session_neural:
bin_idx = np.floor((aligned[mask] - TMIN) / DT).astype(int)
np.add.at(counts, (mapped_trials, bin_idx), 1.0)
fr = smooth_causal_reflect((counts / DT).T, SMOOTH_N, BCTYPE).T
```

iii. From CONVERSION_NOTES Step 4: "Use 10 ms bins for the converted neural time series because this matches the majority of task-analysis scripts, including the fixed-delay/randomized-delay figure scripts and the choice/context decoding pipeline." Step 1 notes: "Tutorial code uses a causal Gaussian kernel with window 15 and boundary condition reflect."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (case-insensitive). Then units with mean firing rate <= 1 Hz are dropped. Sessions with fewer than 10 surviving units are skipped entirely.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

def matlab_quality_ok(quality: str) -> bool:
    return quality.strip().lower() not in QUALITY_EXCLUDE

# In bin_session_neural:
mean_fr = float(np.nanmean(fr))
if mean_fr > LOW_FR_HZ:
    keep_idx.append(unit_idx)
    keep_mats.append(fr.astype(np.float32))

# Session skipping:
if len(neural_trials) < 2 or len(keep_unit_idx) < 10:
    info["skipped"] = True
```

iii. From CONVERSION_NOTES Step 4: "For the decoder conversion, include all non-garbage/non-noisy ALM units that pass >1 Hz, not single-units-only, because the decoder task is closer to 'all other analyses' than to the single-unit selectivity analyses."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times relative to trial start (`trialtm`) are aligned by subtracting the go cue time for that trial. Then spikes in the window [-2.5, 2.5] s are binned.

ii.
```python
trial_idx = unit["trial"] - 1
aligned = unit["trialtm"] - align_times[trial_idx]
...
in_win = (aligned >= TMIN) & (aligned < TMAX)
```

iii. From CONVERSION_NOTES Step 5: "Align everything to go cue and use a common 10 ms grid from -2.5 to 2.5 s."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (`DT = 1/100`), giving 500 bins over the [-2.5, 2.5] s window. No rebinning is applied after the initial binning step.

ii.
```python
DT = 1 / 100  # 10 ms; matches most figure/decoder scripts
```

iii. From CONVERSION_NOTES Step 4: "Use 10 ms bins for the converted neural time series because this matches the majority of task-analysis scripts."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself - bin centers of the uniform grid spanning [-2.5, 2.5] s.

ii.
```python
def build_time_axis() -> np.ndarray:
    edges = np.arange(TMIN, TMAX + DT, DT)
    return edges[:-1] + DT / 2
```

iii. From CONVERSION_NOTES Step 5: "Decoder input is just time-from-go-cue; this is a task-specific export choice built on the same aligned trial grid."

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing - it is the bin-center vector of the uniform time grid, identical for every trial.

ii.
```python
inp = time_axis[np.newaxis, :].astype(np.float32)
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same time grid defines both the neural bin edges and the input values, so they are inherently aligned.

ii.
```python
# Neural uses the same edges:
edges = np.arange(TMIN, TMAX + DT, DT)
# Input uses bin centers of the same edges:
time_axis = edges[:-1] + DT / 2
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. From `bp.R`, `bp.L`, `bp.hit`, `bp.miss`, and `bp.no`. The lick direction is inferred from the instructed side and outcome.

ii.
```python
def lick_direction_value(bp: dict, trial_idx: int) -> int:
    if bp["no"][trial_idx]:
        return 2
    right_choice = (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx])
    left_choice = (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx])
    if left_choice:
        return 0
    if right_choice:
        return 1
    return 2
```

iii. From CONVERSION_NOTES Step 5: "Map actual lick choice: R&hit or L&miss -> right; L&hit or R&miss -> left; no -> none."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A logical combination: hit on an R trial means licked right; miss on an R trial means licked left (wrong side); `no` means no lick. Codes: left=0, right=1, none=2. The value is per-trial, repeated across all time bins.

ii. Same as 4-a above.

iii. From CONVERSION_NOTES Step 5: "This decodes actual behavioral output, not instructed side."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. From `bp.autowater`. Autowater trials are WC context; all others are DR.

ii.
```python
def context_value(bp: dict, trial_idx: int) -> int:
    return 1 if bp["autowater"][trial_idx] else 0
```

iii. From CONVERSION_NOTES Step 5: "autowater==0 -> DR; autowater==1 -> WC."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: DR=0, WC=1. Per-trial, repeated across time bins.

ii. Same as 5-a.

iii. From CONVERSION_NOTES Step 5.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
def outcome_value(bp: dict, trial_idx: int) -> int:
    if bp["miss"][trial_idx]:
        return 0
    if bp["hit"][trial_idx]:
        return 1
    return 2
```

iii. From CONVERSION_NOTES Step 5: "miss -> incorrect; hit -> correct; no -> ignore."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: incorrect=0 (miss), correct=1 (hit), ignore=2 (neither). Per-trial, repeated across time bins.

ii. Same as 6-a.

iii. From CONVERSION_NOTES Step 5.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. From the side-view camera DLC tracking (`tongue` feature in `obj.traj[0]`), including `ts` (x, y, likelihood), `frameTimes`, and the bitcode fields for video offset calculation.

ii.
```python
tongue_x, tongue_y, tongue_vis = align_feature(raw, time_axis, align_times, 0, "tongue")
tongue_speed = compute_speed(tongue_x, tongue_y, "tongue")
```

iii. From CONVERSION_NOTES Step 5: "Raw tongue DLC trajectories from side view."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Frame times are corrected with the video offset and aligned to go cue. (2) x, y positions are interpolated onto the 10ms time grid (with NaN preservation for tongue - gaps are not filled). (3) Speed is computed as `sqrt(gradient(x)^2 + gradient(y)^2)` on the interpolated grid. (4) Discretized at per-session 50th percentile of visible values. (5) Bins where tongue is not visible get class 2.

ii.
```python
x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis)
y = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 1], time_axis)
...
def compute_speed(x, y, feat_name):
    xvel = np.gradient(xx)
    yvel = np.gradient(yy)
    speed[:, tr_idx] = np.sqrt(xvel ** 2 + yvel ** 2)
...
tongue_cat, tongue_thr = discretize_visible_signal(tongue_speed, tongue_vis)
```

iii. From CONVERSION_NOTES Step 5: "Align to go cue using video offset; compute tongue speed from aligned x/y trajectory; per session median threshold over visible timepoints."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. A per-session 50th percentile threshold over visible (finite + tracked) values. Below threshold = 0, at or above = 1, not visible = 2.

ii.
```python
def discretize_visible_signal(values, visible):
    out = np.full(values.shape, 2, dtype=np.int64)
    valid_vals = values[visible & np.isfinite(values)]
    thresh = float(np.nanpercentile(valid_vals, 50))
    low_mask = visible & np.isfinite(values) & (values < thresh)
    high_mask = visible & np.isfinite(values) & ~low_mask
    out[low_mask] = 0
    out[high_mask] = 1
    return out, thresh
```

iii. From the instruction spec: 50th percentile threshold per session.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video offset is computed from the bitcode timing (same as reference `findVideoOffset.m`). Frame times are then corrected by subtracting the video offset and the trial's go cue time. Tongue positions are interpolated onto the same time grid as neural data.

ii.
```python
def find_video_offset(raw):
    bit_start = mode_float(raw["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_float(raw["sglx_bitstart"]) / raw["sglx_fs"]
    return vid_file_offset - bit_start

x = interp_preserve_nan(frame_times - vidshift - align_times[tr_idx], ts[:, 0], time_axis + ADVANCE_MOVEMENT)
```

iii. From CONVERSION_NOTES Step 1: "Alignment to video is explicit: trajectories use frameTimes - vidshift - alignTimes(trial)."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. From both `top_paw` and `bottom_paw` features in the bottom-view camera (`obj.traj[1]`).

ii.
```python
paw_feats = ["top_paw", "bottom_paw"]
paw_speeds = []
paw_vis = []
for feat in paw_feats:
    x, y, vis = align_feature(raw, time_axis, align_times, 1, feat)
    paw_speeds.append(compute_speed(x, y, feat))
    paw_vis.append(vis)
```

iii. From CONVERSION_NOTES Step 5: "Aggregate as max visible paw speed."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) positions are interpolated onto the time grid with nearest-neighbor fill for NaN values (non-tongue features get `fill_nearest_1d`). (2) A baseline drift is subtracted from velocity (median of position differences). (3) Speed is computed as magnitude of x/y gradients. (4) The max speed across the two paws is taken at each time bin. (5) Discretized at per-session 50th percentile.

ii.
```python
# Non-tongue features get nearest fill:
if not is_tongue:
    x = fill_nearest_1d(x)
    y = fill_nearest_1d(y)

# Baseline subtraction for non-tongue:
if not is_tongue:
    diffs = np.column_stack([np.diff(xx), np.diff(yy)])
    basederiv = np.nanmedian(diffs, axis=0)
    xvel = xvel - baseline
    yvel = yvel - baseline
    xvel = fill_nearest_1d(xvel)
    yvel = fill_nearest_1d(yvel)

# Max across paws:
paw_speed = np.max(np.where(paw_finite, paw_stack, -np.inf), axis=0)
```

iii. From CONVERSION_NOTES Step 5: "Aggregate as max visible paw speed; per session median threshold over visible timepoints."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile over visible values. Below = 0, above = 1, not visible = 2.

ii.
```python
paw_cat, paw_thr = discretize_visible_signal(paw_speed, paw_visible)
```

iii. From instruction spec.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same video offset correction as tongue. Paw frame times from the bottom camera are corrected and positions interpolated onto the same time grid.

ii. Same `align_feature` function with `view_idx=1` for bottom camera.

iii. Same alignment logic as all video features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. From separate `motionEnergy_*.mat` files, loaded via `load_motion_energy()`. The function unwraps nested struct layers to get per-trial arrays.

ii.
```python
def load_motion_energy(spec):
    me_path = spec.data_path.with_name(f"motionEnergy_{spec.subject}_{spec.date}.mat")
    me = sio.loadmat(me_path, struct_as_record=False, squeeze_me=True)["me"]
    ...
    while hasattr(payload, "_fieldnames") and "data" in payload._fieldnames:
        ...
        payload = next_payload
```

iii. From CONVERSION_NOTES Step 5: "Motion energy from motionEnergy_*.mat or embedded obj.me."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (already one number per frame) are interpolated onto the neural time grid using `interp_numeric`, then missing values are filled with nearest-neighbor (`fill_nearest_1d`). Discretized at per-session 50th percentile. Bins where no video data exists get class 2.

ii.
```python
sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
aligned[:, tr_idx] = fill_nearest_1d(sig)
...
me_cat, me_thr = discretize_motion_energy(me_aligned)
```

iii. From CONVERSION_NOTES Step 5: "Align/interpolate onto neural time grid as in reference."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile over all finite values. Below = 0, at or above = 1. Class 2 for sessions without motion energy data.

ii.
```python
def discretize_motion_energy(me_aligned):
    ...
    thresh = float(np.nanpercentile(vals, 50))
    out[valid & (me_aligned < thresh)] = 0
    out[valid & (me_aligned >= thresh)] = 1
    return out, thresh
```

iii. From instruction spec.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy frame times come from the side camera's `frameTimes`. After video offset correction and go-cue subtraction, values are interpolated onto the neural time grid, then nearest-filled.

ii.
```python
def align_motion_energy(raw, spec, time_axis, align_times):
    ...
    tt = frame_times - vidshift - align_times[tr_idx]
    sig = interp_numeric(tt, y, time_axis + ADVANCE_MOVEMENT)
    aligned[:, tr_idx] = fill_nearest_1d(sig)
```

iii. From CONVERSION_NOTES Step 1: reference to `loadMotionEnergy.m` which aligns and fills.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials whose entire neural matrix is zero (behavioral trials past the recording) are removed. (2) Missing frame times: if `frameTimes` is empty or all NaN, an artificial time grid is constructed assuming 400 Hz. (3) Non-tongue features with NaN positions are nearest-neighbor filled before velocity computation. (4) Tongue NaN positions are preserved (not filled), and untracked bins get class 2. (5) Motion energy NaN values are nearest-neighbor filled. (6) Sessions with fewer than 10 neurons or 2 trials after filtering are skipped entirely.

ii.
```python
# Artificial frame times fallback:
if frame_times.size == 0:
    frame_times = (np.arange(ts.shape[0]) + 1) / 400.0

# Nearest fill for non-tongue:
if not is_tongue:
    x = fill_nearest_1d(x)
    y = fill_nearest_1d(y)

# Zero-neural trial removal:
nonzero_trial_mask = np.array([np.any(trial != 0) for trial in neural_trials], dtype=bool)
```

iii. From CONVERSION_NOTES Step 10: "Two JEB24 sessions contained invalid late trials with no neural spikes in any selected unit. Resolution: removed all-zero neural trials."

## 11-a. What are the most time-consuming steps of the code?

i. File loading dominates. Mean session processing time is 3.69s, total ~162s for 44 sessions. The HDF5 v7.3 loader is particularly expensive because it dereferences many HDF5 object references.

ii. From conversion_full_out.txt: "Mean session processing time: 3.69s"

iii. From CONVERSION_NOTES Step 7: "Keeps mean runtime at 2.95 s/session."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops remain: (1) Per-trial loops in `align_feature` for interpolating each trial's video data. (2) Per-unit loop in `bin_session_neural` for spike binning. (3) Per-trial loop in `align_motion_energy`. These are difficult to vectorize because trials have different numbers of camera frames.

ii.
```python
for tr_idx in range(ntrials):
    # ... interpolation per trial in align_feature

for unit_idx, unit in enumerate(units):
    # ... spike binning per unit in bin_session_neural
```

iii. Not explicitly discussed in CONVERSION_NOTES.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed fresh each time `align_feature` and `align_motion_energy` are called, rather than once per session. The `find_video_offset` function is called separately for tongue, paw (x2), and motion energy alignment. Also, `parse_manifest_sessions()` is called multiple times (once for session selection, once for printing, once at end).

ii.
```python
# In align_feature:
vidshift = find_video_offset(raw)
# In align_motion_energy:
vidshift = find_video_offset(raw)
```

iii. Not discussed in CONVERSION_NOTES.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items: (1) The loader reads many fields not used in conversion (e.g., `lickL`, `lickR`, `sample`, `delay`, `reward` events, `site`/`channel` for units, `NdroppedFrames`). (2) Both `top_paw` and `bottom_paw` are fully processed even though a single paw might suffice. (3) The `fill_nearest_1d` function is applied to motion energy and paw data - the filled values may not be meaningful and could introduce artifacts. (4) Baseline drift subtraction is computed for paw velocity but the reference code does not do this.

ii.
```python
# Unused fields loaded:
"lickL": [h5_read_cell_numeric_1d(f, ev["lickL"], i) for i in range(ntrials)],
"lickR": [h5_read_cell_numeric_1d(f, ev["lickR"], i) for i in range(ntrials)],
"sample": ev["sample"][()].reshape(-1),
```

iii. Not discussed in CONVERSION_NOTES.
