# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from only the `Ephys_Behavior` folder (fixed-delay sessions), not from `RandomizedDelay_Ephys_Behavior`. It hard-codes 12 sessions (the "Figure 8 context analyses" subset from the paper) in `SESSION_SPECS`. Each session's `data_structure_*.mat` file is opened with `h5py` (HDF5/v7.3 format only). Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    # ... 12 sessions total
]

DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")

def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        ...
```

iii. The agent stated: "I've pinned the reference processing path and the likely target cohort: the 12 alternating-context ALM sessions used in the context analyses." The agent chose to use only the subset of sessions from the paper's Figure 8 context analyses rather than all 44 sessions listed in the authors' loading scripts.

## 1-b. How are the data split into subjects?

i. The subject (animal) is taken from the `animal` field of each `SESSION_SPECS` entry. Subjects are tracked in an `OrderedDict` to maintain insertion order, and `subject_idx` maps each session to its subject.

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    sid = spec["animal"]
    if sid not in subject_lookup:
        subject_lookup[sid] = len(subject_lookup)
        subjects.append(sid)
    data["subject_idx"].append(subject_lookup[sid])
```

iii. The agent derives the subject directly from the session specification rather than parsing filenames.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` corresponds to one session. The AI uses only 12 sessions from the `Ephys_Behavior` folder, not from both task folders. Each session becomes one element in the `neural`, `input`, and `output` lists.

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The agent stated it was targeting "the 12 alternating-context ALM sessions used in the context analyses" from the paper.

## 1-d. How are the data split into trials?

i. Trials are identified by the `Ntrials` field of `bp`. Spike trial indices are read from `clu.trial` (converted to 0-based). Behavioral fields (`hit`, `miss`, `early`, etc.) are read per trial.

ii.
```python
def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        "hit": np.asarray(read_numeric_dataset(bp["hit"]), dtype=bool),
        ...
    }
```

iii. The agent uses the standard trial structure from the Bpod data.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials with `analysis_trial_mask`: keeps only `(hit | miss) & ~early & ~no & ~stim_enable`. This excludes early-lick trials, photostim trials, AND ignore/no-response trials. The reference keeps ignore trials.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

iii. The agent stated: "Trials with early licks, no responses, or stimulation enabled are excluded from the decoder dataset."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu{probe}` — specifically the `trial`, `trialtm`, and `quality` fields of each cluster. Go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
```

iii. The agent correctly identified the cluster data structure in the HDF5 files.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to goCue, binned at 10ms (DT=1/100), converted to firing rate (counts/DT), then causally smoothed using a half-zero `gausswin(15)` kernel with reflect boundary conditions. The smoothing is causal: the first half of the Gaussian window is zeroed out.

ii.
```python
DT = 1 / 100
SMOOTH = 15

def my_smooth(x, n, bctype="none"):
    kern = gausswin(n)
    kern[: n // 2] = 0
    kern /= kern.sum()
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

# In bin_spikes_for_session:
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The agent stated it was following "the MATLAB processing path" with "dt=0.01 s, causally smoothed with smooth=15".

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Quality label check — clusters with quality in `{"garbage", "gabrga", "noisy", "real?"}` are dropped (note: does NOT case-normalize, and does NOT drop "poor"). (2) Mean firing rate threshold: the mean of the PSTH (condition-averaged firing rates) must exceed 1 Hz. The PSTH is computed over 7 context conditions.

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

# Firing rate check uses PSTH mean:
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. The agent followed the reference MATLAB code's quality exclusion list. The firing rate threshold is based on the mean PSTH rather than the mean over all trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `bp.ev.goCue[trial]` for each spike. This is the same subtraction as the reference.

ii.
```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The agent correctly identified `goCue` as the alignment event.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10ms bins (DT = 1/100) with a window from -3.0 to +2.5 seconds, yielding 550 time bins. The reference uses 5ms bins with -2.5 to +2.5 seconds yielding 1000 bins.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The agent stated it was using "10 ms bins" following the reference code's `params.dt = 1/200` — but actually misread the value, since `1/200 = 0.005` (5ms), not 10ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself (`TIME_AXIS`), computed from the bin edges. It is the same for every trial.

ii.
```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The agent constructs the time axis from the bin parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing — it is the bin centers of the time grid, computed once and reused for every trial.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis defines the same bins used for spike counting and smoothing, so input and neural data are aligned by construction.

ii.
```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` (right instruction), `bp.hit`, `bp.L`, and `bp.miss`. The lick direction is inferred: right lick = (R & hit) | (L & miss).

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. The agent infers lick direction from the instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The result is a binary variable: 0=left, 1=right. There is NO "no lick" class — ignore trials have already been excluded by the trial mask. The instructions specify three classes (left, right, none), so a class is missing.

ii.
```python
OUTPUT_VALUES = [
    ["left", "right"],
    ...
]
lick_dir = actual_lick_direction(behavior)[use_trials]
```

iii. The agent excluded ignore trials entirely, so it did not need a "no lick" class, but this conflicts with the instruction requirements.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are WC (0), non-autowater are DR (1).

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. The agent correctly identifies autowater as the context marker.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct boolean inversion: `~autowater` gives DR=1, autowater gives WC=0. This matches the reference's encoding.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. Matches reference encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` only. Correct=1 (hit), incorrect=0 (not hit, which must be miss since ignore trials are excluded).

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. Since the agent excluded ignore trials, it only has two outcome classes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A direct cast of the `hit` boolean to integer. Missing the "ignore" class since those trials were dropped. The instructions specify three classes: incorrect, correct, ignore.

ii.
```python
OUTPUT_VALUES = [
    ...
    ["incorrect", "correct"],
    ...
]
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The agent only has two outcome values, missing the ignore class required by instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` — specifically the `tongue` feature from the side camera (view index 1). Only one camera view is used for the tongue. Frame times and the video offset from `sglx.bitcode` are also used.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
```

iii. The agent uses only the side camera view for the tongue, not both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. (1) x,y positions are extracted for frames where the feature is tracked. (2) For tongue, no smoothing is applied (`my_smooth(coords, 1, "reflect")` is a no-op). (3) Positions are linearly interpolated onto the neural time axis. (4) Velocity is computed as `np.gradient` of x and y, then speed = sqrt(xvel^2 + yvel^2). (5) NaN values (invisible tongue) are set to 0. (6) The 50th percentile threshold is computed from positive values only if the overall median is zero.

ii.
```python
if "tongue" not in feature_name:
    coords = my_smooth(coords, 1, "reflect")

# Interpolate onto time axis
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
vals = interp(time_axis)

# Velocity
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)

# Invisible tongue -> zero
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0

# Threshold with special handling
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
```

iii. The agent noted it sets "invisible tongue velocity to zero as in the MATLAB code" and handles the zero-median issue specially.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two categories only: below 50th percentile (0) and above/equal 50th percentile (1). There is NO "not visible" class (category 2). The percentile is computed from positive values only when the overall median is zero.

ii.
```python
OUTPUT_VALUES = [
    ...
    ["lt_p50", "ge_p50"],
    ...
]
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. The agent lacks the "not visible" class required by the instructions.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue, then positions are linearly interpolated onto the neural time axis (`TIME_AXIS`). This differs from the reference which bins frame-level values into 5ms bins.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
vals = interp(time_axis)
```

iii. The agent uses interpolation rather than binning for alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj`, bottom camera (view index 2). The feature is selected by `choose_paw_feature`, which picks `top_paw` first, falling back to `bottom_paw`.

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    ...
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The agent dynamically selects the paw feature from the available tracking data.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. (1) x,y positions are smoothed with `my_smooth(coords, 1, "reflect")` (effectively no smoothing). (2) Positions are linearly interpolated onto the neural time axis. (3) NaN gaps are filled with nearest-neighbor interpolation. (4) Velocity is computed as gradient, with baseline drift subtracted. (5) Speed = sqrt(xvel^2 + yvel^2). (6) Thresholded at 50th percentile.

ii.
```python
if "tongue" not in feature_name:
    coords = my_smooth(coords, 1, "reflect")
    xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])

# Velocity with baseline subtraction
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
```

iii. The agent applies nearest-neighbor filling and baseline drift subtraction for paw, which the reference does not do.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two categories: below 50th percentile (0), above/equal 50th percentile (1). No "not visible" class.

ii.
```python
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. Missing the "not visible" class required by instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: positions are interpolated onto the neural time axis, then NaN-filled with nearest neighbor.

ii.
```python
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
vals = interp(time_axis)
xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
```

iii. Uses interpolation + nearest fill rather than binning.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_*.mat` files, but only from the `Ephys_Behavior` folder (not `RandomizedDelay_Ephys_Behavior`). The `me.data` field is extracted, and `me.moveThresh` is also read (though not used for thresholding).

ii.
```python
def load_motion_energy(animal: str, date: str, behavior: dict) -> tuple[np.ndarray, float]:
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. The agent loads motion energy from the standard MAT files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy (one value per frame) is interpolated onto the neural time axis using linear interpolation, then NaN gaps are filled with nearest-neighbor interpolation. Thresholded at the 50th percentile.

ii.
```python
def align_motion_energy(f, behavior, raw_motion_energy):
    ...
    interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
    aligned[:, trial_idx] = interp(TIME_AXIS)
    aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. The agent uses interpolation with nearest fill rather than simple binning.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two categories: below 50th percentile (0), above/equal 50th percentile (1). No "no video" class.

ii.
```python
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. Missing the "no video" class required by instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times are corrected by the video offset and go cue, then motion energy values are linearly interpolated onto the neural time axis and NaN-filled with nearest neighbor.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
aligned[:, trial_idx] = interp(TIME_AXIS)
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. Same interpolation approach as other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Missing/NaN dropped frames are detected and trials are skipped. (2) For paw and motion energy, NaN gaps after interpolation are filled with nearest-neighbor values. (3) For tongue, NaN values are set to zero. (4) If a trajectory array has unexpected dimensions, the trial is skipped. (5) If a trial has no valid time points for interpolation, it is left as NaN.

ii.
```python
def fill_nearest(x: np.ndarray) -> np.ndarray:
    ...
    if not mask.any():
        return np.zeros_like(x)
    interp = interp1d(idx[mask], x[mask], kind="nearest", ...)
    return interp(idx)

# Tongue invisible -> zero
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
```

iii. The agent fills missing data with nearest-neighbor or zero rather than preserving it as a separate class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files and the per-trial trajectory interpolation are the most expensive steps. Each session's data structure file is opened and parsed via `h5py`, and then every trial's trajectory is interpolated onto the time axis in a Python loop.

ii.
```python
with h5py.File(data_path, "r") as f:
    behavior = load_behavior(f)
    probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
    ...
    tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
```

iii. N/A

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loops over each unique trial per cluster (`for t in unique_trials`), computing a histogram per trial. This could be vectorized with a 2D histogram as the reference does. The trajectory interpolation also loops per trial.

ii.
```python
for t in unique_trials:
    mask = trial_idx == t
    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The per-trial spike binning loop is the most obvious candidate for vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The `get_video_offset` function is called multiple times per session — once in `load_traj_feature_series` for each feature (tongue, paw) and once in `align_motion_energy`. The PSTH computation for firing rate filtering computes condition averages that are only used for the threshold check and then discarded.

ii.
```python
# Called in load_traj_feature_series:
vidshift = get_video_offset(f, behavior)
# Called again in align_motion_energy:
vidshift = get_video_offset(f, behavior)
```

iii. The video offset could be computed once and reused.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code computes 7 context-condition PSTHs per cluster solely to determine the mean firing rate threshold, then discards them. (2) The `manual_motion_thresh` (moveThresh from the motion energy file) is read but never used for thresholding. (3) `single_unit_flags` and quality labels are tracked but not used in the output format. (4) The `get_context_conditions` function computes elaborate condition masks only used for the firing rate check.

ii.
```python
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
```

iii. The PSTH computation for firing rate filtering is the most significant unnecessary processing.
