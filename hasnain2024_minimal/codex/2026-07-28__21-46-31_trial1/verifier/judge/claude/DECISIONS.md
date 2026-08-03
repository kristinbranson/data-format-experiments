# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from a hardcoded `SESSION_SPECS` list, corresponding to the Figure 8 context analyses subset of the paper. It only looks in the `Ephys_Behavior` folder and does not include the `RandomizedDelay_Ephys_Behavior` folder. Each session's `.mat` file is opened with `h5py` (v7.3 HDF5 format). Motion energy files are loaded separately with `scipy.io.loadmat` from the same folder.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    ...  # 12 sessions total
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

iii. The AI explicitly chose the Figure 8 context analyses subset: "The paper statistics say the main dataset should be 12 two-context ephys sessions from 6 mice" (trajectory step 12). The CONVERSION_NOTES.md states: "This conversion targets the alternating delayed-response / water-cued ALM electrophysiology cohort used for the context analyses in the reference repository."

## 1-b. How are the data split into subjects?

i. Subjects are derived from the `animal` field in `SESSION_SPECS`. They are accumulated in insertion order into an `OrderedDict` as sessions are processed. The result is 7 unique animals from 12 sessions.

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

iii. The animal ID comes directly from the session specification rather than from the filename or from within the data files.

## 1-c. How are the data split into sessions?

i. One session = one entry in `SESSION_SPECS`, corresponding to one `data_structure_<animal>_<date>.mat` file. Only the 12 sessions from `Ephys_Behavior` are processed. The `RandomizedDelay_Ephys_Behavior` folder is not searched.

ii.
```python
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")
# ...
data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
```

iii. The AI states in CONVERSION_NOTES.md: "The session list follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`."

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in `obj.bp`. Each per-trial field (hit, miss, R, L, early, etc.) has `Ntrials` entries. Spike cluster data carries trial indices that map each spike to a trial.

ii.
```python
def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        "R": np.asarray(read_numeric_dataset(bp["R"]), dtype=bool),
        "hit": np.asarray(read_numeric_dataset(bp["hit"]), dtype=bool),
        ...
    }
```

iii. The Bpod behavioral data defines trials directly; no trial boundary reconstruction is needed.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using `analysis_trial_mask`, which keeps only trials where the animal responded (hit or miss), excluding early-lick trials, no-response ("ignore") trials, and photostimulation trials. This is more aggressive than the reference, which keeps ignore/no-response trials.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

iii. CONVERSION_NOTES.md: "The decoder dataset excludes: early trials, no response trials, stim.enable trials. This matches the recurring ~early, ~no, and ~stim.enable filtering used in the repository analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu{probe}` spike-sorted clusters. Each cluster provides `trial` (spike trial index, 1-based), `trialtm` (spike time relative to trial start), and `quality` (curation label). The go cue times from `bp.ev.goCue` provide alignment.

ii.
```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
```

iii. Same raw variables as the reference: spike clusters with trial assignments and trial-relative timing.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`, then binned into 10 ms bins (DT=1/100) spanning -3.0 to 2.5 s (550 bins). The binned counts are converted to firing rates (Hz) by dividing by DT, then smoothed with a **causal** Gaussian kernel of window size 15 bins. The causal kernel is created by zeroing the first half of a `gausswin(15)` window.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15

def my_smooth(x, n, bctype="none"):
    kern = gausswin(n)
    kern[: n // 2] = 0   # causal: only past bins contribute
    kern /= kern.sum()
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

# In bin_spikes_for_session:
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The AI states: "Smoothing: causal Gaussian kernel with N=15, matching mySmooth(..., 15, 'reflect')". The bin size and time window come from the MATLAB code's `getDefaultParams.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, cluster quality labels are checked case-sensitively against `{"garbage", "gabrga", "noisy", "real?"}` -- "poor" is NOT excluded. Second, a firing rate filter removes units whose mean condition-averaged PSTH firing rate is <= 1 Hz. The PSTH is computed across 7 behavioral conditions (matching the Figure 8 analysis), and the mean across all conditions and time points must exceed 1 Hz.

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

# Firing rate filter uses condition-averaged PSTHs:
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. CONVERSION_NOTES.md: "Cluster inclusion before firing-rate filter: all qualities except garbage, gabrga, noisy, real?". The firing rate filter "matched to the MATLAB path" by building 7 context conditions.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike time, then binned using `np.histogram` with edges from TMIN to TMAX.

ii.
```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
```

iii. Same alignment approach as the reference: subtract go cue time from spike time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 10 ms (DT = 1/100), with a time window from -3.0 to 2.5 s, yielding 550 time bins. No rebinning is applied after initial binning.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. CONVERSION_NOTES.md: "Time window: [-3.0 s, 2.5 s]", "Bin size: 10 ms".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself -- the centers of the 550 time bins spanning -3.0 to 2.5 s relative to the go cue, identical for every trial.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
# ...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The time axis is defined by the bin structure, not derived from raw data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing -- the time axis is directly defined from the bin edges.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis IS the neural binning grid. Spikes are histogrammed into TIME_EDGES, and the input is TIME_AXIS (the bin centers), so they share the same time reference.

ii.
```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
# ...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.R` (right-instructed trial), `bp.hit`, `bp.L`, and `bp.miss`. The actual lick direction is inferred from the combination.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. The actual lick direction is not directly recorded but inferred from instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed side, a miss means it licked the other side. The code computes: right=1 if (R & hit) or (L & miss), left=0 otherwise. Only 2 classes (left=0, right=1). No "no lick" class because ignore trials are already excluded by trial filtering.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)

lick_dir = actual_lick_direction(behavior)[use_trials]
```

iii. CONVERSION_NOTES.md: "right if R & hit or L & miss; left if L & hit or R & miss".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are the water-cued (WC) context.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. Autowater flag directly indicates the WC context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC=0, autowater=False -> DR=1. The NOT of autowater is cast to int: ~autowater gives 1 for DR and 0 for WC.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. CONVERSION_NOTES.md: "0 = WC, 1 = DR".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived solely from `bp.hit`. Since ignore trials are already excluded, hit=1 means correct and not-hit means incorrect.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. With ignore trials excluded, outcome is simply the hit flag.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct cast of `hit` to integer: hit=True -> correct=1, hit=False (i.e., miss, since no-response trials are excluded) -> incorrect=0. Only 2 classes -- no "ignore" class.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. CONVERSION_NOTES.md: "0 = incorrect, 1 = correct".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from the DeepLabCut tracking in `obj.traj`, specifically the "tongue" feature from the side camera (view_index=1, which maps to `traj[0]`). Only one camera view is used, unlike the reference which uses both side and bottom cameras. Frame times (`frameTimes`), video offset (`sglx.bitcode`), and go cue times are also used for alignment.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
# view_index=1 -> traj_refs[0] -> side camera only
```

iii. The AI uses only the side camera for tongue tracking.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The processing differs substantially from the reference. (1) Tongue x,y positions are loaded at frame resolution. (2) For tongue, no smoothing is applied (`my_smooth(coords, 1, "reflect")` is a no-op when n=1). Wait -- actually checking more carefully: the code passes `1` for smoothing window for tongue: `if "tongue" not in feature_name: coords = my_smooth(coords, 1, "reflect")` -- this check means tongue is NOT smoothed at all (the condition is False for tongue). (3) The positions are interpolated (linear) from frame times to the neural time axis. (4) NaN positions (invisible tongue) are left as NaN (no nearest fill for tongue). (5) Velocity is computed as `np.gradient` of the interpolated positions, with NaN periods set to zero. (6) Speed = sqrt(xvel^2 + yvel^2).

ii.
```python
# No smoothing for tongue (condition excludes tongue)
if "tongue" not in feature_name:
    coords = my_smooth(coords, 1, "reflect")

# Interpolation to neural time axis
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                  bounds_error=False, fill_value=np.nan)
vals = interp(time_axis)

# Velocity computation
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
# Set NaN to zero for tongue
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0

tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. CONVERSION_NOTES.md: "tongue invisibility periods become zero velocity" and "Position interpolation and velocity computation follow the repository logic."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Binarized at the per-session 50th percentile into 2 classes (0 = below, 1 = above). A special case: if the 50th percentile is 0 (because tongue is mostly invisible and set to zero velocity), the threshold is recomputed from positive values only.

ii.
```python
def percentile_threshold(values, drop_zeros_if_needed=False):
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh

tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
# ...
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)
```

iii. CONVERSION_NOTES.md: "if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by `video_offset` (computed as in `findVideoOffset.m`) and the trial's go cue. The tongue position is then linearly interpolated from frame times directly onto the neural time axis (TIME_AXIS). This means tongue data shares the exact same time bins as neural data.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                  bounds_error=False, fill_value=np.nan, assume_sorted=True)
vals = interp(time_axis)
```

iii. Video offset matches `findVideoOffset.m`: `mode(bitstart)/fs - mode(bitStart)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj`, using `top_paw` (or fallback to `bottom_paw`) from the bottom camera (view_index=2, mapping to `traj[1]`). The AI dynamically selects the paw feature by checking which features are available.

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]  # bottom camera
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. Uses the bottom camera for paw tracking, same as the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. (1) Paw x,y loaded at frame resolution. (2) For paw (non-tongue), `my_smooth(coords, 1, "reflect")` is applied -- but with n=1 this is a no-op. (3) Positions are linearly interpolated to the neural time axis. (4) NaN positions are nearest-filled for paw. (5) Velocity is computed via `np.gradient`. (6) A baseline derivative (`nanmedian` of `diff`) is subtracted from both x and y velocity. (7) NaN velocities are nearest-filled again. (8) Speed = sqrt(xvel^2 + yvel^2).

ii.
```python
# Nearest fill for paw (non-tongue)
xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])

# Velocity with baseline subtraction
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]  # note: subtracts x baseline from y too
```

iii. CONVERSION_NOTES.md: "non-tongue features are nearest-filled" and "non-tongue baseline subtraction reproduces the MATLAB implementation, including subtracting basederiv(1) from both xvel and yvel."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Binarized at the per-session 50th percentile into 2 classes (0 = below, 1 = above). No special handling for zeros.

ii.
```python
paw_thresh = percentile_threshold(paw_selected)
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)
```

iii. Standard 50th percentile split.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then linearly interpolated onto the neural time axis. NaN regions are nearest-filled.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                  bounds_error=False, fill_value=np.nan, assume_sorted=True)
vals = interp(time_axis)
xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
```

iii. Same alignment logic as tongue and motion energy.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<animal>_<date>.mat` files. The `me.data` field contains per-trial motion energy traces (one value per camera frame). The AI also loads `me.moveThresh` but does not use it for thresholding.

ii.
```python
def load_motion_energy(animal, date, behavior):
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. The AI loads motion energy from the same source as the reference but only looks in EPHYS_DIR.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is linearly interpolated from frame times onto the neural time axis, then nearest-filled at edges to remove NaN. No additional processing (the raw values are already reduced to one number per frame).

ii.
```python
def align_motion_energy(f, behavior, raw_motion_energy):
    me_trial = np.asarray(raw_motion_energy[trial_idx], dtype=np.float64).reshape(-1)
    shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
    interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear",
                      bounds_error=False, fill_value=np.nan, assume_sorted=True)
    aligned[:, trial_idx] = interp(TIME_AXIS)
    aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. CONVERSION_NOTES.md: "Motion energy comes from motionEnergy_Animal_Date.mat, is resampled onto the neural time axis, and is nearest-filled at the edges."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Binarized at the per-session 50th percentile into 2 classes (0 = below, 1 = above).

ii.
```python
me_thresh = percentile_threshold(me_selected)
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)
```

iii. Standard 50th percentile threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times, corrected by video offset and go cue, then linearly interpolated onto the neural time axis and nearest-filled.

ii.
```python
frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
aligned[:, trial_idx] = interp(TIME_AXIS)
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. Same video offset and alignment as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Missing/NaN frame times cause the trial to be skipped for video features (`continue` when NdroppedFrames is all NaN). (2) For tongue, invisible periods (NaN coordinates) result in zero velocity. (3) For paw (non-tongue), NaN positions and velocities are nearest-filled. (4) For motion energy, NaN after interpolation is nearest-filled. (5) If a trial has no valid data for interpolation (<2 valid points), it is skipped.

ii.
```python
# Skip trials with NaN dropped frames
ndropped = np.asarray(read_numeric_dataset(f[ndropped_refs[trial_idx]]), dtype=np.float64)
if ndropped.size and np.isnan(ndropped).all():
    continue

# Nearest fill for paw
def fill_nearest(x):
    if not mask.any():
        return np.zeros_like(x)
    interp = interp1d(idx[mask], x[mask], kind="nearest", ...)
    return interp(idx)

# Zero fill for tongue
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
```

iii. The AI fills missing data rather than leaving it as a distinct class, unlike the reference which uses a "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files and computing the trajectory feature series (which involves per-trial interpolation of each kinematic feature). The per-trial loop over all trials for each feature, with interpolation, is expensive.

ii.
```python
# Per-trial interpolation loop
for trial_idx in range(ntrials):
    # ... load frame times, trajectory data
    interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear", ...)
    vals = interp(time_axis)
```

iii. The trajectory loading involves nested loops over trials and features.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop iterates over each unique trial individually, calling `np.histogram` once per trial per cluster. This could be vectorized with `np.histogram2d` as the reference does. The per-trial trajectory interpolation loop could potentially be batched.

ii.
```python
# Per-trial spike binning loop
for t in unique_trials:
    mask = trial_idx == t
    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The reference uses `np.histogram2d` to bin all trials at once for each cluster.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is recomputed every time `load_traj_feature_series` or `align_motion_energy` is called (3 times per session: tongue, paw, and motion energy). The trajectory `ts` array references are re-dereferenced for each feature.

ii.
```python
# get_video_offset called in each of:
# load_traj_feature_series(..., "tongue", 1)
# load_traj_feature_series(..., paw_feature, 2)
# align_motion_energy(...)
vidshift = get_video_offset(f, behavior)
```

iii. The reference computes the offset once per session and reuses it.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The code builds 7 context conditions and condition-averaged PSTHs solely for the firing rate filter, even though the per-trial rates (not the PSTHs) are what goes into the output. (2) It computes `single_unit_flags` and tracks single-unit quality counts for sanity checking, which aren't used in the converted data. (3) `moveThresh` is loaded from motion energy files but never used for thresholding (the 50th percentile is used instead). (4) `basederiv` subtraction adds processing for paw velocity that the reference does not do.

ii.
```python
# PSTH computed only for firing rate filter
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    ...
mean_fr = float(np.mean(psth))

# moveThresh loaded but not used
move_thresh = float(dat.moveThresh)
```

iii. The PSTH-based firing rate filter follows the reference MATLAB code but adds computational overhead.
