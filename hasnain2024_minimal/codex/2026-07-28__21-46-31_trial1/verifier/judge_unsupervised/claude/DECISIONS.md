# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from HDF5-format MATLAB `.mat` files in `/app/data/Ephys_Behavior/` using `h5py`. A hardcoded list `SESSION_SPECS` defines 12 sessions (animal name, date, probe numbers). For each session, it opens `data_structure_{animal}_{date}.mat`, extracts behavioral data from `obj.bp`, spike cluster data from `obj.clu`, trajectory data from `obj.traj`, and separately loads motion energy from `motionEnergy_{animal}_{date}.mat` via `scipy.io.loadmat`.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    # ... 12 sessions total
]

def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        # ...
```

iii. The AI identified the 12-session alternating-context ALM cohort from the Figure 8 scripts in the reference code (`Figure8a_thru_c.m`, `Figure8d.m`). The CONVERSION_NOTES state: "This conversion targets the alternating delayed-response / water-cued ALM electrophysiology cohort used for the context analyses in the reference repository."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field in `SESSION_SPECS`. Unique animal names are accumulated in order into a `subjects` list, and each session is mapped to a subject index via `subject_lookup`.

ii.
```python
for spec in SESSION_SPECS:
    sid = spec["animal"]
    if sid not in subject_lookup:
        subject_lookup[sid] = len(subject_lookup)
        subjects.append(sid)
    data["subject_idx"].append(subject_lookup[sid])
```

iii. The AI noted in CONVERSION_NOTES that this yields 7 animal IDs (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19) while the paper says 6 mice for 12 sessions. The AI kept the implementation anchored to the repository's explicit session loaders rather than forcing a match.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` corresponds to one session. The AI iterates over the 12 session specs sequentially, processing each independently via `build_session_payload()`, then appending neural/input/output data to the dataset lists.

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The session list matches the Figure 8 script loaders in the reference code.

## 1-d. How are the data split into trials?

i. Within each session, the total number of trials is read from `obj.bp.Ntrials`. Spike data per cluster contains per-spike trial indices (`clu.trial`) and spike times (`clu.trialtm`). Behavioral variables (hit, miss, early, etc.) are per-trial boolean arrays. After filtering (see 1-e), kept trials are iterated over and each becomes one entry in the session's trial list.

ii.
```python
out = {
    "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
    "hit": np.asarray(read_numeric_dataset(bp["hit"]), dtype=bool),
    # ...
}
# Later:
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
```

iii. The trial structure follows the MATLAB data objects where each cluster stores spike times with associated trial indices.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies `analysis_trial_mask` which keeps trials that are (hit OR miss) AND NOT early AND NOT no-response AND NOT stimulation-enabled. This excludes early-lick trials, no-response trials, and photostimulation trials.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

iii. CONVERSION_NOTES: "The decoder dataset excludes: early trials, no response trials, stim.enable trials. This matches the recurring ~early, ~no, and ~stim.enable filtering used in the repository analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from spike clusters stored in `obj.clu{probe}`. Each cluster has `quality` (string label), `trial` (per-spike trial index, 1-based in MATLAB, converted to 0-based), and `trialtm` (per-spike time in seconds).

ii.
```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
```

iii. This matches the MATLAB reference code structure where `obj.clu{prbnum}(curClu).trialtm` stores spike times per cluster.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue by subtracting `behavior["ev"]["goCue"][trial_idx]`. Aligned spikes are binned into 10ms time bins (edges from -3.0 to 2.5s), giving spike counts. Counts are divided by `DT` (0.01s) to get instantaneous firing rates in Hz, then smoothed with a causal Gaussian kernel (N=15 bins, reflect boundary condition).

ii.
```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. This replicates the MATLAB `getSeq.m` pipeline: `N = histc(trialtm_aligned, edges); N = N(1:end-1); trialdat = mySmooth(N./params.dt, params.smooth, params.bctype)`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) Cluster quality filter: clusters with quality labels "garbage", "gabrga", "noisy", or "real?" are excluded. (2) Low firing rate filter: condition-averaged PSTHs are computed across 7 context conditions (matching Figure 8), and units with mean PSTH firing rate <= 1 Hz are excluded.

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

# Low FR filter:
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. CONVERSION_NOTES: "Cluster inclusion before firing-rate filter: all qualities except garbage, gabrga, noisy, real?. Low-firing-rate exclusion: remove units with mean PSTH firing rate <= 1 Hz." The reference code uses `params.quality = {'all'}` which keeps all quality levels, relying solely on the FR filter. The AI's additional quality pre-filter is an extra step.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the go cue time for that spike's trial: `trialtm - ev.goCue[trial]`. Spikes are then binned into a fixed time grid from -3.0s to +2.5s relative to go cue onset.

ii.
```python
ALIGN_EVENT = "goCue"
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. This matches the MATLAB `alignSpikes.m`: `obj.clu{prbnum}(clu).trialtm_aligned = obj.clu{prbnum}(clu).trialtm - event`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (DT = 1/100 = 0.01 seconds). No rebinning is applied; spikes are binned directly at this resolution. The time axis spans from -3.0s to +2.5s in 10ms steps, yielding 550 time bins per trial.

ii.
```python
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. Matches reference parameters: `params.dt = 1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the neural time axis itself (`TIME_AXIS`), which is computed from the bin edges defined by `TMIN`, `TMAX`, and `DT`. It is not derived from any raw data variable directly.

ii.
```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The time axis represents bin centers relative to go cue onset, consistent with the instruction to provide "Time from go cue onset in seconds" as a continuous, time-varying decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing is involved. The time axis is a deterministic array of bin centers computed from the time window parameters. Each trial receives the same time axis array.

ii.
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
# Shape: (1, 550) for each trial
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. Straightforward construction of time bin centers.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural time axis since both use the same `TIME_AXIS` array. The time values directly represent seconds relative to go cue onset.

ii.
```python
# Neural bins use TIME_EDGES derived from same TMIN/TMAX/DT
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
# Input uses TIME_AXIS (bin centers of same edges)
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. Perfect alignment by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from four behavioral variables: `R` (right-instruction trials), `L` (left-instruction trials), `hit` (correct trials), and `miss` (incorrect trials).

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. CONVERSION_NOTES: "Defined as actual lick direction, not instructed side: right if R & hit or L & miss, left if L & hit or R & miss."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI computes the actual lick direction (which side the animal licked), not the instructed side. Right lick = 1 when (right-instructed AND correct) OR (left-instructed AND incorrect). Left lick = 0 otherwise. The value is per-trial and is repeated across all time bins.

ii.
```python
lick_dir = actual_lick_direction(behavior)[use_trials]
# Then tiled across time:
repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The instructions specify "left = 0, right = 1, per-trial". The AI chose actual lick direction rather than instructed side, which is the standard interpretation.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` boolean array in `obj.bp`.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. CONVERSION_NOTES: "behavioral_context: 0 = WC, 1 = DR."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater=True` indicates water-cued (WC) trials; `autowater=False` indicates delayed-response (DR) trials. The AI inverts `autowater` to get DR=1, WC=0, matching the instruction specification. The value is per-trial and repeated across time bins.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
# WC (autowater=True) -> ~True = 0
# DR (autowater=False) -> ~False = 1
```

iii. Matches instruction: "WC = 0, DR = 1, per-trial".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` boolean array in `obj.bp`.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. Simple mapping: hit=True -> correct=1, hit=False -> incorrect=0.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Since the trial filter already excludes `no` and `early` trials, the remaining trials are either `hit` (correct) or `miss` (incorrect). The `hit` flag directly maps to the outcome: 1=correct, 0=incorrect. Repeated across time bins.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. Matches instruction: "incorrect = 0, correct = 1, per-trial".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the DeepLabCut-tracked "tongue" feature in the first trajectory view (`obj.traj{1}`). Specifically, the x/y coordinates are extracted from the `ts` (time series) field, with frame timing from `frameTimes`.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
tongue_xvel, tongue_yvel = compute_velocity_from_position(tongue_xpos, tongue_ypos, "tongue")
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. CONVERSION_NOTES: "scalar tongue speed from the reference x/y tongue velocity components."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing pipeline: (1) Extract x,y positions from DLC tracking data for each trial. (2) Align frame times using `frameTimes - vidshift - goCue[trial]`. (3) Interpolate positions to the neural time axis using linear interpolation. (4) NaN positions (tongue invisible) are NOT filled (unlike other features). (5) Compute velocity via `np.gradient` (central differences) on each coordinate. (6) NaN velocities during tongue invisibility are set to 0. (7) Compute speed as `sqrt(xvel^2 + yvel^2)`.

ii.
```python
# In compute_velocity_from_position, for tongue:
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
# tongue NaN -> 0:
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
```

iii. CONVERSION_NOTES: "tongue invisibility periods become zero velocity" matching the MATLAB reference behavior.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile threshold computed from all kept-trial tongue speed values. Special handling: if the threshold is <= 0 (common due to many zero values from invisible tongue periods), the threshold is recomputed from positive speed values only.

ii.
```python
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)

def percentile_threshold(values, drop_zeros_if_needed=False):
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh
```

iii. CONVERSION_NOTES: "because the repository sets invisible-tongue velocity to zero, if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only; otherwise several sessions collapse to a trivial constant label."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position is interpolated to the same time axis used for neural data via linear interpolation of frame times shifted by video offset and go cue time.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                  bounds_error=False, fill_value=np.nan)
vals = interp(time_axis)
```

iii. Matches reference: `frameTimes - vidshift - goCue`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the "top_paw" (or "bottom_paw" as fallback) DLC feature in the second trajectory view (`obj.traj{2}`, bottom camera).

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]  # view 2 (0-indexed)
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The reference code extracts paw kinematics from the bottom camera view.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing pipeline: (1) Extract x,y positions from DLC tracking. (2) Align and interpolate to neural time axis. (3) Fill NaN positions with nearest valid value (`fill_nearest`). (4) Compute velocity via `np.gradient`. (5) Subtract baseline drift: `basederiv[0]` (median of x-position differences) subtracted from BOTH xvel and yvel. (6) Fill NaN velocities with nearest. (7) Compute speed = `sqrt(xvel^2 + yvel^2)`.

ii.
```python
# Non-tongue velocity computation:
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
```

iii. The baseline subtraction replicates the MATLAB `findVelocity.m` behavior including the quirk of subtracting `basederiv(1)` from both x and y velocities.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Per-session 50th percentile threshold computed from all kept-trial paw speed values. No special zero handling (unlike tongue).

ii.
```python
paw_thresh = percentile_threshold(paw_selected)
# Then: (paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)
```

iii. Matches instruction: "0: < 50th percentile, 1: >= 50th percentile".

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue: frame times shifted by `vidshift + goCue`, linearly interpolated to the neural time axis, then nearest-filled for missing values.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(shifted_t[valid], coords[valid, dim], ...)
vals = interp(time_axis)
# Non-tongue positions are nearest-filled:
xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
```

iii. Matches reference video alignment logic.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy comes from separate `motionEnergy_{animal}_{date}.mat` files loaded via `scipy.io.loadmat`. The raw data is in `me.data` (per-trial arrays at video frame rate) and `me.moveThresh` (a scalar threshold from the reference, stored but not used for thresholding).

ii.
```python
def load_motion_energy(animal, date, behavior):
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. CONVERSION_NOTES: "Motion energy comes from motionEnergy_Animal_Date.mat."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Raw per-trial motion energy arrays (at video frame rate) are interpolated to the neural time axis using frame times from the first trajectory view, aligned via `frameTimes - vidshift - goCue`. NaN values at edges are filled with nearest valid values.

ii.
```python
def align_motion_energy(f, behavior, raw_motion_energy):
    # Uses frame times from first trajectory view
    shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
    interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
    aligned[:, trial_idx] = interp(TIME_AXIS)
    aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. CONVERSION_NOTES: "resampled to neural time axis" and "nearest-filled at the edges."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Per-session 50th percentile threshold computed from all kept-trial motion energy values.

ii.
```python
me_thresh = percentile_threshold(me_selected)
# Then: (me_selected[:, local_idx] >= me_thresh).astype(np.int64)
```

iii. Matches instruction: "0: < 50th percentile, 1: >= 50th percentile". Note the reference code uses `me.moveThresh` for a binary move/no-move classification, but the instructions explicitly override this with percentile-based thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated to the same neural time axis using linear interpolation of video frame times corrected for video offset and go cue time.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. Matches reference: `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) NaN positions for non-tongue features are filled with nearest valid values (`fill_nearest`). (2) Tongue invisible periods (NaN) produce zero velocity. (3) All-NaN trial trajectories are skipped (no position/velocity data). (4) Spike trial indices outside valid range are filtered out. (5) Clusters with no valid data after filtering are skipped. (6) If a trajectory feature name is missing for a specific trial, that trial is skipped. (7) If `basederiv` is NaN, it defaults to 0.

ii.
```python
def fill_nearest(x):
    if not mask.any():
        return np.zeros_like(x)  # all-NaN -> zeros
    interp = interp1d(idx[mask], x[mask], kind="nearest", ...)
    return interp(idx)

# Spike trial validation:
valid_trial_spikes = (trial_idx >= 0) & (trial_idx < ntrials)

# basederiv NaN handling:
if not np.isfinite(basederiv[0]):
    basederiv[0] = 0.0
```

iii. CONVERSION_NOTES mention edge-filling and tongue invisibility handling as matching the reference code.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading and parsing the HDF5 files with h5py, especially dereferencing cell arrays of spike data per cluster. (2) Spike binning loop: iterating over each neuron, then over each trial, calling `np.histogram` and `my_smooth` individually. (3) Trajectory feature loading: per-trial loading and interpolation of video tracking data. (4) Motion energy alignment: per-trial interpolation.

ii.
```python
# Per-neuron, per-trial spike binning:
for clu in probe:
    for t in unique_trials:
        counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
        trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. No explicit justification for performance choices in CONVERSION_NOTES.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. (1) The spike binning loop iterates per-neuron per-trial; vectorized histogramming across trials could be faster. (2) The `my_smooth` convolution loop iterates per column; could use 2D convolution. (3) The `compute_velocity_from_position` loop iterates per trial; `np.gradient` along axis=0 could vectorize this. (4) The trajectory loading/interpolation loops per trial could potentially be parallelized.

ii.
```python
# Per-column smoothing loop:
for col in range(x.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

# Per-trial velocity:
for trial_idx in range(xpos.shape[1]):
    xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
```

iii. No discussion of vectorization in CONVERSION_NOTES.

## 11-c. What processing does the code repeat multiple times?

i. (1) Video offset (`get_video_offset`) is computed separately for trajectory loading and motion energy alignment within the same session, despite being the same value. (2) Frame times from the trajectory views are loaded multiple times: once for tongue, once for paw, and once for motion energy. (3) The `deref_string_list` call for feature names is made per trial within `load_traj_feature_series`.

ii.
```python
# Video offset computed in load_traj_feature_series AND align_motion_energy:
vidshift = get_video_offset(f, behavior)  # called in each function
```

iii. No discussion of redundant computation in CONVERSION_NOTES.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The 7-condition PSTH is computed solely for the low-FR filter and then discarded. (2) The `manual_motion_thresh` (moveThresh) from the motion energy file is loaded and stored in session stats but not used for the actual thresholding (which uses percentile-based thresholds per the instructions). (3) The `single_unit_flags` array is computed per session for sanity-check counting but not included in the final data structure. (4) `session_stats` includes extensive metadata that is informational but not used by the decoder.

ii.
```python
# PSTH only used for FR filter:
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
mean_fr = float(np.mean(psth))
# Discarded after filtering

# moveThresh loaded but unused for thresholding:
raw_motion_energy, manual_motion_thresh = load_motion_energy(...)
# manual_motion_thresh only stored in stats
```

iii. The PSTH computation replicates the reference code's approach to FR filtering, which is necessary for matching the reference neuron counts even though the PSTH itself is not retained.
