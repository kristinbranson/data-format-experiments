# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from MATLAB v7.3 `.mat` files (HDF5 format) stored in `/app/data/Ephys_Behavior/`. Each session has a `data_structure_{animal}_{date}.mat` file opened with `h5py` and a separate `motionEnergy_{animal}_{date}.mat` file opened with `scipy.io.loadmat`. A hardcoded list of 12 `SessionSpec` objects defines the exact sessions to load. The `build_dataset()` function iterates over this list, calling `load_session()` for each.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

def build_dataset() -> dict:
    sessions = [load_session(spec) for spec in SESSION_SPECS]
    ...

def load_session(spec: SessionSpec) -> dict:
    ...
    with h5py.File(spec.data_path, "r") as f:
        bp_group = f["obj/bp"]
        ...
```

iii. The agent identified the Figure 8 context-task session roster from `Figure8a_thru_c.m` and the per-animal loader scripts (e.g., `loadJEB6_ALMVideo.m`). The CONVERSION_NOTES.md states: "The session roster was taken from the reference MATLAB context-analysis pipeline in `code/Scripts/Figure 8/Figure8a_thru_c.m` plus the per-animal loader files."

## 1-b. How are the data split into subjects (mice)?

i. Subject identity is derived from the `animal` field of each `SessionSpec`. Unique subjects are collected in insertion order across all sessions. A `subject_idx` array maps each session to its subject index.

ii.
```python
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_to_idx[sess["subject"]] for sess in sessions], dtype=np.int64),
```

iii. The agent preserved the 7 unique subject IDs found in the data files (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19), noting the paper reports 6 mice but choosing not to collapse or rename subjects without evidence.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` defines one session (animal + date + probe number). Sessions are processed independently in `load_session()` and collected into lists in `build_dataset()`. The output data structure has lists indexed by session.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"
```

iii. The agent identified the 12-session roster from the Figure 8 MATLAB scripts, matching each session to a specific probe number from the per-animal loader scripts.

## 1-d. How are the data split into trials?

i. Trial information is read from `obj/bp` in each HDF5 file. The number of trials is read from `bp.Ntrials`. Individual trial properties (hit, miss, early, autowater, R, L, stim_enable) and event times (goCue, sample, delay) are read as vectors indexed by trial number.

ii.
```python
bp = {
    "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
    "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
    "miss": read_h5_vector(bp_group["miss"]).astype(np.int16),
    ...
    "goCue": read_h5_vector(ev_group["goCue"]),
}
```

iii. The agent read the HDF5 structure to extract per-trial behavioral parameters, following the same field names used in the MATLAB reference code.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to include only `hit` or `miss` trials, excluding `early` trials and trials with optogenetic stimulation (`stim.enable`). The `trial_selector()` function computes this mask, and `included_trials` is the set of indices that pass.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)

included_mask = trial_selector(bp)
included_trials = np.flatnonzero(included_mask)
if included_trials.size < 2:
    raise RuntimeError(f"{spec.stem}: fewer than 2 usable trials after filtering")
```

iii. The CONVERSION_NOTES state: "Retained hit and miss trials with stim.enable == 0 and early == 0; ignore/no trials excluded." The agent excluded ignore/no-response trials because the requested outcome target is binary (correct/incorrect) and the paper methods state ignore trials were omitted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `obj/clu/{probe}/trial` (trial assignment per spike) and `obj/clu/{probe}/trialtm` (spike time within trial). The probe index is determined by the `SessionSpec.probe` field.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]

def load_trial_spikes(f, clu_group, clu_idx, align_times):
    trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
    trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
    session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
    trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
    aligned = trialtm - align_times[session_trial]
    return session_trial, trialtm, aligned
```

iii. The agent traced the spike data path through the HDF5 file structure, matching the MATLAB code's `obj.clu` access pattern.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, histogrammed into 10 ms bins over [-3.0, 2.5] s (550 bins), converted to firing rates (counts/dt), and smoothed with a causal Gaussian kernel (window length 15, reflect boundary). This is done both for PSTH computation (for low-FR filtering) and for single-trial data.

ii.
```python
DT = 1 / 100  # 10 ms bins
SMOOTH = 15

def my_smooth(x, n, bctype="none"):
    kern = gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0  # causal: zero out left half
    kern /= kern.sum()
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    ...

def build_neural_trials(session_trial, aligned_spikes, included_trials, edges):
    counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
    out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The agent ported the MATLAB `mySmooth.m` function, matching the causal Gaussian kernel with reflect padding, and used the Figure 8 parameters (smooth=15, bctype='reflect', dt=1/100).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied sequentially: (1) Quality filter: clusters labeled "garbage", "gabrga", "noisy", or "real?" are excluded. (2) Low firing-rate filter: the mean PSTH firing rate across 7 condition groups must exceed 1 Hz.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

for clu_idx in range(qds.shape[0]):
    quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
    if quality in BAD_QUALITIES:
        continue
    ...
    mean_fr = compute_psth_mean_fr(session_trial, aligned, condition_masks, edges)
    if mean_fr <= LOW_FR:
        continue
```

iii. The agent read `findClusters.m` (quality filtering) and `removeLowFRClusters.m` (low-FR filtering with threshold 1 Hz from Figure 8 params).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting the per-trial go cue time: `aligned = trialtm - goCue[trial]`. Aligned spikes are then histogrammed into bins spanning [-3.0, 2.5] s relative to go cue.

ii.
```python
ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5

aligned = trialtm - align_times[session_trial]  # align_times = bp["goCue"]
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
```

iii. The agent followed `params.alignEvent = 'goCue'` from `Figure8a_thru_c.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (dt = 1/100 s), producing 550 bins over the [-3.0, 2.5] s window. No temporal rebinning is applied; the raw spike times are directly histogrammed into the 10 ms bins.

ii.
```python
DT = 1 / 100

def build_edges_and_time():
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    ...
    time = edges[:-1] + DT / 2
```

iii. The agent matched `params.dt = 1/100` from the Figure 8 script.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time bin centers of the neural data bins, which are computed from the alignment parameters (TMIN, TMAX, DT). It is not directly derived from any raw data variable but rather from the constructed time axis.

ii.
```python
time = edges[:-1] + DT / 2  # bin centers

session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The time axis is constructed to represent the center of each 10 ms bin relative to the go cue onset, spanning from -2.995 s to 2.495 s.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time values are computed as bin centers: `edges[:-1] + DT/2`, where edges span from -3.0 to 2.5 in steps of 0.01. The same 1x550 time vector is replicated for every trial.

ii.
```python
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. No special processing is needed; the time axis is a deterministic function of the alignment parameters.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time series is inherently aligned with the neural data because it uses the same time bin centers. Each time bin center represents the midpoint of a 10 ms window relative to go cue onset.

ii.
```python
edges, time = build_edges_and_time()
# time is used for both neural binning (via edges) and input
```

iii. By construction, bin index `i` in the neural data corresponds to time `time[i]` in the input, both relative to go cue onset.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.hit` (whether the trial was correct) and `bp.R` (whether the trial was a right-instruction trial).

ii.
```python
def lick_direction_from_trial(bp, trial_idx):
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The agent reasoned that on correct (hit) trials, the animal licked the instructed side; on incorrect (miss) trials, the animal licked the opposite side. CONVERSION_NOTES: "Correct right trials and incorrect left trials were labeled right; correct left trials and incorrect right trials were labeled left."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Lick direction is computed as a per-trial scalar (0=left, 1=right) and broadcast to all time bins. On hit trials, it matches the instructed side (R=1 -> right=1). On miss trials, it is the opposite of the instructed side.

ii.
```python
lick_dir = lick_direction_from_trial(bp, tr)
tr_out = np.vstack([
    np.full(tongue_speed.shape[0], lick_dir, dtype=np.int16),
    ...
])
```

iii. The per-trial value is replicated across all 550 time bins to match the time-varying output format.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. The agent mapped `autowater=1` to WC (water-cued, value 0) and `autowater=0` to DR (delayed-response, value 1), matching the instruction spec (WC=0, DR=1).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A simple mapping: autowater=1 -> WC=0, autowater=0 -> DR=1. The per-trial value is broadcast to all 550 time bins.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
np.full(tongue_speed.shape[0], context, dtype=np.int16),
```

iii. No complex processing; direct mapping from the autowater flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit`.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. Hit trials are labeled correct (1), miss trials are labeled incorrect (0). Since trial filtering already excluded no-response and early trials, only hit and miss remain.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Simple binary mapping: hit=1 -> correct=1, miss=1 -> incorrect=0. Per-trial value broadcast to all time bins.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
np.full(tongue_speed.shape[0], outcome, dtype=np.int16),
```

iii. Straightforward mapping.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from bottom-view DLC tracking data: `top_tongue` and `bottom_tongue` feature positions from `obj/traj` (the bottom camera view group), and `frameTimes` for temporal alignment. Video offset correction uses `obj/bp/ev/bitStart` and `obj/sglx/bitcode/bitstart`.

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
    ...
}

top_tongue_xvel, top_tongue_yvel = velocities["top_tongue"]
bottom_tongue_xvel, bottom_tongue_yvel = velocities["bottom_tongue"]
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
```

iii. The agent traced the DLC kinematics pipeline through `getKinematicsFromVideo.m`, `findPosition.m`, and `findVelocity.m`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Read x,y positions from DLC tracking for each trial. (2) Align to go cue using video offset correction. (3) Interpolate from frame times to 10 ms time axis. (4) Compute velocity via `np.gradient`. (5) Set NaN velocities to 0 (tongue not visible). (6) Average top_tongue and bottom_tongue velocities. (7) Compute speed as sqrt(xvel^2 + yvel^2). (8) Replace remaining NaNs with 0.

ii.
```python
# Position interpolation
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)

# Velocity computation (tongue case)
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0
yvel[:, tr][np.isnan(yvel[:, tr])] = 0.0

# Tongue speed
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The agent ported the MATLAB kinematics pipeline, noting that tongue NaN values are set to zero (matching `findVelocity.m`'s behavior for tongue features).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is thresholded at the per-session 50th percentile of *positive* tongue speed values (values > 0). Values >= threshold become 1 (high), values < threshold become 0 (low).

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
else:
    tongue_thresh = 0.0

(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16),
```

iii. The agent initially used the 50th percentile over all values but found this collapsed to zero (because most tongue-speed samples are zero when the tongue is not visible). The agent patched to use only positive values, documenting: "Restricting the threshold computation to positive tongue-speed bins preserves the reference-aligned trace while yielding a usable categorical target."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is interpolated onto the same 10 ms time axis as the neural data, using the go-cue-aligned frame times with video offset correction. The time axis includes a configurable `ADVANCE_MOVEMENT` offset (set to 0.0).

ii.
```python
taxis = time + ADVANCE_MOVEMENT  # ADVANCE_MOVEMENT = 0.0

interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The alignment ensures tongue velocity samples correspond to the same time bins as neural activity.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from bottom-view DLC tracking data: `top_paw` and `bottom_paw` feature positions from `obj/traj` (bottom camera view group), plus `frameTimes` and video offset correction variables.

ii.
```python
feat_indices = {
    ...
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}

top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
```

iii. The agent followed the DLC kinematics pipeline for non-tongue features from the reference code.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps: (1) Read x,y positions for top_paw and bottom_paw. (2) Align to go cue with video offset. (3) Interpolate to 10 ms time axis. (4) Fill missing position values with nearest neighbor. (5) Compute velocity via `np.gradient`. (6) Subtract baseline derivative (median of diff). (7) Fill missing velocity values with nearest neighbor. (8) Compute speed for each paw feature. (9) Average the two paw speeds.

ii.
```python
# Non-tongue position: nearest-fill
xpos[:, tr] = fill_nearest_1d(xpos[:, tr])
ypos[:, tr] = fill_nearest_1d(ypos[:, tr])

# Non-tongue velocity: baseline subtraction + nearest-fill
basederiv = np.nanmedian(diff_xy, axis=0)
xvel[:, tr] = xvel[:, tr] - basederiv[0]
yvel[:, tr] = yvel[:, tr] - basederiv[0]  # BUG: should be basederiv[1]
```

iii. The agent ported `findPosition.m` and `findVelocity.m`. Note: there is a bug on line 333 where `yvel` subtracts `basederiv[0]` instead of `basederiv[1]`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded at the per-session 50th percentile across all time bins and included trials. Values >= threshold become 1 (high), values < threshold become 0 (low).

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
(paw_speed[:, tr] >= paw_thresh).astype(np.int16),
```

iii. This follows the decoder task specification directly: 50th percentile per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw velocity is interpolated onto the same 10 ms time axis as neural data, using go-cue-aligned frame times with video offset correction, identical to tongue velocity alignment.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. Same alignment mechanism as all other video-derived features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `motionEnergy_{animal}_{date}.mat` files. The relevant variables are `me.data` (per-trial motion energy traces) and `me.moveThresh` (motion threshold). Frame times come from `obj/traj` side view's `frameTimes`.

ii.
```python
def load_motion_energy(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
```

iii. The agent followed `loadMotionEnergy.m` for loading the separate motion energy files rather than using any embedded field.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load per-trial motion energy traces and move threshold from `.mat` files. (2) For each trial, interpolate motion energy from frame times to the 10 ms time axis using video offset correction. (3) Fill edge NaN values with nearest-neighbor interpolation.

ii.
```python
def aligned_motion_energy(frame_times_by_trial, raw_motion_energy, align_times, vidshift, taxis):
    for tr in range(n_trials):
        frame_times = frame_times_by_trial[tr]
        me_trial = np.asarray(raw_motion_energy[tr], dtype=np.float64).reshape(-1)
        out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
        out[:, tr] = fill_nearest_1d(out[:, tr])
    return out
```

iii. The agent matched `loadMotionEnergy.m`'s alignment logic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the per-session 50th percentile across all time bins and included trials. Values >= threshold become 1 (high), values < threshold become 0 (low).

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
(motion_energy[:, tr] >= me_thresh).astype(np.int16),
```

iii. This follows the decoder task specification. Note that the raw `me.moveThresh` is loaded but not used for thresholding — it is only stored in the sanity check metadata.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the same 10 ms time axis as neural data, using side-view frame times aligned to go cue with video offset correction.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
```

iii. Same alignment principle as tongue and paw velocities.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies handle missing data: (1) Trials with NaN `NdroppedFrames` are skipped for video features. (2) Tongue positions/velocities that are NaN (tongue not visible) are set to 0. (3) Non-tongue positions/velocities use nearest-neighbor filling. (4) Motion energy traces use nearest-neighbor filling for edge NaN values. (5) Units with empty/unreadable quality strings (e.g., uint64 zeros in HDF5) are treated as passing quality filter. (6) `interp_with_nan` handles NaN values in input coordinates by excluding them from interpolation.

ii.
```python
def fill_nearest_1d(x):
    idx = np.flatnonzero(~np.isnan(x))
    if idx.size == 0:
        return np.zeros_like(x)
    out = np.interp(np.arange(x.size), idx.astype(np.float64), x[idx])
    return out

# Tongue NaN handling
xvel[:, tr][np.isnan(xvel[:, tr])] = 0.0

# Skip corrupted video trials
ndropped = read_ndropped_frames(f, view_group, tr)
if np.isnan(ndropped):
    continue
```

iii. The agent followed the MATLAB reference code's handling patterns (e.g., `fillmissing(..., 'nearest')` -> `fill_nearest_1d`, tongue NaN -> 0).

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Per-cluster PSTH computation for low-FR filtering (iterates over all clusters x 7 conditions, computing histograms and smoothing). (2) Video feature alignment (reads DLC data per-trial per-feature from HDF5, interpolates). (3) Motion energy alignment per-trial. (4) Neural trial building (histogram + smooth per cluster per included trial).

ii.
```python
# Per-cluster loop: quality check + PSTH + trial building
for clu_idx in range(qds.shape[0]):
    ...
    mean_fr = compute_psth_mean_fr(session_trial, aligned, condition_masks, edges)
    ...
    neural_by_trial.append(build_neural_trials(...))
```

iii. The HDF5 reading involves many individual dataset accesses via object references, which is inherently slow.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The per-trial loop in `build_neural_trials` computing histograms individually. (2) The per-column convolution loop in `my_smooth`. (3) The per-trial loops in `aligned_feature_position` and `aligned_feature_velocity`. (4) The per-trial loop in `aligned_motion_energy`. (5) The `compute_psth_mean_fr` loop over conditions.

ii.
```python
# Per-trial histogram loop (could batch)
for col, tr in enumerate(included_trials):
    counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
    out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")

# Per-column smoothing loop
for j in range(x_filt.shape[1]):
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
```

iii. The agent did not explicitly discuss vectorization opportunities.

## 11-c. What processing does the code repeat multiple times?

i. (1) Frame times are read from HDF5 for every feature separately in `aligned_feature_position`, but motion energy also reads them in `load_session`. (2) The smoothing function `my_smooth` is called both during PSTH computation (for FR filtering) and during per-trial neural data building — the PSTH smoothing is only for filtering and is discarded. (3) Video offset computation implicitly reads the same bitStart/bitcode data.

ii.
```python
# Frame times read once for motion energy
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]

# Frame times read again inside aligned_feature_position for each feature
frame_times = read_frame_times(f, view_group, tr)  # called 4 times per trial (4 features)
```

iii. The repeated frame time reading is a minor inefficiency but functionally correct.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The PSTH computation (`compute_psth_mean_fr`) builds full smoothed PSTHs across 7 conditions solely to compute a mean firing rate for filtering — only the scalar mean is used. (2) The `compute_condition_masks` function computes 7 condition masks matching the Figure 8 analysis, but the decoder only uses the trial_selector subset. (3) The `sample` and `delay` event times are read from HDF5 but never used. (4) `bp.L` is read but never used (only `bp.R` is used for lick direction). (5) The `raw_move_thresh` from motion energy files is loaded and stored in metadata but not used for thresholding.

ii.
```python
# Condition masks computed but only used for FR filtering
condition_masks = compute_condition_masks(bp)

# sample and delay read but unused
"sample": read_h5_vector(ev_group["sample"]),
"delay": read_h5_vector(ev_group["delay"]),

# L read but unused
"L": read_h5_vector(bp_group["L"]).astype(np.int16),
```

iii. These are minor inefficiencies carried over from replicating the reference MATLAB code's structure.
