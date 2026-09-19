# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from only the `Ephys_Behavior` folder (not `RandomizedDelay_Ephys_Behavior`). It hardcodes 12 sessions (the Figure 8 context-task subset) in `SESSION_SPECS`. Each session's `data_structure_*.mat` is opened with `h5py` (HDF5/MATLAB v7.3 only). Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat` (MATLAB v5). The AI does NOT load all 44 sessions — only the 12 from the context analysis.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    SessionSpec("JEB7", "2021-04-30", 1),
    SessionSpec("EKH1", "2021-08-07", 2),
    SessionSpec("EKH3", "2021-08-11", 2),
    SessionSpec("JGR2", "2021-11-16", 1),
    SessionSpec("JGR2", "2021-11-17", 1),
    SessionSpec("JGR3", "2021-11-18", 1),
    SessionSpec("JEB19", "2023-04-21", 1),
    SessionSpec("JEB19", "2023-04-20", 1),
    SessionSpec("JEB19", "2023-04-19", 1),
    SessionSpec("JEB19", "2023-04-18", 1),
]

with h5py.File(spec.data_path, "r") as f:
    ...
```

iii. The agent stated: "I'm also using the figure-script session roster rather than 'all files with autowater trials', because the paper's context analyses clearly operate on a curated 12-session subset." The agent chose the Figure 8 context-task subset because the decoder outputs require behavioral context (WC vs DR).

## 1-b. How are the data split into subjects?

i. The subject (animal) name is extracted from the `SessionSpec.animal` field, which is set when hardcoding the session list. Unique subjects are collected and indexed via `OrderedDict.fromkeys`.

ii.
```python
subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
```

iii. No explicit justification given. The animal name is part of the session specification already.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` is one session, identified by animal name, date, and probe number. Only 12 sessions from the fixed-delay Ephys_Behavior folder are included — no randomized-delay sessions. Each session is loaded individually via `load_session(spec)`.

ii.
```python
sessions = [load_session(spec) for spec in SESSION_SPECS]
```

iii. The agent chose the 12-session subset from Figure 8 scripts. The agent noted: "The decoder outputs require behavioral context (WC vs DR), so I converted the ALM two-context electrophysiology cohort."

## 1-d. How are the data split into trials?

i. Trials are indexed from `bp['Ntrials']`. Each per-trial field (hit, miss, early, etc.) is read as a vector of that length. Trial indices are 0-based in the code (converted from MATLAB's 1-based).

ii.
```python
bp = {
    "Ntrials": int(round(read_h5_scalar(bp_group["Ntrials"]))),
    "hit": read_h5_vector(bp_group["hit"]).astype(np.int16),
    ...
}
```

iii. No explicit justification — follows the structure of the MATLAB data files.

## 1-e. How are trials filtered based on quality controls?

i. Only `hit` and `miss` trials are included, and trials with `stim.enable == 1` or `early == 1` are excluded. Critically, `no`/ignore trials are **excluded entirely** from the dataset.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. The agent stated: "Ignore / no-response trials were excluded because the requested outcome target is binary correct/incorrect and the paper methods state ignore trials were omitted from analyses." However, the task instructions specify outcome with three categories: incorrect, correct, and ignore — implying ignore trials should be kept.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu{probe}` — specifically the `trial` and `trialtm` fields of each cluster, plus the `quality` label. The go cue times from `bp.ev.goCue` are used for alignment.

ii.
```python
clu_group = f[h5_ref_at(f["obj/clu"], spec.probe - 1)]
session_trial, _trialtm, aligned = load_trial_spikes(f, clu_group, clu_idx, bp[ALIGN_EVENT])
```

```python
def load_trial_spikes(f, clu_group, clu_idx, align_times):
    trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
    trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
    session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
    trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
    aligned = trialtm - align_times[session_trial]
    return session_trial, trialtm, aligned
```

iii. No explicit justification beyond following the MATLAB code structure.

## 2-b. How is the `neural` data processed?

i. Spikes are histogrammed into 10 ms bins, divided by `DT` to get firing rates, then smoothed with a **causal** Gaussian kernel (window size 15). The causal kernel zeroes the first half of a `gausswin(15)` before normalizing — this is a direct port of `mySmooth.m`. Only one probe per session is used (not concatenated).

ii.
```python
DT = 1 / 100  # 10 ms bins
SMOOTH = 15

def my_smooth(x, n, bctype="none"):
    kern = gausswin(n)
    kern[: math.floor(kern.size / 2)] = 0.0  # causal: zero the first half
    kern /= kern.sum()
    out[:, j] = np.convolve(x_filt[:, j], kern, mode="same")
    ...

counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
psth = my_smooth(counts / trix.size / DT, SMOOTH, "reflect")
```

iii. The agent traced these parameters from the Figure 8 scripts and stated: "10 ms spike-rate bins with causal Gaussian smoothing (window 15) and 1 Hz low-FR filtering."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (note: `poor` is NOT in this set, unlike the reference). (2) Units whose mean smoothed PSTH firing rate across multiple condition masks is <= 1 Hz are removed.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}

for clu_idx in range(qds.shape[0]):
    quality = read_h5_string(f, h5_ref_at(qds, clu_idx)).strip().lower()
    if quality in BAD_QUALITIES:
        continue
    mean_fr = compute_psth_mean_fr(session_trial, aligned, condition_masks, edges)
    if mean_fr <= LOW_FR:
        continue
```

iii. The agent said this follows `findClusters(..., {'all'})` from the MATLAB code. The FR filtering uses condition-averaged PSTH means rather than a simple mean rate.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`: `aligned = trialtm - align_times[session_trial]`. This is equivalent to the reference approach.

ii.
```python
aligned = trialtm - align_times[session_trial]
```

iii. The agent confirmed: "ALM ephys only, goCue alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses **10 ms bins** (`DT = 1/100`), spanning [-3.0, 2.5] s, yielding 550 timepoints per trial. This differs from the reference which uses 5 ms bins over [-2.5, 2.5] s (1000 timepoints).

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100

edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. The agent traced these from the Figure 8 context scripts rather than from `getDefaultParams.m`. No explicit reasoning for why 10 ms was preferred over the default 5 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the bin-center time axis, constructed from `TMIN`, `TMAX`, and `DT`. It is not derived from any raw data variable.

ii.
```python
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. No explicit justification beyond the task specification.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is constructed analytically: bin edges from TMIN to TMAX in steps of DT, then bin centers at `edges[:-1] + DT/2`.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same time axis used for binning the neural data, so they share the same grid by construction.

ii. Same `edges` and `time` arrays are used for both neural binning and the input variable.

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.hit` and `bp.R`. On hit trials, the lick direction matches the instructed side (R=1 means right). On miss trials, the lick direction is opposite.

ii.
```python
def lick_direction_from_trial(bp, trial_idx):
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
    if hit:
        return int(is_right_trial)
    return int(not is_right_trial)
```

iii. The agent stated: "Correct right trials and incorrect left trials were labeled right. Correct left trials and incorrect right trials were labeled left."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A binary encoding: left=0, right=1. There is **no "no lick" category** because ignore trials are excluded. The output is per-trial, constant across time bins.

ii.
```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. Since ignore trials are excluded, all remaining trials have a definite lick direction. The agent did not discuss the missing "no lick" / "none" category specified in the instructions.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`: autowater=1 is WC, autowater=0 is DR.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. The agent referenced `WorkingWithDataObjs.m`: "obj.bp.autowater=1 when water was delivered regardless of animal choice... this field can be used as a proxy for obtaining water-cued blocks."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: WC=0, DR=1. Per-trial, constant across time.

ii.
```python
np.full(tongue_speed.shape[0], context, dtype=np.int16),
```

iii. Matches the task spec: WC, DR.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` only. Since ignore trials are excluded, outcome is binary: hit=correct(1), miss=incorrect(0).

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. The agent stated ignore trials were excluded because "the requested outcome target is binary correct/incorrect." However, the task spec explicitly lists three categories: incorrect, correct, ignore.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary encoding: incorrect=0, correct=1. No "ignore" class exists. Per-trial, constant across time.

ii.
```python
"output_values": [
    ...
    ["incorrect", "correct"],
    ...
]
```

iii. See 6-a.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `top_tongue` and `bottom_tongue` features from the **bottom camera** only. The side camera's `tongue` feature is not used. The x,y positions are extracted from `obj.traj[1]` (bottom view) via DLC tracking data (`ts` array).

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
    ...
}
```

iii. The agent stated: "reducing to tongue-tip speed... before session-wise median binarization, which is the least arbitrary way to satisfy the decoder spec without discarding their tracking geometry."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Steps: (1) Extract x,y from DLC tracking for each trial. (2) Interpolate positions onto the neural time axis using video-offset-corrected frame times. (3) Compute velocity as `np.gradient()` of each coordinate. (4) For tongue features, NaN velocities are set to 0 (not "not visible"). (5) Average x-velocities and y-velocities of top_tongue and bottom_tongue. (6) Compute speed as `sqrt(xvel^2 + yvel^2)`. (7) Replace remaining NaN with 0.

ii.
```python
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The agent's justification for averaging: "reducing to tongue-tip speed... which is the least arbitrary way to satisfy the decoder spec without discarding their tracking geometry."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two categories only: low (0) and high (1). The threshold is the per-session 50th percentile over **positive** tongue-speed bins only (to avoid degenerate classes when most bins are zero). There is **no "not visible" category** — invisible bins are coded as 0 speed.

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
else:
    tongue_thresh = 0.0

(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16),
```

iii. The agent explained: "the tongue-velocity target collapsed to a constant class because the reference pipeline sets non-visible tongue velocity to zero, and the session median over all bins becomes zero. I'm patching that target to compute its median over positive tongue-speed bins only."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Positions are interpolated directly onto the neural time axis using the video-offset-corrected frame times: `frameTimes - vidshift - goCue[trial]`. The video offset is computed as `mode(bitcode.bitstart)/fs - mode(bp.ev.bitStart)`.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The agent followed the MATLAB code's `findVideoOffset.m` approach.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from both `top_paw` AND `bottom_paw` features from the bottom camera. The reference uses only `top_paw`.

ii.
```python
feat_indices = {
    ...
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}
```

iii. No explicit reasoning for using both paws.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Steps: (1) Extract x,y from DLC tracking. (2) Interpolate onto neural time axis. (3) Compute velocity with `np.gradient()`. (4) For non-tongue features, subtract baseline derivative (nanmedian of diffs) and fill NaN with nearest. (5) Compute speed for each paw. (6) Average speeds of top_paw and bottom_paw. (7) Replace NaN with 0.

ii.
```python
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. Same as tongue — "reducing to mean paw speed before session-wise median binarization."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two categories: low (0) and high (1) at the per-session 50th percentile. No "not visible" category — NaN values are replaced with 0 before thresholding.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
(paw_speed[:, tr] >= paw_thresh).astype(np.int16),
```

iii. No explicit reasoning beyond following the spec.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue — positions interpolated onto neural time axis after video-offset correction.

ii. Same `interp_with_nan` and `fill_nearest_1d` approach as tongue.

iii. Same as tongue alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from separate `motionEnergy_*.mat` files via `scipy.io.loadmat`. The per-trial arrays are in `me.data`.

ii.
```python
def load_motion_energy(path):
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
    return trial_arrays, float(me.moveThresh)
```

iii. The agent followed the paper code's `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy arrays are interpolated onto the neural time axis using video-offset-corrected side-camera frame times. Edge NaN values are filled with nearest available value. The raw `moveThresh` from the paper is stored in metadata but not used for discretization.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. No explicit reasoning — directly ported from MATLAB code patterns.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two categories: low (0) and high (1) at the per-session 50th percentile over all included-trial bins. No "no video" category.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
(motion_energy[:, tr] >= me_thresh).astype(np.int16),
```

iii. Follows the decoder task spec for 50th percentile thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Interpolated onto the neural time axis using side-camera frame times after video-offset correction: `frame_times - vidshift - align_times[tr]`.

ii.
```python
motion_energy = aligned_motion_energy(
    frame_times_by_trial=frame_times_by_trial,
    raw_motion_energy=raw_motion_energy,
    align_times=bp[ALIGN_EVENT],
    vidshift=vidshift,
    taxis=taxis,
)
```

iii. Same video-offset correction as tongue and paw.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Tongue positions where tongue is not visible: NaN velocities set to 0 (no "not visible" class). (2) Non-tongue features: NaN filled with nearest available value (`fill_nearest_1d`), matching MATLAB's `fillmissing(...,'nearest')`. (3) Motion energy NaN at trial edges: filled with nearest value. (4) Trials with completely missing video: skipped in position extraction (positions remain NaN, velocities become 0). (5) If all diffs are NaN, baseline derivative defaults to 0.

ii.
```python
def fill_nearest_1d(x):
    idx = np.flatnonzero(~np.isnan(x))
    if idx.size == 0:
        return np.zeros_like(x)
    out = np.interp(np.arange(x.size), idx.astype(float), x[idx])
    return out
```

iii. No explicit reasoning — directly ported from MATLAB code patterns.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 data files and the per-cluster neural data processing (looping through all clusters per session to compute PSTHs for firing-rate filtering). Each trial's video features are also individually extracted via `h5py` reference dereferencing.

ii.
```python
with h5py.File(spec.data_path, "r") as f:
    ...
    for clu_idx in range(qds.shape[0]):
        ...
```

iii. No explicit efficiency discussion in the trajectory.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops for video feature extraction (`aligned_feature_position`), motion energy alignment (`aligned_motion_energy`), velocity computation (`aligned_feature_velocity`), and output assembly (`derive_session_outputs`) all iterate one trial at a time. The per-cluster neural processing loop also iterates one cluster at a time.

ii.
```python
for tr in range(n_trials):
    frame_times = frame_times_by_trial[tr]
    me_trial = np.asarray(raw_motion_energy[tr], ...)
    ...
```

iii. No explicit discussion of vectorization opportunities.

## 11-c. What processing does the code repeat multiple times?

i. Frame times are re-read from the HDF5 file for each feature and each trial separately (e.g., `read_frame_times` is called multiple times for the same trial but different features, even though they share the same camera). The `find_feat_index` function loops through trials to find a feature name. The condition masks are computed once but the PSTH mean FR computation loops through all conditions for each cluster.

ii.
```python
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
# But bottom camera frame times are re-read inside aligned_feature_position for each feature
```

iii. No explicit discussion.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `moveThresh` from motion energy files but never uses it for thresholding (uses 50th percentile instead). It computes `bottom_paw` velocity even though the reference only uses `top_paw`. The `compute_condition_masks` function creates 7 condition masks for FR filtering, which is more elaborate than the simple mean rate check the reference uses. The `basederiv` subtraction for non-tongue features may also be unnecessary processing.

ii.
```python
return trial_arrays, float(me.moveThresh)  # moveThresh loaded but not used for discretization

# 7 condition masks computed for FR filtering
def compute_condition_masks(bp):
    return [
        hit | miss | no,
        hit & ~stim_enable & ~autowater,
        ...
    ]
```

iii. No explicit discussion.
