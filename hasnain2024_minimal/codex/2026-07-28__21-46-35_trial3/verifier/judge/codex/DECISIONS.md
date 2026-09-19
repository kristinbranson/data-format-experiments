# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session Figure 8 context-task roster in `SESSION_SPECS`, loads only from `/app/data/Ephys_Behavior`, opens `data_structure_*.mat` with `h5py`, and reads motion energy from separate `motionEnergy_*.mat` files with `scipy.io.loadmat`. It does not discover sessions by globbing and does not support the second task folder or MATLAB v5 session files.

ii.
```python
DATA_DIR = ROOT / "data" / "Ephys_Behavior"

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

with h5py.File(spec.data_path, "r") as f:
    ...

def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
```

iii. In the trajectory, the AI says it chose the Figure 8 context-task roster, `goCue` alignment, `10 ms` bins, and separate motion-energy files because it wanted to match the manuscript’s context analysis rather than use every packaged session (steps 67, 79, 89, 148).

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from `SessionSpec.animal`, and the final `subjects` list preserves first appearance order via `OrderedDict` rather than sorting. Session-to-subject mapping is built from that ordered list.

ii.
```python
return {
    ...
    "subject": spec.animal,
    ...
}

subjects = list(OrderedDict.fromkeys(sess["subject"] for sess in sessions))
subject_to_idx = {subj: i for i, subj in enumerate(subjects)}
```

iii. The trajectory does not separately justify subject parsing; it follows from using the hard-coded session roster, which already stores each animal ID explicitly (steps 67, 89, 148).

## 1-c. How are the data split into sessions?

i. One `SessionSpec` entry is one session. The AI treats the 12 hard-coded Figure 8 sessions as the entire dataset, each with a single named probe, all from `Ephys_Behavior`.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int

sessions = [load_session(spec) for spec in SESSION_SPECS]
```

iii. The AI explicitly says it is using the manuscript Figure 8 context-task ALM session roster instead of a broader session inventory (steps 67, 79, 89, 148).

## 1-d. How are the data split into trials?

i. Trials are indexed by the Bpod per-trial arrays, with `Ntrials` implied by the lengths of those vectors. Included trials are `np.flatnonzero(trial_selector(bp))`, and spikes are assigned to trials by `cluster["trial"] - 1`.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)

included_trials = np.flatnonzero(included_mask)

session_trial = np.asarray(trial_ds[()], dtype=np.float64).reshape(-1).astype(np.int64) - 1
```

iii. The trajectory says the pipeline is trial-aligned on `goCue` and that early and ignore trial handling is part of the Figure 8 setup it intended to mirror (steps 16, 67, 148).

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only hit or miss trials, and excludes early-lick and photostimulation trials. Ignore/no-response trials are dropped entirely. There is no additional filter for trials after recording end.

ii.
```python
def trial_selector(bp: dict[str, np.ndarray]) -> np.ndarray:
    hit_or_miss = bp["hit"].astype(bool) | bp["miss"].astype(bool)
    return hit_or_miss & ~bp["early"].astype(bool) & ~bp["stim_enable"].astype(bool)
```

iii. The AI says it is excluding early and ignore trials because that matched its reading of the paper’s context-task figure scripts (steps 16, 67, 148).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` comes from each cluster’s `trial` and `trialtm` fields plus the trialwise `goCue` times; `quality` is also read to filter units.

ii.
```python
trial_ds = f[h5_ref_at(clu_group["trial"], clu_idx)]
trialtm_ds = f[h5_ref_at(clu_group["trialtm"], clu_idx)]
session_trial = ... - 1
trialtm = np.asarray(trialtm_ds[()], dtype=np.float64).reshape(-1)
aligned = trialtm - align_times[session_trial]
```

iii. The trajectory notes that spike times are stored as per-cluster `trial`/`trialtm` and that the alignment event is `goCue` (steps 44, 67, 89).

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to `goCue`, histogrammed into `10 ms` bins over `[-3.0, 2.5] s`, converted to rates by dividing by `DT`, and smoothed with `my_smooth`, which applies a one-sided Gaussian-like kernel. The script processes only the single probe specified in each `SessionSpec`.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15

counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
out[:, col] = my_smooth(counts / DT, SMOOTH, "reflect")
```

iii. The AI repeatedly justifies `10 ms`, `[-3.0, 2.5]`, and low-FR filtering as part of the Figure 8 context-task pipeline it chose to emulate (steps 67, 89, 148).

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are dropped if `quality` is one of `{"garbage", "gabrga", "noisy", "real?"}`. Remaining units are kept only if their mean firing rate, computed from smoothed condition-averaged PSTHs, exceeds `1 Hz`.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR = 1.0

if quality in BAD_QUALITIES:
    continue
...
mean_fr = compute_psth_mean_fr(...)
if mean_fr <= LOW_FR:
    continue
```

iii. The trajectory says it is applying the repository quality filter and `>1 Hz` low-FR filter from the manuscript code (steps 67, 89, 148).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike’s trial-specific `goCue` time from `trialtm`.

ii.
```python
aligned = trialtm - align_times[session_trial]
```

iii. The AI explicitly says the exported neural data are `goCue`-aligned (steps 67, 89, 148).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use `10 ms` bins from `-3.0` to `+2.5` s around `goCue`. No later temporal rebinning is applied.

ii.
```python
DT = 1 / 100

def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
    ...
    time = edges[:-1] + DT / 2
```

iii. The AI says it adopted `dt = 10 ms` and `t = [-3.0, 2.5] s` from the Figure 8 scripts (steps 67, 148).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This input is not read from a raw time-series variable. It is constructed from the chosen bin grid around the alignment event, using the `goCue`-aligned time axis.

ii.
```python
def build_edges_and_time() -> tuple[np.ndarray, np.ndarray]:
    ...
    time = edges[:-1] + DT / 2

session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The trajectory treats the time input as part of the same `goCue`-aligned figure pipeline as the neural data rather than as a separately loaded signal (steps 67, 89).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code computes bin centers from the chosen edges and reuses that same vector for every trial in the session.

ii.
```python
edges = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. There is no separate trajectory discussion of this input beyond the AI’s choice of the `10 ms`, `[-3.0, 2.5]` alignment grid (steps 67, 148).

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `time` vector that defines the spike histogram bins, so each input sample corresponds to the same binning grid as the neural data.

ii.
```python
counts, _ = np.histogram(aligned_spikes[spike_mask], bins=edges)
...
session_input = [
    np.asarray(time[None, :], dtype=np.float32)
    for _ in range(neural_arr.shape[2])
]
```

iii. The trajectory says the converter reproduces one shared `goCue`-aligned grid for the exported streams (steps 89, 148).

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp["hit"]` and `bp["R"]`; miss trials are handled implicitly because only hit/miss trials survive `trial_selector`.

ii.
```python
def lick_direction_from_trial(bp: dict[str, np.ndarray], trial_idx: int) -> int:
    hit = bool(bp["hit"][trial_idx])
    is_right_trial = bool(bp["R"][trial_idx])
```

iii. The AI says it “derives actual lick direction from instructed side plus hit/miss outcome” after excluding ignore trials (step 148).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. If a trial is a hit, the output is the instructed side. Otherwise, on the retained miss trials, the output is the opposite side. The exported label set is binary left/right only.

ii.
```python
if hit:
    return int(is_right_trial)
return int(not is_right_trial)
```

iii. The trajectory explicitly describes this derivation as “actual lick direction from instructed side plus hit/miss outcome” (step 148).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Context is derived from `bp["autowater"]`.

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. The trajectory does not separately justify context derivation; it is implicit in the Figure 8 context-task framing the AI chose to follow (steps 67, 148).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code maps `autowater == True` to `WC` (`0`) and everything else to `DR` (`1`).

ii.
```python
context = 0 if bp["autowater"][tr] else 1
```

iii. No separate trajectory justification was given beyond following the context-task roster and trial annotations (steps 67, 148).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]` on the already filtered hit/miss subset; miss trials are the fallback case and ignore/no trials are removed before output construction.

ii.
```python
included_mask = trial_selector(bp)
...
outcome = 1 if bp["hit"][tr] else 0
```

iii. The trajectory explicitly says the exported dataset excludes ignore/no trials (step 148).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is binary: hit trials become `1` (`correct`) and miss trials become `0` (`incorrect`). No ignore class is represented.

ii.
```python
outcome = 1 if bp["hit"][tr] else 0
```

iii. The trajectory ties this directly to the choice to exclude ignore trials from the dataset (step 148).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut `ts` coordinates and `frameTimes` in the bottom camera, specifically the `top_tongue` and `bottom_tongue` landmarks, together with `bitStart`, `sglx.bitcode.bitstart`, `sglx.fs`, and `goCue` for alignment.

ii.
```python
feat_indices = {
    "top_tongue": find_feat_index(f, bottom_group, "top_tongue", bp["Ntrials"]),
    "bottom_tongue": find_feat_index(f, bottom_group, "bottom_tongue", bp["Ntrials"]),
}
...
vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)
```

iii. The trajectory says the remaining design choice was how to collapse multiple tracked tongue landmarks and that it chose “tongue-tip speed” from aligned DLC trajectories (step 73).

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The code interpolates tongue positions onto the chosen time grid, computes finite-difference velocities with `np.gradient`, averages the two tongue landmark velocities componentwise, converts that to speed magnitude, and replaces NaNs with zero.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
...
xvel[:, tr] = np.gradient(tsinterp[:, 0])
yvel[:, tr] = np.gradient(tsinterp[:, 1])
...
tongue_tip_xvel = 0.5 * (top_tongue_xvel + bottom_tongue_xvel)
tongue_tip_yvel = 0.5 * (top_tongue_yvel + bottom_tongue_yvel)
tongue_speed = np.sqrt(tongue_tip_xvel ** 2 + tongue_tip_yvel ** 2)
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
```

iii. The trajectory says it used “the same aligned DLC processing as their MATLAB code, then reducing to tongue-tip speed” to satisfy the decoder target definition (step 73).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code thresholds tongue speed at the session median over strictly positive tongue-speed bins. Values at or above the threshold are class `1`; lower values are class `0`. There is no separate not-visible class.

ii.
```python
positive_tongue = tongue_use[tongue_use > 0]
if positive_tongue.size:
    tongue_thresh = float(np.nanpercentile(positive_tongue, 50))
...
(tongue_speed[:, tr] >= tongue_thresh).astype(np.int16)
```

iii. The AI explicitly justified this as a decoder-specific adjustment to avoid the tongue target collapsing to a constant class when invisible bins had been zero-filled (steps 105, 148).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frames are shifted by a session-wise video offset, then by the trial’s `goCue`, and the tongue trajectories are interpolated onto the same `taxis` grid used for the neural bins.

ii.
```python
vidshift = find_video_offset(bp["bitStart"], bitcode_bitstart, fs)
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The trajectory says it reproduced the reference video-offset correction and `goCue` alignment while keeping tongue velocity on the same exported decoder grid as the neural data (steps 67, 89, 148).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera’s `top_paw` and `bottom_paw` DLC landmarks plus `frameTimes`, `bitStart`, `sglx.bitcode.bitstart`, `sglx.fs`, and `goCue`.

ii.
```python
feat_indices = {
    ...
    "top_paw": find_feat_index(f, bottom_group, "top_paw", bp["Ntrials"]),
    "bottom_paw": find_feat_index(f, bottom_group, "bottom_paw", bp["Ntrials"]),
}
```

iii. The trajectory says the AI chose “mean paw speed” from the aligned DLC geometry rather than a single landmark (step 73).

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw positions are interpolated to the time grid, differentiated with `np.gradient`, offset by a baseline derivative term for non-tongue features, converted to speed separately for the two paw landmarks, averaged, and NaNs are replaced with zero.

ii.
```python
if "tongue" not in feat_name:
    xvel[:, tr] = xvel[:, tr] - basederiv[0]
    yvel[:, tr] = yvel[:, tr] - basederiv[0]
    xvel[:, tr] = fill_nearest_1d(xvel[:, tr])
    yvel[:, tr] = fill_nearest_1d(yvel[:, tr])
...
top_paw_speed = np.sqrt(top_paw_xvel ** 2 + top_paw_yvel ** 2)
bottom_paw_speed = np.sqrt(bottom_paw_xvel ** 2 + bottom_paw_yvel ** 2)
paw_speed = 0.5 * (top_paw_speed + bottom_paw_speed)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The trajectory justifies this only at a high level: it says it reduced the aligned DLC paw tracks to a single mean paw speed per session for the decoder outputs (step 73).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is thresholded at the session median and encoded as a binary low/high label. There is no separate not-visible class.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_use, 50))
...
(paw_speed[:, tr] >= paw_thresh).astype(np.int16)
```

iii. The trajectory says the movement variables were “binarized with per-session medians” (step 148).

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are aligned exactly like tongue trajectories: subtract session video offset and trial `goCue`, then interpolate onto `taxis`, the same neural time grid.

ii.
```python
interp_xy = interp_with_nan(frame_times - vidshift - align_times[tr], xy, taxis)
```

iii. The AI says the movement streams use the same video-offset correction and `goCue` alignment as the rest of the exported decoder dataset (steps 89, 148).

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is read from the standalone `motionEnergy_*.mat` files, specifically `me.data`, with side-camera `frameTimes` used for alignment.

ii.
```python
def load_motion_energy(path: Path) -> tuple[np.ndarray, float]:
    mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    me = mat["me"]
    trial_arrays = np.asarray(me.data, dtype=object).reshape(-1)
```

iii. The trajectory says it used the separate motion-energy files because the embedded field was not consistently present (steps 79, 148).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the framewise motion-energy trace is aligned to `goCue`, interpolated onto the export time grid, and any remaining NaNs are nearest-filled.

ii.
```python
out[:, tr] = interp_with_nan(frame_times - vidshift - align_times[tr], me_trial, taxis)
out[:, tr] = fill_nearest_1d(out[:, tr])
```

iii. The trajectory justifies this generally as part of using video-offset correction and aligned motion/video streams on the same decoder grid (steps 23, 79, 148).

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the session median and exported as binary low/high labels. There is no no-video class.

ii.
```python
me_thresh = float(np.nanpercentile(me_use, 50))
...
(motion_energy[:, tr] >= me_thresh).astype(np.int16)
```

iii. The trajectory says the movement variables were binarized using per-session medians (step 148).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side-camera frame times, subtracts the session video offset and trial `goCue`, and is then interpolated to the same `taxis` grid as the neural data.

ii.
```python
frame_times_by_trial = [read_frame_times(f, side_group, tr) for tr in range(bp["Ntrials"])]
motion_energy = aligned_motion_energy(
    frame_times_by_trial=frame_times_by_trial,
    raw_motion_energy=raw_motion_energy,
    align_times=bp[ALIGN_EVENT],
    vidshift=vidshift,
    taxis=taxis,
)
```

iii. The trajectory says motion energy is aligned with the same video-offset correction and shared decoder time base as the other movement streams (steps 79, 89, 148).

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or gappy movement data are mostly filled rather than preserved as missing. `interp_with_nan` leaves gaps where interpolation is impossible, then `fill_nearest_1d` forward/backward-fills many streams, tongue NaNs are later zero-filled, and if no valid values exist `fill_nearest_1d` returns all zeros. Trials with missing dropped-frame metadata are skipped for that feature.

ii.
```python
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    ...
    if idx.size == 0:
        return np.zeros_like(x)

if np.isnan(ndropped):
    continue
...
tongue_speed = np.nan_to_num(tongue_speed, nan=0.0)
paw_speed = np.nan_to_num(paw_speed, nan=0.0)
```

iii. The trajectory only explicitly comments on one consequence of this handling: zero-filled invisible tongue bins made the median threshold collapse, so the AI changed thresholding to use positive tongue-speed bins only (steps 105, 148).

## 11-a. What are the most time-consuming steps of the code?

i. From the code structure, the heavy steps are per-session HDF5/MAT loading, per-feature interpolation across every trial, and per-cluster spike histogram/smoothing loops. The AI did not explicitly rank runtime hotspots in the trajectory.

ii.
```python
for tr in range(n_trials):
    ...
    interp_xy = interp_with_nan(...)

for clu_idx in range(qds.shape[0]):
    ...
    mean_fr = compute_psth_mean_fr(...)
    ...
    neural_by_trial.append(build_neural_trials(...))
```

iii. No explicit trajectory justification was given beyond reporting successful converter and decoder runs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code leaves several obvious Python loops: trial loops in `aligned_motion_energy`, `aligned_feature_position`, and `aligned_feature_velocity`; condition and trial loops in the firing-rate code; and a cluster loop over all units.

ii.
```python
for tr in range(n_trials):
    ...

for tr in range(xpos.shape[1]):
    ...

for clu_idx in range(qds.shape[0]):
    ...
```

iii. The trajectory does not discuss vectorization tradeoffs explicitly.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations: it histogram-bins each cluster once for the low-FR estimate and again per kept trial for exported neural data; it rescans trial feature-name lists for each requested feature; and it rebuilds the edge/time grid inside every `load_session` call.

ii.
```python
edges, time = build_edges_and_time()
...
mean_fr = compute_psth_mean_fr(...)
...
neural_by_trial.append(build_neural_trials(...))

for trial_idx in range(n_trials):
    names = read_feat_names(f, view_group, trial_idx)
```

iii. The trajectory gives no explicit justification for these repeated computations.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes or stores several values that are not used in the exported dataset: `raw_move_thresh`, `kept_cluster_indices`, `prefilter_unit_count`, `cluster_ids`, unused Bpod fields like `L`, `sample`, and `delay`, and the sample subset pickle. Some defined feature-name lists are also unused.

ii.
```python
raw_motion_energy, raw_move_thresh = load_motion_energy(spec.motion_energy_path)
...
kept_cluster_indices = []
prefilter_unit_count = 0
cluster_ids = []
...
"L": read_h5_vector(bp_group["L"]).astype(np.int16),
"sample": read_h5_vector(ev_group["sample"]),
"delay": read_h5_vector(ev_group["delay"]),
```

iii. The trajectory does not explicitly justify these extra computations; they appear to be convenience or auditing artifacts rather than required conversion steps.
