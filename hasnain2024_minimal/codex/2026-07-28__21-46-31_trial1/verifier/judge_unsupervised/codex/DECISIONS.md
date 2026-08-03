# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter does not scan the dataset dynamically. It hard-codes a 12-session cohort in `SESSION_SPECS`, then loops over that list. For each session it opens one HDF5-backed MATLAB file from `/app/data/Ephys_Behavior`, loads behavior, probe clusters, motion energy, tongue trajectories, and paw trajectories, and finally appends the resulting per-session trial lists into the output dataset.

ii. ```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    ...
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]

DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")

for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    ...

def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
        neural_all, kept_quality, single_flags = bin_spikes_for_session(behavior, probes)
        raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
        motion_energy = align_motion_energy(f, behavior, raw_motion_energy)
        tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
        paw_feature = choose_paw_feature(f)
        paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The justification in `CONVERSION_NOTES.md` is that the cohort “follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`.” The trajectory also says the agent “pinned the reference processing path and the likely target cohort” to the 12 alternating-context ALM sessions.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are defined only by the `animal` field in each `SESSION_SPECS` entry. While building the dataset, the script creates a unique ordered list of animal IDs and a `subject_idx` entry per session.

ii. ```python
def build_dataset() -> tuple[dict, dict]:
    subjects = []
    subject_lookup = OrderedDict()
    ...
    for spec in SESSION_SPECS:
        payload = build_session_payload(spec)
        sid = spec["animal"]
        if sid not in subject_lookup:
            subject_lookup[sid] = len(subject_lookup)
            subjects.append(sid)
        ...
        data["subject_idx"].append(subject_lookup[sid])
```

iii. The agent’s justification was to preserve the Figure 8 session loader subset exactly. In `CONVERSION_NOTES.md` it explicitly notes a discrepancy: the loader subset yields 12 sessions across 7 animal IDs, whereas the copied methods text says 12 sessions from 6 mice.

## 1-c. How are the data split into sessions?

i. Each row of `SESSION_SPECS` is treated as one session. A session is identified by the `animal` and `date` pair, plus one or more listed probe numbers.

ii. ```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    {"animal": "JEB7", "date": "2021-04-29", "probes": [1]},
    ...
]

for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. `CONVERSION_NOTES.md` says this session list matches the Figure 8 context-analysis loaders. The trajectory shows the agent deliberately anchored the cohort to those scripts rather than discovering sessions dynamically.

## 1-d. How are the data split into trials?

i. Trials come from the behavior file’s `Ntrials` and boolean trial-level vectors. Neural spikes already carry per-spike trial IDs; kinematic and motion-energy arrays are interpolated into `[time, trial]` matrices. After quality-control masking, the code iterates over `use_trials` and emits one trial matrix at a time into each session list.

ii. ```python
def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        ...
    }

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
...
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
    input_trials.append(TIME_AXIS[None, :].astype(np.float32))
    ...
    output_trials.append(output.astype(np.int64))
```

iii. The justification is implicit: the target format requires lists of trial matrices per session, and the reference processing is trial-aligned around `goCue`. The notes emphasize that all sessions have a common `T=550` bins after alignment.

## 1-e. How are trials filtered based on quality controls?

i. The decoder dataset keeps only `hit` or `miss` trials and excludes `early` trials, `no` trials, and `stim.enable` trials. The resulting mask is applied after the full neural and kinematic matrices are built.

ii. ```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
if use_trials.size < 2:
    raise RuntimeError(f"{spec['animal']} {spec['date']} has fewer than 2 usable trials")
```

iii. `CONVERSION_NOTES.md` says this matches the “recurring `~early`, `~no`, and `~stim.enable` filtering used in the repository analyses,” and also notes that early lick and ignore trials are omitted in the methods text.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from each cluster’s spike times and trial IDs in the `obj.clu` structure, together with the per-trial `goCue` times from `obj.bp.ev.goCue` for alignment.

ii. ```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
    return [
        {
            "quality": qualities[i],
            "trial": np.asarray(trials[i], dtype=np.int64) - 1,
            "trialtm": np.asarray(trialtm[i], dtype=np.float64),
        }
        for i in range(len(qualities))
    ]

aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The notes say the conversion reproduces the Figure 8 neural path: spike trains are aligned to `goCue`, smoothed, then low-FR-filtered.

## 2-b. How is the `neural` data processed?

i. For each neuron and trial, the script bins aligned spike times into 10 ms bins from `-3.0` to `2.5` s around `goCue`, converts counts to rate by dividing by `DT`, and smooths with a causal Gaussian kernel implemented to match MATLAB `mySmooth(..., 'reflect')`.

ii. ```python
DT = 1 / 100
SMOOTH = 15
BCTYPE = "reflect"
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)

counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. `CONVERSION_NOTES.md` explicitly justifies this as matching the reference: “10 ms” bins, `[-3.0 s, 2.5 s]`, and “causal Gaussian kernel with `N=15`, matching `mySmooth(..., 15, 'reflect')`.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script first removes clusters whose quality label is `garbage`, `gabrga`, `noisy`, or `real?`. It then computes seven context-condition PSTHs and removes any remaining unit whose condition-averaged mean firing rate is `<= 1 Hz`.

ii. ```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

conditions = get_context_conditions(behavior)
...
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. The notes justify this directly as reproducing the MATLAB path: “all qualities except `garbage`, `gabrga`, `noisy`, `real?`,” then the same 7 condition PSTHs, then “drop units whose mean across time and conditions is not above 1 Hz.”

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting the trial’s `goCue` time, which is also the alignment event named in the instructions.

ii. ```python
ALIGN_EVENT = "goCue"
...
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The agent’s notes say “Alignment event: `goCue`,” and the trajectory says it anchored the whole pipeline to go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data has 10 ms bins. The code does not perform any later rebinning; it directly bins spikes on the final target axis.

ii. ```python
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
"time_bin_size": float(DT * 1000.0),
```

iii. `CONVERSION_NOTES.md` explicitly records “Bin size: `10 ms`.” The trajectory likewise describes the reference path as using “10 ms bins.”

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not read from a dedicated raw variable. It is derived from the fixed aligned time axis defined by `TMIN`, `TMAX`, and `DT`, with `goCue` chosen as the alignment event for all streams.

ii. ```python
ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The notes justify this as a decoder design choice: the one decoder input channel is “equal to the aligned neural time axis.”

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script constructs the time axis as bin midpoints over the global alignment window and simply reuses that same 1-by-T array for every trial.

ii. ```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. `CONVERSION_NOTES.md` says the input is “one time-varying input channel equal to the aligned neural time axis.”

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is perfectly co-registered with neural data because it reuses the exact same `TIME_AXIS` that was used to bin the neural firing-rate traces.

ii. ```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The justification is implicit in the code and made explicit in the notes: all streams are aligned to `goCue` on the common neural time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the behavior booleans `R`, `L`, `hit`, and `miss`.

ii. ```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` states that lick direction is defined as “actual lick direction, not instructed side,” and then gives the same boolean logic.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script converts trial outcomes into actual response side: right is encoded when the animal was on a right trial and correct, or on a left trial and incorrect; left is the complement on the selected hit/miss trials. It then tiles that scalar label across all time bins.

ii. ```python
lick_dir = actual_lick_direction(behavior)[use_trials]
...
repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The notes justify the use of actual lick side rather than instructed side because that is the behavior to decode.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` trial flag.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. The notes explicitly define the mapping as `0 = WC`, `1 = DR`. The trial-condition definitions used for low-FR filtering also split trials with `autowater` versus `~autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The script turns `autowater` into a binary context code by negating it: `autowater == True` becomes WC (`0`) and `autowater == False` becomes DR (`1`). The label is then repeated across time bins.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
...
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The justification in the notes is that the alternating-context Figure 8 analyses use this DR versus AW/WC split.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the `hit` boolean after restricting to `hit`/`miss` analysis trials.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The notes define `0 = incorrect`, `1 = correct`, which corresponds to miss versus hit on the retained trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The script casts `hit` to `1` and, because analysis trials are already restricted to `hit` or `miss`, all remaining non-hit trials become `0` for incorrect. The label is then tiled across time.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
...
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The justification is straightforward and is recorded in `CONVERSION_NOTES.md` as `0 = incorrect`, `1 = correct`.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-view trajectory feature named `tongue`, using its per-frame 2D coordinates from the trajectory `ts` array together with `frameTimes` and the trial `goCue` time.

ii. ```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
...
frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
coords = ts[:, 0:2, feat_idx]
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The notes justify this as “scalar tongue speed from the reference x/y tongue velocity components.” The trajectory says the agent checked the reference video-alignment path before deciding on the output bins.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The script linearly interpolates tongue x/y positions onto the common aligned neural time axis, computes x/y velocity with `np.gradient`, sets invisible-tongue samples to zero velocity, and then takes speed magnitude `sqrt(xvel^2 + yvel^2)`.

ii. ```python
def compute_velocity_from_position(xpos: np.ndarray, ypos: np.ndarray, feature_name: str):
    ...
    xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
    yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
    ...
    else:
        xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
        yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0

...
tongue_xvel, tongue_yvel = compute_velocity_from_position(tongue_xpos, tongue_ypos, "tongue")
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. `CONVERSION_NOTES.md` says the code follows the repository logic: tongue invisibility becomes zero velocity, and the output is scalar tongue speed.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The script uses a per-session 50th-percentile threshold on all selected tongue-speed samples, except when that median is `<= 0`; in that case it recomputes the threshold from only the positive tongue-speed samples.

ii. ```python
def percentile_threshold(values: np.ndarray, drop_zeros_if_needed: bool = False) -> float:
    ...
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh

...
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
...
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. The justification is explicit in both the notes and trajectory: because invisible tongue samples are set to zero, the ordinary median could collapse to zero and create a trivial constant label. The agent therefore added the positive-only fallback.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue position traces are aligned by subtracting the session’s video offset and the trial’s `goCue`, interpolating onto `TIME_AXIS`, then computing velocity on that aligned axis.

ii. ```python
vidshift = get_video_offset(f, behavior)
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
interp = interp1d(
    shifted_t[valid],
    coords[valid, dim],
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)
vals = interp(time_axis)
```

iii. The notes explicitly justify this as matching the reference video path: `frameTimes - video_offset - goCue`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from a bottom-view trajectory marker. The code chooses `top_paw` if present, otherwise `bottom_paw`, then uses that feature’s 2D positions from the trajectory `ts` array together with `frameTimes` and `goCue`.

ii. ```python
def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]
    first_feats = deref_string_list(f, f[np.array(view_group["featNames"]).reshape(-1, order="F")[0]])
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name
    raise KeyError("No paw feature found in bottom-view trajectory data")

paw_feature = choose_paw_feature(f)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The notes justify this as “scalar paw speed from the `top_paw` bottom-view marker.” The trajectory shows the agent inspected feature names before choosing a paw signal.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script linearly interpolates paw positions onto the aligned neural time axis, subtracts a baseline derivative from both x and y velocity components, nearest-fills missing values, and then takes the speed magnitude.

ii. ```python
if "tongue" not in feature_name:
    xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
    yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
    xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
    yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
...
paw_xvel, paw_yvel = compute_velocity_from_position(paw_xpos, paw_ypos, paw_feature)
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The notes justify this as reproducing the repository’s non-tongue velocity path, including the baseline subtraction quirk where `basederiv(1)` is subtracted from both `xvel` and `yvel`.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. It is binarized with the 50th percentile of all selected paw-speed samples within each session.

ii. ```python
paw_thresh = percentile_threshold(paw_selected)
...
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. The notes justify this directly: “binarized with the per-session 50th percentile.”

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The alignment mirrors tongue velocity: bottom-view `frameTimes` are shifted by the video offset and the trial `goCue`, then positions are interpolated onto the neural `TIME_AXIS` before velocities are computed.

ii. ```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
vals = interp(time_axis)
...
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The notes justify the shared video alignment rule as matching the reference code’s `frameTimes - video_offset - goCue` path.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_{animal}_{date}.mat` file’s `me.data` field, along with video frame times from the first trajectory view and the trial `goCue` times.

ii. ```python
def load_motion_energy(animal: str, date: str, behavior: dict) -> tuple[np.ndarray, float]:
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    ...
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. The notes say motion energy “comes from `motionEnergy_Animal_Date.mat`.” The methods excerpt in the trajectory also describes motion energy as a per-frame video-derived signal.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The script linearly interpolates each trial’s motion-energy trace onto the neural `TIME_AXIS`, nearest-fills edge NaNs, and keeps the continuous aligned value until final discretization.

ii. ```python
interp = interp1d(
    shifted_t[valid],
    me_trial[valid],
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)
aligned[:, trial_idx] = interp(TIME_AXIS)
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. `CONVERSION_NOTES.md` says motion energy is “resampled onto the neural time axis, and is nearest-filled at the edges.”

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. It is binarized with a per-session 50th-percentile threshold over all selected motion-energy samples.

ii. ```python
me_thresh = percentile_threshold(me_selected)
...
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. The justification is the decoder-task requirement in the instructions. The notes also state that the output is “binarized with the per-session 50th percentile.”

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by shifting video frame times by the session video offset and each trial’s `goCue`, then interpolating onto the same `TIME_AXIS` used for the neural data.

ii. ```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. The notes explicitly justify this as matching the reference `loadMotionEnergy` logic.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script uses several defensive fallbacks. Missing or all-NaN dropped-frame metadata causes a kinematic trial to be skipped. Missing frame times for motion energy fall back to a synthetic `np.arange(...)/400` time base. Missing non-tongue samples are nearest-filled; missing tongue velocity samples are set to zero; completely missing trajectories become zeros after fill logic. If `sglx.bitcode.bitstart` is missing, the script assumes a `0.5` s video offset.

ii. ```python
if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
    return 0.5
...
if ndropped.size and np.isnan(ndropped).all():
    continue
...
except Exception:
    ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
...
if not mask.any():
    return np.zeros_like(x)
...
else:
    xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
```

iii. The notes justify several of these choices by reference to MATLAB behavior: nearest-filling non-tongue signals, zeroing invisible tongue velocity, and using the frame-time fallback path seen in `loadMotionEnergy`. The extra `0.5` s default video offset is only implicitly justified by the same fallback convention and is not called out separately.

## 11-a. What are the most time-consuming steps of the code?

i. The slowest parts are the nested per-neuron, per-trial spike binning and smoothing in `bin_spikes_for_session`, plus the repeated per-trial interpolation of tongue, paw, and motion-energy traces onto `TIME_AXIS`.

ii. ```python
for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            mask = trial_idx == t
            counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
            trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)

for trial_idx in range(ntrials):
    ...
    interp = interp1d(...)
    vals = interp(time_axis)
```

iii. This is an inference from the code structure, not a stated design note. The heaviest work sits inside Python loops over many clusters, trials, and interpolations.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious candidates are the cluster-by-cluster and trial-by-trial spike histogram loop, the trial loop in `load_traj_feature_series`, the trial loop in `align_motion_energy`, the trial loop in `compute_velocity_from_position`, and the final loop that constructs each trial’s output matrix one trial at a time.

ii. ```python
for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            ...

for trial_idx in range(ntrials):
    ...

for trial_idx in range(xpos.shape[1]):
    ...

for local_idx, trial_idx in enumerate(use_trials):
    ...
```

iii. This is also an inference from the code. The agent did not justify these loops as intentional; they are just the current implementation strategy.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly interpolates separate behavioral streams to the same `TIME_AXIS`, repeatedly applies `fill_nearest`, repeatedly computes constant per-trial labels and tiles them across time, and repeatedly rebuilds the same `TIME_AXIS[None, :]` input for every trial.

ii. ```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
...
output = np.vstack(
    [
        repeat_labels(...),
        repeat_labels(...),
        repeat_labels(...),
        ...
    ]
)
...
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])
```

iii. This is an inference from the implementation. It is not separately justified in the notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes or loads several things that are only saved as metadata or not used at all by the decoder: `kept_quality`, `single_flags`, `manual_motion_thresh`, event-time summaries relative to go cue, and extensive session statistics. It also loads the motion-energy manual threshold even though discretization uses the 50th percentile instead.

ii. ```python
raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
...
"manual_motion_energy_move_thresh": manual_motion_thresh,
"quality_labels_kept": kept_quality,
"event_times_sec_relative_to_go_cue": {
    ...
},
...
"sanity_checks": {
    ...
},
```

iii. This is mostly a byproduct of the agent documenting its own conversion and sanity checks. `CONVERSION_NOTES.md` emphasizes those validation statistics, but the downstream decoder itself only consumes the core `neural`, `input`, and `output` arrays plus indexing metadata.
