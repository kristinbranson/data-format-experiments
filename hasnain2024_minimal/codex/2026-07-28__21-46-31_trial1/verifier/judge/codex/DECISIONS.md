# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session `SESSION_SPECS` list and only reads files from `/app/data/Ephys_Behavior`. For each listed session it opens `data_structure_<animal>_<date>.mat` with `h5py` and `motionEnergy_<animal>_<date>.mat` with `scipy.io.loadmat`. It does not search `RandomizedDelay_Ephys_Behavior`, does not load the full 44-session cohort, and does not implement the reference `load_mat` fallback for mixed MATLAB formats.

ii.
```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    ...
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]

EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")
```

```python
data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
with h5py.File(data_path, "r") as f:
    behavior = load_behavior(f)
    probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
    raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
```

iii. The trajectory shows the AI intentionally targeted "the 12 alternating-context ALM sessions used in the context analyses" rather than the full paper cohort. It explicitly said the "paper statistics say the main dataset should be 12 two-context ephys sessions from 6 mice" (step 12), later said it had identified "the 12 alternating-context ALM sessions used in the context analyses" (step 110), and summarized the final conversion as following the "Figure 8 alternating-context ALM cohort" (step 174).

## 1-b. How are the data split into subjects?

i. Subjects are taken directly from `spec["animal"]`. The dataset keeps subjects in first-seen order with an `OrderedDict`, so `subjects` and `subject_idx` reflect the order of `SESSION_SPECS`, not a sorted set from the full cohort.

ii.
```python
subjects = []
subject_lookup = OrderedDict()
...
sid = spec["animal"]
if sid not in subject_lookup:
    subject_lookup[sid] = len(subject_lookup)
    subjects.append(sid)
...
data["subject_idx"].append(subject_lookup[sid])
```

iii. The trajectory justification is indirect: the AI said it was using the 12-session Figure 8 cohort (steps 110 and 174), and its README / notes written during the run describe that cohort as 12 sessions across 7 subject IDs. There is no separate justification for preserving first-seen order rather than sorting.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSION_SPECS`, and each session is one `data_structure_<animal>_<date>.mat` file from `Ephys_Behavior`. Each processed session becomes one element of `data["neural"]`, `data["input"]`, and `data["output"]`.

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The AI justified this as matching the "Figure 8 context analyses" cohort (steps 110 and 174). The trajectory does not show any reconsideration of the broader 44-session loading strategy after that decision.

## 1-d. How are the data split into trials?

i. Trials are taken from the behavioral trial count `bp["Ntrials"]`, with per-trial behavioral arrays read directly from `bp`, spikes assigned by `clu["trial"]`, and video / motion-energy aligned by looping over `trial_idx in range(ntrials)`. The final exported trial list per session is the subset of trial indices that pass `analysis_trial_mask`.

ii.
```python
out = {
    "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
    "R": np.asarray(read_numeric_dataset(bp["R"]), dtype=bool),
    ...
}
```

```python
for trial_idx in range(ntrials):
    ...
```

```python
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
...
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
```

iii. The trajectory justification is limited. The AI said it had confirmed "the behavioral event fields are directly available per trial" and that `goCue` is present per trial (step 55). It did not record a more detailed argument about reconstructing trial boundaries, implying it treated the trial structure in the MAT files as direct ground truth.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps only trials that are `hit` or `miss`, and also excludes `early`, `no`, and `stim.enable` trials. This removes ignored trials entirely instead of keeping them as an `outcome` class, and it does not implement the reference drop of trials that extend past the end of the recording.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

```python
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
```

iii. The trajectory shows the AI believed this matched the repository's analysis filters. Its notes written during the run say the decoder dataset excludes `early`, `no`, and `stim.enable` trials, and claim this matches "recurring `~early`, `~no`, and `~stim.enable` filtering" in the repository. The trajectory itself does not mention the reference solution's retention of ignore trials or its extra recording-length trial cut.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `neural` is derived from the cluster quality labels plus per-spike trial assignments and within-trial spike times from `obj.clu`, aligned with the trial-wise `goCue` times from `bp.ev.goCue`.

ii.
```python
return [
    {
        "quality": qualities[i],
        "trial": np.asarray(trials[i], dtype=np.int64) - 1,
        "trialtm": np.asarray(trialtm[i], dtype=np.float64),
    }
    for i in range(len(qualities))
]
```

```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The AI explicitly said it was checking the video and cluster fields because they "determine the exact kinematic alignment and neuron inclusion logic" (step 55), and earlier summarized the neural path as go-cue alignment plus low-rate filtering (step 25).

## 2-b. How is the `neural` data processed?

i. For each kept cluster, spikes are binned trial by trial into a `[-3.0, 2.5]` second window with `10 ms` bins, converted to rate by dividing by `DT`, and smoothed with a custom causal Gaussian-like kernel `my_smooth(..., SMOOTH=15, BCTYPE="reflect")`. The exported neural data are those smoothed per-trial firing rates.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
```

```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The trajectory explicitly states the chosen processing path: "trials are aligned to `goCue`, binned at `dt=0.01` s, causally smoothed with `smooth=15`" (step 25), and later repeats "go-cue alignment, 10 ms bins, causal smoothing" as the reference preprocessing it intended to follow (steps 110 and 174).

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first drops clusters whose quality string is exactly one of `garbage`, `gabrga`, `noisy`, or `real?`. It keeps everything else, including `poor`. It then computes seven condition-averaged PSTHs and retains units whose mean firing rate across time and those condition PSTHs exceeds `1 Hz`.

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}
```

```python
psth = np.zeros((TIME_AXIS.size, len(conditions)), dtype=np.float64)
for ci, mask in enumerate(conditions):
    trix = np.flatnonzero(mask)
    if trix.size:
        psth[:, ci] = np.mean(trial_counts[:, trix], axis=1)
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. The AI justified the low-rate rule explicitly in step 25 as "mean PSTH firing rate >1 Hz", and in its notes it claimed it was reproducing the Figure 8 seven-condition low-FR pipeline. There is no trajectory justification for omitting `poor` from the quality drop list.

## 2-d. How is the per-trial `neural` data aligned to the event described in the instructions?

i. Each spike time is aligned by subtracting the go-cue time for that spike's trial, i.e. `trialtm - goCue[trial]`.

ii.
```python
trial_idx = clu["trial"]
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The AI repeatedly stated that the reference neural alignment event was `goCue` (steps 25, 55, 110, and 174), and used that as one of its main anchoring decisions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use `10 ms` bins from `-3.0 s` to `+2.5 s`, for 550 time bins per trial. The AI bins spikes directly onto that grid; there is no later rebinning step.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The trajectory explicitly records this choice several times: step 25 says `dt=0.01`, and steps 110 and 174 summarize the pipeline as using "10 ms bins".

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input channel is not read from a raw numeric array. It is constructed as the center of the chosen aligned time bins, with `goCue` used only to define the alignment event for all streams.

ii.
```python
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The trajectory justification is implicit rather than explicit: the AI consistently described the decoder input as the neural time axis aligned to `goCue` (steps 110, 174, and the notes it wrote during the run).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computes bin centers from its chosen `[-3.0, 2.5]` second, `10 ms` time grid and repeats that same 1-by-T vector for every trial.

ii.
```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The trajectory provides no separate justification beyond the broader decision to use the same aligned neural time axis for the decoder dataset (steps 110 and 174).

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the exact same time axis used to bin spikes, so timepoint `k` in `input` corresponds to timepoint `k` in `neural`.

ii.
```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The AI stated that the video streams were "resampled onto the neural time axis" (step 110), and it used that same axis as the sole decoder input, so the alignment choice is explicit in the trajectory.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `lick_direction` is derived from the per-trial `R`, `L`, `hit`, and `miss` behavioral flags.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. In its notes, the AI described this output as the "actual lick direction, not instructed side" and defined right as `R & hit` or `L & miss`, with left as the converse. The trajectory itself does not give a more detailed rationale.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI reduces lick direction to a binary label: right is coded as `1`, and everything else in the kept trials is left (`0`). Because `analysis_trial_mask` removes `no` trials, it does not represent a separate no-lick class.

ii.
```python
lick_dir = actual_lick_direction(behavior)[use_trials]
```

```python
return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. The trajectory justification is indirect. The AI's written notes during the run say it intentionally used "actual lick direction" for the kept hit/miss trials, and its trial filter excludes `no` trials entirely, which is how it avoids a third class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `behavioral_context` is derived from the per-trial `autowater` flag.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. The AI did not record a detailed justification in the trajectory beyond treating the alternating DR/WC context analyses as the target cohort and labeling the output as behavioral context in its notes.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. It maps `autowater=True` to `WC=0` and `autowater=False` to `DR=1` by storing `~autowater` as the output code.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. The AI's notes explicitly define `0 = WC` and `1 = DR`; there is no separate trajectory message disputing or elaborating that mapping.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The exported `outcome` is derived from the `hit` flag after pre-filtering trials down to hit/miss only. `miss` and `no` matter for the trial mask, but the stored output is just `hit` converted to `0/1`.

ii.
```python
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
...
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The trajectory justification is again implicit through the chosen trial filter: by excluding `no` trials up front, the AI reduced `outcome` to incorrect/correct only. Its notes written during the run define `0 = incorrect` and `1 = correct` with no ignore class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI stores `outcome` as a binary label, with miss trials becoming `0` and hit trials `1` because only hit/miss trials are retained.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The trajectory does not contain a separate justification beyond the earlier decision to exclude ignore / no-response trials from the dataset.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the side-camera `tongue` trajectory in `obj.traj`, using its `ts` coordinates and `frameTimes`. Video timing also uses `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, `bp.ev.bitStart`, and `bp.ev.goCue` to compute the alignment offset.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
```

```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. The trajectory says the AI believed the reference code "fills non-tongue gaps with nearest values, and sets invisible tongue velocity to zero" (step 89). It did not record any decision to use the second tongue view (`top_tongue`), so the implemented choice is effectively side-view only.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI linearly interpolates tongue x/y coordinates onto the common neural time axis, computes x and y gradients with `np.gradient`, replaces non-finite tongue velocities with zero, and takes speed magnitude `sqrt(xvel^2 + yvel^2)`. It does not smooth by frame time, does not use likelihood runs, and does not normalize / average two tongue views.

ii.
```python
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

```python
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
...
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
```

iii. The AI justified the zero-filling behavior explicitly in step 89 and later in step 138, where it said invisible tongue periods become zero and can collapse the session median to zero. The rest of the processing is only justified indirectly by its claim that it was following the repository's tongue-velocity path.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. It is binarized per session at the 50th percentile. If the all-sample median is `<= 0`, the AI recomputes the threshold using only positive tongue-speed values. The exported labels are only `0` and `1`; there is no separate `not visible` class.

ii.
```python
def percentile_threshold(values: np.ndarray, drop_zeros_if_needed: bool = False) -> float:
    ...
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh
```

```python
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
...
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. Step 138 gives the explicit justification: the verifier found tongue labels were becoming effectively constant because invisible periods were zero, so the AI changed the thresholding rule to use positive tongue-speed samples when the session median was zero.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a session-wide video offset, subtracts that offset and the trial's `goCue` from each frame time, then interpolates tongue position onto the same `TIME_AXIS` used for neural data.

ii.
```python
return matlab_mode(bitstart) / fs - matlab_mode(behavior["ev"]["bitStart"])
```

```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
vals = interp(time_axis)
```

iii. The trajectory states this decision clearly: step 89 says the reference aligns video by `frameTimes - video_offset - goCue` and resamples it onto the neural time axis; steps 110 and 174 repeat that the video streams were aligned / resampled to the neural time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from a bottom-view paw feature in `obj.traj`. The code prefers `top_paw`, but will fall back to `bottom_paw` if needed.

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    ...
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name
```

```python
paw_feature = choose_paw_feature(f)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The trajectory does not contain a separate justification beyond the general claim that it was following the repository's kinematic streams. Its notes written during the run describe the exported paw signal as coming from `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates bottom-view paw x/y coordinates onto the neural time axis, computes gradients with `np.gradient`, subtracts `basederiv[0]` from both x and y velocities, nearest-fills missing values, and takes the speed magnitude.

ii.
```python
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
if "tongue" not in feature_name:
    xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
    yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
    xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
    yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
```

```python
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The trajectory justification is indirect. Step 89 says the AI thought non-tongue gaps should be nearest-filled, and its notes say the non-tongue baseline subtraction reproduces the MATLAB implementation, including subtracting `basederiv(1)` from both velocity components.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is thresholded at the per-session 50th percentile into a binary `0/1` output. Missing periods are nearest-filled before thresholding, so no third visibility class is emitted.

ii.
```python
paw_thresh = percentile_threshold(paw_selected)
...
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. The trajectory does not provide a separate justification for omitting a not-visible class; the AI's notes simply describe a per-session median split for paw velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The paw stream uses the same video-offset correction and go-cue subtraction as the tongue stream, then interpolates onto the same `TIME_AXIS` used by neural activity.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
vals = interp(time_axis)
```

iii. The trajectory's alignment justification is the same as for tongue: step 89 says video is aligned as `frameTimes - video_offset - goCue` and resampled onto the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from `motionEnergy_<animal>_<date>.mat` using the `me.data` field, and aligned using side-camera `frameTimes` from `obj.traj`.

ii.
```python
dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
raw = dat.data
if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
    raw = raw.data
```

```python
traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
view_group = f[traj_refs[0]]
frame_refs = np.array(view_group["frameTimes"]).reshape(-1, order="F")
```

iii. Step 120 shows the AI explicitly debugging the motion-energy loader and step 122 says it tightened the loader to unwrap only MATLAB struct wrappers. Its notes later state that motion energy comes from `motionEnergy_Animal_Date.mat`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates the per-frame motion-energy trace onto the common neural time axis and nearest-fills missing edge values. It does not bin by frame occupancy; it resamples directly to one value per neural timepoint.

ii.
```python
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

iii. The trajectory justification is limited to step 89 / step 110, where the AI says the video streams should be resampled onto the neural time axis, and its notes say motion energy is "resampled onto the neural time axis, and is nearest-filled at the edges."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded at the session median into a binary `0/1` output. There is no separate no-video category.

ii.
```python
me_thresh = percentile_threshold(me_selected)
...
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. The trajectory does not record any separate rationale beyond the overall requirement to discretize motion energy by per-session 50th percentile.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the same video offset and go-cue subtraction as the other camera streams, then it is interpolated onto the shared neural `TIME_AXIS`.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. The trajectory justification is the same as for tongue and paw: step 89 says the video-derived streams are aligned by `frameTimes - video_offset - goCue` and resampled onto the neural time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing values are mostly filled rather than preserved. If `sglx` / `bitcode` is missing, the video offset defaults to `0.5` s. Non-tongue trajectories and motion energy are nearest-filled. Trials with all-missing trajectories become zeros after `fill_nearest`. Tongue non-finite velocity samples are set to zero, and the tongue threshold is adjusted to ignore zeros only if the median collapses to zero.

ii.
```python
def get_video_offset(f: h5py.File, behavior: dict) -> float:
    if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
        return 0.5
```

```python
if not mask.any():
    return np.zeros_like(x)
```

```python
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
...
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
```

iii. The trajectory explicitly justifies the tongue-zero handling in steps 89 and 138. The other missing-data decisions are not directly justified in the trajectory beyond the AI's claim that it was following repository behavior and trying to keep the verifier / decoder outputs sensible.

## 11-a. What are the most time-consuming steps of the code?

i. The AI did not explicitly state this in the trajectory. From the code structure, the most time-consuming steps are likely the nested per-cluster spike binning loop and the per-trial interpolation / gradient passes for tongue, paw, and motion energy, because they iterate over every kept trial or every spiking cluster in every session.

ii.
```python
for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            mask = trial_idx == t
            counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
            trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

```python
for trial_idx in range(ntrials):
    ...
    vals = interp(time_axis)
```

iii. The trajectory gives no explicit runtime analysis. The closest justification is practical: the AI repeatedly waited for full-session builds to finish (steps 132 and 135) and treated the end-to-end conversion as the main runtime bottleneck.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops are left scalarized: the per-cluster / per-trial spike histogramming loop, the per-trial interpolation loop for trajectories and motion energy, and the per-trial gradient loop for velocity. The AI did not attempt a vectorized all-trials spike histogram like the reference `histogram2d`.

ii.
```python
for t in unique_trials:
    mask = trial_idx == t
    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

```python
for trial_idx in range(ntrials):
    ...
    interp = interp1d(...)
```

iii. There is no explicit trajectory justification for keeping these loops. The AI focused on getting the MATLAB-compatible behavior working across file-format variations rather than optimizing the implementation.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several similar passes: separate interpolation code for tongue, paw, and motion energy; per-trial label tiling for every output; and creation of both full and sample datasets from the same assembled data. Within a session, it also recomputes `fill_nearest`-style gap filling on multiple streams.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(...)
paw_xpos, paw_ypos = load_traj_feature_series(...)
motion_energy = align_motion_energy(...)
```

```python
output = np.vstack(
    [
        repeat_labels(...),
        repeat_labels(...),
        repeat_labels(...),
        ...,
    ]
)
```

iii. The trajectory does not contain an explicit justification for this repeated processing. The closest rationale is that the AI wanted both a full dataset and a sample dataset for validation, and separate handling for each output stream was simpler while debugging schema differences.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores extra metadata that the decoder does not need: `single_unit_flags`, `kept_quality`, `manual_motion_thresh`, several event-time summaries, extensive `session_stats`, `reference_processing`, and `sanity_checks`. It also contains logic to build `sample_data.pkl`, which is outside the required final full dataset.

ii.
```python
session_stats = {
    ...
    "manual_motion_energy_move_thresh": manual_motion_thresh,
    "quality_labels_kept": kept_quality,
    "usable_trial_indices_0based": use_trials.tolist(),
    "event_times_sec_relative_to_go_cue": {...},
}
```

```python
def make_sample_dataset(full_data: dict, session_count: int) -> dict:
    sample = copy.deepcopy(full_data)
    ...
```

iii. The trajectory justifies this as validation and documentation work rather than decoder necessity. The AI said it wanted to inspect counts, compare against paper statistics, and write notes / README around those concrete results (steps 135 and 146-like summaries in the later notes-writing phase), so these extra computations were used for sanity checks, not for the downstream decoder itself.
