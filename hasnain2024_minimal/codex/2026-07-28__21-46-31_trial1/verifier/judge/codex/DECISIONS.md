# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-coded a 12-session subset in `SESSION_SPECS`, restricted loading to `/app/data/Ephys_Behavior`, opened each `data_structure_<animal>_<date>.mat` with `h5py.File`, and loaded motion energy from a separate `motionEnergy_<animal>_<date>.mat` in the same folder. It did not search both task folders, did not use the full 44-session cohort, and did not implement a fallback MATLAB-v5 loader for the main session files.

ii. ```python
SESSION_SPECS = [
    {"animal": "JEB6", "date": "2021-04-18", "probes": [2]},
    ...
    {"animal": "JEB19", "date": "2023-04-18", "probes": [1]},
]

DATA_DIR = "/app/data"
EPHYS_DIR = os.path.join(DATA_DIR, "Ephys_Behavior")

def build_session_payload(spec: dict) -> dict:
    data_path = os.path.join(EPHYS_DIR, f"data_structure_{spec['animal']}_{spec['date']}.mat")
    with h5py.File(data_path, "r") as f:
        behavior = load_behavior(f)
        probes = [load_probe_clusters(f, probe_number) for probe_number in spec["probes"]]
```

iii. In `CONVERSION_NOTES.md`, the agent says it targeted the 12-session alternating-context Figure 8 cohort and that the session list follows `Figure8a_thru_c.m` and `Figure8d.m`. In the trajectory it says it had 'pinned the reference processing path and the likely target cohort: the 12 alternating-context ALM sessions'.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the hard-coded `animal` field in each session spec. During assembly, subjects are recorded in first-seen order with an `OrderedDict`, and `subject_idx` stores that insertion-order index for each session.

ii. ```python
sid = spec["animal"]
if sid not in subject_lookup:
    subject_lookup[sid] = len(subject_lookup)
    subjects.append(sid)
...
data["subject_idx"].append(subject_lookup[sid])
```

iii. The notes justify the cohort as a 12-session Figure 8 subset and acknowledge that it yields 7 subject IDs rather than the 6 mice mentioned in the methods text; the agent chose to keep the code-defined subset rather than reconcile to the paper text.

## 1-c. How are the data split into sessions?

i. Each element of `SESSION_SPECS` is treated as one session. One session corresponds to one `data_structure_<animal>_<date>.mat` file from `Ephys_Behavior`, with the listed probe IDs used for that session. The 12 sessions are then appended one-by-one into `data['neural']`, `data['input']`, and `data['output']`.

ii. ```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The agent explicitly says the conversion 'follows the Figure 8 alternating-context ALM cohort' and uses the 12 sessions from those scripts as the session definition.

## 1-d. How are the data split into trials?

i. Trials are indexed directly by the per-trial arrays in `bp`, using `Ntrials` as the trial count. Neural spikes already carry a trial index (`clu['trial']`), and video/motion-energy arrays are iterated trial-by-trial. Final trial inclusion is determined by `use_trials = np.flatnonzero(analysis_trial_mask(behavior))`.

ii. ```python
out = {
    "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
    "R": np.asarray(read_numeric_dataset(bp["R"]), dtype=bool),
    ...
}

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
neural_selected = neural_all[:, :, use_trials]
...
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
```

iii. The agent treated the Bpod trial table and per-trial trial indices in the raw data as the authoritative trial boundaries; there is no attempt to reconstruct trials from continuous recordings.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only hit or miss trials and excludes `early`, `no`, and `stim.enable` trials through `analysis_trial_mask`. It does not keep ignore/no-response trials as a third outcome class, and it does not implement the reference solution's extra cutoff for trials after the neural recording stopped.

ii. ```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])

use_trials = np.flatnonzero(analysis_trial_mask(behavior))
```

iii. `CONVERSION_NOTES.md` says this matches the repository's recurring `~early`, `~no`, and `~stim.enable` filtering and was chosen to keep correct and incorrect response trials in both contexts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from `obj.clu` for the specified probe(s): cluster `quality`, spike `trial`, and spike `trialtm`. Alignment also uses `bp.ev.goCue`.

ii. ```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
```

iii. The notes describe the neural pipeline as the Figure 8 preprocessing path with go-cue alignment and low-firing-rate exclusion.

## 2-b. How is the `neural` data processed?

i. For each kept cluster, the code subtracts the go-cue time from each spike time, bins spikes into `TIME_EDGES` spanning `[-3.0, 2.5]` s at `DT = 1/100` (10 ms), converts counts to Hz by dividing by `DT`, smooths each trial with a custom causal Gaussian-like kernel `my_smooth(..., 15, 'reflect')`, and stores the resulting per-trial firing-rate matrix.

ii. ```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15

aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
...
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. The agent says it was reproducing the Figure 8 preprocessing: 'go-cue alignment, 10 ms bins, causal smoothing, low-FR exclusion.'

## 2-c. How is the `neural` data filtered based on quality controls?

i. Clusters are dropped if `quality` is exactly one of `garbage`, `gabrga`, `noisy`, or `real?`. Surviving clusters are then filtered by mean firing rate computed from condition-averaged PSTHs across seven Figure 8 context conditions, keeping only units with mean PSTH firing rate `> 1 Hz`. The code does not exclude `poor` clusters.

ii. ```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
    return quality not in {"garbage", "gabrga", "noisy", "real?"}

conditions = get_context_conditions(behavior)
...
mean_fr = float(np.mean(psth))
if mean_fr > LOW_FR_HZ:
    kept_neural.append(trial_counts)
```

iii. `CONVERSION_NOTES.md` says the low-FR rule was matched to the MATLAB Figure 8 path by averaging across seven context conditions before applying the 1 Hz cutoff.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike's trial-specific go-cue time from `clu['trialtm']`. The aligned spikes are then histogrammed on the common time grid.

ii. ```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
```

iii. The notes and final trajectory summary both state that the temporal alignment event is `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins (`DT = 1/100`) over a `[-3.0, 2.5]` s window, giving 550 time bins. Spikes are directly histogrammed into those bins; there is no secondary rebinning step afterwards.

ii. ```python
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The notes explicitly say 'Time window: `[-3.0 s, 2.5 s]`' and 'Bin size: `10 ms`'.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw variable directly. The agent defines a synthetic time axis from `TMIN`, `TMAX`, and `DT`, with go cue chosen as the alignment event via `ALIGN_EVENT = 'goCue'`.

ii. ```python
ALIGN_EVENT = "goCue"
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. The notes say the input is 'one time-varying input channel equal to the aligned neural time axis'.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code just computes the centers of the common bin edges and repeats that 1D time vector for each trial as a `(1, T)` array.

ii. ```python
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The agent's notes describe this input as the aligned neural time axis rather than a separately processed behavioral variable.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is the same `TIME_AXIS` that defines the neural histogram edges. Neural spikes are counted into `TIME_EDGES`, and the input channel stores the corresponding bin centers, so input and neural data share the same aligned grid.

ii. ```python
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
...
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The agent consistently described the decoder input as the neural time axis itself.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The lick-direction label is derived from `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`.

ii. ```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. The notes justify this as 'actual lick direction, not instructed side'.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After trial filtering removes `no` trials, the code assigns a binary lick-direction label: right if `(R & hit) | (L & miss)`, otherwise left. It then tiles that scalar across time for each retained trial.

ii. ```python
lick_dir = actual_lick_direction(behavior)[use_trials]
...
repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. The notes say the label should reflect actual lick direction rather than instructed side, and the trial filter was chosen so only hit/miss response trials remain.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. The notes map autowater trials to WC and non-autowater trials to DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code converts `autowater` into a binary context code by negating it: `autowater=True` becomes WC (`0`), and `autowater=False` becomes DR (`1`). The result is repeated across time within each trial.

ii. ```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
...
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. `CONVERSION_NOTES.md` explicitly lists `0 = WC` and `1 = DR`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. The stored outcome label is derived directly from `bp.hit` after the earlier trial filter has already restricted trials to hit-or-miss response trials. `miss` and `no` are loaded but not used to form the final stored outcome code.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. The notes say the decoder dataset excludes `no` trials and retains correct and incorrect response trials, so a binary hit-based coding was treated as sufficient.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is stored as a binary per-trial label with `0` for incorrect and `1` for correct, implemented as `behavior['hit'].astype(np.int64)` on the filtered hit/miss trial set. It is tiled across time.

ii. ```python
outcome = behavior["hit"][use_trials].astype(np.int64)
...
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size)
```

iii. `CONVERSION_NOTES.md` lists only two outcome values, `incorrect` and `correct`, consistent with the agent's decision to drop no-response trials entirely.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The code derives tongue velocity from the side-camera trajectory feature named `tongue` in `obj.traj`, using that view's `featNames`, `frameTimes`, and `ts`. Alignment additionally uses `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, and `bp.ev.goCue`.

ii. ```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)

def load_traj_feature_series(..., feature_name: str, view_index: int):
    view_group = f[traj_refs[view_index - 1]]
    feature_refs = np.array(view_group["featNames"]).reshape(-1, order="F")
    frame_refs = np.array(view_group["frameTimes"]).reshape(-1, order="F")
    ts_refs = np.array(view_group["ts"]).reshape(-1, order="F")
```

iii. The notes say tongue speed comes from 'the reference x/y tongue velocity components' and the verifier-driven trajectory note says invisible-tongue periods were treated as zero velocity.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each trial, the agent linearly interpolates tongue x and y positions from frame times onto the neural `TIME_AXIS`, computes `np.gradient` on the interpolated positions, replaces non-finite tongue velocities with zero, and then takes speed magnitude `sqrt(xvel**2 + yvel**2)`. It does not combine side and bottom tongue views, does not explicitly smooth tongue coordinates, and does not bin framewise speeds by averaging frames per bin.

ii. ```python
interp = interp1d(
    shifted_t[valid],
    coords[valid, dim],
    kind="linear",
    bounds_error=False,
    fill_value=np.nan,
    assume_sorted=True,
)
vals = interp(time_axis)
...
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
...
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
...
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)
```

iii. The notes claim this follows repository logic for tongue invisibility, and trajectory step 138 says the agent changed the thresholding because invisible-tongue zeros made some sessions nearly constant.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue speed is thresholded at a per-session 50th percentile. If the all-sample median is `<= 0`, the code recomputes the percentile using only positive samples. Bins `>= threshold` become `1`, bins below threshold become `0`.

ii. ```python
def percentile_threshold(values: np.ndarray, drop_zeros_if_needed: bool = False) -> float:
    thresh = float(np.nanpercentile(vals, 50))
    if drop_zeros_if_needed and thresh <= 0:
        pos = vals[vals > 0]
        if pos.size:
            thresh = float(np.nanpercentile(pos, 50))
    return thresh

tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
...
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :]
```

iii. The agent explicitly justified this in trajectory step 138: the median split had collapsed to zero in some sessions because tongue invisibility produced too many zeros, so it changed the rule to positive-only samples when needed.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Video frames are shifted by `video_offset` and the trial's go cue, then interpolated directly onto the neural `TIME_AXIS`. Tongue velocity is therefore represented on the same 10 ms bins as neural data because positions are resampled to that axis before differentiation.

ii. ```python
vidshift = get_video_offset(f, behavior)
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
vals = interp(time_axis)
```

iii. The notes say the video alignment matches `frameTimes - video_offset - goCue`, and that the motion streams are resampled onto the neural time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from a single bottom-camera paw feature in `obj.traj`. The helper `choose_paw_feature` prefers `top_paw` and falls back to `bottom_paw` if needed; it then uses that feature's `featNames`, `frameTimes`, and `ts` from view 2.

ii. ```python
def choose_paw_feature(f: h5py.File) -> str:
    ...
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

paw_feature = choose_paw_feature(f)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The notes present this as using the `top_paw` bottom-view marker, but the code keeps a fallback to `bottom_paw` if `top_paw` is absent.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The paw x/y coordinates are linearly interpolated onto `TIME_AXIS`, differentiated with `np.gradient`, baseline-corrected by subtracting `basederiv[0]` from both x and y velocity channels, nearest-filled where needed, and converted to scalar speed magnitude. The code does not compute speed from frame-time derivatives inside contiguous valid-frame runs.

ii. ```python
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
if "tongue" not in feature_name:
    xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
    yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]
    xvel[:, trial_idx] = fill_nearest(xvel[:, trial_idx])
    yvel[:, trial_idx] = fill_nearest(yvel[:, trial_idx])
...
paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. The notes justify this as reproducing the repository's non-tongue filling and baseline-subtraction behavior.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is thresholded at the session median (`50th percentile`), with bins `>= threshold` coded as `1` and bins below threshold coded as `0`.

ii. ```python
paw_thresh = percentile_threshold(paw_selected)
...
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :]
```

iii. The notes say paw speed is 'binarized with the per-session 50th percentile'.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Like tongue velocity, paw positions are aligned by subtracting `video_offset` and the trial's go cue from frame times, then interpolating onto the common neural `TIME_AXIS` before differentiation.

ii. ```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
...
vals = interp(time_axis)
```

iii. The notes say all motion streams are aligned with `frameTimes - video_offset - goCue` and resampled onto the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the separate `motionEnergy_<animal>_<date>.mat` file, specifically `me.data` (with one extra unwrapping step when that field itself has a `.data` attribute). Alignment then uses side-camera `frameTimes` plus `video_offset` and `goCue`.

ii. ```python
def load_motion_energy(animal: str, date: str, behavior: dict) -> tuple[np.ndarray, float]:
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    if not isinstance(raw, np.ndarray) and hasattr(raw, "data"):
        raw = raw.data
    return raw, float(dat.moveThresh)
```

iii. The notes explicitly say motion energy comes from `motionEnergy_Animal_Date.mat`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the motion-energy trace is linearly interpolated from frame times onto `TIME_AXIS`, nearest-filled at edges or gaps, and then used directly for thresholding. The code does not average raw per-frame values within each neural bin.

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

iii. The notes justify this as resampling motion energy onto the neural time axis and nearest-filling the edges.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The aligned motion-energy samples are thresholded at the per-session median, with `>= threshold` mapped to `1` and below-threshold mapped to `0`.

ii. ```python
me_thresh = percentile_threshold(me_selected)
...
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :]
```

iii. The notes say motion energy is 'binarized with the per-session 50th percentile'.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using side-camera frame times shifted by `video_offset` and go cue, then interpolated onto the same neural `TIME_AXIS`. If a trial's `frameTimes` cannot be read, the code synthesizes them from the trajectory length assuming a 400 Hz frame rate.

ii. ```python
try:
    frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), dtype=np.float64)
except Exception:
    ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), dtype=np.float64)
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. The notes say motion energy is aligned with `frameTimes - video_offset - goCue` and resampled onto the neural time axis.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code usually fills or fabricates values rather than preserving explicit missingness. Missing non-tongue trajectories are nearest-filled; an all-missing trajectory becomes zeros via `fill_nearest`; missing tongue velocity samples become zero; motion-energy interpolation is nearest-filled; if `sglx`/bitcode is absent the video offset defaults to `0.5`; and if motion-energy `frameTimes` fail to load, synthetic times at 400 Hz are created. Trials with NaN `NdroppedFrames` are skipped for that feature.

ii. ```python
if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
    return 0.5
...
if not mask.any():
    return np.zeros_like(x)
...
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
...
aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
...
frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
```

iii. The notes justify these choices as following repository logic for invisible tongue periods and nearest-filling non-tongue features, and trajectory step 138 explicitly frames the tongue zeros as an intended behavior.

## 11-a. What are the most time-consuming steps of the code?

i. From the code structure, the most expensive steps are per-session trajectory interpolation and per-cluster/per-trial spike binning: `bin_spikes_for_session` loops over every kept cluster and then unique trial, and the trajectory loaders loop over every trial and dimension for each feature. Decoder-assembly loops are smaller by comparison.

ii. ```python
for probe in probes:
    for clu in probe:
        ...
        for t in unique_trials:
            counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
```

```python
for trial_idx in range(ntrials):
    ...
    for dim in range(2):
        interp = interp1d(...)
```

iii. The agent did not write a separate performance discussion in the notes, but its trajectory shows the full 12-session build taking long enough to require repeated polling while it ran end-to-end.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several obvious loops could have been vectorized: the inner `for t in unique_trials` spike-histogram loop, the per-trial interpolation loops in `load_traj_feature_series`, the per-trial gradient loop in `compute_velocity_from_position`, and the final Python loop that appends one trial at a time into `neural_trials`, `input_trials`, and `output_trials`.

ii. ```python
for t in unique_trials:
    mask = trial_idx == t
    counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
    trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

```python
for trial_idx in range(ntrials):
    ...
for trial_idx in range(xpos.shape[1]):
    ...
for local_idx, trial_idx in enumerate(use_trials):
    ...
```

iii. The trajectory shows the agent prioritized getting a working end-to-end Figure 8 pipeline rather than revisiting these loops for efficiency.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations within a session: `get_video_offset` is recomputed separately for motion energy, tongue positions, and paw positions; the same trial-by-trial interpolation scaffold is run once for tongue and again for paw; and output label tiling is repeated separately for each trial. It also computes condition PSTHs for every cluster solely to decide the 1 Hz cutoff.

ii. ```python
vidshift = get_video_offset(f, behavior)
```

```python
motion_energy = align_motion_energy(f, behavior, raw_motion_energy)
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. The notes treat these as part of matching the repository logic, but the implementation does not cache the session-level video offset or share interpolation results across streams.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script computes and stores several quantities that are not used by the decoder inputs/outputs: `kept_quality`, `single_flags`, `manual_motion_thresh`, detailed `session_stats`, and event-time summary metadata. It also builds seven-condition PSTHs only for unit filtering, and it loads behavioral event fields like `sample`, `delay`, `reward`, and `bitStart` mainly for metadata/alignment support rather than decoder features.

ii. ```python
neural_all, kept_quality, single_flags = bin_spikes_for_session(behavior, probes)
raw_motion_energy, manual_motion_thresh = load_motion_energy(spec["animal"], spec["date"], behavior)
...
session_stats = {
    ...
    "manual_motion_energy_move_thresh": manual_motion_thresh,
    "quality_labels_kept": kept_quality,
    "event_times_sec_relative_to_go_cue": {...},
}
```

iii. The notes emphasize sanity checks and paper-statistic matching, so the extra bookkeeping was added to compare counts against the paper and repository rather than because the decoder itself needed those fields.
