# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI defines a hardcoded list of 12 session specifications (`SESSION_SPECS`) containing animal name, date, and probe numbers. For each session, it opens the corresponding HDF5 file (`data_structure_{animal}_{date}.mat`) from the `Ephys_Behavior` directory using `h5py`, and separately loads the motion energy `.mat` file using `scipy.io.loadmat`. This mirrors the Figure 8 script's approach of using per-animal loading scripts to define the session list and then calling `loadSessionData`/`processData`.

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
        ...
```

iii. From CONVERSION_NOTES.md: "This conversion targets the alternating delayed-response / water-cued ALM electrophysiology cohort used for the context analyses in the reference repository. The session list follows the loader combination used in `code/Scripts/Figure 8/Figure8a_thru_c.m` and `code/Scripts/Figure 8/Figure8d.m`."

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field in `SESSION_SPECS`. An `OrderedDict` (`subject_lookup`) maps unique animal names to indices. Each session's animal is looked up or added to the subject list. The AI identifies 7 unique subjects: JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19.

ii.
```python
def build_dataset() -> tuple[dict, dict]:
    subjects = []
    subject_lookup = OrderedDict()
    ...
    for spec in SESSION_SPECS:
        sid = spec["animal"]
        if sid not in subject_lookup:
            subject_lookup[sid] = len(subject_lookup)
            subjects.append(sid)
        data["subject_idx"].append(subject_lookup[sid])
```

iii. From CONVERSION_NOTES.md: "The loader subset used by the Figure 8 scripts yields 12 sessions across 7 animal IDs, whereas the copied methods text says 12 sessions from 6 mice." The AI chose to follow the code rather than the paper text.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` defines one session (unique animal + date + probe combination). The AI iterates over all 12 specs, calling `build_session_payload` for each. Results are appended to lists in the output data dictionary (one entry per session for `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx`).

ii.
```python
for spec in SESSION_SPECS:
    payload = build_session_payload(spec)
    data["neural"].append(payload["neural_trials"])
    data["input"].append(payload["input_trials"])
    data["output"].append(payload["output_trials"])
```

iii. The AI follows the Figure 8 script structure which loads sessions one by one from per-animal loading scripts. Each file `data_structure_{animal}_{date}.mat` corresponds to one session.

## 1-d. How are the data split into trials?

i. The number of trials per session is read from `obj.bp.Ntrials` in the HDF5 file. Spike times and video data are organized per trial via the `trial` and `trialtm` fields in each cluster, and per-trial `frameTimes` / trajectory data. After filtering (see 1-e), the kept trial indices are used to extract per-trial neural, input, and output data.

ii.
```python
def load_behavior(f: h5py.File) -> dict:
    bp = f["obj"]["bp"]
    out = {
        "Ntrials": int(np.asarray(read_numeric_dataset(bp["Ntrials"])).item()),
        ...
    }

# In build_session_payload:
use_trials = np.flatnonzero(analysis_trial_mask(behavior))
for local_idx, trial_idx in enumerate(use_trials):
    neural_trials.append(neural_selected[:, :, local_idx].astype(np.float32))
```

iii. Trials are naturally defined by the behavioral protocol structure stored in the data files, consistent with the MATLAB reference where `obj.bp.Ntrials` defines total trial count.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered using `analysis_trial_mask`, which keeps only trials that are either `hit` or `miss`, and excludes `early` (early lick), `no` (no response/ignore), and `stim_enable` (stimulation) trials.

ii.
```python
def analysis_trial_mask(behavior: dict) -> np.ndarray:
    return (behavior["hit"] | behavior["miss"]) & (~behavior["early"]) & (~behavior["no"]) & (~behavior["stim_enable"])
```

iii. From CONVERSION_NOTES.md: "The decoder dataset excludes: early trials, no response trials, stim.enable trials. This matches the recurring ~early, ~no, and ~stim.enable filtering used in the repository analyses."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times stored in `obj.clu{probe}` for each cluster (neuron). Specifically, the `trial` field (which trial each spike belongs to) and `trialtm` field (spike time within each trial) are used. Cluster quality labels from `obj.clu{probe}.quality` are used for filtering.

ii.
```python
def load_probe_clusters(f: h5py.File, probe_number: int):
    clu_cells = np.array(f["obj"]["clu"]).reshape(-1, order="F")
    probe_group = f[clu_cells[probe_number - 1]]
    qualities = deref_string_list(f, probe_group["quality"])
    trials = deref_numeric_list(f, probe_group["trial"])
    trialtm = deref_numeric_list(f, probe_group["trialtm"])
```

iii. This directly follows the MATLAB reference where spikes are stored per-cluster in `obj.clu{prbnum}(clu).trial` and `obj.clu{prbnum}(clu).trialtm`.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting the goCue event time. Aligned spikes are binned into 10ms time bins spanning [-3.0s, 2.5s]. Binned spike counts are converted to firing rates by dividing by the bin width (DT = 0.01s). Firing rates are then smoothed with a causal Gaussian kernel of window size 15 bins, using 'reflect' boundary conditions.

ii.
```python
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
trial_counts = np.zeros((TIME_AXIS.size, ntrials), dtype=np.float64)
...
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
trial_counts[:, t] = my_smooth(counts / DT, SMOOTH, BCTYPE)
```

iii. From CONVERSION_NOTES.md: "Alignment event: goCue, Time window: [-3.0 s, 2.5 s], Bin size: 10 ms, Smoothing: causal Gaussian kernel with N=15, matching mySmooth(..., 15, 'reflect')."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering:
1. **Cluster quality filter**: Clusters with quality labels 'garbage', 'gabrga', 'noisy', or 'real?' are excluded. All other quality labels are kept (matching the 'all' quality mode in the reference code's `findClusters`).
2. **Low firing rate filter**: Condition-averaged PSTHs are computed using 7 conditions (matching Figure 8). Units whose mean firing rate across all conditions and time points is not above 1 Hz are removed.

ii.
```python
def cluster_quality_is_kept(quality: str) -> bool:
    quality = str(quality).strip()
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

iii. From CONVERSION_NOTES.md: "Cluster inclusion before firing-rate filter: all qualities except garbage, gabrga, noisy, real?. Low-firing-rate exclusion: remove units with mean PSTH firing rate <= 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `behavior["ev"]["goCue"][trial_idx]` from each spike's time. This creates spike times relative to go cue onset, which are then binned into the time axis centered on t=0 (go cue).

ii.
```python
ALIGN_EVENT = "goCue"
aligned = clu["trialtm"][valid_trial_spikes] - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. From CONVERSION_NOTES.md: "Alignment event: goCue." The instructions explicitly state: "Temporally align based on Go cue onset."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10 ms (DT = 1/100 s). This matches the Figure 8 script's `params.dt = 1/100`. No temporal rebinning is applied — spikes are binned directly at this resolution. The time axis spans from -3.0s to 2.5s relative to go cue, producing 550 time bins.

ii.
```python
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. From CONVERSION_NOTES.md: "Bin size: 10 ms." The metadata reports `time_bin_size: 10.0` (ms).

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from any raw data variable. It is the time axis itself (TIME_AXIS), which represents the center of each time bin relative to go cue onset.

ii.
```python
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. The instructions specify "Time from go cue onset in seconds (continuous, time-varying)" as a decoder input. Since the time axis is defined by the binning parameters and alignment event, it is the same for every trial.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers from the time edges array. `TIME_EDGES` is created using `np.arange(TMIN, TMAX + DT, DT)`, and `TIME_AXIS` is the midpoint of consecutive edges: `(TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2`. This gives values ranging from approximately -2.995 to 2.495 in 0.01s steps.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
```

iii. This matches the MATLAB reference: `edges = params.tmin:params.dt:params.tmax; obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1);`

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data time axis because both use `TIME_AXIS`. Since the neural data is binned using `TIME_EDGES` (with bin centers at `TIME_AXIS`), and the input is directly `TIME_AXIS`, they are perfectly aligned by construction.

ii.
```python
# Neural data binned with:
counts, _ = np.histogram(aligned[mask], bins=TIME_EDGES)
# Input is:
input_trials.append(TIME_AXIS[None, :].astype(np.float32))
```

iii. No explicit alignment step is needed since both share the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from two behavioral variables: `R` (right trial indicator) and `L` (left trial indicator), combined with `hit` (correct response) and `miss` (incorrect response). These are boolean arrays from `obj.bp`.

ii.
```python
def actual_lick_direction(behavior: dict) -> np.ndarray:
    return ((behavior["R"] & behavior["hit"]) | (behavior["L"] & behavior["miss"])).astype(np.int64)
```

iii. From CONVERSION_NOTES.md: "Defined as actual lick direction, not instructed side: right if R & hit or L & miss; left if L & hit or R & miss."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The actual lick direction is computed by combining instructed side (R/L) with outcome (hit/miss). A right lick (=1) occurs when the animal was instructed right and responded correctly (R & hit) or was instructed left and responded incorrectly (L & miss). The resulting per-trial label is repeated across all time bins for each trial.

ii.
```python
lick_dir = actual_lick_direction(behavior)[use_trials]
# Then in the output construction:
repeat_labels(np.asarray([lick_dir[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. The AI correctly infers the actual lick direction from the available behavioral flags, since the raw data stores instructed direction (R/L) and outcome (hit/miss) rather than the actual lick direction directly.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `autowater` boolean array in `obj.bp`. When `autowater` is True, the trial is a water-cued (WC) trial; when False, it is a delayed-response (DR) trial.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
```

iii. From CONVERSION_NOTES.md: "behavioral_context: 0 = WC, 1 = DR." This uses the fact that autowater=True indicates WC context, matching the reference code's condition definitions where `~autowater` = DR trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` boolean is inverted (negated) so that DR maps to 1 and WC maps to 0, matching the instruction specification. The per-trial label is then repeated across all time bins.

ii.
```python
context = (~behavior["autowater"][use_trials]).astype(np.int64)
# In output:
repeat_labels(np.asarray([context[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. Instructions specify: "Behavioral context (WC = 0, DR = 1, per-trial)."

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from the `hit` boolean array in `obj.bp`. A `hit` (True) indicates a correct response; otherwise (miss), it is incorrect.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
```

iii. Since the trial filter already ensures only `hit` or `miss` trials are kept, `hit=True` maps to correct (1) and `hit=False` (which must be `miss`) maps to incorrect (0).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The `hit` boolean is cast to integer (1=correct, 0=incorrect). The per-trial label is repeated across all time bins.

ii.
```python
outcome = behavior["hit"][use_trials].astype(np.int64)
repeat_labels(np.asarray([outcome[local_idx]], dtype=np.int64), TIME_AXIS.size),
```

iii. Instructions specify: "Outcome (incorrect = 0, correct = 1, per-trial)."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the tongue tracking data stored in `obj.traj{1}` (side camera, view 1). The x and y positions of the 'tongue' feature are extracted from the `ts` array (tracking coordinates) and `frameTimes` (video frame timestamps). The video offset (computed from `obj.sglx.bitcode.bitstart`, `obj.sglx.fs`, and `obj.bp.ev.bitStart`) is used for temporal alignment.

ii.
```python
tongue_xpos, tongue_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, "tongue", 1)
# In load_traj_feature_series:
coords = ts[:, 0:2, feat_idx]  # x,y positions from DLC tracking
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
```

iii. From CONVERSION_NOTES.md: "scalar tongue speed from the reference x/y tongue velocity components."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps:
1. Extract x,y positions from DLC tracking for the 'tongue' feature (side camera view 1)
2. Align to go cue: `frameTimes - video_offset - goCue`
3. Interpolate positions to the neural time axis using linear interpolation
4. Do NOT fill missing (NaN) tongue positions (tongue invisible periods remain NaN)
5. Compute velocity as `np.gradient` of positions
6. Set NaN velocities to 0 (tongue invisible = zero velocity)
7. Compute speed magnitude: `sqrt(xvel^2 + yvel^2)`

ii.
```python
tongue_xvel, tongue_yvel = compute_velocity_from_position(tongue_xpos, tongue_ypos, "tongue")
tongue_speed = np.sqrt(tongue_xvel**2 + tongue_yvel**2)

# In compute_velocity_from_position for tongue:
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
# tongue: no baseline subtraction, NaN -> 0
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0
yvel[:, trial_idx][~np.isfinite(yvel[:, trial_idx])] = 0.0
```

iii. From CONVERSION_NOTES.md: "tongue invisibility periods become zero velocity" and "velocity_definition: Speed magnitude sqrt(xvel^2 + yvel^2)..."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue speed is discretized per-session using the 50th percentile (median) of all tongue speed values across kept trials and time points. Values >= threshold map to 1, < threshold map to 0. If the median is <= 0 (due to many zero-velocity invisible-tongue periods), the threshold is recomputed from positive values only.

ii.
```python
tongue_thresh = percentile_threshold(tongue_selected, drop_zeros_if_needed=True)
# In percentile_threshold:
thresh = float(np.nanpercentile(vals, 50))
if drop_zeros_if_needed and thresh <= 0:
    pos = vals[vals > 0]
    if pos.size:
        thresh = float(np.nanpercentile(pos, 50))
# In output:
(tongue_selected[:, local_idx] >= tongue_thresh).astype(np.int64)[None, :],
```

iii. From CONVERSION_NOTES.md: "because the repository sets invisible-tongue velocity to zero, if the all-sample median is zero the threshold is recomputed from positive tongue-speed samples only; otherwise several sessions collapse to a trivial constant label."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated onto the same time axis (`TIME_AXIS`) used for the neural data. This is done by computing `shifted_t = frameTimes - video_offset - goCue[trial]` and using `scipy.interpolate.interp1d` to interpolate from video frame times to neural time bins.

ii.
```python
interp = interp1d(shifted_t[valid], coords[valid, dim], kind="linear",
                  bounds_error=False, fill_value=np.nan, assume_sorted=True)
vals = interp(time_axis)
```

iii. This matches the reference code: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the paw tracking data in `obj.traj{2}` (bottom camera, view 2). The AI dynamically selects the paw feature name by checking for 'top_paw' or 'bottom_paw' in the feature list. The x,y positions from DLC tracking and `frameTimes` are used.

ii.
```python
def choose_paw_feature(f: h5py.File) -> str:
    traj_refs = np.array(f["obj"]["traj"]).reshape(-1, order="F")
    view_group = f[traj_refs[1]]  # bottom view
    first_feats = deref_string_list(f, f[...])
    for name in ("top_paw", "bottom_paw"):
        if name in first_feats:
            return name

paw_feature = choose_paw_feature(f)
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
```

iii. From CONVERSION_NOTES.md: "scalar paw speed from the top_paw bottom-view marker."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps:
1. Extract x,y positions from DLC tracking for the paw feature (bottom camera view 2)
2. Smooth positions with `mySmooth(ts, 1, 'reflect')` (effectively no smoothing since N=1)
3. Align to go cue and interpolate to neural time axis
4. Fill missing positions with nearest valid value
5. Compute velocity as `np.gradient` of positions
6. Subtract baseline derivative (`basederiv[0]`) from BOTH x and y velocities (matching the reference MATLAB code's quirk)
7. Fill missing velocities with nearest valid value
8. Compute speed magnitude: `sqrt(xvel^2 + yvel^2)`

ii.
```python
# Position: non-tongue features are nearest-filled
if "tongue" not in feature_name:
    xpos[:, trial_idx] = fill_nearest(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest(ypos[:, trial_idx])

# Velocity: baseline subtraction
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
yvel[:, trial_idx] = np.gradient(tsinterp[:, 1])
if "tongue" not in feature_name:
    xvel[:, trial_idx] = xvel[:, trial_idx] - basederiv[0]
    yvel[:, trial_idx] = yvel[:, trial_idx] - basederiv[0]

paw_speed = np.sqrt(paw_xvel**2 + paw_yvel**2)
```

iii. From CONVERSION_NOTES.md: "non-tongue baseline subtraction reproduces the MATLAB implementation, including subtracting basederiv(1) from both xvel and yvel."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is discretized per-session using the 50th percentile (median) of all paw speed values across kept trials and time points. Values >= threshold map to 1, < threshold map to 0. Unlike tongue, no special handling for zero values.

ii.
```python
paw_thresh = percentile_threshold(paw_selected)
(paw_selected[:, local_idx] >= paw_thresh).astype(np.int64)[None, :],
```

iii. Instructions specify: "Paw velocity: discretized into two bins with per-session threshold: 0: < 50th percentile, 1: >= 50th percentile."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment approach as tongue: paw positions are interpolated from video frame times onto the neural time axis via `interp1d` after applying the video offset and go cue alignment. Non-tongue features are additionally nearest-filled after interpolation.

ii.
```python
paw_xpos, paw_ypos = load_traj_feature_series(f, behavior, TIME_AXIS, paw_feature, 2)
# Same interpolation as tongue in load_traj_feature_series
```

iii. Matches the reference code: `interp1(traj(trix).frameTimes-vidshift-obj.bp.ev.(alignEv)(trix), ts, taxis)` followed by `fillmissing(...,'nearest')`.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from separate `.mat` files (`motionEnergy_{animal}_{date}.mat`). The `me.data` field contains per-trial motion energy time series (at 400 Hz video frame rate). The `me.moveThresh` is also loaded but only stored in metadata, not used for thresholding.

ii.
```python
def load_motion_energy(animal: str, date: str, behavior: dict):
    path = os.path.join(EPHYS_DIR, f"motionEnergy_{animal}_{date}.mat")
    dat = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["me"]
    raw = dat.data
    move_thresh = float(dat.moveThresh)
    return raw, move_thresh
```

iii. From CONVERSION_NOTES.md: "Motion energy comes from motionEnergy_Animal_Date.mat."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing steps:
1. Load raw motion energy per trial from `.mat` file
2. Align to go cue using side camera (view 1) frame times: `frameTimes - video_offset - goCue`
3. Interpolate from video frame rate (400 Hz) to neural time axis (100 Hz) using linear interpolation
4. Fill NaN values with nearest valid value

ii.
```python
def align_motion_energy(f, behavior, raw_motion_energy):
    ...
    for trial_idx in range(ntrials):
        me_trial = np.asarray(raw_motion_energy[trial_idx], dtype=np.float64).reshape(-1)
        shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
        interp = interp1d(shifted_t[valid], me_trial[valid], kind="linear", ...)
        aligned[:, trial_idx] = interp(TIME_AXIS)
        aligned[:, trial_idx] = fill_nearest(aligned[:, trial_idx])
```

iii. From CONVERSION_NOTES.md: "is resampled onto the neural time axis, and is nearest-filled at the edges." Matches reference: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix),me.data{trix},taxis)` followed by `fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized per-session using the 50th percentile (median) of all motion energy values across kept trials and time points. Values >= threshold map to 1, < threshold map to 0. The manual `moveThresh` from the data file is NOT used for thresholding (only stored in metadata).

ii.
```python
me_thresh = percentile_threshold(me_selected)
(me_selected[:, local_idx] >= me_thresh).astype(np.int64)[None, :],
```

iii. Instructions specify 50th percentile threshold. The reference code uses `me.move = me.data > me.moveThresh` (a manual threshold), but the AI correctly followed the decoder task instructions which specify percentile-based thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the same video alignment approach: side camera frame times are shifted by `video_offset + goCue` and then linearly interpolated onto the neural time axis. This is the same time base used for neural spike binning.

ii.
```python
shifted_t = frame_times - vidshift - behavior["ev"][ALIGN_EVENT][trial_idx]
aligned[:, trial_idx] = interp(TIME_AXIS)
```

iii. Matches the reference: `interp1(obj.traj{1}(trix).frameTimes-vidshift-alignTimes(trix),me.data{trix},taxis)`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
- **Missing tongue positions**: Left as NaN; when velocity is computed, NaN velocities are set to 0 (tongue invisible = no movement).
- **Missing non-tongue positions**: Filled with nearest valid value after interpolation.
- **Missing video frames**: Trials with all-NaN `NdroppedFrames` are skipped (left as NaN in position arrays).
- **Invalid spike trial indices**: Spikes with trial indices outside [0, Ntrials) are filtered out.
- **Missing sglx bitcode**: If `obj.sglx.bitcode` doesn't exist, video offset defaults to 0.5s.
- **Missing stim field**: If `bp.stim.enable` doesn't exist, defaults to all-False.
- **NaN event times**: Finite checks are applied to shifted times before interpolation.
- **Insufficient data**: Sessions with fewer than 2 usable trials raise an error.

ii.
```python
# Tongue invisible -> velocity = 0
xvel[:, trial_idx][~np.isfinite(xvel[:, trial_idx])] = 0.0

# Video offset fallback
if "sglx" not in f["obj"] or "bitcode" not in f["obj"]["sglx"]:
    return 0.5

# Spike trial index validation
valid_trial_spikes = (trial_idx >= 0) & (trial_idx < ntrials)

# Missing stim field
if "stim" in bp and "enable" in bp["stim"]:
    out["stim_enable"] = ...
else:
    out["stim_enable"] = np.zeros(out["Ntrials"], dtype=bool)
```

iii. The AI implemented defensive handling for multiple edge cases. The tongue handling matches the MATLAB reference (`isnan -> 0`). The video offset fallback of 0.5 is a guess when sglx data is missing.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are:
1. **Loading and parsing HDF5 files**: Each call to `build_session_payload` opens a large HDF5 file and dereferences many cell array references.
2. **Video trajectory loading and interpolation**: `load_traj_feature_series` loops over all trials, dereferencing HDF5 references and performing interpolation for each trial.
3. **Spike binning and smoothing**: `bin_spikes_for_session` loops over all clusters and all trials, performing histogram and smoothing operations for each.
4. **Motion energy alignment**: `align_motion_energy` loops over all trials with interpolation.

ii.
```python
# HDF5 dereferencing in tight loops:
for trial_idx in range(ntrials):
    ndropped = np.asarray(read_numeric_dataset(f[ndropped_refs[trial_idx]]), ...)
    frame_times = np.asarray(read_numeric_dataset(f[frame_refs[trial_idx]]), ...)
    ts = np.asarray(read_numeric_dataset(f[ts_refs[trial_idx]]), ...)
```

iii. The HDF5 dereferencing is particularly expensive because each reference requires a separate file access operation. The MATLAB code benefits from loading the entire object structure into memory at once.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could potentially be vectorized:
1. **Spike binning loop** (per-trial histogram in `bin_spikes_for_session`): Could use vectorized multi-trial histogram approaches.
2. **Smoothing column loop** in `my_smooth`: `for col in range(x.shape[1])` applies convolution column-by-column; could use scipy's 2D convolution.
3. **Velocity computation loop** in `compute_velocity_from_position`: Per-trial gradient and baseline subtraction could be vectorized.
4. **Position interpolation loop** in `load_traj_feature_series`: Per-trial interpolation is inherently sequential due to different frame times per trial.

ii.
```python
# my_smooth column loop:
for col in range(x.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

# Velocity computation loop:
for trial_idx in range(xpos.shape[1]):
    xvel[:, trial_idx] = np.gradient(tsinterp[:, 0])
```

iii. The interpolation loops are difficult to vectorize since each trial has different frame times. The smoothing and gradient loops are more amenable to vectorization.

## 11-c. What processing does the code repeat multiple times?

i. Repeated processing includes:
1. **Video offset computation**: `get_video_offset` is called separately for tongue position, paw position, and motion energy alignment (3 times per session).
2. **HDF5 trajectory reference dereferencing**: The trajectory reference array `f["obj"]["traj"]` is read in `load_traj_feature_series` (twice: tongue and paw) and `align_motion_energy` (once).
3. **Feature name dereferencing**: In `load_traj_feature_series`, feature names are dereferenced for every trial to find the feature index, even though they're usually constant.
4. **Context conditions**: `get_context_conditions` is called once per session within `bin_spikes_for_session`, but the same conditions could be precomputed.

ii.
```python
# Video offset computed in load_traj_feature_series (twice) and align_motion_energy (once):
vidshift = get_video_offset(f, behavior)  # called 3x per session
```

iii. The repeated video offset computation is relatively cheap (just mode calculations), but the repeated HDF5 trajectory reference reading is more expensive.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Unnecessary processing includes:
1. **Full-trial video/kinematic processing**: Tongue, paw, and motion energy data are loaded and processed for ALL trials, but only the kept trials (after `analysis_trial_mask`) are used.
2. **Smoothing with N=1**: `my_smooth(coords, 1, "reflect")` for non-tongue features returns the input unchanged, performing a no-op.
3. **PSTH computation for low-FR filtering**: The 7-condition PSTHs are computed solely for the firing rate filter and then discarded; only the single-trial data is kept.
4. **Single-unit quality tracking**: `single_unit_flags` and quality labels are computed and stored in metadata but not used by the decoder.
5. **Manual motion energy threshold**: `move_thresh` from the motion energy file is loaded and stored in metadata but not used for thresholding (50th percentile is used instead per instructions).

ii.
```python
# Processing all trials then selecting:
tongue_xpos, tongue_ypos = load_traj_feature_series(...)  # all Ntrials
tongue_selected = tongue_speed[:, use_trials]  # subset

# No-op smoothing:
if "tongue" not in feature_name:
    coords = my_smooth(coords, 1, "reflect")  # N=1, returns input unchanged
```

iii. Processing all trials before filtering is a design choice for code simplicity. The PSTH computation for filtering is necessary by design (needed to determine which neurons to keep).
