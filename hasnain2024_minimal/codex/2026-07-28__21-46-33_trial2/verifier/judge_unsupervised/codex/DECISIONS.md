# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a 12-session roster in `SESSION_SPECS`, then iterates over it in `convert_dataset()`. For each session it loads the main MATLAB object with `mat73.loadmat(...)["obj"]`, loads the paired motion-energy `.mat` file with `scipy.io.loadmat`, and then constructs trial-wise neural, input, and output arrays for every kept trial.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    ...
    me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. In `CONVERSION_NOTES.md`, the agent says the roster was taken from `code/Scripts/Figure 8/Figure8a_thru_c.m` and the companion loader files, and that this cohort was chosen because it is the paper's two-context ALM electrophysiology set.

## 1-b. How are the data split into subjects?

i. Subjects are defined by `spec.animal`. The script builds `subjects` dynamically in first-seen order and records one `subject_idx` per session.

ii.
```python
subject_to_idx: dict[str, int] = {}
subjects: list[str] = []

subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)

full_data["subject_idx"].append(subject_idx)
```

iii. The notes explicitly acknowledge a paper/code mismatch here: the paper text says 6 mice, but the released Figure 8 roster yields 7 animal IDs, and the agent chose to follow the released code roster.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` is treated as one session. The output lists `neural`, `input`, `output`, and `brain_region_idx` each receive one entry per `SessionSpec`.

ii.
```python
for spec in SESSION_SPECS:
    ...
    neural_trials = []
    input_trials = []
    output_trials = []
    ...
    full_data["neural"].append(neural_trials)
    full_data["input"].append(input_trials)
    full_data["output"].append(output_trials)
```

iii. The notes say the roster follows the Figure 8 context-analysis loader exactly, so the session split is by those 12 explicit session/date/probe combinations.

## 1-d. How are the data split into trials?

i. Within each loaded session, trials are indexed by the raw `bp["Ntrials"]` count. After filtering, the kept raw trial indices are enumerated and each kept trial becomes one `(n_neurons, n_timepoints)` neural matrix, one input row vector, and one `(6, n_timepoints)` output matrix.

ii.
```python
ntrials = int(bp["Ntrials"])
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)

for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
    input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The agent's notes describe the exported dataset as session lists containing trial lists, and the script implements exactly that structure.

## 1-e. How are trials filtered based on quality controls?

i. Exported trials are restricted to `(hit | miss) & ~early & ~no & ~stim`. That excludes early-lick trials, ignore/no-response trials, and `stim.enable` trials. A session is rejected if fewer than 2 valid trials remain.

ii.
```python
def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim

if kept_trial_indices.size < 2:
    raise ValueError(f"{spec.stem}: fewer than 2 valid trials remained after filtering")
```

iii. `CONVERSION_NOTES.md` says the agent followed the paper text by excluding early licks and no-response trials and also excluded stimulation trials, while keeping hit and miss trials from both contexts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data come from one selected probe per session, using `obj["clu"][spec.probe - 1]`, specifically the per-unit `trial` and `trialtm` arrays, aligned to `bp["ev"]["goCue"]`.

ii.
```python
probe = obj["clu"][spec.probe - 1]
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()

spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. The notes say the neural path was meant to match the shared MATLAB pipeline: selected Figure 8 probe, aligned to `goCue`, then binned and smoothed.

## 2-b. How is the `neural` data processed?

i. For each quality-passing unit, the script subtracts the per-spike trial's `goCue`, bins spikes into 10 ms bins from `-3.0` to `2.5` s, converts counts to Hz by dividing by `DT`, and applies a causal half-Gaussian smoother (`SMOOTH=15`, `reflect` padding) that was ported from `mySmooth.m`.

ii.
```python
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. `CONVERSION_NOTES.md` says neural alignment and smoothing match the MATLAB pipeline, and the trajectory shows the agent explicitly reading `alignSpikes.m`, `getSeq.m`, and `mySmooth.m`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The script first drops units whose `quality` label is `garbage`, `gabrga`, `noisy`, or `real?`. It then computes a low-firing-rate mask from PSTHs averaged over seven task-condition masks and keeps only units with mean firing rate `> 1 Hz`. Finally, sessions with fewer than 10 units after filtering are rejected.

ii.
```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad

mean_frs = psth.mean(axis=0).mean(axis=1)
keep = mean_frs > LOW_FR

if trialdat.shape[1] < 10:
    raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")
```

iii. The notes say unit filtering was intended to match `params.quality = {'all'}` plus `removeLowFRClusters.m`, and the trajectory shows the agent reading `findClusters.m` and `removeLowFRClusters.m` before implementing this.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to `goCue`. For each spike, the script subtracts the `goCue` time of that spike's trial, and the exported time axis spans `-3.0` to `2.5` s relative to that event.

ii.
```python
ALIGN_EVENT = "goCue"
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. Both the task instructions and the notes state go-cue alignment. The notes also say the same session field is used for WC trials because that is how the released codebase treats the common response cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported neural data use 10 ms bins (`DT = 1/100`). Raw spikes are rebinned into this common grid; no additional coarser rebinning is applied afterward.

ii.
```python
DT = 1 / 100
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. The notes say `dt = 0.01` s and that this was meant to reproduce the Figure 8 parameters.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not copied from a single raw field. It is constructed from the global aligned time grid defined by `TMIN`, `TMAX`, and `DT`, with its meaning anchored to `bp["ev"]["goCue"]`.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
EDGES, TIME = build_time_axis()
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes describe the input as `time_from_go_cue_s`, shape `(1, 550)`, which reflects a constructed aligned time basis rather than a raw stored signal.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script builds a single session-independent vector of bin centers from `-3.0` to `2.5` s at 10 ms spacing, then reuses that same vector for every kept trial.

ii.
```python
def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. The agent's notes justify this as the decoder input required by the prompt: a time-varying representation of time from go cue.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The same `TIME` vector used for decoder input is also the time axis used when binning spikes and resampling kinematics, so all streams share the same 550-bin go-cue-centered grid.

ii.
```python
taxis = TIME + ADVANCE_MOVEMENT
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
...
neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes repeatedly state that motion, neural, and decoder input/output variables were all put onto the same go-cue-aligned time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. It is derived from `bp["R"]`; the code uses that boolean/int flag directly for each kept trial.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
```

iii. The notes say `lick_direction` is taken from `bp.R`, with `left=0` and `right=1`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The script casts `bp["R"]` to integer, subsets to kept trials, and then broadcasts the per-trial scalar across all 550 time bins in the output matrix.

ii.
```python
np.full((1, NT), lick_direction[out_pos], dtype=np.int16)
```

iii. The justification in the notes is decoder-driven: lick direction is a per-trial categorical label, so it is stored as a time-constant row.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp["autowater"]`, which the original tutorial code describes as a proxy for distinguishing WC and DR blocks.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The trajectory shows the agent reading `WorkingWithDataObjs.m`, which explicitly says `obj.bp.autowater` can be used as a proxy for water-cued versus delayed-response blocks.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The agent negates `autowater`, so `autowater=True` becomes WC=`0` and `autowater=False` becomes DR=`1`, then broadcasts that per-trial label across time.

ii.
```python
"output_values": [
    ["left", "right"],
    ["WC", "DR"],
    ...
]

context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
np.full((1, NT), context[out_pos], dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` states that behavioral context is derived from `bp.autowater` with `WC=0` and `DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]`.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. The notes explicitly say `outcome` uses `bp.hit`, with `incorrect=0` and `correct=1`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code casts `bp["hit"]` to integer, subsets to kept trials, and repeats the per-trial outcome across all time bins.

ii.
```python
np.full((1, NT), outcome[out_pos], dtype=np.int16)
```

iii. The justification is decoder-format driven: outcome is a categorical trial label, so it is stored as a constant time series.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. It is derived from DeepLabCut trajectories in `obj["traj"]`, using seven tongue landmarks across the side and bottom cameras: `tongue`, `left_tongue`, `right_tongue`, `top_tongue`, `topleft_tongue`, `bottom_tongue`, and `bottomleft_tongue`.

ii.
```python
TONGUE_FEATURES = [
    (0, "tongue"),
    (0, "left_tongue"),
    (0, "right_tongue"),
    (1, "top_tongue"),
    (1, "topleft_tongue"),
    (1, "bottom_tongue"),
    (1, "bottomleft_tongue"),
]
```

iii. The notes say tongue velocity was built from tracked tongue landmarks from both cameras after matching the paper's interpolation and velocity logic.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue landmark, the script interpolates x/y position onto the neural time grid, differentiates with `np.gradient`, converts NaNs in tongue velocity to zero when the tongue is not visible, computes per-landmark speed `sqrt(xvel^2 + yvel^2)`, and then averages those speeds across the seven tongue landmarks.

ii.
```python
xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
xvel, yvel = find_velocity(xpos, ypos, feat_name)
speeds.append(np.sqrt(xvel**2 + yvel**2))
...
stacked = np.stack(speeds, axis=0)
out = np.nanmean(stacked, axis=0)
```

iii. The notes justify this as a decoder-specific scalar built after preserving the paper code's position/velocity preprocessing and its special tongue missing-data handling.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code computes a per-session threshold from strictly positive tongue-speed values only, not from all values. It then labels all kept time points as `1` if `>=` that threshold and `0` otherwise.

ii.
```python
tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed,
    keep_trials,
    ignore_zeros_for_threshold=True,
)

if ignore_zeros_for_threshold:
    threshold_source = kept[kept > 0]
threshold = float(np.nanpercentile(threshold_source, 50))
binary = (kept >= threshold).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent ignored zeros when estimating the tongue threshold because the MATLAB pipeline uses zero as a placeholder when the tongue is not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue trajectories are interpolated onto the same `TIME`/`taxis` grid used for neural data, using the same per-trial alignment times (`goCue`) and the same video-offset correction as motion energy.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

iii. The notes say kinematic alignment matches `findPosition.m` and `findVelocity.m`, and the trajectory shows the agent reading those exact files.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. It is derived from bottom-camera DeepLabCut paw landmarks in `obj["traj"]`: `top_paw` and `bottom_paw`.

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
```

iii. The notes describe paw velocity as a scalar built from the bottom-camera paw landmarks after reusing the paper's interpolation and velocity routines.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The script interpolates each paw landmark onto the neural time axis, computes x/y velocity with baseline subtraction and nearest-fill for missing values, converts to speed, and averages the two paw speeds into one time-varying scalar.

ii.
```python
if not is_tongue:
    xvel[:, trix] = xvel[:, trix] - basederiv[0]
    yvel[:, trix] = yvel[:, trix] - basederiv[0]
    xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
    yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
...
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The notes justify this as matching the MATLAB feature-velocity path and then collapsing it to a decoder-required scalar.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The code uses the 50th percentile of all kept paw-speed values within each session and labels time points below that threshold as `0` and at-or-above it as `1`.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
...
threshold = float(np.nanpercentile(threshold_source, 50))
binary = (kept >= threshold).astype(np.int64)
```

iii. The notes say paw velocity is thresholded at the session median over exported trials and time points.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw position and velocity are resampled onto the same go-cue-aligned `taxis` used for spikes and the input time axis.

ii.
```python
taxis = TIME + ADVANCE_MOVEMENT
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The notes group paw velocity with the shared kinematic alignment path used for all DLC features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. It is derived from the companion `motionEnergy_*.mat` file for each session, specifically `me.data`.

ii.
```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
```

iii. The notes say motion energy loading was intended to match `loadMotionEnergy.m`.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The code loads the per-trial motion-energy traces, corrects video timing with `find_video_offset`, interpolates each trial onto the neural time axis, and fills missing values with nearest neighbors.

ii.
```python
vidshift = find_video_offset(obj)
...
old_t = frame_times - vidshift - align_times[trix]
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. `CONVERSION_NOTES.md` says the agent matched `loadMotionEnergy.m`: same companion file, same video-offset logic, same interpolation, same nearest-fill behavior.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The script computes the 50th percentile of the resampled motion-energy values across all kept trials and time points in a session, then binarizes at that threshold.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. The notes say motion energy was discretized at the session median because the decoder task required binary outputs even though the paper's original use of motion energy was continuous plus a separate manual move threshold.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. It is aligned by resampling each trial's motion-energy trace onto the same go-cue-centered `taxis` that the neural data use.

ii.
```python
align_times = go_cue
taxis = TIME + ADVANCE_MOVEMENT
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. The notes explicitly say motion energy was interpolated onto the neural time axis after video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The script has several fallback paths. Missing DLC frame times trigger a synthetic `np.arange(...)/400 - 0.5` time base. Trials with invalid dropped-frame metadata are skipped in `find_position`. Non-tongue missing positions and velocities are nearest-filled, tongue missing velocities are set to zero, fully missing arrays can be filled with zeros by `fill_nearest_1d`, and motion-energy NaNs are nearest-filled. Non-finite spikes and out-of-range trial indices are discarded.

ii.
```python
use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]

if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
else:
    xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)

finite = (spike_trials >= 0) & (spike_trials < ntrials) & np.isfinite(spike_times)
```

iii. The notes justify these choices as attempts to preserve the MATLAB behavior, especially the special-case handling for tongue visibility and nearest filling elsewhere.

## 11-a. What are the most time-consuming steps of the code?

i. The heaviest work is the per-session, per-unit neural binning/smoothing loop in `build_aligned_trialdat`, followed by the repeated per-feature, per-trial kinematic interpolation/velocity computation in `compute_feature_speed`, `find_position`, and `find_velocity`, plus the per-trial motion-energy interpolation in `load_motion_energy`.

ii.
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    np.add.at(counts, (spike_trials, bins), 1.0)
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)

for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(...)
    xvel, yvel = find_velocity(...)
```

iii. This is an inference from the code structure: these are the deepest nested loops over large arrays, and the trajectory also shows the full conversion run taking noticeable time while iterating through all sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The unit loop in `build_aligned_trialdat`, the trial loop in `find_position`, the trial loop in `find_velocity`, the motion-energy trial loop in `load_motion_energy`, and the repeated loop over output trials when stacking constant rows are all candidates for vectorization or batched operations.

ii.
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...

for trix in range(ntrials):
    ...

for trix in range(xpos.shape[1]):
    ...

for trix, trial_me in enumerate(raw_trials):
    ...
```

iii. The agent did not justify these as design choices; this is simply what the final code does.

## 11-c. What processing does the code repeat multiple times?

i. It repeats nearest-fill and interpolation logic for each feature and again for motion energy; it rebuilds per-condition PSTHs from already smoothed single-trial neural data for low-FR filtering; and it repeatedly casts the same `TIME` vector to `float32` inside the trial loop.

ii.
```python
xpos[:, trix] = interp_with_nan(...)
ypos[:, trix] = interp_with_nan(...)
...
resampled[:, trix] = interp_with_nan(...)

psth[:, :, cond_idx] = trialdat[:, :, trials].mean(axis=2)

input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. This is visible directly in the code. The notes do not discuss efficiency, but they do confirm the repeated feature-processing path was intentional to mirror the MATLAB helpers.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The script builds `sample_data` even though the main decoder uses `converted_data.pkl`; it computes and stores session-summary statistics and metadata fields that are not used by decoder training; it loads `me["moveThresh"]` but never uses it for the exported output; and it computes `mean_frs` beyond the boolean keep mask only for reporting.

ii.
```python
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
    "vidshift": vidshift,
}

session_summary = {
    ...
    "mean_fr_hz_min": float(mean_frs[low_fr_keep].min()),
    "mean_fr_hz_max": float(mean_frs[low_fr_keep].max()),
}

sample_data = build_sample_dataset(full_data)
```

iii. The notes justify some of this as validation and documentation work rather than core conversion logic, especially the extra sample artifact and the extensive session summaries.
