# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-coded a 12-session roster from the Figure 8 context-analysis pipeline and loaded only `/app/data/Ephys_Behavior`. For each session it loaded `data_structure_<animal>_<date>.mat` with `mat73.loadmat` and loaded `motionEnergy_<animal>_<date>.mat` separately with `scipy.io.loadmat`.

ii. ```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]
...
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    ...
    me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. In the trajectory, the AI said it had identified “the 12 two-context ALM ephys sessions,” cross-checked them against Figure 8, and planned to “hardcode the context-session roster from the paper’s Figure 8 pipeline.” It also said it was trying to match the paper’s context cohort size.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the `animal` field of each hard-coded `SessionSpec`. During assembly, the script appends new subject IDs in first-seen order and stores one `subject_idx` per session.

ii. ```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int
...
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
...
full_data["subject_idx"].append(subject_idx)
```

iii. The trajectory did not contain a separate explicit justification for subject splitting beyond using the hard-coded session roster; the choice is implicit in the `SessionSpec` design and the goal of matching the Figure 8 cohort.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` entry is treated as one session. The AI did not merge across folders or discover sessions dynamically; every session corresponds to one `data_structure_*.mat` file plus one matching `motionEnergy_*.mat` file in `Ephys_Behavior`.

ii. ```python
@property
def data_path(self) -> Path:
    return DATA_DIR / f"data_structure_{self.stem}.mat"

@property
def motion_energy_path(self) -> Path:
    return DATA_DIR / f"motionEnergy_{self.stem}.mat"
...
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. The trajectory justification was the same as 1-a: the AI explicitly chose the Figure 8 context-session roster and treated those 12 entries as the target dataset.

## 1-d. How are the data split into trials?

i. Trials are indexed by the raw Bpod trial count `bp["Ntrials"]`. Neural trial membership comes from per-spike `probe["trial"]`, and exported trials are the subset of raw trial indices selected by `keep_trials`.

ii. ```python
ntrials = int(bp["Ntrials"])
...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
...
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)
...
for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
```

iii. In the trajectory the AI said the validator confirmed `input` and `output` should be stored as per-trial arrays. Otherwise the trial split is implicit from the raw session structure rather than separately justified.

## 1-e. How are trials filtered based on quality controls?

i. The AI kept only hit or miss trials, and dropped early-lick, no-response, and stimulation trials. It did not apply the reference solution’s extra cutoff for trials beyond the end of recording.

ii. ```python
def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim
```

iii. The trajectory repeatedly called “exact trial exclusions” a critical design choice. The code and metadata show the AI chose these exclusions to fit a binary left/right and correct/incorrect decoder framing.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from a selected probe in `obj["clu"]`, specifically its `trial`, `trialtm`, and `quality` fields, together with `bp["ev"]["goCue"]` for alignment.

ii. ```python
probe = obj["clu"][spec.probe - 1]
...
qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. The trajectory justification was that the AI was porting the MATLAB helpers for spike alignment and cluster filtering from the paper code.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to `goCue`, binned from `TMIN=-3.0` to `TMAX=2.5` at `DT=0.01` s, converted to Hz, and smoothed with a causal half-Gaussian window of length 15 using reflect padding.

ii. ```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
...
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. The AI explicitly said in the trajectory that the loader code showed neural data were “binned at `dt = 0.01 s`, smoothed causally, aligned to `goCue`,” and it implemented that interpretation directly.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI first removed clusters labeled `garbage`, `gabrga`, `noisy`, or `real?`. It then computed mean firing rates from condition-averaged PSTHs and kept only units above `LOW_FR = 1.0` Hz. It also raised an error if fewer than 10 units remained in a session.

ii. ```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad
...
mean_frs = psth.mean(axis=0).mean(axis=1)
keep = mean_frs > LOW_FR
...
if trialdat.shape[1] < 10:
    raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")
```

iii. The trajectory shows the AI inspected the reference MATLAB cluster-filtering code and later described its neural preprocessing as excluding `garbage/noisy/real?` clusters and removing low-firing units. It did not give a separate explanation for omitting `poor` units or for the extra 10-unit minimum.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned by subtracting that trial’s `goCue`, then assigning the aligned spike to a time bin on the common neural grid.

ii. ```python
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
...
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The trajectory explicitly says the AI interpreted the paper code as aligning neural data to `goCue`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The exported data use 10 ms bins (`DT = 1/100`). No later rebinning is applied; neural, time input, and kinematic outputs are all placed directly on that 10 ms grid.

ii. ```python
DT = 1 / 100
...
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
...
rates = counts.T / DT
```

iii. The trajectory justification was the AI’s reading of the paper loader code: it repeatedly stated that the reference used `dt = 0.01 s`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The exported input is a synthetic common time axis, not a directly stored raw signal. It is defined relative to the session’s `goCue` alignment choice and the fixed constants `TMIN`, `TMAX`, and `DT`.

ii. ```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
...
EDGES, TIME = build_time_axis()
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The trajectory justification is implicit: the AI confirmed with the validator that a per-trial time-varying input was acceptable, then used the neural time grid itself as the decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The AI computed the input as the centers of uniform 10 ms bins spanning -3.0 s to +2.5 s around the go cue, and repeated the same vector for every retained trial.

ii. ```python
def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
...
for out_pos, trial_idx in enumerate(kept_trial_indices):
    input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The AI did not give a separate detailed justification beyond treating the decoder input as the common aligned time axis and using the paper-derived `dt = 0.01 s` interpretation.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses exactly the same `TIME` vector that defines neural bin centers, so the input and neural matrices share one time base.

ii. ```python
EDGES, TIME = build_time_axis()
...
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The trajectory justification is implicit in the AI’s repeated statement that kinematics and motion energy should be placed on the neural time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derived lick direction only from `bp["R"]` after applying the trial mask. It did not use `hit` and `miss` to infer actual lick side on incorrect trials, and it did not keep a no-lick class.

ii. ```python
def build_output_constants(bp: dict, keep_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
    outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
    return lick_direction, context, outcome
```

iii. The trajectory does not contain a dedicated lick-direction rationale. The decision is consistent with the AI’s broader choice to export only hit/miss trials and keep binary outputs.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. There is almost no processing: the `R` flag is copied as a binary left/right label and then repeated across all time bins of the kept trial.

ii. ```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), lick_direction[out_pos], dtype=np.int16)
```

iii. No explicit trajectory justification was recorded beyond the AI’s decision to use binary trial-level outputs for this reduced context cohort.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp["autowater"]`.

ii. ```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The trajectory did not give a separate justification here; it is a direct reading of the task context fields after the AI chose the two-context Figure 8 cohort.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI inverted the `autowater` boolean so `WC` becomes 0 and `DR` becomes 1, then repeated that per-trial label across time bins.

ii. ```python
"output_values": [
    ["left", "right"],
    ["WC", "DR"],
    ...
]
...
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
...
np.full((1, NT), context[out_pos], dtype=np.int16)
```

iii. The trajectory justification is implicit: the AI targeted the paper’s two-context analysis and encoded the two contexts in the decoder format requested.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived only from `bp["hit"]` after the script has already removed no-response trials and retained only hit/miss trials.

ii. ```python
keep_trials = build_keep_trial_mask(bp)
...
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. The trajectory did not contain a separate outcome-specific rationale. The choice follows the AI’s broader decision to exclude no-response trials and treat outcome as a binary correct/incorrect variable.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The `hit` flag is copied directly as the outcome code, so miss trials become 0 (`incorrect`) and hit trials become 1 (`correct`). The code does not represent an ignore class.

ii. ```python
"output_values": [
    ["left", "right"],
    ["WC", "DR"],
    ["incorrect", "correct"],
    ...
]
...
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), outcome[out_pos], dtype=np.int16)
```

iii. The trajectory justification is again implicit in the AI’s trial-exclusion choice and binary-output framing.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from several DeepLabCut tongue landmarks across both cameras in `obj["traj"]`, using each camera’s `ts` and `frameTimes`, plus session-level video offset from `obj["sglx"]` and behavioral alignment from `bp["ev"]["goCue"]`.

ii. ```python
TONGUE_FEATURES = [
    (0, "tongue"),
    (0, "left_tongue"),
    (0, "right_tongue"),
    (1, "top_tongue"),
    (1, "topleft_tongue"),
    (1, "bottom_tongue"),
    (1, "bottomleft_tongue"),
]
...
ts = get_trial_ts(cam, trix)[:, :2, feat_idx]
frame_times = get_trial_frame_times(cam, trix)
...
old_t = frame_times - vidshift - align_times[trix]
```

iii. In the trajectory, the AI said the paper code did not define one canonical tongue-velocity scalar, so it searched for “the cleanest mapping” and decided to expose a summary built from the available tongue landmarks.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each selected tongue landmark, the AI interpolated x and y onto the neural time axis, computed gradients, converted NaNs to zeros, took speed magnitude, and averaged speeds across all tongue landmarks. It did not use likelihood thresholding, per-run smoothing, or per-view normalization.

ii. ```python
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
...
xvel[:, trix] = np.gradient(tsinterp[:, 0])
yvel[:, trix] = np.gradient(tsinterp[:, 1])
...
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)
...
speeds.append(np.sqrt(xvel**2 + yvel**2))
...
out = np.nanmean(stacked, axis=0)
```

iii. The trajectory justification was that the kinematics pipeline aligned positions by interpolation to the neural time axis and used first differences. Later, the AI explicitly adjusted only the tongue thresholding rule because zero placeholders were collapsing the median.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI thresholded tongue speed into only two categories using the 50th percentile computed after dropping zero-valued samples from the threshold calculation. It did not create a third “not visible” category.

ii. ```python
def binarize_session_signal(
    signal: np.ndarray,
    keep_trials: np.ndarray,
    ignore_zeros_for_threshold: bool = False,
) -> tuple[np.ndarray, float]:
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    ...
    threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
```

iii. The trajectory explicitly says the AI changed this rule after observing many zero medians: it treated zero placeholders from invisible tongue frames as “below threshold” rather than letting them define the percentile.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI estimated one session-wide video offset from `sglx.bitcode.bitstart` and `bp.ev.bitStart`, subtracted that offset and the trial’s `goCue` from camera frame times, and interpolated tongue positions onto the neural time grid.

ii. ```python
def find_video_offset(obj: dict) -> float:
    bit_start = mode_scalar(obj["bp"]["ev"]["bitStart"])
    bitstart = as_array(obj["sglx"]["bitcode"]["bitstart"], np.float64).ravel()
    fs = float(np.asarray(obj["sglx"]["fs"]).reshape(-1)[0])
    return float(mode_scalar(bitstart) / fs - bit_start)
...
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. The trajectory explicitly said motion and kinematics should be placed on the neural time axis after video-offset correction, and that this part of the pipeline was clear from the paper code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera paw landmarks, `top_paw` and `bottom_paw`, from `obj["traj"]`, together with frame times, video offset, and go-cue alignment.

ii. ```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
...
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. In the trajectory, the AI noted uncertainty because the figure code omitted paws while the raw sessions still appeared to contain paw tracks. It chose to keep paw outputs by summarizing the available paw landmarks instead of dropping the sessions.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolated paw positions to the neural grid, computed gradients, subtracted a baseline derivative term, nearest-filled missing values, converted to speed magnitude, averaged across the two paw landmarks, and then median-thresholded the result.

ii. ```python
xvel[:, trix] = np.gradient(tsinterp[:, 0])
yvel[:, trix] = np.gradient(tsinterp[:, 1])
if not is_tongue:
    xvel[:, trix] = xvel[:, trix] - basederiv[0]
    yvel[:, trix] = yvel[:, trix] - basederiv[0]
    xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
    yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
...
out = np.nanmean(stacked, axis=0)
```

iii. The trajectory justification was that the AI believed the kinematics code implied interpolation plus first differences and nearest-value filling for non-tongue features.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is thresholded into two categories by a session median, with no explicit “not visible” category.

ii. ```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
...
binary = (kept >= threshold).astype(np.int64)
```

iii. The trajectory did not record a separate paw-threshold justification; this follows the AI’s general binary discretization scheme for continuous outputs.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw frame times are shifted by the session video offset and by the trial `goCue`, then interpolated onto the common neural time axis.

ii. ```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

iii. The trajectory justification matches 7-d: the AI said it was following the code path that aligns video features to the neural grid after video-offset correction.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the companion `motionEnergy_*.mat` file, specifically `me.data`, together with side-camera frame times from `obj["traj"][0]`, plus video offset and go-cue alignment.

ii. ```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
if hasattr(raw_data, "data"):
    raw_data = raw_data.data
...
cam = obj["traj"][0]
vidshift = find_video_offset(obj)
```

iii. The trajectory explicitly states that motion-energy traces were read separately and interpolated onto the neural time axis after video-offset correction.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI resampled motion energy onto the common time grid by interpolation and nearest-value filling, then discretized the resampled values with a per-session median threshold.

ii. ```python
for trix, trial_me in enumerate(raw_trials):
    ...
    old_t = frame_times - vidshift - align_times[trix]
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
    resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
...
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. The trajectory justification was explicit: the AI said the loader code showed motion-energy traces were interpolated to the neural time axis.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is thresholded into two categories by the session median of kept-trial values. There is no third “no video” category.

ii. ```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
...
binary = (kept >= threshold).astype(np.int64)
```

iii. The trajectory did not record a motion-energy-specific threshold justification beyond the general decision to produce binary kinematic outputs for the decoder.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame values are time-shifted using the session video offset and per-trial `goCue`, then interpolated onto the same neural time axis used everywhere else.

ii. ```python
cam = obj["traj"][0]
vidshift = find_video_offset(obj)
...
old_t = frame_times - vidshift - align_times[trix]
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. The trajectory explicitly says motion energy was aligned to the neural time axis after video-offset correction.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally fills through missing data rather than preserving a missing-data class. If frame times are absent it fabricates fallback 400 Hz frame times; for non-tongue positions and motion energy it nearest-fills NaNs with neighboring values or zeros; for tongue velocities it converts NaNs to zeros before thresholding.

ii. ```python
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]
...
if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
    ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)
...
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
...
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The trajectory justification was that the paper kinematics code used interpolation, nearest filling for non-tongue features, and zero placeholders for tongue invisibility. The exact fallback-frame-time behavior was not explicitly justified in the trajectory.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive parts are the session-by-session spike binning loops and the video interpolation / feature-speed calculations for kinematics and motion energy.

ii. ```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    np.add.at(counts, (spike_trials, bins), 1.0)
...
for trix in range(ntrials):
    ...
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
...
for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
```

iii. The trajectory says this directly: “The converter is spending most of its time in session-by-session spike binning and video interpolation.”

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious vectorization opportunities are the per-unit loop in `build_aligned_trialdat`, the per-trial interpolation loops in `find_position` and `load_motion_energy`, and the per-feature loop in `compute_feature_speed`.

ii. ```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
for trix in range(ntrials):
    ...
for view_index, feat_name in feature_specs:
    ...
```

iii. The trajectory did not explicitly discuss vectorization opportunities. The AI only noted that these loops were the main runtime cost.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats interpolation and velocity calculation separately for every selected tongue and paw landmark, even though all of those features are ultimately collapsed to one average speed per modality. It also rebuilds scratch arrays inside the per-unit and per-trial loops.

ii. ```python
for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
    xvel, yvel = find_velocity(xpos, ypos, feat_name)
    speeds.append(np.sqrt(xvel**2 + yvel**2))
...
counts = np.zeros((ntrials, NT), dtype=np.float64)
```

iii. The trajectory did not call this out as a separate design choice. It only noted that the runtime hotspot was spike binning plus video interpolation.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The biggest discarded work is computing separate x/y traces and velocities for many individual tongue and paw landmarks and then immediately averaging them away into one binary stream. The motion-energy loader also reads and stores `moveThresh` even though the downstream export never uses it.

ii. ```python
TONGUE_FEATURES = [
    (0, "tongue"),
    (0, "left_tongue"),
    (0, "right_tongue"),
    (1, "top_tongue"),
    (1, "topleft_tongue"),
    (1, "bottom_tongue"),
    (1, "bottomleft_tongue"),
]
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
...
out = np.nanmean(stacked, axis=0)
...
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
    "vidshift": vidshift,
}
```

iii. The trajectory rationale was that the AI wanted scalar tongue and paw outputs and could not find one canonical variable in the source code, so it built composite summaries instead. It did not discuss the efficiency cost of discarding the per-feature intermediates.
