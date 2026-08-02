# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a 12-session roster, then iterates through it and loads one session `.mat` object with `mat73` plus one companion motion-energy `.mat` file with `scipy.io.loadmat`. It does not use the MATLAB metadata loaders at runtime; instead it manually reproduces the Figure 8 roster in `SESSION_SPECS`.

ii. ```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    bp = obj["bp"]
    ...
    me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. In `CONVERSION_NOTES.md`, the agent says it followed the Figure 8 context-analysis loader and lists the same 12-session roster explicitly.

## 1-b. How are the data split into subjects?

i. Sessions are grouped into subjects by the `animal` field in each `SessionSpec`. A subject list is built in first-seen order, and each session gets a `subject_idx` pointing into that list.

ii. ```python
subject_to_idx: dict[str, int] = {}
subjects: list[str] = []

subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)

full_data["subject_idx"].append(subject_idx)
```

iii. In `CONVERSION_NOTES.md`, the agent justifies this by saying it kept the released-code session roster even though the paper text says 6 mice, because the Figure 8 loader explicitly names 7 animal IDs.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` corresponds to one exported session. The outer loop over `SESSION_SPECS` creates one entry in `full_data["neural"]`, `full_data["input"]`, `full_data["output"]`, and `brain_region_idx` per session.

ii. ```python
for spec in SESSION_SPECS:
    ...
    neural_trials = []
    input_trials = []
    output_trials = []
    ...
    full_data["neural"].append(neural_trials)
    full_data["input"].append(input_trials)
    full_data["output"].append(output_trials)
    full_data["subject_idx"].append(subject_idx)
    full_data["brain_region_idx"].append(np.zeros(trialdat.shape[1], dtype=np.int64))
```

iii. The agent says in `CONVERSION_NOTES.md` that it converted the paper's "two-context ALM ephys cohort" and used the Figure 8 session roster directly.

## 1-d. How are the data split into trials?

i. Within each session, the code creates one exported trial per kept raw trial index. It stores one `(neurons, time)` neural matrix, one `(1, time)` input matrix, and one `(6, time)` output matrix for each kept trial.

ii. ```python
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)

for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
    input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
    output_trials.append(
        np.vstack([...])
    )
```

iii. The notes say the export keeps hit and miss trials from both contexts after trial filtering, and the trajectory says the final dataset has 2415 exported trials across 12 sessions.

## 1-e. How are trials filtered based on quality controls?

i. Exported trials are restricted to hit or miss trials, excluding early-lick, no-response, and `stim.enable` trials. Sessions are also rejected if fewer than 2 valid trials remain.

ii. ```python
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

iii. `CONVERSION_NOTES.md` states that trial export filtering follows the paper text: exclude early-lick, no-response, and stim trials, while keeping hit and miss trials from DR and WC.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data come from the chosen probe's spike-trial assignments and spike times, aligned by the session's `bp.ev.goCue` values. Unit quality labels are used for QC.

ii. ```python
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
probe = obj["clu"][spec.probe - 1]

spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
```

iii. The agent's notes say neural alignment and unit filtering were intended to match the shared MATLAB pipeline: align to `goCue`, use the Figure 8 probe choice, and filter by quality and firing rate.

## 2-b. How is the `neural` data processed?

i. For each unit, spikes are aligned to per-trial `goCue`, binned on a fixed `-3.0` to `2.5` s window with `dt = 0.01` s, converted to firing rates by dividing by `DT`, then smoothed with a causal half-Gaussian window of length 15 using reflect padding.

ii. ```python
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent matched the MATLAB neural pipeline: `goCue` alignment, `dt = 0.01`, `tmin=-3`, `tmax=2.5`, and causal Gaussian smoothing with `smooth = 15`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The code first drops units whose `quality` labels are `garbage`, `gabrga`, `noisy`, or `real?`. It then computes mean firing rate across the seven Figure 8 conditions and drops units with mean FR `<= 1 Hz`. Sessions with fewer than 10 units remaining are rejected.

ii. ```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad

mean_frs = psth.mean(axis=0).mean(axis=1)
keep = mean_frs > LOW_FR

if trialdat.shape[1] < 10:
    raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")
```

iii. The notes say this was meant to match the shared code path: `quality={'all'}`, remove `garbage/gabrga/noisy/real?`, remove low-FR units, and keep only sessions with at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to each trial's `goCue` value by subtracting `go_cue[trial]` from every spike time before binning. The same alignment is used in both DR and WC trials.

ii. ```python
ALIGN_EVENT = "goCue"
...
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
...
aligned = spike_times - go_cue[spike_trials]
```

iii. The notes say the agent aligned to the session object's `goCue` field and treated that field as the shared go-cue / water-drop alignment reference, following the released code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins. There is no second rebinning stage after that; all exported trials stay on the same 550-bin grid.

ii. ```python
DT = 1 / 100

def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. The notes explicitly list `dt = 0.01 s`, and the trajectory's verification output reports `T = 550` for every trial.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not derived from a raw sampled stream. The exported input is the fixed analysis time axis defined by `TMIN`, `TMAX`, and `DT`, interpreted relative to the chosen alignment event `goCue`.

ii. ```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
EDGES, TIME = build_time_axis()
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes say `input[trial]` is `time_from_go_cue_s`, shape `(1, 550)`, which shows the agent treated it as a decoder-specific relative-time regressor rather than a separate raw variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs bin centers from the fixed `goCue`-aligned window and exports the same float32 1D time axis for every kept trial.

ii. ```python
def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time

input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes justify this only briefly: `input[trial]` is `time_from_go_cue_s`, consistent with the decoder instructions.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It uses exactly the same `TIME` grid as the neural binning window, so each neural time bin and each input time sample have identical centers relative to `goCue`.

ii. ```python
EDGES, TIME = build_time_axis()
...
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The notes state that all trials have 550 time bins and are aligned to `goCue`, which is the agent's main justification for using the same axis for neural and input streams.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The code derives it from `bp["R"]` alone, after trial filtering. It does not use lick-event times or combine `R/L` with `hit/miss` to infer actual chosen side.

ii. ```python
def build_output_constants(bp: dict, keep_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    ...
    return lick_direction, context, outcome
```

iii. In `CONVERSION_NOTES.md`, the agent explicitly says `lick_direction` is taken from `bp.R` with `left=0`, `right=1`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. After selecting kept trials, the code casts `bp.R` to integers and repeats the per-trial label across all 550 time bins, making the variable constant in time.

ii. ```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), lick_direction[out_pos], dtype=np.int16)
```

iii. The notes say `lick_direction` is "constant over time" and is one of the decoder-specific output constructions.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. It is derived from `bp["autowater"]`. The code interprets autowater trials as WC and non-autowater trials as DR.

ii. ```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The notes say behavioral context is derived from `bp.autowater` and encoded `WC=0`, `DR=1`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code negates the boolean `autowater` mask so that WC becomes 0 and DR becomes 1, then repeats that session trial label across all time bins.

ii. ```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
...
np.full((1, NT), context[out_pos], dtype=np.int16)
```

iii. The notes justify this as the direct mapping from the codebase's context proxy: autowater indicates WC blocks, and the decoder requested `WC=0`, `DR=1`.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]` after the trial mask removes no-response and early trials. Misses become 0 and hits become 1.

ii. ```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. The notes say `outcome` is derived from `bp.hit` with `incorrect=0`, `correct=1`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code uses the hit flag directly on kept trials and tiles the per-trial outcome label across the full time axis.

ii. ```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), outcome[out_pos], dtype=np.int16)
```

iii. The notes justify this as a decoder-specific categorical output, constant over time.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DeepLabCut trajectories in `obj["traj"]` using seven tongue landmarks: side-camera `tongue`, `left_tongue`, `right_tongue`, and bottom-camera `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue`.

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
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The notes say tongue velocity was computed as the mean speed across tracked tongue landmarks from both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue landmark, the code interpolates x/y positions onto the neural time axis, keeps tongue NaNs until the velocity step, computes velocity as `np.gradient` for x and y, converts missing tongue velocities to zero, computes Euclidean speed, and averages speed across all listed tongue landmarks.

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

iii. The notes say this preserves the MATLAB special handling for tongue visibility, including the use of zero as the placeholder when the tongue is not visible.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The code binarizes tongue speed with a per-session median, but it estimates that threshold only from strictly positive samples on kept trials. Zeros, which mostly reflect tongue invisibility placeholders, are excluded from threshold estimation and then classified using the resulting threshold.

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

tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed,
    keep_trials,
    ignore_zeros_for_threshold=True,
)
```

iii. `CONVERSION_NOTES.md` gives the justification explicitly: the released MATLAB code uses `0` when the tongue is not visible, so the agent chose to estimate the 50th-percentile threshold from strictly positive tongue-speed samples instead.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are resampled onto `taxis = TIME + ADVANCE_MOVEMENT`, where `TIME` is the neural bin-center grid and `ADVANCE_MOVEMENT` is 0, using `goCue`-relative timestamps and the video offset correction. The binned categorical tongue output is then exported on exactly that same time grid.

ii. ```python
align_times = go_cue
taxis = TIME + ADVANCE_MOVEMENT
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
...
tongue_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False)
```

iii. The notes say the agent matched `findPosition.m` and `findVelocity.m` by interpolating tracked positions onto the neural time axis after video-offset correction.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera DeepLabCut trajectories for `top_paw` and `bottom_paw`.

ii. ```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
...
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The notes say paw velocity is the mean speed across `top_paw` and `bottom_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The code interpolates paw x/y coordinates onto the neural time axis, fills missing values with nearest available samples, computes x/y velocity with `np.gradient`, subtracts the per-trial baseline derivative for non-tongue features, converts to Euclidean speed, and averages across the two paw landmarks.

ii. ```python
if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
    ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)
...
xvel[:, trix] = np.gradient(tsinterp[:, 0])
yvel[:, trix] = np.gradient(tsinterp[:, 1])
...
xvel[:, trix] = xvel[:, trix] - basederiv[0]
yvel[:, trix] = yvel[:, trix] - basederiv[0]
...
out = np.nanmean(stacked, axis=0)
```

iii. The notes say this was intended to preserve the MATLAB baseline-subtraction behavior for non-tongue velocities.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is thresholded at the 50th percentile of the kept session samples, with no special exclusion of zeros.

ii. ```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

iii. The notes say paw velocity was thresholded at the session median over exported trials and time points.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories are interpolated to the same `TIME`-based neural axis after video-offset correction and `goCue` alignment, and the categorical paw signal is exported on that same grid.

ii. ```python
taxis = TIME + ADVANCE_MOVEMENT
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
...
paw_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False)
```

iii. The notes say kinematic alignment matches the paper code by interpolating tracked positions onto the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate session-level `motionEnergy_*.mat` file, specifically `me.data` for each trial, with auxiliary timing information from the session object's side-camera frame times and SpikeGLX bitcode fields for video offset correction.

ii. ```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
...
vidshift = find_video_offset(obj)
frame_times = get_trial_frame_times(cam, trix)
```

iii. The notes say motion energy loading matches `loadMotionEnergy.m`: load the companion file, correct video timing with the same offset logic, interpolate, and fill missing values.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, raw motion-energy values are interpolated from video frame times onto the neural time axis after subtracting video offset and per-trial `goCue`. When frame times are missing, a 400 Hz fallback axis shifted by 0.5 s is used. Missing values are filled with nearest samples.

ii. ```python
for trix, trial_me in enumerate(raw_trials):
    trial_me = as_array(trial_me, np.float64).ravel()
    frame_times = get_trial_frame_times(cam, trix)
    use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
    if use_fallback:
        frame_times = np.arange(1, trial_me.size + 1, dtype=np.float64) / 400.0
        old_t = frame_times - 0.5 - align_times[trix]
    else:
        old_t = frame_times - vidshift - align_times[trix]
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
    resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The notes say this mirrors the released MATLAB motion-energy loader.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The continuous resampled motion-energy signal is binarized at the per-session 50th percentile over kept trials and time points. The code loads `me.moveThresh` from the source file but does not use that manual paper threshold for export.

ii. ```python
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
    "vidshift": vidshift,
}
...
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. The notes justify this as a decoder-specific discretization choice: the paper's motion-energy threshold is continuous/manual, but the decoder task required p50 binning.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is explicitly resampled onto the neural `TIME` axis after subtracting video offset and trialwise `goCue`, then the binarized motion-energy output is emitted on that same aligned grid.

ii. ```python
align_times = go_cue
taxis = TIME + ADVANCE_MOVEMENT
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
...
motion_binary[:, out_pos][np.newaxis, :].astype(np.int16, copy=False)
```

iii. The notes say motion energy was interpolated onto the neural time axis using the same `findVideoOffset` logic as the MATLAB code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several irregularities defensively: missing `stim.enable` becomes all-false; missing string formats are flattened; missing frame times fall back to a synthetic 400 Hz axis shifted by 0.5 s; dropped-frame trials are skipped in kinematic interpolation; non-tongue NaNs are nearest-filled; tongue velocity NaNs become 0; all-NaN arrays get filled with a constant 0 by `fill_nearest_1d`; and missing motion-energy values are nearest-filled.

ii. ```python
if isinstance(stim, dict) and "enable" in stim:
    return as_array(stim["enable"], bool).ravel()
return np.zeros(int(bp["Ntrials"]), dtype=bool)

if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]

if not mask.any():
    out[:] = fill_value
    return out

xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
...
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The notes justify most of this by saying the agent wanted to preserve the MATLAB missing-value handling for non-tongue versus tongue features and for motion energy. The trajectory also shows a runtime warning from an all-NaN slice in `find_velocity`, which the code absorbs by resetting the baseline derivative to zero.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive steps are the nested per-session/per-unit/per-trial neural binning and smoothing, plus repeated per-feature kinematic interpolation and velocity calculation for every trial. Motion-energy interpolation across every trial is another large cost.

ii. ```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    np.add.at(counts, (spike_trials, bins), 1.0)
    ...
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)

for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
    xvel, yvel = find_velocity(xpos, ypos, feat_name)

for trix, trial_me in enumerate(raw_trials):
    ...
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. There is no explicit efficiency justification in the notes. The implementation reflects a direct Python port of the MATLAB loops.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest vectorization opportunities are the per-unit spike histogram loop in `build_aligned_trialdat`, the per-column convolution loop inside `my_smooth`, the per-trial loops in `find_position`, `find_velocity`, and `load_motion_energy`, and the final Python loop that constructs one trial object at a time.

ii. ```python
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")

for trix in range(ntrials):
    ...

for unit_pos, unit_idx in enumerate(keep_units):
    ...

for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(...)
    input_trials.append(...)
    output_trials.append(...)
```

iii. The notes do not discuss vectorization. This is an implicit decision to prioritize direct source matching over optimization.

## 11-c. What processing does the code repeat multiple times?

i. The code repeatedly computes interpolation plus nearest-fill for every feature and every trial, repeats the same gradient-based velocity logic separately for each feature, recomputes session medians for three outputs through the same helper, and repeatedly materializes constant-over-time output arrays for per-trial labels.

ii. ```python
xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
xvel, yvel = find_velocity(xpos, ypos, feat_name)
speeds.append(np.sqrt(xvel**2 + yvel**2))

tongue_binary, tongue_threshold = binarize_session_signal(...)
paw_binary, paw_threshold = binarize_session_signal(...)
motion_binary, motion_threshold = binarize_session_signal(...)

np.full((1, NT), lick_direction[out_pos], dtype=np.int16)
np.full((1, NT), context[out_pos], dtype=np.int16)
np.full((1, NT), outcome[out_pos], dtype=np.int16)
```

iii. No explicit justification is given. The repeated structure comes from reusing the same helper pattern for different exported variables.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code loads `moveThresh` from the motion-energy file but never uses it for the exported dataset, computes `mean_frs` for all surviving units only to keep summary stats, computes session summaries and sample-subset metadata that are not used by downstream decoder training, and constructs a stratified `sample_data` dataset even though the main deliverable is the full dataset.

ii. ```python
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
    "vidshift": vidshift,
}
...
session_summary = {
    ...
    "mean_fr_hz_min": float(mean_frs[low_fr_keep].min()),
    "mean_fr_hz_max": float(mean_frs[low_fr_keep].max()),
    ...
}
...
sample_data = build_sample_dataset(full_data)
```

iii. The notes justify most of this as documentation and sanity-check support rather than as part of the decoder inputs themselves.
