# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent hard-codes a 12-session roster in `SESSION_SPECS`, all under `/app/data/Ephys_Behavior`, and loads each `data_structure_<animal>_<date>.mat` with `mat73.loadmat`. Motion energy is loaded separately from the companion `motionEnergy_*.mat` file with `scipy.io.loadmat`. The code does not search `RandomizedDelay_Ephys_Behavior`, does not use the 44-session roster from the paper loaders, and does not implement the reference's dual MATLAB v7.3/v5 reader for the main session files.

ii. 
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    ...
    SessionSpec("JEB19", "2023-04-18", 1),
]
```

```python
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
```

```python
def load_motion_energy(obj: dict, motion_energy_path: Path, taxis: np.ndarray, align_times: np.ndarray) -> dict:
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
```

iii. In `CONVERSION_NOTES.md`, the agent says the session roster "follows the paper's Figure 8 context-analysis loader" and that it kept the released-code roster because it was "explicit and task-specific" even though it yielded 12 sessions, 7 subjects, and 520 units instead of the paper text's 6 mice and 522 units.

## 1-b. How are the data split into subjects?

i. Subjects are defined by the `animal` field in each `SessionSpec`. During assembly, the code assigns each unique animal a `subject_idx` in order of first appearance and appends the subject names to `subjects`.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int
```

```python
subject_to_idx: dict[str, int] = {}
subjects: list[str] = []
...
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
...
full_data["subject_idx"].append(subject_idx)
```

iii. The justification is implicit in the hard-coded roster and explicit in `CONVERSION_NOTES.md`, which reports that the Figure 8 roster yields 7 distinct animal IDs and says the agent kept that released-code roster.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` is treated as one session. One probe is chosen per session from the hard-coded 12-session Figure 8 list, and each session becomes one element of `neural`, `input`, and `output`.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    ...
]
```

```python
for spec in SESSION_SPECS:
    ...
    full_data["neural"].append(neural_trials)
    full_data["input"].append(input_trials)
    full_data["output"].append(output_trials)
```

iii. The agent's stated reason is that the "released-code session roster" from `Figure8a_thru_c.m` is explicit and task-specific, so it used that roster rather than the paper counts.

## 1-d. How are the data split into trials?

i. Trials are indexed by the Bpod trial number `0..Ntrials-1`. The code reads `bp["Ntrials"]`, builds full trial-aligned arrays with trial as the last axis, and then exports one trial at a time using `kept_trial_indices`.

ii.
```python
ntrials = int(bp["Ntrials"])
```

```python
for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32, copy=False))
    input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
    output_trials.append(
        np.vstack(
            [
                np.full((1, NT), lick_direction[out_pos], dtype=np.int16),
                ...
            ]
        )
    )
```

iii. No separate justification is given beyond using the session object's Bpod trial structure; this is the indexing convention the rest of the conversion code assumes.

## 1-e. How are trials filtered based on quality controls?

i. Trials are kept only if they are hit or miss trials and are not early-lick, no-response, or `stim.enable` trials. Sessions are also required to retain at least 2 such trials. The code does not implement the reference's "recording stopped before behavior ended" trial cutoff.

ii.
```python
def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim
```

```python
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)
if kept_trial_indices.size < 2:
    raise ValueError(f"{spec.stem}: fewer than 2 valid trials remained after filtering")
```

iii. `CONVERSION_NOTES.md` says this "follows the paper text": exclude early-lick, no-response, and `stim.enable` trials, and keep hit and miss trials from both DR and WC contexts.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from one selected probe in `obj["clu"]`, specifically the cluster-level `trial`, `trialtm`, and `quality` fields, together with `bp["ev"]["goCue"]` for alignment.

ii.
```python
probe = obj["clu"][spec.probe - 1]
trialdat = build_aligned_trialdat(probe, go_cue, ntrials)
quality_mask = get_quality_mask(probe)
```

```python
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
...
aligned = spike_times - go_cue[spike_trials]
```

iii. In `CONVERSION_NOTES.md`, the agent says its neural alignment and filtering match the shared MATLAB pipeline, specifically aligning to `goCue` and filtering clusters by the code path's quality and firing-rate rules.

## 2-b. How is the `neural` data processed?

i. The agent bins aligned spikes from `-3.0` s to `2.5` s in `10` ms bins, converts counts to firing rates by dividing by `DT`, and smooths with a custom causal half-Gaussian window `SMOOTH = 15` using `my_smooth(..., bctype="reflect")`.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15
```

```python
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. `CONVERSION_NOTES.md` explicitly says the agent chose `-3.0` to `2.5` s, `dt = 0.01` s, and "causal Gaussian smoothing with `smooth = 15` and the same boundary handling as `mySmooth.m`."

## 2-c. How is the `neural` data filtered based on quality controls?

i. First, the code keeps all clusters except those labeled `garbage`, `gabrga`, `noisy`, or `real?`. Then it computes a low-firing-rate filter from condition-averaged PSTHs over seven trial-condition masks and keeps units whose mean PSTH across conditions exceeds `1 Hz`. Finally, it requires at least 10 units to remain in each exported session.

ii.
```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad
```

```python
def apply_low_fr_filter(trialdat: np.ndarray, bp: dict) -> tuple[np.ndarray, np.ndarray]:
    conds = build_low_fr_conditions(bp)
    psth = np.zeros((trialdat.shape[0], trialdat.shape[1], len(conds)), dtype=np.float32)
    for cond_idx, cond_mask in enumerate(conds):
        trials = np.flatnonzero(cond_mask)
        if trials.size:
            psth[:, :, cond_idx] = trialdat[:, :, trials].mean(axis=2)
    mean_frs = psth.mean(axis=0).mean(axis=1)
    keep = mean_frs > LOW_FR
    return keep, mean_frs
```

```python
if trialdat.shape[1] < 10:
    raise ValueError(f"{spec.stem}: fewer than 10 units remained after filtering")
```

iii. `CONVERSION_NOTES.md` says this matches the shared code path: start from `quality = {'all'}`, drop the four listed quality labels, drop units with mean firing rate `<= 1 Hz`, and keep only sessions with at least 10 units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. For each spike, the code subtracts the trial's `goCue` time from `trialtm`, then bins the aligned spike time into the fixed analysis window.

ii.
```python
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
in_window = (bins >= 0) & (bins < NT)
```

iii. The justification is explicit in `CONVERSION_NOTES.md`: the neural data are "aligned to the session object's `goCue` field" to match the shared MATLAB pipeline.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use a fixed `10 ms` bin size (`DT = 1/100`) over 550 bins from `-3.0` to `2.5` s. Spikes are counted directly into that grid; there is no later temporal rebinning step.

ii.
```python
DT = 1 / 100
...
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

```python
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. `CONVERSION_NOTES.md` states that the neural bins are 10 ms and that all trials have exactly 550 time bins after conversion.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is a synthetic time axis built from `TMIN`, `TMAX`, and `DT`, with trial alignment conceptually tied to `bp["ev"]["goCue"]`. It is not computed from a separate per-sample raw variable.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
...
EDGES, TIME = build_time_axis()
```

```python
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. `CONVERSION_NOTES.md` says `input[trial]` is `time_from_go_cue_s`, shape `(1, 550)`, and that the conversion is aligned to the session object's `goCue` field.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The code constructs evenly spaced bin edges with `np.arange`, converts them to bin centers, and reuses that same `TIME` vector for every trial.

ii.
```python
def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

```python
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The only explicit justification is in `CONVERSION_NOTES.md`, which describes the decoder input as a `(1, 550)` time-from-go-cue trace.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input is exactly the same bin-center grid used to define the neural bins. Each trial gets the same `TIME` vector, and `NT` is shared between the neural arrays and the input arrays.

ii.
```python
EDGES, TIME = build_time_axis()
NT = TIME.size
```

```python
trialdat = np.zeros((NT, n_units, ntrials), dtype=np.float32)
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The justification is implicit: by reusing the same `TIME`/`NT` grid for all modalities, the agent ensures the input and neural data are on the same timeline.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The agent derives lick direction directly from `bp["R"]` on the kept trials. It does not use `hit`, `miss`, or `no` to reconstruct the actual lick side.

ii.
```python
def build_output_constants(bp: dict, keep_trials: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
    outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
    return lick_direction, context, outcome
```

iii. `CONVERSION_NOTES.md` explicitly says `lick_direction` is "taken from `bp.R`" and is coded `left=0`, `right=1`.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No extra processing is applied beyond filtering trials and taking the `bp.R` value. That 0/1 value is repeated across all time bins of the trial.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
```

```python
np.full((1, NT), lick_direction[out_pos], dtype=np.int16)
```

iii. The justification given in `CONVERSION_NOTES.md` is that `lick_direction` should be a constant per-trial output and that the agent chose to take it directly from `bp.R`.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the `bp["autowater"]` trial flag.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` explicitly says `behavioral_context` is derived from `bp.autowater`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The code inverts the `autowater` boolean so that `WC=0` and `DR=1`, then repeats the result across all time bins of the trial.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

```python
np.full((1, NT), context[out_pos], dtype=np.int16)
```

iii. `CONVERSION_NOTES.md` says the output is coded `WC=0`, `DR=1`, matching the decoder task.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]` on the kept trials. Because the agent already removed `bp["no"]` trials and only keeps hit/miss trials, `hit=1` becomes correct and `hit=0` becomes incorrect.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. `CONVERSION_NOTES.md` explicitly says `outcome` is "derived from `bp.hit`" and coded `incorrect=0`, `correct=1`.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The code uses the filtered `bp["hit"]` values directly as a 0/1 trial label and repeats that label across the full time axis of the trial.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

```python
np.full((1, NT), outcome[out_pos], dtype=np.int16)
```

iii. The agent's stated reason in `CONVERSION_NOTES.md` is that the decoder task requires `incorrect=0` and `correct=1`, so it exported the per-trial outcome that way.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from multiple DeepLabCut tongue landmarks listed in `TONGUE_FEATURES`, using both camera views in `obj["traj"]`, each trial's `frameTimes`, and the video/behavior offset computed from `bp["ev"]["bitStart"]`, `sglx["bitcode"]["bitstart"]`, and `sglx["fs"]`. Alignment uses per-trial `goCue`.

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

```python
def find_video_offset(obj: dict) -> float:
    bit_start = mode_scalar(obj["bp"]["ev"]["bitStart"])
    bitcode = obj["sglx"]["bitcode"]
    bitstart = as_array(bitcode["bitstart"], np.float64).ravel()
    fs = float(np.asarray(obj["sglx"]["fs"]).reshape(-1)[0])
    vid_file_offset = mode_scalar(bitstart) / fs
    return float(vid_file_offset - bit_start)
```

```python
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. `CONVERSION_NOTES.md` says tongue velocity is "computed as mean speed across the tracked tongue landmarks from both cameras" and that the kinematic alignment matches `findPosition.m` and `findVelocity.m`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue landmark, the agent interpolates x/y coordinates onto the neural time axis, computes frame-to-frame gradients on that interpolated trace, converts NaNs to zero for tongue features, computes speed as `sqrt(xvel**2 + yvel**2)`, and averages those speeds across all seven tongue landmarks.

ii.
```python
def find_position(...):
    ...
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

```python
def find_velocity(xpos: np.ndarray, ypos: np.ndarray, feat_name: str) -> tuple[np.ndarray, np.ndarray]:
    ...
    xvel[:, trix] = np.gradient(tsinterp[:, 0])
    yvel[:, trix] = np.gradient(tsinterp[:, 1])
    ...
    else:
        xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
        yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)
```

```python
def compute_feature_speed(...):
    speeds = []
    for view_index, feat_name in feature_specs:
        xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
        xvel, yvel = find_velocity(xpos, ypos, feat_name)
        speeds.append(np.sqrt(xvel**2 + yvel**2))
    ...
    out = np.nanmean(stacked, axis=0)
```

iii. The justification in `CONVERSION_NOTES.md` is that this follows the MATLAB `findPosition.m`/`findVelocity.m` path, preserves the "special tongue handling," and averages tracked tongue landmarks from both cameras.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is binarized per session at the 50th percentile, but the threshold is estimated from strictly positive tongue-speed samples only (`ignore_zeros_for_threshold=True`). The exported tongue output is then `0` below threshold and `1` at or above threshold. There is no separate "not visible" class.

ii.
```python
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
    return binary, threshold
```

```python
tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed,
    keep_trials,
    ignore_zeros_for_threshold=True,
)
```

iii. `CONVERSION_NOTES.md` explicitly justifies the positive-only threshold source by saying the shared MATLAB code uses `0` as a placeholder when the tongue is not visible, so the agent estimated the median from strictly positive samples and then applied it to the full time series.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The code subtracts a session-wide video offset and the trial's `goCue` from each frame time, then interpolates the tongue trajectories onto the same `TIME` grid used for the neural data. If frame times are missing, it fabricates a `400 Hz` frame-time axis and aligns that instead.

ii.
```python
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
```

```python
taxis = TIME + ADVANCE_MOVEMENT
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. `CONVERSION_NOTES.md` says motion and kinematics are "correct[ed for] video timing by the same `findVideoOffset` logic" and "interpolate[d] onto the neural time axis."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera DeepLabCut landmarks, `top_paw` and `bottom_paw`, plus their `frameTimes`, the session video offset, and per-trial `goCue` times.

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
```

```python
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. `CONVERSION_NOTES.md` explicitly says paw velocity is "computed as mean speed across `top_paw` and `bottom_paw`."

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw positions are interpolated onto the neural time axis, differentiated with `np.gradient`, baseline-subtracted using `basederiv[0]` for both x and y, nearest-filled where missing, converted to speed magnitude, and averaged across `top_paw` and `bottom_paw`.

ii.
```python
if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
    ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)
```

```python
if not is_tongue:
    xvel[:, trix] = xvel[:, trix] - basederiv[0]
    yvel[:, trix] = yvel[:, trix] - basederiv[0]
    xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
    yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
```

```python
out = np.nanmean(stacked, axis=0)
```

iii. The justification in `CONVERSION_NOTES.md` is that the kinematic code preserves the MATLAB baseline-subtraction behavior for non-tongue velocities.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is binarized at the session median over the kept trials/timepoints. Samples below the threshold are `0` and samples at or above threshold are `1`. There is no explicit missing-data category.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

```python
threshold = float(np.nanpercentile(threshold_source, 50))
binary = (kept >= threshold).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says paw velocity is "thresholded at the session median over exported trials/timepoints."

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw trajectories use the same alignment path as tongue trajectories: frame time minus session video offset minus trial `goCue`, then interpolation onto the shared neural `TIME` grid.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
...
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

```python
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The stated justification is the same as for tongue velocity: match `findVideoOffset`, `findPosition.m`, and `findVelocity.m`, then put the kinematics on the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the separate `motionEnergy_<session>.mat` file, specifically `me.data`, together with side-camera `frameTimes`, the session video offset, and per-trial `goCue` times for alignment.

ii.
```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
if hasattr(raw_data, "data"):
    raw_data = raw_data.data
raw_trials = list(np.asarray(raw_data, dtype=object).ravel())
```

```python
cam = obj["traj"][0]
vidshift = find_video_offset(obj)
```

iii. `CONVERSION_NOTES.md` says motion energy loads the companion `motionEnergy_*.mat` file and uses the same `findVideoOffset` logic as the video streams.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The agent unwraps one nested `me.data` layer if needed, then interpolates each trial's motion-energy trace onto the neural time grid and fills missing bins by nearest value.

ii.
```python
raw_data = me.data
if hasattr(raw_data, "data"):
    raw_data = raw_data.data
```

```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. `CONVERSION_NOTES.md` explicitly says the motion-energy trace is "interpolate[d] onto the neural time axis" and that missing values are "fill[ed] with nearest samples."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is binarized per session at the 50th percentile over the kept trials/timepoints. The output is `0` below threshold and `1` at or above threshold.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

```python
threshold = float(np.nanpercentile(threshold_source, 50))
binary = (kept >= threshold).astype(np.int64)
```

iii. `CONVERSION_NOTES.md` says motion energy is "thresholded at the session median over exported trials/timepoints."

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by subtracting the session video offset and trial `goCue` from each side-camera frame time, then interpolating the trace onto the shared neural `TIME` grid. If frame times are missing, the code uses a synthetic `400 Hz` frame-time axis.

ii.
```python
if use_fallback:
    frame_times = np.arange(1, trial_me.size + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]
else:
    old_t = frame_times - vidshift - align_times[trix]
```

```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. The justification in `CONVERSION_NOTES.md` is that motion energy uses the same video-offset correction and interpolation-to-neural-axis strategy as the other kinematic streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The agent generally imputes rather than preserves missingness. If a trial's `NdroppedFrames` entry begins with `NaN`, that trajectory is skipped. If `frameTimes` are missing or all non-finite, the code invents a `400 Hz` time axis. Interpolation returns NaNs outside the observed support, then non-tongue positions/velocities and motion energy are nearest-filled, while tongue velocities are converted to zero. There is no dedicated "not visible" output class.

ii.
```python
dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
if dropped.size and np.isnan(dropped[0]):
    continue
```

```python
use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
```

```python
xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)
...
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)
...
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The main stated justification is in `CONVERSION_NOTES.md`: motion energy should "fill missing values with nearest samples," and for tongue thresholding the agent says the MATLAB code uses `0` as a placeholder when the tongue is not visible.

## 11-a. What are the most time-consuming steps of the code?

i. The agent does not explicitly document runtime profiling. From the code, the most time-consuming work is likely the per-unit spike binning/smoothing loop in `build_aligned_trialdat`, the per-trial/per-feature interpolation and velocity computation in `find_position` and `find_velocity`, and the per-trial motion-energy interpolation in `load_motion_energy`.

ii.
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    np.add.at(counts, (spike_trials, bins), 1.0)
    ...
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

```python
for trix in range(ntrials):
    ...
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
```

```python
for trix, trial_me in enumerate(raw_trials):
    ...
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. No explicit justification was documented in `CONVERSION_NOTES.md`; this is an inference from the structure of the code.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The code keeps several Python loops that could potentially be reduced or batched: the unit loop in `build_aligned_trialdat`, the per-trial loops in `find_position`, `find_velocity`, and `load_motion_energy`, and the per-trial export loop that assembles `neural`, `input`, and `output`.

ii.
```python
for unit_pos, unit_idx in enumerate(keep_units):
    ...
```

```python
for trix in range(ntrials):
    ...
```

```python
for out_pos, trial_idx in enumerate(kept_trial_indices):
    ...
```

iii. The agent gives no explicit efficiency rationale. The notes focus on matching the MATLAB logic rather than optimizing the Python implementation.

## 11-c. What processing does the code repeat multiple times?

i. Similar interpolation and velocity work is repeated separately for every tongue and paw landmark. The constant `TIME[np.newaxis, :]` input array is also re-cast once per exported trial. The code does reuse some session-wide quantities, such as the video offset and the hard-coded time axis, but otherwise processes each feature independently.

ii.
```python
for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
    xvel, yvel = find_velocity(xpos, ypos, feat_name)
    speeds.append(np.sqrt(xvel**2 + yvel**2))
```

```python
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The implicit justification is fidelity to the referenced MATLAB helper functions; `CONVERSION_NOTES.md` repeatedly says the agent chose to preserve those code paths.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several computations are only used for reporting or are discarded after binarization: continuous tongue/paw/motion-energy arrays are computed only to threshold them; `moveThresh` is read from the motion-energy file but not used downstream; `mean_frs` is retained mainly for session summaries; and unused feature-name constants (`SIDE_FEATURES`, `BOTTOM_FEATURES`) remain in the file.

ii.
```python
SIDE_FEATURES = ["tongue", "left_tongue", "right_tongue", "jaw", "trident", "nose"]
BOTTOM_FEATURES = [
    "top_tongue",
    ...
]
```

```python
return {
    "data": resampled,
    "moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),
    "vidshift": vidshift,
}
```

```python
tongue_speed = compute_feature_speed(...)
paw_speed = compute_feature_speed(...)
...
tongue_binary, tongue_threshold = binarize_session_signal(...)
paw_binary, paw_threshold = binarize_session_signal(...)
motion_binary, motion_threshold = binarize_session_signal(...)
```

iii. No explicit justification is given beyond the notes' emphasis on validation reporting and on matching MATLAB preprocessing before applying decoder-specific discretization.
