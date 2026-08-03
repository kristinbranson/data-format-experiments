# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the Figure 8 context-analysis cohort (defined in `Figure8a_thru_c.m`), not the full 44 sessions from the `load<ANM>_ALMVideo.m` scripts. It reads only from `data/Ephys_Behavior/`, not from `RandomizedDelay_Ephys_Behavior/`. Each session is loaded using `mat73.loadmat()` for v7.3 HDF5 files, and `scipy.io.loadmat()` for motion energy files. The session roster is hard-coded in `SESSION_SPECS`.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

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

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. From the CONVERSION_NOTES.md: "The session roster follows the paper's Figure 8 context-analysis loader in Figure8a_thru_c.m." The trajectory shows the agent identified these 12 sessions as the "two-context ALM ephys cohort" and decided this was the target subset based on the Figure 8 analysis scripts.

## 1-b. How are the data split into subjects?

i. The subject (animal) name is extracted from the `SessionSpec.animal` field. Subjects are accumulated in order of first appearance into a list, and each session gets an index into that list. The AI finds 7 subjects from 12 sessions.

ii.
```python
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
```

iii. From CONVERSION_NOTES.md: "The released Figure 8 session loader names 12 sessions from 7 distinct animal IDs."

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` entry corresponds to one session. Sessions are only loaded from `Ephys_Behavior/`, not from `RandomizedDelay_Ephys_Behavior/`. This yields 12 sessions total, all from the fixed-delay task.

ii.
```python
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    # ... process session ...
    full_data["neural"].append(neural_trials)
```

iii. The agent chose the Figure 8 context-analysis roster because it is "explicit and task-specific." The trajectory shows the agent deliberated on which sessions to include and settled on the 12-session Figure 8 cohort.

## 1-d. How are the data split into trials?

i. Trials are identified by the `Ntrials` field of `bp`. Each trial has corresponding entries in the behavioral fields (`hit`, `miss`, `R`, `autowater`, `early`, `stim.enable`). Spike clusters carry a `trial` field (1-based) identifying which trial each spike belongs to.

ii.
```python
ntrials = int(bp["Ntrials"])
# ...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
```

iii. This follows the standard structure of the MATLAB data objects.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials, no-response trials (`bp.no`), and photostimulation trials (`bp.stim.enable`). Only hit and miss trials are kept. This is implemented in `build_keep_trial_mask()`.

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

iii. From CONVERSION_NOTES.md: "Trial export filtering follows the paper text: exclude early-lick trials, exclude no-response trials, exclude stim.enable trials, keep hit and miss trials from both DR and WC contexts."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu[probe-1]`, specifically the `trial` field (1-based trial assignments), `trialtm` field (spike times relative to trial start), and `quality` field (manual curation label). The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
probe = obj["clu"][spec.probe - 1]
# ...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. This matches the standard spike-sorting structure in the data files.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting `goCue[trial]`. Spikes are binned at 10 ms resolution (`DT = 1/100`) from -3.0 to 2.5 s, giving 550 time bins. Bin counts are converted to firing rates by dividing by `DT`. Rates are then smoothed with a **causal** half-Gaussian kernel of window size 15 with reflect boundary handling, matching `mySmooth.m`. The kernel zeros out its first half to make it causal.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15

bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

The smoothing kernel:
```python
def my_smooth(x, n, bctype="none"):
    kernel = gausswin(n)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```

iii. From CONVERSION_NOTES.md: "Neural alignment matches the shared MATLAB pipeline: align to goCue, bin from -3.0 s to 2.5 s, dt = 0.01 s, causal Gaussian smoothing with smooth = 15 and the same boundary handling as mySmooth.m."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied. First, clusters labeled `garbage`, `gabrga`, `noisy`, or `real?` are excluded via `get_quality_mask()`. Then, a low firing rate filter removes units with mean rate <= 1 Hz. The low-FR filter uses a complex condition-averaged PSTH approach with 7 behavioral conditions (matching the MATLAB code's `lowFR` function).

ii.
```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad

def apply_low_fr_filter(trialdat, bp):
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

iii. From CONVERSION_NOTES.md: "Unit filtering matches the shared code path: start from quality = {'all'}, drop garbage, gabrga, noisy, and real?, drop units with mean firing rate <= 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spikes are aligned to the go cue by subtracting `bp.ev.goCue[trial]` from each spike's `trialtm`. The aligned spike times are then binned into the time window.

ii.
```python
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This follows the standard alignment procedure described in the paper code.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (`DT = 1/100`), spanning -3.0 to 2.5 s from the go cue, giving 550 time bins. No rebinning is applied after initial binning.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
# ...
EDGES, TIME = build_time_axis()
NT = TIME.size  # 550
```

iii. From CONVERSION_NOTES.md: "dt = 0.01 s" and "input[trial] is time_from_go_cue_s, shape (1, 550)."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable defined by the time axis itself. It is the bin centers of the time grid, spanning -3.0 to 2.5 s.

ii.
```python
EDGES, TIME = build_time_axis()
# ...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The input is the same time axis used for all neural and behavioral data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed once as the bin centers of the evenly-spaced edges.

ii.
```python
def build_time_axis():
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. No processing beyond defining the grid.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the same time grid used for binning neural spikes, so alignment is inherent.

ii.
```python
# Same EDGES used for spike binning:
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
# Same TIME used for input:
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI uses `bp.R` directly. `bp.R` is a binary field indicating right-instructed trials. The AI interprets it as the lick direction itself.

ii.
```python
def build_output_constants(bp, keep_trials):
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    # ...
```

iii. From CONVERSION_NOTES.md: "lick_direction: constant over time, left=0, right=1, taken from bp.R."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI reads `bp.R` directly as an integer (0 or 1) and uses it without any derivation from hit/miss. Since `no-response` trials are already excluded by the trial filter, the AI assumes all remaining trials have a lick direction matching the instructed side. There is no "no lick" class.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
# Output values:
["left", "right"],  # only 2 classes
```

iii. The AI excluded no-response trials before computing lick direction, so it treats `bp.R` (instructed side) as the actual lick direction. However, this conflates the instructed side with the actual lick direction on miss trials (where the animal licked the wrong side).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater trials are WC (water-cued) context, non-autowater are DR (delayed-response).

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. From CONVERSION_NOTES.md: "behavioral_context: constant over time, WC=0, DR=1, derived from bp.autowater."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The boolean negation of `autowater` gives DR=1 and WC=0. `~autowater` maps `True` (autowater/WC) to `0` and `False` (non-autowater/DR) to `1`.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The coding convention matches the instructions: WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` directly.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. Since no-response trials are excluded, remaining trials are either hit (correct=1) or miss (incorrect=0).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `bp.hit` is read as an integer. Since no-response/ignore trials are already filtered out, hit=1 maps to correct and miss (hit=0) maps to incorrect. There is no "ignore" class.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
# Output values:
["incorrect", "correct"],  # only 2 classes
```

iii. The AI's approach works because it pre-filters no-response trials, but the reference solution keeps them as a third class.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from DLC tracking in `obj.traj`. The AI uses **all 7 tongue-related features** across both cameras: `tongue`, `left_tongue`, `right_tongue` from the side camera (view 0), and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` from the bottom camera (view 1).

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
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The trajectory shows the agent could not identify a single canonical tongue velocity variable and decided to average all tongue-related features.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) Positions are **interpolated** onto the neural time axis using `interp_with_nan`. (2) NaN values in tongue positions are replaced with 0 using `nan_to_num`. (3) Velocity is computed as `np.gradient` of the interpolated position (first differences). (4) Speed is `sqrt(xvel^2 + yvel^2)`. (5) All 7 tongue feature speeds are averaged with `np.nanmean`. (6) Any remaining NaN is set to 0. There is no likelihood-based filtering or per-run smoothing.

ii.
```python
def find_position(obj, view_index, feat_name, taxis, align_times, vidshift):
    # ... interpolates positions onto neural time axis ...
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
    ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)

def find_velocity(xpos, ypos, feat_name):
    # for tongue features:
    xvel[:, trix] = np.gradient(tsinterp[:, 0])
    yvel[:, trix] = np.gradient(tsinterp[:, 1])
    xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
    yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)

def compute_feature_speed(obj, feature_specs, taxis, align_times, vidshift):
    stacked = np.stack(speeds, axis=0)
    out = np.nanmean(stacked, axis=0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
```

iii. The AI follows the MATLAB `findPosition.m` and `findVelocity.m` approach of interpolating onto the neural time grid and computing first differences. However, it does not apply any likelihood filtering or smoothing before differentiation.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses 2 classes (below/above 50th percentile). For tongue specifically, it ignores zeros when computing the threshold, because the MATLAB code uses 0 as a placeholder for invisible tongue. The threshold is computed from strictly positive tongue-speed values within the kept trials.

ii.
```python
tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed, keep_trials, ignore_zeros_for_threshold=True,
)

def binarize_session_signal(signal, keep_trials, ignore_zeros_for_threshold=False):
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
    return binary, threshold
```

iii. From CONVERSION_NOTES.md: "because the shared MATLAB code uses 0 as a placeholder when the tongue is not visible, I estimated the 50th-percentile threshold from strictly positive tongue-speed samples."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are directly interpolated onto the neural time axis after video-offset correction. Frame times are corrected by `vidshift = findVideoOffset()` and the trial's go cue time. Then `interp_with_nan` maps positions from frame times to the neural time grid.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. From CONVERSION_NOTES.md: "Interpolated DLC traces and motion energy onto the neural time grid after video-offset correction."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from DLC tracking in `obj.traj`, using **both** `top_paw` and `bottom_paw` from the bottom camera (view 1).

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The AI uses both paw features, averaging their speeds.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) Positions are interpolated onto the neural time axis. (2) Missing values are filled with nearest-neighbor interpolation (`fill_nearest_1d`). (3) Velocity is computed as `np.gradient` of position minus a baseline derivative (median of differences). (4) The y-velocity subtracts the x baseline derivative (matching MATLAB code's baseline subtraction). (5) Speed is `sqrt(xvel^2 + yvel^2)`. (6) The two paw feature speeds are averaged.

ii.
```python
# For non-tongue features, fill missing values:
xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)

# Baseline subtraction for non-tongue:
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trix] = xvel[:, trix] - basederiv[0]
yvel[:, trix] = yvel[:, trix] - basederiv[0]  # both subtract x baseline
```

iii. From CONVERSION_NOTES.md: "preserve the MATLAB baseline-subtraction behavior for non-tongue velocities."

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is binarized at the session 50th percentile over kept trials/timepoints, with 2 classes (0=below, 1=at or above).

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

iii. Straightforward 50th percentile split as specified in the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: positions are directly interpolated onto the neural time axis after video-offset correction.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. Same interpolation approach as all kinematic features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from a companion file `motionEnergy_<animal>_<date>.mat`. The `me.data` field is extracted, handling nested wrapping.

ii.
```python
def load_motion_energy(obj, motion_energy_path, taxis, align_times):
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
    raw_data = me.data
    if hasattr(raw_data, "data"):
        raw_data = raw_data.data
```

iii. From CONVERSION_NOTES.md: "Motion energy loading matches loadMotionEnergy.m."

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy traces are interpolated onto the neural time axis using `interp_with_nan`, then missing values are filled with nearest-neighbor interpolation (`fill_nearest_1d`).

ii.
```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. From CONVERSION_NOTES.md: "interpolate onto the neural time axis, fill missing values with nearest samples."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is binarized at the session 50th percentile over kept trials/timepoints, with 2 classes.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. Straightforward 50th percentile split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the neural time axis using side camera frame times after video-offset correction, same as other camera-derived signals.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. Same alignment approach as kinematic features.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) For missing frame times (NaN or empty), a fallback time axis is synthesized assuming 400 Hz frame rate. (2) For tongue features, NaN velocities are replaced with 0. (3) For non-tongue features, nearest-neighbor interpolation fills gaps. (4) For tongue thresholding, zeros are excluded from the percentile computation. (5) Dropped frames are detected via `NdroppedFrames`.

ii.
```python
# Fallback for missing frame times:
use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]

# Check dropped frames:
dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
if dropped.size and np.isnan(dropped[0]):
    continue
```

iii. The AI fills gaps rather than marking them as a separate class, unlike the reference which uses a "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files with `mat73.loadmat()` dominates. The kinematic interpolation and spike binning are also non-trivial but secondary. The agent noted in the trajectory that the "full conversion path is expensive."

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. The trajectory mentions the converter "spending most of its time in session-by-session spike binning and video interpolation."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop over units uses `np.add.at` per unit per trial, which is slower than a single `histogram2d` call. The position interpolation and velocity computation loop over trials and features individually. The smoothing convolves column-by-column.

ii.
```python
# Per-unit spike binning loop:
for unit_pos, unit_idx in enumerate(keep_units):
    # ... per-trial add.at ...
    np.add.at(counts, (spike_trials, bins), 1.0)

# Per-column smoothing:
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```

iii. The reference solution bins all trials at once with `histogram2d`, which is more efficient.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session (in `find_video_offset`), but the `load_motion_energy` function computes it independently again via its own call to `find_video_offset`. The feature names are extracted repeatedly for each trial in `find_dlc_feat_index`. Position finding and velocity computation are called separately for each feature in `TONGUE_FEATURES` and `PAW_FEATURES`.

ii.
```python
# Video offset computed in main loop:
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
# But also inside load_motion_energy:
vidshift = find_video_offset(obj)

# And separately:
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The redundant video offset computation is minor but unnecessary.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `build_low_fr_conditions` function computes 7 different behavioral condition masks and averages PSTHs over all of them just to get a mean firing rate for filtering. The `build_output_constants` function computes `n_left_kept`, `n_right_kept`, etc. counts that are only used for summary statistics. The `fill_nearest_1d` function is applied to all non-tongue kinematics, filling gaps that would otherwise provide useful "not visible" information.
