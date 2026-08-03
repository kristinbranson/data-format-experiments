# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only a subset of 12 sessions from the "two-context ALM" Figure 8 analysis, not all 44 sessions available. Session specs are hard-coded in `CONTEXT_SESSION_SPECS`, listing only sessions from `Ephys_Behavior/` (fixed-delay with context variation). All 12 sessions are loaded with `mat73.loadmat()` for the main data file and `scipy.io.loadmat()` for motion energy files.

ii.
```python
CONTEXT_SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 1),
    SessionSpec("JEB7", "2021-04-29", 0),
    SessionSpec("JEB7", "2021-04-30", 0),
    SessionSpec("EKH1", "2021-08-07", 1),
    SessionSpec("EKH3", "2021-08-11", 1),
    SessionSpec("JGR2", "2021-11-16", 0),
    SessionSpec("JGR2", "2021-11-17", 0),
    SessionSpec("JGR3", "2021-11-18", 0),
    SessionSpec("JEB19", "2023-04-21", 0),
    SessionSpec("JEB19", "2023-04-20", 0),
    SessionSpec("JEB19", "2023-04-19", 0),
    SessionSpec("JEB19", "2023-04-18", 0),
]

obj = mat73.loadmat(spec.data_path)["obj"]
me = load_motion_energy(spec)
```

iii. The AI justified using only 12 sessions because the decoder outputs include behavioral context (WC vs DR), and only the two-context sessions have context variation. The AI explicitly followed the Figure 8 session loader list from the reference code. The AI documented this in CONVERSION_NOTES.md Step 4-5.

## 1-b. How are the data split into subjects?

i. The subject (animal) name is taken from `SessionSpec.animal`, which is the prefix of the session name (e.g., "JEB6"). At assembly, subjects are sorted unique names with indices mapping each session. This yields 7 subjects across the 12 sessions.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int

subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. Subject splitting is straightforward from the session metadata. The AI noted a discrepancy with the paper (which says 6 mice) but retained 7 because the code and data both show 7 animal IDs.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` entry becomes one session. Only 12 sessions from the Figure 8 two-context analysis are included. All are from `Ephys_Behavior/` (fixed-delay). Randomized-delay sessions (19 additional sessions in the reference) are excluded entirely. Each session uses only one probe (specified by `probe_index`), unlike the reference which concatenates units from both probes for dual-probe sessions.

ii.
```python
session_specs = CONTEXT_SESSION_SPECS[:2] if use_sample else CONTEXT_SESSION_SPECS

@property
def data_path(self) -> Path:
    return Path("data/Ephys_Behavior") / f"data_structure_{self.session_id}.mat"
```

iii. The AI chose this subset because the context decoder output requires WC/DR variation, which exists only in the two-context sessions.

## 1-d. How are the data split into trials?

i. Trials are indexed by the behavior protocol fields in `obj.bp`. Each trial has associated behavioral labels (`hit`, `miss`, `early`, `no`, `stim.enable`, `autowater`). The AI uses 0-based trial indices, filtering to keep only valid trials.

ii.
```python
def select_valid_trials(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    early = ensure_1d_numeric(bp["early"]) != 0
    no = ensure_1d_numeric(bp["no"]) != 0
    hit = ensure_1d_numeric(bp["hit"]) != 0
    miss = ensure_1d_numeric(bp["miss"]) != 0
    stim_enable = ensure_1d_numeric(bp["stim"]["enable"]) != 0
    valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
    return np.flatnonzero(valid)
```

iii. The AI splits trials using the behavioral protocol table, same as the reference.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies three filters: (1) exclude early-lick trials (`bp.early`), (2) exclude photostimulation trials (`bp.stim.enable`), and (3) exclude ignore/no-response trials (`bp.no`). Only hit and miss trials are kept. Additionally, trials where a valid first lick direction cannot be determined are excluded. This yields 2,415 trials across 12 sessions.

ii.
```python
valid = (~early) & (~no) & (~stim_enable) & (hit | miss)
...
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
    kept_trials.append(int(trial_idx))
```

iii. The AI justified excluding ignore trials because the outcome output requires a well-defined hit/miss label, and lick direction requires an actual lick event. The reference, by contrast, keeps ignore trials and assigns them "no lick" and "ignore" classes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu[probe_index]`, specifically `clu.trialtm` (spike times relative to trial start), `clu.trial` (trial number, 1-based), and `clu.quality` (manual curation label). The go cue times `bp.ev.goCue` are used for alignment. Only one probe per session is used (specified by `probe_index`).

ii.
```python
clu = obj["clu"][spec.probe_index]
...
trialtm = ensure_1d_numeric(clu["trialtm"][unit_idx])
trial_numbers_1based = np.asarray(clu["trial"][unit_idx], dtype=np.int64)
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
```

iii. This matches the reference variables, except the AI uses only a single probe while the reference concatenates units from both probes for dual-probe sessions.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5 ms bins spanning -2.5 to +2.5 s (1000 bins), divided by `dt` to get firing rates in Hz, then smoothed with a causal half-Gaussian kernel of length 15 samples. The smoothing uses `my_smooth` which zeroes the first half of a `gausswin(15)` kernel and applies it via convolution with reflect boundary conditions.

ii.
```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
aligned_counts = np.zeros((kept_trials_0based.size, TIME_AXIS.size), dtype=np.float64)
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
```

The smoothing kernel:
```python
def my_smooth(x, n, bctype="none"):
    kern = gaussian_window(n)
    kern[: len(kern) // 2] = 0.0
    kern /= np.sum(kern)
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

iii. The AI aimed to replicate the reference's `mySmooth` function, which uses a causal Gaussian (half the kernel zeroed). The reference solution instead uses `scipy.ndimage.gaussian_filter1d` with a symmetric Gaussian of sigma=2.8 bins (14 ms) and 'reflect' mode. Both approaches attempt to follow the reference code's smoothing, but implement it differently.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters are applied: (1) quality label exclusion matching `{garbage, gabrga, noisy, real?}` using case-sensitive exact string matching (after stripping whitespace), and (2) units with mean firing rate <= 1 Hz across all trials in the window are excluded. This yields 519 units across 12 sessions.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

def keep_quality(quality: str) -> bool:
    if quality is None:
        quality = ""
    quality = str(quality).strip()
    return quality not in QUALITY_EXCLUDE

def mean_firing_rate_window(...) -> float:
    aligned = trialtm - align_times[trial_numbers_1based - 1]
    in_window = (aligned >= TMIN) & (aligned < TMAX)
    return float(np.sum(in_window) / (align_times.size * (TMAX - TMIN)))
```

iii. The AI followed `findClusters.m` for quality labels but does not include "poor" in the exclusion set (the reference does). The AI uses case-sensitive matching while the reference uses case-insensitive matching. The 1 Hz threshold matches both the paper and reference. The mean firing rate is computed over all trials (not just kept trials), which differs from the reference which computes it over kept trials only.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting `bp.ev.goCue[trial]` from each spike's `trialtm`. Spikes are then binned into the -2.5 to +2.5 s window.

ii.
```python
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. This matches the reference approach of `trialtm - goCue[trial]`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5 ms bins from -2.5 to +2.5 s (1000 bins), matching the reference's `params.dt = 1/200`. No rebinning is applied.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. Matches the reference parameters.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is not derived from raw data variables. It is the time axis itself, defined as the centers of the 1000 bins spanning -2.5 to +2.5 s.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
...
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The time axis is a synthetic variable defined by the binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing; the time axis is directly computed from the bin parameters.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same binning grid as the neural data, so they are inherently aligned.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

iii. Same grid ensures alignment.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw lick event times `bp.ev.lickL` and `bp.ev.lickR`, which record the times of left and right lick port contacts. The go cue time is also used to select only post-go-cue licks.

ii.
```python
def get_first_lick_direction(obj, trial_idx, go_time):
    lick_l = event_list_to_array(obj["bp"]["ev"]["lickL"][trial_idx])
    lick_r = event_list_to_array(obj["bp"]["ev"]["lickR"][trial_idx])
    lick_l = lick_l[lick_l >= go_time]
    lick_r = lick_r[lick_r >= go_time]
    first_l = lick_l[0] if lick_l.size else math.inf
    first_r = lick_r[0] if lick_r.size else math.inf
    if math.isinf(first_l) and math.isinf(first_r):
        return None
    return 0 if first_l < first_r else 1
```

iii. The AI chose to use actual lick events rather than deriving direction from instructed side + outcome (hit/miss). The reference derives lick direction from instructed side (`bp.R`) and outcome (`bp.hit`, `bp.miss`).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The first lick event after the go cue is found from both `lickL` and `lickR` event arrays. The side with the earlier first lick determines the direction: left=0, right=1. Trials with no post-go-cue lick are excluded entirely. Only two classes exist (no "no lick" class).

ii.
```python
return 0 if first_l < first_r else 1
...
output_trial = np.vstack([
    np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
    ...
])
```

iii. The AI used actual behavioral lick events rather than inferring from instructed side and outcome. The reference uses instructed side + hit/miss logic and includes a "no lick" third class for ignore trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`, which flags water-cued (WC) trials.

ii.
```python
autowater = ensure_1d_numeric(bp["autowater"])
...
0 if autowater[trial_idx] != 0 else 1,
```

iii. Same source variable as the reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater trials become WC (0), non-autowater become DR (1).

ii.
```python
0 if autowater[trial_idx] != 0 else 1,
```

iii. Matches the reference coding (WC=0, DR=1).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`. Since ignore trials are excluded, `hit` alone determines outcome (hit=correct=1, miss=incorrect=0).

ii.
```python
hit = ensure_1d_numeric(bp["hit"])
...
1 if hit[trial_idx] != 0 else 0,
```

iii. Because the AI excludes ignore/no-response trials, only two outcome classes are needed.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Binary mapping: hit trials get outcome=1 (correct), miss trials get outcome=0 (incorrect). No "ignore" class exists because those trials are filtered out. The instructions specify "incorrect = 0, correct = 1" which this matches, though it only has 2 classes vs the reference's 3.

ii.
```python
1 if hit[trial_idx] != 0 else 0,
```

iii. The AI justified this by excluding ignore trials in the trial filter step.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from the side camera (view index 0) DLC tracking for the "tongue" feature. Uses `obj.traj[0]` with fields `ts` (x, y, likelihood), `frameTimes`, and `featNames`. Also uses `bp.ev.goCue` and `sglx` bitcode fields for video clock correction.

ii.
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The AI uses only the side camera for tongue, while the reference uses both side and bottom cameras and averages them after normalizing by each camera's 90th percentile.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The processing pipeline: (1) Extract x, y positions from DLC tracking at frame resolution. (2) Interpolate positions onto the 5 ms time axis using `np.interp`, with NaN for out-of-range times. (3) For tongue features, NaN values in velocity are replaced with 0 (not filled with nearest neighbor). (4) Compute velocity as `np.gradient` of interpolated positions. (5) Compute speed as `sqrt(vx^2 + vy^2)`.

ii.
```python
# Interpolation onto time axis
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)

# Velocity computation for tongue (no baseline subtraction)
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
xv = np.where(np.isfinite(xv), xv, 0.0)
yv = np.where(np.isfinite(yv), yv, 0.0)

tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The AI's approach differs significantly from the reference: (1) interpolates positions to the time axis first, then differentiates, rather than computing velocity at frame resolution and binning; (2) no smoothing of positions before differentiation (smoothing is only applied for non-tongue features with n=1 which is a no-op); (3) no per-run velocity computation across contiguous visible frames; (4) only uses one camera view instead of two.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Two categories: 0 (below session 50th percentile) and 1 (at or above). The threshold is computed from tongue-visible timepoints only (where tongue tracking is finite). When the tongue is not visible, the bin is assigned 0 (below threshold).

ii.
```python
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0

tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The AI computes the median only from visible frames, assigning invisible frames to 0. The reference uses `np.nanpercentile` over all values (including NaN-excluded), and assigns invisible bins to a third "not visible" class (value 2). The AI only has 2 output classes instead of 3.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset (computed from bitcode pulse comparison between recording and behavior clocks), then the go cue time is subtracted. Positions are interpolated onto the 5 ms neural time axis using `np.interp`.

ii.
```python
def compute_vidshift(obj):
    bit_start = mode_value(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_value(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)

shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. The video offset computation matches the reference's `findVideoOffset.m`. The difference is that the AI interpolates positions to the time axis, while the reference bins frame-level velocities into the time axis.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom camera (view index 1) DLC tracking for both "top_paw" and "bottom_paw" features. Uses `obj.traj[1]`.

ii.
```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
    paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
```

iii. The AI averages both paws, while the reference uses only `top_paw` because `bottom_paw` tracking is unreliable during the delay period.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The processing: (1) Extract x, y positions for both top_paw and bottom_paw from bottom camera DLC tracking. (2) Apply causal smoothing (`my_smooth` with n=1, which is a no-op). (3) Interpolate positions onto the 5 ms time axis. (4) Fill NaN values with nearest valid value (for non-tongue features). (5) Compute velocity as `np.gradient`, subtracting the baseline median drift. (6) Fill remaining NaN velocities with nearest value. (7) Compute speed as `sqrt(vx^2 + vy^2)`. (8) Average speeds from both paws.

ii.
```python
# For non-tongue features: smooth (no-op with n=1), interpolate, fill nearest
if "tongue" not in feat_name:
    xy = my_smooth(xy, 1, "reflect")  # n=1, no-op
...
xpos[:, trial_idx] = np.interp(...)
if "tongue" not in feat_name:
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])

# Velocity with baseline subtraction
basederiv = np.nanmedian(diffs, axis=0)
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
xv = xv - basederiv[0]
yv = yv - basederiv[0]  # Bug: should be basederiv[1] for y

# Average both paws
paw_speed = np.nansum(paw_stack, axis=0) / paw_counts
```

iii. Key differences from the reference: uses both paws instead of only top_paw; interpolates positions to time axis then differentiates rather than computing velocity at frame resolution then binning; subtracts baseline drift; fills missing values with nearest neighbor instead of using a "not visible" class. Also note: the y-velocity baseline subtraction uses `basederiv[0]` (x-component) instead of `basederiv[1]` (y-component), which is a bug.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Two categories at the session 50th percentile. All values including NaN-filled ones contribute to the threshold computation.

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
...
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64),
```

iii. Two classes only (0 and 1), no "not visible" class. The reference uses three classes with a "not visible" category for untracked bins.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, go cue subtraction, interpolation onto the 5 ms time axis.

ii. Same `align_feature_positions` function as tongue, using bottom camera (view index 1).

iii. Same alignment approach as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<anm>_<date>.mat` files. The file contains `me.data` (per-trial traces) and `me.moveThresh`.

ii.
```python
def load_motion_energy(spec):
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}
```

iii. Same source files as the reference, though the AI's loading code is less robust to the varying file layouts (it expects `me.data` directly rather than handling nested wrapping).

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy is interpolated onto the 5 ms time axis using `np.interp`, then NaN values are filled with nearest-neighbor interpolation (`fill_nearest_1d`). The result is then discretized at the session median.

ii.
```python
def align_motion_energy(obj, me, align_times, time_axis):
    ...
    shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
    aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
    aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
    return aligned
```

iii. The reference bins frame-level motion energy into the time bins (averaging frames within each bin) rather than interpolating. The AI also fills NaN with nearest-neighbor, while the reference keeps NaN and uses a "not visible" class.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Two categories at the session 50th percentile. After nearest-neighbor filling, no bins should be NaN.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
...
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64),
```

iii. Two classes only (0 and 1), no "not visible" class since NaN values are filled. The reference uses three classes.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times are corrected by video offset and go cue, then motion energy values are interpolated onto the neural time axis. The motion energy uses the side camera's frame times.

ii.
```python
frame_times = view_dict["frameTimes"][trial_idx]
...
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. Same video offset approach as other camera streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) Missing/all-NaN frame times fall back to a nominal 400 Hz grid. (2) For tongue, NaN velocities are replaced with 0. (3) For non-tongue features, NaN positions and velocities are filled with nearest-neighbor interpolation. (4) Motion energy NaN values are filled with nearest-neighbor interpolation. (5) Trials with no valid lick direction are excluded.

ii.
```python
# Fallback for missing frame times
if frame_times.size == 0 or np.all(np.isnan(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Nearest-neighbor fill for non-tongue features
if "tongue" not in feat_name:
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])

# Zero-fill for tongue velocity NaN
xv = np.where(np.isfinite(xv), xv, 0.0)
```

iii. The reference uses a "not visible" third class for missing data rather than filling it in. The reference does not interpolate or fill missing values. The AI's nearest-neighbor filling may introduce artifacts.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB files with `mat73.loadmat()` is the most time-consuming step. The AI estimated ~7.4 s/session, with total conversion of ~1.5-2 minutes for 12 sessions. Kinematic alignment (interpolation per feature per trial) is also relatively expensive.

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. File I/O dominates, consistent with the reference.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several per-trial loops could potentially be vectorized: (1) the trial loop in `align_feature_positions` for interpolating positions, (2) the trial loop in `align_motion_energy`, (3) the per-unit loop for spike binning. The spike binning uses `np.add.at` which is semi-vectorized.

ii.
```python
for trial_idx in range(n_trials):
    ...
    xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], ...)
```

iii. Variable frame counts per trial make full vectorization difficult.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`compute_vidshift`) is computed twice for one session: once inside `align_feature_positions` (passed as parameter) and once at the session level for motion energy alignment. The `normalize_trial_struct` function is called repeatedly for each trial of each feature.

ii.
```python
vidshift = compute_vidshift(obj)
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
...
# Also computed again inside align_motion_energy:
def align_motion_energy(obj, me, align_times, time_axis):
    vidshift = compute_vidshift(obj)
```

iii. The video offset is a session constant that should only be computed once.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `moveThresh` value is loaded from motion energy files but never used (the code uses the session median instead). (2) The `my_smooth(xy, 1, "reflect")` call for non-tongue features is a no-op since n=1. (3) Full MATLAB session objects are loaded into memory including fields never used. (4) The `NdroppedFrames` field is checked but doesn't affect processing.

ii.
```python
return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}  # moveThresh unused

xy = my_smooth(xy, 1, "reflect")  # n=1, no-op

dropped = trial.get("NdroppedFrames", np.nan)  # checked but no functional impact
```

iii. These are minor inefficiencies that don't affect correctness.
