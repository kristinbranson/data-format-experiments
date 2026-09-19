# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only the 12 two-context ALM sessions specified in the Figure 8 analysis scripts from the reference code. These are hard-coded in `CONTEXT_SESSION_SPECS`. Each session's `.mat` file is loaded using `mat73.loadmat()` for the data structure (v7.3 HDF5 format) and `scipy.io.loadmat()` for the motion energy files (v5 format). The AI only looks in `data/Ephys_Behavior/` and does not include `RandomizedDelay_Ephys_Behavior/` sessions.

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

iii. The AI chose the 12-session two-context subset because the decoder outputs include behavioral context (WC vs DR), which only varies in the two-context sessions. The AI documented this as "the most paper-consistent primary dataset" for context analyses, referencing the Figure 8 loader scripts.

## 1-b. How are the data split into subjects?

i. Subject identity is taken from the `animal` field of the `SessionSpec` dataclass, which is hard-coded for each session. The subjects list is the sorted set of unique animal names, and `subject_idx` maps each session to its subject.

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

iii. The AI identified 7 unique subjects from the 12 sessions, noting a discrepancy with the paper's claim of 6 mice but resolving in favor of the code/data evidence.

## 1-c. How are the data split into sessions?

i. Each session is one entry in `CONTEXT_SESSION_SPECS`, corresponding to one `.mat` file. The AI processes 12 sessions total, all from `data/Ephys_Behavior/`. The `SessionSpec` dataclass also specifies which probe (0-based index) to use for each session. Each session becomes one element in the `neural`, `input`, and `output` lists.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe_index: int  # 0-based

    @property
    def data_path(self) -> Path:
        return Path("data/Ephys_Behavior") / f"data_structure_{self.session_id}.mat"
```

iii. The AI followed the Figure 8 loader scripts from the reference code to determine which sessions and probes to include.

## 1-d. How are the data split into trials?

i. Trials are the rows of `obj.bp`, with `Ntrials` giving the count. For each trial, the AI reads behavioral fields and event arrays indexed by trial number. Trial indices are 0-based in the AI's code.

ii.
```python
n_trials = int(obj["bp"]["Ntrials"])
valid_trials = select_valid_trials(obj)
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
```

iii. The AI uses `bp.Ntrials` to determine trial count and indexes per-trial fields directly.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies multiple filters: (1) early-lick trials excluded (`bp.early != 0`), (2) ignore/no-response trials excluded (`bp.no != 0`), (3) photostimulation trials excluded (`bp.stim.enable != 0`), (4) only hit or miss trials kept (`hit | miss`), and (5) trials without a detectable post-go-cue lick are also dropped (when `get_first_lick_direction` returns `None`).

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

# Then additionally:
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
```

iii. The AI justified excluding ignore trials because it wanted well-defined outcome labels (only correct/incorrect). The AI also required a detectable lick for lick direction.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu[probe_index]`, specifically the `trialtm` (spike times relative to trial start), `trial` (trial number, 1-based), and `quality` fields for each unit. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
clu = obj["clu"][spec.probe_index]
# For each unit:
trialtm = clu["trialtm"][unit_idx]
trial_numbers = clu["trial"][unit_idx]
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
```

iii. The AI correctly identified the spike cluster data as the source of neural activity.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtracting `align_times[trial-1]` from `trialtm`. Aligned spikes are binned into 5ms bins from -2.5 to +2.5s using `np.floor((aligned - TMIN) / DT)` and `np.add.at`. Counts are converted to firing rates by dividing by `DT`. Rates are smoothed using a causal Gaussian kernel: `my_smooth(rates.T, SMOOTH_N, BCTYPE).T` with `SMOOTH_N=15` and `BCTYPE='reflect'`. The smoothing uses a half-Gaussian (the left half of the kernel is zeroed out).

ii.
```python
bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
aligned_counts = np.zeros((kept_trials_0based.size, TIME_AXIS.size), dtype=np.float64)
np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
rates = aligned_counts / DT
rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T

def my_smooth(x, n, bctype):
    kern = gaussian_window(n)
    kern[: len(kern) // 2] = 0.0  # causal: zero out left half
    kern /= np.sum(kern)
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

iii. The AI implemented a causal Gaussian smoothing to match the reference code's `mySmooth.m` function, which uses a half-Gaussian kernel with `bctype='reflect'`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) quality label filter excluding units with labels in `{'garbage', 'gabrga', 'noisy', 'real?'}`, and (2) mean firing rate filter removing units with mean FR <= 1 Hz. The quality comparison is case-sensitive (exact string match, not lower-cased).

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

def keep_quality(quality: str) -> bool:
    if quality is None:
        quality = ""
    quality = str(quality).strip()
    return quality not in QUALITY_EXCLUDE

# Firing rate filter:
mean_fr = mean_firing_rate_window(...)
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The AI followed `findClusters.m` from the reference code for quality exclusions and used the paper's 1 Hz threshold for firing rate filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times in `trialtm` are aligned to the go cue by subtracting `align_times[trial-1]` (where `align_times = bp.ev.goCue`). This puts all spikes in seconds relative to go cue onset.

ii.
```python
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])  # ALIGN_EVENT = "goCue"
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. The AI correctly aligns to `bp.ev.goCue` as specified in the instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5ms (`DT = 0.005`), spanning -2.5 to +2.5s from the go cue, yielding 1000 bins. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
```

iii. The AI matched the reference code's `params.dt = 1/200` and `params.tmin`/`params.tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself, which is defined by the binning parameters. It is not derived from any raw data variable.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The time axis is computed from the bin parameters and represents bin centers.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing from raw data. The time axis is computed as the bin centers of the 5ms bins spanning -2.5 to +2.5s.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The time axis is the bin center grid itself, so it is inherently aligned with the neural data which is binned onto the same grid.

ii.
```python
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The input shares the same time grid as the neural data by construction.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the raw lick event times `bp.ev.lickL` and `bp.ev.lickR`, finding the first lick after the go cue on each trial.

ii.
```python
def get_first_lick_direction(obj: dict, trial_idx: int, go_time: float) -> int | None:
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

iii. The AI chose to use actual behavioral lick direction from lick port events rather than the instructed side, reasoning that the user explicitly requested "lick direction" which should reflect actual behavior.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI finds the first left and first right lick event after the go cue time. Whichever occurs first determines the lick direction: left (0) or right (1). Trials with no detectable post-go-cue lick are dropped entirely. There is no "no lick" class.

ii.
```python
return 0 if first_l < first_r else 1
# Trials with no lick:
if math.isinf(first_l) and math.isinf(first_r):
    return None  # trial is dropped
```

iii. The AI has only 2 classes (left=0, right=1), with no "no lick" class. Trials without licks are excluded.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`.

ii.
```python
autowater = ensure_1d_numeric(bp["autowater"])
context_label = 0 if autowater[trial_idx] != 0 else 1
```

iii. The AI correctly identified `autowater` as the context indicator.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct mapping: autowater trials become WC (0), non-autowater trials become DR (1).

ii.
```python
context_label = 0 if autowater[trial_idx] != 0 else 1
# In output_values:
["WC", "DR"]
```

iii. Matches the convention specified in the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit`. Since ignore trials are already excluded, the remaining trials are either hit or miss.

ii.
```python
hit = ensure_1d_numeric(bp["hit"])
outcome_label = 1 if hit[trial_idx] != 0 else 0
```

iii. With ignore trials excluded, hit=correct and not-hit=incorrect.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. A binary mapping: hit trials become correct (1), non-hit trials (which must be miss, since ignore is excluded) become incorrect (0). There is no "ignore" class.

ii.
```python
outcome_label = 1 if hit[trial_idx] != 0 else 0
# In output_values:
["incorrect", "correct"]
```

iii. The AI excluded ignore trials upstream, so outcome is binary. The instructions specify three classes (incorrect, correct, ignore), but the AI only has two.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI derives tongue velocity from the DLC tracking in `obj.traj[0]` (side camera view only), using the `tongue` feature. It reads `ts` (tracked positions), `frameTimes`, and `featNames`.

ii.
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
```

iii. The AI used only the side camera view (view index 0) for tongue tracking.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI: (1) extracts x,y positions from the DLC tracking for the `tongue` feature on the side camera, (2) interpolates them onto the 5ms time grid using `np.interp` (NaN outside the data range), (3) computes velocity using `np.gradient` on the interpolated positions, (4) replaces NaN velocities with 0.0, (5) computes speed as `sqrt(vx^2 + vy^2)`.

ii.
```python
# Interpolation to time grid:
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
ypos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 1], left=np.nan, right=np.nan)

# Velocity (tongue-specific: no baseline subtraction, NaN->0):
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
xv = np.where(np.isfinite(xv), xv, 0.0)
yv = np.where(np.isfinite(yv), yv, 0.0)

tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. Unlike the reference which computes velocity at frame resolution within contiguous tracked runs, the AI interpolates positions to the time grid first and then differentiates. NaN velocities are replaced with 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI computes the 50th percentile of tongue speed across all kept trials, but only from tongue-visible timepoints. The threshold splits into two classes: 0 (below threshold and not visible) and 1 (above threshold and visible).

ii.
```python
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0

tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The AI uses only 2 classes (0 and 1), without a separate "not visible" class (class 2). Invisible tongue timepoints are assigned to class 0.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected for the video offset (computed from bitcode alignment) and the trial's go cue time: `shifted_time = frameTimes - vidshift - align_times[trial_idx]`. The positions are then interpolated onto the same 5ms time grid as the neural data.

ii.
```python
def compute_vidshift(obj: dict) -> float:
    bit_start = mode_value(obj["bp"]["ev"]["bitStart"])
    vid_file_offset = mode_value(obj["sglx"]["bitcode"]["bitstart"]) / float(obj["sglx"]["fs"])
    return float(vid_file_offset - bit_start)

shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. The video offset computation matches the reference's `findVideoOffset.m`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI derives paw velocity from DLC tracking of both `top_paw` and `bottom_paw` features from the bottom camera view (`obj.traj[1]`).

ii.
```python
paw_speeds = []
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
    paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
```

iii. The AI chose to use both paw features and average them, treating them as two views of paw movement.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature (`top_paw`, `bottom_paw`): (1) x,y positions are extracted and smoothed with `my_smooth(xy, 1, "reflect")` (which is a no-op since n=1), (2) interpolated onto the 5ms time grid with `np.interp`, (3) missing values filled with `fill_nearest_1d` (nearest-neighbor interpolation), (4) velocity computed with `np.gradient` with baseline subtraction (`nanmedian` of diffs), (5) speed as magnitude. The two paw speeds are then averaged (where available).

ii.
```python
# For paw features (not tongue):
if "tongue" not in feat_name:
    xy = my_smooth(xy, 1, "reflect")  # no-op
    xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
    ypos[:, trial_idx] = fill_nearest_1d(ypos[:, trial_idx])

# Velocity with baseline subtraction:
basederiv = np.nanmedian(diffs, axis=0)
xv = xv - basederiv[0]
yv = yv - basederiv[0]  # BUG: should be basederiv[1]

# Averaging two paws:
paw_stack = np.stack(paw_speeds, axis=0)
paw_speed = np.nansum(paw_stack, axis=0) / paw_counts
```

iii. The AI applies nearest-neighbor fill and baseline subtraction for paw features but not for tongue, following different handling in the reference code.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile of paw speed across all kept trials/timepoints is computed. Values above the threshold are class 1, below are class 0. There is no "not visible" class.

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64)
```

iii. The AI uses only 2 classes (0 and 1) for paw velocity, without a "not visible" class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: frame times corrected by video offset and go cue, then positions interpolated onto the 5ms time grid. Missing values are filled with nearest-neighbor interpolation.

ii.
```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
```

iii. Same alignment method as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the companion `motionEnergy_<animal>_<date>.mat` file. The AI reads `me.data` and `me.moveThresh`.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {
        "data": np.atleast_1d(me.data),
        "moveThresh": float(me.moveThresh),
    }
```

iii. The AI correctly identified the separate motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated onto the 5ms time grid using `np.interp`, then missing values are filled with `fill_nearest_1d` (nearest-neighbor interpolation).

ii.
```python
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. The AI applies nearest-neighbor fill to motion energy, which means bins outside the video data range get filled rather than remaining NaN.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile of motion energy across all kept trials/timepoints is computed. Values above the threshold are class 1, below are class 0. There is no "no video" class.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64)
```

iii. The AI uses only 2 classes (0 and 1), without a "no video" class as specified in the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Frame times from the side camera are corrected by the video offset and go cue time, then motion energy values are interpolated onto the 5ms time grid.

ii.
```python
frame_times = view_dict["frameTimes"][trial_idx]
shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
```

iii. Same video offset and alignment method as other camera-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing data by: (1) falling back to a nominal 400 Hz frame grid when `frameTimes` is missing or all-NaN, (2) using `fill_nearest_1d` to interpolate missing values for paw positions and motion energy, (3) replacing NaN tongue velocities with 0.0, (4) skipping trials where DLC `ts` data is missing entirely. The AI does not drop trials with missing video data but instead fills in estimated values.

ii.
```python
# Missing frameTimes fallback:
if frame_times.size == 0 or np.all(np.isnan(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Nearest-neighbor fill:
def fill_nearest_1d(x: np.ndarray) -> np.ndarray:
    good = np.isfinite(x)
    x[~good] = np.interp(idx[~good], idx[good], x[good])
    return x

# Tongue NaN -> 0:
xv = np.where(np.isfinite(xv), xv, 0.0)
```

iii. The AI chose to fill missing data rather than mark it as a separate class, which differs from the instructions that specify "not visible" (class 2) for tongue/paw and "no video" (class 2) for motion energy.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the MATLAB session files with `mat73.loadmat()` is the most time-consuming step, as noted in the conversion notes (~7.4s per session). The kinematic alignment with per-trial interpolation loops is also significant.

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. The AI estimated total conversion time at ~1.5-2 minutes for 12 sessions.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `align_feature_positions`, `align_motion_energy`, and `compute_velocity` could potentially be vectorized but are kept as loops because each trial has a variable number of frames. The spike binning is already vectorized with `np.add.at`.

ii.
```python
# Per-trial loop in align_feature_positions:
for trial_idx in range(n_trials):
    # ... per-trial interpolation
    xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], ...)

# Per-trial loop in compute_velocity:
for trial_idx in range(xpos.shape[1]):
    xv = np.gradient(tsinterp[:, 0])
```

iii. Variable frame counts per trial prevent straightforward vectorization.

## 11-c. What processing does the code repeat multiple times?

i. The video offset (`compute_vidshift`) is called once per session but the trajectory struct normalization (`normalize_trial_struct`) is called for every trial for every feature separately. The DLC data for each trial is accessed multiple times across different feature extractions (tongue, top_paw, bottom_paw).

ii.
```python
# vidshift computed once, but traj accessed per trial per feature:
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", ...)
paw_x, paw_y = align_feature_positions(obj, 1, "top_paw", ...)
paw_x, paw_y = align_feature_positions(obj, 1, "bottom_paw", ...)
```

iii. Each call to `align_feature_positions` loops over all trials independently.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes velocities for `bottom_paw` in addition to `top_paw`, which adds processing. The `moveThresh` value is loaded from motion energy files but never used (the AI uses session median instead). The `my_smooth(xy, 1, "reflect")` call for paw features is a no-op (smoothing with window size 1 does nothing).

ii.
```python
# bottom_paw processing:
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, ...)

# Unused moveThresh:
"moveThresh": float(me.moveThresh)

# No-op smoothing:
if "tongue" not in feat_name:
    xy = my_smooth(xy, 1, "reflect")  # n=1 means no smoothing
```

iii. The AI processes both paw features when only `top_paw` would be sufficient per the reference.
