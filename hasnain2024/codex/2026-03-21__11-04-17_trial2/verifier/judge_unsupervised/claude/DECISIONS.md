# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from a hard-coded list of 12 specific sessions (`CONTEXT_SESSION_SPECS`) corresponding to the Figure 8 two-context ALM analysis in the reference paper. Each session's MATLAB v7.3/HDF5 file is loaded via `mat73.loadmat()`, and the companion motion energy file is loaded via `scipy.io.loadmat()`. Sessions are processed sequentially in a loop.

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

def convert_session(spec: SessionSpec, make_plot: bool = False) -> dict:
    obj = mat73.loadmat(spec.data_path)["obj"]
    me = load_motion_energy(spec)
```

iii. The AI explicitly chose to use the Figure 8 two-context ALM session list from the reference code rather than all sessions in the data directory, because the decoder outputs include behavioral context (WC vs DR) which only varies in these sessions. This is documented in CONVERSION_NOTES.md Step 5, Key Decision 1 and 2.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are determined by the `animal` field of each `SessionSpec`. The unique animal names are extracted from the converted sessions and sorted alphabetically. A `subject_idx` array maps each session to its subject index.

ii.
```python
subjects = sorted({sess["subject"] for sess in converted_sessions})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
dataset["subject_idx"] = np.asarray([subject_to_idx[sess["subject"]] for sess in converted_sessions], dtype=np.int64)
```

iii. The AI noted that the paper text says "six mice" but the Figure 8 loader code names 7 animal IDs (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19). The AI resolved this in favor of the code and raw data, retaining 7 subjects.

## 1-c. How are the data split into sessions?

i. Each entry in `CONTEXT_SESSION_SPECS` defines one session, identified by animal name, date, and probe index. Each session is processed independently by `convert_session()`, producing separate neural, input, and output lists.

ii.
```python
for sess_idx, spec in enumerate(session_specs):
    converted_sessions.append(convert_session(spec, make_plot=make_plot))
```

iii. The session list directly mirrors the reference code's Figure 8 loader scripts which specify 12 sessions across 7 mice. Each unique animal+date combination is one session.

## 1-d. How are the data split into trials?

i. Within each session, trials are identified by integer indices from 0 to `Ntrials-1`. Valid trials are selected via `select_valid_trials()`, then further filtered to require a valid first lick direction after the go cue. Each kept trial becomes one entry in the session's neural/input/output lists.

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

# Further filtering for valid lick direction:
for trial_idx in valid_trials:
    lick_dir = get_first_lick_direction(obj, int(trial_idx), align_times[trial_idx])
    if lick_dir is None:
        continue
    kept_trials.append(int(trial_idx))
```

iii. The AI documented this matches the paper's omission of early/ignore trials and ensures outcome is well-defined (hit or miss). The additional lick-direction filter ensures every trial has a meaningful lick direction output.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by: (1) excluding early-lick trials (`early != 0`), (2) excluding no-response/ignore trials (`no != 0`), (3) excluding stimulation trials (`stim.enable != 0`), (4) requiring hit or miss outcome, and (5) requiring a valid first post-go-cue lick direction (left or right).

ii. (Same code as 1-d above)

iii. The AI's CONVERSION_NOTES.md Step 5 Key Decision 3 states: "Keep non-stim, non-early, non-ignore neural trials with a valid lick outcome (hit or miss). Rationale: this matches the paper's omission of early/ignore trials and keeps outcome well-defined."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu[probe_index]` — specifically the spike times (`trialtm`), trial assignments (`trial`), and quality labels (`quality`) for each unit on the selected probe.

ii.
```python
clu = obj["clu"][spec.probe_index]
# Per unit:
clu["trialtm"][unit_idx]  # spike times relative to trial
clu["trial"][unit_idx]    # 1-based trial number for each spike
clu["quality"]            # quality label for each unit
```

iii. The AI identified that the reference neural decoding scripts operate on `obj.trialdat`, which is derived from spike trains via binning and smoothing, confirming that sorted spike times are the raw source.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5 ms bins over a [-2.5, 2.5] s window, converted to firing rates (counts/dt), and smoothed with a causal Gaussian kernel (window size 15, reflect boundary condition).

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH_N = 15
BCTYPE = "reflect"

def bin_unit_spikes(...):
    aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
    bin_idx = np.floor((aligned - TMIN) / DT).astype(np.int64)
    np.add.at(aligned_counts, (spike_local_trial[valid_bins], bin_idx[valid_bins]), 1.0)
    rates = aligned_counts / DT
    rates = my_smooth(rates.T, SMOOTH_N, BCTYPE).T
    return rates

def my_smooth(x, n, bctype="none"):
    kern = gaussian_window(n)
    kern[: len(kern) // 2] = 0.0  # causal: zero out first half
    kern /= np.sum(kern)
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

iii. The AI documented that this matches the reference `getSeq` function which bins spikes, divides by dt to get firing rates, and applies `mySmooth` with a causal Gaussian kernel.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) quality label filtering excludes units labeled `garbage`, `gabrga`, `noisy`, or `real?`; (2) firing rate filtering excludes units with mean firing rate <= 1 Hz across the alignment window.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0

def keep_quality(quality: str) -> bool:
    return quality not in QUALITY_EXCLUDE

# FR filter:
mean_fr = mean_firing_rate_window(clu["trialtm"][unit_idx], ...)
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The AI noted that `findClusters` with `quality='all'` excludes exactly these four labels, and that Figure 8 scripts set `lowFR = 1` (overriding the default 0.5).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All neural data is aligned to `bp.ev.goCue`. Spike times are subtracted by the go cue time for each trial before binning.

ii.
```python
ALIGN_EVENT = "goCue"
align_times = ensure_1d_numeric(bp["ev"][ALIGN_EVENT])
# In bin_unit_spikes:
aligned = trialtm[valid_spikes] - align_times[trial_numbers_1based[valid_spikes] - 1]
```

iii. The AI documented that `goCue` is the default alignment in the reference code and that WC trials still contain a meaningful `goCue`-equivalent event in the raw data, making unified alignment feasible.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 5 ms (DT = 0.005 s), producing 1000 time bins over the [-2.5, 2.5] s window. No temporal rebinning is applied after the initial binning.

ii.
```python
DT = 0.005
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0  # 1000 bins
```

iii. The AI chose 5 ms based on `getDefaultParams.m` which sets `dt = 1/200` (5 ms), and the paper's methods which mention 5 ms bins for lagged kinematic prediction.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself — the center of each 5 ms time bin relative to the go cue alignment event. No raw behavioral variable is used directly.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
# Per trial:
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. The AI stated this is a "continuous time-from-go-cue vector repeated for every trial" as requested by the decoder task specification.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `arange(TMIN, TMAX, DT) + DT/2`. This gives 1000 values from -2.4975 to 2.4975 seconds. The same array is used for every trial, shaped as `(1, n_timepoints)`.

ii.
```python
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
time_input = TIME_AXIS[None, :].astype(np.float32)
```

iii. No complex processing — just constructing a uniform time grid centered at 0 (go cue).

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same time grid used for neural spike binning, so alignment is inherent — both share the same `TIME_AXIS`.

ii.
```python
# Same TIME_AXIS used for spike binning and input:
EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_AXIS = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2.0
```

iii. Documented as a deliberate design choice — the time input is simply the aligned time axis that all signals share.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.ev.lickL` and `bp.ev.lickR` — the timestamps of left and right lick port contacts — and `bp.ev.goCue` for the alignment time.

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

iii. The AI chose to use actual behavioral lick direction (first post-go-cue lick) rather than the instructed trial side (R/L), as the user explicitly requested "lick direction."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the first left and right lick events occurring at or after the go cue time are found. The direction of the earliest lick determines the label: left = 0, right = 1. Trials with no post-go-cue licks are excluded. The label is replicated as a constant across all time bins.

ii.
```python
output_trial = np.vstack([
    np.full(TIME_AXIS.size, lick_dir, dtype=np.int64),
    ...
])
```

iii. The AI documented this as Key Decision 8: "Use actual first post-alignment lick side from lickport events, not the instructed side (R/L)."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater` — whether automatic water delivery was enabled for the trial.

ii.
```python
autowater = ensure_1d_numeric(bp["autowater"])
# Per trial:
context_label = 0 if autowater[trial_idx] != 0 else 1
```

iii. The AI documented that `autowater` is the exact context variable used throughout the paper/code: `autowater` = WC (water-cued), `~autowater` = DR (delayed-response).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` field is read as a numeric array. For each trial, if `autowater != 0`, context = 0 (WC); otherwise context = 1 (DR). The label is replicated as a constant across all time bins.

ii.
```python
context_label = 0 if autowater[trial_idx] != 0 else 1
np.full(TIME_AXIS.size, context_label, dtype=np.int64)
```

iii. Documented as straightforward binary encoding matching the reference code's DR/WC distinction.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp.hit` — whether the animal responded correctly.

ii.
```python
hit = ensure_1d_numeric(bp["hit"])
outcome_label = 1 if hit[trial_idx] != 0 else 0
```

iii. The AI noted this matches the reference `getOutcome` function, with trial filtering already ensuring only hit or miss trials are included.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. For each kept trial (already filtered to hit or miss), if `hit != 0` then outcome = 1 (correct), else outcome = 0 (incorrect). The label is replicated as a constant across all time bins.

ii.
```python
outcome_label = 1 if hit[trial_idx] != 0 else 0
np.full(TIME_AXIS.size, outcome_label, dtype=np.int64)
```

iii. Simple binary encoding. No additional processing beyond the trial-level filtering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC trajectories in `obj.traj[0]` (side camera view), specifically the `tongue` feature's x,y position time series (`ts[:, 0:2, feat_index]`) and frame timestamps (`frameTimes`), plus video offset information from `bp.ev.bitStart` and `sglx.bitcode.bitstart`.

ii.
```python
tongue_x, tongue_y = align_feature_positions(obj, 0, "tongue", align_times, TIME_AXIS, vidshift)
tongue_vx, tongue_vy = compute_velocity(tongue_x, tongue_y, "tongue")
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The AI identified "tongue" from the side camera (view 0) as the relevant feature, matching the reference code's `traj_features` listing.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Processing steps: (1) Extract tongue x,y positions per trial from DLC trajectories. (2) Align frame times to the go cue via video offset subtraction. (3) Interpolate positions onto the 5 ms neural time axis. (4) No smoothing for tongue (skipped when feature name contains "tongue"). (5) Compute velocity using `np.gradient` (central differences). (6) For tongue, NaN velocities are set to 0 (no baseline drift subtraction). (7) Compute scalar speed as `sqrt(vx^2 + vy^2)`.

ii.
```python
# Position alignment (no smoothing for tongue):
if "tongue" not in feat_name:
    xy = my_smooth(xy, 1, "reflect")
# Velocity computation:
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
if "tongue" not in feat_name:
    xv = xv - basederiv[0]
    yv = yv - basederiv[0]
else:
    xv = np.where(np.isfinite(xv), xv, 0.0)
    yv = np.where(np.isfinite(yv), yv, 0.0)
tongue_speed = np.sqrt(tongue_vx**2 + tongue_vy**2)
```

iii. The AI followed the reference `findVelocity.m` logic: gradient-based velocity, no baseline subtraction for tongue, NaN-to-zero for invisible tongue periods.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile (median) of tongue speed is computed per session across all kept trials, but only from time points where the tongue is visible. The binary output is: 1 if speed >= threshold AND tongue is visible, 0 otherwise (including all invisible periods).

ii.
```python
tongue_visible_kept = tongue_visible[:, kept_trials]
if np.any(tongue_visible_kept):
    tongue_threshold = float(np.nanpercentile(tongue_speed[:, kept_trials][tongue_visible_kept], 50))
else:
    tongue_threshold = 0.0
# Per trial:
tongue_bin = ((tongue_speed[:, trial_idx] >= tongue_threshold) & tongue_visible[:, trial_idx]).astype(np.int64)
```

iii. The AI initially had issues with tongue discretization collapsing because invisible periods dominated the median. Fixed by computing the median only from visible timepoints and assigning invisible periods to the low bin (0).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue positions are interpolated from video frame times onto the same 5 ms time axis used for neural data, after subtracting the video offset and go cue alignment time. This ensures temporal alignment with neural data.

ii.
```python
shifted_time = frame_times - vidshift - align_times[trial_idx]
xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], left=np.nan, right=np.nan)
```

iii. Same alignment procedure as the reference `findPosition` / `getKinematicsFromVideo` functions.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC trajectories in `obj.traj[1]` (bottom camera view), specifically the `top_paw` and `bottom_paw` features' x,y position time series.

ii.
```python
for paw_feat in ("top_paw", "bottom_paw"):
    paw_x, paw_y = align_feature_positions(obj, 1, paw_feat, align_times, TIME_AXIS, vidshift)
    paw_vx, paw_vy = compute_velocity(paw_x, paw_y, paw_feat)
    paw_speeds.append(np.sqrt(paw_vx**2 + paw_vy**2))
```

iii. The AI identified `top_paw` and `bottom_paw` from the bottom camera (view 1) as the relevant features, matching the reference code's `traj_features` listing.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing steps for each paw feature: (1) Extract x,y positions per trial. (2) Apply `my_smooth(xy, 1, "reflect")` (effectively no smoothing since N=1). (3) Align to go cue via video offset. (4) Interpolate onto 5 ms time axis. (5) Fill NaN positions with nearest-neighbor interpolation. (6) Compute velocity via `np.gradient` with baseline drift subtraction. (7) Fill NaN velocities with nearest-neighbor. (8) Compute scalar speed. (9) Average speed across `top_paw` and `bottom_paw`.

ii.
```python
# Non-tongue position processing:
xy = my_smooth(xy, 1, "reflect")  # effectively no-op
xpos[:, trial_idx] = fill_nearest_1d(xpos[:, trial_idx])
# Velocity:
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
xv = xv - basederiv[0]
yv = yv - basederiv[0]  # Note: reference bug replicated - uses basederiv[0] for both
xv = fill_nearest_1d(xv)
yv = fill_nearest_1d(yv)
# Averaging:
paw_speed = np.nanmean(paw_stack, axis=0)  # mean of top_paw and bottom_paw speeds
```

iii. The AI replicated the reference code's velocity computation including the baseline drift subtraction bug (using `basederiv[0]` for both x and y velocity).

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile of paw speed is computed per session across all kept trials and all timepoints. Binary output: 1 if speed >= threshold, 0 otherwise.

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[:, kept_trials], 50))
# Per trial:
(paw_speed[:, trial_idx] >= paw_threshold).astype(np.int64)
```

iii. This follows the user's required 50th percentile discretization. The resulting distribution is exactly 50/50 by construction.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment procedure as tongue: paw positions are interpolated from video frame times onto the 5 ms neural time axis after subtracting video offset and go cue time.

ii. (Same `align_feature_positions` function as tongue, called with view index 1 and paw feature names)

iii. Matches reference code alignment approach.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from separate companion files (`motionEnergy_<animal>_<date>.mat`) containing per-trial motion energy time series (`me.data`) and a movement threshold (`me.moveThresh`). Frame timestamps from `obj.traj[0]` are used for temporal alignment.

ii.
```python
def load_motion_energy(spec: SessionSpec) -> dict:
    me_mat = loadmat(spec.motion_energy_path, squeeze_me=True, struct_as_record=False)
    me = me_mat["me"]
    return {"data": np.atleast_1d(me.data), "moveThresh": float(me.moveThresh)}
```

iii. The AI identified the motion energy companion files as matching the reference `loadMotionEnergy.m` function.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Processing: (1) Load per-trial motion energy values and frame timestamps from the side camera view. (2) Compute video offset (`vidshift`). (3) For each trial, align frame times by subtracting vidshift and go cue time. (4) Interpolate motion energy onto the 5 ms neural time axis. (5) Fill NaN values with nearest-neighbor interpolation.

ii.
```python
def align_motion_energy(obj, me, align_times, time_axis):
    vidshift = compute_vidshift(obj)
    for trial_idx in range(n_trials):
        shifted_time = frame_times[valid] - vidshift - align_times[trial_idx]
        aligned[:, trial_idx] = np.interp(time_axis, shifted_time, me_trial[valid], left=np.nan, right=np.nan)
        aligned[:, trial_idx] = fill_nearest_1d(aligned[:, trial_idx])
```

iii. Matches the reference `loadMotionEnergy.m` alignment procedure using `interp1` and `fillmissing('nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile of aligned motion energy is computed per session across all kept trials and timepoints. Binary output: 1 if motion energy >= threshold, 0 otherwise.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[:, kept_trials], 50))
(motion_energy[:, trial_idx] >= me_threshold).astype(np.int64)
```

iii. Follows the user's required 50th percentile discretization. Note: the reference code uses a manually chosen `moveThresh` per session, but the instruction requires median-based thresholding instead.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the same 5 ms time axis used for neural data, using frame timestamps from the side camera view aligned to the go cue event.

ii. (Same `align_motion_energy` function shown in 9-b)

iii. Uses the same temporal framework as neural and kinematic data.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Missing/NaN frame timestamps are replaced with a nominal 400 Hz grid. (2) Missing tongue positions are left as NaN (tongue visibility tracked). (3) Missing non-tongue positions and velocities are filled with nearest-neighbor interpolation (`fill_nearest_1d`). (4) Missing motion energy values are filled with nearest-neighbor. (5) Trials with no valid post-go-cue lick are excluded. (6) Dropped video frames are detected and trials with all-NaN dropped frames are skipped.

ii.
```python
# Frame time fallback:
if frame_times is None:
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Nearest-neighbor fill for non-tongue:
def fill_nearest_1d(x):
    x[~good] = np.interp(idx[~good], idx[good], x[good])

# Tongue visibility:
tongue_visible = np.isfinite(tongue_x) & np.isfinite(tongue_y)
```

iii. The AI documented finding 2 raw trials with empty/all-NaN frameTimes and 527 raw trials with completely invisible tongue, confirming these edge cases are handled.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the large MATLAB v7.3/HDF5 session files via `mat73.loadmat()` and the trial-by-trial kinematic alignment loops (interpolating DLC positions for tongue, top_paw, and bottom_paw features). Total conversion for 12 sessions takes ~72 seconds.

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]  # Most expensive I/O
# Kinematic alignment loops:
for trial_idx in range(n_trials):
    xpos[:, trial_idx] = np.interp(time_axis, shifted_time[valid], xy[valid, 0], ...)
```

iii. The AI documented ~7.4 s/session average, with I/O being the dominant cost.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-by-trial loops in `align_feature_positions`, `compute_velocity`, and `align_motion_energy` could potentially be vectorized, though the per-trial varying valid-sample masks make this challenging. The per-column smoothing loop in `my_smooth` could use `scipy.ndimage.convolve1d`. The unit-by-unit loop for spike binning and FR filtering could be partially vectorized.

ii.
```python
# Trial loop in align_feature_positions (lines 207-244):
for trial_idx in range(n_trials):
    ...
# Column loop in my_smooth (lines 98-99):
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
# Unit loop for spike binning (lines 436-453):
for unit_idx, use_unit in enumerate(quality_keep):
    ...
```

iii. The AI noted that spike binning was vectorized within each unit using `np.add.at` but kinematic alignment remained trial-by-trial.

## 11-c. What processing does the code repeat multiple times?

i. Video offset (`compute_vidshift`) is computed twice per session — once in `convert_session` for kinematics, and once inside `align_motion_energy`. Frame time extraction from `obj.traj` is done multiple times (once for tongue, once for each paw feature, once for motion energy).

ii.
```python
vidshift = compute_vidshift(obj)  # Called in convert_session
# Also called inside:
def align_motion_energy(obj, me, align_times, time_axis):
    vidshift = compute_vidshift(obj)  # Computed again
```

iii. Minor inefficiency but not significant given the overall runtime.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes `me.moveThresh` (loaded from the motion energy file) but never uses it — the decoder task requires 50th percentile thresholding instead. Position smoothing for non-tongue features (`my_smooth(xy, 1, "reflect")`) is a no-op (N=1 returns input unchanged). The `summary` dict computed per session is used only for logging, not for the final dataset.

ii.
```python
# moveThresh loaded but unused for thresholding:
"moveThresh": float(me.moveThresh)  # Only stored, not used for output discretization

# No-op smoothing:
if "tongue" not in feat_name:
    xy = my_smooth(xy, 1, "reflect")  # N=1 returns unchanged
```

iii. The `moveThresh` is the paper's manual threshold but the instructions require median-based thresholding.
