# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, corresponding to the Figure 8 two-context analysis cohort. It uses only h5py for HDF5 (v7.3) files and scipy.io.loadmat for v5 motion-energy files. The session list is hard-coded in `SESSION_SPECS`. The `RandomizedDelay_Ephys_Behavior` folder is not accessed at all. `DATA_ROOT` is hard-coded to only the `Ephys_Behavior` directory.

ii.
```python
DATA_ROOT = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    SessionSpec("JEB7", "2021-04-29", (1,)),
    SessionSpec("JEB7", "2021-04-30", (1,)),
    SessionSpec("EKH1", "2021-08-07", (2,)),
    SessionSpec("EKH3", "2021-08-11", (2,)),
    SessionSpec("JGR2", "2021-11-16", (1,)),
    SessionSpec("JGR2", "2021-11-17", (1,)),
    SessionSpec("JGR3", "2021-11-18", (1,)),
    SessionSpec("JEB19", "2023-04-21", (1,)),
    SessionSpec("JEB19", "2023-04-20", (1,)),
    SessionSpec("JEB19", "2023-04-19", (1,)),
    SessionSpec("JEB19", "2023-04-18", (1,)),
)
```

Loading a session:
```python
with h5py.File(spec.data_path, "r") as handle:
    bp = load_behavior(handle)
```

iii. The AI justified using only 12 sessions by arguing these are the Figure 8 two-context cohort that provides balanced WC/DR context classes. In CONVERSION_NOTES.md Step 4, it states: "Randomized-delay sessions are a distinct task and nearly lack WC examples in the files, so including them would change the task and heavily distort context classes." It also excluded the DR-only fixed-delay sessions (JEB13, JEB14, JEB15) for similar reasons.

## 1-b. How are the data split into subjects?

i. The subject (animal) is extracted from the `SessionSpec.animal` field, which is the prefix before the underscore in the session name. The AI identifies 7 unique subjects from its 12 sessions: JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19.

ii.
```python
subjects = list(dict.fromkeys(spec.animal for spec in specs))
subject_idx = np.array([subjects.index(spec.animal) for spec in specs], dtype=np.int64)
```

iii. The AI notes that the paper reports 6 mice for this cohort but the data contains 7 unique IDs, and chose to preserve the source identifiers rather than collapse them.

## 1-c. How are the data split into sessions?

i. One session is one `SessionSpec` entry, corresponding to one HDF5 file in `Ephys_Behavior`. The AI processes exactly 12 sessions. Each becomes one element of `neural`, `input`, and `output`.

ii.
```python
specs = SESSION_SPECS[:2] if args.sample else SESSION_SPECS
for spec in specs:
    session, diag = convert_session(spec)
    converted.append(session)
```

iii. The AI explicitly chose the 12-session two-context subset based on the Figure 8 loader scripts, arguing these are the sessions with the alternating WC/DR context design.

## 1-d. How are the data split into trials?

i. Yes, trials are split correctly. The AI reads `Ntrials` from `obj/bp/Ntrials` and validates that all behavioral fields match this count. Each trial is one entry in the behavioral arrays. The AI also validates that hit/miss/no are mutually exhaustive and R/L are mutually exhaustive.

ii.
```python
ntrials = int(np.asarray(handle["obj/bp/Ntrials"]).squeeze())
if any(len(value) != ntrials for value in bp.values()):
    raise ValueError("Behavior fields do not all match Ntrials")
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
if not np.all(outcome_sum == 1):
    raise ValueError("hit/miss/no are not mutually exhaustive")
```

iii. The AI uses HDF5 direct field reads and validates consistency before proceeding.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`). It does not have a check for trials past the end of the neural recording (unlike the reference). Across the 12 sessions, this keeps 3,116 of 3,626 trials.

ii.
```python
keep_trials = ~bp["early"] & ~bp["stim"]
raw_trials = np.flatnonzero(keep_trials)
if len(raw_trials) < 2:
    raise ValueError(f"{spec.session_id}: fewer than two retained trials")
```

iii. The AI follows the paper's exclusion of early-lick and photostimulation trials, documented in CONVERSION_NOTES.md Step 4: "Exclude stimulation and early trials. Retain hit, miss, and no/ignore because the requested output explicitly requires all three outcomes."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI uses `obj.clu{probe}` cluster data: `trial` (1-based spike trial IDs), `trialtm` (within-trial spike times), and `quality` (manual curation label). Go cue times `bp.ev.goCue` provide alignment.

ii.
```python
quality_clusters = selected_clusters(handle, spec.probes)
# In selected_clusters:
trial_refs = matlab_cell_refs(group["trial"])
trialtm_refs = matlab_cell_refs(group["trialtm"])
trials = deref_vector(handle, trial_refs[cluster_index], np.int64) - 1
trial_times = deref_vector(handle, trialtm_refs[cluster_index], np.float64)
```

iii. Same source variables as the reference code's spike data.

## 2-b. How is the `neural` data processed?

i. The AI bins spike counts into 5ms bins using `np.add.at`, divides by dt to get firing rates (Hz), then applies a **causal** Gaussian kernel. The kernel is `gausswin(15)` with the first 7 coefficients zeroed and the remainder normalized, applied via `scipy.signal.lfilter` with leading padding. This faithfully reproduces the MATLAB `mySmooth.m` function.

ii.
```python
def causal_gaussian_kernel(n: int = 15) -> np.ndarray:
    kernel = gaussian(n, std=(n - 1) / (2 * 2.5), sym=True).astype(np.float32)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel

def reference_smooth(values, n=15):
    padded = np.concatenate((values[..., :n], values), axis=-1)
    causal_coefficients = KERNEL[n // 2 :]
    filtered = lfilter(causal_coefficients, [1.0], padded, axis=-1)
    return filtered[..., n:].astype(np.float32, copy=False)

# In build_neural:
return reference_smooth(counts / DT)
```

iii. The AI explicitly matched the MATLAB code's causal smoothing: "Match MATLAB gausswin(n, 2.5), then the released causalization." The AI verified numerical agreement with explicit `np.convolve` to float32 precision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (note: 'poor' is NOT excluded, unlike the reference). Second, the AI implements a complex PSTH-based firing rate filter that recreates the reference `removeLowFRClusters.m`: it computes condition-specific PSTHs across 7 conditions (all trials, hit/miss x DR/WC, hit-no-early x DR/WC) in 10ms bins from -3 to +2.5s, applies causal smoothing, then takes the mean across all conditions and timepoints, requiring strict > 1 Hz. This yields 521 units across 12 sessions.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}

def low_fr_filter(clusters, bp):
    conditions = (
        np.ones(ntrials, dtype=bool),
        hit & ~stim & ~aw,
        hit & ~stim & aw,
        miss & ~stim & ~aw,
        miss & ~stim & aw,
        hit & ~stim & ~aw & ~early,
        hit & ~stim & aw & ~early,
    )
    edges = np.arange(-3.0, 2.5 + 0.005, 0.01, dtype=np.float64)
    # ... computes PSTH per condition, smooths, takes mean
    keep = mean_rates > LOW_FR_HZ
```

iii. The AI noted that `findClusters.m` excludes exactly these 4 labels and reproduced the reference's complex FR calculation. The AI does not exclude 'poor' quality clusters, which the reference does.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to go cue is done by subtracting `bp.ev.goCue` from each spike's trial time, then binning into the -2.5 to +2.5s window.

ii.
```python
aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
in_window = (bins >= 0) & (bins < N_TIME)
np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
```

iii. Same approach as the reference code's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 5ms bins spanning -2.5 to +2.5s from go cue, yielding 1000 time points. No rebinning is applied.

ii.
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.005
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
N_TIME = len(TIME)
```

iii. Matches the reference code's `params.dt = 1/200` and `params.tmin/tmax`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. Derived from the bin centers of the 5ms grid, which is defined by the go cue alignment. Not from a raw data variable.

ii.
```python
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The time axis is analytically defined, matching the neural binning grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing from raw data; the time axis is generated as bin centers from -2.4975 to +2.4975 in steps of 0.005.

ii. Same as 3-a.

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is the neural binning grid itself. The same TIME array is used to define both the spike binning and the input.

ii. Same as 3-a.

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.R` (instructed right), `bp.hit`, `bp.miss`, and `bp.no`. The actual lick direction is inferred from the combination of instructed side and outcome.

ii.
```python
bp = {field: direct_field(handle, f"obj/bp/{field}", bool) for field in fields}
# In static_trial_outputs:
if bp["no"][raw]:
    lick = 2
elif bp["hit"][raw]:
    lick = 1 if bp["R"][raw] else 0
elif bp["miss"][raw]:
    lick = 0 if bp["R"][raw] else 1
```

iii. The AI derives actual response direction from the combination of instructed side and outcome, consistent with the task logic.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit trials licked the instructed port (R=1 -> right, L=1 -> left), miss trials licked the opposite port, and no/ignore trials map to "none" (class 2). Codes: left=0, right=1, none=2. The value is broadcast across all time bins.

ii. Same as 4-a.

iii. The AI correctly implements the logical derivation of actual lick direction.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. One per-trial field: `bp.autowater`. Autowater=True indicates WC context.

ii.
```python
bp["autowater"] = direct_field(handle, "obj/bp/autowater", bool)
# In static_trial_outputs:
context = 0 if bp["autowater"][raw] else 1
```

iii. Direct read from the trial table.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True -> WC (0), autowater=False -> DR (1). Broadcast across time bins.

ii. Same as 5-a.

iii. Codes follow the prompt's specification.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss`, `bp.no`. The AI validates they are mutually exhaustive.

ii.
```python
outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
```

iii. Direct from the behavioral fields with validation.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Relabelling: miss -> incorrect (0), hit -> correct (1), no -> ignore (2). Broadcast across time bins.

ii. Same as 6-a.

iii. Matches the instruction specification.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses only the **side camera** view (view index 0), specifically the "tongue" feature's DLC x, y, and confidence from `obj.traj`. Frame times and bitcode fields for clock synchronization are also used.

ii.
```python
tongue[kept_trial], tongue_visible[kept_trial], tongue_x, tongue_y = feature_speed(
    handle, side, int(raw_trial), "tongue", aligned_side_time, True
)
```

iii. The AI uses only the side-camera tongue, unlike the reference which also uses the bottom-camera "top_tongue" and averages the two views.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI: (1) Interpolates DLC x/y positions from frame times onto the 5ms TIME grid using `np.interp`. (2) Computes `np.gradient` of the interpolated x and y (in pixels per 5ms bin). (3) Takes the Euclidean magnitude (`np.hypot`). (4) Sets speed to NaN where the interpolated position was NaN (not visible). (5) Thresholds at the session median of visible values.

ii.
```python
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
y = interpolate_with_nans(aligned_time, trajectory[:, 1, feature_index], TIME)
visible = np.isfinite(x) & np.isfinite(y)
# For tongue (is_tongue=True), no nearest-fill:
x_for_velocity, y_for_velocity = x, y
x_velocity = np.gradient(x_for_velocity)
y_velocity = np.gradient(y_for_velocity)
speed = np.hypot(x_velocity, y_velocity)
speed[~visible] = np.nan
```

iii. The AI interpolates positions to the neural time grid first, then differentiates. This differs from the reference which computes velocity at frame resolution within contiguous valid runs using Gaussian smoothing, then bins into 5ms bins. The AI also does not apply any Gaussian smoothing to positions before differentiation and does not use the second camera view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session median of all visible tongue speed values is used as the threshold. Values >= median -> 1, < median -> 0, not visible -> 2.

ii.
```python
tongue_threshold = float(np.nanmedian(tongue[tongue_visible]))
classes = np.full(N_TIME, 2, dtype=np.int8)
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. Matches the instruction specification for 50th percentile thresholding.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are aligned by subtracting the per-session bitcode-derived video offset and the trial's go cue time. The aligned positions are then interpolated onto the same 5ms TIME grid as the neural data.

ii.
```python
aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
# Then interpolated via:
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. The video offset is computed identically to the reference's `findVideoOffset.m`. The alignment approach (interpolation to grid) differs from the reference's approach (compute at frame rate, then bin).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The bottom camera (view index 1), specifically the "top_paw" feature's DLC x, y, and confidence from `obj.traj`.

ii.
```python
paw[kept_trial], paw_visible[kept_trial], paw_x, paw_y = feature_speed(
    handle, bottom, int(raw_trial), "top_paw", aligned_bottom_time, False
)
```

iii. Same feature selection as the reference.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI: (1) Interpolates DLC x/y positions onto the 5ms grid. (2) Applies nearest-fill (`fill_nearest`) to paw positions (non-tongue features are filled). (3) Computes `np.gradient` of the filled positions. (4) **Subtracts a baseline derivative** (the median of frame-to-frame differences) from both x and y velocity components. Notably, uses x baseline for both x_velocity and y_velocity. (5) Takes Euclidean magnitude. (6) Sets NaN where original positions were not visible.

ii.
```python
# For paw (is_tongue=False):
x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
stacked = np.column_stack((x_for_velocity, y_for_velocity))
differences = np.diff(stacked, axis=0)
baseline_derivative = np.array([
    np.median(column[np.isfinite(column)]) if np.any(np.isfinite(column)) else 0.0
    for column in differences.T
])
# Match the released function, including its use of x baseline for y.
x_velocity = x_velocity - baseline_derivative[0]
y_velocity = y_velocity - baseline_derivative[0]
speed = np.hypot(x_velocity, y_velocity)
```

iii. The AI claims to match the reference `findVelocity.m` which includes baseline subtraction. The comment "Match the released function, including its use of x baseline for y" suggests this is intentional. The reference solution does not include this baseline subtraction step.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session median of visible values as threshold. Values >= median -> 1, < median -> 0, not visible -> 2.

ii. Same discretization logic as tongue in `discretize_outputs`.

iii. Matches the instruction specification.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: frame times corrected by video offset and go cue, then interpolated onto the 5ms grid. Uses the bottom camera's frame times.

ii.
```python
aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]
```

iii. Same alignment principle as tongue, using the appropriate camera's frame times.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Separate `motionEnergy_<animal>_<date>.mat` files, loaded via scipy.io.loadmat. Contains `me.data` with one trace per trial.

ii.
```python
def load_motion_energy(path):
    motion = loadmat(path, simplify_cells=True)["me"]
    data = motion["data"]
    if isinstance(data, dict):
        data = data["data"]
    return np.atleast_1d(data)
```

iii. Handles the nested wrapping structure of the motion energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI: (1) Aligns motion energy frames using side camera frame times (when available) or a fallback 400 Hz nominal clock with 0.5s shift (when frame times are NaN). (2) **Interpolates** motion energy values onto the 5ms TIME grid using `np.interp`. (3) Applies **nearest-fill** to fill edge NaNs. (4) Thresholds at session median.

ii.
```python
if len(trial_motion) == len(side_frames) and np.sum(np.isfinite(side_frames)) >= 2:
    motion_times = aligned_side_time
else:
    motion_times = np.arange(1, len(trial_motion) + 1, dtype=np.float64) / 400
    motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The AI reproduces the MATLAB `loadMotionEnergy.m` fallback for missing frame times. The interpolation and nearest-fill approach differs from the reference which averages frames within each bin.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session median of all valid motion energy values. Values >= median -> 1, < median -> 0, no video -> 2.

ii.
```python
motion_threshold = float(np.nanmedian(motion[valid_motion]))
```

iii. Matches instruction specification.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same as tongue: side camera frame times corrected by video offset and go cue. When frame times are all NaN, uses nominal 400 Hz clock with 0.5s offset (matching MATLAB reference fallback). Values are interpolated onto the TIME grid.

ii. See 9-b code snippet.

iii. The fallback for missing frame times matches `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Missing frame times (all NaN): For motion energy, falls back to nominal 400 Hz clock. For DLC tracking, `interpolate_with_nans` returns all NaN, yielding class 2 (not visible). (2) Untracked DLC frames: positions with NaN x/y after interpolation are marked not visible. (3) Paw gaps: nearest-filled before velocity computation, but visibility mask is preserved for class assignment. (4) The AI validates data integrity extensively (outcome_sum == 1, side_sum == 1, finite go cues).

ii.
```python
def interpolate_with_nans(times, values, target):
    finite_time = np.isfinite(times)
    if np.sum(finite_time) < 2:
        return np.full(len(target), np.nan, dtype=np.float64)
    return np.interp(target, x, y, left=np.nan, right=np.nan)

def fill_nearest(values):
    # ... nearest-neighbor fill for non-tongue features
```

iii. The AI documents handling of the all-NaN frame times case in CONVERSION_NOTES.md Step 10 as a bug found and fixed during review.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files dominates runtime. The full conversion of 12 sessions runs in about 43 seconds.

ii.
```python
with h5py.File(spec.data_path, "r") as handle:
    bp = load_behavior(handle)
    # ... all processing within the h5py context
```

iii. File I/O is the main bottleneck, consistent with the reference solution's observation.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `align_kinematics_and_motion` iterates over every trial to load and process DLC trajectories individually. The per-unit loop in `build_neural` uses `np.add.at` but still loops over clusters. The `low_fr_filter` function loops over units and conditions.

ii.
```python
for kept_trial, raw_trial in enumerate(raw_trials):
    # ... loads trajectory, computes speed for each trial
for unit, cluster in enumerate(clusters):
    # ... spike binning per unit
```

iii. Variable-length DLC data prevents easy vectorization of the trial loop. The spike binning could potentially be more vectorized but the current approach is already reasonably efficient.

## 11-c. What processing does the code repeat multiple times?

i. The DLC trajectory data is loaded twice per trial for the side camera: once for tongue and once implicitly when loading frame times for motion energy alignment. The code calls `trajectory_for_trial` separately each time.

ii.
```python
# First call for tongue:
side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
# Later for motion energy:
time = self._frame_time(trial, SIDE)  # reads frame times again
```

iii. This is a minor inefficiency since the HDF5 reads are cached at the OS level.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `low_fr_filter` function computes full condition-specific PSTHs (7 conditions, ~550 bins each) with causal smoothing for every quality-eligible cluster, just to determine a mean rate for filtering. A simple mean firing rate would suffice. The code also stores extensive diagnostic information (`diagnostics` dict) and computes `mean_rates_all_quality_units` for plotting even in non-plotting mode.

ii.
```python
def low_fr_filter(clusters, bp):
    # Computes 7 condition PSTHs for every cluster just to get mean rate
    for condition in conditions:
        counts = np.histogram(aligned[use], bins=edges)[0].astype(np.float32)
        rate = counts / (float(np.sum(condition)) * 0.01)
        condition_psths.append(reference_smooth(rate))
    mean_rates[unit] = np.nanmean(np.stack(condition_psths))
```

iii. The AI intentionally reproduces the exact reference MATLAB calculation for FR filtering rather than using a simpler approximation. The diagnostic computation is cheap but technically unnecessary when not plotting.
