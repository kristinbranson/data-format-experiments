# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes a 12-session subset in `SESSION_SPECS`, all from `Ephys_Behavior`, and opens each session directly with `h5py.File`. It does not scan both task folders, does not include the randomized-delay sessions, and does not implement the reference's general `load_mat` logic for v5/v7.3 session files. Motion energy is loaded separately with `scipy.io.loadmat`.

ii.
```python
DATA_SUBDIR = "Ephys_Behavior"
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
    {"subject": "JEB19", "date": "2023-04-21", "probe": 1},
]

def process_session(data_root: Path, spec: dict) -> dict:
    session_tag = f"{spec['subject']}_{spec['date']}"
    session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
    motion_path = data_root / DATA_SUBDIR / f"motionEnergy_{session_tag}.mat"

    with h5py.File(session_path, "r") as mat:
        beh = load_basic_behavior(mat)
```

iii. In the trajectory, the AI explicitly said it would use the "12 hand-picked two-context ALM sessions" from the MATLAB context scripts rather than sweeping every ephys file, and cited that list as matching the paper's reported ALM unit count.

## 1-b. How are the data split into subjects?

i. Subjects are taken from the hard-coded `subject` field in each `SESSION_SPECS` entry. The exported `subjects` list preserves first appearance order, and `subject_idx` maps each session to that list.

ii.
```python
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    ...
]

subjects = []
subject_to_idx = {}
subject_idx = []
for sess in sessions:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The trajectory does not give a separate justification for subject splitting beyond using the hand-picked session list and carrying the `subject` field through assembly.

## 1-c. How are the data split into sessions?

i. One session is one hard-coded `{subject, date, probe}` entry. Each session becomes one element of `neural`, `input`, and `output`, and the build step iterates only over that fixed list.

ii.
```python
def build_dataset(data_root: Path) -> dict:
    sessions = [process_session(data_root, spec) for spec in SESSION_SPECS]

    data = {
        "neural": [sess["neural"] for sess in sessions],
        "input": [sess["input"] for sess in sessions],
        "output": [sess["output"] for sess in sessions],
```

iii. The AI justified this in the trajectory by saying the decoder targets only made sense for the "two-context electrophysiology sessions" and that the MATLAB context scripts used a hard-coded session list.

## 1-d. How are the data split into trials?

i. Trials are taken directly from per-trial behavior arrays in `obj.bp`. The script keeps a boolean `valid_trials` mask over `ntrials`, uses `valid_idx` as the retained trial numbers, and slices neural, video, and per-trial labels by those indices.

ii.
```python
data = {
    "ntrials": int(read_h5_numeric(bp["Ntrials"])),
    "R": read_h5_numeric(bp["R"]).astype(bool),
    ...
    "goCue": read_h5_numeric(bp["ev"]["goCue"]).astype(np.float64),
}

valid_trials = ~beh["early"] & ~beh["stim_enable"]
valid_idx = np.flatnonzero(valid_trials)

valid_neural = neural_all_trials[:, valid_idx, :]
for kept_trial_pos, trial_idx in enumerate(valid_idx):
    neural_trials.append(valid_neural[:, kept_trial_pos, :].astype(np.float32))
```

iii. The trajectory justification was implicit: the AI repeatedly described the pipeline as trial selection first, followed by go-cue alignment and export of non-early, non-stim trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters out early-lick and photostimulation trials only, using `~early & ~stim_enable`. It keeps hit, miss, and ignore trials, and does not implement the reference solution's additional cutoff for trials after the recording stopped.

ii.
```python
valid_trials = ~beh["early"] & ~beh["stim_enable"]
valid_idx = np.flatnonzero(valid_trials)
if valid_idx.size < 2:
    raise ValueError(f"{session_tag} has fewer than two valid trials after filtering")
```

iii. The trajectory states that the export would include only "non-early and non-stim trials" and would keep correct, incorrect, and ignore outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from `obj.clu[probe]`, specifically each unit's `quality`, `trial`, and `trialtm`, plus `goCue` from behavior for alignment. The code also reads `site` or `channel`, but only for metadata-like bookkeeping and not for the final decoder arrays.

ii.
```python
quality_refs = np.asarray(clu["quality"][()]).squeeze()
trial_refs = np.asarray(clu["trial"][()]).squeeze()
trialtm_refs = np.asarray(clu["trialtm"][()]).squeeze()
...
spike_trials = read_trial_ref_numeric(mat, trial_refs[unit_idx]).astype(np.int64)
spike_trialtm = read_trial_ref_numeric(mat, trialtm_refs[unit_idx]).astype(np.float64)
aligned_times = spike_trialtm - go_cue[spike_trials]
```

iii. The trajectory repeatedly described the neural pipeline as "spikes aligned to `goCue`" with low-rate units removed after quality filtering.

## 2-b. How is the `neural` data processed?

i. For each kept unit and each trial, spike times are aligned to go cue, histogrammed into a fixed grid from `-3.0` to `2.5` s with `DT = 10 ms`, converted to rates by dividing by `DT`, and smoothed with a causal half-Gaussian built by `my_smooth(..., SMOOTH=15, BOUNDARY="reflect")`. There is no baseline subtraction or z-scoring.

ii.
```python
TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0
SMOOTH = 15

counts, _ = np.histogram(spk, bins=edges)
rates = counts.astype(np.float64) / DT
trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)
```

iii. In the trajectory and final summary, the AI said it was using the paper's causal Gaussian smoothing and a `goCue`-aligned fixed window.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are first filtered by cluster quality labels, dropping only `garbage`, `gabrga`, `noisy`, and `real?`. Surviving units are then filtered by mean firing rate `> 1.0`, where mean firing rate is computed after building trial-by-time firing-rate arrays and averaging PSTHs across several condition masks.

ii.
```python
def quality_is_usable(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

...
mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
return trialdat[use], sites[use]
```

iii. The trajectory says the AI chose the paper's `>1 Hz` rule and validated its session/unit counts against the MATLAB context-session scripts. It did not separately justify omitting `poor` from the quality drop list.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting the trial's `goCue` time from `trialtm` after converting 1-based MATLAB trial indices to 0-based Python indices.

ii.
```python
spike_trials = spike_trials - 1
aligned_times = spike_trialtm - go_cue[spike_trials]
```

iii. The trajectory explicitly described the neural activity as "aligned to `goCue`" and read the MATLAB `alignSpikes.m` code before implementation.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The neural data are represented on a 10 ms grid spanning `-3.0` to `2.5` s from go cue. Spikes are histogrammed directly into that grid; there is no second-stage temporal rebinning afterward.

ii.
```python
DT = 1.0 / 100.0

def get_edges() -> np.ndarray:
    return np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
```

iii. The AI's final summary in the trajectory explicitly says "bins neural activity at 10 ms from `-3.0` to `2.5` s."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time axis is not read from a raw per-trial variable. It is a constructed template based on the global `TIME_MIN`, `TIME_MAX`, and `DT` constants, intended to represent time relative to the go-cue alignment event.

ii.
```python
def get_time_axis() -> np.ndarray:
    edges = np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
    return edges[:-1] + DT / 2.0

taxis = get_time_axis().astype(np.float32)
input_template = taxis[None, :]
```

iii. The trajectory justification is indirect: the AI described the exported inputs as a fixed `goCue`-aligned time axis shared with the neural data.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The script computes bin centers from the global time edges and repeats that same 1D array for every kept trial.

ii.
```python
edges = np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
return edges[:-1] + DT / 2.0

...
input_trials.append(input_template.copy())
```

iii. No separate processing justification appears in the trajectory beyond the decision to use a fixed `goCue`-aligned bin grid.

## 3-c. How is `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `get_time_axis()` grid that neural spike counts use through `get_edges()`, so each input sample index corresponds to the same time binning scheme as the neural data.

ii.
```python
edges = get_edges()
...
counts, _ = np.histogram(spk, bins=edges)

taxis = get_time_axis().astype(np.float32)
input_template = taxis[None, :]
```

iii. The trajectory says the AI aligned spikes and video to `goCue` and used a fixed exported time axis, implying the input is the same temporal grid as the neural output.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the raw left-lick and right-lick event-time cell arrays `bp.ev.lickL` and `bp.ev.lickR`, together with `goCue` to determine which licks count as post-go. The trial-level `hit`/`miss`/`R` fields are not used for this label.

ii.
```python
data["lickL_refs"] = np.asarray(bp["ev"]["lickL"][()]).squeeze()
data["lickR_refs"] = np.asarray(bp["ev"]["lickR"][()]).squeeze()

lick_dir = first_post_go_lick_direction(mat, beh["lickL_refs"], beh["lickR_refs"], beh["goCue"])
```

iii. In the trajectory, the AI said "lick direction is the first post-go lick" and checked lick timing around go cue before finalizing the export.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. For each trial, the AI finds the first left lick and first right lick occurring at or after go cue. If the left lick happens first, the class is `0`; if the right lick happens first, the class is `1`; if neither exists, the class remains `2` (`none`).

ii.
```python
lick_dir = np.full((go_cue.size,), 2, dtype=np.int64)
...
post_l = lick_l[lick_l >= go_cue[trial_idx] - 1e-9]
post_r = lick_r[lick_r >= go_cue[trial_idx] - 1e-9]

first_l = post_l[0] if post_l.size else np.inf
first_r = post_r[0] if post_r.size else np.inf
if first_l < first_r:
    lick_dir[trial_idx] = 0
elif first_r < first_l:
    lick_dir[trial_idx] = 1
```

iii. The trajectory explicitly states that lick direction would be defined from the "first post-go lick."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from the per-trial `autowater` flag in `obj.bp`.

ii.
```python
"autowater": read_h5_numeric(bp["autowater"]).astype(bool),
...
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
```

iii. The trajectory explicitly says context comes from `autowater` and maps to `WC` versus `DR`.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. `autowater == True` is relabeled to `WC` (`0`), and all other trials become `DR` (`1`).

ii.
```python
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
```

iii. The trajectory states this mapping directly in the final summary.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the per-trial `hit` and `miss` flags, with all remaining trials treated as ignore. The code also reads `no`, but outcome construction does not directly index it.

ii.
```python
"hit": read_h5_numeric(bp["hit"]).astype(bool),
"miss": read_h5_numeric(bp["miss"]).astype(bool),
"no": read_h5_numeric(bp["no"]).astype(bool),

outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)
outcome[beh["miss"]] = 0
outcome[beh["hit"]] = 1
```

iii. The trajectory says outcome comes from `hit/miss/no`, but the actual code implements ignore as the default class for anything not marked hit or miss.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome labels are `incorrect = 0` on miss trials, `correct = 1` on hit trials, and `ignore = 2` otherwise.

ii.
```python
outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)
outcome[beh["miss"]] = 0
outcome[beh["hit"]] = 1
```

iii. The trajectory summary states that the export keeps correct, incorrect, and ignore outcomes.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from `obj.traj` tracking data, using several tongue-related features from both cameras: `tongue`, `left_tongue`, `right_tongue`, `top_tongue`, `topleft_tongue`, `bottom_tongue`, and `bottomleft_tongue`. It also uses per-trial `frameTimes`, `goCue`, and the session-wide video/behavior offset from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`.

ii.
```python
TONGUE_FEATURES = [
    (1, "tongue"),
    (1, "left_tongue"),
    (1, "right_tongue"),
    (2, "top_tongue"),
    (2, "topleft_tongue"),
    (2, "bottom_tongue"),
    (2, "bottomleft_tongue"),
]

vidshift = get_video_offset(mat, beh["bitStart"])
...
x = ts[:, 0, feat_idx]
y = ts[:, 1, feat_idx]
```

iii. The trajectory says the AI intended to use "the same video alignment and feature-velocity logic as the MATLAB code," but the trajectory does not separately justify the expansion from two tongue features to seven.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates each tongue feature's x and y coordinates from camera frame times onto the common exported time grid, marks bins visible when both coordinates are finite, computes velocity with simple gradients, replaces tongue NaN gradients with zeros, converts each feature to speed magnitude, and averages speed across all visible tongue features. It does not apply the reference likelihood cutoff, per-run smoothing, per-view normalization, or frame-to-bin averaging.

ii.
```python
x_interp = interp_with_nans(old_t, x, taxis)
y_interp = interp_with_nans(old_t, y, taxis)
visible = np.isfinite(x_interp) & np.isfinite(y_interp)

is_tongue = "tongue" in feat_name
...
xvel, yvel = compute_velocity(x_proc, y_proc, is_tongue=is_tongue)
speed = np.sqrt(xvel**2 + yvel**2)

visible_counts = per_feature_visible.sum(axis=0)
speed_sum = np.where(per_feature_visible, np.nan_to_num(per_feature_speed), 0.0).sum(axis=0)
composite_speed = np.divide(speed_sum, np.maximum(visible_counts, 1), ...)
```

iii. The trajectory justification is limited to the claim that tongue, paw, and motion used the MATLAB-style video alignment before session-wise discretization; there is no explicit defense of the interpolation-and-average processing actually implemented.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The continuous tongue-speed values are discretized per session with a 50th percentile threshold computed over visible bins from kept trials only. Visible bins below threshold become `0`, visible bins at or above threshold become `1`, and invisible bins become `2`.

ii.
```python
if np.any(keep_visible):
    threshold = float(np.nanpercentile(keep_values[keep_visible], 50))
...
classes = np.full(keep_values.shape, absent_code, dtype=np.int64)
if threshold is not None:
    present = keep_visible
    classes[present] = (keep_values[present] >= threshold).astype(np.int64)
```

iii. The trajectory explicitly says the movement variables use "per-session median discretization."

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Each camera frame time is shifted by a session-wide offset from bitcode timing and then shifted by the trial's `goCue`; the resulting time series is interpolated directly onto the common output time axis used for neural data.

ii.
```python
def get_video_offset(mat: h5py.File, bit_start: np.ndarray) -> float:
    fs = float(read_h5_numeric(mat["obj"]["sglx"]["fs"]))
    bitcode_starts = read_h5_numeric(mat["obj"]["sglx"]["bitcode"]["bitstart"]).astype(np.float64)
    return matlab_mode(bitcode_starts) / fs - matlab_mode(bit_start)

old_t = frame_times - vidshift - align_time
x_interp = interp_with_nans(old_t, x, taxis)
```

iii. The trajectory shows the AI inspected `findVideoOffset.m` and repeatedly said it was aligning video to `goCue` with the shared export grid.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from two bottom-camera features in `obj.traj`: `top_paw` and `bottom_paw`, together with `frameTimes`, `goCue`, and the session video offset.

ii.
```python
PAW_FEATURES = [
    (2, "top_paw"),
    (2, "bottom_paw"),
]

paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
```

iii. The trajectory only says paw used the same video alignment and feature-velocity logic as the MATLAB code; it does not separately justify averaging two paws instead of using only `top_paw`.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI uses the same interpolation-based feature pipeline as for the tongue, but for non-tongue features it nearest-fills missing coordinates before taking gradients and subtracts a baseline derivative estimated from x-position differences. The final paw speed is the average over whichever paw features are visible at each time bin.

ii.
```python
if not is_tongue:
    x_proc = fill_nearest_1d(x_proc)
    y_proc = fill_nearest_1d(y_proc)

...
basederiv_x = np.nanmedian(np.diff(np.column_stack([xpos, ypos]), axis=0)[:, 0])
...
xvel = xvel - basederiv_x
yvel = yvel - basederiv_x
```

iii. The trajectory does not provide a specific justification for the baseline subtraction or for combining `top_paw` and `bottom_paw`; that rationale has to be inferred from the implementation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is discretized the same way as tongue speed: session-wise median split over visible bins from kept trials, with invisible bins assigned code `2`.

ii.
```python
paw_classes, paw_threshold = discretize_with_visibility(
    paw_speed, paw_visible, valid_trials, absent_code=2
)
```

iii. The trajectory explicitly says movement outputs use session-wise median discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw tracking is aligned by subtracting the session video offset and each trial's go cue from frame times, then interpolating onto the same output time axis used for neural data.

ii.
```python
vidshift = get_video_offset(mat, beh["bitStart"])
...
old_t = frame_times - vidshift - align_time
x_interp = interp_with_nans(old_t, x, taxis)
```

iii. The trajectory justification is the same as for tongue velocity: shared bitcode-based video alignment to the `goCue` timebase.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the per-session `motionEnergy_<session>.mat` file, specifically `me.data`, and aligned using side-camera `frameTimes` from `obj.traj`, plus `goCue` and the video offset.

ii.
```python
motion_file = sio.loadmat(motion_path, squeeze_me=True, struct_as_record=False)
me = motion_file["me"]
motion_trials = np.asarray(me.data, dtype=object).reshape(-1)

bundle = load_trial_video_bundle(mat, traj_groups[0], trial_idx)
...
old_t = frame_times - vidshift - beh["goCue"][trial_idx]
```

iii. The trajectory says motion used the same video alignment as the tracking signals and that motion energy came from the separate motion-energy files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI linearly interpolates each trial's motion-energy trace from frame times onto the common exported time axis. It marks interpolated finite values as visible and then nearest-fills those interpolated values across gaps before storing them.

ii.
```python
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
visible[trial_idx, :] = np.isfinite(trial_aligned)
if np.any(visible[trial_idx, :]):
    aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)
```

iii. The trajectory does not give a separate justification for interpolation or nearest-filling; it only claims the same high-level video-alignment logic as the MATLAB pipeline.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is discretized with the same helper as the other movement signals: per-session median threshold over visible bins from kept trials, with absent bins assigned code `2`.

ii.
```python
motion_classes, motion_threshold = discretize_with_visibility(
    motion_energy, motion_visible, valid_trials, absent_code=2
)
```

iii. The trajectory states that motion energy also uses per-session median discretization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion-energy frame times are aligned by subtracting the bitcode-derived session offset and the trial `goCue`, then interpolating onto the shared neural/input/output time axis.

ii.
```python
vidshift = get_video_offset(mat, beh["bitStart"])
...
old_t = frame_times - vidshift - beh["goCue"][trial_idx]
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
```

iii. The trajectory explicitly says motion signals were aligned to `goCue` using the same video-alignment logic as the other camera-derived outputs.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly fills or substitutes rather than preserving gaps. If `frameTimes` are missing or all-NaN, it fabricates a 400 Hz timeline; if non-tongue coordinates are missing after interpolation, it nearest-fills them before differentiating; if motion energy has gaps after interpolation, it nearest-fills there too. If an entire stream is missing, the output bins are assigned the absent code.

ii.
```python
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

if not is_tongue:
    x_proc = fill_nearest_1d(x_proc)
    y_proc = fill_nearest_1d(y_proc)

if np.any(visible[trial_idx, :]):
    aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)
```

iii. The trajectory does not contain an explicit discussion of missing-data philosophy; this behavior is visible in the code rather than spelled out in the agent's written justification.

## 11-a. What are the most time-consuming steps of the code?

i. The most expensive parts are likely the per-unit/per-trial spike histogram-and-smoothing loop in `load_neural_session` and the repeated per-trial/per-feature interpolation and gradient calculations in `compute_composite_speed`, plus per-trial interpolation of motion energy.

ii.
```python
for out_idx, unit_idx in enumerate(keep_unit_indices):
    ...
    for tr in np.unique(spike_trials):
        counts, _ = np.histogram(spk, bins=edges)
        trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)
```

```python
for trial_idx in range(ntrials):
    ...
    for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
        ...
        x_interp = interp_with_nans(old_t, x, taxis)
        y_interp = interp_with_nans(old_t, y, taxis)
```

iii. In the trajectory, the AI described the long-running parts as rebuilding "smoothed trial-by-trial firing rates and aligned video features session by session."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could have been reduced or consolidated: the nested unit/trial loop for spike binning, the trial/feature loops in `compute_composite_speed`, the second pass over conditions to estimate `mean_frs`, and the separate pass over trials for motion-energy interpolation.

ii.
```python
for out_idx, unit_idx in enumerate(keep_unit_indices):
    ...
    for tr in np.unique(spike_trials):
        ...

for trial_idx in range(ntrials):
    ...
    for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
        ...

for mask in condition_masks:
    if np.any(mask):
        psth = trialdat[:, mask, :].mean(axis=1)
```

iii. The trajectory does not contain an explicit vectorization analysis, but the implemented code leaves multiple obvious Python-level loops in the hot path.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several session-level computations: `get_time_axis()` is called in multiple helpers, and `get_video_offset`, `load_traj_groups`, and `get_feature_maps` are recomputed separately for tongue, paw, and motion energy instead of being shared once per session.

ii.
```python
def compute_composite_speed(...):
    taxis = get_time_axis() + ADVANCE_MOVEMENT
    vidshift = get_video_offset(mat, beh["bitStart"])
    traj_groups = load_traj_groups(mat)
    feature_maps = get_feature_maps(mat, traj_groups, ntrials)

def load_motion_energy_aligned(...):
    taxis = get_time_axis() + ADVANCE_MOVEMENT
    vidshift = get_video_offset(mat, beh["bitStart"])
    traj_groups = load_traj_groups(mat)
```

iii. The trajectory does not acknowledge this repetition; it only says the code follows the same high-level alignment logic for each movement stream.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes and returns `sites` from spike clusters but never uses them in the exported dataset. It also builds condition-specific PSTHs only to collapse them immediately into a mean-rate filter, and it loads several behavior fields (`L`, `sample`, `delay`) that do not affect the final output arrays.

ii.
```python
sites = np.zeros((n_units_pre,), dtype=np.int64)
...
if site_refs is not None:
    sites[out_idx] = int(read_trial_ref_numeric(mat, site_refs[unit_idx]))
...
mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
return trialdat[use], sites[use]
```

iii. No explicit justification for this extra work appears in the trajectory; these are implementation leftovers visible from the code structure.
