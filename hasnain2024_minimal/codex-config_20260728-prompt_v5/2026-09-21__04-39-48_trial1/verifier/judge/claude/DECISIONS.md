# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions (the "Figure 8 two-context cohort") rather than all 44 sessions present in the data. Sessions are hard-coded in `SESSION_SPECS` as `SessionSpec` dataclass instances specifying animal, date, folder, and probe. All files are loaded via `h5py` (HDF5/v7.3 format only); motion energy files are loaded via `scipy.io.loadmat`. There is no fallback for v5-format data structure files.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", "Ephys_Behavior", 2),
    SessionSpec("JEB7", "2021-04-29", "Ephys_Behavior", 1),
    SessionSpec("JEB7", "2021-04-30", "Ephys_Behavior", 1),
    SessionSpec("EKH1", "2021-08-07", "Ephys_Behavior", 2),
    SessionSpec("EKH3", "2021-08-11", "Ephys_Behavior", 2),
    SessionSpec("JGR2", "2021-11-16", "Ephys_Behavior", 1),
    SessionSpec("JGR2", "2021-11-17", "Ephys_Behavior", 1),
    SessionSpec("JGR3", "2021-11-18", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-18", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-19", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-20", "Ephys_Behavior", 1),
    SessionSpec("JEB19", "2023-04-21", "Ephys_Behavior", 1),
]
```

```python
def process_session(spec: SessionSpec):
    with h5py.File(spec.data_path, "r") as h5:
        bp = load_bp_fields(h5)
        ...
```

iii. The AI's trajectory shows it deliberately narrowed to the Figure 8 cohort: "The context cohort is now clear: the Figure 8 loader set gives exactly 12 sessions." It chose this subset because these sessions have substantial WC blocks, excluding DR-only and randomized-delay sessions.

## 1-b. How are the data split into subjects?

i. The animal name comes from the `SessionSpec.animal` field. Subjects are accumulated in insertion order (not sorted) into a list, with a lookup dictionary mapping animal name to index.

ii.
```python
subjects = []
subject_lookup = {}
subject_idx = []
for spec in SESSION_SPECS:
    if spec.animal not in subject_lookup:
        subject_lookup[spec.animal] = len(subjects)
        subjects.append(spec.animal)
    subject_idx.append(subject_lookup[spec.animal])
```

iii. The animal name is embedded in the session spec, derived from the filename convention `<animal>_<date>`.

## 1-c. How are the data split into sessions?

i. One session corresponds to one `SessionSpec` entry and one `.mat` file. Only 12 sessions from the fixed-delay `Ephys_Behavior` folder are included. No randomized-delay sessions are used.

ii.
```python
sessions = [process_session(spec) for spec in SESSION_SPECS]
```

iii. The AI chose the Figure 8 two-context cohort only, reasoning these are the sessions with both WC and DR contexts.

## 1-d. How are the data split into trials?

i. The number of trials is read from `bp.Ntrials`. All behavioral arrays are truncated to this length. Trials are indexed by their position (0-based). Valid trials are those passing the early-lick and photostim filter.

ii.
```python
"Ntrials": int(np.asarray(bp["Ntrials"])[0, 0]),
...
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

iii. Trials are defined by the Bpod table rows, consistent with the reference approach.

## 1-e. How are trials filtered based on quality controls?

i. Two filters are applied: early-lick trials (`bp.early`) and photostimulation trials (`bp.stim.enable`) are excluded. Unlike the reference, trials that extend past the end of the recording are NOT explicitly checked or dropped.

ii.
```python
valid_mask = (~bp["early"]) & (~bp["stim_enable"])
valid_trials = np.flatnonzero(valid_mask)
```

iii. The AI's trajectory mentions filtering with `~early & ~stim.enable`, following the paper's exclusion criteria.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` — the spike-sorted clusters. Each cluster has `trial` (1-based trial index per spike), `trialtm` (spike time relative to trial start), and `quality` (curation label). `bp.ev.goCue` provides the go cue time for alignment.

ii.
```python
clusters.append({
    "quality": quality,
    "trial": read_ref_array(h5, probe_group["trial"][clu_idx, 0]).astype(np.int64),
    "trialtm": read_ref_array(h5, probe_group["trialtm"][clu_idx, 0]).astype(np.float64),
})
```

iii. The AI identified these as the relevant fields by inspecting the HDF5 structure.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue (`trialtm - goCue[trial-1]`), then histogrammed into 10 ms bins over [-3.0, 2.5] s (550 bins). Counts are converted to Hz by dividing by `DT`. Smoothing uses a **causal** Gaussian kernel: `gausswin(15)` with the first half zeroed out, applied via convolution with `reflect` boundary conditions. The smoothing is applied to condition-averaged PSTHs for the firing rate filter, then again to per-trial rates.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
SMOOTH_WINDOW = 15
BCTYPE = "reflect"

def my_smooth(x, n, bctype="none"):
    ...
    kernel = gausswin(n)
    kernel[: len(kernel) // 2] = 0.0  # causal: zero out first half
    kernel /= kernel.sum()
    ...

cluster_trials = counts[valid_mask] / DT
cluster_trials = my_smooth(cluster_trials.T, SMOOTH_WINDOW, BCTYPE).T
```

iii. The AI stated it matched "the repository's spike binning/smoothing" and used "10 ms bins, causal Gaussian smoothing."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters. First, clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (note: "poor" is NOT excluded, unlike the reference). Second, the mean firing rate is computed from condition-averaged PSTHs (not overall mean rate), and units with mean FR <= 1 Hz are excluded.

ii.
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
...
quality = str(cluster["quality"]).strip().lower()
if quality in QUALITY_EXCLUDE:
    continue
...
# Mean FR computed from condition-averaged PSTHs
psths = []
for cond_mask in condition_masks:
    ...
    psth = counts[cond_mask].sum(axis=0) / n_cond / DT
    psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
    psths.append(psth)
mean_fr = float(np.mean(np.stack(psths, axis=1)))
if mean_fr <= LOW_FR_HZ:
    continue
```

iii. The AI mentioned "all non-garbage/non-noisy units with mean firing rate > 1 Hz" in its metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned to the go cue by subtracting `goCue[trial-1]` from `trialtm`, then histogrammed into time bins. This is the same approach as the reference.

ii.
```python
aligned_times = trialtm - bp["goCue"][trial_ids - 1]
counts = align_and_histogram(trial_ids, aligned_times, n_trials)
```

iii. The AI identified go cue alignment from the paper and instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 10 ms (`DT = 0.01`), and the time window is [-3.0, 2.5] s, giving 550 time bins. No rebinning is applied — spikes are directly counted into these bins.

ii.
```python
DT = 0.01
TMIN = -3.0
TMAX = 2.5
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

iii. The AI stated it used "10 ms bins over [-3.0, 2.5] s" matching the repository's `params.dt = 1/200` but incorrectly — `1/200 = 0.005` s = 5 ms, not 10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is defined by the time bin centers, computed from the bin edges. It is not derived from any raw data variable — it is the time axis itself.

ii.
```python
TIME = EDGES[:-1] + (DT / 2.0)
inputs = [TIME[np.newaxis, :].astype(np.float32).copy() for _ in valid_trials]
```

iii. Same approach as reference — the time axis serves as the decoder input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing the bin centers from the edges.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same bin grid used for binning neural spikes, so alignment is automatic.

ii.
```python
EDGES = np.arange(TMIN, TMAX + 1e-9, DT)
TIME = EDGES[:-1] + (DT / 2.0)
```

iii. Both neural and input share the same time grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` (right instruction), `bp.L` (left instruction), `bp.hit`, `bp.miss`, and `bp.no`. The logic checks instructed side and outcome to infer lick direction.

ii.
```python
def trial_choice_code(bp, trial_idx):
    if bp["no"][trial_idx]:
        return 2  # none
    if (bp["R"][trial_idx] and bp["hit"][trial_idx]) or (bp["L"][trial_idx] and bp["miss"][trial_idx]):
        return 1  # right
    if (bp["L"][trial_idx] and bp["hit"][trial_idx]) or (bp["R"][trial_idx] and bp["miss"][trial_idx]):
        return 0  # left
```

iii. The AI infers lick direction from the combination of instructed side and outcome, same logic as reference.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit means the animal licked the instructed side; a miss means it licked the opposite side; `no` (ignore) means no lick. Codes: left=0, right=1, none=2. This matches the reference encoding.

ii. Same as 4-a code snippet.

iii. Straightforward relabeling based on trial outcome and instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. Autowater=True indicates WC context; otherwise DR.

ii.
```python
context_code = 0 if bp["autowater"][trial_idx] else 1
```

iii. The AI correctly identified `autowater` as the WC-context flag.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabeling: autowater → WC (0), non-autowater → DR (1).

ii.
```python
context_code = 0 if bp["autowater"][trial_idx] else 1
```

iii. Matches reference encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
def trial_outcome_code(bp, trial_idx):
    if bp["miss"][trial_idx]:
        return 0  # incorrect
    if bp["hit"][trial_idx]:
        return 1  # correct
    if bp["no"][trial_idx]:
        return 2  # ignore
```

iii. The AI uses all three flags explicitly (hit, miss, no), while the reference derives ignore implicitly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct relabeling: miss → incorrect (0), hit → correct (1), no → ignore (2).

ii. Same as 6-a.

iii. Matches the prompt's specified encoding.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from `obj.traj` tracking data. The AI uses multiple tongue-related features from both cameras: side view features `["tongue", "left_tongue", "right_tongue"]` and bottom view features `["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]`. This contrasts with the reference which uses only `tongue` (side) and `top_tongue` (bottom).

ii.
```python
tongue_indices = {
    0: [feat_names[0].index(name) for name in ["tongue", "left_tongue", "right_tongue"] if name in feat_names[0]],
    1: [feat_names[1].index(name) for name in ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]
        if name in feat_names[1]],
}
```

iii. The AI inspected the DLC feature names and decided to use all tongue-related features from both cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Raw x,y coordinates are extracted for each feature. (2) Coordinates are **interpolated** onto the 10ms time grid using `interp1d` (linear). (3) Speed is computed as the magnitude of `np.gradient` of the interpolated coordinates. NaN values in the velocity are set to 0. (4) Speeds from multiple features are averaged where visible. (5) The session's 50th percentile is used to threshold into categories (0: below, 1: above, 2: not visible).

No likelihood filtering is applied — the AI does not filter by the DLC likelihood threshold (0.9). Instead, it relies on NaN values in the coordinates to determine visibility.

ii.
```python
def interpolate_coords(coords, frame_times, align_time, vidshift):
    interp = interp1d(frame_times - vidshift - align_time, coords, axis=0,
                      kind="linear", bounds_error=False, fill_value=np.nan)
    return np.asarray(interp(TIME), dtype=np.float64)

def feature_speed(coords_interp, tongue_feature):
    visible = np.isfinite(coords_interp).all(axis=1)
    if tongue_feature:
        vel = np.gradient(coords_interp, axis=0)
        vel[~np.isfinite(vel)] = 0.0
    ...
    speed = np.sqrt((vel ** 2).sum(axis=1))
    return speed, visible
```

iii. The AI adopted interpolation onto the time grid rather than binning frame-resolution velocities.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile of all visible tongue velocity values across the session is used as the threshold. Values below → 0, values at or above → 1, not visible → 2.

ii.
```python
tongue_thresh = float(np.nanpercentile(tongue_series[tongue_visible], 50))
trial_output[3] = np.where(~tongue_visible[out_idx], 2,
                           (tongue_series[out_idx] >= tongue_thresh).astype(np.int64))
```

iii. Matches the instructions' specified thresholding scheme.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from bitcode timestamps (`sglx.bitcode.bitstart / fs - mode(bp.ev.bitStart)`), same approach as reference. Frame times are corrected by this offset and the trial's go cue time. However, instead of binning frame-resolution values, the AI **interpolates** the raw coordinates onto the neural time grid, then computes velocity on the interpolated grid.

ii.
```python
def get_video_shift(h5, bp):
    fs = float(np.asarray(h5["obj"]["sglx"]["fs"]).reshape(-1)[0])
    bitstart = np.asarray(h5["obj"]["sglx"]["bitcode"]["bitstart"]).reshape(-1)
    return matlab_mode(bitstart) / fs - matlab_mode(bp["bitStart"])

def interpolate_coords(coords, frame_times, align_time, vidshift):
    interp = interp1d(frame_times - vidshift - align_time, coords, ...)
    return np.asarray(interp(TIME), dtype=np.float64)
```

iii. The AI reproduced the reference's `findVideoOffset.m` logic for the offset computation.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from `obj.traj` bottom camera tracking. The AI uses **both** `top_paw` and `bottom_paw` features, whereas the reference uses only `top_paw`.

ii.
```python
paw_indices = {
    1: [feat_names[1].index(name) for name in ["top_paw", "bottom_paw"] if name in feat_names[1]],
}
```

iii. The AI included both paw features visible in the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For paw features, the AI applies a different processing path than tongue: (1) Coordinates are interpolated onto the time grid. (2) Missing values are filled using `fill_nearest` (nearest-neighbor interpolation). (3) Velocity is computed via `np.gradient`, then a baseline drift correction is applied by subtracting the median derivative. (4) The result is filled again with `fill_nearest`. (5) Speeds from multiple features are **averaged** (not aggregated by visibility). (6) Thresholded at session 50th percentile.

ii.
```python
def feature_speed(coords_interp, tongue_feature):
    ...
    else:  # paw
        coords_filled = fill_nearest(coords_interp)
        vel = np.gradient(coords_filled, axis=0)
        base_deriv = np.nanmedian(np.diff(coords_filled, axis=0), axis=0)
        vel[:, 0] = vel[:, 0] - base_deriv[0]
        vel[:, 1] = vel[:, 1] - base_deriv[0]  # note: uses base_deriv[0] for both dims
        vel = fill_nearest(vel)
    speed = np.sqrt((vel ** 2).sum(axis=1))
```

iii. The AI applied nearest-fill and baseline correction for paw, treating it differently from tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile of visible paw velocity values across the session is used as threshold. Below → 0, at or above → 1, not visible → 2.

ii.
```python
paw_thresh = float(np.nanpercentile(paw_series[paw_visible], 50))
trial_output[4] = np.where(~paw_visible[out_idx], 2,
                           (paw_series[out_idx] >= paw_thresh).astype(np.int64))
```

iii. Matches instruction scheme.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same interpolation approach as tongue: raw coordinates are interpolated onto the neural time grid after applying the video offset correction.

ii. Same `interpolate_coords` function as tongue (see 7-d).

iii. Same alignment method for all video-derived outputs.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<animal>_<date>.mat` files via `scipy.io.loadmat`. The `me.data` field provides per-trial traces and `me.moveThresh` provides a threshold (though it is not used in the final discretization).

ii.
```python
def read_motion_energy(path):
    loaded = loadmat(path, squeeze_me=True, struct_as_record=False)
    me = loaded["me"]
    data = me.data
    if not isinstance(data, np.ndarray) and hasattr(data, "data"):
        data = data.data
    return data, float(me.moveThresh)
```

iii. The AI identified the motion energy companion files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The per-trial motion energy trace is **interpolated** onto the neural time grid using `interp1d` (linear), then `fill_nearest` is applied to fill any remaining NaN values. This contrasts with the reference which simply bins frame-resolution values. The 50th percentile threshold is then applied.

ii.
```python
def interpolate_motion_energy(me_trace, frame_times, align_time, vidshift):
    interp = interp1d(frame_times - vidshift - align_time, me_trace,
                      kind="linear", bounds_error=False, fill_value=np.nan)
    return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The AI used interpolation + nearest fill rather than simple binning.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile of visible motion energy values across the session is the threshold. Below → 0, at or above → 1, no video → 2.

ii.
```python
me_thresh = float(np.nanpercentile(me_series[me_visible], 50))
trial_output[5] = np.where(~me_visible[out_idx], 2,
                           (me_series[out_idx] >= me_thresh).astype(np.int64))
```

iii. Matches instruction scheme.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The side camera's frame times are used (corrected by video offset and go cue). Motion energy is interpolated onto the neural time grid rather than binned.

ii.
```python
def interpolate_motion_energy(me_trace, frame_times, align_time, vidshift):
    ...
    interp = interp1d(frame_times - vidshift - align_time, me_trace, ...)
    return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. Same interpolation approach as other video streams.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) If a trial's video data has no valid frames (`NdroppedFrames` is empty/NaN), the trial returns `None` for ts and frame_times. (2) If frame_times are missing or all NaN, synthetic frame times are generated at 400 Hz: `np.arange(1, n+1) / 400.0`. (3) For paw and motion energy, `fill_nearest` interpolation fills NaN gaps. (4) For tongue, NaN velocities are set to 0. (5) If `frame_times.size != me_trace.size`, synthetic frame times are generated.

This is notably different from the reference, which keeps NaN values and marks gaps as "not visible" without interpolation.

ii.
```python
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0

# In feature_speed for tongue:
vel[~np.isfinite(vel)] = 0.0

# In feature_speed for paw:
coords_filled = fill_nearest(coords_interp)

# In interpolate_motion_energy:
return fill_nearest(np.asarray(interp(TIME), dtype=np.float64))
```

iii. The AI chose to interpolate and fill missing values rather than preserving NaN gaps.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading HDF5 files — each session is opened twice (once for neural, once for behavioral outputs). (2) Per-trial video interpolation and feature extraction, which involves opening the HDF5 file again and reading per-trial trajectory data.

ii.
```python
def process_session(spec):
    with h5py.File(spec.data_path, "r") as h5:
        ...
    outputs, thresholds = compute_behavioral_outputs(spec, bp, valid_trials)
    # compute_behavioral_outputs opens the file again:
    with h5py.File(spec.data_path, "r") as h5:
        ...
```

iii. The dual file opening and per-trial HDF5 reads are the dominant costs.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial video processing loop in `compute_behavioral_outputs` iterates over every valid trial, reading and interpolating video data one trial at a time. The per-cluster loop in `compute_neural_trials` processes each cluster individually. The condition PSTH computation loops over 7 condition masks per cluster.

ii.
```python
for trial_idx in valid_trials:
    ...
    for feat_idx in tongue_indices[0]:
        ...
    for feat_idx in tongue_indices[1]:
        ...

for cluster in clusters:
    ...
    for cond_mask in condition_masks:
        ...
```

iii. The per-trial video loop is the main candidate for vectorization, though the variable frame counts per trial make this difficult.

## 11-c. What processing does the code repeat multiple times?

i. The session's HDF5 file is opened twice — once in `process_session` for neural data and once in `compute_behavioral_outputs` for video data. The condition masks are computed once via `make_condition_masks` but the PSTH smoothing is done per-cluster (7 conditions × N clusters). Per-trial smoothing is also applied separately from the PSTH smoothing.

ii.
```python
# First open in process_session:
with h5py.File(spec.data_path, "r") as h5:
    bp = load_bp_fields(h5)
    clusters = load_probe_clusters(h5, spec.probe)
    neural_trials, n_neurons = compute_neural_trials(bp, clusters, valid_mask)

# Second open in compute_behavioral_outputs:
with h5py.File(spec.data_path, "r") as h5:
    vidshift = get_video_shift(h5, bp)
    view_groups, feat_names = load_view_info(h5)
```

iii. The double file open is unnecessary; both could share a single context.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes condition-averaged PSTHs (7 conditions per cluster) solely for the purpose of computing a mean firing rate for the low-FR filter. These PSTHs themselves are never used in the output. The `make_condition_masks` function creates 7 condition masks that are not needed for the final output. The `moveThresh` from motion energy files is loaded but never used.

ii.
```python
psths = []
for cond_mask in condition_masks:
    n_cond = int(cond_mask.sum())
    if n_cond == 0:
        psth = np.zeros(TIME.size, dtype=np.float64)
    else:
        psth = counts[cond_mask].sum(axis=0) / n_cond / DT
        psth = my_smooth(psth, SMOOTH_WINDOW, BCTYPE)
    psths.append(psth)
mean_fr = float(np.mean(np.stack(psths, axis=1)))
# psths are discarded after this

return data, float(me.moveThresh)  # moveThresh never used
```

iii. The condition-averaged PSTH computation is a significant amount of work that only serves the firing rate filter and could be replaced by a simpler overall mean rate calculation.
