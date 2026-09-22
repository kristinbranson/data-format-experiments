# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, hard-coded in a `SESSIONS` list. It treats these as the "published DR+WC ALM-video cohort." Each session is a MATLAB v7.3 HDF5 file opened with `h5py`. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`. The AI does NOT load sessions from the `RandomizedDelay_Ephys_Behavior` folder.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

SESSIONS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    {"subject": "JEB7", "date": "2021-04-29", "probe": 1},
    {"subject": "JEB7", "date": "2021-04-30", "probe": 1},
    {"subject": "EKH1", "date": "2021-08-07", "probe": 2},
    {"subject": "EKH3", "date": "2021-08-11", "probe": 2},
    {"subject": "JGR2", "date": "2021-11-16", "probe": 1},
    {"subject": "JGR2", "date": "2021-11-17", "probe": 1},
    {"subject": "JGR3", "date": "2021-11-18", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-21", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-20", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-19", "probe": 1},
    {"subject": "JEB19", "date": "2023-04-18", "probe": 1},
]
```

```python
with h5py.File(data_path, "r") as h5file:
    bp = h5file["obj/bp"]
    ...
```

iii. The agent stated: "I've pinned down the cohort split from the authors' own scripts: the first 12 `Ephys_Behavior` ALM-video sessions are the DR+WC recordings." The agent identified these 12 sessions as the two-context cohort and excluded the RandomizedDelay sessions and other Ephys_Behavior sessions (JEB13, JEB14, JEB15 multi-probe sessions).

## 1-b. How are the data split into subjects?

i. The subject is taken from the `subject` field of the hard-coded `SESSIONS` list. Subjects are accumulated in insertion order as sessions are processed, and a `subject_to_idx` mapping is built incrementally. The result is 7 unique subjects from the 12 sessions.

ii.
```python
for session_info in SESSIONS:
    session_data, session_meta = convert_session(session_info)
    subject = str(session_meta["subject"])
    if subject not in subject_to_idx:
        subject_to_idx[subject] = len(subjects)
        subjects.append(subject)
```

iii. The agent did not provide specific reasoning for this choice; it follows directly from the session list structure.

## 1-c. How are the data split into sessions?

i. One session is one entry in the `SESSIONS` list. Each session maps to one `.mat` file in the `Ephys_Behavior` folder. Only 12 sessions are included (all from the fixed-delay task folder). The `RandomizedDelay_Ephys_Behavior` folder is not used at all.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")
# ...
for session_info in SESSIONS:
    session_data, session_meta = convert_session(session_info)
```

iii. The agent identified these 12 sessions as the "published DR+WC ALM-video cohort" based on the authors' loading scripts, specifically focusing on the two-context (WC + DR) subset rather than all available sessions.

## 1-d. How are the data split into trials?

i. Trials are indexed by the behavioral fields in `obj.bp`. The code reads boolean arrays like `hit`, `miss`, `R`, `L`, `early`, `stim.enable`, and `autolearn` from the Bpod structure. A mask is created to identify kept trials, and `keep_trials` is the array of indices of valid trials.

ii.
```python
keep_mask = (~stim_enable) & (~early) & (~autolearn)
keep_trials = np.flatnonzero(keep_mask)
```

iii. The agent described filtering as: "Trials are filtered to control/non-early trials (~stim.enable & ~early, plus ~autolearn if present)."

## 1-e. How are trials filtered based on quality controls?

i. Three filters are applied: (1) photostimulation trials (`stim.enable`) are excluded, (2) early-lick trials (`early`) are excluded, and (3) `autolearn` trials are excluded if the field exists. Unlike the reference, the AI does NOT filter trials that run past the end of the recording. The AI also adds the `autolearn` filter, which the reference does not use.

ii.
```python
stim_enable = as_bool_1d(bp["stim/enable"])
early = as_bool_1d(bp["early"])
autolearn = (
    as_bool_1d(bp["autolearn"]) if "autolearn" in bp.keys() else np.zeros_like(early, dtype=bool)
)
keep_mask = (~stim_enable) & (~early) & (~autolearn)
keep_trials = np.flatnonzero(keep_mask)
```

iii. The agent stated the trial filter as "~stim.enable & ~early & ~autolearn". The agent did not mention filtering for trials past the end of the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu` (spike-sorted clusters). For each cluster, the code reads `trial` (spike trial indices, 1-based), `trialtm` (spike times relative to trial start), and `quality` (curation label). The go cue times `bp.ev.goCue` are used for temporal alignment.

ii.
```python
clu_group = h5file[h5file["obj/clu"][probe_num - 1, 0]]
qualities = clu_group["quality"]
trials = clu_group["trial"]
trial_times = clu_group["trialtm"]
# ...
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The agent described using "goCue alignment, 5 ms bins, causal Gaussian smoothing."

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue by subtraction, then binned into 5 ms bins spanning -3.0 to 2.5 s (1100 bins). Spike counts are converted to firing rates by dividing by bin width (DT = 0.005). The rates are then smoothed using a **causal** Gaussian kernel: a `gausswin(15)` with the first half zeroed out and reflected leading padding. This differs from the reference's symmetric Gaussian smoothing.

ii.
```python
def build_smoothing_kernel(n: int = SMOOTH_N) -> np.ndarray:
    x = np.arange(1, n + 1, dtype=np.float64)
    sigma = np.std(x)
    kernel = np.exp(-0.5 * ((x - (n + 1) / 2.0) / sigma) ** 2)
    kernel[: n // 2] = 0.0   # make it causal
    kernel /= kernel.sum()
    return kernel

def smooth_causal_reflect(x, kernel=SMOOTH_KERNEL):
    # pads with reflected leading boundary, convolves
    padded = np.concatenate([x[:, :n], x], axis=1)
    out[i] = np.convolve(padded[i], kernel, mode="same")
    out = out[:, n:]
```

```python
rates = smooth_causal_reflect(counts / DT).astype(np.float32)
```

iii. The agent stated: "causal Gaussian smoothing (N=15) with reflected leading padding."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels `garbage`, `gabrga`, `noisy`, or `real?` are excluded. Unlike the reference, `poor` is NOT excluded. Quality labels are matched as-is (not lower-cased). (2) Units with mean firing rate <= 1 Hz are excluded. Additionally, sessions with fewer than 10 surviving units raise a RuntimeError and would be skipped.

ii.
```python
quality = str(read_matlab_any(h5file, qualities[clu_idx, 0]) or "").strip()
if quality in {"garbage", "gabrga", "noisy", "real?"}:
    continue
# ...
mean_fr = float(rates.mean())
if mean_fr > LOW_FR_HZ:
    kept_unit_data.append(rates)

if len(kept_unit_data) < 10:
    raise RuntimeError(...)
```

iii. The agent stated removing "garbage/noisy clusters" and keeping units with "mean firing rate > 1 Hz."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times (`trialtm`) are aligned to the go cue by subtracting `goCue[trial]` from each spike time. This is the same approach as the reference.

ii.
```python
clu_trial_times = as_1d_float(read_matlab_any(h5file, trial_times[clu_idx, 0]))
aligned_times = clu_trial_times - go_cue[clu_trials]
```

iii. The agent described "goCue alignment."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 5 ms (DT = 1/200), matching the reference. However, the time window is -3.0 to 2.5 s (1100 bins), whereas the reference uses -2.5 to 2.5 s (1000 bins). The extra 0.5 s before the go cue means more data is included on the pre-cue side.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1.0 / 200.0
TIME_EDGES = np.arange(TMIN, TMAX + DT, DT, dtype=np.float64)
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
NT = TIME_BINS.size
```

iii. The agent stated: "bins neural activity at 5 ms on [-3.0, 2.5] s."

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the center of each time bin in the binning grid, relative to the go cue. It is constructed from the time edges, not from any raw data variable.

ii.
```python
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
input_trials = [TIME_BINS[np.newaxis, :].astype(np.float32) for _ in range(keep_trials.size)]
```

iii. No specific justification needed; this is a derived quantity.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is defined by the binning grid. No processing beyond computing bin centers.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis IS the neural binning grid itself. Bin centers are used, matching the neural bins exactly.

ii.
```python
TIME_BINS = TIME_EDGES[:-1] + DT / 2.0
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. `bp.R`, `bp.L`, `bp.hit`, and `bp.miss`. The instructed side (`R` or `L`) combined with outcome (`hit` or `miss`) determines the actual lick direction.

ii.
```python
r = as_bool_1d(bp["R"])
l = as_bool_1d(bp["L"])
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
```

iii. The agent stated: "Lick direction is derived from R/L together with hit/miss/no."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. A hit on an R trial means licked right, a miss on an R trial means licked left (and vice versa). The logic: `right_choice = (r & hit) | (l & miss)`, `left_choice = (l & hit) | (r & miss)`. Everything else is "none" (code 2). Codes are left=0, right=1, none=2.

ii.
```python
lick_direction = np.full(keep_trials.size, 2, dtype=np.int64)
right_choice = (r & hit) | (l & miss)
left_choice = (l & hit) | (r & miss)
lick_direction[left_choice[keep_trials]] = 0
lick_direction[right_choice[keep_trials]] = 1
```

iii. The agent described the lick direction derivation from R/L combined with hit/miss/no.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. `bp.autowater`. Autowater trials are WC context, the rest are DR context.

ii.
```python
autowater = as_bool_1d(bp["autowater"])
context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)  # WC, DR
```

iii. The agent stated: "context from autowater."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct relabelling: autowater=True maps to WC (0), autowater=False maps to DR (1). This matches the reference.

ii.
```python
context = np.where(autowater[keep_trials], 0, 1).astype(np.int64)
```

iii. N/A

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss`. Trials that are neither are classified as "ignore."

ii.
```python
hit = as_bool_1d(bp["hit"])
miss = as_bool_1d(bp["miss"])
```

iii. The agent stated: "outcome from hit/miss/no."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit trials are "correct" (1), miss trials are "incorrect" (0), everything else is "ignore" (2). This matches the reference.

ii.
```python
outcome = np.full(keep_trials.size, 2, dtype=np.int64)
outcome[miss[keep_trials]] = 0
outcome[hit[keep_trials]] = 1
```

iii. N/A

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DeepLabCut tracking in `obj.traj`. The AI uses multiple tongue-related features from each camera view: side view features `["tongue", "left_tongue", "right_tongue"]` and bottom view features `["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"]`. The reference uses only `"tongue"` from side and `"top_tongue"` from bottom. Also uses `frameTimes`, `featNames`, `ts`, and the bitcode fields for video offset.

ii.
```python
side_tongue_speed, side_tongue_visible, _ = get_feature_positions(
    h5file, side_view, int(trial_idx),
    ["tongue", "left_tongue", "right_tongue"],
    align_time, video_shift,
)
bottom_tongue_speed, bottom_tongue_visible, _ = get_feature_positions(
    h5file, bottom_view, int(trial_idx),
    ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
    align_time, video_shift,
)
```

iii. The agent did not specifically justify using multiple tongue features vs. the single feature used in the reference code.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The processing differs significantly from the reference:
1. Position (x, y) for each feature is extracted from `ts`.
2. Positions are **interpolated** onto the 5 ms time grid using `interpolate_visible_segments`, rather than computing velocity at frame resolution and then binning.
3. NaN positions are filled with nearest-neighbor (`fill_nearest_1d`).
4. Velocity is computed as `np.gradient` of the filled, interpolated positions.
5. Speed is the magnitude of the velocity vector.
6. Multiple tongue features from each view are averaged together using `average_over_visible`.
7. Side and bottom views are averaged together.
8. No normalization by percentile before averaging the two views (the reference normalizes each view by its 90th percentile).
9. The result is discretized at the session 50th percentile.

ii.
```python
# Interpolate positions to neural grid
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
# Fill NaNs
filled_xy[:, 0] = fill_nearest_1d(filled_xy[:, 0])
filled_xy[:, 1] = fill_nearest_1d(filled_xy[:, 1])
# Compute velocity
vel = np.gradient(filled_xy, axis=0)
speed = np.sqrt((vel**2).sum(axis=1))
# Average views
tongue_speed[out_trial_idx], tongue_visible[out_trial_idx] = average_over_visible(
    tongue_speeds, tongue_vis
)
# Discretize
tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
```

iii. The agent described "feature positions linearly interpolated onto the neural grid, visibility preserved for tongue/paw categorization."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session-wide 50th percentile of visible tongue speed values is used as the threshold. Values below threshold are 0 ("low"), at or above threshold are 1 ("high"), and not-visible bins are 2 ("not_visible"). This is consistent with the reference approach.

ii.
```python
tongue_threshold = float(np.nanpercentile(tongue_speed[tongue_visible], 50))
tongue_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
tongue_disc[tongue_visible & (tongue_speed < tongue_threshold)] = 0
tongue_disc[tongue_visible & (tongue_speed >= tongue_threshold)] = 1
```

iii. N/A

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed from `sglx.bitcode.bitstart / fs` minus `bp.ev.bitStart`, using **median** (not mode as in the reference). Frame times are corrected by subtracting the offset and the trial's go cue time. Positions are then interpolated directly onto the neural time grid (`TIME_BINS`), so alignment is achieved through interpolation rather than binning.

ii.
```python
def get_video_shift(h5file, bit_start):
    bitcode_start = as_1d_float(h5file["obj/sglx/bitcode/bitstart"])
    fs = float(np.asarray(h5file["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(bitcode_start / fs) - np.nanmedian(bit_start))

# Frame times aligned to go cue
frame_times = frame_times - video_shift - align_time
# Interpolated onto neural grid
interp_xy = interpolate_visible_segments(frame_times, xy, TIME_BINS)
```

iii. The agent described: "Video trajectories aligned with the paper's video offset."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The DLC tracking in `obj.traj`, using features `["top_paw", "bottom_paw"]` from the bottom view. The reference uses only `"top_paw"`.

ii.
```python
paw_speed_trial, paw_visible_trial, _ = get_feature_positions(
    h5file, bottom_view, int(trial_idx),
    ["top_paw", "bottom_paw"],
    align_time, video_shift,
)
```

iii. The agent did not explicitly justify using both paws vs. just one.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue: interpolate positions to neural grid, fill NaN with nearest, compute gradient, compute speed magnitude. Additionally, for non-tongue features (including paw), a **baseline drift subtraction** is applied: the median of the frame-to-frame position difference is subtracted from the velocity. This is not present in the reference.

ii.
```python
vel = np.gradient(filled_xy, axis=0)
baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
if "tongue" not in feature_name:
    vel[:, 0] = vel[:, 0] - baseline_deriv[0]
    vel[:, 1] = vel[:, 1] - baseline_deriv[0]  # Note: both use baseline_deriv[0]
speed = np.sqrt((vel**2).sum(axis=1))
```

iii. The agent did not explicitly describe the baseline drift subtraction in its reasoning.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same approach as tongue: session 50th percentile threshold, codes 0 (low), 1 (high), 2 (not visible).

ii.
```python
paw_threshold = float(np.nanpercentile(paw_speed[paw_visible], 50))
paw_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
paw_disc[paw_visible & (paw_speed < paw_threshold)] = 0
paw_disc[paw_visible & (paw_speed >= paw_threshold)] = 1
```

iii. N/A

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset is computed via median of bitcode times, frame times are corrected, and positions are interpolated directly onto the neural time grid.

ii. Same video offset and interpolation as tongue (see 7-d).

iii. N/A

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. The `motionEnergy_*.mat` file, loaded with `scipy.io.loadmat`. The AI accesses `me['data']` directly (one level of unwrapping). The reference unwraps in a loop to handle multiple nesting levels.

ii.
```python
me_raw = loadmat(me_path, simplify_cells=True)["me"]["data"]
```

iii. The agent described using motion energy files aligned with the reference code.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The motion energy values (one per frame) are interpolated onto the neural time grid using `interp_motion`, which does linear interpolation and then **nearest-neighbor fill** at the edges. The reference simply bins the raw frame values into 5 ms bins (mean per bin). The nearest-fill means that bins outside the range of valid frame times get filled with the nearest valid value rather than being left as NaN.

ii.
```python
def interp_motion(src_t, src_y, target_t):
    valid = np.isfinite(src_t) & np.isfinite(src_y)
    out = np.interp(target_t, src_t[valid], src_y[valid], left=np.nan, right=np.nan)
    return fill_nearest_1d(out)

aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
```

iii. The agent described "motion energy aligned as in the reference code."

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Session 50th percentile of available motion energy values. Codes: 0 (low), 1 (high), 2 (no_video). Matches the reference approach.

ii.
```python
me_threshold = float(np.nanpercentile(motion_energy[motion_available], 50))
me_disc = np.full((keep_trials.size, NT), 2, dtype=np.int64)
me_disc[motion_available & (motion_energy < me_threshold)] = 0
me_disc[motion_available & (motion_energy >= me_threshold)] = 1
```

iii. N/A

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Same video offset as other camera streams. Motion energy uses the side camera's frame times, corrected by the video offset and go cue time. Values are then linearly interpolated onto the neural time grid. If frame times and motion energy arrays have mismatched sizes, a fallback using synthetic frame times from the `ts` array is attempted.

ii.
```python
frame_times = as_1d_float(read_matlab_any(h5file, side_view["frameTimes"][trial_idx, 0]))
frame_times = frame_times - video_shift - align_time
me_trial = np.asarray(me_raw[trial_idx], dtype=np.float64).reshape(-1)
# Fallback for size mismatch
if frame_times.size != me_trial.size or ...:
    # ... uses synthetic frame times
aligned_me = interp_motion(frame_times, me_trial, TIME_BINS)
```

iii. The agent mentioned handling edge cases where frame times are NaN or mismatched.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies:
- Missing/NaN frame times: A fallback using synthetic times computed from the `ts` array shape is used for motion energy. For DLC tracking, `interpolate_visible_segments` handles gaps by only interpolating within valid segments.
- NaN positions: `fill_nearest_1d` fills NaN values with the nearest valid value, which means invalid tracking data gets filled rather than left as missing.
- Untracked frames: Tracked as "not visible" in the discretization.
- Frame count mismatches: Handled by the fallback mechanism for motion energy.
- The AI does NOT handle trials that run past the end of the recording (unlike the reference).

ii.
```python
def fill_nearest_1d(y):
    good = np.flatnonzero(np.isfinite(y))
    if good.size == 0:
        return y
    y[: good[0]] = y[good[0]]
    y[good[-1] + 1 :] = y[good[-1]]
    return y
```

iii. The agent described finding and fixing edge cases with NaN frame times and motion energy mismatches.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files is the primary bottleneck, as each session file must be parsed. The per-trial video processing loop (computing feature positions, interpolation, velocity for each trial individually) is also expensive since it involves many per-trial HDF5 reads and numpy operations. The code reopens the HDF5 file in `compute_video_outputs` after already reading it in `convert_session`.

ii.
```python
with h5py.File(data_path, "r") as h5file:
    # Neural processing
    ...
# Then in compute_video_outputs:
with h5py.File(data_path, "r") as h5file:
    # Video processing - reopens the same file
```

iii. N/A

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial video feature extraction loop iterates over each trial individually, performing HDF5 reads, interpolation, and velocity computation. The per-cluster neural processing loop also iterates over clusters individually. The `smooth_causal_reflect` function loops over rows for convolution. The spike binning uses `np.add.at` per cluster rather than a single `histogram2d` over all clusters.

ii.
```python
for out_trial_idx, trial_idx in enumerate(keep_trials):
    # Per-trial HDF5 reads and interpolation
    side_tongue_speed, ... = get_feature_positions(...)
    bottom_tongue_speed, ... = get_feature_positions(...)
    paw_speed_trial, ... = get_feature_positions(...)
```

```python
for clu_idx in range(qualities.shape[0]):
    # Per-cluster processing
    np.add.at(counts, (mapped_trials, bin_idx), 1.0)
```

iii. N/A

## 11-c. What processing does the code repeat multiple times?

i. The HDF5 session file is opened twice: once in `convert_session` for neural data and behavioral labels, and again in `compute_video_outputs` for video processing. Within the video loop, `read_matlab_any` is called for every trial for every feature, re-reading field names and tracking data from HDF5 each time. The go cue array is read in both the neural and video processing paths.

ii.
```python
def convert_session(session_info):
    with h5py.File(data_path, "r") as h5file:
        # First open for neural + behavior
        ...
    tongue_disc, paw_disc, me_disc, thresholds = compute_video_outputs(session_info, keep_trials)
    # compute_video_outputs opens the same file again

def compute_video_outputs(session_info, keep_trials):
    with h5py.File(data_path, "r") as h5file:
        # Second open of same file
```

iii. N/A

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The `fill_nearest_1d` fills NaN values in position and motion energy arrays, but these filled values at the edges are then potentially overridden by the "not visible" classification. The baseline drift subtraction for paw is an extra computation not present in the reference. Reading `autolearn` from the Bpod structure is unnecessary processing since the reference does not use this filter. Using multiple tongue/paw features and averaging them is additional computation beyond what the reference does.

ii.
```python
filled_xy[:, 0] = fill_nearest_1d(filled_xy[:, 0])
# ...
baseline_deriv = np.nanmedian(np.diff(filled_xy, axis=0), axis=0)
if "tongue" not in feature_name:
    vel[:, 0] = vel[:, 0] - baseline_deriv[0]
```

iii. N/A
