# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` folder, hard-coded in `SESSION_SPECS`. Each session is opened as an HDF5 file using `h5py`. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`. The AI does NOT load sessions from the `RandomizedDelay_Ephys_Behavior` folder, missing 19 of the 44 sessions the authors analyzed.

ii.
```python
DATA_SUBDIR = "Ephys_Behavior"
SESSION_SPECS = [
    {"subject": "JEB6", "date": "2021-04-18", "probe": 2},
    {"subject": "JEB7", "date": "2021-04-29", "probe": 1},
    # ... 12 sessions total, all from Ephys_Behavior only
]

session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
with h5py.File(session_path, "r") as mat:
    beh = load_basic_behavior(mat)
```

iii. The AI decided to use only "the 12 hand-picked two-context ALM electrophysiology sessions from the paper's context scripts." The AI reasoned that the decoder targets only make sense for sessions where both DR and WC contexts appear, and that the MATLAB context analysis scripts use a specific hand-written session list. However, this interpretation is too narrow -- the authors' `load<ANM>_ALMVideo.m` scripts list all 44 sessions across both folders.

## 1-b. How are the data split into subjects?

i. The subject is taken from the `spec["subject"]` field in `SESSION_SPECS`. Subjects are collected in order of first appearance (not sorted).

ii.
```python
subjects = []
subject_to_idx = {}
for sess in sessions:
    if sess["subject"] not in subject_to_idx:
        subject_to_idx[sess["subject"]] = len(subjects)
        subjects.append(sess["subject"])
    subject_idx.append(subject_to_idx[sess["subject"]])
```

iii. The AI hard-codes the subject name in each session spec rather than extracting it from the filename or the data file, which works but produces subjects in encounter order rather than sorted order.

## 1-c. How are the data split into sessions?

i. One session is one entry in `SESSION_SPECS`. Only the `Ephys_Behavior` subfolder is searched (via `DATA_SUBDIR = "Ephys_Behavior"`). The `RandomizedDelay_Ephys_Behavior` folder is not searched at all. Only 12 of the 44 sessions are included. Multi-probe sessions (e.g., JEB15 with probes [1,2]) are loaded with only a single probe.

ii.
```python
DATA_SUBDIR = "Ephys_Behavior"
session_path = data_root / DATA_SUBDIR / f"data_structure_{session_tag}.mat"
```

iii. The AI believed these 12 sessions are the "two-context" subset and that the randomized delay sessions are a separate experiment. This is incorrect -- the reference includes all 44 sessions from both folders.

## 1-d. How are the data split into trials?

i. Trials are indexed by the `Ntrials` field in `obj.bp`. Each trial has its associated behavioral variables (hit, miss, early, etc.), go cue time, and neural/video data.

ii.
```python
data = {
    "ntrials": int(read_h5_numeric(bp["Ntrials"])),
    ...
}
```

iii. The trial structure follows the standard Bpod table layout, which is correct.

## 1-e. How are trials filtered based on quality controls?

i. Early-lick trials (`early`) and photostimulation trials (`stim.enable`) are removed. The AI does NOT check for trials that run past the end of the recording (the reference does this). The AI also requires at least 2 valid trials per session.

ii.
```python
valid_trials = ~beh["early"] & ~beh["stim_enable"]
valid_idx = np.flatnonzero(valid_trials)
if valid_idx.size < 2:
    raise ValueError(f"{session_tag} has fewer than two valid trials after filtering")
```

iii. The AI correctly identified that early-lick and photostim trials should be removed, following the paper. It does not handle the edge case of trials extending past the recording end.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu` (the spike-sorted clusters). Each cluster has `trial` (spike trial assignment), `trialtm` (spike times relative to trial start), and `quality` (curation label). The go cue times `bp.ev.goCue` provide the alignment event.

ii.
```python
clu_ref = mat["obj"]["clu"][()][probe - 1, 0]
clu = mat[clu_ref]
trial_refs = np.asarray(clu["trial"][()]).squeeze()
trialtm_refs = np.asarray(clu["trialtm"][()]).squeeze()
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue by subtracting `goCue[trial]` from `trialtm`, then histogrammed into bins of 10 ms (DT = 1/100) spanning -3.0 to 2.5 s (550 bins). Counts are converted to firing rates (Hz) by dividing by DT, then smoothed with a **causal** Gaussian filter: `gausswin(15)` with the first half zeroed out, applied via convolution with `reflect` boundary. This differs from the reference which uses a symmetric Gaussian with sigma=14ms and 5ms bins spanning -2.5 to 2.5s.

ii.
```python
TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0  # 10 ms bins
SMOOTH = 15
BOUNDARY = "reflect"

def my_smooth(x, n, bctype):
    kern = gausswin(n)
    kern[: len(kern) // 2] = 0.0  # causal: zero out first half
    kern = kern / np.sum(kern)
    out[:, col] = np.convolve(arr_filt[:, col], kern, mode="same")
```

iii. The AI states it applies "the paper's causal Gaussian filter (window 15, reflect boundary)." The reference code's `my_smooth.m` does use a half-zeroed `gausswin(15)`, so the AI is faithfully replicating that function, though the reference solution instead uses a symmetric Gaussian with sigma=14ms.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are filtered by quality label: labels matching `garbage`, `gabrga`, `noisy`, or `real?` are excluded. Note: the AI does NOT exclude `poor` quality units (the reference does). Then the mean firing rate is computed from condition-averaged PSTHs (not the raw per-trial rates), and units with mean rate <= 1 Hz are dropped.

ii.
```python
def quality_is_usable(label: str) -> bool:
    label = label.strip().lower()
    return label not in {"garbage", "gabrga", "noisy", "real?"}

# Mean rate computed from condition-averaged PSTHs:
psth_by_cond = []
for mask in condition_masks:
    psth = trialdat[:, mask, :].mean(axis=1)
    psth_by_cond.append(psth)
psth_by_cond = np.stack(psth_by_cond, axis=2)
mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
```

iii. The AI followed the reference MATLAB code's `findClusters.m` for the quality labels. However, it omits `poor` from the exclusion list. The mean rate computation from condition-averaged PSTHs is unusual -- the reference computes it as the overall mean rate across all trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting `goCue[trial]` from each spike's `trialtm`, then histogramming into time bins. This is the same approach as the reference.

ii.
```python
aligned_times = spike_trialtm - go_cue[spike_trials]
for tr in np.unique(spike_trials):
    spk = aligned_times[spike_trials == tr]
    counts, _ = np.histogram(spk, bins=edges)
```

iii. Standard go-cue alignment, matching the reference.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT = 1/100) spanning -3.0 to 2.5 s, yielding 550 time bins. The reference uses 5 ms bins spanning -2.5 to 2.5 s, yielding 1000 bins. No rebinning is applied.

ii.
```python
TIME_MIN = -3.0
TIME_MAX = 2.5
DT = 1.0 / 100.0  # 10 ms
```

iii. The AI states it uses the paper's `params.dt = 1/200`, but actually implements `DT = 1/100` (10 ms), which is twice the paper's bin size. The time window also starts at -3.0s instead of -2.5s.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is a time axis computed from the bin edges, identical in concept to the reference. It is the center of each time bin from -3.0 to 2.5 s at 10 ms resolution.

ii.
```python
def get_time_axis() -> np.ndarray:
    edges = np.arange(TIME_MIN, TIME_MAX + DT, DT, dtype=np.float64)
    return edges[:-1] + DT / 2.0
```

iii. Same concept as the reference, but with different parameters.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing bin centers from the defined time axis.

ii. N/A

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis uses the same bin edges as the neural data, so alignment is inherent.

ii.
```python
taxis = get_time_axis().astype(np.float32)
input_template = taxis[None, :]
```

iii. Correct -- same grid as the neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. The AI derives lick direction from the actual lick event times: `bp.ev.lickL` and `bp.ev.lickR`. It finds the first lick after the go cue to determine direction. The reference instead derives lick direction from `bp.R`, `bp.hit`, and `bp.miss` (the instructed side and outcome).

ii.
```python
def first_post_go_lick_direction(mat, lick_l_refs, lick_r_refs, go_cue):
    for trial_idx in range(go_cue.size):
        lick_l = read_trial_ref_numeric(mat, lick_l_refs[trial_idx])
        lick_r = read_trial_ref_numeric(mat, lick_r_refs[trial_idx])
        post_l = lick_l[lick_l >= go_cue[trial_idx] - 1e-9]
        post_r = lick_r[lick_r >= go_cue[trial_idx] - 1e-9]
        first_l = post_l[0] if post_l.size else np.inf
        first_r = post_r[0] if post_r.size else np.inf
        if first_l < first_r:
            lick_dir[trial_idx] = 0  # left
        elif first_r < first_l:
            lick_dir[trial_idx] = 1  # right
```

iii. The AI chose to use the actual lick timestamps, which is a more direct measurement. However, the reference's approach of inferring from hit/miss + R/L is equivalent for hit/miss trials and more clearly handles ignore trials.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The first post-go-cue lick event (lickL or lickR) determines direction. If neither lick occurs, the trial is coded as 2 (none). The reference derives this from instructed side and outcome (hit -> licked instructed, miss -> licked opposite, else -> no lick).

ii. See 4-a code snippet.

iii. Both approaches should produce equivalent results for hit and miss trials. For ignore trials (no lick), both code as 2.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The `autowater` field from `bp`. Autowater trials are WC (0), others are DR (1).

ii.
```python
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
```

iii. Matches the reference exactly.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater -> WC (0), not autowater -> DR (1).

ii.
```python
context = np.where(beh["autowater"], 0, 1).astype(np.int64)
```

iii. Same as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit` and `bp.miss` flags, with remaining trials as ignore. The AI also reads `bp.no` but does not use it for the outcome coding.

ii.
```python
outcome = np.full((beh["ntrials"],), 2, dtype=np.int64)  # ignore
outcome[beh["miss"]] = 0  # incorrect
outcome[beh["hit"]] = 1   # correct
```

iii. Same approach as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Three-class coding: miss -> incorrect (0), hit -> correct (1), else -> ignore (2).

ii. See 6-a code snippet.

iii. Matches the reference.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. The AI uses 7 tongue-related features from both cameras: `tongue`, `left_tongue`, `right_tongue` from the side camera (view 1) and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` from the bottom camera (view 2). The reference uses only 2 features: `tongue` from the side camera and `top_tongue` from the bottom camera.

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
```

iii. The AI chose to include all tracked tongue features, reasoning that more features provide better coverage. However, the reference uses only the main tongue tracking point from each camera.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates the raw x,y positions onto the decoder's time axis (10ms bins) using linear interpolation, then computes velocity as `np.gradient` of the interpolated positions. For tongue features, NaN velocities are replaced with 0. No per-run smoothing is applied. The composite speed is the mean across all visible features at each time bin.

The reference instead computes velocity at the original frame times: it applies a 5ms Gaussian smooth to x,y within each contiguous run of tracked frames, then differentiates, then bins the resulting speeds into the decoder time bins. The two camera views are normalized by their 90th percentile before averaging.

ii.
```python
x_interp = interp_with_nans(old_t, x, taxis)
y_interp = interp_with_nans(old_t, y, taxis)
# For tongue: NaN -> 0
xvel, yvel = compute_velocity(x_proc, y_proc, is_tongue=True)
speed = np.sqrt(xvel**2 + yvel**2)
```

```python
def compute_velocity(xpos, ypos, is_tongue):
    xvel = np.gradient(xpos)
    yvel = np.gradient(ypos)
    if is_tongue:
        xvel[np.isnan(xvel)] = 0.0
        yvel[np.isnan(yvel)] = 0.0
    return xvel, yvel
```

iii. The AI's approach differs significantly from the reference: it interpolates first then differentiates, rather than differentiating at frame times then binning. It also doesn't normalize the two cameras before averaging, and it replaces NaN velocities with 0 rather than leaving them as NaN for a "not visible" class assignment.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of visible values is the threshold. Values >= threshold -> 1, < threshold -> 0, not visible -> 2.

ii.
```python
def discretize_with_visibility(values, visible, keep_mask, absent_code):
    threshold = float(np.nanpercentile(keep_values[keep_visible], 50))
    classes[present] = (keep_values[present] >= threshold).astype(np.int64)
```

iii. The thresholding approach matches the reference in concept (50th percentile split), though the underlying speed values differ due to different processing.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed the same way as the reference (bitcode alignment), then frame times are corrected and used to interpolate directly onto the decoder time axis.

ii.
```python
def get_video_offset(mat, bit_start):
    fs = float(read_h5_numeric(mat["obj"]["sglx"]["fs"]))
    bitcode_starts = read_h5_numeric(mat["obj"]["sglx"]["bitcode"]["bitstart"])
    return matlab_mode(bitcode_starts) / fs - matlab_mode(bit_start)

old_t = frame_times - vidshift - align_time
x_interp = interp_with_nans(old_t, x, taxis)
```

iii. The video offset computation matches the reference. The alignment approach (interpolation to time axis) is different from the reference (bin averaging).

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. The AI uses both `top_paw` and `bottom_paw` from the bottom camera (view 2). The reference uses only `top_paw`.

ii.
```python
PAW_FEATURES = [
    (2, "top_paw"),
    (2, "bottom_paw"),
]
```

iii. The reference explains that `bottom_paw` drops out during the delay epoch and is unreliable, so only `top_paw` is used. The AI includes both.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same interpolation-then-gradient approach as tongue velocity, but with different NaN handling: for non-tongue features, the AI subtracts a baseline derivative (median of diff) and uses nearest-neighbor filling for NaN values. The reference uses the same per-run Gaussian smooth + gradient approach as for tongue.

ii.
```python
def compute_velocity(xpos, ypos, is_tongue):
    xvel = np.gradient(xpos)
    yvel = np.gradient(ypos)
    if not is_tongue:
        basederiv_x = np.nanmedian(np.diff(np.column_stack([xpos, ypos]), axis=0)[:, 0])
        xvel = xvel - basederiv_x
        yvel = yvel - basederiv_x
        xvel = fill_nearest_1d(xvel)
        yvel = fill_nearest_1d(yvel)
    return xvel, yvel
```

iii. The AI applies a baseline correction and nearest-neighbor fill for paw velocity, which the reference does not do.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile threshold, with not-visible class for untracked bins.

ii.
```python
paw_classes, paw_threshold = discretize_with_visibility(
    paw_speed, paw_visible, valid_trials, absent_code=2
)
```

iii. Matches the reference in concept.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: video offset correction, then interpolation onto the decoder time axis.

ii. Same as 7-d.

iii. Same alignment mechanism as tongue velocity.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<subject>_<date>.mat` using `scipy.io.loadmat`. The AI accesses `me.data` (as a structured array). Frame times come from the side camera's `traj` group.

ii.
```python
motion_file = sio.loadmat(motion_path, squeeze_me=True, struct_as_record=False)
me = motion_file["me"]
motion_trials = np.asarray(me.data, dtype=object).reshape(-1)
```

iii. Same source data as the reference, though the reference handles three different file layouts with a `while isinstance(me, dict)` loop.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI interpolates motion energy onto the decoder time axis using linear interpolation, then fills NaN gaps with nearest-neighbor interpolation (`fill_nearest_1d`). The reference simply bins the frame-rate values into 5ms bins by averaging (no interpolation or gap-filling).

ii.
```python
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
if np.any(visible[trial_idx, :]):
    aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)
```

iii. The nearest-neighbor filling means bins that had no camera frame will get an imputed value rather than being marked as "no video". This changes the semantics of the "no_video" class.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same per-session 50th percentile split as other movement variables. The absent code is 2 (no_video).

ii.
```python
motion_classes, motion_threshold = discretize_with_visibility(
    motion_energy, motion_visible, valid_trials, absent_code=2
)
```

iii. Conceptually matches the reference, but the nearest-neighbor filling of gaps means the threshold is computed over more data points (including filled values).

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Video offset correction using bitcode alignment, then linear interpolation onto the decoder time axis. Uses the side camera (view index 0) frame times.

ii.
```python
bundle = load_trial_video_bundle(mat, traj_groups[0], trial_idx)
old_t = frame_times - vidshift - beh["goCue"][trial_idx]
trial_aligned = interp_with_nans(old_t, trial_motion, taxis)
```

iii. The interpolation approach differs from the reference's bin-averaging approach.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses nearest-neighbor filling (`fill_nearest_1d`) for NaN gaps in motion energy and paw velocity. For tongue velocity, NaN values are replaced with 0. If `NdroppedFrames` is NaN, the trial's video data is skipped. If `frameTimes` are empty or all-NaN, synthetic frame times are generated (`(arange + 1) / 400`). The reference never interpolates or fills -- it marks missing data as the "not visible" class.

ii.
```python
# Synthetic frame times when missing:
if frame_times.size == 0 or not np.any(np.isfinite(frame_times)):
    frame_times = (np.arange(ts.shape[0], dtype=np.float64) + 1.0) / 400.0

# Nearest-neighbor fill for motion energy:
aligned[trial_idx, :] = fill_nearest_1d(trial_aligned).astype(np.float32)
```

iii. The AI's approach of filling/imputing missing data is different from the reference's approach of representing missing data honestly with a "not visible" class.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files and computing per-trial velocities for all tongue/paw features. The AI processes 7 tongue features and 2 paw features per trial, which is more work than the reference's 2 tongue + 1 paw. The loop over trials for video processing is the inner bottleneck.

ii.
```python
for trial_idx in range(ntrials):
    for feat_out_idx, (view, feat_name) in enumerate(feature_specs):
        # load, interpolate, compute velocity per feature per trial
```

iii. The per-trial per-feature loop with HDF5 reads is expensive.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial spike binning loop could be replaced with a single `histogram2d` call (as the reference does). The per-trial velocity computation involves interpolation and gradient that are hard to vectorize due to variable frame counts.

ii.
```python
for tr in np.unique(spike_trials):
    spk = aligned_times[spike_trials == tr]
    counts, _ = np.histogram(spk, bins=edges)
    rates = counts.astype(np.float64) / DT
    trialdat[out_idx, tr, :] = my_smooth(rates, SMOOTH, BOUNDARY).astype(np.float32)
```

iii. The reference vectorizes spike counting with `np.histogram2d` over all trials at once, which is significantly faster.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed once per session (via `get_video_offset`), but `load_trial_video_bundle` is called separately for each feature group (tongue and paw), loading the same trial's `ts` and `frameTimes` multiple times. Motion energy also reloads the traj group for frame times. The smoothing is applied per-trial per-unit, which is necessary.

ii.
```python
tongue_speed, tongue_visible = compute_composite_speed(mat, beh, TONGUE_FEATURES)
paw_speed, paw_visible = compute_composite_speed(mat, beh, PAW_FEATURES)
# Each calls load_trial_video_bundle for every trial
```

iii. The repeated loading of trial video bundles across tongue and paw processing is redundant.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes condition-averaged PSTHs (`psth_by_cond`) solely to filter units by mean firing rate. These PSTHs are not used in the output and represent extra computation. It also reads `bp.no`, `bp.L`, and lick event references that are not strictly necessary for the output (though `lickL`/`lickR` are used for lick direction). The `sites` array is computed but only stored as `brain_region_idx` (all zeros), making the site computation pointless.

ii.
```python
psth_by_cond = []
for mask in condition_masks:
    psth = trialdat[:, mask, :].mean(axis=1)
    psth_by_cond.append(psth)
mean_frs = psth_by_cond.mean(axis=(1, 2))
use = mean_frs > 1.0
```

iii. The PSTH computation is a detour for what could be a simple overall mean rate check.
