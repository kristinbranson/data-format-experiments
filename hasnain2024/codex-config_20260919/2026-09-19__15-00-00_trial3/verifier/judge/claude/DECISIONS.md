# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only 12 sessions from the `Ephys_Behavior` directory (the "two-context" cohort used in the paper's Figure 8 and context neural analyses). Sessions are hard-coded in a `SESSIONS` list of `(animal, date, probes)` tuples. Each session is opened as HDF5 via `h5py`. Motion energy is loaded separately via `scipy.io.loadmat`. The AI explicitly chose NOT to include the 25-session fixed-delay cohort or the 19-session randomized-delay cohort, reasoning that only the 12 two-context sessions have genuine alternating WC/DR blocks.

ii.
```python
SESSIONS = [
    ("JEB6", "2021-04-18", (2,)),
    ("JEB7", "2021-04-29", (1,)),
    ...
    ("JEB19", "2023-04-18", (1,)),
]

# In process_session:
with h5py.File(data_path, "r") as f:
    ...
```

iii. From CONVERSION_NOTES.md Step 4: "Use the 12 sessions explicitly loaded by Figure 8/two-context neural scripts. This is the only neural cohort in which `autowater` represents alternating WC blocks rather than occasional assistance." The AI argues that the other 13 fixed-delay sessions contain only occasional autowater assistance trials, not genuine WC context blocks, and the randomized-delay sessions test a different task variant.

## 1-b. How are the data split into subjects?

i. The animal ID is the first element of each `SESSIONS` tuple (e.g., `"JEB6"`). Subjects are accumulated in order of first appearance. `subject_idx` maps each session to its subject index. The result is 7 unique subjects across the 12 sessions.

ii.
```python
subjects = []
for index, (animal, date, probes) in enumerate(selected):
    if animal not in subjects:
        subjects.append(animal)
subject_idx = np.asarray([subjects.index(animal) for animal, _, _ in selected], dtype=np.int64)
```

iii. The animal IDs are taken from the hard-coded session list, consistent with how the authors' loading scripts identify animals.

## 1-c. How are the data split into sessions?

i. Each entry in the `SESSIONS` list is one session. The AI processes only sessions from the `Ephys_Behavior` directory (not `RandomizedDelay_Ephys_Behavior`). The `DATA_DIR` is hard-coded to `/app/data/Ephys_Behavior`.

ii.
```python
DATA_DIR = APP / "data" / "Ephys_Behavior"
data_path = DATA_DIR / f"data_structure_{session_id}.mat"
motion_path = DATA_DIR / f"motionEnergy_{session_id}.mat"
```

iii. The 12 sessions are from the two-context cohort as described above in 1-a.

## 1-d. How are the data split into trials?

i. The number of trials is read from `obj.bp.Ntrials`. All per-trial fields (`R`, `L`, `hit`, `miss`, `no`, `early`, `autowater`, `stim.enable`, `goCue`) are loaded as boolean or float arrays and validated to have length `n_trials`.

ii.
```python
n_trials = int(np.asarray(f["obj/bp/Ntrials"])[0, 0])
fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}
bp["stim"] = np.asarray(f["obj/bp/stim/enable"]).ravel().astype(bool)
go = np.asarray(f["obj/bp/ev/goCue"]).ravel().astype(np.float64)
```

iii. The trial count comes directly from the data file's `Ntrials` field, consistent with the reference code.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes trials that are early-lick, stimulation-enabled, have non-finite go cue times, do not have exactly one outcome flag (hit+miss+no != 1), or do not have exactly one side flag (R+L != 1). This is more aggressive filtering than the reference, which only filters early-lick and photostim trials.

ii.
```python
outcome_sum = bp["hit"].astype(int) + bp["miss"].astype(int) + bp["no"].astype(int)
side_sum = bp["R"].astype(int) + bp["L"].astype(int)
retained = np.flatnonzero(~bp["early"] & ~bp["stim"] & np.isfinite(go)
                          & (outcome_sum == 1) & (side_sum == 1))
```

iii. From CONVERSION_NOTES.md: "Retain trials satisfying `~early & ~stim.enable` and having exactly one outcome flag and one instructed/reward-side flag." The extra checks (outcome_sum==1, side_sum==1, finite go cue) are data integrity validation rather than paper-specified filtering.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data comes from `obj.clu{probe}` clusters. For each cluster on the selected probe(s), `trial` (spike trial assignment), `trialtm` (spike time relative to trial start), and `quality` (manual curation label) are read. `bp.ev.goCue` provides the alignment event times.

ii.
```python
for cluster_index in range(group["quality"].shape[0]):
    quality = h5_string(f, group["quality"][cluster_index, 0]).strip()
    ...
    trials = h5_vector(f, group["trial"][cluster_index, 0], np.float64)
    trial_times = h5_vector(f, group["trialtm"][cluster_index, 0], np.float64)
```

iii. This matches the reference approach of using sorted spike clusters aligned to go cue.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to go cue (`trialtm - goCue[trial]`), binned into 10 ms bins spanning [-2.5, 2.5) (500 bins), divided by dt (0.01s) to get firing rates in Hz, then smoothed with a 15-sample **causal** Gaussian kernel that matches the reference code's `mySmooth.m`. The smoothing prepends a reflected block of the signal before convolution and zeros out the acausal half of the kernel.

ii.
```python
DT = 0.010
SMOOTH_N = 15

def matlab_gausswin_causal(n=SMOOTH_N, alpha=2.5):
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    kernel = np.exp(-0.5 * (alpha * x / (n / 2)) ** 2)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel

def smooth_reference(counts):
    prefix = counts[..., :SMOOTH_N]
    padded = np.concatenate((prefix, counts), axis=-1)
    ...
    smoothed = np.einsum("...k,k->...", windows, KERNEL[::-1], optimize=True)
    return smoothed[..., SMOOTH_N:]

counts = bin_one_unit(trials, trial_times, go, n_trials)
rates = smooth_reference(counts / DT).astype(np.float32)
```

iii. From CONVERSION_NOTES.md: The AI explicitly chose 10 ms bins and causal Gaussian smoothing based on the reference code's `getSeq` and `mySmooth.m` functions, which use `dt=1/100` and a 15-sample causal Gaussian with reflected boundaries.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels in `{garbage, gabrga, noisy, real?}` are excluded (note: `poor` is NOT excluded), (2) units with mean firing rate <= 1 Hz are dropped.

ii.
```python
BAD_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
if quality.lower() in BAD_QUALITIES:
    continue
...
mean_rate = float(rates.mean())
if mean_rate > 1.0:
    all_rates.append(rates)
```

iii. The AI follows `findClusters.m`'s quality={`all`} setting which excludes exactly `garbage`, `gabrga`, `noisy`, and `real?`. The 1 Hz cutoff matches the paper's statement about including units with firing rates exceeding 1 Hz.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's time relative to trial start (`trialtm`) is aligned to the go cue by subtracting `goCue[trial]`. The aligned times are then binned.

ii.
```python
aligned = trial_times[valid_trial] - go[tr0]
bins = np.searchsorted(EDGES, aligned, side="right") - 1
```

iii. This matches the reference code's `alignSpikes.m` which subtracts the alignment event time from each spike's within-trial time.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT=0.010), giving 500 time bins spanning [-2.5, 2.5). No rebinning is applied after the initial binning.

ii.
```python
DT = 0.010
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size  # 500
```

iii. From CONVERSION_NOTES.md Step 4: "Use the common decoder/kinematics window `[-2.5,2.5)` and 10 ms; it matches both the tutorial and choice decoding." The AI chose 10 ms based on the reference code's `dt=1/100` in the tutorial and main analysis scripts, noting that `getDefaultParams.m` uses `dt=1/200` (5 ms) but the actual analysis pipelines override to 10 ms.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself -- the centers of the 500 time bins spanning [-2.5, 2.5) at 10 ms resolution. It is not derived from any raw data variable but constructed from the binning parameters.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
time_input = TIME.astype(np.float32)[None, :]
```

iii. The bin centers define the time from go cue onset for each time point.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing; the time axis is analytically constructed from bin parameters (start, end, bin size).

ii. Same as 3-a above.

iii. N/A.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis IS the neural binning grid (bin centers). Both neural data and the input share the same 500-bin time axis.

ii.
```python
# Neural binning uses the same edges:
EDGES = TMIN + np.arange(N_TIME + 1, dtype=np.float64) * DT
# Input is the bin centers:
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```

iii. N/A.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Derived from `bp.R` (right instruction), `bp.L` (left instruction), `bp.hit`, `bp.miss`, and `bp.no` (outcome flags).

ii.
```python
fields = ("R", "L", "hit", "miss", "no", "early", "autowater")
bp = {name: np.asarray(f[f"obj/bp/{name}"]).ravel().astype(bool) for name in fields}

def trial_labels(bp, trial):
    right, left = bool(bp["R"][trial]), bool(bp["L"][trial])
    hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
    ...
    if no:
        lick = 2
    elif hit:
        lick = 1 if right else 0
    else:
        lick = 0 if right else 1
```

iii. The lick direction is inferred from the instructed side and outcome, since the actual lick direction is not directly recorded.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. On hit trials, the lick direction matches the instructed side (right=1, left=0). On miss trials (incorrect), the lick direction is the opposite of the instructed side. On no-response (ignore) trials, lick=2 (none). The value is broadcast across all 500 time bins (per-trial constant).

ii.
```python
if no:
    lick = 2
elif hit:
    lick = 1 if right else 0
else:  # incorrect
    lick = 0 if right else 1
...
output[0] = lick  # broadcast across all time bins
```

iii. This matches the logic of inferring lick direction from the outcome and instructed side.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. When autowater is true, the trial is WC (water-cued); otherwise DR (delayed-response).

ii.
```python
context = 0 if bool(bp["autowater"][trial]) else 1
```

iii. The `autowater` flag directly marks WC context trials.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Direct mapping: autowater=True -> WC (0), autowater=False -> DR (1). Broadcast as per-trial constant across all time bins.

ii.
```python
context = 0 if bool(bp["autowater"][trial]) else 1
output[1] = context
```

iii. Matches the expected WC=0, DR=1 encoding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit`, `bp.miss`, and `bp.no`.

ii.
```python
hit, miss, no = bool(bp["hit"][trial]), bool(bp["miss"][trial]), bool(bp["no"][trial])
outcome = 1 if hit else (0 if miss else 2)
```

iii. The three mutually exclusive outcome flags directly determine the outcome class.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct mapping: hit -> correct (1), miss -> incorrect (0), no -> ignore (2). Broadcast across all time bins.

ii.
```python
outcome = 1 if hit else (0 if miss else 2)
output[2] = outcome
```

iii. Matches the required encoding of incorrect=0, correct=1, ignore=2.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from the side camera's DLC tracking data in `obj.traj[0]` (view index 0). The feature `tongue` provides x, y coordinates. Frame times from `traj.frameTimes` and the video clock offset (from `sglx.bitcode.bitstart`, `sglx.fs`, and `bp.ev.bitStart`) are used for temporal alignment.

ii.
```python
tongue_ix = side_names.index("tongue")
...
side_ts, side_ft = load_trial_traj(f, side, trial)
if side_ts is not None and tongue_ix < side_ts.shape[2]:
    aligned_ft = side_ft - vidshift - go[trial]
    x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
    y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
    tongue[trial], _ = velocity_from_xy(x, y, tongue=True)
```

iii. Only the side camera's tongue tracking is used. The AI does not use the bottom camera's `top_tongue` feature.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. Five steps: (1) Video clock offset is computed via bitcode alignment. (2) Frame times are aligned to go cue: `frameTimes - vidshift - goCue[trial]`. (3) Tongue x, y positions are **linearly interpolated** from frame times to the 10 ms bin centers. (4) Velocity is computed as the Euclidean magnitude of `np.gradient(x)` and `np.gradient(y)` on the interpolated positions, with NaN gradients set to 0 for tongue. (5) The continuous velocity is discretized at the per-session 50th percentile.

ii.
```python
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
tongue[trial], _ = velocity_from_xy(x, y, tongue=True)

def velocity_from_xy(x, y, tongue=True):
    visible = np.isfinite(x) & np.isfinite(y)
    xf, yf = x.copy(), y.copy()  # for tongue, no fill
    xv = np.gradient(xf)
    yv = np.gradient(yf)
    xv[~np.isfinite(xv)] = 0
    yv[~np.isfinite(yv)] = 0
    speed = np.hypot(xv, yv)
    speed[~visible] = np.nan
    return speed, visible
```

iii. The AI interpolates positions to bin centers before computing velocity, rather than computing velocity at frame resolution and then binning. For tongue, no nearest-fill is applied (unlike paw), and NaN gradients are zeroed. This differs from the reference approach of computing velocity at camera frame rate and averaging into bins.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The session-wide 50th percentile of finite tongue velocity values (computed only over retained trials) is the threshold. Values below the threshold get class 0, at or above get class 1, and NaN (not visible) gets class 2.

ii.
```python
def discretize_session(values, retained):
    subset = values[retained]
    finite = np.isfinite(subset)
    threshold = float(np.nanpercentile(subset, 50))
    labels = np.full(subset.shape, 2, dtype=np.int64)
    labels[finite & (subset < threshold)] = 0
    labels[finite & (subset >= threshold)] = 1
    return labels, threshold
```

iii. Matches the instructions: 0 = <50th percentile, 1 = >=50th percentile, 2 = not visible.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Frame times are corrected by the video offset and aligned to go cue. Then x, y positions are linearly interpolated from frame times onto the same 10 ms bin centers used for neural data. Velocity is then computed on those interpolated bin-center values.

ii.
```python
aligned_ft = side_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, side_ts[:, 0, tongue_ix], target_absolute)
y = interp_matlab(aligned_ft, side_ts[:, 1, tongue_ix], target_absolute)
```

iii. By interpolating to bin centers, the tongue velocity shares the same time axis as the neural data.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from the bottom camera's DLC tracking data (`obj.traj[1]`). The AI uses BOTH `top_paw` and `bottom_paw` features (averaging them when both are available), unlike the reference which uses only `top_paw`.

ii.
```python
paw_indices = [bottom_names.index(x) for x in ("top_paw", "bottom_paw") if x in bottom_names]
...
for feature_ix in paw_indices:
    x = interp_matlab(aligned_ft, bottom_ts[:, 0, feature_ix], target_absolute)
    y = interp_matlab(aligned_ft, bottom_ts[:, 1, feature_ix], target_absolute)
    speed, _ = velocity_from_xy(x, y, tongue=False)
    paw_speeds.append(speed)
if paw_speeds:
    stack = np.stack(paw_speeds)
    paw[trial] = np.divide(np.nansum(stack, axis=0), np.sum(np.isfinite(stack), axis=0), ...)
```

iii. The AI averages speed from both paw features when available. This is different from the reference which uses only `top_paw` because it is more reliably tracked.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Similar pipeline to tongue but with differences: (1) positions are **nearest-filled** before computing velocity (matching `findVelocity.m`'s behavior for non-tongue features), (2) a baseline gradient correction is applied (subtracting the median of `diff` along coordinates), (3) speed from both paw markers is averaged. Then discretized at session 50th percentile.

ii.
```python
def velocity_from_xy(x, y, tongue=False):
    xf, yf = fill_nearest(x), fill_nearest(y)
    xv = np.gradient(xf)
    yv = np.gradient(yf)
    baseline_x = np.nanmedian(np.diff(np.column_stack((xf, yf)), axis=0), axis=0)[0]
    xv -= baseline_x
    yv -= baseline_x
    speed = np.hypot(xv, yv)
    speed[~visible] = np.nan
    return speed, visible
```

iii. The AI attempts to match `findVelocity.m`'s processing including nearest-fill and baseline correction. However, the baseline correction implementation appears to differ from the reference.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: session-wide 50th percentile of finite values over retained trials. Values below -> 0, at or above -> 1, NaN -> 2.

ii.
```python
paw_labels, paw_threshold = discretize_session(paw, retained)
```

iii. Same discretization logic as tongue velocity.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same approach as tongue: frame times corrected by video offset and go cue, positions interpolated to 10 ms bin centers, velocity computed on interpolated values.

ii.
```python
aligned_ft = bottom_ft - vidshift - go[trial]
x = interp_matlab(aligned_ft, bottom_ts[:, 0, feature_ix], target_absolute)
y = interp_matlab(aligned_ft, bottom_ts[:, 1, feature_ix], target_absolute)
```

iii. Same temporal alignment as all other streams.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Loaded from `motionEnergy_<animal>_<date>.mat` files via `scipy.io.loadmat`. The `me.data` field contains one trace per trial.

ii.
```python
def load_motion_file(path, n_trials):
    loaded = scipy_io.loadmat(path, simplify_cells=True)["me"]
    raw = loaded["data"]
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    raw = np.atleast_1d(raw)
    ...
```

iii. The motion energy files are loaded separately from the main data structure, consistent with the reference approach.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy traces are **linearly interpolated** from side-camera frame times (corrected by video offset and go cue) onto the 10 ms bin centers. Then discretized at the session 50th percentile.

ii.
```python
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]
    motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
```

iii. The AI uses linear interpolation to map frame-rate motion energy onto the bin centers, matching the reference code's `loadMotionEnergy.m` which also uses `interp1`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same discretization as other behavior streams: session-wide 50th percentile, with class 2 for NaN/missing.

ii.
```python
motion_labels, motion_threshold = discretize_session(motion, retained)
```

iii. Same logic as tongue and paw.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Side-camera frame times are corrected by video offset and go cue, then motion energy is linearly interpolated onto the same 10 ms bin centers as neural data.

ii.
```python
aligned_ft = side_ft - vidshift - go[trial]
motion[trial] = interp_matlab(aligned_ft, me, target_absolute)
```

iii. Uses the side camera's frame times for alignment since motion energy is computed from side-view video.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases: (1) Trials with `NdroppedFrames` that is NaN are treated as having no trajectory data. (2) Trials where `frameTimes` is empty or all non-finite return None. (3) For tongue, NaN positions from low DLC likelihood remain NaN (speed set to NaN, encoded as class 2). (4) For paw, positions are nearest-filled before velocity computation, but the original visibility mask is preserved for class 2 assignment. (5) Missing motion energy trials get NaN (class 2). (6) If interpolation target is outside source coverage, NaN is returned. (7) Extra data integrity checks: `outcome_sum == 1` and `side_sum == 1` filter out any malformed trials.

ii.
```python
if "NdroppedFrames" in traj:
    dropped = h5_vector(f, traj["NdroppedFrames"][trial, 0])
    if dropped.size and np.isnan(dropped[0]):
        return None, None

def interp_matlab(x, y, target):
    ...
    return np.interp(target, x, y, left=np.nan, right=np.nan)

# For paw, nearest-fill before velocity:
xf, yf = fill_nearest(x), fill_nearest(y)
```

iii. The AI documents in CONVERSION_NOTES.md that missing video data is handled by assigning class 2, preserving the trial for other outputs rather than dropping it entirely.

## 11-a. What are the most time-consuming steps of the code?

i. Reading the HDF5 files dominates runtime. The full 12-session conversion completes in about 28-41 seconds. The AI uses direct HDF5 field access rather than reading entire objects.

ii.
```python
with h5py.File(data_path, "r") as f:
    ...
```

iii. From CONVERSION_NOTES.md: "Use HDF5 field/reference access" as a speedup over loading entire MATLAB objects.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `load_behavior_streams` iterates over all trials, loading trajectory data and computing interpolated velocities one trial at a time. This is difficult to vectorize because each trial has different numbers of camera frames. The per-cluster loop in `load_neural` also iterates over clusters individually, though spike binning within each cluster is vectorized.

ii.
```python
for trial in range(n_trials):
    side_ts, side_ft = load_trial_traj(f, side, trial)
    ...
    # all velocity/interpolation done per trial

for cluster_index in range(group["quality"].shape[0]):
    ...
    counts = bin_one_unit(trials, trial_times, go, n_trials)
```

iii. The variable frame counts per trial prevent straightforward vectorization of the behavior stream processing.

## 11-c. What processing does the code repeat multiple times?

i. Frame times for the side camera are loaded and aligned multiple times per trial: once for tongue velocity and again for motion energy alignment. The video offset computation is done once per session (efficient). The smoothing kernel is computed once at module level.

ii.
```python
# In the trial loop, side frame times are loaded for tongue:
side_ts, side_ft = load_trial_traj(f, side, trial)
aligned_ft = side_ft - vidshift - go[trial]
# Then later for motion energy:
if me is not None and side_ft is not None:
    aligned_ft = side_ft - vidshift - go[trial]  # recomputed
```

iii. The repeated frame time alignment is minor overhead since `side_ft` is already loaded and the alignment is just subtraction.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads `bp.no` explicitly even though it could be inferred from not-hit and not-miss. The `NdroppedFrames` check loads extra data from the HDF5 file. The `fill_nearest` function and baseline correction for paw velocity implement reference-style processing that may not be necessary for the downstream categorical output. The code also tracks and stores extensive metadata (per-unit quality labels, source trial indices, etc.) that is not used by the decoder.

ii.
```python
# Extensive metadata stored:
session_info = {
    "session_id": session_id,
    "retained_source_trial_indices_1based": (retained + 1).tolist(),
    "units": unit_info,
    ...
}
```

iii. The metadata is useful for auditing but not consumed by downstream analysis.
