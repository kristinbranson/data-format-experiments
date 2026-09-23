# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI only loads sessions from the `Ephys_Behavior` folder (fixed-delay task), selecting sessions whose animal name appears in a hardcoded `ALM_PROBE` dictionary mapping 7 animals to their ALM probe number. It discovers session files by globbing for `data_structure_*.mat` in that single directory, filtering to the 7 known animals. All files are opened with `h5py` (HDF5/v7.3 format). Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`. This yields 12 sessions total.

ii.
```python
DATA_DIR = Path(__file__).resolve().parent / "data" / "Ephys_Behavior"

ALM_PROBE = {
    "JEB6": 2,
    "JEB7": 1,
    "EKH1": 2,
    "EKH3": 2,
    "JGR2": 1,
    "JGR3": 1,
    "JEB19": 1,
}

def session_files(data_dir: Path) -> list[tuple[Path, str, str]]:
    sessions = []
    pattern = re.compile(r"data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat$")
    for path in sorted(data_dir.glob("data_structure_*.mat")):
        match = pattern.match(path.name)
        if match and match.group(1) in ALM_PROBE:
            sessions.append((path, match.group(1), match.group(2)))
    if len(sessions) != 12:
        raise RuntimeError(f"Expected 12 two-context sessions, found {len(sessions)}")
    return sessions
```

iii. The AI identified this as the "two-context neural cohort" from the paper's Figure 8 scripts (12 sessions, six mice, 522 units). It stated: "The paper's exact two-context neural cohort is identifiable from the figure scripts: 12 sessions from six mice."

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the filename pattern `data_structure_<animal>_<date>.mat` by regex capture group. A sorted set of unique animal names forms the `subjects` list, and `subject_idx` maps each session to its index.

ii.
```python
pattern = re.compile(r"data_structure_([^_]+)_(\d{4}-\d{2}-\d{2})\.mat$")
# ...
subjects = sorted({animal for _, animal, _ in files})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx.append(subject_lookup[animal])
```

iii. The animal name is reliably encoded in the filename.

## 1-c. How are the data split into sessions?

i. One session equals one `.mat` file in `data/Ephys_Behavior/` whose animal name is in `ALM_PROBE`. Only files from the single `Ephys_Behavior` folder are used; the `RandomizedDelay_Ephys_Behavior` folder is entirely excluded. This yields 12 sessions from 7 animals.

ii.
```python
def session_files(data_dir: Path) -> list[tuple[Path, str, str]]:
    sessions = []
    for path in sorted(data_dir.glob("data_structure_*.mat")):
        match = pattern.match(path.name)
        if match and match.group(1) in ALM_PROBE:
            sessions.append((path, match.group(1), match.group(2)))
```

iii. The AI interpreted the task as targeting the "two-context electrophysiology cohort" used in the paper's Figure 8 analyses.

## 1-d. How are the data split into trials?

i. Within each session, a trial corresponds to one entry indexed by the go cue array `bp.ev.goCue`. The total number of trials is `go.size`. Per-trial fields (`hit`, `miss`, `no`, `early`, `autowater`, `R`, `stim.enable`) are read from `bp` and indexed by trial number.

ii.
```python
go = np.asarray(bp["ev/goCue"]).ravel().astype(np.float64)
n_trials = go.size
hit = np.asarray(bp["hit"]).ravel().astype(bool)
# ... etc
```

iii. The Bpod data directly defines trials, one per go cue entry.

## 1-e. How are trials filtered based on quality controls?

i. Trials with early licks (`early`), photostimulation (`stim.enable`), non-finite go cue times, or trials that are not hit/miss/ignore are excluded. The mask is: `~early & ~stim & np.isfinite(go) & (hit | miss | ignore)`.

ii.
```python
trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
kept_trials = np.flatnonzero(trial_mask)
```

iii. The AI stated: "Paper analyses use control trials and omit early licks. Ignore trials remain because ignore is a required decoder output in this task."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `obj.clu{probe}` cluster data: for each unit, `trial` (1-based spike trial index) and `trialtm` (spike time relative to trial start). Also `bp.ev.goCue` for alignment. The probe number comes from the `ALM_PROBE` dictionary.

ii.
```python
cluster_group = h5[h5["obj/clu"][probe_number - 1, 0]]
spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
trial_time = referenced_array(h5, cluster_group["trialtm"][unit, 0]).ravel().astype(np.float64)
aligned_time = trial_time - go[spike_trial]
```

iii. This follows the reference code's spike alignment approach.

## 2-b. How is the `neural` data processed?

i. Spikes are binned at 10 ms resolution (DT=0.01) into 500 bins spanning -2.5 to +2.5 s. The counts are then smoothed with a causal Gaussian kernel (15 bins, reproducing the MATLAB `gausswin(15)` with the left half zeroed out) applied via `lfilter`, then divided by the bin width to convert to firing rates (Hz). A "reflect" boundary condition is approximated by prepending the first 15 samples.

ii.
```python
DT = 0.01
SMOOTH_BINS = 15

def gaussian_causal_kernel(n: int = SMOOTH_BINS) -> np.ndarray:
    x = np.arange(n, dtype=np.float64) - (n - 1) / 2
    kernel = np.exp(-0.5 * (2.5 * x / ((n - 1) / 2)) ** 2)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    return kernel[n // 2 :].astype(np.float32)

def smooth_rates(counts: np.ndarray) -> np.ndarray:
    padded = np.concatenate((counts[..., :SMOOTH_BINS], counts), axis=-1)
    filtered = lfilter(CAUSAL_KERNEL, [1.0], padded, axis=-1)
    return (filtered[..., SMOOTH_BINS:] / DT).astype(np.float32)
```

iii. The AI stated: "Neural activity will use the repository's 10 ms bins and 15-bin causal Gaussian smoothing over -2.5 to +2.5 s."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) Units whose quality label (case-sensitive) is in `{garbage, gabrga, noisy, real?}` are excluded. (2) Units whose mean firing rate (computed via a condition-based PSTH approach) is <= 1 Hz are excluded.

ii.
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
# ...
quality = matlab_char(h5, cluster_group["quality"][unit, 0])
if quality in REJECTED_QUALITIES:
    continue
# ...
mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
if mean_rate > 1.0:
    selected_units.append(...)
```

iii. The AI explicitly stated: "Deliberately case-sensitive, matching findClusters.m/isMember." The firing rate calculation uses condition-based PSTHs matching the `removeLowFRClusters` approach in the paper's code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to go cue is done by subtracting `go[trial]` from `trialtm`: `aligned_time = trial_time - go[spike_trial]`. This puts spikes in seconds from go cue onset.

ii.
```python
aligned_time = trial_time - go[spike_trial]
```

iii. This matches the reference's `alignSpikes.m`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The bin size is 10 ms (DT=0.01), producing 500 time bins over -2.5 to +2.5 s. No rebinning is applied after the initial binning.

ii.
```python
DT = 0.01
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size  # 500
```

iii. The AI chose 10 ms to match the MATLAB code's `params.dt = 1/100`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. This is a synthetic variable: the center of each time bin in the window [-2.5, 2.5] s, computed from the bin parameters.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
time_input = TIME.astype(np.float32)[None, :]
```

iii. By construction, the time axis is defined by the binning grid itself.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing; it is the bin center array itself, with shape (1, 500).

ii.
```python
time_input = TIME.astype(np.float32)[None, :]
input_trials.append(time_input.copy())
```

iii. N/A

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is identical to the neural binning grid, so alignment is inherent.

ii.
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```

iii. N/A

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Three per-trial fields: `bp.hit`, `bp.miss` (originally also `bp.no` for ignore), and `bp.R` (right target flag).

ii.
```python
hit = np.asarray(bp["hit"]).ravel().astype(bool)
miss = np.asarray(bp["miss"]).ravel().astype(bool)
ignore = np.asarray(bp["no"]).ravel().astype(bool)
right_target = np.asarray(bp["R"]).ravel().astype(bool)
```

iii. The lick direction is inferred from the instructed side and outcome.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. Hit trials: lick = instructed side. Miss trials: lick = opposite of instructed side. Ignore trials: lick = none (2). Codes: left=0, right=1, none=2.

ii.
```python
if ignore[original_trial]:
    lick_direction = 2  # none
elif hit[original_trial]:
    lick_direction = 1 if right_target[original_trial] else 0
else:
    lick_direction = 0 if right_target[original_trial] else 1
```

iii. The AI stated: "derive lick direction from target side plus outcome, so incorrect trials are labeled by the animal's actual opposite-side response."

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. The per-trial field `bp.autowater`.

ii.
```python
autowater = np.asarray(bp["autowater"]).ravel().astype(bool)
```

iii. Autowater indicates the WC context.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. Autowater=True maps to WC (0), otherwise DR (1).

ii.
```python
context = 0 if autowater[original_trial] else 1
```

iii. Direct mapping following the instructions.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. `bp.hit`, `bp.miss`, and `bp.no` (ignore).

ii.
```python
hit = np.asarray(bp["hit"]).ravel().astype(bool)
miss = np.asarray(bp["miss"]).ravel().astype(bool)
ignore = np.asarray(bp["no"]).ravel().astype(bool)
```

iii. These three flags are mutually exclusive.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Hit=correct (1), miss=incorrect (0), ignore=ignore (2).

ii.
```python
if ignore[original_trial]:
    outcome = 2
elif hit[original_trial]:
    outcome = 1
else:
    outcome = 0
```

iii. Follows the task instructions.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. DLC tracking data in `obj.traj`. The side camera (index 0) `tongue` feature is used. Per-trial `ts` (x, y, likelihood per frame) and `frameTimes` are extracted, plus `sglx` bitcode data for video offset.

ii.
```python
trajectory_refs = np.asarray(h5["obj/traj"]).ravel()
side = h5[trajectory_refs[0]]
tongue_index = find_feature_indices(h5, side, "tongue")
tongue = trajectory_signal(h5, side, int(trial), tongue_index, go[trial], offset)
```

iii. The AI uses only the side camera for tongue, whereas the reference uses both side and bottom cameras.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. DLC x/y positions are interpolated onto the neural time axis (10 ms bins) using linear interpolation that preserves NaN intervals. Speed is computed as `np.hypot(np.gradient(x), np.gradient(y))` - a simple frame-to-frame Euclidean speed on the interpolated grid. No smoothing is applied to positions before differentiation. No normalization across cameras since only one camera is used.

ii.
```python
def speed_from_position(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.gradient(x), np.gradient(y))
    speed[~(np.isfinite(x) & np.isfinite(y))] = np.nan
    return speed

x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
```

iii. The AI computes velocity after interpolation rather than at the original frame rate.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Per-session 50th percentile of finite values is the threshold. Values >= threshold get class 1, below get class 0, NaN (not visible) get class 2.

ii.
```python
def categorize_session_signal(signals: list[np.ndarray]) -> ...:
    threshold = float(np.percentile(np.concatenate(finite_parts), 50))
    out[visible] = (signal[visible] >= threshold).astype(np.int64)
    # NaN -> class 2
```

iii. Follows the instructions for per-session median discretization.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The video offset is computed per session using `sglx.bitcode.bitstart / fs` vs `bp.ev.bitStart`, taking the median of each. Frame times are corrected by subtracting offset and go cue, then x/y positions are linearly interpolated onto the neural TIME grid.

ii.
```python
def video_offset(h5: h5py.File) -> float:
    return float(np.nanmedian(video_bit_start / fs) - np.nanmedian(bit_start))

x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
```

iii. The AI stated it used "the repository's per-session video offset before interpolation onto the neural time axis."

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. DLC tracking from the bottom camera (index 1), `top_paw` feature.

ii.
```python
bottom = h5[trajectory_refs[1]]
paw_index = find_feature_indices(h5, bottom, "top_paw")
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. Same approach as tongue but from bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same as tongue: interpolate x/y onto neural time grid, compute speed as `np.hypot(np.gradient(x), np.gradient(y))`. No smoothing.

ii.
```python
paw_velocity.append(
    np.full(N_TIME, np.nan) if paw is None else speed_from_position(paw[0], paw[1])
)
```

iii. Identical processing pipeline to tongue.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same as tongue: per-session 50th percentile threshold, classes 0/1/2.

ii.
```python
paw_cat, paw_threshold = categorize_session_signal(paw_velocity)
```

iii. Follows the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: video offset correction, then linear interpolation onto neural TIME grid.

ii.
```python
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. Same alignment pipeline.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. `motionEnergy_<animal>_<date>.mat` file, loaded via `scipy.io.loadmat`. The `me.data` field (handling possible nested wrapping).

ii.
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    mat = loadmat(path, simplify_cells=True)
    raw = mat["me"]["data"]
    if isinstance(raw, dict):
        raw = raw["data"]
```

iii. The standalone motion energy file is used.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values (one per frame) are interpolated onto the neural time grid using linear interpolation from the side camera frame times. After interpolation, NaN values are filled with nearest-neighbor interpolation (`nearest_fill`).

ii.
```python
me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
motion_energy.append(nearest_fill(me))
```

iii. The nearest-fill step fills NaN values at the edges of the interpolated trace.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same as tongue/paw: per-session 50th percentile threshold, classes 0/1/2.

ii.
```python
motion_cat, motion_threshold = categorize_session_signal(motion_energy)
```

iii. Follows the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy uses the side camera's frame times (from the tongue trajectory signal). After video offset correction, the values are linearly interpolated onto the neural TIME grid, then nearest-filled.

ii.
```python
n = min(raw_me.size, tongue[2].size)
me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
motion_energy.append(nearest_fill(me))
```

iii. This follows the reference MATLAB code's `loadMotionEnergy.m` which uses `interp1` and `fillmissing(..., 'nearest')`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several approaches: (1) If a trajectory signal returns None (e.g., dropped frames, bad dimensions), the trial gets NaN for that variable, which maps to "not visible" class 2. (2) Motion energy applies nearest-neighbor fill after interpolation. (3) Trials with non-finite go cues are excluded. (4) Spikes with trial indices outside valid range are filtered out.

ii.
```python
tongue_velocity.append(
    np.full(N_TIME, np.nan) if tongue is None else speed_from_position(tongue[0], tongue[1])
)
# ...
motion_energy.append(nearest_fill(me))
# ...
valid = (spike_trial >= 0) & (spike_trial < n_trials)
```

iii. Missing video data maps to the "not visible" category rather than being interpolated.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the HDF5 files and reading per-unit spike data are the most expensive operations. The motion energy loading and trajectory signal extraction also involve disk I/O and per-trial array reads.

ii.
```python
with h5py.File(path, "r") as h5:
    # ... all processing happens inside a single h5py context
```

iii. File I/O dominates the runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit loop for reading spike data and computing counts could potentially be vectorized if all units were read together. The per-trial loop for trajectory signals and motion energy could also be vectorized if frame counts were uniform.

ii.
```python
for unit in range(cluster_group["quality"].shape[0]):
    # ... reads each unit's data individually
for trial in kept_trials:
    tongue = trajectory_signal(...)
    paw = trajectory_signal(...)
```

iii. HDF5 reference-based storage forces per-unit/per-trial reads.

## 11-c. What processing does the code repeat multiple times?

i. The `trajectory_signal` function is called separately for tongue and paw on each trial, meaning `NdroppedFrames` is read twice per trial (once per feature). The video offset is computed once per session.

ii.
```python
tongue = trajectory_signal(h5, side, int(trial), tongue_index, go[trial], offset)
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. Each call reads separate HDF5 references from different camera views, so the duplication is minimal.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes condition-based PSTH masks (`condition_masks`) for the firing rate filter that go beyond what is needed for just a mean rate computation. It also stores per-unit qualities and mean rates in the session result dict, then immediately pops them during assembly. The `nearest_fill` on motion energy fills NaN edges that may not affect downstream results since they could be handled by the "no video" class.

ii.
```python
condition_masks = [
    hit | miss | ignore,
    hit & ~autowater,
    hit & autowater,
    miss & ~autowater,
    miss & autowater,
    hit & ~autowater & ~early,
    hit & autowater & ~early,
]
# ...
result.pop("qualities")
result.pop("mean_rates_hz")
```

iii. The condition-based PSTH firing rate computation is more complex than a simple mean rate, but matches the paper's `removeLowFRClusters` approach.
