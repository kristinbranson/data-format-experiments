# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses a hardcoded list of 12 sessions (`SESSION_SPECS`) derived from the paper's Figure 8 context analysis script (`Figure8a_thru_c.m`), rather than the full 44-session roster from the authors' `load<ANM>_ALMVideo.m` scripts. Each session is loaded using the `mat73` library. Sessions are only searched in `Ephys_Behavior`, not in `RandomizedDelay_Ephys_Behavior`. Motion energy is loaded separately from `motionEnergy_*.mat` files using `scipy.io.loadmat`.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    SessionSpec("JEB7", "2021-04-30", 1),
    SessionSpec("EKH1", "2021-08-07", 2),
    SessionSpec("EKH3", "2021-08-11", 2),
    SessionSpec("JGR2", "2021-11-16", 1),
    SessionSpec("JGR2", "2021-11-17", 1),
    SessionSpec("JGR3", "2021-11-18", 1),
    SessionSpec("JEB19", "2023-04-21", 1),
    SessionSpec("JEB19", "2023-04-20", 1),
    SessionSpec("JEB19", "2023-04-19", 1),
    SessionSpec("JEB19", "2023-04-18", 1),
]

# Loading:
obj = mat73.loadmat(spec.data_path)["obj"]
me = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)['me']
```

iii. The agent identified the Figure 8 context cohort of 12 sessions from the paper's analysis scripts and deliberately chose this subset. The agent stated: "the Figure 8 context analysis explicitly uses the 12 sessions from JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, and JEB19." The agent also confirmed "12 sessions and 520 filtered units, which is very close to the paper's 522-unit context cohort."

## 1-b. How are the data split into subjects?

i. The animal name is extracted from the `SessionSpec.animal` field, which is the first part of the session name (e.g., `JEB6`). Subjects are accumulated in encounter order as sessions are processed, with a `subject_to_idx` mapping.

ii.
```python
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
```

iii. The animal ID is encoded in the session spec directly, taken from the filename convention used by the authors.

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` entry corresponds to one session. Only one data folder (`Ephys_Behavior`) is searched, so only fixed-delay sessions are included. The 12 sessions come from 7 animals. Each session becomes one element of the `neural`, `input`, and `output` lists.

ii.
```python
DATA_DIR = Path("/app/data/Ephys_Behavior")

@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probe: int

    @property
    def data_path(self) -> Path:
        return DATA_DIR / f"data_structure_{self.stem}.mat"
```

iii. The agent chose to use the Figure 8 context analysis sessions, which are all in the fixed-delay folder. The agent noted this cohort was "very close to the paper's 522-unit context cohort."

## 1-d. How are the data split into trials?

i. Trials are defined by the `Ntrials` field in `bp`. Each trial has associated fields indexed by trial number. Spike times carry their trial assignment in the `trial` field.

ii.
```python
ntrials = int(bp["Ntrials"])
# ...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
```

iii. The trial structure is read directly from the Bpod metadata, consistent with the raw data format.

## 1-e. How are trials filtered based on quality controls?

i. The AI excludes early-lick trials, no-response (ignore) trials, and photostimulation trials. This is more aggressive than the reference, which keeps ignore trials and only drops early-lick and photostim trials. The AI also does not check for trials past the end of the recording.

ii.
```python
def build_keep_trial_mask(bp: dict) -> np.ndarray:
    hit = as_array(bp["hit"], bool).ravel()
    miss = as_array(bp["miss"], bool).ravel()
    early = as_array(bp["early"], bool).ravel()
    no = as_array(bp["no"], bool).ravel()
    stim = get_stim_enable(bp)
    return (hit | miss) & ~early & ~no & ~stim
```

iii. The agent stated: "Excluded early-lick, no-response, and stim-enable trials from the exported trial set." The agent did not explain why ignore trials were excluded beyond the paper's standard practice.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `obj.clu[probe]`, specifically the `trial`, `trialtm`, and `quality` fields of each cluster. The go cue times `bp.ev.goCue` are used for alignment.

ii.
```python
probe = obj["clu"][spec.probe - 1]
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. The agent used the same raw spike data fields as the reference, deriving neural activity from the spike-sorted cluster data.

## 2-b. How is the `neural` data processed?

i. Spike counts are binned at 10 ms resolution (DT = 1/100), converted to firing rates (Hz), then smoothed with a causal half-Gaussian kernel of window size 15 with reflect padding. This differs from the reference which uses 5 ms bins and a symmetric Gaussian with 14 ms sigma. The AI's smoothing zeros out the first half of the Gaussian kernel to make it causal, matching the MATLAB code's `my_smooth` function.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
SMOOTH = 15

bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

The `my_smooth` function:
```python
def my_smooth(x, n, bctype="none"):
    kernel = gausswin(n)
    kernel[: n // 2] = 0
    kernel /= kernel.sum()
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```

iii. The agent stated it "applied the MATLAB causal half-Gaussian smoother with window 15 and reflect padding" and "binned at 10 ms," directly porting the MATLAB code's `my_smooth` function.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filters: (1) clusters with quality labels in `['garbage', 'gabrga', 'noisy', 'real?']` are excluded (matching the reference's `findClusters.m`), and (2) units with mean firing rate <= 1 Hz are removed. The mean rate is computed across multiple conditions (hit, miss, combinations with stim/autowater/early), not just the overall mean. The AI does not include 'poor' in the drop list.

ii.
```python
def get_quality_mask(probe: dict) -> np.ndarray:
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]], dtype=object)
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad

def apply_low_fr_filter(trialdat, bp):
    conds = build_low_fr_conditions(bp)
    psth = np.zeros((trialdat.shape[0], trialdat.shape[1], len(conds)), dtype=np.float32)
    for cond_idx, cond_mask in enumerate(conds):
        trials = np.flatnonzero(cond_mask)
        if trials.size:
            psth[:, :, cond_idx] = trialdat[:, :, trials].mean(axis=2)
    mean_frs = psth.mean(axis=0).mean(axis=1)
    keep = mean_frs > LOW_FR
    return keep, mean_frs
```

iii. The agent stated it "excluded garbage/noisy/real? clusters, and removed units with mean firing rate <= 1 Hz." The condition-based FR computation mirrors the MATLAB code's `findLowFR` function more closely than the reference's simple overall mean.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment to the go cue is done by subtracting the go cue time from each spike time: `aligned = spike_times - go_cue[spike_trials]`. This is the same approach as the reference.

ii.
```python
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The agent used the standard alignment approach, subtracting goCue times from spike times.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI uses 10 ms bins (DT = 1/100) spanning -3.0 to 2.5 s, yielding 550 time bins. The reference uses 5 ms bins spanning -2.5 to 2.5 s, yielding 1000 time bins. The AI's time window starts 0.5 s earlier than the reference.

ii.
```python
TMIN = -3.0
TMAX = 2.5
DT = 1 / 100
```

iii. The agent stated it used "10 ms" bins. The MATLAB reference code uses `params.dt = 1/200` (5 ms), but the agent chose `1/100` based on what it found in some MATLAB scripts. The window of -3.0 to 2.5 differs from the reference's `params.tmin = -2.5`, `params.tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is the time axis itself, computed from the bin edges. It is the center of each time bin, spanning from -3.0 to 2.5 s at 10 ms resolution.

ii.
```python
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
# ...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The agent computed the time axis from the bin parameters, same conceptual approach as the reference but with different bin size and window.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond computing bin centers from the time window parameters. The same time array is tiled for each trial.

ii.
```python
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. Same approach as the reference.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is the same grid used for binning the neural data, so alignment is inherent. Both use the same `TMIN`, `TMAX`, and `DT` parameters.

ii.
```python
EDGES, TIME = build_time_axis()
# Used for both neural binning and input
```

iii. Same conceptual approach as the reference.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp.R`, which indicates whether the instructed side is right. The AI uses `bp.R` directly as the lick direction (0 = left, 1 = right), without considering the trial outcome.

ii.
```python
def build_output_constants(bp, keep_trials):
    lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
    # ...
```

iii. The agent maps `bp.R` directly to lick direction. Since ignore trials are already excluded, every remaining trial is either a hit or miss, and the agent treats `R` as the *instructed* direction. However, on miss trials the animal licked the *opposite* of the instructed side, so this is actually the instructed side, not the lick direction.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI takes `bp.R` directly: right=1, left=0. There is no "no lick" class because ignore trials are excluded. The reference derives lick direction from the combination of instructed side (`R`) and outcome (`hit`/`miss`), assigning the opposite direction on miss trials and a "no lick" class for ignores.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
# Stored as: 0=left, 1=right (no "no lick" class)
```

iii. The agent did not explain the reasoning for this approach in the trajectory. By excluding ignore trials and using `R` directly, the agent conflates instructed side with lick direction on miss trials.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Derived from `bp.autowater`. The AI inverts it: `context = (~autowater).astype(np.int64)`, so autowater=True becomes WC=0, and autowater=False becomes DR=1.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The agent correctly maps autowater to WC and non-autowater to DR, consistent with the paper and reference.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. A direct boolean inversion and cast: WC=0 (autowater), DR=1 (not autowater). This matches the reference's encoding.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. Same logic as the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Derived from `bp.hit` directly. The AI uses hit=1 as correct and hit=0 as incorrect. Since ignore trials are excluded, the outcome is binary rather than three-class.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. The agent excluded ignore trials upstream, so outcome becomes binary (correct/incorrect) rather than the reference's three-class (incorrect/correct/ignore).

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Direct use of `bp.hit` as a binary: 0=incorrect, 1=correct. No "ignore" class since those trials are dropped. The output_values list reflects only two classes: `["incorrect", "correct"]`.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
# output_values: ["incorrect", "correct"]
```

iii. The agent's approach is simpler than the reference because ignore trials are excluded, but the instructions explicitly specify three outcome classes: "incorrect, correct, ignore."

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Derived from DLC tracking in `obj.traj`, using multiple tongue-related features from both cameras: `tongue`, `left_tongue`, `right_tongue` from the side camera (view 0), and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` from the bottom camera (view 1). The reference uses only `tongue` (side) and `top_tongue` (bottom).

ii.
```python
TONGUE_FEATURES = [
    (0, "tongue"),
    (0, "left_tongue"),
    (0, "right_tongue"),
    (1, "top_tongue"),
    (1, "topleft_tongue"),
    (1, "bottom_tongue"),
    (1, "bottomleft_tongue"),
]
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The agent stated: "Mean speed across side-camera tongue/left_tongue/right_tongue and bottom-camera top/bottom tongue landmarks."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates DLC positions onto the neural time axis using `interp_with_nan`, then computes velocity as the first difference (`np.gradient`). For tongue features, NaN values in the velocity are replaced with zeros (`np.nan_to_num`). The speeds from all tongue features are averaged. The result is then binarized at the session 50th percentile (ignoring zeros for threshold computation). This differs significantly from the reference, which computes velocity at frame resolution within contiguous tracked runs, bins to 5 ms, normalizes each camera view by its 90th percentile, and averages the two views.

ii.
```python
# Position interpolation
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
ypos[:, trix] = interp_with_nan(old_t, ts[:, 1], taxis)
# For tongue: NaN -> 0
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)
# Speed
speeds.append(np.sqrt(xvel**2 + yvel**2))
# Average across features
out = np.nanmean(stacked, axis=0)
out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
# Binarize
tongue_binary, tongue_threshold = binarize_session_signal(tongue_speed, keep_trials, ignore_zeros_for_threshold=True)
```

iii. The agent stated it "Interpolated DLC traces and motion energy onto the neural time grid after video-offset correction; computed feature velocities from first differences using the same missing-value handling as the MATLAB code."

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses binary thresholding at the 50th percentile, ignoring zeros when computing the threshold. Values >= threshold get class 1, values < threshold get class 0. There is no "not visible" (class 2) category because NaN values are replaced with 0 before thresholding. The output_values list shows only two classes: `["below_p50", "at_or_above_p50"]`.

ii.
```python
def binarize_session_signal(signal, keep_trials, ignore_zeros_for_threshold=False):
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
    return binary, threshold
```

iii. The agent does not include a "not visible" class, contrary to the instructions which specify three classes (0: below 50th percentile, 1: >= 50th percentile, 2: not visible).

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a video offset using `find_video_offset` (same bitcode-based approach as the reference), then interpolates DLC positions directly onto the neural time axis after correcting frame times. This contrasts with the reference which computes velocity at frame resolution and then bins.

ii.
```python
def find_video_offset(obj):
    bit_start = mode_scalar(obj["bp"]["ev"]["bitStart"])
    bitstart = as_array(bitcode["bitstart"], np.float64).ravel()
    fs = float(np.asarray(obj["sglx"]["fs"]).reshape(-1)[0])
    vid_file_offset = mode_scalar(bitstart) / fs
    return float(vid_file_offset - bit_start)

old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. The agent correctly uses the video offset correction and go cue alignment, but the interpolation approach differs from the reference's frame-resolution processing.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Derived from DLC tracking of two paw features from the bottom camera: `top_paw` and `bottom_paw`. The reference uses only `top_paw`.

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The agent chose to include both paw features. The reference explicitly chose only `top_paw` because `bottom_paw` drops out through the delay epoch.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue: positions are interpolated onto the neural time axis, but for non-tongue features, missing values are filled using nearest-neighbor interpolation (`fill_nearest_1d`) rather than left as NaN. Velocity is computed as first differences with a baseline drift subtraction. The speeds from both paw features are averaged.

ii.
```python
# For non-tongue features: nearest-fill NaN
xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
ypos[:, trix] = fill_nearest_1d(ypos[:, trix], fill_value=0.0)
# Velocity with baseline subtraction
basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
xvel[:, trix] = xvel[:, trix] - basederiv[0]
yvel[:, trix] = yvel[:, trix] - basederiv[0]  # note: subtracts x baseline from y
```

iii. The agent ported the MATLAB code's velocity computation, including the baseline subtraction. There appears to be a bug: `basederiv[0]` (x-component) is subtracted from both x and y velocities.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Binary thresholding at the 50th percentile without excluding zeros. Values >= threshold get class 1, values < threshold get class 0. No "not visible" class. The output_values list shows only two classes.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

iii. Same approach as tongue but without the zero-exclusion for threshold computation. Missing the "not visible" class specified in the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: DLC positions are interpolated directly onto the neural time grid after video offset correction.

ii.
```python
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
# then fill_nearest_1d for non-tongue
```

iii. Same alignment approach as all other camera-derived signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Derived from `motionEnergy_<anm>_<date>.mat` files, reading `me.data` (with unwrapping for nested structures). The motion energy is also aligned using the side camera's frame times.

ii.
```python
def load_motion_energy(obj, motion_energy_path, taxis, align_times):
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
    raw_data = me.data
    if hasattr(raw_data, "data"):
        raw_data = raw_data.data
```

iii. The agent correctly loads the motion energy from the separate file and handles the nested structure.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Motion energy values are interpolated onto the neural time axis using `interp_with_nan`, then missing values are filled with nearest-neighbor interpolation (`fill_nearest_1d`). The reference simply averages frames into bins without interpolation or filling.

ii.
```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The agent interpolates and fills, while the reference bins and leaves NaN where no frames fall.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Binary thresholding at the 50th percentile. Values >= threshold get class 1, values < threshold get class 0. No "no video" class. The instructions specify three classes including "no video" (class 2).

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. The agent uses the same binary thresholding as tongue and paw, missing the "no video" class.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned using the side camera's frame times with video offset correction, then interpolated onto the neural time axis.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. Same video offset correction approach as the reference.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI uses several strategies: (1) For tongue features, NaN velocities are replaced with 0. (2) For non-tongue features (paw), positions are filled using nearest-neighbor interpolation. (3) Trials with all-NaN frame times use a fallback synthetic frame timing (frames at 400 Hz with a 0.5s offset). (4) Motion energy missing values are filled with nearest-neighbor interpolation. This contrasts with the reference, which preserves NaN/missing information through a "not visible" class.

ii.
```python
# Fallback for missing frame times
use_fallback = frame_times.size == 0 or np.all(~np.isfinite(frame_times))
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]

# Tongue: NaN -> 0
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)

# Paw: nearest-fill
xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
```

iii. The agent chose to fill missing values rather than preserve them as a separate class, which means the "not visible" information is lost in the converted data.

## 11-a. What are the most time-consuming steps of the code?

i. Loading the `.mat` files with `mat73` and the per-trial DLC position interpolation are the most expensive operations. The agent noted the conversion was "spending most of its time in session-by-session spike binning and video interpolation."

ii.
```python
obj = mat73.loadmat(spec.data_path)["obj"]
# Per-trial interpolation loop
for trix in range(ntrials):
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. The agent noted: "The first pass is slower than I want, likely because mat73 is rehydrating several large session objects and the Python port is still too literal."

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `find_position` and `find_velocity` iterate over all trials for each feature. The spike binning uses `np.add.at` in a per-unit loop rather than vectorized histogram operations. The `my_smooth` function loops over columns.

ii.
```python
# Per-unit spike binning loop
for unit_pos, unit_idx in enumerate(keep_units):
    # ...
    np.add.at(counts, (spike_trials, bins), 1.0)

# Per-trial position loop
for trix in range(ntrials):
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)

# Per-column smoothing loop
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")
```

iii. The agent acknowledged the code was slow but prioritized correctness over optimization.

## 11-c. What processing does the code repeat multiple times?

i. The video offset is computed twice: once in `compute_feature_speed` (via `find_position` which calls `find_video_offset` implicitly through the vidshift parameter) and once in `load_motion_energy`. The `get_quality_mask` function is called twice (once during `build_aligned_trialdat` and once separately). Trial-level metadata like `go_cue`, `hit`, `miss`, `early` etc. are extracted multiple times across different functions.

ii.
```python
# Video offset computed in main loop:
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
# load_motion_energy also calls find_video_offset internally:
vidshift = find_video_offset(obj)
```

iii. The agent did not address this redundancy.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI loads and processes all DLC features (7 tongue features, 2 paw features) when the reference uses only 2 tongue features and 1 paw feature. The `build_low_fr_conditions` function computes 7 different condition masks for firing rate filtering, which is more elaborate than needed. The `fill_nearest_1d` interpolation on paw positions creates artificial velocity signals that are then discarded by the thresholding. The code also builds a sample dataset that is not required by the instructions.

ii.
```python
# 7 tongue features (reference uses 2)
TONGUE_FEATURES = [
    (0, "tongue"), (0, "left_tongue"), (0, "right_tongue"),
    (1, "top_tongue"), (1, "topleft_tongue"), (1, "bottom_tongue"), (1, "bottomleft_tongue"),
]

# Sample dataset creation
sample_data = build_sample_dataset(full_data)
save_pickle(args.sample_out, sample_data)
```

iii. The agent processed more features than necessary, increasing computation time without clear benefit for the decoder task.
