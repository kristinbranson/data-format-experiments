# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from `.mat` files in `/app/data/Ephys_Behavior/` using `mat73.loadmat()`. Each session has a `data_structure_<animal>_<date>.mat` file and a companion `motionEnergy_<animal>_<date>.mat` file. The AI iterates over a hardcoded list of 12 `SessionSpec` objects (animal, date, probe index), loading each session's `obj` dictionary. It extracts `obj['bp']` for behavioral/trial data, `obj['clu'][probe-1]` for spike clusters, and `obj['traj']` for DLC tracking data.

ii.
```python
SESSION_SPECS = [
    SessionSpec("JEB6", "2021-04-18", 2),
    SessionSpec("JEB7", "2021-04-29", 1),
    # ... 12 total
]

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    bp = obj["bp"]
    # ...
    probe = obj["clu"][spec.probe - 1]
    me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. The AI states in CONVERSION_NOTES.md: "The session roster follows the paper's Figure 8 context-analysis loader in Figure8a_thru_c.m." The trajectory shows the agent read `Figure8a_thru_c.m` and all individual loader files (e.g., `loadJEB6_ALMVideo.m`) to determine the exact sessions and probe numbers.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field in each `SessionSpec`. The AI maintains a `subject_to_idx` mapping and a `subjects` list, assigning each new animal name a sequential index. The final `subject_idx` array maps each session to its subject.

ii.
```python
subject_to_idx: dict[str, int] = {}
subjects: list[str] = []
# ...
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
```

iii. The AI notes in CONVERSION_NOTES.md that the released code roster yields 7 distinct animal IDs (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19), while the paper text reports 6 mice. The AI kept the code roster because "it is explicit and task-specific."

## 1-c. How are the data split into sessions?

i. Each `SessionSpec` in the hardcoded list of 12 represents one session. The AI processes them sequentially, and each produces one entry in the `neural`, `input`, `output`, `brain_region_idx`, and `subject_idx` lists. Some animals contribute multiple sessions (e.g., JEB7 has 2 sessions, JEB19 has 4 sessions).

ii.
```python
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    # ... process session ...
    full_data["neural"].append(neural_trials)
    full_data["input"].append(input_trials)
    full_data["output"].append(output_trials)
    full_data["subject_idx"].append(subject_idx)
```

iii. The trajectory shows the AI determined the session roster from `Figure8a_thru_c.m`, which loads 12 sessions using individual loader functions. Each loader specifies which probe to use and which .mat files to load.

## 1-d. How are the data split into trials?

i. Within each session, the AI extracts all trials from the session object (`ntrials = int(bp["Ntrials"])`), then filters them. The kept trials are indexed via `kept_trial_indices = np.flatnonzero(keep_trials)`, and data for each kept trial is appended to the per-session lists.

ii.
```python
ntrials = int(bp["Ntrials"])
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)
# ...
for out_pos, trial_idx in enumerate(kept_trial_indices):
    neural_trials.append(trialdat[:, :, trial_idx].T.astype(np.float32))
    input_trials.append(TIME[np.newaxis, :].astype(np.float32))
    output_trials.append(np.vstack([...]))
```

iii. The AI documented that trial counts per session range from 146 to 322, totaling 2415 trials across 12 sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI filters trials using `build_keep_trial_mask`, which keeps only `(hit | miss) & ~early & ~no & ~stim` trials. This excludes early-lick trials, no-response trials, and photostimulation-enabled trials. Both correct (hit) and incorrect (miss) trials from both DR and WC contexts are kept.

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

iii. The AI documented in CONVERSION_NOTES.md: "Excluded early-lick trials, no-response trials, and stim.enable trials." This matches the paper methods text which states early-lick and ignore (no-response) trials are omitted.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times in `obj['clu'][probe_index]`. Specifically, the fields `probe['trial']` (1-based trial assignment for each spike), `probe['trialtm']` (trial-relative spike times), and `probe['quality']` (cluster quality labels) are used.

ii.
```python
probe = obj["clu"][spec.probe - 1]
# In build_aligned_trialdat:
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
```

iii. The trajectory shows the agent read `WorkingWithDataObjs.m` to understand the data structure: `obj.clu` contains spike cluster data with fields `trial`, `trialtm`, and `quality`.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to the go cue, binned into 10ms bins over a [-3.0, 2.5]s window (550 time bins), converted to firing rates (counts/dt), and smoothed with a causal half-Gaussian kernel (window=15, reflect boundary). The smoothing matches the MATLAB `mySmooth.m` implementation: a `gausswin(N)` kernel with the first half zeroed out and then normalized.

ii.
```python
TMIN = -3.0; TMAX = 2.5; DT = 1/100; SMOOTH = 15

aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
# ... bin spikes ...
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)

def my_smooth(x, n, bctype="none"):
    kernel = gausswin(n)
    kernel[:n // 2] = 0  # causal
    kernel /= kernel.sum()
    # convolve with 'same' mode, reflect padding
```

iii. The AI states: "Selected the Figure 8 context-session roster, aligned spikes to goCue, binned at 10 ms, applied the MATLAB causal half-Gaussian smoother with window 15 and reflect padding."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two filtering steps: (1) Quality filter removes clusters labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Low firing rate filter removes units with mean FR <= 1 Hz, computed as the mean across time and across 7 condition-specific PSTHs matching the Figure 8 analysis.

ii.
```python
def get_quality_mask(probe):
    qualities = np.array([flatten_string(q).strip().lower() for q in probe["quality"]])
    bad = np.isin(qualities, ["garbage", "gabrga", "noisy", "real?"])
    return ~bad

def apply_low_fr_filter(trialdat, bp):
    conds = build_low_fr_conditions(bp)  # 7 conditions matching Figure 8
    # compute PSTHs for each condition
    mean_frs = psth.mean(axis=0).mean(axis=1)
    keep = mean_frs > LOW_FR  # LOW_FR = 1.0
```

iii. The agent read `findClusters.m` and `removeLowFRClusters.m` to replicate these filters. It used `lowFR = 1.0` from the Figure 8 script (not the default 0.5 from `getDefaultParams.m`). Final count: 520 units vs. paper's 522.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue (`bp['ev']['goCue']`). Spike times are made relative to go cue onset by subtracting the per-trial go cue time: `aligned = spike_times - go_cue[spike_trials]`. The window spans from -3.0s to +2.5s relative to the go cue.

ii.
```python
ALIGN_EVENT = "goCue"
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The AI notes: "For WC trials the codebase uses the same field as the water-drop-aligned response cue." The trajectory confirms the agent verified that `goCue` is finite for WC trials across all sessions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 10ms (`DT = 1/100`). No rebinning is applied -- spikes are directly binned at this resolution from the raw spike times. This matches the `dt = 1/100` parameter in `Figure8a_thru_c.m`.

ii.
```python
DT = 1 / 100
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. The metadata records `time_bin_size: 10.0` (ms). The agent chose 10ms from the Figure 8 script, not the 5ms default in `getDefaultParams.m`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the time axis itself -- it is the array of time bin centers relative to the go cue onset. It does not come from a raw data variable per se; it is constructed from the alignment parameters `TMIN`, `TMAX`, and `DT`.

ii.
```python
EDGES, TIME = build_time_axis()
# TIME = edges[:-1] + DT/2, ranging from -2.995 to +2.495

input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. The AI states: "`input[trial]` is `time_from_go_cue_s`, shape `(1, 550)`." Since the alignment event is the go cue, the time axis directly encodes time from go cue onset.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No processing beyond constructing the time axis. The bin centers are computed as `edges[:-1] + DT/2` where edges span from -3.0 to +2.5 in steps of 0.01. The same time vector is used for every trial.

ii.
```python
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. This is standard -- the time from go cue is the same across all trials because all trials use the same time window relative to alignment.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time vector is identical to the neural data's time axis, as both use the same `TIME` array. Thus they are inherently aligned -- each time bin in the input corresponds to the same time bin in the neural data.

ii.
```python
# Neural uses TIME for binning edges
# Input is TIME directly
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. No separate alignment is needed since both are constructed from the same time axis.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp['R']`, which indicates the trial type / response direction (0=left, 1=right).

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
```

iii. The AI states: "`lick_direction`: constant over time, `left=0`, `right=1`, taken from `bp.R`."

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The raw `bp.R` values are directly used (cast to int64 and filtered to kept trials). The value is broadcast as a constant across all 550 time bins for each trial.

ii.
```python
np.full((1, NT), lick_direction[out_pos], dtype=np.int16),
```

iii. No additional processing. Per the instruction, lick direction is per-trial, and the AI represents it as time-constant (same value at every time bin).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp['autowater']`. The `autowater` field indicates WC (water-cued) trials. The AI inverts it so that WC=0 and DR=1.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The AI states: "`behavioral_context`: constant over time, `WC=0`, `DR=1`, derived from `bp.autowater`."

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The `autowater` boolean is inverted (`~autowater`) so that WC trials map to 0 and DR trials map to 1, matching the instruction specification (WC=0, DR=1). The value is broadcast as a constant across all time bins.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
np.full((1, NT), context[out_pos], dtype=np.int16),
```

iii. Straightforward inversion of the autowater flag.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp['hit']`, which indicates whether the trial was correct (hit=1) or incorrect (hit=0, i.e., miss).

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. The AI states: "`outcome`: constant over time, `incorrect=0`, `correct=1`, derived from `bp.hit`."

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The raw `bp.hit` values are used directly (cast to int64, filtered to kept trials). The value is broadcast as a constant across all time bins.

ii.
```python
np.full((1, NT), outcome[out_pos], dtype=np.int16),
```

iii. No additional processing beyond casting and filtering.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC (DeepLabCut) tracking data in `obj['traj']`. It uses 7 tongue-related landmarks: `tongue`, `left_tongue`, `right_tongue` from the side camera (view 0), and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` from the bottom camera (view 1). The x,y positions come from `cam['ts'][trial][:, :2, feat_idx]`, and frame timing comes from `cam['frameTimes'][trial]`.

ii.
```python
TONGUE_FEATURES = [
    (0, "tongue"), (0, "left_tongue"), (0, "right_tongue"),
    (1, "top_tongue"), (1, "topleft_tongue"),
    (1, "bottom_tongue"), (1, "bottomleft_tongue"),
]
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The agent read `getKinematicsFromVideo.m` and the DLC feature lists to identify the relevant landmarks. These match the paper's tracked body parts.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) x,y positions are interpolated onto the neural time axis using `interp_with_nan`, with video offset correction. (2) For tongue features, NaN positions are NOT filled (unlike non-tongue features). (3) Velocities are computed using `np.gradient()`. (4) NaN velocities are set to 0. (5) Speed = sqrt(xvel^2 + yvel^2). (6) The mean speed across all 7 tongue features is the final tongue velocity signal.

ii.
```python
# In find_position: tongue NaNs left in place (no fill_nearest_1d for tongue)
# In find_velocity for tongue:
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)
yvel[:, trix] = np.nan_to_num(yvel[:, trix], nan=0.0)

# In compute_feature_speed:
speeds.append(np.sqrt(xvel**2 + yvel**2))
stacked = np.stack(speeds, axis=0)
out = np.nanmean(stacked, axis=0)
```

iii. The AI matched the MATLAB code's special handling: tongue positions have NaN when not visible, and the corresponding velocities are set to 0.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue velocity is binarized per session using the 50th percentile (median) of strictly positive values (excluding zeros, which are placeholders for tongue-not-visible periods). Values >= threshold map to 1, values < threshold map to 0.

ii.
```python
tongue_binary, tongue_threshold = binarize_session_signal(
    tongue_speed, keep_trials, ignore_zeros_for_threshold=True,
)

def binarize_session_signal(signal, keep_trials, ignore_zeros_for_threshold=False):
    kept = signal[:, keep_trials]
    threshold_source = kept
    if ignore_zeros_for_threshold:
        threshold_source = kept[kept > 0]
    threshold = float(np.nanpercentile(threshold_source, 50))
    binary = (kept >= threshold).astype(np.int64)
```

iii. The AI initially found that 11/12 sessions had tongue threshold = 0.0 (because zeros dominate), so it fixed this by excluding zeros from the threshold computation. Documented as a decoder-specific adaptation.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. DLC positions are interpolated onto the neural time axis (`TIME`) using `interp_with_nan`, after correcting for video timing offset. The video offset is computed as `mode(sglx.bitcode.bitstart)/fs - mode(bp.ev.bitStart)`. This produces a tongue velocity signal at the same 550 time bins as the neural data.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. The alignment uses the same `findVideoOffset` logic as the MATLAB code.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC tracking of `top_paw` and `bottom_paw` from the bottom camera (view 1).

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The agent identified these as the paw-related DLC features from the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. For each paw feature: (1) x,y positions are interpolated onto the neural time axis with video offset correction. (2) For non-tongue features, NaN positions are filled with nearest valid values. (3) Velocities are computed using `np.gradient()`. (4) Baseline derivative (median of first differences) is subtracted from both x and y velocity (faithfully reproducing a MATLAB bug where `basederiv(1)` is used for both axes). (5) NaN velocities are filled with nearest. (6) Speed = sqrt(xvel^2 + yvel^2). (7) Mean speed across both paw features.

ii.
```python
# In find_velocity for non-tongue:
xvel[:, trix] = xvel[:, trix] - basederiv[0]
yvel[:, trix] = yvel[:, trix] - basederiv[0]  # MATLAB bug faithfully reproduced
xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
```

iii. The AI noted and reproduced the MATLAB code's use of `basederiv(1)` for both x and y subtraction.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw velocity is binarized per session using the 50th percentile (median) of all values across kept trials and timepoints. Values >= threshold map to 1, < threshold map to 0. Unlike tongue, zeros are NOT excluded from threshold computation.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

iii. Straightforward median thresholding as specified in the instructions.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue: DLC positions are interpolated onto the neural time axis using `interp_with_nan` with video offset correction. The resulting paw velocity signal is on the same 550-bin time grid as the neural data.

ii.
```python
xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
# Position then used to compute velocity, all on the same taxis
```

iii. Uses the same `findVideoOffset`-based alignment as all other kinematic signals.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from companion `motionEnergy_<animal>_<date>.mat` files. The raw data comes from `me.data` (a cell array of per-trial motion energy traces), and `me.moveThresh` provides the movement threshold.

ii.
```python
mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
me = mat["me"]
raw_data = me.data
```

iii. The AI followed `loadMotionEnergy.m` to load these companion files.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the raw motion energy trace is interpolated onto the neural time axis using `interp_with_nan`, with video offset correction. NaN values are filled with nearest valid samples.

ii.
```python
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. The AI matched the `loadMotionEnergy.m` pipeline: interpolate ME onto neural time grid using video-corrected frame times, then fill NaN with nearest.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is binarized per session using the 50th percentile (median) of all values across kept trials and timepoints. Values >= threshold map to 1, < threshold map to 0.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. Standard median thresholding as specified in the instructions.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the neural time axis using the same video offset correction as other kinematic signals. Frame times from the side camera (`obj['traj'][0]`) are used for timing, corrected by `vidshift` and aligned to the go cue.

ii.
```python
old_t = frame_times - vidshift - align_times[trix]
resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
```

iii. Same alignment approach as the MATLAB `loadMotionEnergy.m`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies: (1) Missing/NaN positions for non-tongue features are filled with nearest valid values via `fill_nearest_1d`. (2) Missing tongue positions are left as NaN, and resulting NaN velocities are set to 0. (3) Dropped video frames (NaN in `NdroppedFrames`) cause the trial to be skipped for that feature. (4) Missing frame times fall back to synthetic times at 400Hz with a 0.5s offset. (5) Spikes with non-finite times or out-of-range trial indices are excluded. (6) The `interp_with_nan` function handles NaN in both time and signal arrays, requiring at least 2 valid points. (7) Tongue velocity binarization excludes zeros to avoid degenerate thresholds.

ii.
```python
# Fallback frame times
if use_fallback:
    frame_times = np.arange(1, ts.shape[0] + 1, dtype=np.float64) / 400.0
    old_t = frame_times - 0.5 - align_times[trix]

# Spike filtering
finite = (spike_trials >= 0) & (spike_trials < ntrials) & np.isfinite(spike_times)

# NaN velocity for tongue
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)

# Fill NaN for non-tongue
xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)
```

iii. The AI's handling matches the MATLAB code patterns. The trajectory shows the agent encountered `RuntimeWarning: All-NaN slice` during velocity computation and handled it by defaulting `basederiv` to `[0.0, 0.0]`.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are: (1) Loading the large `.mat` files with `mat73.loadmat()`, which are complex HDF5 structures. (2) Building aligned trial data (`build_aligned_trialdat`), which loops over every unit and every spike to bin and smooth. (3) Computing kinematic features (`compute_feature_speed`), which loops over 7 tongue features and 2 paw features, each involving per-trial interpolation across all trials. (4) Loading and resampling motion energy per trial.

ii.
```python
# mat73 loading (I/O bound)
obj = mat73.loadmat(spec.data_path)["obj"]

# Per-unit spike binning and smoothing (CPU bound)
for unit_pos, unit_idx in enumerate(keep_units):
    # bin all spikes for this unit, then smooth
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect")

# Per-feature, per-trial interpolation
for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
    xvel, yvel = find_velocity(xpos, ypos, feat_name)
```

iii. No explicit justification given, but the session-by-session loop with per-unit and per-feature inner loops dominate runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several inner loops could be vectorized: (1) The per-unit spike binning loop in `build_aligned_trialdat` processes one unit at a time; all units could be binned together using vectorized operations. (2) The per-trial interpolation loops in `find_position` and `load_motion_energy` iterate over trials individually. (3) The per-column convolution in `my_smooth` loops over columns. (4) The per-trial velocity computation in `find_velocity`.

ii.
```python
# Per-unit loop (could vectorize across units)
for unit_pos, unit_idx in enumerate(keep_units):
    spike_trials = as_array(probe["trial"][unit_idx], ...)
    # ...

# Per-column smoothing loop
for col in range(arr_filt.shape[1]):
    out[:, col] = np.convolve(arr_filt[:, col], kernel, mode="same")

# Per-trial interpolation
for trix in range(ntrials):
    xpos[:, trix] = interp_with_nan(old_t, ts[:, 0], taxis)
```

iii. No justification given; the AI chose clarity over optimization.

## 11-c. What processing does the code repeat multiple times?

i. (1) The video offset (`find_video_offset`) is computed once in `load_motion_energy` and then the `vidshift` result is reused for kinematic features, so this is not repeated. (2) Quality mask is computed twice -- once in `build_aligned_trialdat` (implicitly via `get_quality_mask`) and once outside for logging. (3) The `build_keep_trial_mask` conditions overlap significantly with `build_low_fr_conditions`, both computing the same boolean arrays (`hit`, `miss`, `early`, `no`, `stim`). (4) `as_array(bp["autowater"], bool)` is computed in both `build_keep_trial_mask` (via `stim`) and `build_output_constants`.

ii.
```python
# quality_mask computed in build_aligned_trialdat:
quality_mask = get_quality_mask(probe)

# And also outside:
quality_mask = get_quality_mask(probe)

# bp fields parsed multiple times:
hit = as_array(bp["hit"], bool).ravel()  # in build_keep_trial_mask
hit = as_array(bp["hit"], np.int64).ravel()  # in build_output_constants
```

iii. No justification -- minor redundancy that doesn't affect correctness.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `build_low_fr_conditions` function computes 7 condition masks matching the Figure 8 PSTH analysis, including conditions for autowater and non-autowater trials. This is used only for computing mean firing rates for the low-FR filter; the PSTHs themselves are discarded. (2) The `mean_frs` array is computed for all quality-passing units but only the `low_fr_keep` subset is retained. (3) The `motion_energy_path` loads `me.moveThresh` but this threshold is not used (the AI uses its own 50th percentile threshold instead). (4) The `build_sample_dataset` function creates a sample subset that is separate from the main output. (5) Position data (xpos, ypos) is computed for all features but only the derived speed is used in the final output.

ii.
```python
# moveThresh loaded but not used for binarization
"moveThresh": float(np.asarray(me.moveThresh).reshape(-1)[0]),

# PSTHs computed only for FR filtering, then discarded
psth = np.zeros((trialdat.shape[0], trialdat.shape[1], len(conds)), dtype=np.float32)
# ... used only to compute mean_frs
```

iii. The `moveThresh` is loaded to match the MATLAB loader but the instructions specify using 50th percentile thresholding instead.
