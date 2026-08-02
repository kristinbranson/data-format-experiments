# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI identifies 12 sessions from the Figure 8 analysis script (`Figure8a_thru_c.m`) and its animal-specific loader files. Each session's data is loaded from a `.mat` file (`data_structure_<animal>_<date>.mat`) using `mat73.loadmat`. Motion energy is loaded from companion `motionEnergy_<animal>_<date>.mat` files using `scipy.io.loadmat`. The sessions are hardcoded in the `SESSION_SPECS` list.

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

for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
```

iii. The AI states: "The session roster follows the paper's Figure 8 context-analysis loader in Figure8a_thru_c.m." This matches the MATLAB code which loads sessions via `loadJEB6_ALMVideo`, `loadJEB7_ALMVideo`, etc.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the `animal` field in each `SessionSpec`. A subject-to-index mapping is built dynamically as sessions are processed. The AI found 7 unique subjects (JEB6, JEB7, EKH1, EKH3, JGR2, JGR3, JEB19).

ii.
```python
subject_idx = subject_to_idx.get(spec.animal)
if subject_idx is None:
    subject_idx = len(subjects)
    subject_to_idx[spec.animal] = subject_idx
    subjects.append(spec.animal)
```

iii. The AI notes the paper reports 6 mice but the code loader scripts list 7 distinct animal IDs. The AI follows the code.

## 1-c. How are the data split into sessions?

i. Each entry in `SESSION_SPECS` defines one session (animal + date + probe number). The 12 sessions are processed sequentially in the main loop. Each session corresponds to one element in `full_data["neural"]`, `full_data["input"]`, and `full_data["output"]`.

ii.
```python
for spec in SESSION_SPECS:
    obj = mat73.loadmat(spec.data_path)["obj"]
    bp = obj["bp"]
    ...
    probe = obj["clu"][spec.probe - 1]
    ...
    full_data["neural"].append(neural_trials)
```

iii. The AI matched the session structure from the Figure 8 loader scripts, where each loader function adds sessions with specific probe numbers to the `meta` array.

## 1-d. How are the data split into trials?

i. The total number of trials per session comes from `bp["Ntrials"]`. All trials are initially processed for neural data (spike binning, smoothing). Then a trial quality mask is applied to select trials for export.

ii.
```python
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
ntrials = int(bp["Ntrials"])
...
trialdat = build_aligned_trialdat(probe, go_cue, ntrials)
...
keep_trials = build_keep_trial_mask(bp)
kept_trial_indices = np.flatnonzero(keep_trials)
```

iii. The AI processes all trials for neural data and kinematics, then filters to kept trials for the final output. This matches the MATLAB pipeline where `getSeq.m` computes trialdat for all `Ntrials` and condition-based analysis is applied afterward.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered to keep only hit and miss trials, excluding early-lick, no-response, and stim-enabled trials: `(hit | miss) & ~early & ~no & ~stim`.

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

iii. The AI states: "Trial export filtering follows the paper text: exclude early-lick trials, exclude no-response trials, exclude stim.enable trials, keep hit and miss trials from both DR and WC contexts." This is consistent with the paper methods which says "early lick trials were omitted from analyses" and the Figure 8 conditions which exclude stim and early trials for analysis.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from spike times (`probe["trialtm"]`) and trial assignments (`probe["trial"]`) from the probe's cluster data (`obj["clu"][probe_index]`). The alignment event is `bp["ev"]["goCue"]`.

ii.
```python
probe = obj["clu"][spec.probe - 1]
...
spike_trials = as_array(probe["trial"][unit_idx], np.int64).ravel() - 1
spike_times = as_array(probe["trialtm"][unit_idx], np.float64).ravel()
...
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
```

iii. This matches the MATLAB `alignSpikes.m` which aligns `trialtm` to the event, and `getSeq.m` which bins these aligned spike times.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 10ms bins (dt=0.01s) over a window from -3.0s to +2.5s (550 bins), converted to firing rates (spikes/bin / dt), and smoothed with a causal half-Gaussian kernel (window=15, reflect boundary padding).

ii.
```python
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
counts = np.zeros((ntrials, NT), dtype=np.float64)
np.add.at(counts, (spike_trials, bins), 1.0)
rates = counts.T / DT
trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect").astype(np.float32)
```

iii. The AI's `my_smooth` function replicates the MATLAB `mySmooth.m`: a Gaussian window of size N, with the first half zeroed (causal), and 'reflect' boundary padding. Parameters match Figure 8: tmin=-3, tmax=2.5, dt=0.01, smooth=15, bctype='reflect'.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Quality-based: exclude units labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Firing rate-based: exclude units with mean firing rate <= 1 Hz, computed by averaging PSTHs across 7 trial-type conditions and time.

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

iii. Quality filtering matches `findClusters.m` with `quality='all'`, which excludes 'garbage', 'gabrga', 'noisy', 'real?'. The low FR threshold of 1 Hz matches Figure 8's `params.lowFR = 1`. The low FR filter conditions match Figure 8's `params.condition` list.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue (`bp.ev.goCue`). Each spike time is adjusted by subtracting the go cue time for its trial, then binned relative to that alignment.

ii.
```python
ALIGN_EVENT = "goCue"
go_cue = as_array(bp["ev"][ALIGN_EVENT], np.float64).ravel()
aligned = spike_times - go_cue[spike_trials]
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. Matches the instructions ("Temporally align based on Go cue onset") and the MATLAB code which uses `params.alignEvent = 'goCue'`.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 10ms (dt=0.01s), giving 550 time bins from -3.0s to +2.5s. No temporal rebinning is applied beyond the initial binning.

ii.
```python
DT = 1 / 100  # 0.01s = 10ms
TMIN = -3.0
TMAX = 2.5
edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
time = edges[:-1] + DT / 2
```

iii. Matches Figure 8's `params.dt = 1/100` and `params.tmin = -3; params.tmax = 2.5`.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the computed time axis (bin centers), which is constructed from the alignment parameters (TMIN, TMAX, DT). It is not directly derived from a raw data variable but from the binning parameters.

ii.
```python
EDGES, TIME = build_time_axis()
...
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. TIME represents the center of each 10ms bin, ranging from -2.995s to +2.495s. This provides continuous, time-varying input as required by the instructions.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No complex processing. The time axis is computed as bin centers: `edges[:-1] + DT/2`, where edges span from TMIN to TMAX in steps of DT.

ii.
```python
def build_time_axis() -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
    time = edges[:-1] + DT / 2
    return edges, time
```

iii. This matches the MATLAB approach in `getSeq.m`: `obj.time = edges + params.dt/2; obj.time = obj.time(1:end-1);`.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural time axis since both use the same bin centers from `build_time_axis()`. They are inherently aligned.

ii.
```python
# Neural: binned into same time axis
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
# Input: same TIME array
input_trials.append(TIME[np.newaxis, :].astype(np.float32, copy=False))
```

iii. Since the go cue is at time 0 in the time axis, and neural data is aligned to the go cue, the input correctly represents time from go cue onset.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp["R"]`, which indicates right-lick trials.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
```

iii. `bp.R` is a standard field in the data objects indicating right-choice trials (1=right, 0=left), consistent with the paper's behavioral paradigm.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. No processing beyond indexing. `bp.R` values are directly used (0=left, 1=right). The per-trial value is broadcast across all time bins.

ii.
```python
lick_direction = as_array(bp["R"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), lick_direction[out_pos], dtype=np.int16),
```

iii. Matches instructions: "left = 0, right = 1, per-trial".

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp["autowater"]`. Autowater=True indicates WC (water-cued) trials.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
```

iii. The `autowater` field in the data structure indicates water-cued (WC) trials. The AI inverts it so that WC=0 and DR=1.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The autowater boolean is inverted: `~autowater` gives DR=1, WC=0 (since autowater=True for WC). The per-trial value is broadcast across all time bins.

ii.
```python
context = (~as_array(bp["autowater"], bool).ravel()[keep_trials]).astype(np.int64)
...
np.full((1, NT), context[out_pos], dtype=np.int16),
```

iii. Matches instructions: "WC = 0, DR = 1, per-trial".

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `bp["hit"]`, which indicates correct (hit) trials.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
```

iii. `bp.hit` indicates trials where the animal responded correctly.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. No processing beyond indexing. `bp.hit` values are directly used (0=incorrect/miss, 1=correct/hit). The per-trial value is broadcast across all time bins.

ii.
```python
outcome = as_array(bp["hit"], np.int64).ravel()[keep_trials]
...
np.full((1, NT), outcome[out_pos], dtype=np.int16),
```

iii. Matches instructions: "incorrect = 0, correct = 1, per-trial".

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC-tracked tongue features from both cameras: side camera ('tongue', 'left_tongue', 'right_tongue') and bottom camera ('top_tongue', 'topleft_tongue', 'bottom_tongue', 'bottomleft_tongue'). Each feature's x,y positions come from `obj.traj[view].ts[:, :2, feat_idx]` and frame times from `obj.traj[view].frameTimes`.

ii.
```python
TONGUE_FEATURES = [
    (0, "tongue"), (0, "left_tongue"), (0, "right_tongue"),
    (1, "top_tongue"), (1, "topleft_tongue"), (1, "bottom_tongue"), (1, "bottomleft_tongue"),
]
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. The AI uses tongue landmarks from both camera views, consistent with the methods text: "The tongue, jaw and nose were tracked using both cameras."

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) x,y positions are interpolated from video frame times onto the neural time axis using linear interpolation, with video offset correction. (2) Velocity is computed as the gradient (first difference). (3) For tongue features, NaN velocities are set to 0 (tongue not visible). (4) Speed = sqrt(xvel^2 + yvel^2). (5) Speeds are averaged across all tongue features.

ii.
```python
def compute_feature_speed(obj, feature_specs, taxis, align_times, vidshift):
    speeds = []
    for view_index, feat_name in feature_specs:
        xpos, ypos = find_position(obj, view_index, feat_name, taxis, align_times, vidshift)
        xvel, yvel = find_velocity(xpos, ypos, feat_name)
        speeds.append(np.sqrt(xvel**2 + yvel**2))
    stacked = np.stack(speeds, axis=0)
    out = np.nanmean(stacked, axis=0)
    return out
```

iii. The position interpolation and velocity computation match the MATLAB `findPosition.m` and `findVelocity.m` functions. The AI intentionally replicates the MATLAB bug where `basederiv[0]` (x-baseline) is subtracted from both x and y velocities for non-tongue features; however, for tongue features this subtraction is not applied, so the bug doesn't affect tongue velocity.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The 50th percentile threshold is computed from strictly positive tongue speed values (ignoring zeros) within the kept trials. Values >= threshold get category 1, below get category 0.

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
    return binary, threshold
```

iii. The AI justifies: "because the shared MATLAB code uses 0 as a placeholder when the tongue is not visible, I estimated the 50th-percentile threshold from strictly positive tongue-speed samples." This deviates from a literal reading of the instructions ("50th percentile") but is motivated by data characteristics.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is computed on the same time axis as neural data (`taxis = TIME + ADVANCE_MOVEMENT`), with `ADVANCE_MOVEMENT = 0.0`. Video frame times are adjusted by the video offset before interpolation onto this time axis.

ii.
```python
taxis = TIME + ADVANCE_MOVEMENT  # ADVANCE_MOVEMENT = 0.0
tongue_speed = compute_feature_speed(obj, TONGUE_FEATURES, taxis, align_times, me["vidshift"])
```

iii. Matches the MATLAB code: `taxis = obj.time + params.advance_movement` with `advance_movement = 0`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC-tracked paw features from the bottom camera only: 'top_paw' and 'bottom_paw'. Position data comes from `obj.traj[1].ts[:, :2, feat_idx]`.

ii.
```python
PAW_FEATURES = [(1, "top_paw"), (1, "bottom_paw")]
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. Matches the methods text: "the paws were tracked using only the bottom view." Features match the default params `traj_features` for the bottom camera.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity, but with paw-specific handling: (1) Positions are interpolated onto neural time axis. (2) Missing values are filled with nearest available value (unlike tongue, which keeps NaNs). (3) Velocity = gradient of position, with baseline subtraction (median derivative subtracted from both x and y velocity). (4) Speed = sqrt(xvel^2 + yvel^2). (5) Speeds averaged across top_paw and bottom_paw.

ii.
```python
# In find_velocity, for non-tongue features:
xvel[:, trix] = xvel[:, trix] - basederiv[0]
yvel[:, trix] = yvel[:, trix] - basederiv[0]  # replicates MATLAB behavior
xvel[:, trix] = fill_nearest_1d(xvel[:, trix], fill_value=0.0)
yvel[:, trix] = fill_nearest_1d(yvel[:, trix], fill_value=0.0)
```

iii. Matches `findVelocity.m` behavior, including the intentional replication of the MATLAB code's subtraction of basederiv(1) from both x and y velocity components.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The 50th percentile threshold is computed from all paw speed values (including zeros) within the kept trials. Values >= threshold get category 1, below get category 0.

ii.
```python
paw_binary, paw_threshold = binarize_session_signal(paw_speed, keep_trials)
```

iii. Straightforward application of the instructions' "50th percentile" per-session threshold.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same as tongue velocity - computed on the neural time axis with video offset correction.

ii.
```python
paw_speed = compute_feature_speed(obj, PAW_FEATURES, taxis, align_times, me["vidshift"])
```

iii. Same alignment approach as all kinematic features.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from companion `motionEnergy_<animal>_<date>.mat` files. The raw data comes from `me.data` (a cell array of per-trial motion energy traces computed from video at 400 Hz). The threshold `me.moveThresh` is also loaded but not used for binarization.

ii.
```python
def load_motion_energy(obj, motion_energy_path, taxis, align_times):
    mat = sio.loadmat(motion_energy_path, struct_as_record=False, squeeze_me=True)
    me = mat["me"]
    raw_data = me.data
    ...
    raw_trials = list(np.asarray(raw_data, dtype=object).ravel())
```

iii. Matches the MATLAB `loadMotionEnergy.m` which loads the companion motion energy file.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces are interpolated from video frame times onto the neural time axis using linear interpolation, with video offset correction (same as for kinematic features). Missing values are filled with nearest available value.

ii.
```python
for trix, trial_me in enumerate(raw_trials):
    trial_me = as_array(trial_me, np.float64).ravel()
    frame_times = get_trial_frame_times(cam, trix)
    ...
    old_t = frame_times - vidshift - align_times[trix]
    resampled[:, trix] = interp_with_nan(old_t, trial_me, taxis)
    resampled[:, trix] = fill_nearest_1d(resampled[:, trix], fill_value=0.0)
```

iii. Matches `loadMotionEnergy.m`: interpolation using `interp1(frameTimes-vidshift-alignTimes, me.data{trix}, taxis)`, followed by `fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. The 50th percentile threshold is computed from all motion energy values within the kept trials. Values >= threshold get category 1, below get category 0.

ii.
```python
motion_binary, motion_threshold = binarize_session_signal(me["data"], keep_trials)
```

iii. Straightforward application of the instructions' "50th percentile" per-session threshold. Note: the loaded `moveThresh` from the mat file (a manually set threshold) is stored but not used for binarization.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated onto the neural time axis (`taxis = TIME + ADVANCE_MOVEMENT`) with video offset correction, identical to the kinematic features.

ii.
```python
taxis = TIME + ADVANCE_MOVEMENT
me = load_motion_energy(obj, spec.motion_energy_path, taxis, align_times)
```

iii. Matches the MATLAB code: `taxis = obj.time + params.advance_movement`.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used:
- **Dropped frames**: Trials with NaN `NdroppedFrames` are skipped for kinematic extraction.
- **Missing frame times**: A fallback generates frame times from frame count and assumed 400 Hz frame rate.
- **Missing positions (non-tongue)**: Filled with nearest available value via `fill_nearest_1d`.
- **Missing positions (tongue)**: Left as NaN (tongue not visible).
- **NaN velocities (non-tongue)**: Filled with nearest available value.
- **NaN velocities (tongue)**: Set to 0.
- **Non-finite spike times/trials**: Filtered out before binning.
- **Duplicate time points in interpolation**: Deduplicated before interpolation.

ii.
```python
# Dropped frames check
dropped = np.asarray(cam["NdroppedFrames"][trix]).reshape(-1)
if dropped.size and np.isnan(dropped[0]):
    continue

# Fill missing for non-tongue
if not is_tongue:
    xpos[:, trix] = fill_nearest_1d(xpos[:, trix], fill_value=0.0)

# Tongue velocity NaN handling
xvel[:, trix] = np.nan_to_num(xvel[:, trix], nan=0.0)

# Finite spike filtering
finite = (spike_trials >= 0) & (spike_trials < ntrials) & np.isfinite(spike_times)
```

iii. These strategies match the MATLAB reference code: `fillmissing('nearest')` for non-tongue features, NaN-to-0 for tongue velocity, and frame-level validation.

## 11-a. What are the most time-consuming steps of the code?

i. The most time-consuming steps are likely:
1. **Loading .mat files** with `mat73.loadmat` (large data structures).
2. **Spike binning and smoothing** in `build_aligned_trialdat` — iterates over all units, bins spikes per trial, convolves each column.
3. **Kinematic feature extraction** in `compute_feature_speed` — calls `find_position` and `find_velocity` for each of 7 tongue features and 2 paw features, each iterating over all trials.
4. **Motion energy loading and interpolation** — interpolates per trial.

ii.
```python
# Spike processing: loop over units, bin all trials, smooth
for unit_pos, unit_idx in enumerate(keep_units):
    ...
    np.add.at(counts, (spike_trials, bins), 1.0)
    rates = counts.T / DT
    trialdat[:, unit_pos, :] = my_smooth(rates, SMOOTH, bctype="reflect")

# Kinematic processing: loop over features, then trials
for view_index, feat_name in feature_specs:
    xpos, ypos = find_position(...)  # loops over ntrials
    xvel, yvel = find_velocity(...)  # loops over ntrials
```

iii. The AI did not explicitly discuss performance in CONVERSION_NOTES.md.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized:
1. **`find_position`**: The trial loop could potentially be vectorized using batch interpolation.
2. **`find_velocity`**: The trial loop computing gradient and baseline subtraction could use vectorized numpy operations.
3. **`my_smooth`**: The column-by-column convolution loop could use scipy.signal operations.
4. **`build_aligned_trialdat`**: The outer loop over units is necessary for `np.add.at`, but the inner smoothing loop (via `my_smooth`) applies convolution column-by-column.

ii.
```python
# Example: find_velocity loops over trials
for trix in range(xpos.shape[1]):
    tsinterp = np.column_stack([xpos[:, trix], ypos[:, trix]])
    basederiv = np.nanmedian(np.diff(tsinterp, axis=0), axis=0)
    xvel[:, trix] = np.gradient(tsinterp[:, 0])
    yvel[:, trix] = np.gradient(tsinterp[:, 1])
```

iii. No explicit discussion of vectorization in the AI's notes.

## 11-c. What processing does the code repeat multiple times?

i.
1. **Video offset computation**: `find_video_offset` is called in `load_motion_energy`, and the returned `vidshift` is reused for `compute_feature_speed`. However, the motion energy function computes it internally and also returns it. So it's computed once per session effectively.
2. **Quality mask computation**: `get_quality_mask` is called both in `build_aligned_trialdat` and separately to log stats, computing the same mask twice.
3. **Trial mask computation**: `build_keep_trial_mask` is called once but `build_low_fr_conditions` computes overlapping boolean arrays from the same `bp` fields.

ii.
```python
# Quality mask computed in build_aligned_trialdat
quality_mask = get_quality_mask(probe)  # inside build_aligned_trialdat
...
# And again outside
quality_mask = get_quality_mask(probe)  # in convert_dataset loop
```

iii. Not discussed in the AI's notes.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
1. **All-trial neural processing**: `build_aligned_trialdat` processes all `ntrials` trials, but only kept trials (after quality filtering) are exported. The excluded trials' neural data is computed but discarded.
2. **All-trial kinematic processing**: `find_position`, `find_velocity`, and `load_motion_energy` process all trials, but only kept trials are used in the final output.
3. **PSTH conditions for low FR filter**: Seven condition-based PSTHs are computed just for the mean firing rate calculation, and the PSTHs themselves are discarded.
4. **Motion energy `moveThresh`**: Loaded and stored but not used for binarization (the 50th percentile is used instead).

ii.
```python
# All trials processed but only kept_trial_indices exported
trialdat = build_aligned_trialdat(probe, go_cue, ntrials)  # all ntrials
...
for out_pos, trial_idx in enumerate(kept_trial_indices):  # only kept trials exported
    neural_trials.append(trialdat[:, :, trial_idx].T)
```

iii. The AI acknowledges in CONVERSION_NOTES that the processing pipeline follows the MATLAB code structure, which processes all trials first and filters later.
