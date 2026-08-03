# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the reference MATLAB loader files (e.g., `loadJEB6_ALMVideo.m`, `loadJEB11_ALMVideo.m`, etc.) to extract session specifications (subject, date, probe numbers, data folder). It then loads each session's `.mat` file using `pymatreader.read_mat()`. Motion energy is loaded from separate `motionEnergy_*.mat` files when available, or from embedded `obj.me` as a fallback. Two groups of loaders are processed: 10 fixed-delay loaders and 4 randomized-delay loaders, yielding 44 total sessions.

ii.
```python
FIXED_DELAY_LOADERS = [
    "loadJEB6_ALMVideo.m", "loadJEB7_ALMVideo.m", "loadEKH1_ALMVideo.m",
    "loadEKH3_ALMVideo.m", "loadJGR2_ALMVideo.m", "loadJGR3_ALMVideo.m",
    "loadJEB13_ALMVideo.m", "loadJEB14_ALMVideo.m", "loadJEB15_ALMVideo.m",
    "loadJEB19_ALMVideo.m",
]
RANDOMIZED_DELAY_LOADERS = [
    "loadJEB11_ALMVideo.m", "loadJEB12_ALMVideo.m",
    "loadJEB23_ALMVideo.m", "loadJEB24_ALMVideo.m",
]

def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions

# Loading per session:
obj = read_mat(spec.data_path)["obj"]
```

iii. The AI justified this approach by stating it follows the same session inclusion as the reference code's figure scripts (e.g., `Figure3d.m`, `Figure3i.m`), which define the exact set of analyzed sessions. This ensures consistency with the paper's session selection rather than naively including all raw files in the data directory. Behavior-only sessions are excluded since they lack neural data.

## 1-b. How are the data split into subjects?

i. Subjects are extracted from the loader file names using a regex pattern (e.g., `loadJEB6_ALMVideo.m` -> subject `JEB6`). A stable-unique list of subjects is maintained across sessions, and each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)

# In build_dataset:
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. The AI documented 14 distinct subjects across the loader files, noting a discrepancy with the paper which reports 9 mice for fixed-delay and 4 for randomized-delay. The AI chose to follow the code/data (14 subjects) rather than the paper text, treating the difference as a text/reporting discrepancy.

## 1-c. How are the data split into sessions?

i. Each entry in a loader file (identified by a date + probe specification followed by a `datapth = fullfile` line) becomes one `SessionSpec`. Sessions are uniquely identified by `{subject}_{date}`. The loader parser extracts all sessions from each loader file.

ii.
```python
@dataclass(frozen=True)
class SessionSpec:
    subject: str
    date: str
    probe: Tuple[int, ...]
    folder: str
    task: str

    @property
    def session_id(self) -> str:
        return f"{self.subject}_{self.date}"

    @property
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.subject}_{self.date}.mat"
```

iii. The AI noted that this produces 25 fixed-delay sessions and 19 randomized-delay sessions (44 total), matching the paper's session counts. Multi-probe sessions (e.g., `JEB15` with `probe = [1 2]`) are handled as single sessions with neurons concatenated from both probes.

## 1-d. How are the data split into trials?

i. Trials are indexed from the raw data structure's `obj.bp.Ntrials`. A trial mask is built based on quality control filters, and only passing trials (`keep_trials = np.flatnonzero(trial_mask)`) are used. Neural data, inputs, and outputs are all indexed by these kept trial indices.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid
```

iii. The AI stated this follows the reference code's `findTrials` logic, which excludes stimulation, early-lick, and no-response trials from analyses.

## 1-e. How are trials filtered based on quality controls?

i. Three trial-level filters are applied: (1) exclude `stim.enable` trials, (2) exclude `early` lick trials, (3) exclude `no`-response trials. Additionally, a neural-coverage filter drops trials that occur after the last trial with spikes in the selected neuron set. Sessions with fewer than 2 valid trials are skipped.

ii.
```python
valid = (~stim_enable) & (~early) & (~no)

# Neural coverage filtering:
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
if max_trial_with_spikes > 0:
    neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
    refined_trial_mask = trial_mask & neural_coverage_mask
```

iii. The AI justified the neural-coverage filter by noting that some JEB24 sessions had behavior-valid trials occurring after the last trial with neural spikes, which would produce all-zero neural inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from per-probe spike data: `obj.clu{probe}.trial` (1-based trial index per spike), `obj.clu{probe}.trialtm` (time within trial per spike), and `obj.bp.ev.goCue` (go cue event times for alignment).

ii.
```python
probes = normalize_probe_container(obj.get("clu"))
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)

# Per neuron:
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
```

iii. The AI documented that these are the same variables used by the reference code's `alignSpikes` and `getSeq` functions.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5 ms bins over [-2.5, +2.5] seconds (1000 bins), converted to firing rates (counts/dt), and smoothed with a causal Gaussian kernel (n=15 bins, alpha=2.5, reflect boundary condition). The output is a `(n_neurons, n_timepoints)` matrix per trial.

ii.
```python
DT = 0.005; TMIN = -2.5; TMAX = 2.5; SMOOTH = 15; BCTYPE = "reflect"

def binned_neuron_trials(probe, neuron_index, align_times, keep_trials_0based, edges):
    # ... align spikes to go cue ...
    aligned = trialtm[mask] - align_times[trials[mask] - 1]
    bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
    # ... bin spikes ...
    rates = counts / DT
    return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)

def causal_gaussian_smooth(x, n, bctype=BCTYPE):
    # Gaussian window with causal half zeroed out
    kern = gaussian_window(n)
    kern[: n // 2] = 0.0
    kern /= kern.sum()
    # Reflect boundary padding, then convolve
```

iii. The AI stated this matches the reference `getSeq` function's binning and `mySmooth` smoothing logic. The 5 ms bin size was chosen to match the paper's decoding methods ("each bin is 5 ms").

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-layer neuron curation: (1) quality filter excludes clusters labeled "garbage", "gabrga", "noisy", or "real?" (matching `findClusters(..., 'all')`), and (2) low firing rate filter excludes neurons with mean firing rate <= 1 Hz. Sessions with fewer than 10 remaining units are skipped entirely.

ii.
```python
LOW_FR = 1.0; MIN_UNITS_PER_SESSION = 10

def quality_keep_mask(qualities):
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)

def neuron_mean_fr(probe, neuron_index, align_times, trial_mask):
    n_spikes = count_window_spikes(trials, trialtm, align_times, trial_mask)
    n_trials = int(trial_mask.sum())
    return n_spikes / (n_trials * (TMAX - TMIN))

# Selection:
if mean_fr > LOW_FR:
    kept_indices.append(int(neuron_index))
```

iii. The AI documented this as matching the reference `findClusters` and `removeLowFRClusters` functions. The firing rate is computed from raw spike counts in the [-2.5, 2.5] window across valid trials, rather than from the smoothed PSTH mean (a minor implementation difference from the reference).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike time is aligned to the go cue by subtracting the go cue event time for that trial: `aligned_time = trialtm - goCue_time`. Spikes are then binned into the common [-2.5, +2.5] second window.

ii.
```python
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. The AI stated this matches the reference `alignSpikes` function with `params.alignEvent = 'goCue'`, consistent with both the paper's main decoding analyses and the decoder task instructions.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 5 ms (DT = 0.005 s), producing 1000 time bins over [-2.5, +2.5] seconds. No rebinning is applied; raw spike times are binned directly into 5 ms bins.

ii.
```python
DT = 0.005
def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. The AI chose 5 ms based on the paper's decoding methods statement ("each bin is 5 ms"), prioritizing this over the 10 ms bins used in some other reference figure scripts.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the common time axis (bin centers), which is computed from the bin edges defined by TMIN, TMAX, and DT. It is not derived from any per-trial raw variable.

ii.
```python
def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0

# Per trial:
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The AI noted that this is the same time axis used for neural data, ensuring alignment. The values range from -2.4975 to +2.4975 seconds (bin centers of the [-2.5, +2.5] edge range).

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time axis is computed as bin centers: `edges[:-1] + DT/2`. No further processing is applied. The same array is used for every trial in every session, broadcast to shape `(1, n_timepoints)`.

ii.
```python
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. The AI described this as a deterministic function of the bin parameters, identical across all trials and sessions.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is identical to the neural data's time axis by construction -- both use the same bin centers derived from the same TMIN, TMAX, DT parameters.

ii.
```python
# Both neural and input use the same edges/centers:
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
# Neural uses time_edges for binning; input uses time_centers directly
```

iii. The AI's justification is implicit: since the input is the time axis itself, alignment is guaranteed.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R` (right lick indicator), where R=1 means right lick and R=0 means left lick.

ii.
```python
R = to_vector(bp["R"], float).astype(int)
lick_direction = R[keep_trials]
```

iii. The AI noted this matches the reference code's trial labeling convention where `R` indicates right choice and maps to left=0, right=1.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The raw `R` field is directly used as the lick direction value (0=left, 1=right). It is broadcast as a per-trial constant across all time bins.

ii.
```python
np.full(time_centers.size, lick_direction[tr], dtype=np.int64)
```

iii. The AI stated this is a direct mapping consistent with the decoder task specification (left=0, right=1).

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, where autowater=1 indicates water-cued (WC) trials and autowater=0 indicates delayed-response (DR) trials.

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
context = 1 - autowater[keep_trials]
```

iii. The AI documented that the polarity is flipped (`1 - autowater`) to match the decoder task specification (WC=0, DR=1), since the raw `autowater` field uses the opposite convention.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The raw `autowater` field is inverted (`1 - autowater`) to produce WC=0, DR=1. It is broadcast as a per-trial constant across all time bins.

ii.
```python
context = 1 - autowater[keep_trials]
# Per trial:
np.full(time_centers.size, context[tr], dtype=np.int64)
```

iii. The AI justified the inversion by noting that the raw code uses `autowater==1` for WC, while the decoder task specifies WC=0, DR=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, where hit=1 indicates a correct trial and hit=0 indicates an incorrect trial (miss).

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
outcome = hit[keep_trials]
```

iii. The AI verified that `hit` and `miss` are mutually exclusive on kept trials (`hit + miss == 1`), which is guaranteed by the trial filter excluding no-response trials.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The raw `hit` field is directly used as the outcome value (0=incorrect/miss, 1=correct/hit). It is broadcast as a per-trial constant across all time bins. A validation check ensures `hit + miss == 1` for all kept trials.

ii.
```python
outcome = hit[keep_trials]
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(f"{spec.session_id}: hit/miss are not mutually exclusive on kept trials")

# Per trial:
np.full(time_centers.size, outcome[tr], dtype=np.int64)
```

iii. The AI stated this directly maps to the decoder task specification (incorrect=0, correct=1).

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC trajectory data in `obj.traj` for tongue-related tracked features. View 1 (side camera): "tongue", "left_tongue", "right_tongue". View 2 (bottom camera): "top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue". The x,y coordinates are in `obj.traj{view}.ts[:, :2, feat_index]` and frame times from `obj.traj{view}.frameTimes`.

ii.
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}

# Position extraction:
xy = ts[:, :2, feat_index]
```

iii. The AI documented these as the tongue-related DLC features used in the reference kinematics processing, matching the tracked points described in the methods.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) Extract x,y positions from DLC trajectories, (2) Correct frame times by video offset and go cue alignment, (3) Interpolate positions to the neural time axis using linear interpolation, (4) Compute x and y velocity via `np.gradient`, (5) Replace NaN velocities with 0 (tongue-specific handling), (6) Compute speed as `sqrt(xvel^2 + yvel^2)`. All tongue feature speeds are then averaged (mean across features, ignoring NaNs).

ii.
```python
# Interpolation:
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
xy_aligned = interp(taxis)

# Velocity (tongue-specific NaN handling):
xv = np.gradient(tsinterp[:, 0])
yv = np.gradient(tsinterp[:, 1])
if "tongue" not in feat_name:
    xv = xv - base[0]; yv = yv - base[1]
else:
    xv = np.nan_to_num(xv, nan=0.0); yv = np.nan_to_num(yv, nan=0.0)

# Speed and aggregation:
speed = np.sqrt(np.square(xvel) + np.square(yvel))
agg = np.divide(summed, np.maximum(count, 1))  # mean across features
```

iii. The AI stated this reproduces the reference `getKinematicsFromVideo` and `findVelocity` processing, with tongue-specific handling (NaN->0 instead of nearest-neighbor filling, no baseline subtraction).

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. The AI uses a rank-based median split: the flattened per-session trace is sorted, and the bottom half is assigned 0, the top half assigned 1. A separate `percentile_threshold` function computes the 50th percentile value but this value is only used for plotting, not for the actual discretization.

ii.
```python
def discretize_trace(traces: np.ndarray, threshold: float) -> np.ndarray:
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)

tongue_thr = percentile_threshold(tongue_speed, 50.0)  # used for plotting only
tongue_bin = discretize_trace(tongue_speed, tongue_thr)
```

iii. The AI documented this as producing a per-session median threshold, yielding exactly 50/50 class balance. However, the `discretize_trace` function ignores the `threshold` parameter and uses rank-based splitting instead.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by interpolating DLC trajectories onto the neural time axis. Frame times are corrected by the video offset (computed from bitcode synchronization) and the per-trial go cue time: `old_time = frameTimes - vidshift - goCue_time`. The interpolation target is `time_centers + ADVANCE_MOVEMENT` (ADVANCE_MOVEMENT=0.0).

ii.
```python
vidshift = compute_video_offset(obj)
taxis = time_centers + ADVANCE_MOVEMENT  # ADVANCE_MOVEMENT = 0.0
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
xy_aligned = interp(taxis)
```

iii. The AI stated this matches the reference `getKinematicsFromVideo` and `findVideoOffset` functions for temporal alignment.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC trajectory data in `obj.traj` view 2 (bottom camera) for paw features: "top_paw" and "bottom_paw".

ii.
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
```

iii. The AI documented that paws are tracked only from the bottom camera view, consistent with the methods description.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Same pipeline as tongue velocity with key differences for non-tongue features: (1) NaN positions are filled with nearest-neighbor interpolation (`fill_nearest_1d`), (2) Velocity baseline is subtracted (median of position differences), (3) NaN velocities are also filled with nearest-neighbor values.

ii.
```python
# Non-tongue NaN filling:
xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
ypos[:, trial] = fill_nearest_1d(ypos[:, trial])

# Velocity baseline subtraction (non-tongue):
base = np.nanmedian(diffs, axis=0)
xv = xv - base[0]
yv = yv - base[1]
xv = fill_nearest_1d(xv)
yv = fill_nearest_1d(yv)
```

iii. The AI stated this matches the reference `findVelocity` function's non-tongue processing path: baseline-subtract and nearest-fill.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same rank-based median split as tongue velocity: `discretize_trace` sorts the per-session trace and assigns bottom half to 0, top half to 1.

ii.
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. Same as tongue velocity thresholding. Produces exactly 50/50 per session.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same alignment as tongue velocity: frame times corrected by video offset and go cue time, linearly interpolated onto the neural time axis.

ii.
```python
# Same aligned_position and aggregate_speed pipeline as tongue
paw_speed_all, paw_feats = aggregate_speed(obj, PAW_FEATURES, time_centers, align_times, vidshift)
```

iii. Same justification as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from external `motionEnergy_*.mat` files (field `me.data`), or from embedded `obj.me` when external files are unavailable. Each trial contains a 1D array of per-frame motion energy values. Frame times come from `obj.traj[0].frameTimes`.

ii.
```python
def load_motion_energy_raw(obj, spec):
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        return {"data": unwrap_embedded_motion_energy(raw_me), "moveThresh": ...}
    if "me" in obj:
        raw_me = obj["me"]
        return {"data": unwrap_embedded_motion_energy(raw_me), ...}
    return None
```

iii. The AI documented that external motion energy files exist for most ephys sessions, matching the reference `loadMotionEnergy` function's loading path.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces are interpolated to the neural time axis using linear interpolation. Frame times are corrected by video offset and go cue time. NaN values after interpolation are filled with nearest-neighbor values, then remaining NaNs are replaced with zeros.

ii.
```python
def aligned_motion_energy(obj, raw_me, time_centers, align_times, vidshift):
    for trial in range(min(len(me_data), n_trials)):
        me_trial = np.asarray(me_data[trial], dtype=np.float32).reshape(-1)
        frame_times = get_frame_times(trial_view, ts.shape[0])
        old_time = frame_times - vidshift - float(align_times[trial])
        interp = interp1d(old_time, me_trial, kind="linear", bounds_error=False, fill_value=np.nan)
        aligned[:, trial] = interp(taxis)
        aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
    return np.nan_to_num(aligned, nan=0.0)
```

iii. The AI stated this matches the reference `loadMotionEnergy` function's interpolation and NaN-filling logic.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same rank-based median split as tongue and paw velocity.

ii.
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. Same as other movement variable thresholding.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned by interpolating onto the neural time axis using frame times corrected by video offset and go cue time. When frame times don't match motion energy length, a default 400 Hz frame rate with 0.5s offset is used as fallback.

ii.
```python
if frame_times.size != me_trial.size:
    frame_times = (np.arange(me_trial.size, dtype=np.float32) + 1.0) / 400.0
    old_time = frame_times - 0.5 - float(align_times[trial])
else:
    old_time = frame_times - vidshift - float(align_times[trial])
```

iii. The AI documented the fallback for mismatched frame time/motion energy lengths, using the default 400 Hz and 0.5s offset from the reference code.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Sessions without neural data (no `clu` field) are excluded. (2) Behavior-valid trials beyond neural coverage are dropped. (3) NaN positions are filled with nearest-neighbor for non-tongue features, zeros for tongue velocities. (4) Missing motion energy files use embedded `obj.me` or produce zeros. (5) Nested motion energy structs (e.g., JEB15) are recursively unwrapped. (6) Dropped video frames (all-NaN `NdroppedFrames`) cause the trial's kinematic data to be skipped. (7) Missing probe location metadata defaults to "ALM".

ii.
```python
# Neural coverage:
neural_coverage_mask = (np.arange(trial_mask.size) + 1) <= max_trial_with_spikes

# Dropped frames:
dropped = trial_view.get("NdroppedFrames")
if dropped is not None:
    if np.isnan(np.asarray(dropped_arr, dtype=float)).all():
        continue

# Missing probe location:
return loc if loc else "ALM"
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md Step 10, noting specific fixes for JEB15 nested motion energy and JEB24 neural coverage gaps.

## 11-a. What are the most time-consuming steps of the code?

i. Based on the conversion output (~480s for 44 sessions, ~10-12s per session), the most time-consuming steps are: (1) Loading large MATLAB .mat files via `pymatreader.read_mat()`, (2) Per-neuron spike binning and smoothing (iterates over all selected neurons), (3) Kinematic interpolation (iterates over all features and trials for position and velocity computation).

ii.
```python
# Each session takes 3-20 seconds depending on size
# Session loading:
obj = read_mat(spec.data_path)["obj"]
# Per-neuron binning:
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(...)
# Per-feature kinematic processing:
for view_index, feat_names in feature_map.items():
    for feat_name in feat_names:
        xpos, ypos = aligned_position(...)
```

iii. The AI estimated ~8 minutes for full conversion, which matched the actual runtime (~480s).

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The per-neuron spike binning loop iterates over each neuron individually; could be vectorized by processing all neurons' spikes at once with sparse matrix operations. (2) The per-trial kinematic interpolation loop processes each trial separately; could use vectorized batch interpolation. (3) The per-column smoothing loop in `causal_gaussian_smooth` convolves each column independently; could use 2D FFT-based convolution. (4) The per-trial feature velocity loop processes each trial separately.

ii.
```python
# Per-neuron loop:
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(...)

# Per-trial kinematic loop:
for trial in range(n_trials):
    interp = interp1d(old_time, xy, ...)

# Per-column smoothing:
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")
```

iii. The AI mentioned optimizations were applied (two-pass neural processing, restricting video processing to needed features) but these core loops remain sequential.

## 11-c. What processing does the code repeat multiple times?

i. (1) When neural coverage filtering triggers, neuron selection is re-run (`select_neurons` called twice, along with `max_supported_trial`). (2) The `percentile_threshold` function computes the 50th percentile for each movement variable, but the result is not actually used by `discretize_trace` -- it's only used for plotting. (3) Video offset is computed once per session (not repeated). (4) The trial mask and keep_trials are recomputed after neural coverage refinement.

ii.
```python
# First selection pass:
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
# Possible second pass after refinement:
if not np.array_equal(refined_trial_mask, trial_mask):
    selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
    max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
```

iii. The AI noted the double neuron selection only occurs for sessions with neural coverage gaps (rare: only 2 of 44 sessions).

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `percentile_threshold` function computes thresholds that are used only for plotting, not for actual discretization. (2) The code computes continuous tongue speed, paw speed, and motion energy traces, but only the discretized binary versions are stored in the output. The continuous traces are discarded after discretization. (3) The `baselineFR` computation described in the reference code is not implemented (correctly omitted since z-scoring is not needed for the decoder task). (4) Session metadata (`probe_summaries`, feature lists) is stored but not used by the decoder.

ii.
```python
# Threshold computed but not used by discretize_trace:
tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)  # threshold param ignored

# Continuous traces computed then discarded:
tongue_speed = tongue_speed_all[:, keep_trials]
# ... only tongue_bin is stored in output
```

iii. The AI did not explicitly discuss this inefficiency, but the continuous traces are necessary intermediate steps for computing the discretized outputs.
