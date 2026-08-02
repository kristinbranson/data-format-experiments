# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI parses the MATLAB loader files in `code/DataLoadingScripts/Recording and video/` to extract session specifications (subject, date, probe numbers). It then loads each session's `.mat` file via `pymatreader.read_mat()`. Fixed-delay sessions come from `data/Ephys_Behavior/` and randomized-delay sessions from `data/RandomizedDelay_Ephys_Behavior/`. The loader files parsed are explicitly listed in `FIXED_DELAY_LOADERS` and `RANDOMIZED_DELAY_LOADERS`.

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

def parse_loader_file(loader_name: str, folder: str, task: str) -> List[SessionSpec]:
    path = LOADER_ROOT / loader_name
    # ... parses date, probe from MATLAB loader files

def get_reference_sessions() -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for loader in FIXED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="Ephys_Behavior", task="fixed_delay"))
    for loader in RANDOMIZED_DELAY_LOADERS:
        sessions.extend(parse_loader_file(loader, folder="RandomizedDelay_Ephys_Behavior", task="randomized_delay"))
    return sessions

# In process_session:
obj = read_mat(spec.data_path)["obj"]
```

iii. The AI justified this approach by following the reference code's session selection logic. The CONVERSION_NOTES.md states: "Include only neural sessions represented in the reference ephys loaders. Fixed-delay set: all 25 sessions in Ephys_Behavior. Randomized-delay set: the 19-session subset encoded by the loader files." This matches the reference code's `loadSessionData.m` which uses these same loader files.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by parsing the animal ID from the loader filename (e.g., `loadJEB6_ALMVideo.m` -> `JEB6`). A stable-order list of unique subjects is maintained, and each session is assigned a `subject_idx` pointing into this list.

ii.
```python
subject_match = re.match(r"load([A-Z0-9]+)_ALMVideo\.m", loader_name)
subject = subject_match.group(1)

# In build_dataset:
if spec.subject not in subject_order:
    subject_order.append(spec.subject)
subject_idx.append(subject_order.index(spec.subject))
```

iii. The AI followed the reference convention where each loader file corresponds to one subject, and subject identity is encoded in the filename. This matches how the reference code associates sessions with animals.

## 1-c. How are the data split into sessions?

i. Each session is uniquely identified by a `(subject, date)` pair, derived from the loader files. Each session corresponds to one `.mat` file (`data_structure_<subject>_<date>.mat`). Sessions can have one or more probes; the probe numbers from the loader files are included in the `SessionSpec`. Each session is processed independently in `process_session()`.

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
    def data_path(self) -> Path:
        return DATA_ROOT / self.folder / f"data_structure_{self.subject}_{self.date}.mat"
```

iii. The session splitting directly mirrors the reference code's loader structure, where each `meta(end).date` entry defines one session. Multi-probe sessions (e.g., `JEB15` with `probe = [1 2]`) concatenate neurons across probes within the same session.

## 1-d. How are the data split into trials?

i. Within each session, trials are identified from `obj.bp.Ntrials` (total trial count). A boolean mask filters trials based on quality controls. The kept trial indices (`keep_trials`) define which raw trials become output trials.

ii.
```python
def build_trial_mask(obj: dict) -> np.ndarray:
    bp = obj["bp"]
    stim_enable = to_vector(bp["stim"]["enable"], float).astype(bool)
    early = to_vector(bp["early"], float).astype(bool)
    no = to_vector(bp["no"], float).astype(bool)
    valid = (~stim_enable) & (~early) & (~no)
    return valid

keep_trials = np.flatnonzero(trial_mask)
```

iii. The AI's approach excludes stimulation, early-lick, and no-response (ignore) trials, retaining both hit and miss trials. This deviates from the reference code's default condition definitions which only keep hit trials, but the decoder task requires outcome decoding (correct vs incorrect), necessitating the inclusion of miss trials.

## 1-e. How are trials filtered based on quality controls?

i. Trials are excluded if any of these conditions hold: (1) photostimulation was enabled (`stim.enable`), (2) the mouse licked early (`early`), (3) the mouse did not respond (`no`). Additionally, trials beyond the last trial with neural spike coverage are excluded to avoid all-zero neural data.

ii.
```python
valid = (~stim_enable) & (~early) & (~no)

# Neural coverage filtering:
max_trial_with_spikes = max_supported_trial(selected_neurons, probes)
if max_trial_with_spikes > 0:
    neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes
    refined_trial_mask = trial_mask & neural_coverage_mask
```

iii. The stim/early/no exclusion matches the reference code's trial filtering logic used throughout the figure scripts. The neural-coverage filter was added to handle edge cases in JEB24 sessions where some trials lacked neural data.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `obj.clu{probe}.trial` (1-based trial assignments per spike), `obj.clu{probe}.trialtm` (spike times within each trial), and `obj.bp.ev.goCue` (go cue event times for alignment).

ii.
```python
probes = normalize_probe_container(obj.get("clu"))
align_times = to_vector(obj["bp"]["ev"]["goCue"], float)

# In binned_neuron_trials:
trials = to_vector(probe["trial"][neuron_index], int)
trialtm = to_vector(probe["trialtm"][neuron_index], float)
```

iii. This matches the reference code's `alignSpikes` and `getSeq` functions which use the same raw variables to construct binned firing rates.

## 2-b. How is the `neural` data processed?

i. Spike times are aligned to the go cue, binned into 5 ms bins from -2.5s to +2.5s (1000 time bins), converted to firing rates (Hz) by dividing by the bin width, and smoothed with a causal Gaussian kernel (window size 15, 'reflect' boundary condition).

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5
SMOOTH = 15
BCTYPE = "reflect"

def binned_neuron_trials(probe, neuron_index, align_times, keep_trials_0based, edges):
    # ... align spikes, bin into edges
    aligned = trialtm[mask] - align_times[trials[mask] - 1]
    bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
    counts = np.zeros((edges.size - 1, keep_trials_0based.size), dtype=np.float32)
    np.add.at(counts, (bin_index[valid], keep_index[valid]), 1.0)
    rates = counts / DT
    return causal_gaussian_smooth(rates, SMOOTH, BCTYPE)
```

iii. The processing replicates the reference `getSeq.m` pipeline: `histc` for binning, division by `dt` for firing rates, and `mySmooth` for causal Gaussian smoothing. The 5 ms bin size matches the paper's decoding specification. The 'reflect' boundary condition matches all figure scripts (though the default in `processData.m` is 'none').

## 2-c. How is the `neural` data filtered based on quality controls?

i. Two-stage filtering: (1) Cluster quality filter excludes neurons labeled 'garbage', 'gabrga', 'noisy', or 'real?'. (2) Neurons with mean firing rate <= 1 Hz are removed. Sessions with fewer than 10 remaining neurons are excluded entirely.

ii.
```python
LOW_FR = 1.0
MIN_UNITS_PER_SESSION = 10

def quality_keep_mask(qualities):
    bad = {"garbage", "gabrga", "noisy", "real?"}
    return np.array([q not in bad for q in cleaned], dtype=bool)

def neuron_mean_fr(probe, neuron_index, align_times, trial_mask):
    n_spikes = count_window_spikes(trials, trialtm, align_times, trial_mask)
    n_trials = int(trial_mask.sum())
    return n_spikes / (n_trials * (TMAX - TMIN))

# In select_neurons:
if mean_fr > LOW_FR:
    kept_indices.append(int(neuron_index))
```

iii. Quality labels match the reference `findClusters.m` with quality='all'. The 1 Hz firing rate threshold matches all figure scripts (which set `params.lowFR = 1`), though the default in `getDefaultParams.m` is 0.5 Hz. The 10-unit minimum per session matches the paper's statement.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike's within-trial time (`trialtm`) is aligned by subtracting the go cue time for that trial. The aligned spike times are then binned into the common time grid centered on the go cue.

ii.
```python
aligned = trialtm[mask] - align_times[trials[mask] - 1]
bin_index = np.floor((aligned - TMIN) / DT).astype(np.int64)
```

iii. This matches the reference `alignSpikes.m` which computes `trialtm_aligned = trialtm - event` where event is `obj.bp.ev.goCue`. The instructions explicitly require go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The time bin size is 5 ms (0.005 s), producing 1000 time bins over the [-2.5, 2.5) second window. No temporal rebinning is applied after the initial binning.

ii.
```python
DT = 0.005
TMIN = -2.5
TMAX = 2.5

def make_time_edges() -> np.ndarray:
    return np.arange(TMIN, TMAX + DT * 0.5, DT, dtype=np.float64)
```

iii. The paper states "each bin is 5 ms" for decoding analyses. The reference code default `params.dt = 1/200 = 0.005`. Some figure scripts use 10 ms bins, but the 5 ms bin matches the paper's decoding specification.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input is derived from the common time axis computed from the binning parameters (TMIN, TMAX, DT), not from any specific raw data variable. It represents the bin centers of the neural data time grid.

ii.
```python
def make_time_centers(edges: np.ndarray) -> np.ndarray:
    return edges[:-1] + DT / 2.0

# In process_session:
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. Since all trials are aligned to the go cue and share the same time grid, the input is simply the time axis (bin centers) relative to the go cue, identical for every trial. This follows the instructions which specify "Time from go cue onset in seconds" as a continuous, time-varying input.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. The time centers are computed as `edge + DT/2` for each bin edge, producing values from -2.4975 to 2.4975 seconds. The result is broadcast to shape `(1, n_timepoints)` for each trial.

ii.
```python
time_centers = make_time_centers(time_edges)  # edges[:-1] + DT/2.0
input_trials.append(time_centers[None, :].astype(np.float32))
```

iii. This is a straightforward computation that provides the time from go cue onset at each bin center.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input time axis is derived from the same bin edges used for neural data binning, so alignment is inherent. Both neural and input data share the same 1000-bin time grid.

ii.
```python
time_edges = make_time_edges()
time_centers = make_time_centers(time_edges)
# Same time_centers used for both neural binning and input construction
```

iii. No separate alignment step is needed because the input IS the time axis of the neural data.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `obj.bp.R`, which is 1 for right-lick trials and 0 for left-lick trials.

ii.
```python
R = to_vector(bp["R"], float).astype(int)
lick_direction = R[keep_trials]
```

iii. The reference code uses `obj.bp.R` and `obj.bp.L` to distinguish trial types. The AI uses only `R` since it's binary (R=1 means right, R=0 means left), matching the decoder task specification (left=0, right=1).

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The raw `R` values are directly used as per-trial labels (0 for left, 1 for right), then broadcast across all time bins for each trial. No additional processing is applied.

ii.
```python
output_trials.append(
    np.vstack([
        np.full(time_centers.size, lick_direction[tr], dtype=np.int64),
        # ...
    ])
)
```

iii. The instructions specify lick direction as left=0, right=1, which directly maps to the raw `R` field.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `obj.bp.autowater`, where `autowater=1` indicates water-cued (WC) trials and `autowater=0` indicates delayed-response (DR) trials.

ii.
```python
autowater = to_vector(bp["autowater"], float).astype(int)
context = 1 - autowater[keep_trials]
```

iii. The CONVERSION_NOTES.md states: "Raw code uses autowater==1 as WC." The polarity is inverted to match the decoder task specification (WC=0, DR=1).

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The raw `autowater` values are inverted (`1 - autowater`) to produce the target encoding (WC=0, DR=1), then broadcast across time bins per trial.

ii.
```python
context = 1 - autowater[keep_trials]
# Then broadcast:
np.full(time_centers.size, context[tr], dtype=np.int64),
```

iii. Simple polarity inversion to match the decoder task's encoding convention.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `obj.bp.hit`, where `hit=1` indicates correct trials and `hit=0` indicates incorrect (miss) trials.

ii.
```python
hit = to_vector(bp["hit"], float).astype(int)
miss = to_vector(bp["miss"], float).astype(int)
outcome = hit[keep_trials]
# Validation:
if not np.all((hit[keep_trials] + miss[keep_trials]) == 1):
    raise ValueError(...)
```

iii. The AI validates that hit and miss are mutually exclusive on kept trials. No-response and early trials are excluded by the trial mask, so remaining trials are either hit or miss.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The raw `hit` values are directly used as per-trial labels (0 for incorrect, 1 for correct), broadcast across time bins. A validation check confirms hit+miss=1 for all kept trials.

ii.
```python
outcome = hit[keep_trials]
np.full(time_centers.size, outcome[tr], dtype=np.int64),
```

iii. The instructions specify incorrect=0, correct=1, which maps directly to hit=0/1.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from DLC trajectory data in `obj.traj`, specifically tongue-related features from camera views 1 and 2: `tongue`, `left_tongue`, `right_tongue` (view 1) and `top_tongue`, `topleft_tongue`, `bottom_tongue`, `bottomleft_tongue` (view 2).

ii.
```python
TONGUE_FEATURES = {
    1: ["tongue", "left_tongue", "right_tongue"],
    2: ["top_tongue", "topleft_tongue", "bottom_tongue", "bottomleft_tongue"],
}
```

iii. These features match the tongue-related DLC tracking points described in the reference code's `getKinematicsFromVideo.m` and `getKinematics.m`.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each tongue feature: (1) x/y positions are extracted from DLC trajectory arrays and interpolated to the neural time axis using the video offset correction. (2) Velocity is computed as `np.gradient` of position. (3) For tongue features, NaN velocities are set to 0 (tongue not visible = not moving). (4) Speed is computed as `sqrt(xvel^2 + yvel^2)`. (5) Speeds from all tongue features are averaged (mean across features).

ii.
```python
def aligned_position(obj, view_index_one_based, feat_name, time_centers, align_times, vidshift):
    # Interpolates DLC positions to neural time axis
    old_time = frame_times - vidshift - float(align_times[trial])
    interp = interp1d(old_time, xy, axis=0, kind="linear", ...)
    # Tongue: no nearest-fill

def feature_velocity(xpos, ypos, feat_name):
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    # Tongue: nan_to_num with 0
    xv = np.nan_to_num(xv, nan=0.0)
    yv = np.nan_to_num(yv, nan=0.0)

def aggregate_speed(obj, feature_map, time_centers, align_times, vidshift):
    speed = np.sqrt(np.square(xvel) + np.square(yvel))
    # Average across features
    agg = np.divide(summed, np.maximum(count, 1), dtype=np.float32)
```

iii. The velocity computation matches `findVelocity.m` (gradient of position, tongue NaN->0). The aggregation to a single scalar speed is a necessary deviation from the reference code (which keeps features separate) because the decoder task requires one tongue velocity output.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue speed is discretized into two bins using a rank-based median split per session. Values in the bottom 50% of all (time, trial) values get label 0; the top 50% get label 1.

ii.
```python
def percentile_threshold(traces, percentile=50.0):
    return float(np.percentile(flat, percentile))

def discretize_trace(traces, threshold):
    flat = traces.reshape(-1)
    order = np.argsort(flat, kind="mergesort")
    out = np.zeros(flat.size, dtype=np.int64)
    out[order[flat.size // 2 :]] = 1
    return out.reshape(traces.shape)

tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)
```

iii. The instructions specify per-session 50th percentile thresholding (0: < 50th percentile, 1: >= 50th percentile). The AI uses a rank-based approach that ensures exactly 50/50 split. Note: the `threshold` parameter is computed but not actually used in `discretize_trace`; the function relies on rank ordering instead.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Tongue velocity is aligned by interpolating DLC positions to the same neural time axis (time_centers) using video-offset-corrected frame times relative to the go cue.

ii.
```python
taxis = time_centers + ADVANCE_MOVEMENT  # ADVANCE_MOVEMENT = 0.0
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, xy, axis=0, kind="linear", ...)
xy_aligned = interp(taxis)
```

iii. This matches the reference code's `findPosition.m` / `getKinematicsFromVideo.m` alignment logic: `interp1(frameTimes - vidshift - alignTimes(trix), ts, taxis)`.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from DLC trajectory data for paw features from the bottom camera view (view 2): `top_paw` and `bottom_paw`.

ii.
```python
PAW_FEATURES = {
    2: ["top_paw", "bottom_paw"],
}
```

iii. The paper states "paws were tracked using only the bottom view," which matches this selection.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Processing follows the same pipeline as tongue velocity but with non-tongue handling: (1) positions interpolated to neural time axis, (2) missing positions filled with nearest values, (3) velocity computed as gradient of position, (4) baseline velocity subtracted (median of diff), (5) NaN velocities filled with nearest values, (6) speed = sqrt(xvel^2 + yvel^2), (7) speeds averaged across paw features.

ii.
```python
def feature_velocity(xpos, ypos, feat_name):
    basederiv = np.nanmedian(diffs, axis=0)
    xv = np.gradient(tsinterp[:, 0])
    yv = np.gradient(tsinterp[:, 1])
    if "tongue" not in feat_name:
        xv = xv - base[0]
        yv = yv - base[1]
        xv = fill_nearest_1d(xv)
        yv = fill_nearest_1d(yv)

def aligned_position(...):
    if "tongue" not in feat_name:
        xpos[:, trial] = fill_nearest_1d(xpos[:, trial])
        ypos[:, trial] = fill_nearest_1d(ypos[:, trial])
```

iii. The baseline subtraction and nearest-fill for non-tongue features matches `findVelocity.m` exactly. The aggregation to a single paw speed is a necessary deviation.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Same rank-based median split as tongue velocity, applied per session.

ii.
```python
paw_thr = percentile_threshold(paw_speed, 50.0)
paw_bin = discretize_trace(paw_speed, paw_thr)
```

iii. Same approach as tongue velocity discretization.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Same interpolation-based alignment as tongue velocity, using video-offset-corrected frame times.

ii.
```python
# Same alignment logic via aligned_position -> interp1d
old_time = frame_times - vidshift - float(align_times[trial])
```

iii. Same approach as tongue velocity alignment.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from external `motionEnergy_<subject>_<date>.mat` files (field `me.data`) or from embedded `obj.me` if no external file exists. Frame times come from `obj.traj[0]` (view 1).

ii.
```python
def load_motion_energy_raw(obj, spec):
    if spec.motion_energy_path.exists():
        raw_me = read_mat(spec.motion_energy_path).get("me")
        # ...
    if "me" in obj:
        raw_me = obj["me"]
        # ...

@property
def motion_energy_path(self) -> Path:
    return DATA_ROOT / self.folder / f"motionEnergy_{self.subject}_{self.date}.mat"
```

iii. The reference code's `loadMotionEnergy.m` loads from external files, and `loadMotionEnergy_Behav.m` handles embedded `obj.me`. The AI handles both paths.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. Per-trial motion energy traces are interpolated from the video frame rate (~400 Hz) to the neural time axis using video-offset-corrected frame times. NaN values are filled with nearest-neighbor interpolation. The continuous trace is then discretized.

ii.
```python
def aligned_motion_energy(obj, raw_me, time_centers, align_times, vidshift):
    taxis = time_centers + ADVANCE_MOVEMENT
    for trial in range(min(len(me_data), n_trials)):
        old_time = frame_times - vidshift - float(align_times[trial])
        interp = interp1d(old_time, me_trial, kind="linear", ...)
        aligned[:, trial] = interp(taxis)
        aligned[:, trial] = fill_nearest_1d(aligned[:, trial])
    return np.nan_to_num(aligned, nan=0.0)
```

iii. This matches the reference `loadMotionEnergy.m`: `interp1(frameTimes - vidshift - alignTimes(trix), me.data{trix}, taxis)` followed by `fillmissing(me.data,'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Same rank-based per-session 50th percentile split as tongue and paw velocity.

ii.
```python
me_thr = percentile_threshold(motion_energy, 50.0)
me_bin = discretize_trace(motion_energy, me_thr)
```

iii. The instructions specify per-session 50th percentile thresholding. The paper uses a manual per-session threshold for move/non-move classification, but the decoder task explicitly requires the 50th percentile approach.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is interpolated to the same neural time axis using video-offset-corrected frame times, identical to the kinematic alignment approach.

ii.
```python
old_time = frame_times - vidshift - float(align_times[trial])
interp = interp1d(old_time, me_trial, kind="linear", ...)
aligned[:, trial] = interp(taxis)
```

iii. Matches the reference `loadMotionEnergy.m` alignment logic.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled: (1) Trials with `NdroppedFrames` all NaN are skipped for video data. (2) Missing video frame times fall back to synthetic 400 Hz timing. (3) NaN positions are filled with nearest-neighbor for non-tongue features; tongue NaN velocities become 0. (4) Motion energy NaN values are filled with nearest-neighbor then remaining NaN become 0. (5) Sessions where JEB24 had behavior trials beyond the last neural trial are trimmed. (6) Nested motion energy file structures (JEB15) are recursively unwrapped. (7) Sessions with <2 valid trials or <10 neurons are skipped.

ii.
```python
# Dropped frames check:
if dropped is not None:
    dropped_arr = np.asarray(dropped)
    if dropped_arr.size and np.isnan(np.asarray(dropped_arr, dtype=float)).all():
        continue

# Fallback frame times:
def get_frame_times(trial_view, n_frames):
    if frame_times is None:
        return (np.arange(n_frames, dtype=np.float32) + 1.0) / 400.0

# Neural coverage trimming:
neural_coverage_mask = (np.arange(trial_mask.size, dtype=np.int32) + 1) <= max_trial_with_spikes

# Nested ME unwrapping:
def unwrap_embedded_motion_energy(raw_me):
    while isinstance(data, dict) and "data" in data and id(data) not in seen:
        data = data["data"]
```

iii. The CONVERSION_NOTES.md documents fixes for JEB15 nested motion energy and JEB24 neural coverage gaps. The AI handles multiple edge cases robustly.

## 11-a. What are the most time-consuming steps of the code?

i. According to the CONVERSION_NOTES.md and code structure: (1) Loading large MATLAB `.mat` files via `pymatreader` is the main I/O cost. (2) Per-neuron spike binning across all kept trials in `binned_neuron_trials`. (3) Kinematic interpolation for multiple DLC features across multiple views and trials in `aligned_position`.

ii.
```python
# Loading:
obj = read_mat(spec.data_path)["obj"]

# Per-neuron binning (looped per neuron):
for out_idx, selected in enumerate(selected_neurons):
    rates = binned_neuron_trials(probe, selected.neuron_index, align_times, keep_trials, time_edges)

# Per-feature, per-trial interpolation:
for trial in range(n_trials):
    interp = interp1d(old_time, xy, axis=0, kind="linear", ...)
```

iii. The CONVERSION_NOTES.md reports ~10-12 s/session, totaling ~8 minutes for 44 sessions, which is below the 15-minute optimization threshold.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could be vectorized: (1) The per-trial loop in `aligned_position` for interpolating DLC features could be vectorized. (2) The per-trial loop in `feature_velocity` for computing gradients. (3) The per-column loop in `causal_gaussian_smooth` for convolution. (4) The per-trial loop in `aligned_motion_energy`.

ii.
```python
# Per-trial position interpolation loop:
for trial in range(n_trials):
    interp = interp1d(old_time, xy, axis=0, kind="linear", ...)

# Per-column smoothing loop:
for col in range(x_filt.shape[1]):
    out[:, col] = np.convolve(x_filt[:, col], kern, mode="same")

# Per-trial velocity loop:
for trial in range(xpos.shape[1]):
    xv = np.gradient(tsinterp[:, 0])
```

iii. These loops process trials/neurons independently and could potentially use vectorized operations or scipy's `fftconvolve` for batch processing. However, the total runtime is acceptable.

## 11-c. What processing does the code repeat multiple times?

i. (1) The `select_neurons` function is called twice when neural coverage filtering triggers re-selection (once before and once after trimming). (2) Video offset (`compute_video_offset`) is computed once per session but position interpolation happens separately for each feature within the same view. (3) Frame time extraction (`get_frame_times`) is repeated for each feature and trial rather than cached per view.

ii.
```python
# Double neuron selection:
selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)
# ... neural coverage check ...
if not np.array_equal(refined_trial_mask, trial_mask):
    selected_neurons, probe_summaries = select_neurons(obj, spec, trial_mask)

# Frame times recomputed per feature:
frame_times = get_frame_times(trial_view, ts.shape[0])
```

iii. The double neuron selection is functionally necessary to handle the neural coverage edge case. Frame time recomputation is redundant but has minimal performance impact.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. (1) The `percentile_threshold` is computed for each velocity output but is NOT actually used in `discretize_trace` (the function uses rank-based splitting instead). It's only used for diagnostic plots. (2) The `find_feature_index` function scans all trials' feature names repeatedly instead of caching once per view. (3) The code computes `tongue_speed_all`, `paw_speed_all`, and `motion_energy_all` for ALL trials (including filtered-out ones) before subsetting to `keep_trials`.

ii.
```python
# Threshold computed but not used in discretization:
tongue_thr = percentile_threshold(tongue_speed, 50.0)
tongue_bin = discretize_trace(tongue_speed, tongue_thr)  # threshold arg unused

# All-trial computation before subsetting:
tongue_speed_all, tongue_feats = aggregate_speed(obj, TONGUE_FEATURES, time_centers, align_times, vidshift)
tongue_speed = tongue_speed_all[:, keep_trials]  # subset after
```

iii. Computing behavioral features for all trials before subsetting is wasteful when many trials are filtered out. The percentile threshold is effectively dead code in the discretization path.
