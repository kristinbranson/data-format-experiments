# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not load the full 44-session corpus used in the human reference. It hard-codes a 12-session subset in `SESSION_SPECS`, assumes all `data_structure_*.mat` files are under `/app/data/Ephys_Behavior`, reads those session files directly with `h5py`, and reads the paired `motionEnergy_*.mat` files with `scipy.io.loadmat`.

ii. 
```python
DATA_ROOT = Path("/app/data/Ephys_Behavior")

SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    ...
    SessionSpec("JEB19", "2023-04-18", (1,)),
)
```

```python
with h5py.File(spec.data_path, "r") as handle:
    bp = load_behavior(handle)
    ...
```

```python
def load_motion_energy(path: Path) -> np.ndarray:
    motion = loadmat(path, simplify_cells=True)["me"]
```

iii. The justification in `CONVERSION_NOTES.md` is that the AI treated the “Figure 8” two-context cohort as the applicable task, rather than the full release. It explicitly says it chose “the exact 12 sessions enumerated by Figure 8 context code” because those sessions contain substantial WC and DR context blocks.

## 1-b. How are the data split into subjects?

i. Subjects are encoded directly in the hard-coded `SessionSpec.animal` field. At assembly time the AI creates `subjects` by preserving first appearance order from `SESSION_SPECS`, then maps each session to that subject index.

ii. 
```python
@dataclass(frozen=True)
class SessionSpec:
    animal: str
    date: str
    probes: tuple[int, ...]
```

```python
subjects = list(dict.fromkeys(spec.animal for spec in specs))
subject_idx = np.array([subjects.index(spec.animal) for spec in specs], dtype=np.int64)
```

iii. The notes justify this as preserving “seven explicit source identifiers” rather than collapsing or renaming subjects to force agreement with the paper’s reported count.

## 1-c. How are the data split into sessions?

i. One `SessionSpec` entry is treated as one session, and one session becomes one element of `neural`, `input`, `output`, and `brain_region_idx`. The AI only uses the hard-coded 12-session subset from `Ephys_Behavior`.

ii. 
```python
SESSION_SPECS = (
    SessionSpec("JEB6", "2021-04-18", (2,)),
    ...
)
```

```python
for spec in specs:
    session, diag = convert_session(spec)
    converted.append(session)
```

iii. The notes say this was an intentional cohort decision: “Use exactly the 12 sessions enumerated by Figure 8 context code. Exclude DR-only, randomized-delay, and behavior-only sessions.”

## 1-d. How are the data split into trials?

i. Trials are defined by the per-trial behavioral arrays in `obj/bp`; the AI validates that those arrays all match `Ntrials`. After that, trial indices are handled as raw integer indices into those behavioral arrays, and retained trials are `np.flatnonzero` of the trial filter.

ii. 
```python
def load_behavior(handle: h5py.File) -> dict[str, np.ndarray]:
    ...
    ntrials = int(np.asarray(handle["obj/bp/Ntrials"]).squeeze())
    if any(len(value) != ntrials for value in bp.values()):
        raise ValueError("Behavior fields do not all match Ntrials")
```

```python
keep_trials = ~bp["early"] & ~bp["stim"]
raw_trials = np.flatnonzero(keep_trials)
```

iii. The AI’s notes justify this by treating `obj.bp` as the authoritative trial table and by using raw trial indices consistently through neural and behavioral alignment.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered only by excluding `early` and photostimulation (`stim.enable`) trials. Ignore trials are kept. The AI does not implement the human reference’s extra drop of trials that extend beyond available ephys recording.

ii. 
```python
keep_trials = ~bp["early"] & ~bp["stim"]
raw_trials = np.flatnonzero(keep_trials)
if len(raw_trials) < 2:
    raise ValueError(f"{spec.session_id}: fewer than two retained trials")
```

iii. The notes say this was deliberate: “Trials exclude `early` or `stim.enable`, but retain hit/miss/no,” because the decoder task explicitly requires incorrect and ignore outcomes.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data are derived from cluster-level spike-sorted variables in `obj/clu`: `quality`, `trial`, and `trialtm`, together with per-trial `bp.ev.goCue` for alignment.

ii. 
```python
quality_refs = matlab_cell_refs(group["quality"])
trial_refs = matlab_cell_refs(group["trial"])
trialtm_refs = matlab_cell_refs(group["trialtm"])
```

```python
bp["goCue"] = direct_field(handle, "obj/bp/ev/goCue", np.float64)
```

iii. The notes describe this as the paper’s extracellular ephys source: sorted spikes aligned by subtracting each trial’s go cue.

## 2-b. How is the `neural` data processed?

i. Spikes are aligned to go cue, binned into 5 ms bins, converted to spikes/s, and then smoothed with the AI’s custom “reference” smoother: a 15-sample causalized Gaussian implemented with a left prefix and `lfilter`. Dual probes are concatenated before downstream use.

ii. 
```python
def build_neural(clusters: list[dict], bp: dict[str, np.ndarray], raw_trials: np.ndarray) -> np.ndarray:
    ...
    bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
    np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
    return reference_smooth(counts / DT)
```

```python
def reference_smooth(values: np.ndarray, n: int = 15) -> np.ndarray:
    padded = np.concatenate((values[..., :n], values), axis=-1)
    causal_coefficients = KERNEL[n // 2 :]
    filtered = lfilter(causal_coefficients, [1.0], padded, axis=-1)
    return filtered[..., n:].astype(np.float32, copy=False)
```

iii. The notes justify this as reproducing MATLAB `gausswin(15)` after causalization, and they explicitly say the AI chose “the exact causal kernel” used in the released figure workflow.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two neuron filters. First it excludes clusters whose lower-cased `quality` label is in `{"garbage", "gabrga", "noisy", "real?"}`. Then it applies a strict `>1 Hz` low-firing-rate filter computed from condition-averaged PSTHs over seven conditions.

ii. 
```python
QUALITY_EXCLUDE = {"garbage", "gabrga", "noisy", "real?"}
LOW_FR_HZ = 1.0
```

```python
if quality in QUALITY_EXCLUDE:
    continue
```

```python
mean_rates[unit] = np.nanmean(np.stack(condition_psths))
keep = mean_rates > LOW_FR_HZ
```

iii. The notes justify this as matching the “context-workflow” scripts rather than the generic loader defaults, and they emphasize the reconstructed count of 521 retained units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each spike is aligned by subtracting the go cue of its own trial: `trial_times - bp["goCue"][spike_trials]`.

ii. 
```python
aligned = cluster["trial_times"][use] - bp["goCue"][spike_trials]
```

iii. The notes explicitly identify go-cue alignment as the paper/code convention and as the decoder’s required temporal anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 5 ms bins over a `[-2.5, 2.5)` s window, giving 1000 time bins. No secondary temporal rebinning is applied.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.005
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
N_TIME = len(TIME)
```

iii. The notes justify 5 ms as the paper/reference bin size and treat `TIME` as the common grid for neural, input, and dynamic outputs.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. It is not read from a raw data variable. The AI defines it analytically from the chosen go-cue-centered bin grid, so it is derived from the alignment convention and the `TMIN/TMAX/DT` constants.

ii. 
```python
TIME = (np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2).astype(np.float32)
...
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The notes justify this as using the neural bin centers themselves as the decoder input variable.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No extra processing is applied beyond constructing the 1000 bin centers and repeating them for every retained trial.

ii. 
```python
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The notes explicitly describe this as “one-row continuous float32 time series” equal to the analytical bin centers.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. It is exactly the same time grid used for neural binning, so each input time point corresponds to the center of the neural bin at the same index.

ii. 
```python
bins = np.floor((aligned - TMIN) / DT).astype(np.int64)
...
input_trials = [TIME[None, :].copy() for _ in raw_trials]
```

iii. The AI’s notes justify this by making `TIME` the shared “same aligned time centers as neural data.”

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from `bp["R"]`, `bp["hit"]`, `bp["miss"]`, and also `bp["no"]` for the no-lick class.

ii. 
```python
if bp["no"][raw]:
    lick = 2
elif bp["hit"][raw]:
    lick = 1 if bp["R"][raw] else 0
elif bp["miss"][raw]:
    lick = 0 if bp["R"][raw] else 1
```

iii. The notes justify this as decoding actual lick direction rather than instructed side, and they explicitly say “miss trials invert R/L; ignores map to none.”

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps no-response trials to class 2, hit trials to the instructed side, and miss trials to the opposite side. That per-trial class is then broadcast across all 1000 time bins.

ii. 
```python
static[kept] = (lick, context, outcome)
```

```python
out = np.empty((6, N_TIME), dtype=np.int8)
out[:3] = static[trial, :, None]
```

iii. The justification in the notes is that this implements “actual lick rather than instructed side” and preserves the task-required “none” class.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived directly from `bp["autowater"]`.

ii. 
```python
context = 0 if bp["autowater"][raw] else 1
```

iii. The notes explicitly justify `autowater=1` as WC and `autowater=0` as DR.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The AI directly relabels `autowater` into categorical classes `WC=0` and `DR=1`, then broadcasts the result across time within each trial.

ii. 
```python
context = 0 if bp["autowater"][raw] else 1
...
out[:3] = static[trial, :, None]
```

iii. The notes say this follows the context scripts and the prompt’s requested output coding.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the behavioral outcome flags, primarily `bp["hit"]` and `bp["miss"]`, with the remaining case treated as ignore.

ii. 
```python
outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
```

iii. The notes justify this as keeping hit, miss, and no/ignore trials because the decoder task explicitly requires three outcome classes.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps miss to `incorrect=0`, hit to `correct=1`, and all other retained trials to `ignore=2`. The resulting per-trial label is broadcast across time.

ii. 
```python
outcome = 0 if bp["miss"][raw] else (1 if bp["hit"][raw] else 2)
...
out[:3] = static[trial, :, None]
```

iii. The notes state this is a task-driven exception to the paper’s usual omission of ignore trials.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side-camera trajectory feature named `"tongue"` in `obj/traj`, together with that camera’s `frameTimes`, the session video offset, and `bp["goCue"]`. The AI does not use the bottom-camera tongue feature.

ii. 
```python
tongue[kept_trial], tongue_visible[kept_trial], tongue_x, tongue_y = feature_speed(
    handle, side, int(raw_trial), "tongue", aligned_side_time, True
)
```

iii. The notes justify using the “main side-camera `tongue` landmark” as the tongue-tip representation and emphasize shared go-cue/video-offset alignment.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. The AI interpolates tongue x and y coordinates onto the 5 ms neural grid, treats finite interpolated x/y as visible, computes the Euclidean magnitude of `np.gradient` on that aligned grid, sets non-visible bins to NaN, and later median-splits visible values per session. There is no two-view combination, no per-view normalization, and no explicit per-run Gaussian smoothing.

ii. 
```python
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
y = interpolate_with_nans(aligned_time, trajectory[:, 1, feature_index], TIME)
visible = np.isfinite(x) & np.isfinite(y)
...
x_velocity = np.gradient(x_for_velocity)
y_velocity = np.gradient(y_for_velocity)
speed = np.hypot(x_velocity, y_velocity)
speed[~visible] = np.nan
```

iii. The notes justify this as “Euclidean magnitude of x/y first derivatives” with “visibility precedes filling,” but they do not claim to use both tongue views.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. Tongue speed is thresholded per session at the median of all visible tongue-speed samples. Bins below the median become class 0, bins at or above the median become class 1, and bins not marked visible become class 2.

ii. 
```python
tongue_threshold = float(np.nanmedian(tongue[tongue_visible]))
```

```python
classes = np.full(N_TIME, 2, dtype=np.int8)
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. The notes explicitly justify “per-session median over retained visible/valid aligned samples” and a distinct class 2 for missing visibility.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. The AI computes a per-session video offset from bitcode timing, subtracts that offset and the trial’s go cue from frame times, then interpolates tongue positions onto the neural `TIME` grid before differentiating them.

ii. 
```python
def video_offset_seconds(handle: h5py.File, bp: dict[str, np.ndarray]) -> float:
    neural_bit_start = direct_field(handle, "obj/sglx/bitcode/bitstart", np.float64)
    fs = float(np.asarray(handle["obj/sglx/fs"]).squeeze())
    return mode_value(neural_bit_start) / fs - mode_value(bp["bitStart"])
```

```python
aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
...
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. The notes justify the bitcode-derived per-session offset, especially because they found JEB19 sessions where a fixed 0.5 s shift would be wrong.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom-camera `"top_paw"` feature in `obj/traj`, together with bottom-camera `frameTimes`, the video offset, and `bp["goCue"]`.

ii. 
```python
paw[kept_trial], paw_visible[kept_trial], paw_x, paw_y = feature_speed(
    handle, bottom, int(raw_trial), "top_paw", aligned_bottom_time, False
)
```

iii. The notes justify this as using `top_paw` as the paper’s reliable paw feature.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. The AI interpolates paw x and y onto the neural time grid, nearest-fills gaps for velocity computation, computes `np.gradient` on the aligned grid, subtracts a baseline derivative estimate, converts to speed magnitude, then masks non-visible bins back to NaN before later thresholding.

ii. 
```python
if is_tongue:
    x_for_velocity, y_for_velocity = x, y
else:
    x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
...
x_velocity = np.gradient(x_for_velocity)
y_velocity = np.gradient(y_for_velocity)
...
speed = np.hypot(x_velocity, y_velocity)
speed[~visible] = np.nan
```

iii. The notes justify this with “visibility precedes filling” so class 2 is preserved, while filling is used only to compute neighboring visible derivatives.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. Paw speed is split at the per-session median of visible paw-speed samples. Visible bins are coded 0/1 depending on whether they are below or at/above the median, and non-visible bins are class 2.

ii. 
```python
paw_threshold = float(np.nanmedian(paw[paw_visible]))
```

```python
classes = np.full(N_TIME, 2, dtype=np.int8)
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. The notes explicitly justify a per-session median threshold and a separate missing-visibility class.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. The AI aligns bottom-camera `frameTimes` by subtracting the per-session video offset and the trial’s go cue, then interpolates the paw trajectory onto the neural `TIME` grid.

ii. 
```python
aligned_bottom_time = bottom_frames - offset - bp["goCue"][raw_trial]
...
x = interpolate_with_nans(aligned_time, trajectory[:, 0, feature_index], TIME)
```

iii. The notes justify using one common neural/video time axis and a per-session bitcode-derived offset.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is derived from the standalone `motionEnergy_<session>.mat` file, specifically from the unwrapped `me["data"]` array returned by `load_motion_energy`.

ii. 
```python
def load_motion_energy(path: Path) -> np.ndarray:
    motion = loadmat(path, simplify_cells=True)["me"]
    data = motion["data"]
    if isinstance(data, dict):
        data = data["data"]
    return np.atleast_1d(data)
```

iii. The notes justify using the paired motion-energy files rather than recomputing motion from video pixels.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. The AI takes the per-trial motion-energy trace, chooses a time base, interpolates it to the neural `TIME` grid, nearest-fills edge/interior NaNs, and later median-splits the aligned values per session. If frame times do not match the motion trace or are unusable, it falls back to a nominal 400 Hz clock with a 0.5 s shift before go-cue alignment.

ii. 
```python
if len(trial_motion) == len(side_frames) and np.sum(np.isfinite(side_frames)) >= 2:
    motion_times = aligned_side_time
else:
    motion_times = np.arange(1, len(trial_motion) + 1, dtype=np.float64) / 400
    motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The notes justify the 400 Hz fallback by saying the first converter version marked one JEB19 motion trace as “no video,” but the reference MATLAB code instead recovers it with this fallback.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy is split at the per-session median of valid aligned samples. Valid bins are coded 0/1 below vs. at/above the median, and invalid bins default to class 2.

ii. 
```python
valid_motion = np.isfinite(motion)
motion_threshold = float(np.nanmedian(motion[valid_motion]))
```

```python
classes = np.full(N_TIME, 2, dtype=np.int8)
classes[valid] = (values[trial, valid] >= thresholds[output_index - 3]).astype(np.int8)
```

iii. The notes explicitly say the task overrides the paper’s manual `moveThresh` and requires a per-session 50th-percentile split.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. The AI aligns motion energy using the side-camera timing when possible: side-camera `frameTimes`, minus session video offset, minus trial go cue, then interpolation to the neural `TIME` grid. If those timestamps are unusable it substitutes a nominal 400 Hz clock shifted by 0.5 s before go-cue alignment.

ii. 
```python
aligned_side_time = side_frames - offset - bp["goCue"][raw_trial]
...
interpolated = interpolate_with_nans(motion_times, trial_motion, TIME)
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The justification in the notes is the explicit JEB19 repair: the fallback was added because the released MATLAB loader would recover that trace rather than discarding it.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI generally preserves trials and repairs missingness rather than dropping data. Missing features return NaN trajectories and therefore class 2 after discretization. Paw trajectories are nearest-filled for derivative computation but then remasked using the original visibility mask. Motion energy with unusable timestamps is reconstructed with a 400 Hz fallback clock and then nearest-filled after interpolation. Behavior inconsistencies, non-finite go cues, or structural mismatches raise errors.

ii. 
```python
if feature not in names:
    nan = np.full(N_TIME, np.nan, dtype=np.float64)
    return nan, np.zeros(N_TIME, dtype=bool), nan, nan
```

```python
else:
    x_for_velocity, y_for_velocity = fill_nearest(x), fill_nearest(y)
...
speed[~visible] = np.nan
```

```python
motion_times = np.arange(1, len(trial_motion) + 1, dtype=np.float64) / 400
motion_times = motion_times - 0.5 - bp["goCue"][raw_trial]
...
motion[kept_trial] = fill_nearest(interpolated)
```

iii. The notes explicitly defend the JEB19 motion fallback as restoring a trace the MATLAB loader would keep, and they defend visibility-before-filling so that class 2 is not erased for tongue or paw.

## 11-a. What are the most time-consuming steps of the code?

i. The AI’s code is likely dominated by per-session HDF5 reads plus repeated trial-wise trajectory extraction and interpolation for tongue, paw, and motion. Neural spike accumulation is vectorized within each unit, but camera processing still loops over retained trials.

ii. 
```python
for kept_trial, raw_trial in enumerate(raw_trials):
    side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
    ...
    tongue[kept_trial], ... = feature_speed(...)
    paw[kept_trial], ... = feature_speed(...)
```

```python
for unit, cluster in enumerate(clusters):
    ...
    np.add.at(counts[unit], (mapped_trials[in_window], bins[in_window]), 1)
```

iii. The notes say variable-length MATLAB references and variable-length DLC traces are the main runtime cost, and they contrast this with a rejected fully materialized `mat73`-style load that was slower and more memory-heavy.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The obvious non-vectorized loops are the retained-trial loop in `align_kinematics_and_motion`, the cluster loop in `selected_clusters`, the cluster loop in `low_fr_filter`, and the per-trial output construction loop in `discretize_outputs`. The AI did at least vectorize within-unit spike accumulation using `np.add.at`.

ii. 
```python
for kept_trial, raw_trial in enumerate(raw_trials):
    ...
```

```python
for unit, cluster in enumerate(clusters):
    ...
```

```python
for trial in range(len(static)):
    out = np.empty((6, N_TIME), dtype=np.int8)
```

iii. The notes justify keeping the trial-wise video loops because the camera traces are ragged, with variable numbers of frames per trial.

## 11-c. What processing does the code repeat multiple times?

i. The code re-reads some trajectory data for the same trial more than once. In `align_kinematics_and_motion` it first loads side and bottom trajectories to get frame times, then `feature_speed` loads the same trial trajectory again and also looks up feature names again. Subject indexing at assembly also repeatedly calls `subjects.index`.

ii. 
```python
side_trajectory, side_frames = trajectory_for_trial(handle, side, int(raw_trial))
del side_trajectory
...
tongue[kept_trial], ... = feature_speed(handle, side, int(raw_trial), "tongue", aligned_side_time, True)
```

```python
names = feature_names_for_trial(handle, group, trial)
trajectory, frame_times = trajectory_for_trial(handle, group, trial)
```

iii. I did not find an explicit justification in the notes for this repeated work. The only implicit justification is implementation convenience while operating directly on nested HDF5 references.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code builds several diagnostics that are not part of the saved dataset: full `mean_rates_all_quality_units`, sampled neural/kinematic traces, visibility fractions, first-trial raw x/y traces, and optional processing plots. These are useful for auditing but are discarded from the final pickle.

ii. 
```python
diagnostics = {
    "session_id": spec.session_id,
    "mean_rates_all_quality_units": mean_rates,
    "neural_sample": neural[: min(12, len(clusters)), 0].copy(),
    "kinematic_sample": kin_info["sample"],
    "thresholds": kin_info["thresholds"],
    "static": static,
    "output_sample": outputs[0],
    "visibility_fractions": visibility.mean(axis=(1, 2)),
}
```

```python
if args.show_processing:
    for diag in diagnostics[:2]:
        path = plot_processing(diag)
```

iii. The notes justify this as part of the required sanity-checking and critical-review workflow rather than as output-generation logic.
