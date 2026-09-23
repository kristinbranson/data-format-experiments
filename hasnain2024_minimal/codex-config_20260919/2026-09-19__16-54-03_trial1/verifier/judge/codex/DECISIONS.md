# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads only `data_structure_*.mat` files from `/app/data/Ephys_Behavior`, not from both electrophysiology folders. It discovers sessions by globbing the directory and then keeps only animals listed in `ALM_PROBE`, expecting exactly 12 sessions total. Each session is opened directly with `h5py`, while motion energy is loaded separately from the paired `motionEnergy_*.mat` file with `scipy.io.loadmat`.

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

```python
with h5py.File(path, "r") as h5:
    ...
motion_trials = load_motion_energy(motion_path)
```

iii. The trajectory says the AI decided that the "requested WC/DR context output points to the fixed-delay electrophysiology collection" and later that the "paper’s exact two-context neural cohort is identifiable from the figure scripts: 12 sessions." That is the stated reason it narrowed loading to one folder and one 12-session cohort.

## 1-b. How are the data split into subjects?

i. Subjects are derived from the filename prefix before the date, e.g. `JEB19` from `data_structure_JEB19_2023-04-19.mat`. The script then builds `subjects` as the sorted unique animal names from the discovered session files and assigns `subject_idx` per session from that lookup.

ii. 
```python
files = session_files(data_dir)
subjects = sorted({animal for _, animal, _ in files})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
...
subject_idx.append(subject_lookup[animal])
```

iii. The trajectory repeatedly refers to the cohort by animal name, e.g. "12 sessions from six mice" and later reports the final per-subject counts from those filename-derived ids. There is no separate justification beyond using the manifest-like animal prefixes in filenames.

## 1-c. How are the data split into sessions?

i. One `.mat` file is treated as one session. The script iterates over the 12 globbed `data_structure_*.mat` files in `Ephys_Behavior`; each processed file becomes one entry in the top-level `neural`, `input`, and `output` session lists.

ii. 
```python
for number, (path, animal, date) in enumerate(files, start=1):
    result = process_session(path, animal, date)
    neural.append(result.pop("neural"))
    inputs.append(result.pop("input"))
    outputs.append(result.pop("output"))
```

iii. The trajectory justifies this by saying the Figure 8 scripts define the "exact two-context neural cohort." There is no attempt to merge fixed-delay and randomized-delay folders; the AI believed the requested task should use only this one collection.

## 1-d. How are the data split into trials?

i. Trials are defined by the length of `bp.ev.goCue`, and all other per-trial arrays are indexed against that length. After building a boolean `trial_mask`, the kept trials are `np.flatnonzero(trial_mask)`, and every per-trial stream is then built by iterating over those original trial indices.

ii. 
```python
go = np.asarray(bp["ev/goCue"]).ravel().astype(np.float64)
n_trials = go.size
...
trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
kept_trials = np.flatnonzero(trial_mask)
```

```python
for local_trial, original_trial in enumerate(kept_trials):
    neural_trials.append(np.ascontiguousarray(rates[local_trial], dtype=np.float32))
    ...
    output_trials.append(output)
```

iii. The trajectory does not dwell on trial splitting specifically; it assumes the Bpod trial arrays are authoritative. The code comments justify the retained set as "control trials" with hits, misses, or ignores and finite go cues.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered by excluding early-lick trials, photostimulation trials, trials with non-finite go cues, and any trial that is not labeled as hit, miss, or ignore. The AI does not apply the reference solution’s additional cutoff for trials that continue after neural recording ends; instead it later maps spikes only onto the retained trial list.

ii. 
```python
trial_mask = ~early & ~stim & np.isfinite(go) & (hit | miss | ignore)
kept_trials = np.flatnonzero(trial_mask)
if kept_trials.size < 2:
    raise RuntimeError(f"{animal} {date} has fewer than two usable trials")
```

iii. The trajectory explicitly says "only non-photostimulation control trials are used" and that it will "retain all non-early trials including ignores because they are an explicitly requested outcome." There is no trajectory evidence that it noticed or handled trials extending beyond the recording.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from the selected probe’s cluster table in `obj/clu`, specifically each unit’s `trial`, `trialtm`, and `quality`, together with `bp.ev.goCue` for alignment. The AI restricts this to one manifest-selected ALM probe per animal.

ii. 
```python
probe_number = ALM_PROBE[animal]
cluster_group = h5[h5["obj/clu"][probe_number - 1, 0]]
...
quality = matlab_char(h5, cluster_group["quality"][unit, 0])
spike_trial = referenced_array(h5, cluster_group["trial"][unit, 0]).ravel().astype(np.int64) - 1
trial_time = referenced_array(h5, cluster_group["trialtm"][unit, 0]).ravel().astype(np.float64)
aligned_time = trial_time - go[spike_trial]
```

iii. The trajectory says probe selection comes from the "Figure 8 scripts" and that ALM units are restricted to the "manifest-selected probe." It also describes go-cue alignment as part of matching the source analysis.

## 2-b. How is the `neural` data processed?

i. For each curated unit, the AI bins aligned spike times into 10 ms bins from -2.5 s to 2.5 s, smooths those binned counts with a 15-bin causal Gaussian-like kernel implemented with `lfilter`, and divides by `DT` to express the result as firing rate. The result per trial is a `(n_units, 500)` float32 matrix.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
SMOOTH_BINS = 15
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
```

```python
def smooth_rates(counts: np.ndarray) -> np.ndarray:
    padded = np.concatenate((counts[..., :SMOOTH_BINS], counts), axis=-1)
    filtered = lfilter(CAUSAL_KERNEL, [1.0], padded, axis=-1)
    return (filtered[..., SMOOTH_BINS:] / DT).astype(np.float32)
```

iii. The trajectory explicitly says, "Neural activity will use the repository’s 10 ms bins and 15-bin causal Gaussian smoothing over -2.5 to +2.5 s." The script docstring also says this was meant to follow the Figure 8 code path rather than the broader reference pipeline.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI applies two filters. First it excludes units whose exact string `quality` equals one of `garbage`, `gabrga`, `noisy`, or `real?`, using a case-sensitive comparison. Then it computes a source-style mean firing rate from several Figure 8 condition PSTHs and keeps only units with mean rate greater than 1 Hz. It also aborts a session if fewer than 10 curated units remain.

ii. 
```python
REJECTED_QUALITIES = {"garbage", "gabrga", "noisy", "real?"}
...
quality = matlab_char(h5, cluster_group["quality"][unit, 0])
if quality in REJECTED_QUALITIES:
    continue
...
mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
if mean_rate > 1.0:
    selected_units.append((aligned_time, spike_trial, quality, mean_rate))

if len(selected_units) < 10:
    raise RuntimeError(f"{animal} {date} has only {len(selected_units)} curated units")
```

iii. The trajectory says this was meant to match `findClusters.m/isMember` and a Figure 8 low-firing-rate filter: "ALM units are restricted to the manifest-selected probe, accepted quality labels, and firing rate above 1 Hz." It also explicitly notes the choice was "deliberately case-sensitive."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Spike times are aligned by subtracting each spike’s trial-specific go cue time from `trialtm`, so time zero is `bp.ev.goCue` on that trial. Those aligned times are then binned into the common neural time axis.

ii. 
```python
aligned_time = trial_time - go[spike_trial]
...
bins = np.floor((aligned_time - TMIN) / DT).astype(np.int64)
```

iii. The trajectory repeatedly says the dataset is "aligned based on Go cue onset" and that the code matches the repository’s go-cue alignment. No additional offset or interpolation is used for spikes.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data use 10 ms bins. Spikes are rebinned onto this 10 ms grid over the -2.5 s to +2.5 s window, giving 500 time bins per trial.

ii. 
```python
DT = 0.01
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
N_TIME = TIME.size
...
return (filtered[..., SMOOTH_BINS:] / DT).astype(np.float32)
```

iii. The trajectory states this choice directly: "500 × 10 ms bins per trial" and "10 ms bins." The AI believed this matched the Figure 8 decoding scripts it chose to follow.

## 3-a. What variables in the raw data is `input` *Time from go cue onset in seconds* derived from?

i. The input time variable is not read directly from a raw field. It is a synthetic time axis derived from the chosen alignment window around the go cue, represented by the center of each 10 ms bin in `TIME`.

ii. 
```python
TMIN = -2.5
TMAX = 2.5
DT = 0.01
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
...
time_input = TIME.astype(np.float32)[None, :]
```

iii. The trajectory does not offer a separate argument here beyond saying everything is aligned to go cue onset. The synthetic bin-center time axis is implied by its chosen neural grid.

## 3-b. What processing is involved in computing `input` *Time from go cue onset in seconds*?

i. No raw-data transformation is applied beyond constructing the fixed `TIME` vector once and copying it into every trial. Each trial’s input is a `(1, 500)` float32 array containing those bin centers.

ii. 
```python
time_input = TIME.astype(np.float32)[None, :]
...
input_trials.append(time_input.copy())
```

iii. There is no explicit trajectory discussion of this step. The code treats the time axis as the decoder input required by the prompt.

## 3-c. How is the `input` *Time from go cue onset in seconds* aligned with the neural data?

i. The input uses the same `TIME` array that defines the neural bins, so each input time point corresponds exactly to the center of the neural bin at that column. Neural spike times are histogrammed onto that same window and `DT`.

ii. 
```python
TIME = np.arange(TMIN, TMAX, DT, dtype=np.float64) + DT / 2
...
bins = np.floor((aligned_time - TMIN) / DT).astype(np.int64)
...
time_input = TIME.astype(np.float32)[None, :]
```

iii. The trajectory justification is implicit: the AI states that both neural data and decoder input are aligned to go cue onset on the same 10 ms grid.

## 4-a. What variables in the raw data is `output` *Lick direction* derived from?

i. Lick direction is derived from the trial outcome flags `bp.hit`, `bp.miss`, and `bp.no`, together with the instructed/right target flag `bp.R`. The AI uses `bp.no` explicitly to identify ignore trials and assign the `none` class.

ii. 
```python
hit = np.asarray(bp["hit"]).ravel().astype(bool)
miss = np.asarray(bp["miss"]).ravel().astype(bool)
ignore = np.asarray(bp["no"]).ravel().astype(bool)
right_target = np.asarray(bp["R"]).ravel().astype(bool)
```

iii. The trajectory says it would "derive lick direction from target side plus outcome, so incorrect trials are labeled by the animal’s actual opposite-side response." It also says ignore trials are retained because they are a required decoder output.

## 4-b. What processing is involved in computing `output` *Lick direction*?

i. The AI maps ignore trials to class 2 (`none`), hit trials to the instructed side from `R`, and miss trials to the opposite side of `R`. It then repeats that scalar label across all time bins of the trial.

ii. 
```python
if ignore[original_trial]:
    lick_direction = 2  # none
elif hit[original_trial]:
    lick_direction = 1 if right_target[original_trial] else 0
else:
    lick_direction = 0 if right_target[original_trial] else 1

output[0] = lick_direction
```

iii. The trajectory justifies this explicitly: incorrect trials should be labeled by the animal’s "actual opposite-side response," while ignores are kept as their own class for the requested decoder task.

## 5-a. What variables in the raw data is `output` *Behavioral context* derived from?

i. Behavioral context is derived from `bp.autowater`. The AI interprets `autowater=True` as the water-cued (WC) context and `False` as delayed response (DR).

ii. 
```python
autowater = np.asarray(bp["autowater"]).ravel().astype(bool)
...
context = 0 if autowater[original_trial] else 1
```

iii. The trajectory refers to the WC/DR context distinction as the key reason it selected this cohort. There is no separate debate in the trajectory about how `autowater` should be mapped.

## 5-b. What processing is involved in computing `output` *Behavioral context*?

i. The processing is a direct binary relabeling: WC is coded as 0 when `autowater` is true, otherwise DR is coded as 1. That per-trial label is repeated across time bins.

ii. 
```python
context = 0 if autowater[original_trial] else 1
...
output[1] = context
```

iii. The code comment and metadata both treat this as a straightforward decoder label rather than a signal requiring further processing.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from the trial-level hit, miss, and ignore flags: `bp.hit`, `bp.miss`, and `bp.no`. The AI uses `bp.no` explicitly instead of inferring ignore as "not hit and not miss."

ii. 
```python
hit = np.asarray(bp["hit"]).ravel().astype(bool)
miss = np.asarray(bp["miss"]).ravel().astype(bool)
ignore = np.asarray(bp["no"]).ravel().astype(bool)
```

iii. The trajectory emphasizes retaining ignore trials because ignore is "a required decoder output in this task." That is the main justification for using the explicit ignore flag.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome is mapped to three classes: ignore to 2, hit to 1 (`correct`), and miss to 0 (`incorrect`). The per-trial value is then broadcast across the full time axis.

ii. 
```python
if ignore[original_trial]:
    outcome = 2
elif hit[original_trial]:
    outcome = 1
else:
    outcome = 0

output[2] = outcome
```

iii. The trajectory explicitly says the ignore outcome is "deliberately retained" even though the paper often compares only hits and misses.

## 7-a. What variables in the raw data is `output` *Tongue velocity* derived from?

i. Tongue velocity is derived from the side camera trajectory in `obj.traj`, using the DLC feature named `tongue`. The AI uses `frameTimes`, `ts`, the per-session video offset from `obj.sglx.bitcode.bitstart` and `obj.bp.ev.bitStart`, and the trial-specific go cue. It does not use the bottom camera’s `top_tongue` feature.

ii. 
```python
trajectory_refs = np.asarray(h5["obj/traj"]).ravel()
side = h5[trajectory_refs[0]]
bottom = h5[trajectory_refs[1]]
tongue_index = find_feature_indices(h5, side, "tongue")
...
tongue = trajectory_signal(h5, side, int(trial), tongue_index, go[trial], offset)
```

iii. The trajectory shows the AI inspected both camera feature lists, but the implemented code follows its summary that video processing uses "DLC tongue (side camera) and top-paw (bottom camera)." It did not justify excluding the bottom tongue view beyond that design choice.

## 7-b. What processing is involved in computing `output` *Tongue velocity*?

i. For each kept trial, the side-camera tongue x/y coordinates are temporally interpolated onto the 10 ms neural grid. The AI then computes `np.gradient` of the interpolated x and y traces and takes the Euclidean norm as tongue speed, leaving bins with non-finite coordinates as `NaN`. There is no explicit DLC likelihood threshold, no smoothing of x/y before differentiation, no separation into contiguous visible runs, and no normalization/averaging across two tongue views.

ii. 
```python
def trajectory_signal(...):
    ...
    aligned_frame_time = frame_time[:n] - offset - go_cue
    x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
    y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
    return x, y, aligned_frame_time
```

```python
def speed_from_position(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    speed = np.hypot(np.gradient(x), np.gradient(y))
    speed[~(np.isfinite(x) & np.isfinite(y))] = np.nan
    return speed
```

iii. The trajectory justification is mostly indirect. It says camera streams should use the source video offset and later describes "DLC tongue (side camera) ... Euclidean frame-to-frame velocity." It does not justify omitting likelihood-thresholding, smoothing, or the second tongue view.

## 7-c. How is `output` *Tongue velocity* thresholded into categories?

i. After all tongue-speed traces for a session are computed, the AI pools all finite values across trials, takes the 50th percentile, and labels each finite bin as 0 if below threshold or 1 if at/above threshold. Non-finite bins are labeled 2 (`not_visible`).

ii. 
```python
def categorize_session_signal(signals: list[np.ndarray]) -> tuple[list[np.ndarray], float | None]:
    finite_parts = [x[np.isfinite(x)] for x in signals if np.any(np.isfinite(x))]
    ...
    threshold = float(np.percentile(np.concatenate(finite_parts), 50))
    ...
    out = np.full(N_TIME, 2, dtype=np.int64)
    visible = np.isfinite(signal)
    out[visible] = (signal[visible] >= threshold).astype(np.int64)
```

iii. The trajectory says movement variables use "per-session visible-sample medians," so the threshold rule was chosen directly from the prompt’s median split requirement.

## 7-d. How is `output` *Tongue velocity* aligned with the neural data?

i. Alignment is done by computing one per-session video offset as the difference between the medians of the video bit-start times and Bpod bit-start times, subtracting that offset and the trial’s go cue from frame times, and then interpolating tongue x/y positions onto the same 10 ms `TIME` grid used for neural data.

ii. 
```python
def video_offset(h5: h5py.File) -> float:
    bit_start = np.asarray(h5["obj/bp/ev/bitStart"]).ravel()
    video_bit_start = np.asarray(h5["obj/sglx/bitcode/bitstart"]).ravel()
    fs = float(np.asarray(h5["obj/sglx/fs"]).squeeze())
    return float(np.nanmedian(video_bit_start / fs) - np.nanmedian(bit_start))
```

```python
aligned_frame_time = frame_time[:n] - offset - go_cue
x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
```

iii. The trajectory states that "camera and motion-energy streams use the repository's per-session video offset before interpolation onto the neural time axis." That is the AI’s explicit alignment rationale.

## 8-a. What variables in the raw data is `output` *Paw velocity* derived from?

i. Paw velocity is derived from the bottom camera trajectory in `obj.traj`, using the DLC feature `top_paw`, its `frameTimes`, and the same session-wide video offset and per-trial go cue used for tongue alignment.

ii. 
```python
bottom = h5[trajectory_refs[1]]
paw_index = find_feature_indices(h5, bottom, "top_paw")
...
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

iii. The trajectory summary and metadata explicitly describe the paw feature as "top-paw (bottom camera)." There is no further justification beyond matching that feature name in the tracking data.

## 8-b. What processing is involved in computing `output` *Paw velocity*?

i. Paw x/y positions are interpolated from frame times onto the 10 ms `TIME` grid and converted to speed by taking the Euclidean norm of their discrete gradients. As with tongue velocity, there is no explicit likelihood thresholding, no pre-differentiation smoothing, and no contiguous-run handling.

ii. 
```python
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
paw_velocity.append(
    np.full(N_TIME, np.nan) if paw is None else speed_from_position(paw[0], paw[1])
)
```

iii. The trajectory only justifies this generically as video-derived "Euclidean frame-to-frame velocity" aligned by the source video offset. It does not provide a separate argument for paw-specific preprocessing.

## 8-c. How is `output` *Paw velocity* thresholded into categories?

i. The AI applies the same session-level median split as for tongue velocity: all finite paw-speed values in a session are pooled, thresholded at the 50th percentile, and then categorized as 0 below threshold, 1 at/above threshold, or 2 for non-finite bins.

ii. 
```python
paw_cat, paw_threshold = categorize_session_signal(paw_velocity)
...
output[4] = paw_cat[local_trial]
```

iii. The trajectory’s generic "per-session visible-sample medians" statement applies here too; no paw-specific thresholding justification was given.

## 8-d. How is `output` *Paw velocity* aligned with the neural data?

i. Paw alignment uses the same `video_offset` and `goCue` subtraction as tongue, followed by interpolation onto the neural `TIME` grid. This makes each paw-velocity output vector the same length and sample times as the neural data.

ii. 
```python
offset = video_offset(h5)
...
paw = trajectory_signal(h5, bottom, int(trial), paw_index, go[trial], offset)
```

```python
aligned_frame_time = frame_time[:n] - offset - go_cue
x = interp_preserving_missing(aligned_frame_time, x_raw[:n], TIME)
y = interp_preserving_missing(aligned_frame_time, y_raw[:n], TIME)
```

iii. The trajectory does not distinguish paw from tongue here; both are treated as video streams that should be aligned by the same session-wide offset before interpolation to the neural time axis.

## 9-a. What variables in the raw data is `output` *Motion energy* derived from?

i. Motion energy is loaded from the session’s paired `motionEnergy_*.mat` file, specifically the `me["data"]` payload after unwrapping one nested `data` level when necessary. For alignment it reuses the side-camera frame times returned with the tongue trajectory.

ii. 
```python
def load_motion_energy(path: Path) -> list[np.ndarray]:
    mat = loadmat(path, simplify_cells=True)
    raw = mat["me"]["data"]
    if isinstance(raw, dict):
        raw = raw["data"]
```

```python
motion_path = path.with_name(path.name.replace("data_structure_", "motionEnergy_"))
motion_trials = load_motion_energy(motion_path)
```

iii. The trajectory notes that motion-energy files sit alongside the session files and that their structure can vary. It also states that motion-energy streams use the same video offset as other camera-derived signals.

## 9-b. What processing is involved in computing `output` *Motion energy*?

i. For each trial, the raw motion-energy trace is paired with side-camera frame times, linearly interpolated onto the 10 ms `TIME` grid, and then missing bins are filled by nearest-neighbor imputation with `nearest_fill`. The resulting continuous signal is later discretized by a session median split.

ii. 
```python
if n < 2 or not np.any(np.isfinite(raw_me[:n])):
    motion_energy.append(np.full(N_TIME, np.nan))
else:
    me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
    motion_energy.append(nearest_fill(me))
```

iii. The only explicit justification in the trajectory is that motion energy should use the "per-session video offset before interpolation onto the neural time axis." The nearest-neighbor fill is justified only by the helper docstring, which cites MATLAB `fillmissing(..., 'nearest')`.

## 9-c. How is `output` *Motion energy* thresholded into categories?

i. Motion energy uses the same categorizer as tongue and paw: the 50th percentile over all finite values in the session defines the threshold, finite bins become class 0 or 1, and non-finite bins become class 2 (`no_video` in the output label list).

ii. 
```python
motion_cat, motion_threshold = categorize_session_signal(motion_energy)
...
output[5] = motion_cat[local_trial]
```

iii. The trajectory’s stated rationale again is the prompt’s per-session median split for movement variables.

## 9-d. How is `output` *Motion energy* aligned with the neural data?

i. Motion energy is aligned indirectly through the side camera. For each trial, the AI takes the aligned side-camera frame times from the tongue trajectory, subtracting the session `video_offset` and trial `goCue`, and interpolates the raw motion-energy values onto the same `TIME` grid used by neural data.

ii. 
```python
tongue = trajectory_signal(h5, side, int(trial), tongue_index, go[trial], offset)
...
me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
motion_energy.append(nearest_fill(me))
```

iii. The trajectory explicitly says motion-energy streams use the repository’s video offset and are interpolated to the neural time axis. It does not mention any separate side-camera binning step.

## 10. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several missing-data cases by propagating `NaN` and then converting those bins to class 2 at discretization time. If `NdroppedFrames` is empty or non-finite, or if frame/trajectory arrays are malformed, `trajectory_signal` returns `None` and the whole trial’s tongue or paw trace becomes all-`NaN`. Motion energy becomes all-`NaN` when there is no matching tongue timing or insufficient finite samples; otherwise missing motion-energy bins are nearest-filled after interpolation. It also requires finite `goCue` to retain a trial at all.

ii. 
```python
if np.size(dropped) == 0 or not np.all(np.isfinite(dropped)):
    return None
...
if source_time.size < 2:
    return np.full(target_time.shape, np.nan, dtype=np.float64)
```

```python
if tongue is None or trial >= len(motion_trials):
    motion_energy.append(np.full(N_TIME, np.nan))
...
me = interp_preserving_missing(tongue[2][:n], raw_me[:n], TIME)
motion_energy.append(nearest_fill(me))
```

iii. The trajectory gives no extended missing-data discussion beyond saying video streams are aligned by interpolation and that ignore/control-trial filtering is preserved. The helper docstrings are the only explicit justification for the nearest-fill behavior.

## 11-a. What are the most time-consuming steps of the code?

i. The likely most time-consuming steps are the per-session loops over all units to compute low-firing-rate screening and trial-by-time histograms, plus the per-trial video interpolation and gradient calculations for tongue, paw, and motion energy. The code opens each session file once, but most bespoke processing happens inside those nested loops.

ii. 
```python
for unit in range(cluster_group["quality"].shape[0]):
    ...
    mean_rate = unit_mean_rate(aligned_time, spike_trial, condition_masks)
    if mean_rate > 1.0:
        selected_units.append((aligned_time, spike_trial, quality, mean_rate))
```

```python
for trial in kept_trials:
    tongue = trajectory_signal(...)
    paw = trajectory_signal(...)
    ...
```

iii. The trajectory does not explicitly benchmark bottlenecks. The closest justification is that the AI prioritized reproducing its chosen Figure 8 processing rather than optimizing runtime.

## 11-b. What loops in the code could have been vectorized to improve efficiency?

i. The main vectorizable loops are the loop over units for low-FR calculation and spike histogramming, and the loop over trials for tongue, paw, and motion-energy interpolation/categorization. `categorize_session_signal` also loops over session signals after concatenating them for thresholds.

ii. 
```python
for unit in range(cluster_group["quality"].shape[0]):
    ...
for unit_index, (aligned_time, spike_trial, _, _) in enumerate(selected_units):
    ...
for trial in kept_trials:
    ...
for signal in signals:
    ...
```

iii. The trajectory never raises vectorization as a goal. Its justifications focus on following the source scripts and obtaining valid decoder performance.

## 11-c. What processing does the code repeat multiple times?

i. The code repeats several computations: `unit_mean_rate` smooths and averages seven condition-specific PSTHs for every unit, `trajectory_signal` separately interpolates x and y for every trial and feature, and the same per-trial loop structure is repeated for tongue, paw, and motion energy. The script also stores `quality` and `mean_rate` for every selected unit even though both are later dropped from the top-level dataset.

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
...
for mask in condition_masks:
    ...
```

```python
selected_units.append((aligned_time, spike_trial, quality, mean_rate))
...
result.pop("qualities")
result.pop("mean_rates_hz")
```

iii. The trajectory explicitly mentions using "Figure 8 condition definitions" for the low-FR filter, which explains the repeated per-condition PSTH work. It does not justify the later-discarded bookkeeping beyond session metadata assembly.

## 11-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The clearest discarded work is collecting `qualities` and `mean_rates_hz` for each selected unit and then immediately removing them before building the final dataset. The code also computes and retains per-session thresholds and some session metadata that are not used by downstream decoder training, but those remain in metadata rather than being discarded. More broadly, interpolating full x/y trajectories just to reduce them to thresholded categorical speeds is more work than the final saved representation preserves.

ii. 
```python
selected_units.append((aligned_time, spike_trial, quality, mean_rate))
...
"qualities": [item[2] for item in selected_units],
"mean_rates_hz": [float(item[3]) for item in selected_units],
```

```python
result.pop("qualities")
result.pop("mean_rates_hz")
```

iii. The trajectory does not call this out. It only reports successful conversion, validation, and decoder accuracy, so this explanation is inferred from the implementation itself.
