# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI does not use the ONE API as the primary loader. Instead, it reads the frozen release manifest `code/code_zhang2025/data/bwm_release.csv`, groups it by `eid`, constructs filesystem paths under `/app/data/one_cache/...`, chooses revisions by filename globbing, then loads Parquet and NumPy files directly with `pd.read_parquet` and `np.load`.

ii. 
```python
RELEASE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def build_release_specs() -> tuple[list[SessionSpec], dict[str, int]]:
    release = pd.read_csv(RELEASE_CSV, dtype={"date": str, "subject": str, "lab": str})
    ...
    for eid, rows in release.groupby("eid", sort=False):
        ...
        trial_table=_find_file(alf, "_ibl_trials.table.pqt", "#2025-03-03#"),
        wheel_times=_find_file(alf, "_ibl_wheel.timestamps.npy"),
        wheel_position=_find_file(alf, "_ibl_wheel.position.npy"),
```

```python
trials = pd.read_parquet(spec.trial_table)
motion_times = np.asarray(np.load(motion_times_path, mmap_mode="r"), dtype=np.float64)
wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says the local cache contains extra sessions/probes beyond the paper freeze, so it chose `bwm_release.csv` as the authoritative release list and resolved files directly from disk rather than through ONE.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken from the `subject` column of the frozen release CSV. Each session spec stores its subject string, and the final output uses sorted unique subject names plus a `subject_idx` lookup per retained session.

ii.
```python
specs.append(SessionSpec(
    eid=str(eid), subject=str(first["subject"]), lab=str(first["lab"]),
    date=str(first["date"]), number=number, session_dir=session_dir,
    ...
))
```

```python
subjects = sorted({s["spec"].subject for s in sessions})
subject_lookup = {s: i for i, s in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[s["spec"].subject] for s in sessions], dtype=np.int32),
```

iii. The AI’s notes say the frozen release CSV is the authoritative subject/session inventory for the conversion, so no path parsing or API subject lookup is used beyond that manifest.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The code groups the CSV rows by `eid` and creates one `SessionSpec` per `eid`, combining all probes listed for that session.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    first = rows.iloc[0]
    ...
    probes: list[ProbeFiles] = []
    for row in rows.itertuples(index=False):
        probe_root = alf / row.probe_name / "pykilosort"
        ...
    specs.append(SessionSpec(... probes=tuple(probes)))
```

iii. In the notes, the AI explicitly treats the frozen release as a session list and says simultaneous probes belong to one behavioral session.

## 1-d. How are the data split into trials?

i. Trials are taken as rows of the session trial table `_ibl_trials.table.pqt`. After loading the DataFrame, the code uses boolean masks and retained row indices to choose which trial rows survive.

ii.
```python
trials = pd.read_parquet(spec.trial_table)
raw_n = len(trials)
ref_mask = reference_trial_mask(trials)
raw_idx = np.flatnonzero(ref_mask)
```

iii. The AI follows the standard ALF interpretation that the trials table is already one row per trial; no extra segmentation logic is introduced.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies a two-stage trial filter. First, `reference_trial_mask` requires non-missing `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`, keeps reaction times in `[0.08, 2.0]` seconds, removes no-choice trials, and removes trials whose `feedback_times - goCue_times` exceeds 10 s. Second, it drops trials whose wheel or whisker traces do not cover the full `[-0.5, 1.5]` s stimulus-aligned window, and it excludes sessions left with fewer than two trials.

ii.
```python
def reference_trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = ["stimOn_times", "choice", "feedback_times", "probabilityLeft",
                "firstMovement_times", "feedbackType"]
    mask = trials[required].notna().all(axis=1).to_numpy().copy()
    rt = (trials["firstMovement_times"] - trials["stimOn_times"]).to_numpy()
    duration = (trials["feedback_times"] - trials["goCue_times"]).to_numpy()
    mask &= (rt >= 0.08) & (rt <= 2.0)
    mask &= ~(duration > 10.0)
    mask &= trials["choice"].to_numpy() != 0
    return mask
```

```python
wheel_good, _, wheel_ie = coverage_mask(wheel_times, begins, ends)
motion_good, _, motion_ie = coverage_mask(motion_times, begins, ends)
stream_good = wheel_good & motion_good
raw_idx = raw_idx[stream_good]
if raw_idx.size < 2:
    return None, "fewer than two trials with complete wheel/whisker coverage"
```

iii. In `CONVERSION_NOTES.md` Step 4 and Step 5, the AI says it is following the supplied Zhang reference code’s missing-event, RT, no-choice, and trial-duration mask, then adding complete wheel/whisker coverage because the requested outputs require those continuous traces.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derives `neural` from `spikes.times.npy` and `spikes.clusters.npy` for every probe in a session. It also loads `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy`, but those are used only to assign region labels, not to build the neural time series themselves.

ii.
```python
pf = ProbeFiles(
    name=row.probe_name,
    spikes_times=_find_file(probe_root, "spikes.times.npy", "#2024-05-06#"),
    spikes_clusters=_find_file(probe_root, "spikes.clusters.npy", "#2024-05-06#"),
    clusters_channels=_find_file(probe_root, "clusters.channels.npy", "#2024-05-06#"),
    channel_ids=_find_file(probe_root, "channels.brainLocationIds_ccf_2017.npy", "#2024-05-06#"),
)
```

```python
spike_times = np.load(probe.spikes_times, mmap_mode="r")
spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
...
flat = clusters[keep] * N_BINS + bins[keep]
counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
```

iii. The notes describe the source as frozen-release Neuropixels spike times plus synchronized trial/behavior streams, with cluster/channel metadata used for anatomy.

## 2-b. How is the `neural` data processed?

i. For each session, the AI merges all probes listed in the frozen release, keeps every cluster on those probes, bins spikes into half-open 20 ms bins over `[-0.5, 1.5)` relative to stimulus onset, and stores the result as float32 spike counts. It does not divide by bin width, so the stored values are counts rather than firing rates.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

```python
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
neural = [np.zeros((n_neurons, N_BINS), dtype=np.float32) for _ in range(n_trials)]
...
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
flat = clusters[keep] * N_BINS + bins[keep]
counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
neural[j][offset:offset + n_clusters] = counts
```

iii. In `CONVERSION_NOTES.md` Step 5 and Step 6, the AI says it intentionally kept “all-unit ... spike counts in half-open 20 ms bins” to match what it believed was the reference cache’s use of all clusters.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI does not apply a neural quality-control filter. It keeps all clusters in the frozen release for every probe, and it keeps all Beryl labels produced by the atlas mapping, including `root` and `void`.

ii.
```python
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
```

```python
native = br.id2acronym(channel_ids[cluster_channels])
region_names.extend(br.acronym2acronym(native, mapping="Beryl").tolist())
```

```python
"neural_representation": "all-unit Kilosort 2.5 spike counts in half-open 20 ms bins, float32",
```

iii. The AI’s notes repeatedly justify this by saying the Zhang decoder cache uses `qc=None` and therefore “all clusters,” even though the data paper also reports a well-isolated-neuron subset.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The neural data are aligned to `trials.stimOn_times`. For each retained trial, the code forms a window from `stimOn_times - 0.5` s to `stimOn_times + 1.5` s and bins spikes relative to the start of that window.

ii.
```python
stim = behavior["stim_times"]
begins, ends = stim + OFF_START, stim + OFF_END
...
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    ...
    bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
```

iii. The notes say the explicit decoder task requires a common stimulus-onset alignment, so the code uses one shared stimulus-centered window for every stream.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 20 ms bins (`BIN_SIZE = 0.020`) and 100 bins per trial over a 2 s window. No further temporal rebinning is applied.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

```python
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
```

iii. The AI’s notes cite the reference cache and method paper as using a 2 s window with 20 ms bins, and keep that resolution unchanged.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI derives this input from the session’s stimulus-onset alignment event `trials["stimOn_times"]`, but the actual values stored are a fixed relative time vector `RELATIVE_BIN_ENDS` rather than per-trial event times.

ii.
```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

```python
stim = trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx]
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
```

iii. In the notes, the AI says it chose bin-end coordinates because it interpreted the reference behavior interpolation code as labeling each spike-count bin by its right edge.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code does not compute a trial-specific signal from the raw table. Instead, it defines a fixed 100-sample vector running from `-0.48` to `1.50` seconds in 20 ms steps and repeats that same vector for every retained trial.

ii.
```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

```python
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. The AI explicitly justifies this in `CONVERSION_NOTES.md` Step 5: it wanted a common time-varying input and chose the right edge of each half-open spike-count bin as that coordinate.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input is aligned bin-for-bin with the neural data. The neural counts use half-open bins over `[-0.5, 1.5)`, and the time input stores the right edge of each of those bins.

ii.
```python
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
```

```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
...
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. The notes say this was intentional: the behavior code appeared to return bin-end timestamps, so the AI used those same bin edges as the common decoder time axis.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw trial-table column `probabilityLeft`.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    ...
```

```python
probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
block_no_all = trial_number_in_block(probs_all)
```

iii. The AI’s notes say there is no explicit block-id field in the raw table, so block membership must be reconstructed from runs of constant `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The code finds block boundaries where `probabilityLeft` changes, then assigns each trial a zero-based index within its block. It computes this on the unfiltered trial sequence and only afterwards subsets to retained trials, so excluded trials still advance the counter.

ii.
```python
starts = np.r_[0, np.flatnonzero(probability_left[1:] != probability_left[:-1]) + 1]
ends = np.r_[starts[1:], n]
out = np.empty(n, dtype=np.int32)
for start, end in zip(starts, ends):
    out[start:end] = np.arange(end - start, dtype=np.int32)
```

```python
"block_trial": block_no_all[raw_idx].astype(np.float32),
```

iii. `CONVERSION_NOTES.md` Step 5 says this preserves the trial’s real experimental position in the original block rather than compressing after filtering.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from the raw `trials["choice"]` column.

ii.
```python
choices = trials["choice"].to_numpy(dtype=np.float64)[raw_idx]
```

```python
"choice": (choices == 1).astype(np.int8),
```

iii. The AI does not introduce a secondary source for choice; all choice labels come straight from the retained trial rows.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI converts each retained choice to a binary label with `(choices == 1).astype(np.int8)` and repeats that single label across all 100 time bins for the trial. The code comments and metadata indicate the AI believed source `-1` means left and source `+1` means right.

ii.
```python
"choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
```

```python
outputs.append(np.vstack((
    np.full(N_BINS, s["choice"][j], dtype=np.int8),
    np.full(N_BINS, s["prior"][j], dtype=np.int8),
    ...
)))
```

```python
"choice_mapping": "source -1 (left) -> 0; source +1 (right) -> 1",
```

iii. The AI did not justify the sign convention from the human reference; its notes mostly focus on the need to repeat static per-trial outputs across time to keep array shapes consistent.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It is derived from `trials["probabilityLeft"]`.

ii.
```python
probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
priors = probs_all[raw_idx]
```

iii. The AI’s notes say the task prior is already represented by the trial-table `probabilityLeft` values.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps retained `probabilityLeft` values `0.2`, `0.5`, and `0.8` to categorical codes `0`, `1`, and `2` using `np.searchsorted`, then repeats the result across all 100 time bins of the trial.

ii.
```python
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
```

```python
np.full(N_BINS, s["prior"][j], dtype=np.int8)
```

iii. In Step 5 of the notes, the AI says this exact mapping is required by the decoder specification.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It is derived from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
wheel_raw_position = np.asarray(np.load(spec.wheel_position, mmap_mode="r"), dtype=np.float64)
```

iii. The notes say the raw dataset stores wheel position and timestamps, so speed has to be reconstructed from them.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates wheel position to a uniform 1 kHz grid with the bundled IBL wheel helper, filters/differentiates it with `velocity_filtered`, takes the absolute velocity as speed, and linearly resamples that speed onto the 100 stimulus-aligned target times for each trial.

ii.
```python
wheel_position, wheel_times = interpolate_position(wheel_raw_times, wheel_raw_position, freq=1000)
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
wheel_speed = np.abs(wheel_velocity)
```

```python
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. In `CONVERSION_NOTES.md` Step 5, the AI says it intentionally reused the bundled IBL wheel implementation rather than rewriting wheel filtering logic itself.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI computes global 1/3 and 2/3 quantiles across all retained wheel-speed samples from all retained sessions, then assigns classes `0`, `1`, and `2` with `np.searchsorted`.

ii.
```python
wheel_values = np.concatenate([s["wheel"].ravel() for s in behavior_sessions])
wheel_q = np.quantile(wheel_values, [1 / 3, 2 / 3])
```

```python
np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8)
```

iii. The notes and trajectory explicitly justify this as a deliberate choice: the AI wanted class meanings to be globally consistent across sessions rather than session-specific.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-centered trial window as the neural data and sampled at the same 100 target times stored in `RELATIVE_BIN_ENDS`.

ii.
```python
stim = trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx]
begins, ends = stim + OFF_START, stim + OFF_END
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. The AI’s notes say all outputs had to share one stimulus-aligned tensor, so wheel speed was moved onto the same grid as the spike counts.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It is derived from either the left or right camera motion-energy stream: `<side>Camera.ROIMotionEnergy.npy` together with `_ibl_<side>Camera.times.npy`. The code prefers the left view and falls back to the right.

ii.
```python
def choose_motion_stream(alf: Path) -> tuple[str, Path, Path] | None:
    for view, revision in (("left", "#2025-05-29#"), ("right", "#2025-05-31#")):
        times = _optional_file(alf, f"_ibl_{view}Camera.times.npy")
        values = _optional_file(alf, f"{view}Camera.ROIMotionEnergy.npy", revision)
        if times is not None and values is not None:
            return view, times, values
    return None
```

iii. The notes describe this as “left preferred; right fallback,” following the reference behavior-loading logic as the AI understood it.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace as-is, checks monotonicity and finiteness, and linearly interpolates it onto the 100 stimulus-aligned target times for each retained trial. It does not filter or normalize the motion-energy values before discretization.

ii.
```python
motion_times = np.asarray(np.load(motion_times_path, mmap_mode="r"), dtype=np.float64)
motion_values = np.asarray(np.load(motion_values_path, mmap_mode="r"), dtype=np.float64)
if not (np.all(np.diff(motion_times) > 0) and np.all(np.isfinite(motion_values))):
    return None, "nonmonotonic/nonfinite whisker-motion stream"
```

```python
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. The notes say the released trace is already the desired whisker motion-energy quantity, so the AI only resamples it onto the common decoder grid.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Like wheel speed, whisker motion energy is discretized with global 1/3 and 2/3 quantiles computed across all retained aligned whisker samples from all retained sessions.

ii.
```python
whisker_values = np.concatenate([s["whisker"].ravel() for s in behavior_sessions])
whisker_q = np.quantile(whisker_values, [1 / 3, 2 / 3])
```

```python
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8)
```

iii. The AI explicitly justifies this in the notes as a deliberate “global tertiles” design choice for cross-session consistency.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The whisker trace is aligned to each trial’s `stimOn_times` and resampled at the same 100 relative target times used by the neural data and wheel output.

ii.
```python
stim = trials["stimOn_times"].to_numpy(dtype=np.float64)[raw_idx]
begins, ends = stim + OFF_START, stim + OFF_END
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. The AI’s notes say all dynamic outputs must share the same common stimulus-aligned time axis as the neural tensor.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing or invalid behavior streams usually cause trial or session exclusion: no usable whisker stream, malformed wheel/motion arrays, nonmonotonic timestamps, nonfinite values, or fewer than two surviving trials all exclude a session. Trials with incomplete wheel or whisker coverage are dropped. For neural data, however, the code does not check spike-stream coverage trial-by-trial; if a trial window contains no spikes, the neural matrix row stays all zeros.

ii.
```python
motion = choose_motion_stream(spec.session_dir / "alf")
if motion is None:
    return None, "missing paired left and right whisker-motion streams"
...
if raw_idx.size < 2:
    return None, "fewer than two trials with complete wheel/whisker coverage"
```

```python
if behavior is None:
    excluded.append({"eid": spec.eid, "reason": str(reason)})
    print(f"Excluded {spec.eid}: {reason}", flush=True)
    continue
```

```python
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    if i1 <= i0:
        continue
```

iii. The notes say missing required outputs should be dropped, while README text justifies all-zero neural trials as faithful representations when the spike recording ends before late behavioral trials.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies release inventory building, spike loading/binning, and writing the enormous pickle as the dominant costs. In practice, the code spends most per-session work in spike binning and a fixed up-front pass reading probe headers to validate the frozen release inventory.

ii.
```python
cluster_inventory += int(np.load(pf.clusters_channels, mmap_mode="r").shape[0])
```

```python
for probe, n_clusters in zip(spec.probes, n_per_probe):
    spike_times = np.load(probe.spikes_times, mmap_mode="r")
    spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
    ...
```

```python
with args.outpicklefile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. `CONVERSION_NOTES.md` Step 6 and Step 7 explicitly discuss the cost of broad spike-array I/O, per-session spike binning, and the large pickle write.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The main remaining nontrivial loop is the nested per-trial spike-binning loop inside each probe. The block-counter loop and the per-trial input/output assembly loop are also scalar Python loops. Behavior interpolation is already mostly vectorized over all targets at once.

ii.
```python
for start, end in zip(starts, ends):
    out[start:end] = np.arange(end - start, dtype=np.int32)
```

```python
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    ...
    counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
    neural[j][offset:offset + n_clusters] = counts
```

```python
for j in range(n_trials):
    inputs.append(...)
    outputs.append(...)
```

iii. The AI’s notes argue that its main optimization already replaced full-session rescans with `searchsorted` plus `np.bincount`, but these per-trial loops still remain in the implementation.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats some work for checking and reporting: it reloads the first probe’s spike arrays after binning to perform a direct spot check, applies wheel/whisker discretization again in plotting and again in final assembly, and repeatedly filesystem-globs for specific revisioned files while constructing session specs.

ii.
```python
p0 = spec.probes[0]
st = np.load(p0.spikes_times, mmap_mode="r")
sc = np.load(p0.spikes_clusters, mmap_mode="r")
direct = np.sum((st >= begins[0]) & (st < begins[0] + BIN_SIZE) & (sc == 0))
```

```python
wheel_cls = np.searchsorted(np.asarray(thresholds["wheel_speed"]), session["wheel"][0], side="right")
whisk_cls = np.searchsorted(np.asarray(thresholds["whisker_motion_energy"]), session["whisker"][0], side="right")
```

```python
np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8),
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8),
```

iii. The AI did not frame these as major problems in the notes, but they are visible in the code as extra verification and reporting work layered onto the main conversion.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several diagnostic steps are not needed by downstream decoders: optional plotting captures raw wheel position and motion traces, `make_processing_plot` renders PNG audits, the raw neural spot check reloads data only to assert correctness, `internal_validate` recomputes class counts and shape checks after assembly, and large metadata fields such as `excluded_sessions`, `source_release_inventory`, and `source_trial_indices` are preserved even though the downstream decoder does not consume them.

ii.
```python
if capture_plot:
    ...
    result["plot_capture"] = {
        "wheel_times": wheel_times[wi] - stim[j],
        "wheel_position": wheel_position[wi],
        "wheel_speed": wheel_speed[wi],
        "motion_times": motion_times[mi] - stim[j],
        "motion_values": motion_values[mi],
    }
```

```python
if args.show_processing:
    for session in sessions[:2]:
        make_processing_plot(session, thresholds)
```

```python
stats = internal_validate(data)
print("Internal validation:", json.dumps(stats, indent=2), flush=True)
```

iii. The notes present these as sanity checks and documentation aids rather than core conversion steps, so they are intentionally extra processing that downstream training does not require.
