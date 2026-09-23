# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI did not use the ONE API as the main loading path. It loaded a fixed session/probe inventory from `code_zhang2025/data/bwm_release.csv`, turned each row group into a session keyed by `eid`, then opened local ALF/Parquet/NumPy files directly under `/app/data/one_cache/<lab>/Subjects/<subject>/<date>/<session>/alf`. It also resolved revisions by globbing for the newest matching file and preferred specific staged revisions for trials and pykilosort outputs.

ii.
```python
FREEZE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def load_freeze() -> tuple[pd.DataFrame, list[dict]]:
    freeze = pd.read_csv(FREEZE_CSV)
    ...
    for eid, group in freeze.groupby("eid", sort=False):
        ...
        sessions.append({
            "eid": str(eid),
            "lab": str(first["lab"]),
            "subject": str(first["subject"]),
            "date": str(first["date"]),
            "session_number": int(first["session_number"]),
            "pids": group["pid"].astype(str).tolist(),
            "probe_names": group["probe_name"].astype(str).tolist(),
        })
```

```python
def session_path(row: pd.Series) -> Path:
    return (
        DATA_ROOT / str(row["lab"]) / "Subjects" / str(row["subject"])
        / str(row["date"]) / f"{int(row['session_number']):03d}"
    )
```

```python
def newest_file(base: Path, pattern: str, preferred_revision: str | None = None) -> Path | None:
    paths = sorted(base.glob(f"**/{pattern}"), key=revision_key)
    ...
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as matching the staged local release more reliably than offline ONE resolution: it wanted to “use the CSV freeze list and merge all probes from each EID,” “resolve paths only through `/app/data/one_cache`,” and “prefer staged `#2024-05-06#` pykilosort products, `#2025-03-03#` trial tables.” It framed direct ALF loading as a pragmatic substitute for unavailable or incomplete offline ONE revision resolution.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are taken directly from the freeze CSV’s `subject` column while sessions are enumerated. At dataset assembly, subjects are the sorted unique subject names and each session gets a `subject_idx` into that list.

ii.
```python
sessions.append({
    ...
    "subject": str(first["subject"]),
    ...
})
```

```python
subjects = sorted({r["info"]["subject"] for r in results})
subject_lookup = {subject: i for i, subject in enumerate(subjects)}
subject_idx = np.asarray(
    [subject_lookup[r["info"]["subject"]] for r in results], dtype=np.int32,
)
```

iii. The justification in the notes is that the freeze file already defines release membership and carries subject metadata, so no subject identity needs to be inferred from filenames beyond the session-path construction.

## 1-c. How are the data split into sessions?

i. Sessions are defined by unique `eid` values in `bwm_release.csv`. The AI groups freeze rows by `eid`, preserving file order, and treats each group as one session with one or more probes.

ii.
```python
for eid, group in freeze.groupby("eid", sort=False):
    first = group.iloc[0]
    sessions.append({
        "eid": str(eid),
        ...
        "pids": group["pid"].astype(str).tolist(),
        "probe_names": group["probe_name"].astype(str).tolist(),
    })
```

iii. In the notes, the AI explicitly resolved a consistency question in favor of “the 459/699/139 paper freeze” and said session order should “follow first EID occurrence in the freeze CSV.”

## 1-d. How are the data split into trials?

i. The AI loads one trials table per session and treats each row as one trial. After applying trial masks and stream-coverage masks, it keeps the surviving row indices in `source_indices`, and each retained row becomes one trial in `neural`, `input`, and `output`.

ii.
```python
def load_trials(info: dict) -> tuple[pd.DataFrame, Path]:
    ...
    trials = pd.read_parquet(path)
    return trials, path
```

```python
code_mask = trial_mask(trials)
code_indices = np.flatnonzero(code_mask)
...
source_indices = code_indices[keep_in_code]
...
"source_trial_indices": source_indices.astype(int).tolist(),
```

iii. The AI’s notes treat the trial table as the natural trial split and emphasize preserving “raw trial ordinal for trial-in-block calculation, then subset.”

## 1-e. How are trials filtered based on quality controls?

i. The AI used a broader filter than the human reference. First it required finite `stimOn_times`, `choice`, `feedback_times`, `probabilityLeft`, `firstMovement_times`, and `feedbackType`; reaction time had to be between 0.08 and 2.0 s; `choice` had to be nonzero; and, when present, `feedback_times - goCue_times` had to be at most 10 s. Then it required per-trial wheel coverage, whisker-motion coverage, and an additional “common neural coverage” condition ensuring every trial window fell within the overlapping spike-time extent of all probes in the session.

ii.
```python
def trial_mask(trials: pd.DataFrame) -> np.ndarray:
    required = [
        "stimOn_times", "choice", "feedback_times", "probabilityLeft",
        "firstMovement_times", "feedbackType",
    ]
    ...
    good &= reaction_time >= 0.08
    good &= reaction_time <= 2.0
    good &= trials["choice"].to_numpy() != 0
    if "goCue_times" in trials:
        duration = (
            trials["feedback_times"].to_numpy(dtype=float)
            - trials["goCue_times"].to_numpy(dtype=float)
        )
        good &= ~(duration > 10.0)
    return good
```

```python
neural_good = (
    (stim_code + OFF_START >= neural_coverage_start)
    & (stim_code + OFF_END <= neural_coverage_end)
)
...
stream_good = wheel_good & motion_good & neural_good
keep_in_code = np.flatnonzero(stream_good)
```

iii. The notes say to “apply the exact supplied mask” from the Zhang caching code, then exclude trials lacking valid wheel/whisker coverage, and later added neural coverage after the validator exposed three all-zero trials after ephys had ended. The trajectory explicitly records that this extra filter was added to remove those invalid trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural array is derived from `spikes.times.npy` and `spikes.clusters.npy` for each probe. The cluster metadata tables are also read to interpret cluster IDs and assign Beryl regions, but the activity matrix itself comes from spike times and spike-cluster assignments.

ii.
```python
spikes_times_path = directory / "spikes.times.npy"
spikes_clusters_path = directory / "spikes.clusters.npy"
...
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
```

```python
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
...
clusters = map_spike_clusters(raw_clusters, probe["cluster_ids"])
```

iii. The notes map “Revised `spikes.times`, `spikes.clusters` from every freeze-list probe” to `neural` and describe cluster/channel metadata as supporting region assignment rather than the core activity signal.

## 2-b. How is the `neural` data processed?

i. The AI merged all probes within a session, binned spikes into 100 half-open 20 ms bins over a `[-0.5, 1.5)` s window around stimulus onset, and stored raw spike counts as `float32`. It did not divide by bin width to convert counts to firing rates, did not smooth, and used bounded chunked `np.bincount` passes for efficiency.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

```python
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
keep = (bins >= 0) & (bins < N_BINS)
...
counts = np.bincount(
    flat, minlength=(last - first) * n_clusters * N_BINS,
).reshape(last - first, n_clusters, N_BINS)
output[first:last] = counts
```

```python
neural = np.zeros((len(source_indices), n_neurons, N_BINS), dtype=np.float32)
...
neural[:, offset:offset + n] = bin_probe(probe, starts, ends)
```

iii. The notes justify this by saying the Zhang caching pipeline uses spike counts rather than rates: “retain all sorted clusters, including multiunit; no smoothing/rate division,” and the overview line of `convert_data.py` says it follows “20 ms spike-count bins.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI chose not to filter neural data by QC labels. It kept all clusters present in each probe’s metrics table, counted spikes from all of them, and only recorded the number of `label >= 1` clusters as metadata. It also did not drop Beryl `void` or `root` clusters.

ii.
```python
metrics = pd.read_parquet(metrics_path)
cluster_ids = metrics["cluster_id"].to_numpy(dtype=np.int64)
n_clusters = len(metrics)
...
"n_good": int((metrics["label"].to_numpy() >= 1).sum()),
```

```python
n_neurons = sum(p["n_clusters"] for p in probes)
neural = np.zeros((len(source_indices), n_neurons, N_BINS), dtype=np.float32)
...
"n_good_label_clusters": int(sum(p["n_good"] for p in probes)),
```

```python
"neuron_filter": "all Kilosort clusters (qc=None), matching decoder reference code",
```

iii. This was a deliberate decision documented repeatedly in `CONVERSION_NOTES.md`: the AI argued that the Zhang decoder code uses `qc=None` and “all neurons,” so it should “preserve all 621,733 clusters” rather than apply the data paper’s 75,708-unit well-isolated filter.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trials are aligned to stimulus onset. For each retained trial, the code defines `starts = stim_times - 0.5` and `ends = stim_times + 1.5`, slices spikes in that absolute-time window, and bins them relative to `starts[trial]`, so each trial spans `[-0.5, 1.5)` around `stimOn_times`.

ii.
```python
stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
starts = stim_times + OFF_START
ends = stim_times + OFF_END
```

```python
lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
...
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
```

iii. The notes say the conversion should use “common stimulus-aligned 20 ms counts” and the trajectory repeatedly describes the representation as “stimulus-onset alignment, `[-0.5, +1.5)`.”

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Neural data use 20 ms bins and 100 time points per trial. There is no coarser or finer rebinned neural representation beyond assigning each spike to one 20 ms bin.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

```python
bins = np.floor((times - starts[trial]) / BIN_SIZE).astype(np.int64)
```

iii. The AI cites the Zhang cache representation and the task instructions as justification: “stimulus-onset alignment, `[-0.5, 1.5)` s windows, 20 ms spike-count bins.”

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The time input is not read as a standalone raw variable. It is constructed from the stimulus-onset alignment choice, using `stimOn_times` to define each trial window and a fixed relative time grid stored in `REL_SAMPLE_TIMES`.

ii.
```python
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

```python
stim_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
```

iii. The notes justify this as the common task-driven stimulus-aligned grid. They explicitly describe the input as a “fixed event-relative sample grid.”

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI constructs a fixed 100-sample relative-time vector from `-0.48` to `1.50` s using `np.linspace`, then repeats that same vector for every retained trial. It uses the right-edge convention of each 20 ms bin rather than bin centers.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
...
time_input = np.broadcast_to(REL_SAMPLE_TIMES, (len(source_indices), N_BINS))
inputs = np.stack((time_input, block_input), axis=1).astype(np.float32, copy=True)
```

iii. The justification in the notes is explicit: it chose “the 100 bin-right-edge times used by reference behavior interpolation” and described this as the “right-edge time convention.”

## 3-c. How is `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligns the time input to the right edge of each neural spike-count bin. In other words, each time value labels the end of the preceding 20 ms spike-count bin rather than its center.

ii.
```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
...
"behavior_sample_convention": "right edge of each neural time bin",
```

```python
"Window: [-0.5, +1.5) s; 20 ms bins",
"Behavior samples label the preceding spike-count bin end.",
```

iii. The notes justify this as matching the behavior interpolation convention from the Zhang code, and Step 5 explicitly says this makes outputs “causal with the spike bin ending at the same time.”

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the raw `probabilityLeft` column of the trials table. A new block starts whenever `probabilityLeft` changes.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    probability_left = np.asarray(probability_left, dtype=float)
    changes = np.ones(len(probability_left), dtype=bool)
    if len(probability_left) > 1:
        changes[1:] = ~np.isclose(
            probability_left[1:], probability_left[:-1], rtol=0.0, atol=1e-8,
            equal_nan=False,
        )
```

iii. The notes say trial number in block should come from “trial-table `probabilityLeft` runs.”

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI finds block starts by detecting changes in `probabilityLeft`, computes the zero-based count of each trial since the current block start, and crucially does this on the raw trial order before any filtering. It then subsets retained trials and broadcasts each block number across all 100 time bins of the corresponding trial.

ii.
```python
starts = np.maximum.accumulate(np.where(changes, np.arange(len(probability_left)), 0))
return (np.arange(len(probability_left)) - starts).astype(np.float32)
```

```python
raw_block_numbers = trial_number_in_block(trials["probabilityLeft"].to_numpy())
...
block_numbers = raw_block_numbers[source_indices]
...
block_input = np.broadcast_to(block_numbers[:, None], (len(source_indices), N_BINS))
```

iii. The notes justify this as preserving “the animal’s real position in the block” and say it should be “computed before trial filtering.”

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. Choice comes from the trials table’s raw `choice` column.

ii.
```python
raw_choice = trials["choice"].to_numpy()[source_indices]
if not np.all(np.isin(raw_choice, [-1, 1])):
    raise ValueError("Unexpected retained choice value")
```

iii. The notes and trajectory both mention the IBL convention explicitly: raw `+1` means left, `-1` means right, and no-go trials (`0`) are removed by filtering.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After filtering out `choice == 0` trials, the AI maps raw `+1` to class `0` and raw `-1` to class `1`, then repeats that categorical choice across all 100 time bins of a trial.

ii.
```python
# IBL convention: +1 is a leftward choice and -1 is rightward.
# Target convention required here: left=0, right=1.
choices = (raw_choice == -1).astype(np.int8)
```

```python
choice_output = np.broadcast_to(choices[:, None], (len(source_indices), N_BINS))
```

iii. The trajectory records that the AI initially reversed this mapping, then fixed it during “Critical Review 2” because it violated the requested left/right semantics even though decoder accuracy did not reveal the bug.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. Prior probability of left is derived from the trials table’s `probabilityLeft` column.

ii.
```python
raw_prior = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
```

iii. The notes describe this as a direct recoding of the raw task block variable.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI maps raw `probabilityLeft` values `0.2`, `0.5`, and `0.8` to categorical outputs `0`, `1`, and `2`, checks that no other values survive, and broadcasts the result across all 100 bins of the trial.

ii.
```python
priors = np.full(len(raw_prior), -1, dtype=np.int8)
for value, category in ((0.2, 0), (0.5, 1), (0.8, 2)):
    priors[np.isclose(raw_prior, value, rtol=0.0, atol=1e-8)] = category
if np.any(priors < 0):
    raise ValueError(f"Unexpected probabilityLeft values: {np.unique(raw_prior[priors < 0])}")
```

```python
prior_output = np.broadcast_to(priors[:, None], (len(source_indices), N_BINS))
```

iii. The notes justify this as the task-mandated mapping of the block prior values.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. Wheel speed is derived from raw wheel timestamps and positions: `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
timestamp_path = newest_file(alf, "_ibl_wheel.timestamps.npy")
position_path = newest_file(alf, "_ibl_wheel.position.npy")
...
raw_times = np.load(timestamp_path, mmap_mode="r")
raw_position = np.load(position_path, mmap_mode="r")
```

iii. The notes justify this as reproducing `SessionLoader`-equivalent wheel processing from the reference code.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI interpolates raw wheel position to 1 kHz, filters/differentiates it with `velocity_filtered(..., corner_frequency=20, order=8)`, takes absolute velocity as speed, then resamples that speed onto the trial-aligned 100-sample grid.

ii.
```python
position, times = interpolate_position(raw_times, raw_position, freq=1000)
velocity, _ = velocity_filtered(position, fs=1000, corner_frequency=20, order=8)
speed = np.abs(velocity)
sampled, good = sample_behavior(times, speed, stim_times)
```

iii. The notes explicitly say to “reproduce `SessionLoader` wheel processing” and describe “1-kHz wheel interpolation and 20-Hz low-pass filtered velocity magnitude.”

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. After interpolation, the AI computes within-session empirical tertile thresholds over all finite retained wheel-speed samples, imputes any non-finite samples with the session median, and then applies `np.digitize(..., right=False)` to get categories `0`, `1`, `2`.

ii.
```python
def discretize_tertiles(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    finite = np.isfinite(values)
    ...
    thresholds = np.quantile(values[finite], [1 / 3, 2 / 3]).astype(np.float64)
    median = float(np.median(values[finite]))
    clean = np.where(finite, values, median)
    categories = np.digitize(clean, thresholds, right=False).astype(np.int8)
    return categories, thresholds, imputed, median
```

```python
wheel_categories, wheel_thresholds, wheel_imputed, wheel_median = discretize_tertiles(
    wheel_values
)
```

iii. The notes justify this as the task-mandated discretization and argue that session-specific tertiles are the categorical analogue of the reference code’s per-session scaling.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is aligned to the same stimulus-onset-centered trial window as neural data, but sampled at the right edge of each 20 ms neural bin rather than at bin centers.

ii.
```python
sampled, good = sample_behavior(times, speed, stim_times)
```

```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
...
"behavior_sample_convention": "right edge of each neural time bin",
```

iii. The notes call this the “right-edge time convention” and explicitly say wheel samples should “label the preceding spike-count bin end.”

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. Whisker motion energy comes from `<side>Camera.ROIMotionEnergy.npy` together with `_ibl_<side>Camera.times.npy`. The AI prefers the left camera and falls back to the right camera if left is unavailable.

ii.
```python
for view in ("left", "right"):
    value_path = newest_file(alf, f"{view}Camera.ROIMotionEnergy.npy")
    time_path = newest_file(alf, f"_ibl_{view}Camera.times.npy")
    ...
    return sampled, good, aux
```

iii. The notes state: “Prefer left camera, fallback right.”

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI loads the released motion-energy trace as-is, trims timestamps from the front if there are more timestamps than values, samples the trace onto the common trial grid with linear interpolation, and otherwise does no smoothing or normalization before discretization.

ii.
```python
values = np.load(value_path, mmap_mode="r")
times = np.load(time_path, mmap_mode="r")
if len(times) < len(values):
    failures.append(f"{view}: timestamps shorter than data")
    continue
if len(times) > len(values):
    times = times[-len(values):]
sampled, good = sample_behavior(times, values, stim_times)
```

iii. The notes justify the timestamp trimming as matching the staged data quirk they found, and otherwise describe the whisker trace as using “reference linear interpolation” with no additional normalization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same session-specific tertile discretization as wheel speed: thresholds at the 1/3 and 2/3 empirical quantiles of retained finite samples, median imputation for non-finite values, then `np.digitize`.

ii.
```python
motion_categories, motion_thresholds, motion_imputed, motion_median = discretize_tertiles(
    motion_values
)
```

iii. The notes justify this by saying wheel and whisker are continuous task outputs that must be turned into three classes, and session-specific tertiles keep classes populated within each session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Like wheel speed, whisker motion energy is aligned to stimulus onset and sampled at the right edge of each 20 ms neural bin.

ii.
```python
sampled, good = sample_behavior(times, values, stim_times)
```

```python
REL_SAMPLE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS).astype(np.float32)
```

iii. The notes use the same “right-edge time convention” justification here as for wheel speed.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several classes of data issues explicitly. Missing required files cause a session to be skipped. Trials missing required task fields or with bad RT/no-go are filtered out. Trials lacking wheel, whisker, or neural-window coverage are dropped. Sessions with fewer than two retained trials are skipped. If camera timestamps are longer than motion-energy values, extra leading timestamps are trimmed. If continuous outputs contain isolated non-finite values after interpolation, those samples are imputed with the session median before categorization.

ii.
```python
if timestamp_path is None or position_path is None:
    raise FileNotFoundError("Wheel timestamps or position missing")
```

```python
if len(times) > len(values):
    times = times[-len(values):]
```

```python
stream_good = wheel_good & motion_good & neural_good
...
if len(source_indices) < 2:
    raise ValueError(f"Only {len(source_indices)} trials after stream coverage")
```

```python
median = float(np.median(values[finite]))
clean = np.where(finite, values, median)
```

iii. The notes justify these decisions as necessary to keep only scientifically valid trials and to handle “minor issues in the data” without inventing entire missing targets. The trajectory specifically records adding the neural-coverage filter after discovering three post-recording trials with all-zero neural activity.

## 10-a. What are the most time-consuming steps of the code?

i. The AI’s own documentation says the expensive work is dominated by neural processing and file size: loading and binning spikes probe-by-probe, then writing the very large pickle. It treats behavior interpolation as comparatively cheap.

ii.
```python
spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
...
counts = np.bincount(...)
```

```python
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. In Step 6 and Step 7 of `CONVERSION_NOTES.md`, the AI says large spike files motivated memory mapping and chunked binning, and estimates that full “pickle assembly/write” would take several minutes because the dataset is about 100 GB.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining obvious per-trial loops are in `sample_behavior`, which interpolates one trial at a time, and in the inner part of `bin_probe`, which still iterates over trials within a chunk to locate trial-specific spike windows and build encoded bincount indices. The AI partially vectorized around these loops by batching trials into chunks and vectorizing later input/output assembly.

ii.
```python
for trial in range(len(stim_times)):
    lo, hi = int(ibeg[trial]), int(iend[trial])
    ...
    sampled[trial] = _linear_interp_extrapolate(tx, vy, query).astype(np.float32)
```

```python
for first in range(0, n_trials, trials_per_chunk):
    ...
    for local_trial, trial in enumerate(range(first, last)):
        lo = int(np.searchsorted(spike_times, starts[trial], side="left"))
        hi = int(np.searchsorted(spike_times, ends[trial], side="left"))
        ...
```

iii. The notes explicitly say the AI tried to “vectorize target/input construction and block numbering” and to “batch multiple trials into one vectorized `np.bincount`,” implying these remaining loops were left because fully removing them was not necessary for runtime targets.

## 10-c. What processing does the code repeat multiple times?

i. The AI code repeats some work. It loads `spikes.times.npy` once to compute neural coverage bounds and then again inside `bin_probe` to bin spikes. It also repeatedly casts `REL_SAMPLE_TIMES` to `float64` inside every `sample_behavior` trial loop and repeatedly maps spike cluster IDs for each trial chunk.

ii.
```python
for probe in probes:
    spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
    ...
    probe_ends.append(float(spike_times[-1]))
```

```python
def bin_probe(...):
    spike_times = np.load(probe["spikes_times_path"], mmap_mode="r")
    spike_clusters = np.load(probe["spikes_clusters_path"], mmap_mode="r")
```

```python
query = stim_times[trial] + REL_SAMPLE_TIMES.astype(np.float64)
```

iii. The AI did not explicitly call these repetitions out in the notes. The closest justification is Step 6’s claim that it tried to “avoid redundant file loads,” so these remaining repetitions appear to be pragmatic leftovers rather than intentional repeated analyses.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI code computes and stores auditing information that the downstream decoder does not need: optional plot payloads, per-session tertile thresholds and imputation medians, counts of `label >= 1` clusters despite not filtering on them, and detailed session provenance fields. These are useful for validation and documentation but are not used by the decoder itself.

ii.
```python
plot_payload = make_plot_payload(...)
```

```python
"n_good_label_clusters": int(sum(p["n_good"] for p in probes)),
"wheel_tertiles": wheel_thresholds.tolist(),
"whisker_tertiles": motion_thresholds.tolist(),
"wheel_imputed_samples": wheel_imputed,
"whisker_imputed_samples": motion_imputed,
"wheel_imputation_median": wheel_median,
"whisker_imputation_median": motion_median,
```

```python
"session_info": [r["info"] for r in results],
"skipped_sessions": skipped,
```

iii. The notes justify this extra work as part of the required sanity checks, review, and visualization workflow. No separate justification says the decoder needs these values; they are there to support auditability.
