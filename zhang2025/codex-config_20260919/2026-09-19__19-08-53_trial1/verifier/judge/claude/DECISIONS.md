# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI reads data directly from disk using the frozen release CSV (`bwm_release.csv`) rather than using the ONE API. It constructs `SessionSpec` dataclasses that resolve file paths under the `one_cache` directory hierarchy. For each session in the release CSV, it builds paths to trial tables, wheel data, and per-probe spike files. It validates the inventory against expected counts (139 subjects, 459 sessions, 699 probes, 621,733 clusters).

ii.
```python
RELEASE_CSV = APP / "code" / "code_zhang2025" / "data" / "bwm_release.csv"

def build_release_specs() -> tuple[list[SessionSpec], dict[str, int]]:
    release = pd.read_csv(RELEASE_CSV, dtype={"date": str, "subject": str, "lab": str})
    assert release["pid"].nunique() == EXPECTED_RELEASE["probes"]
    assert release["eid"].nunique() == EXPECTED_RELEASE["sessions"]
    assert release["subject"].nunique() == EXPECTED_RELEASE["subjects"]
    ...
    for eid, rows in release.groupby("eid", sort=False):
        first = rows.iloc[0]
        session_dir = (DATA_ROOT / first["lab"] / "Subjects" / first["subject"] /
                       first["date"] / f"{number:03d}")
        ...
```

iii. The AI chose to bypass the ONE API and read files directly from disk paths, using the release CSV as the authoritative source. The justification is that direct file access avoids network/API ambiguity and ensures reproducibility against the exact frozen release.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the `subject` column in the release CSV. After processing, unique subjects are sorted and assigned integer indices.

ii.
```python
subjects = sorted({s["spec"].subject for s in sessions})
subject_lookup = {s: i for i, s in enumerate(subjects)}
```

iii. The subject identity comes directly from the frozen release CSV, requiring no path parsing or API lookups.

## 1-c. How are the data split into sessions?

i. Sessions are identified by unique `eid` values in the release CSV. Each `eid` groups one or more probes into a single session. The `groupby("eid")` operation produces one `SessionSpec` per session.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    ...
    specs.append(SessionSpec(eid=str(eid), subject=str(first["subject"]), ...))
```

iii. Sessions are the natural unit of the release; the CSV groups probes by eid.

## 1-d. How are the data split into trials?

i. Trials come from the per-session Parquet trial table (`_ibl_trials.table.pqt`), which has one row per trial.

ii.
```python
trials = pd.read_parquet(spec.trial_table)
```

iii. No splitting decision needed; the trial table is already one row per trial.

## 1-e. How are trials filtered based on quality controls?

i. Five criteria are applied: (1) required fields must be non-NaN (stimOn_times, choice, feedback_times, probabilityLeft, firstMovement_times, feedbackType); (2) reaction time between 0.08 and 2.0 s; (3) trial duration (feedback_times - goCue_times) must not exceed 10 s; (4) choice must not be 0 (no-response); (5) both wheel and whisker motion energy must have sufficient coverage of the trial window. Sessions with fewer than 2 valid trials are excluded.

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

Then stream coverage is applied:
```python
wheel_good, _, wheel_ie = coverage_mask(wheel_times, begins, ends)
motion_good, _, motion_ie = coverage_mask(motion_times, begins, ends)
stream_good = wheel_good & motion_good
```

iii. The AI closely follows the reference code's `load_trials_and_mask` function, including the duration > 10s check. Stream coverage is added as a necessary requirement for the decoder outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Two arrays per probe: `spikes.times.npy` (spike timestamps) and `spikes.clusters.npy` (cluster assignments). Additionally `clusters.channels.npy` and `channels.brainLocationIds_ccf_2017.npy` for region mapping.

ii.
```python
spike_times = np.load(probe.spikes_times, mmap_mode="r")
spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")
```

iii. Spike times and cluster assignments are the standard inputs for binned spike count computation.

## 2-b. How is the `neural` data processed?

i. Spikes are counted into 100 half-open 20 ms bins spanning [-0.5, 1.5) s relative to stimulus onset. The counts are stored as float32 (spike counts, NOT firing rates). Multiple probes in a session are concatenated along the neuron dimension.

ii.
```python
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
keep = (bins >= 0) & (bins < N_BINS)
flat = clusters[keep] * N_BINS + bins[keep]
counts = np.bincount(flat, minlength=n_clusters * N_BINS).reshape(n_clusters, N_BINS)
neural[j][offset:offset + n_clusters] = counts
```

iii. The AI uses the same binning approach as the reference code. Notably, the AI stores raw spike counts (not divided by bin width to get firing rates), which differs from the human reference that converts to Hz.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron quality filtering is applied. ALL sorted clusters from the frozen release are retained, regardless of their quality label. The AI explicitly decided against applying the `label >= 1` filter that the data paper uses for well-isolated neurons.

ii.
```python
# In bin_spikes_for_session - no quality filtering:
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
neural = [np.zeros((n_neurons, N_BINS), dtype=np.float32) for _ in range(n_trials)]
# All clusters are used directly, no filtering by label
```

iii. The AI argues that the methods paper cache code calls `load_spiking_data(qc=None)` and the paper says it uses "all neurons." The AI chose to match the decoder code's behavior rather than the data paper's analysis-level filtering.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Each trial's spikes are aligned to stimulus onset by subtracting `stimOn_times` from spike timestamps. The window spans [-0.5, 1.5) s around stimulus onset.

ii.
```python
stim = behavior["stim_times"]
begins, ends = stim + OFF_START, stim + OFF_END
...
left = np.searchsorted(spike_times, begins, side="left")
right = np.searchsorted(spike_times, ends, side="left")
...
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
```

iii. The session clock is shared across all data streams; aligning to stimulus onset is a simple subtraction.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 20 ms bins, 100 bins total over the 2 s window. No rebinning is applied; spikes are counted directly into these bins from raw spike times.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. Matches the reference code's `binsize=0.02` and the method paper's 100 time steps.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. Derived from the bin timing constants and `stimOn_times`. The time input is the right edge of each 20 ms bin, running from -0.48 to 1.50 s.

ii.
```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

iii. The AI chose bin right edges rather than bin centers, arguing this matches the reference behavior code's convention for labeling count bins.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. No processing — the values are defined by the bin grid parameters. They are the same for every trial.

ii.
```python
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
...
inputs.append(np.vstack((RELATIVE_BIN_ENDS,
                         np.full(N_BINS, s["block_trial"][j]))).astype(np.float32))
```

iii. The time input is a deterministic function of the bin grid.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The time input values are the right edges of the same bins used for spike counting, so they are aligned by construction.

ii.
```python
# Neural bins:
bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
# Time input:
RELATIVE_BIN_ENDS = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS, dtype=np.float64)
```

iii. Both use the same 100-bin temporal grid.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. From `probabilityLeft` in the trials table. A block boundary is detected where probabilityLeft changes value.

ii.
```python
def trial_number_in_block(probability_left: np.ndarray) -> np.ndarray:
    n = probability_left.size
    starts = np.r_[0, np.flatnonzero(probability_left[1:] != probability_left[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    out = np.empty(n, dtype=np.int32)
    for start, end in zip(starts, ends):
        out[start:end] = np.arange(end - start, dtype=np.int32)
    return out
```

iii. The trials table carries no explicit block identifier, so blocks are recovered from changes in probabilityLeft.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The trial number is the zero-based position within each contiguous run of the same probabilityLeft value. It is computed on the full unfiltered trial sequence before trial filtering, so filtered-out trials still advance the count.

ii.
```python
probs_all = trials["probabilityLeft"].to_numpy(dtype=np.float64)
block_no_all = trial_number_in_block(probs_all)
...
"block_trial": block_no_all[raw_idx].astype(np.float32),
```

iii. Computing on the unfiltered sequence preserves the animal's true position in the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. From `trials.choice`, which takes values +1 (left in IBL convention), -1 (right), or 0 (no response). No-response trials are excluded by the trial mask.

ii.
```python
choices = trials["choice"].to_numpy(dtype=np.float64)[raw_idx]
...
"choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
```

iii. The AI states in comments and metadata that -1 is left and +1 is right, but this is the **opposite** of the IBL convention (+1 = left, -1 = right).

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI maps choice using `(choices == 1).astype(np.int8)`. With IBL convention (+1 = left, -1 = right), this produces: left (+1) → 1, right (-1) → 0. However, the instructions specify left = 0, right = 1. The mapping is **reversed**.

ii.
```python
"choice": (choices == 1).astype(np.int8),  # -1 left -> 0; +1 right -> 1
```

The output values are declared as:
```python
["left", "right"],  # index 0 = left, index 1 = right
```

iii. The AI's comment claims "-1 left -> 0; +1 right -> 1" but in IBL, +1 is left and -1 is right. The mapping is reversed relative to the instructions' requirement of "left = 0, right = 1."

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. From `trials.probabilityLeft`, which takes values 0.2, 0.5, or 0.8.

ii.
```python
priors = probs_all[raw_idx]
...
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
```

iii. The prior is directly available in the trials table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are mapped using `np.searchsorted([0.2, 0.5, 0.8], prior)`: 0.2 → 0, 0.5 → 1, 0.8 → 2. This matches the instructions.

ii.
```python
"prior": np.searchsorted(np.array([0.2, 0.5, 0.8]), priors).astype(np.int8),
```

iii. Direct mapping as specified in the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. From `_ibl_wheel.position.npy` and `_ibl_wheel.timestamps.npy`.

ii.
```python
wheel_raw_times = np.asarray(np.load(spec.wheel_times, mmap_mode="r"), dtype=np.float64)
wheel_raw_position = np.asarray(np.load(spec.wheel_position, mmap_mode="r"), dtype=np.float64)
```

iii. These are the standard wheel data files in the IBL release.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. (1) Position is interpolated to a uniform 1 kHz grid using the reference brainbox `interpolate_position` function. (2) Velocity is computed with an order-8, 20 Hz Butterworth low-pass filter using `velocity_filtered`. (3) Speed is the absolute value of velocity. (4) The speed trace is linearly interpolated onto the 100 bin right-edge time points for each trial, with extrapolation for the last bin to match the reference code's `fill_value='extrapolate'` behavior. (5) Discretization into 3 classes using global tertiles.

ii.
```python
wheel_position, wheel_times = interpolate_position(wheel_raw_times, wheel_raw_position, freq=1000)
wheel_velocity, _ = velocity_filtered(wheel_position, fs=1000, corner_frequency=20, order=8)
wheel_speed = np.abs(wheel_velocity)
...
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. The AI imports and uses the actual reference brainbox wheel functions, ensuring identical preprocessing.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Global tertiles: the 1/3 and 2/3 quantiles are computed over ALL retained aligned wheel speed samples across ALL sessions. `np.searchsorted` then assigns each value to class 0, 1, or 2.

ii.
```python
wheel_values = np.concatenate([s["wheel"].ravel() for s in behavior_sessions])
wheel_q = np.quantile(wheel_values, [1 / 3, 2 / 3])
...
np.searchsorted(qwheel, s["wheel"][j], side="right").astype(np.int8),
```

iii. The AI argues global thresholds keep class definitions physically consistent across sessions and approximately balance classes overall. The human reference uses per-session percentiles instead.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The wheel speed is interpolated onto the same 100 time points (bin right edges) used for the neural bins, measured from stimulus onset, so they share a temporal grid.

ii.
```python
targets = stim[:, None] + RELATIVE_BIN_ENDS[None, :]
wheel = interval_interpolate(wheel_times, wheel_speed, targets, ends, wheel_ie)
```

iii. Both neural and behavioral data are on the same stimulus-aligned temporal grid.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. From `{left,right}Camera.ROIMotionEnergy.npy` with matching `_ibl_{left,right}Camera.times.npy`. Left camera is preferred, right is fallback.

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

iii. Left-first/right-fallback matches the reference code's `load_target_behavior`.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion energy trace is used as-is (no filtering or normalization). It is linearly interpolated onto the 100 bin right-edge time points, with last-bin extrapolation. Then discretized into 3 classes using global tertiles.

ii.
```python
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
...
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8),
```

iii. No additional processing beyond interpolation and discretization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. Same as wheel speed: global 1/3 and 2/3 quantiles over all retained aligned whisker motion energy samples across all sessions.

ii.
```python
whisker_values = np.concatenate([s["whisker"].ravel() for s in behavior_sessions])
whisker_q = np.quantile(whisker_values, [1 / 3, 2 / 3])
...
np.searchsorted(qwhisker, s["whisker"][j], side="right").astype(np.int8),
```

iii. Global tertiles for consistency across sessions.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Identical approach to wheel speed: interpolated onto the same 100 bin right-edge time points as the neural data.

ii.
```python
whisker = interval_interpolate(motion_times, motion_values, targets, ends, motion_ie)
```

iii. Shares the same stimulus-aligned temporal grid as neural and wheel data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions missing required whisker motion energy streams are excluded entirely. Sessions with fewer than 2 valid trials after filtering are excluded. Trials without full wheel/whisker coverage are dropped individually. Non-monotonic or non-finite behavioral streams cause session exclusion. Behavior processing errors are caught and the session is excluded with a logged reason.

ii.
```python
if behavior is None:
    excluded.append({"eid": spec.eid, "reason": str(reason)})
    print(f"Excluded {spec.eid}: {reason}", flush=True)
    continue
```

```python
if raw_idx.size < 2:
    return None, "fewer than two trials with complete wheel/whisker coverage"
```

iii. Missing data is handled by exclusion at either the session or trial level, with all exclusions logged.

## 10-a. What are the most time-consuming steps of the code?

i. Two main bottlenecks: (1) Building the frozen release inventory (14s for assertions and file discovery); (2) Spike binning across all sessions, which dominates total wall time. The AI reports total conversion time of ~455 s.

ii.
```python
# Inventory building
specs, inventory = build_release_specs()
# Spike binning (parallelized)
with ThreadPoolExecutor(max_workers=workers) as pool:
    sessions = list(pool.map(bin_spikes_for_session, behavior_sessions))
```

iii. The spike binning is inherently I/O and compute intensive due to the large volume of spike data (21+ billion events).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Two per-trial loops: (1) the spike binning loop iterates over trials within `bin_spikes_for_session`; (2) the behavior interpolation loop in `trial_traces` / `interval_interpolate` is vectorized with `np.interp` over a reshaped array but still loops implicitly. The trial-number-in-block computation also uses a Python loop over blocks.

ii.
```python
for j, (i0, i1, begin) in enumerate(zip(left, right, begins)):
    ...
    bins = np.floor((np.asarray(spike_times[i0:i1]) - begin) / BIN_SIZE).astype(np.int64)
    ...
```

```python
for start, end in zip(starts, ends):
    out[start:end] = np.arange(end - start, dtype=np.int32)
```

iii. The per-trial spike binning loop could theoretically be vectorized using offset indexing, but the AI notes that the per-trial slicing is clear and the cost is dominated by I/O rather than Python overhead.

## 10-c. What processing does the code repeat multiple times?

i. The AI loads spike files twice for the first probe of each session: once during the main binning and once for the spot-check assertion at the end of `bin_spikes_for_session`. It also loads `clusters.channels.npy` twice per probe: once in `build_release_specs` for inventory counting and once in `bin_spikes_for_session` for region mapping.

ii.
```python
# First load during binning:
spike_times = np.load(probe.spikes_times, mmap_mode="r")
spike_clusters = np.load(probe.spikes_clusters, mmap_mode="r")

# Second load for spot check:
st = np.load(p0.spikes_times, mmap_mode="r")
sc = np.load(p0.spikes_clusters, mmap_mode="r")
```

iii. The repeated loads are for validation/assertion purposes and use memory mapping, so the actual I/O cost is minimal.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. By retaining ALL sorted clusters (621,733) rather than filtering to well-isolated neurons (75,708), the AI processes and stores approximately 8x more neural data than the human reference. The `void` region units (channels outside the brain) are also retained. Additionally, continuous wheel and whisker values are computed and stored in the intermediate `behavior` dict before being discretized in the final assembly step.

ii.
```python
# All clusters retained, no quality filtering
n_per_probe = [int(np.load(p.clusters_channels, mmap_mode="r").shape[0]) for p in spec.probes]
n_neurons = int(sum(n_per_probe))
```

iii. The AI argues this matches the methods paper decoder code. However, it results in much larger output files (~99 GiB vs a smaller file for filtered data) and includes noise from low-quality and out-of-brain units.
