# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI built a composite in-memory ONE index from two staged cache releases, intersected their session ids, optionally restricted to `DATALIMIT_SUBSET.csv`, and then loaded per-session data through `SessionLoader` and `SpikeSortingLoader`. It also patched the trial-table path in the ONE cache to the staged `2025-03-03` revision instead of relying on the unrevisioned `Brainwidemap` row.

ii. 
```python
def build_one(verbose=True) -> tuple[One, list[str]]:
    one = One(cache_dir=CACHE_ROOT)
    one.load_cache(SPIKE_RELEASE)
    behavior_one = One(cache_dir=CACHE_ROOT)
    behavior_one.load_cache(BEHAVIOR_RELEASE)
    ...
    trial_rows = datasets["rel_path"].eq("alf/_ibl_trials.table.pqt")
    datasets.loc[trial_rows, "rel_path"] = (
        f"alf/#{TRIAL_REVISION}#/_ibl_trials.table.pqt"
    )
    ...
    spike_eids = set(map(str, one.search()))
    behavior_eids = set(map(str, behavior_one.search()))
    eids = sorted(spike_eids & behavior_eids)
```

```python
sess = SessionLoader(one=one, eid=eid)
...
loader = SpikeSortingLoader(eid=eid, pname=pname, one=one)
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as a narrowly scoped repair for a staged-cache inconsistency: the frozen BWM table pointed at an unstaged trials path, so it merged the newer behavior release into the ONE index and corrected only the in-memory trial revision while still using ONE and brainbox loaders throughout.

## 1-b. How are the data split into subjects (mice)?

i. The AI took the subject id from ONE session metadata (`one.get_details`) for each retained session. At assembly time it preserved subjects in first-seen session order and built `subject_idx` from that order.

ii. 
```python
details = one.get_details(eid, full=False)
...
info = {
    "eid": eid,
    "subject": str(details["subject"]),
    ...
}
...
subjects.append(info["subject"])
...
subject_names = list(dict.fromkeys(subjects))
subject_lookup = {name: i for i, name in enumerate(subject_names)}
```

iii. The notes say the API already provides subject identity, so no filename parsing was needed; the implementation simply carries that metadata through conversion.

## 1-c. How are the data split into sessions?

i. The AI treated each `eid` as a session. It built a list of eligible session ids from the ONE cache and processed one `eid` at a time.

ii. 
```python
spike_eids = set(map(str, one.search()))
behavior_eids = set(map(str, behavior_one.search()))
eids = sorted(spike_eids & behavior_eids)
...
for eid, converted, error in results:
    ...
    return eid, convert_session(worker_one, eid, do_plot), None
```

iii. Its notes describe the session as the release-organized unit, so there was no further splitting logic beyond selecting eligible `eid`s.

## 1-d. How are the data split into trials?

i. The AI used the trials table loaded by `SessionLoader`; each row was treated as one trial. Trial-level arrays were then formed by indexing the retained row indices.

ii. 
```python
sess = SessionLoader(one=one, eid=eid)
sess.load_trials(revision=TRIAL_REVISION)
...
trials, wheel, motion, whisker_view = load_behavior(one, eid)
...
selected = np.flatnonzero(keep)
for row, trial_idx in enumerate(selected):
    ...
```

iii. The notes treat the trials table as the native trial split, so the AI did not infer trials from timestamps or behavior streams.

## 1-e. How are trials filtered based on quality controls?

i. The AI filtered trials by requiring several finite trial-table columns, reaction time between 0.08 and 2.0 s, feedback duration no longer than 10 s, nonzero choice, valid `probabilityLeft`, complete wheel coverage, complete whisker-motion coverage, and at least two retained trials per session. After spike binning it also dropped trials whose neural matrix was all zeros, treating them as invalid recording gaps.

ii. 
```python
REQUIRED_TRIAL_COLUMNS = (
    "stimOn_times", "choice", "feedback_times", "probabilityLeft",
    "firstMovement_times", "feedbackType", "goCue_times",
)
...
mask &= (rt >= 0.08) & (rt <= 2.0)
mask &= duration <= 10.0
mask &= np.isin(choice, (-1, 1))
mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
```

```python
wheel_interp, wheel_good = interpolate_trials(...)
whisker_interp, whisker_good = interpolate_trials(...)
keep = base_mask & wheel_good & whisker_good
selected = np.flatnonzero(keep)
...
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
if not np.all(neural_valid):
    selected = selected[neural_valid]
```

iii. In the notes, the AI justified these as the intersection of paper/code trial filters plus complete-modality coverage. A later trajectory entry says the zero-neural exclusion was added after auditing a few trials that fell into recording gaps.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The AI derived neural data from per-spike times and cluster ids loaded from spike sorting. It also loaded cluster acronyms and QC labels, but those were used only for metadata and region mapping rather than for constructing the counts themselves.

ii. 
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
...
spike_cluster = np.asarray(spikes["clusters"], dtype=np.int64)
all_times.append(np.asarray(spikes["times"], dtype=np.float64)[valid])
all_clusters.append(spike_cluster[valid] + offset)
all_regions.append(np.asarray(clusters["acronym"], dtype=object).astype(str))
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
```

iii. The notes explicitly frame this as electrophysiology processing from the Zhang cache path: load spike times/clusters for each probe, merge probes, and carry cluster metadata alongside them.

## 2-b. How is the `neural` data processed?

i. The AI merged probes within session, reindexed cluster ids across probes, sorted spikes by time, and then counted spikes into 100 half-open 20 ms bins over `[-0.5, 1.5)` around stimulus onset. It kept unsmoothed spike counts as `float32`; it did not convert them to firing rates.

ii. 
```python
all_clusters.append(spike_cluster[valid] + offset)
...
order = np.argsort(times, kind="stable")
return (
    times[order], cluster_ids[order], np.concatenate(all_regions),
    np.concatenate(all_good), used_probes,
)
```

```python
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
tb = np.searchsorted(edges, st, side="right") - 1
valid = (tb >= 0) & (tb < N_BINS)
flat = sc[valid] * N_BINS + tb[valid]
count = np.bincount(flat, minlength=n_clusters * N_BINS)
trials.append(count.reshape(n_clusters, N_BINS).astype(np.float32))
```

iii. The notes say the AI chose “unsmoothed spike counts in half-open 20-ms bins” because it believed the reference decoder cache consumed counts rather than rates. It also justified probe merging as following the reference session-level cache.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI did not filter the neural array by cluster QC. It retained all Kilosort clusters, merely recording a `good_units` mask where `label >= 1` and storing the count of such units in metadata. It also did not drop `void` clusters before constructing `neural`.

ii. 
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
...
spike_times, spike_clusters, regions, good_units, probes = load_and_merge_spikes(one, eid)
neural = bin_spikes(spike_times, spike_clusters, stim[selected], len(regions))
...
"n_good_units": int(np.sum(good_units)),
...
"neuron_filter": "all Kilosort clusters (reference decoder cache qc=None)",
```

iii. The notes repeatedly justify this as following the published decoder cache path, which the AI interpreted as `qc=None` and therefore “all Kilosort clusters,” while only recording the stricter `label >= 1` count as metadata.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data are aligned to `trials.stimOn_times`. For each retained trial, the AI formed stimulus-relative bin edges from `-0.5` to `+1.5` s and counted spikes within those edges.

ii. 
```python
stim = trials["stimOn_times"].to_numpy(dtype=float)
...
for stim in np.asarray(stim_times, dtype=np.float64):
    edges = stim + edges_rel
    lo = np.searchsorted(times, edges[0], side="left")
    hi = np.searchsorted(times, edges[-1], side="left")
```

iii. The notes say this was the explicit reconciliation between the paper’s variable-specific windows and the task’s required common stimulus-aligned window.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The AI used 20 ms bins and 100 bins per 2 s trial window. It did not apply any temporal rebinning or smoothing beyond the single 20 ms binning step.

ii. 
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

```python
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
```

iii. The notes justify this as matching the common 2 s, 20 ms stimulus-aligned cache convention used by the reference decoder code.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. The AI effectively derived this input from the alignment event `stimOn_times` plus a fixed relative time grid. The stored values themselves are not copied from raw data; they are the chosen bin-end offsets relative to stimulus onset.

ii. 
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
stim = trials["stimOn_times"].to_numpy(dtype=float)
...
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. The notes justify this as part of the fixed common grid required by the decoder format: a continuous “time since stimulus onset” input represented on the same 20 ms trial grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The AI computed this input as the 100 bin-end timestamps for the 20 ms stimulus-aligned window, running from `-0.48` to `1.50` s. It then copied that same vector into every retained trial.

ii. 
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. The trajectory and notes say the AI intentionally sampled continuous behavior at bin ends and used the same bin-end grid for the explicit time input.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. The AI aligned the time input to the neural data by using the same trial window and binning scheme. The neural array is spike counts over half-open 20 ms bins, while the time input is the bin-end timestamp of each corresponding bin.

ii. 
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
...
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. The AI’s stated rationale was that all streams should share one common stimulus-aligned 20 ms grid; in its implementation, the representative time for each neural bin is the bin end.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. The AI derived trial number in block from `trials.probabilityLeft`, treating consecutive runs of equal probability as blocks.

ii. 
```python
def block_trial_numbers(probability_left: np.ndarray) -> np.ndarray:
    p = np.asarray(probability_left, dtype=float)
    out = np.zeros(len(p), dtype=np.int32)
    for i in range(1, len(p)):
        out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
    return out
```

iii. The notes justify this by noting that the trials table does not contain an explicit block id, so blocks must be reconstructed from changes in `probabilityLeft`.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. The AI counted trials within each consecutive run of identical `probabilityLeft`, starting from zero and resetting whenever the probability changed. It computed this on the full chronological trials table before later trial filtering, then broadcast the resulting scalar across all 100 bins of each retained trial.

ii. 
```python
block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())
...
inp = np.vstack([
    BIN_END_TIMES.astype(np.float32),
    np.full(N_BINS, block_num[trial_idx], dtype=np.float32),
])
```

iii. The notes explicitly justify computing the ordinal before trial exclusion so that dropped trials still advance the animal’s true position within the block.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. The AI derived choice from `trials.choice`. It interpreted native `-1` as left and `+1` as right, then remapped those to decoder categories `0` and `1`.

ii. 
```python
choice = trials["choice"].to_numpy()
mask &= np.isin(choice, (-1, 1))
...
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
...
np.full(N_BINS, choice, dtype=np.int8)
```

iii. In `CONVERSION_NOTES.md`, the AI explicitly justified this by claiming the IBL convention is `-1 = left` and `+1 = right`, and therefore mapping `-1 -> 0` and `+1 -> 1`.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The AI dropped trials with `choice == 0`, then converted the remaining `choice` value to a binary category and broadcast that category over all 100 time bins of the trial.

ii. 
```python
mask &= np.isin(choice, (-1, 1))
...
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
...
out = np.vstack([
    np.full(N_BINS, choice, dtype=np.int8),
    ...
])
```

iii. The notes frame this as a straightforward recoding of the native signed choice variable into the requested left/right decoder labels.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. The AI derived prior probability of left from `trials.probabilityLeft`.

ii. 
```python
prior = trials["probabilityLeft"].to_numpy()
mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
...
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
...
p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
```

iii. The notes justify this by saying the task instruction explicitly requests the categorical map `0.2 -> 0`, `0.5 -> 1`, `0.8 -> 2`, so it used the observed block prior directly rather than any inferred latent belief.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. The AI rounded `probabilityLeft`, required it to be one of the three canonical values, mapped it to `0/1/2`, and broadcast the category across all 100 time bins of the retained trial.

ii. 
```python
mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
...
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
...
np.full(N_BINS, prior_map[p], dtype=np.int8)
```

iii. The notes describe this as an exact task-driven categorical recoding, with no extra modeling.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. The AI derived wheel speed from the wheel stream loaded by `SessionLoader`, specifically from `wheel["velocity"]`, which `SessionLoader.load_wheel()` computes from wheel position and timestamps. It then took the absolute value.

ii. 
```python
sess.load_wheel()
...
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
```

iii. The notes justify this as following the reference code path for wheel speed: use the standard IBL wheel loader and take absolute velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. The AI used the smoothed wheel velocity produced by `SessionLoader`, took its absolute value, linearly interpolated it onto each trial’s 100 stimulus-aligned bin-end timestamps, and then discretized the retained aligned values into session-wise tertiles.

ii. 
```python
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
wheel_interp, wheel_good = interpolate_trials(
    wheel["times"].to_numpy(), wheel_speed, stim
)
...
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
```

iii. The notes explicitly justify interpolation at bin ends and per-session tertiles as matching the intended behavior-cache representation while handling session-specific scaling.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The AI thresholded wheel speed into three categories using the 1/3 and 2/3 quantiles of all finite aligned wheel-speed samples from the session’s final retained trials. `np.digitize` then converted each time point to class `0`, `1`, or `2`.

ii. 
```python
def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds
...
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
```

iii. The notes justify this as session-wise tertile discretization, chosen to normalize across labs/cameras and to produce balanced three-class decoder targets.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. The AI aligned wheel speed by interpolating it onto the same stimulus-aligned 20 ms trial grid used for neural binning. In the implementation, wheel speed is sampled at bin-end timestamps corresponding to each neural count bin.

ii. 
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
...
result[i] = np.interp(target, local_t, local_v)
...
wheel_interp, wheel_good = interpolate_trials(
    wheel["times"].to_numpy(), wheel_speed, stim
)
```

iii. The notes and trajectory describe this as the common-grid alignment strategy: everything is represented on the same stimulus-aligned 20 ms bins, with continuous traces sampled at the bin ends.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. The AI derived whisker motion energy from the camera motion-energy stream and its timestamps. It preferred the left camera and fell back to the right camera if the left whisker-motion stream was unavailable.

ii. 
```python
for view in ("left", "right"):
    try:
        sess.load_motion_energy(views=[view])
        key = f"{view}Camera"
        frame = sess.motion_energy[key]
        if "whiskerMotionEnergy" in frame and len(frame) > 1:
            whisker_view = view
            motion = frame
            break
```

iii. The notes justify the left-preferred/right-fallback rule as matching the reference behavior-loading logic and the staged data availability.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The AI used the released `whiskerMotionEnergy` trace without additional filtering or normalization, linearly interpolated it onto each trial’s bin-end timestamps, and discretized the retained samples into session-wise tertiles.

ii. 
```python
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)
...
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. The notes describe this as using the released motion-energy product directly and applying the same common-grid interpolation and tertile discretization used for wheel speed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. The AI thresholded whisker motion energy with the same session-wise tertile rule as wheel speed: the 1/3 and 2/3 quantiles of finite aligned samples from final retained trials, then `np.digitize` to produce class `0`, `1`, or `2`.

ii. 
```python
def discretize_tertiles(values: np.ndarray):
    finite = values[np.isfinite(values)]
    thresholds = np.quantile(finite, [1 / 3, 2 / 3])
    labels = np.digitize(values, thresholds, right=False).astype(np.int8)
    return labels, thresholds
...
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. The notes justify the per-session tertiles as a way to absorb session-specific scale differences while satisfying the required 3-class decoder output.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. The AI aligned whisker motion energy by interpolating it onto the same stimulus-aligned 20 ms grid as the neural data. As with wheel speed, the sampled time points are the bin ends of the neural count bins.

ii. 
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
...
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)
```

iii. The notes present this as the same common-grid alignment choice used for all time-varying behavioral streams.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI mostly handled missing or malformed data by exclusion rather than imputation. It raised errors for missing essential trial columns or absent behavior streams, marked trials invalid when wheel/whisker coverage was incomplete or timestamps were unsorted, skipped sessions with fewer than two valid trials, and later excluded fully zero neural trials as recording gaps.

ii. 
```python
missing = [col for col in REQUIRED_TRIAL_COLUMNS if col not in trials]
if missing:
    raise RuntimeError(f"missing trial columns: {missing}")
...
if whisker_view is None:
    raise RuntimeError("no left or right whisker motion-energy stream")
...
if len(selected) < 2:
    raise RuntimeError(f"only {len(selected)} valid trials")
...
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
```

iii. The notes and trajectory justify this as preserving only fully supported session/trial windows. A trajectory audit specifically says missing whisker streams “cannot be imputed” without changing the requested output, and zero-neural trials were treated as invalid acquisition gaps rather than silence.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified session behavior loading/interpolation, spike loading, spike binning, and especially dense neural serialization as the main runtime and memory costs. In the implementation, full conversion parallelizes session processing to hide some of the I/O and NumPy cost.

ii. 
```python
sess.load_wheel()
...
spikes, clusters, channels = loader.load_spike_sorting()
...
count = np.bincount(flat, minlength=n_clusters * N_BINS)
...
executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
```

iii. `CONVERSION_NOTES.md` says wheel smoothing is expensive, spike I/O is heavy, and dense target-format neural matrices dominate memory and output size.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The clearest remaining per-trial loops are the interpolation loop in `interpolate_trials`, the spike-binning loop in `bin_spikes`, and the simple sequential loop used to compute block ordinals. These are the main places where the AI traded some vectorization for straightforward session-by-session logic.

ii. 
```python
for i, target in enumerate(targets):
    ...
    result[i] = np.interp(target, local_t, local_v)
```

```python
for stim in np.asarray(stim_times, dtype=np.float64):
    ...
    count = np.bincount(flat, minlength=n_clusters * N_BINS)
```

```python
for i in range(1, len(p)):
    out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
```

iii. The notes mention vectorized binning as a deliberate optimization, but they do not claim these loops were fully eliminated; the remaining loops reflect a readability/performance tradeoff.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats a few setup and transformation steps: it rebuilds the composite ONE object separately inside worker threads, it runs the same interpolation/discretization pipeline twice per session (once for wheel and once for whisker), and it maps regions to Beryl only after first carrying raw acronyms through session processing.

ii. 
```python
if not hasattr(worker_state, "one"):
    worker_state.one, _ = build_one(verbose=False)
```

```python
wheel_interp, wheel_good = interpolate_trials(...)
whisker_interp, whisker_good = interpolate_trials(...)
...
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. The notes do not explicitly justify all of this repetition. Where they do comment, the justification is pragmatic: ONE state is not thread-safe, so each worker needs its own instance; the duplicated wheel/whisker path keeps the two behavior streams symmetric.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code computes some quantities that do not affect the final decoder tensors: it builds a `good_units` QC mask without using it to filter `neural`, stores per-session `lab`, `date`, and tertile thresholds only as metadata, and includes optional plotting machinery used for diagnostics rather than downstream analysis.

ii. 
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
...
"lab": str(details["lab"]),
"date": str(details["date"]),
"n_good_units": int(np.sum(good_units)),
"wheel_tertiles": wheel_q.tolist(),
"whisker_tertiles": whisker_q.tolist(),
```

```python
def plot_processing(...):
    ...
    fig.savefig(f"/app/processing_{eid}.png", dpi=140)
```

iii. The notes justify these mostly as audit/debug support: the QC counts, thresholds, and plots were kept to validate the conversion, not because the decoder consumes them.
