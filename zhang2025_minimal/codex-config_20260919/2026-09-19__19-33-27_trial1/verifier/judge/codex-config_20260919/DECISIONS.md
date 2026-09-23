# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the BWM release CSV, optionally restricts it using `DATALIMIT_SUBSET.csv`, groups its rows into session jobs, and uses a local-cache ONE client. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads every listed probe. Sessions are processed in parallel into restartable pickle shards and then assembled.

ii.
```python
release = pd.read_csv(release_file, index_col=0)
release = release[release["eid"].astype(str).isin(subset)]
for eid, rows in release.groupby("eid", sort=False):
    jobs.append({"eid": str(eid), "pids": rows["pid"].astype(str).tolist(), ...})
```
```python
loader = SessionLoader(one=one, eid=eid)
loader.load_trials(); loader.load_wheel()
spikes, clusters, channels = spike_loader.load_spike_sorting()
```

iii. The trajectory says the full release, rather than a sample, was required and describes ONE as the supplied cache interface. Shards and multiprocessing were chosen because the source and converted datasets are very large.

## 1-b. How are the data split into subjects?

i. Subject names come directly from the release CSV. At assembly, unique names are sorted and each session receives an integer `subject_idx`.

ii.
```python
"subject": str(first["subject"]),
subjects = sorted({session["subject"] for session in sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
```

iii. The agent relied on the release metadata's existing unique subject identifier; the trajectory reports 136 subjects after conversion.

## 1-c. How are the data split into sessions?

i. Rows of the release manifest are grouped by `eid`; all probe rows with the same `eid` form one session job and are merged into one session population.

ii.
```python
for eid, rows in release.groupby("eid", sort=False):
    jobs.append({"eid": str(eid), "pids": rows["pid"].astype(str).tolist(), ...})
```

iii. The agent treated the BWM `eid` as the native session boundary and merged probes because they share behavior.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one row per trial. Retained row indices (`candidate`) select trial onsets and labels, and neural/behavior arrays are sliced or interpolated separately for each retained onset.

ii.
```python
loader.load_trials()
candidate = np.flatnonzero(mask)
onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
neural = [neural_array[i] for i in range(len(candidate))]
```

iii. The trajectory follows the supplied reference utilities and uses a common two-second stimulus-aligned trial window for all requested variables.

## 1-e. How are trials filtered based on quality controls?

i. Required trial fields must all be finite; reaction time (`firstMovement_times - stimOn_times`) must be 0.08–2 s; go-cue-to-feedback duration must be at most 10 s; choice must be nonzero; prior must be 0.2, 0.5, or 0.8; and wheel and motion-energy streams must cover the full window. Sessions with fewer than two surviving trials are rejected.

ii.
```python
mask = (finite & (reaction_time >= 0.08) & (reaction_time <= 2.0)
        & (trial_length <= 10.0) & (choice_raw != 0)
        & np.isin(probability, [0.2, 0.5, 0.8]))
behavior_good = wheel_good & motion_good
candidate = candidate[behavior_good]
```

iii. The agent calls these the BWM trial exclusions and says missing behavior explains nearly all excluded sessions. Its finite-all-fields and 10 s duration requirements are stricter than the human conversion.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from each probe's `spikes["times"]` and `spikes["clusters"]`. The merged cluster table determines cluster count and anatomical regions, but not a quality filter.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
local_ids = np.asarray(spikes["clusters"], dtype=np.int64)
spike_parts.append((np.asarray(spikes["times"]), local_ids, offset))
```

iii. The agent reasoned that the methods repository calls `prepare_data(qc=None)`, so it deliberately retained every Kilosort cluster.

## 2-b. How is the `neural` data processed?

i. Probes are merged by offsetting probe-local cluster IDs. Spikes are counted, without smoothing, into 100 nonoverlapping 20 ms bins from -0.5 to 1.5 s. The saved values are raw counts (`float32`), not rates in Hz.

ii.
```python
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
clu = clusters[lo:hi] + offset
counts = np.bincount(flat, minlength=size)
return counts.reshape(len(onsets), n_neurons, N_BINS).astype(np.float32)
```

iii. The docstring says the reference bins spike counts and does not smooth. Unlike the human solution, the code never divides counts by 0.02 to obtain firing rates.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not QC-filtered: all sorted clusters, including low-label and `void` units, are retained. Only malformed cluster IDs or a session with zero clusters causes failure.

ii.
```python
# Load and retain every Kilosort cluster, exactly as prepare_data(qc=None)
n_clusters = len(merged)
region_parts.append(mapped)
offset += n_clusters
```

iii. The agent explicitly chose `qc=None` to match the methods code. This conflicts with the human decision to require cluster label 1 and exclude Beryl `void` units.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows are defined relative to `stimOn_times`, from 0.5 s before through 1.5 s after onset. Absolute spike times are searched within each window and binned relative to its start.

ii.
```python
starts = onsets + OFF_START
ends = onsets + OFF_END
lo = np.searchsorted(times, begin, side="left")
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
```

iii. The trajectory states that one shared stimulus-aligned window is needed so all requested targets coexist on identical trials.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Resolution is 20 ms, with 100 bins over two seconds. Spike events are binned once; there is no later neural resampling or smoothing.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. The agent selected the 20 ms resolution and two-second window used by the supplied methods code and paper.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is derived from the configured offsets and bin size, with `stimOn_times` providing each trial's zero point. The stored relative grid is -0.48 through 1.50 s (right bin edges).

ii.
```python
RELATIVE_TIMES = np.linspace(OFF_START + BIN_SIZE, OFF_END, N_BINS)
candidate_onsets = trials["stimOn_times"].to_numpy(dtype=float)[candidate]
```

iii. The agent believed the reference behavior helper associates values with bin right edges.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. A fixed 100-element float32 grid is created once and copied into every trial; no raw timestamp subtraction is saved explicitly.

ii.
```python
RELATIVE_TIMES = np.linspace(-0.5 + 0.020, 1.5, 100).astype(np.float32)
inp[0] = RELATIVE_TIMES
```

iii. The grid was intended to match the reference behavior sampling convention.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. It has 100 entries corresponding by index to the 100 neural bins, but denotes each bin's right edge, whereas the human solution denotes neural-bin centers.

ii.
```python
bins = np.floor((times[lo:hi] - begin) / BIN_SIZE).astype(np.int64)
inp[0] = RELATIVE_TIMES
```

iii. The agent asserted this was “exactly” the grid used by the reference behavior function, prioritizing behavioral right-edge sampling over the human center convention.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from consecutive values of trial-table `probabilityLeft`; a changed or nonfinite value begins a new block.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
trial_in_block = _trial_number_in_block(probability)
```

iii. The agent inferred blocks from prior changes because there is no separate block identifier.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A loop assigns a one-based position: 1 on the first trial of a block, then increments. It runs before trial filtering, and the scalar is broadcast across all 100 time bins.

ii.
```python
if i == 0 or not np.isfinite(value) or value != previous:
    number = 1
else:
    number += 1
...
inp[1] = trial_in_block[trial_index]
```

iii. The agent intentionally computes before exclusions so rejected trials still advance the animal's true block position, but chose one-based rather than the human's zero-based count.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from trial-table `choice`; zero/no-response trials are excluded. The code maps raw +1 to output 1 and everything retained (-1) to 0.

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)
mask = ... & (choice_raw != 0)
choice = (choice_raw[candidate] == 1).astype(np.int8)
```

iii. The inline comment claims “left=-1, right=1,” which is opposite the IBL convention used by the human reference; consequently the requested left=0/right=1 coding is reversed.

## 5-b. What processing is involved in computing `output` *Choice*?

i. Choice is binarized, then its per-trial scalar is broadcast across 100 time points.

ii.
```python
choice = (choice_raw[candidate] == 1).astype(np.int8)
out[0] = choice[row]
```

iii. The agent treated choice as a categorical per-trial output represented as a constant time series for compatibility with the decoder format.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials["probabilityLeft"]` after restricting values to 0.2, 0.5, and 0.8.

ii.
```python
probability = trials["probabilityLeft"].to_numpy(dtype=float)
np.isin(probability, [0.2, 0.5, 0.8])
```

iii. These are the three task block priors specified in the instructions.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. A lookup maps 0.2→0, 0.5→1, and 0.8→2; the per-trial value is broadcast across time.

ii.
```python
prior_lookup = {0.2: 0, 0.5: 1, 0.8: 2}
prior = np.array([prior_lookup[float(x)] for x in probability[candidate]], dtype=np.int8)
out[1] = prior[row]
```

iii. This is the exact mapping required by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It uses `SessionLoader`'s wheel `times` and `velocity`; speed is the absolute velocity.

ii.
```python
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy()
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy())
```

iii. The agent followed the reference utility's use of absolute wheel velocity.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. `SessionLoader` supplies processed velocity. Timestamps are sorted/deduplicated, the absolute velocity is linearly interpolated at each trial's 100 right-edge times, and nonfinite/incompletely covered trials are rejected.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
row = np.interp(grid, segment_t, segment_v)
```

iii. The agent states linear interpolation and coverage checks reproduce the reference helper, although the human solution samples bin centers.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. Two quantiles, 1/3 and 2/3, are computed across every retained wheel trial-time sample in a session. `np.digitize` maps values to low=0, medium=1, high=2.

ii.
```python
cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
labels = np.digitize(values, cuts, right=False).astype(np.int8)
```

iii. Session-wise tertiles were selected to turn the continuous target into three approximately equal-sized decoder classes.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel is interpolated relative to the same `stimOn_times` and has 100 samples matched by index to neural bins, but at right edges (-0.48…1.50 s) rather than human-reference centers (-0.49…1.49 s).

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
wheel, wheel_good = _interpolate_trials(wheel_times, wheel_speed, candidate_onsets)
```

iii. The agent viewed the reference behavioral right-edge grid as the correct alignment convention.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses camera `times` and `whiskerMotionEnergy`, preferring `leftCamera` and falling back to `rightCamera`.

ii.
```python
for view in ("left", "right"):
    session_loader.load_motion_energy(views=[view])
    values = frame["whiskerMotionEnergy"].to_numpy()
```

iii. The fallback was described as part of the reference pipeline and prevents loss of sessions lacking the preferred side camera.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released values receive no filtering or normalization. They are sorted/deduplicated, interpolated at the 100 trial-relative right edges, checked for complete finite coverage, then discretized session-wise.

ii.
```python
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
motion_cat, motion_cuts = _session_tertiles(motion)
```

iii. The agent followed the supplied behavior interpolation logic and applied only the discretization required for categorical decoder outputs.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the session-wide 1/3 and 2/3 quantiles over all retained trial-time values, producing labels 0, 1, and 2.

ii.
```python
cuts = np.quantile(values, [1.0 / 3.0, 2.0 / 3.0])
labels = np.digitize(values, cuts, right=False).astype(np.int8)
```

iii. The same tertile rule as wheel speed was chosen to make balanced categorical classes.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. It is interpolated using the same stimulus onset and 100-index grid, but the samples are neural-bin right edges rather than centers.

ii.
```python
grid = onset + RELATIVE_TIMES.astype(np.float64)
motion, motion_good = _interpolate_trials(motion_times, motion_values, candidate_onsets)
```

iii. The agent believed the supplied reference behavior utility defined values at those right edges.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing required columns, malformed streams/cluster IDs, no units, or fewer than two trials cause a session to fail. Nonfinite required trial fields and behavior windows with incomplete/nonfinite coverage are dropped. Duplicate behavior timestamps retain the last value. Camera side falls back left-to-right. Failed sessions are omitted because only existing shards are assembled.

ii.
```python
if missing: raise ValueError(...)
_, reverse_idx = np.unique(times[::-1], return_index=True)
behavior_good = wheel_good & motion_good
if len(candidate) < 2: raise ValueError(...)
if not shard.exists(): continue
```

iii. The trajectory reports that 15 sessions lacking usable motion energy or complete aligned trials were excluded; restartable shards make failures recoverable.

## 10-a. What are the most time-consuming steps of the code?

i. Loading very large spike-sorting arrays, binning spikes for every trial/probe, processing 459 sessions, and assembling/writing the 99 GB final pickle dominate. Verification must also stream that full pickle.

ii.
```python
spikes, clusters, channels = spike_loader.load_spike_sorting()
neural_array = _bin_spikes(spike_parts, onsets, offset)
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory identifies a roughly 570 GB source release and uses up to eight workers plus session shards to make the full conversion tractable.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The trial-number loop, per-probe/per-trial spike-window loop, per-trial behavior interpolation loop, per-trial input/output construction loop, and region/prior list comprehensions could be vectorized. Spike binning already vectorizes within each trial using flat indices and `bincount`.

ii.
```python
for i, value in enumerate(probability_left): ...
for trial, (begin, end) in enumerate(zip(starts, ends)): ...
for i, onset in enumerate(onsets): ...
for row, trial_index in enumerate(candidate): ...
```

iii. The agent favored clear trial slices and parallelized across sessions; the trajectory does not provide a separate explicit justification for each remaining loop.

## 10-c. What processing does the code repeat multiple times?

i. Each worker constructs and loads its own ONE client/cache tables and `BrainRegions`; every trial receives fresh input/output arrays containing repeated time grids and repeated scalar choice/prior/block values; existing shards are reopened during assembly.

ii.
```python
one = _one(job["cache_dir"])
brain_regions = BrainRegions()
inp = np.empty((2, N_BINS), dtype=np.float32)
out = np.empty((4, N_BINS), dtype=np.int8)
```

iii. Per-worker setup avoids unsafe shared state, while repeated time-series broadcasting satisfies the target format. Shards support restarts and bounded conversion memory.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads/merges cluster and channel metadata beyond the acronyms/counts ultimately retained, computes and stores shard metadata/quantile edges used only as metadata, sorts streams even when already ordered, and writes then rereads huge shards before optionally deleting them. It also loads `feedbackType` solely for a finite-data mask.

ii.
```python
merged = SpikeSortingLoader.merge_clusters(spikes, clusters, channels).to_df()
order = np.argsort(times, kind="stable")
"wheel_speed_tertile_edges": wheel_cuts,
with shard.open("rb") as stream: payload = pickle.load(stream)
```

iii. Most overhead supports robustness, provenance, and restartability rather than decoder computations. The agent explicitly chose shards because the full dataset is too large to keep all session processing state in memory.
