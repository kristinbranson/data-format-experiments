# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the frozen `bwm_release.csv`, optionally restricts its EIDs with `DATALIMIT_SUBSET.csv`, creates an Alyx-backed `ONE` client pointed at the local cache, and processes each unique EID. `SessionLoader` loads trials, wheel, and camera motion energy; `SpikeSortingLoader` loads every insertion listed for that EID in the release table. Sessions that raise any exception are recorded and skipped.

ii.
```python
release = pd.read_csv(RELEASE_CSV)
release = release[release["eid"].astype(str).isin(allowed)]
eids = _ordered_unique(release["eid"].astype(str))
one = ONE(base_url="https://openalyx.internationalbrainlab.org",
          silent=True, cache_dir=str(CACHE_DIR))
for number, eid in enumerate(eids, start=1):
    rows = release[release["eid"].astype(str) == eid]
    neural, decoder_input, decoder_output, regions, info = make_session(one, rows)
```

iii. The trajectory says revised ALF objects are not all represented correctly by the static tables, so the agent deliberately used ONE's revision-resolving loaders. It considered the CSV a frozen insertion list and documented every skipped session rather than silently filling unavailable data.

## 1-b. How are the data split into subjects?

i. Subject names come from the release-table rows. After successful sessions are assembled, unique names are sorted and each session receives an integer `subject_idx`.

ii.
```python
"subject": str(eid_rows.iloc[0]["subject"]),
subjects = sorted({info["subject"] for info in session_info})
subject_lookup = {name: i for i, name in enumerate(subjects)}
subject_idx = np.asarray(
    [subject_lookup[info["subject"]] for info in session_info], dtype=np.int64
)
```

iii. The agent relied on the release metadata's explicit subject field; its trajectory notes that session ordering and source identity were preserved for reproducibility.

## 1-c. How are the data split into sessions?

i. The EID is the session key. All release rows with the same EID (usually separate probes) are grouped and passed once to `make_session`; one successful EID becomes one element of each top-level session list.

ii.
```python
eids = _ordered_unique(release["eid"].astype(str))
rows = release[release["eid"].astype(str) == eid]
neural, decoder_input, decoder_output, regions, info = make_session(one, rows)
neural_all.append(neural)
```

iii. The trajectory treats the release EID as the repository's native behavioral-session identifier and merges multiple probes because they share that session's behavior.

## 1-d. How are the data split into trials?

i. `SessionLoader.load_trials()` supplies one table row per trial. After masks are applied, the code loops over retained row indices and creates one neural, input, and output array per retained trial.

ii.
```python
session_loader.load_trials()
trials = session_loader.trials.copy()
source_indices = np.flatnonzero(source_mask)
for i in range(len(source_indices)):
    neural.append(np.ascontiguousarray(neural_3d[i]))
    decoder_input.append(...)
    decoder_output.append(...)
```

iii. The agent regarded the trial table as already defining trial boundaries and preserved retained source indices in metadata.

## 1-e. How are trials filtered based on quality controls?

i. It requires finite stimulus, choice, feedback, prior, first-movement, feedback-type, and go-cue fields; first movement 0.08–2.0 s after stimulus; duration from go cue to feedback at most 10 s; and nonzero choice. It then requires usable wheel and camera samples covering the complete two-second window. A session must retain at least two trials.

ii.
```python
mask = np.all(np.isfinite(trials[required].to_numpy(dtype=float)), axis=1)
mask &= reaction_time >= 0.08
mask &= reaction_time <= 2.0
mask &= duration <= 10.0
mask &= trials["choice"].to_numpy(dtype=float) != 0
...
source_indices = source_indices[behavior_good]
if len(source_indices) < 2:
    raise ValueError("fewer than two trials have complete behavior coverage")
```

iii. The docstring says this implements the repository's default trial mask, including the `max_trial_len=10` argument used by `prepare_data`, while retaining the unbiased block. The trajectory repeatedly describes this as the paper's reaction-time/event-completeness mask.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays are derived from each probe's `spikes.times` and `spikes.clusters`. Cluster metadata are used to enumerate units and assign Beryl anatomical regions, but not to QC-filter units.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
spike_times = np.asarray(spikes["times"], dtype=float)
spike_clusters = np.asarray(spikes["clusters"], dtype=np.int64)
cluster_ids = np.unique(spike_clusters)
```

iii. The trajectory explicitly says it chose the repository's “all-cluster spike processing” and retained all Kilosort clusters exactly as the methods code does.

## 2-b. How is the `neural` data processed?

i. For each probe and trial, spikes in `[-0.5, 1.5)` relative to stimulus onset are assigned to nonoverlapping 20 ms bins with `floor`, counted by cluster with `bincount`, and stored as counts. Counts from multiple probes are concatenated along the neuron dimension and converted to `float32`; they are not divided by bin width or smoothed.

ii.
```python
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
flat = local_cluster[in_range] * N_TIME + bin_idx[in_range]
hist = np.bincount(flat, minlength=len(cluster_ids) * N_TIME)
counts[trial_i] = hist.reshape(len(cluster_ids), N_TIME)
...
neural_3d = np.concatenate(probe_counts, axis=1).astype(np.float32)
```

iii. The agent believed the reference method uses unsmoothed spike counts. Its trajectory highlights nonoverlapping 20 ms “spike-count bins” and merging probes from the same session.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It applies no cluster-quality or anatomical filter: every cluster ID that has at least one spike is retained, including clusters mapped to `void` or `root`.

ii.
```python
cluster_ids = np.unique(spike_clusters)
# All sorted clusters are retained; no QC label filter.
regions = cluster_table.iloc[cluster_ids]["acronym"].fillna("void").astype(str)
```

iii. The agent reasoned that `prepare_data` calls the repository spike loader without a QC argument, and repeatedly stated in the trajectory that keeping all sorted clusters matched the methods code.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute spike timestamps are sliced from 0.5 s before through 1.5 s after each trial's `stimOn_times`; subtracting the window start implicitly places stimulus onset at 0.5 s in the binned array.

ii.
```python
begin = stimulus_time + OFF_START_S
end = stimulus_time + OFF_END_S
ib = np.searchsorted(spike_times, begin, side="left")
ie = np.searchsorted(spike_times, end, side="left")
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. The trajectory identifies stimulus alignment and the paper's −0.5 to +1.5 s window as key methodological constraints; all streams use the synchronized session clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, yielding 100 bins over two seconds. Raw spike times are histogrammed directly into that grid; there is no further neural rebinning or interpolation.

ii.
```python
BIN_SIZE_S = 0.020
N_TIME = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
```

iii. The agent cites the papers/repository's 20 ms configuration and confirmed the 100-bin shape throughout validation.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is constructed from the fixed offset start and bin size associated with alignment to `trials.stimOn_times`; the stored vector does not numerically use each onset because it is identical for every trial.

ii.
```python
stimulus_times = trials["stimOn_times"].to_numpy(dtype=float)[source_indices]
TIME_FROM_STIMULUS = OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
```

iii. The trajectory says the stimulus-aligned −0.5 to +1.5 s window and 20 ms bins came directly from the paper and repository.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It generates 100 values `-0.50, -0.48, ..., 1.48`, i.e. the left edge of each neural bin, and copies the same vector into every trial input.

ii.
```python
TIME_FROM_STIMULUS = OFF_START_S + np.arange(N_TIME, dtype=np.float32) * BIN_SIZE_S
decoder_input.append(np.vstack([TIME_FROM_STIMULUS, ...]))
```

iii. The agent intended the vector to represent the shared fixed temporal grid; no separate raw signal processing was considered necessary.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `t` is the left boundary of neural count bin `t`; both use the same `OFF_START_S`, 20 ms spacing, and 100 positions.

ii.
```python
TIME_FROM_STIMULUS = OFF_START_S + np.arange(N_TIME) * BIN_SIZE_S
bin_idx = np.floor((local_time - begin) / BIN_SIZE_S).astype(np.int64)
```

iii. The agent treated this as direct bin-for-bin alignment, although it used bin starts rather than the human reference's bin centers.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived solely from consecutive values of the trials-table `probabilityLeft` column; a value change starts a new block.

ii.
```python
block_number = trial_numbers_in_block(
    trials["probabilityLeft"].to_numpy(dtype=float)
)
```

iii. The agent inferred block boundaries from the prior because the trials table does not supply a separate block identifier.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. Before filtering, it initializes every block at 1 and increments while `probabilityLeft` equals the preceding trial. The retained trial's one-indexed number is then repeated over all 100 time bins.

ii.
```python
out = np.ones(len(probability_left), dtype=np.float32)
for i in range(1, len(out)):
    if probability_left[i] == probability_left[i - 1]:
        out[i] = out[i - 1] + 1.0
...
np.full(N_TIME, trial_in_block[i], np.float32)
```

iii. The trajectory emphasizes preserving original trial ordinals, so excluded trials still advance the count. Metadata explicitly records that indexing is one-based.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It is derived from `trials.choice`, after no-choice (`0`) trials are removed. The code maps raw `+1` to categorical `1` and raw `-1` to `0`.

ii.
```python
choice_raw = trials["choice"].to_numpy(dtype=float)[source_indices]
choice = (choice_raw == 1.0).astype(np.int8)
```

iii. The code validates that only ±1 remain, but the trajectory gives no explicit justification for the direction of this mapping. It appears to assume `+1` denotes the class labelled `right`, which conflicts with IBL's convention.

## 5-b. What processing is involved in computing `output` *Choice*?

i. After binary recoding, the single per-trial class is repeated across all 100 time bins so it can be stacked with time-varying outputs.

ii.
```python
np.full(N_TIME, choice[i], np.int8)
```

iii. The module docstring explains that static targets are repeated over time so all four targets can share one output matrix.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes from `trials.probabilityLeft` for each retained trial.

ii.
```python
prior_raw = trials["probabilityLeft"].to_numpy(dtype=float)[source_indices]
```

iii. The agent identifies this as the task's block prior, already present in the trial table.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 are recoded to 0, 1, and 2, respectively; unexpected values raise an error. The class is repeated across 100 time bins.

ii.
```python
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_raw, value)] = label
np.full(N_TIME, prior[i], np.int8)
```

iii. This is the exact mapping requested in the task; repetition permits stacking with dynamic targets.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel()` loads wheel timestamps and derives velocity from raw wheel position/timestamps; the agent takes the absolute velocity as speed.

ii.
```python
loader.load_wheel()
wheel_times = loader.wheel["times"].to_numpy(dtype=float)
wheel_speed = np.abs(loader.wheel["velocity"].to_numpy(dtype=float))
```

iii. The agent followed the repository-defined wheel speed and regarded `SessionLoader`'s interpolation/filtering as the standard IBL processing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. After `SessionLoader` derives filtered velocity, the absolute value is linearly interpolated/extrapolated within each valid trial window at 20 ms right-edge times. Trials lacking near-complete endpoint coverage are rejected. The resulting retained session-wide values are discretized into tertiles.

ii.
```python
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
interp = interp1d(local_t, local_v, kind="linear",
                  fill_value="extrapolate")(stimulus_time + relative_endpoints)
wheel_class, wheel_thresholds = discretize_tertiles(wheel)
```

iii. The trajectory says right-edge interpolation reproduces the repository's `get_behavior_per_interval`, and session-level tertiles accommodate session/camera scale differences.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles of all retained wheel trial-time values in a session define low (`0`), medium (`1`), and high (`2`) using `np.digitize`.

ii.
```python
thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The agent chose within-session tertiles to make the three requested classes balanced and robust to scale differences; the trajectory reports that all dynamic classes were balanced by construction.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are queried on the same stimulus-relative two-second interval and at the right edge of each 20 ms neural count bin (−0.48 through 1.50 s), so output index `t` is paired with neural bin `t`.

ii.
```python
query = stimulus_time + relative_endpoints
relative_endpoints = OFF_START_S + BIN_SIZE_S * np.arange(1, N_TIME + 1)
```

iii. The agent believed endpoint sampling was the repository's exact interval convention and therefore the appropriate pairing for binned spike counts.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the released `whiskerMotionEnergy` trace and timestamps from the left camera when usable, otherwise the right camera.

ii.
```python
for view in ("left", "right"):
    loader.load_motion_energy(views=[view])
    motion_df = loader.motion_energy[f"{view}Camera"]
    motion_df["whiskerMotionEnergy"].to_numpy(dtype=float)
```

iii. The agent says this follows the reference loader's left-first fallback and excludes a session if neither camera has any usable trace.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released trace receives no filtering or normalization in this script. It is linearly interpolated/extrapolated at the same 20 ms right-edge grid as wheel speed, after a coverage test, then classified using retained session-wide tertiles.

ii.
```python
motion, motion_good = interpolate_trials(
    motion_df["times"].to_numpy(dtype=float),
    motion_df["whiskerMotionEnergy"].to_numpy(dtype=float), stimulus_times)
motion_class, motion_thresholds = discretize_tertiles(motion)
```

iii. The trajectory states that revised motion-energy objects were selected where available and that session-level tertiles handle camera-specific scales.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. As with wheel speed, all retained values in one session are split at the 1/3 and 2/3 quantiles and labelled 0, 1, and 2.

ii.
```python
thresholds = np.quantile(values.reshape(-1), [1.0 / 3.0, 2.0 / 3.0])
labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The agent intended balanced low/medium/high classes within each session.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy uses the same absolute stimulus onset and two-second interval but is sampled at each 20 ms bin's right edge; output index `t` is paired with neural count bin `t`.

ii.
```python
query = stimulus_time + relative_endpoints
interp1d(local_t, local_v, kind="linear", fill_value="extrapolate")(query)
```

iii. The agent justified this as the methods repository's endpoint-sampling convention on the common session clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Missing/nonfinite trial fields and incomplete behavior windows remove trials. Left-camera failure falls back to right. Sessions with fewer than two usable trials, no spikes, no usable camera trace, malformed data, or any other processing exception are skipped and their exception is stored in metadata. Output is written through a temporary file and atomically renamed.

ii.
```python
try:
    ...
except Exception as exc:
    skipped_sessions.append({"eid": eid, "reason": repr(exc)})
...
temporary = output_path.with_suffix(output_path.suffix + ".tmp")
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
temporary.replace(output_path)
```

iii. The trajectory repeatedly states that absent streams are excluded and documented rather than padded or silently filled. Fifteen sessions were ultimately excluded for unusable whisker motion energy.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large per-probe spike arrays, trial-wise spike binning for hundreds of trials and thousands of clusters, holding the full dense result, and finally serializing/validating the roughly 99 GB pickle dominate runtime. Spike volume, rather than trial count alone, drove per-session time.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
for trial_i, stimulus_time in enumerate(stimulus_times):
    ...
    hist = np.bincount(...)
with temporary.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The trajectory directly reports that dense two-probe recordings take longer because of spike volume and that serialization and validator scanning were expensive due to dense float32 matrices.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loops in `interpolate_trials` and `bin_probe_spikes`, the loop computing trial number within a block, the per-trial assembly loop, and region-index list comprehensions could be vectorized or batched. The spike and behavior loops are the principal candidates.

ii.
```python
for i, stimulus_time in enumerate(stimulus_times):
    ... interp1d(...)(query)
for trial_i, stimulus_time in enumerate(stimulus_times):
    ... np.bincount(...)
for i in range(1, len(out)):
    ...
for i in range(len(source_indices)):
    neural.append(...)
```

iii. The trajectory does not explicitly discuss vectorization; its focus was correctness and full-release completion. The loops make variable session slices straightforward but contribute Python overhead.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly filters the release DataFrame once per EID, instantiates loaders per session/probe, converts the same columns to arrays, computes equivalent behavior coverage/interpolation separately for wheel and each attempted camera, creates a new `BrainRegions` object per probe, and allocates repeated constant rows for every trial.

ii.
```python
rows = release[release["eid"].astype(str) == eid]
...
beryl = BrainRegions().acronym2acronym(regions.to_numpy(), mapping="Beryl")
...
np.full(N_TIME, choice[i], np.int8)
np.full(N_TIME, prior[i], np.int8)
```

iii. No specific justification appears in the trajectory. The repeated operations simplify modular session/probe processing and are small relative to spike I/O and dense output serialization.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It loads and merges the full cluster/channel metadata although only acronyms are ultimately retained; computes/stores extensive session diagnostics and thresholds not used by decoder tensors; constructs behavior traces for initially valid trials before dropping behavior-incomplete ones; and represents static targets and trial inputs as 100 repeated values. Most significantly, it processes and stores all low-quality/`void` clusters, which the human reference would discard at QC.

ii.
```python
cluster_table = SpikeSortingLoader.merge_clusters(
    spikes, clusters, channels, compute_metrics=False).to_df()
...
info = {"retained_trial_indices": ..., "wheel_speed_tertile_thresholds": ...}
...
np.full(N_TIME, trial_in_block[i], np.float32)
np.full(N_TIME, choice[i], np.int8)
```

iii. The trajectory justifies the large all-cluster representation as matching its reading of the methods repository, and static repetition as necessary to stack static and dynamic variables. It does not identify these as discardable costs.
