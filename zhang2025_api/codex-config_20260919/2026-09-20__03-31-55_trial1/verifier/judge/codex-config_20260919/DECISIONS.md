# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent constructs two local `One` indexes, one from `Brainwidemap` and one from `2025_Q3_IBL_et_al_BWM`, merges their dataset tables in memory, repairs trial-table revision paths in memory, and selects the intersection of their session IDs. If `DATALIMIT_SUBSET.csv` exists, it further restricts the IDs. Scientific arrays are then loaded through `SessionLoader` and `SpikeSortingLoader`, not opened directly.

ii.
```python
one = One(cache_dir=CACHE_ROOT)
one.load_cache(SPIKE_RELEASE)
behavior_one = One(cache_dir=CACHE_ROOT)
behavior_one.load_cache(BEHAVIOR_RELEASE)
datasets = pd.concat([one._cache["datasets"], behavior_one._cache["datasets"]])
datasets = datasets[~datasets.index.duplicated(keep="last")].sort_index()
one._cache["datasets"] = datasets
eids = sorted(set(map(str, one.search())) & set(map(str, behavior_one.search())))
```

iii. The notes say the two releases are complementary, the newer cache contains corrected behavioral products, and patching only the in-memory index preserves the requirement to use ONE. The intersection initially contains 459 task sessions; 441 ultimately convert.

## 1-b. How are the data split into subjects?

i. Each session's subject is read from `one.get_details`. Unique subjects are retained in first-session order, and each session gets an integer lookup index.

ii.
```python
details = one.get_details(eid, full=False)
info = {"subject": str(details["subject"]), ...}
subject_names = list(dict.fromkeys(subjects))
subject_lookup = {name: i for i, name in enumerate(subject_names)}
"subject_idx": np.asarray([subject_lookup[x] for x in subjects], dtype=np.int32)
```

iii. The agent notes that ONE exposes the subject identity directly, avoiding filename parsing. Four of the 139 candidate subjects disappear because all their sessions are excluded.

## 1-c. How are the data split into sessions?

i. ONE session IDs (`eid`s) define sessions. Each `eid` is converted independently, including all of its pykilosort probe collections, and each successful result becomes one top-level session.

ii.
```python
for collection in one.list_collections(eid):
    ...
return sorted(names)
...
neural_all.append(neural)
input_all.append(inp)
output_all.append(out)
```

iii. The notes justify this as matching the release and reference code, where probes from the same behavioral session are merged rather than treated as independent sessions.

## 1-d. How are the data split into trials?

i. Trial rows from `SessionLoader.load_trials` define trials. After a Boolean mask is built, retained row indices are used consistently to slice events and behavioral traces and to create one neural/input/output item per trial.

ii.
```python
sess.load_trials(revision=TRIAL_REVISION)
...
selected = np.flatnonzero(keep)
neural = bin_spikes(spike_times, spike_clusters, stim[selected], len(regions))
for row, trial_idx in enumerate(selected):
    session_input.append(inp)
    session_output.append(out)
```

iii. The agent treats the trials table as authoritative and reports independent checks that converted raw-trial indices reproduce source values.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have finite required fields, reaction time from 0.08 through 2 s, go-cue-to-feedback duration at most 10 s, a nonzero valid choice, a prior in {0.2, 0.5, 0.8}, and complete wheel/camera interpolation coverage. The agent also removes trials whose entire neural population is zero in the two-second window and requires at least two retained trials per session.

ii.
```python
for col in REQUIRED_TRIAL_COLUMNS:
    mask &= np.isfinite(trials[col].to_numpy(dtype=float))
mask &= (rt >= 0.08) & (rt <= 2.0)
mask &= duration <= 10.0
mask &= np.isin(choice, (-1, 1))
mask &= np.isin(np.round(prior, 6), (0.2, 0.5, 0.8))
keep = base_mask & wheel_good & whisker_good
neural_valid = np.asarray([np.any(x) for x in neural], dtype=bool)
```

iii. The notes attribute the event, reaction-time, duration, and no-choice rules to the supplied Zhang code. The zero-population rule was added after validation found three trials in an apparent recording gap.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural matrices are derived from spike timestamps and spike cluster IDs returned by `SpikeSortingLoader`. Cluster/channel data supply the number, identity, anatomical acronym, and QC metadata of units.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
clusters = loader.merge_clusters(spikes, clusters, channels)
all_times.append(np.asarray(spikes["times"], dtype=np.float64)[valid])
all_clusters.append(spike_cluster[valid] + offset)
all_regions.append(np.asarray(clusters["acronym"], dtype=object).astype(str))
```

iii. The notes state this follows the reference loaders and merges all probes belonging to one session.

## 2-b. How is the `neural` data processed?

i. Probes are concatenated with cluster-ID offsets and spikes stable-sorted by time. For each trial, spikes are counted in 100 half-open 20-ms bins over [-0.5, 1.5). The saved values are unsmoothed raw spike counts (`float32`), not counts divided by bin width.

ii.
```python
order = np.argsort(times, kind="stable")
...
tb = np.searchsorted(edges, st, side="right") - 1
flat = sc[valid] * N_BINS + tb[valid]
count = np.bincount(flat, minlength=n_clusters * N_BINS)
trials.append(count.reshape(n_clusters, N_BINS).astype(np.float32))
```

iii. The agent says the supplied decoder cache uses unsmoothed spike counts and half-open bins, and records `"neural_representation": "unsmoothed spike counts in half-open 20-ms bins"`.

## 2-c. How is the `neural` data filtered based on quality controls?

i. It is not filtered by unit QC. The code computes `label >= 1` only for metadata (`n_good_units`) but retains every valid Kilosort cluster, including clusters later mapped to `void` or `root`.

ii.
```python
label = np.asarray(clusters.get("label", np.zeros(n_clusters)), dtype=float)
all_good.append(label >= 1)
...
neural = bin_spikes(..., len(regions))
"neuron_filter": "all Kilosort clusters (reference decoder cache qc=None)"
```

iii. The notes explicitly reconcile the paper's 75,708 well-isolated units with the supplied cache call `qc=None` and choose all 621,733 clusters because they judged the decoder code the more directly applicable reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Every trial's absolute spike times are assigned to edges formed by adding `stimOn_times` to relative edges from -0.5 to +1.5 s.

ii.
```python
edges_rel = np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
for stim in np.asarray(stim_times, dtype=np.float64):
    edges = stim + edges_rel
```

iii. The agent says this exactly follows the requested stimulus-onset alignment and supplied cache settings.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, producing 100 bins across two seconds. Spikes are binned once; there is no smoothing or later neural rebinning.

ii.
```python
BIN_SIZE = 0.020
OFF_START = -0.5
OFF_END = 1.5
N_BINS = 100
```

iii. The notes cite the methods-paper cache representation of 100 non-overlapping 20-ms bins.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is based on the stimulus-onset-aligned bin grid, whose absolute placement derives from `trials.stimOn_times`; the saved relative vector itself is the fixed ends of the 100 bins.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
stim = trials["stimOn_times"].to_numpy(dtype=float)
```

iii. The agent says bin-end timestamps match the supplied reference behavior interpolation.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. The code creates the arithmetic sequence -0.48, -0.46, ..., 1.50 seconds and casts it to `float32`; the same sequence is reused for every trial.

ii.
```python
BIN_END_TIMES = np.arange(1, N_BINS + 1, dtype=np.float64) * BIN_SIZE + OFF_START
inp = np.vstack([BIN_END_TIMES.astype(np.float32), ...])
```

iii. The notes report this as a direct task-defined coordinate and describe its range explicitly.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `t` is the end, not the center, of neural bin `t`. Thus the first input value -0.48 labels the neural interval [-0.50, -0.48), and the final value 1.50 labels [1.48, 1.50).

ii.
```python
edges_rel = np.arange(N_BINS + 1) * BIN_SIZE + OFF_START
BIN_END_TIMES = np.arange(1, N_BINS + 1) * BIN_SIZE + OFF_START
```

iii. The agent considers bin-end labeling aligned because it follows the behavior code and found no visible one-bin shift in plots.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the complete unfiltered `trials.probabilityLeft` sequence; each run of equal adjacent prior values is treated as one block.

ii.
```python
block_num = block_trial_numbers(trials["probabilityLeft"].to_numpy())
```

iii. The notes identify `probabilityLeft` as the available block marker and retain initial 0.5-prior trials.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments when the current prior equals the previous prior and resets to zero when it changes. It is computed before filtering, so excluded trials still count, then broadcast over all 100 time points.

ii.
```python
for i in range(1, len(p)):
    out[i] = out[i - 1] + 1 if p[i] == p[i - 1] else 0
...
np.full(N_BINS, block_num[trial_idx], dtype=np.float32)
```

iii. The agent calls this the animal's true ordinal within the original block and reports unit/spot checks of the result.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from `trials.choice`, after excluding values other than -1 and +1.

ii.
```python
choice = trials["choice"].to_numpy()
mask &= np.isin(choice, (-1, 1))
```

iii. The notes identify the trials table as the authoritative source and independently spot-check raw labels.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The agent maps raw -1 to class 0 and raw +1 to class 1, broadcasts the class over time, and labels classes 0/1 as left/right in metadata.

ii.
```python
choice = 0 if trials.iloc[trial_idx]["choice"] == -1 else 1
np.full(N_BINS, choice, dtype=np.int8)
...
["left", "right"]
```

iii. The notes claim raw-label spot checks passed, but do not justify reversing the IBL convention used by the human reference (+1 = left, -1 = right).

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from `trials.probabilityLeft`.

ii.
```python
p = float(np.round(trials.iloc[trial_idx]["probabilityLeft"], 6))
```

iii. The agent identifies this column as the task's block prior and explicitly retains the unbiased 0.5 class.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values are rounded to six decimals, mapped 0.2→0, 0.5→1, and 0.8→2, then broadcast over 100 bins as `int8`.

ii.
```python
prior_map = {0.2: 0, 0.5: 1, 0.8: 2}
np.full(N_BINS, prior_map[p], dtype=np.int8)
```

iii. This is the mapping explicitly required by the task; rounding protects against floating-point representation differences.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. `SessionLoader.load_wheel` supplies wheel timestamps and filtered wheel velocity. Speed is the absolute value of that velocity.

ii.
```python
sess.load_wheel()
wheel_speed = np.abs(wheel["velocity"].to_numpy(dtype=float))
wheel_interp, wheel_good = interpolate_trials(wheel["times"].to_numpy(), wheel_speed, stim)
```

iii. The notes say this matches the supplied reference, including `SessionLoader`'s standard wheel smoothing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Absolute velocity is linearly interpolated at each trial's 100 bin-end timestamps. Trials without adequate boundary coverage or finite interpolated values are rejected, then the retained aligned values are discretized by session-wide tertiles.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
result[i] = np.interp(target, local_t, local_v)
...
wheel_labels, wheel_q = discretize_tertiles(wheel_interp[selected])
```

iii. The agent attributes interpolation at bin ends to `get_behavior_per_interval` and uses session tertiles because the task requires categorical outputs.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The 1/3 and 2/3 quantiles of all finite aligned wheel samples in the final retained trials of a session define class boundaries; `np.digitize` gives 0, 1, or 2. Collapsed thresholds reject a session.

ii.
```python
thresholds = np.quantile(finite, [1 / 3, 2 / 3])
if not thresholds[0] < thresholds[1]:
    raise RuntimeError(...)
labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The agent chose session-wise tertiles to create balanced decoder classes and avoid cross-session scale differences.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel values are evaluated at absolute `stimOn_times + BIN_END_TIMES`; each saved value therefore corresponds to the end of the same-index neural bin, rather than its center.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
```

iii. The notes say the reference behavioral helper samples bin ends and independent plots showed no visible shift.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses the `whiskerMotionEnergy` column and timestamps loaded by `SessionLoader.load_motion_energy`, preferring the left camera and falling back to the right.

ii.
```python
for view in ("left", "right"):
    sess.load_motion_energy(views=[view])
    frame = sess.motion_energy[f"{view}Camera"]
    if "whiskerMotionEnergy" in frame and len(frame) > 1:
        motion = frame
```

iii. The agent says this is the camera fallback used by the supplied reference and avoids fabricating a missing modality.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. The released motion-energy trace is linearly interpolated at trial bin ends without additional filtering or normalization. Inadequately covered trials are dropped, and retained values are discretized using session-wide tertiles.

ii.
```python
whisker_interp, whisker_good = interpolate_trials(
    motion["times"].to_numpy(), motion["whiskerMotionEnergy"].to_numpy(), stim
)
whisker_labels, whisker_q = discretize_tertiles(whisker_interp[selected])
```

iii. The agent describes the released trace as already processed and applies only the interpolation and task-required categorization.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same per-session 1/3 and 2/3 quantiles and `np.digitize` rule as wheel speed, yielding low/medium/high classes 0/1/2.

ii.
```python
thresholds = np.quantile(finite, [1 / 3, 2 / 3])
labels = np.digitize(values, thresholds, right=False).astype(np.int8)
```

iii. The notes justify tertiles as balanced categorical targets and record each session's thresholds for auditability.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Camera energy is sampled at absolute stimulus onset plus each neural bin's end time, so same-index entries label neural bins by their right boundary.

ii.
```python
targets = np.asarray(stim_times)[:, None] + BIN_END_TIMES[None, :]
result[i] = np.interp(target, local_t, local_v)
```

iii. The agent says the streams share the synchronized session clock and that bin-end sampling follows the reference behavior helper.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Invalid cluster IDs are removed. Missing trial columns, probes, both camera streams, insufficient trials, nonfinite/incomplete behavior, flat tertile thresholds, or unusable sessions raise an exception; the session is logged and skipped. Trials with incomplete behavior or all-zero population activity are dropped. The conversion does not impute. Eighteen sessions were excluded.

ii.
```python
valid = (spike_cluster >= 0) & (spike_cluster < n_clusters)
...
except Exception as exc:
    return eid, None, f"{type(exc).__name__}: {exc}"
...
failures.append({"eid": eid, "error": error})
```

iii. The notes say imputation would fabricate mandatory outputs. They document 14 camera-absent sessions, two missing-prior sessions, one collapsed stream, and one with no valid trials.

## 10-a. What are the most time-consuming steps of the code?

i. Loading/merging large spike sorting products, `SessionLoader`'s dense 1-kHz wheel processing, per-trial spike binning, and serializing the roughly 98-GiB dense pickle dominate. The code parallelizes independent sessions across up to 16 threads.

ii.
```python
spikes, clusters, channels = loader.load_spike_sorting()
sess.load_wheel()
neural = bin_spikes(...)
executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
pickle.dump(data, stream, protocol=5)
```

iii. The notes measured about 5.95 seconds per sample session before serialization and 1,235.8 seconds for the corrected full run.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The Python loops over trials in `interpolate_trials`, `bin_spikes`, output assembly, and session validation could be vectorized or batched further. The zero-activity scan and region-index construction are also list-based, though much cheaper. Session conversion is parallel rather than vectorized.

ii.
```python
for i, target in enumerate(targets): ...
for stim in np.asarray(stim_times, dtype=np.float64): ...
for row, trial_idx in enumerate(selected): ...
for x, i, o in zip(neural, inputs, outputs): ...
```

iii. The notes emphasize that sorted `searchsorted`, flattened `bincount`, one stable spike sort, and session parallelism already remove larger costs; they do not claim every remaining loop is optimal.

## 10-c. What processing does the code repeat multiple times?

i. Wheel and whisker streams independently run the same interpolation/coverage logic and the same tertile discretization. Every full-conversion worker also builds and merges its own pair of ONE cache indexes. Validation loops over all dense trial arrays after construction, and processing plots repeat access to already interpolated data when requested.

ii.
```python
wheel_interp, wheel_good = interpolate_trials(...)
whisker_interp, whisker_good = interpolate_trials(...)
wheel_labels, wheel_q = discretize_tertiles(...)
whisker_labels, whisker_q = discretize_tertiles(...)
if not hasattr(worker_state, "one"):
    worker_state.one, _ = build_one(verbose=False)
```

iii. The separate modality calls share tested helper functions. Worker-local ONE construction was deliberately introduced after shared mutable ONE state caused nondeterministic failures.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes and concatenates the good-unit Boolean flags but uses them only for session metadata, while all units enter `neural`. It loads merged cluster/channel information beyond the acronyms and labels ultimately retained, creates large diagnostic plots only on request, and broadcasts per-trial choice, prior, and trial number into 100 repeated values because the target format/trainer expects time-shaped arrays.

ii.
```python
all_good.append(label >= 1)
...
"n_good_units": int(np.sum(good_units))
...
np.full(N_BINS, choice, dtype=np.int8)
np.full(N_BINS, prior_map[p], dtype=np.int8)
```

iii. The notes retain good-unit counts for comparison with paper statistics and broadcast static variables to satisfy the decoder schema; these are auditable conveniences rather than scientific transformations.
