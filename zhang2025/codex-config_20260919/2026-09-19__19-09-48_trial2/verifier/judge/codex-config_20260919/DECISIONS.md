# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent reads the publication-freeze CSV, groups its probe rows by session EID, constructs each cached session path from lab/subject/date/session number, and directly opens parquet/NumPy files. Full mode processes every freeze group, with four threads.

ii.
```python
freeze = pd.read_csv(FREEZE, index_col=0)
groups = list(freeze.groupby("eid", sort=False))
path = session_path(first)
trials, trial_file = load_trials(path)
with ThreadPoolExecutor(max_workers=4) as pool:
    for result in pool.map(run_group, groups):
```

iii. The notes justify the freeze as matching the published 459-session/699-probe release exactly, excluding two unrelated cached sessions, and direct memory-mapped loading as faster and more memory efficient than materializing full session tables.

## 1-b. How are the data split into subjects?

i. Subject names come from the freeze rows. After conversion, subjects are unique in first-appearance order and each retained session receives an integer `subject_idx`.

ii.
```python
subjects = list(dict.fromkeys(x["session_info"]["subject"] for x in converted))
subject_lookup = {name: i for i, name in enumerate(subjects)}
"subject_idx": np.asarray([subject_lookup[x["session_info"]["subject"]]
                           for x in converted], dtype=np.int64)
```

iii. The agent treats the release's subject string as the authoritative mouse identifier and records that 136 of 139 release mice remain after stream/session exclusions.

## 1-c. How are the data split into sessions?

i. Rows with the same EID are one session; simultaneous probes are merged within that group in sorted probe-name order.

ii.
```python
groups = list(freeze.groupby("eid", sort=False))
for _, row in rows.sort_values("probe_name").iterrows():
    info = load_probe_units(probe_path, brain_regions)
```

iii. The notes say probes share session behavior and should not be treated as independent recordings, matching the papers' session-level dependence rationale.

## 1-d. How are the data split into trials?

i. Each row of `_ibl_trials.table.pqt` is a native trial. Retained row indices define stimulus-aligned windows and become separate nested-list trial arrays.

ii.
```python
trial_indices = np.flatnonzero(valid)
begins = begins_all[trial_indices]
ends = ends_all[trial_indices]
for i in range(len(trial_indices)):
    neural_trials.append(counts[i].astype(np.float32))
```

iii. The trial table already supplies trial boundaries/events; original indices are retained in metadata for auditability.

## 1-e. How are trials filtered based on quality controls?

i. Trials must have six required nonmissing fields, reaction time 0.08–2 s, feedback-minus-go-cue duration at most 10 s, nonzero choice, and complete wheel and camera coverage of the two-second window. Sessions with fewer than two surviving trials are dropped.

ii.
```python
present &= trials[name].notna().to_numpy()
rt_ok = (rt >= 0.08) & (rt <= 2.00)
duration_ok = duration <= 10.0
choice_ok = trials["choice"].to_numpy() != 0
mask = present & rt_ok & duration_ok & choice_ok
valid = base_mask & motion_ok & wheel_ok
```

iii. The agent says this reproduces the supplied `load_trials_and_mask` settings and avoids imputing uncovered continuous outputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural arrays derive from `spikes.times.npy` and `spikes.clusters.npy`; cluster metrics identify good cluster IDs, while cluster channels and atlas IDs provide regions.

ii.
```python
spike_times_path = preferred(probe_path.glob("**/spikes.times.npy"))
spike_clusters_path = preferred(probe_path.glob("**/spikes.clusters.npy"))
metrics = pd.read_parquet(metrics_path)
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
```

iii. The notes identify these as the released sorted-spike sources and retain cluster/probe identities in metadata for verification.

## 2-b. How is the `neural` data processed?

i. Good-unit spikes are binned as raw counts into 100 half-open 20-ms bins per trial. Probes are concatenated, counts are built as `uint16`, and each final trial is cast to `float32`; there is no smoothing or division by bin width.

ii.
```python
bins = np.floor(rel_t / BIN_SIZE).astype(np.int64)
destination[trial] = np.bincount(flat, minlength=width).reshape(
    destination.shape[1], N_BINS)
neural_trials.append(counts[i].astype(np.float32))
```

iii. The agent explicitly chose raw counts, claiming this matches `bin_spiking_data`; `uint16` construction and memmaps reduce memory, while the target's final float32 type is preserved.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only rows with cluster metric `label >= 1` are retained. Peak-channel atlas IDs are mapped to Allen and then Beryl acronyms, but no `void`-region exclusion is applied.

ii.
```python
good_rows = np.flatnonzero(metrics["label"].to_numpy() >= 1)
allen = brain_regions.id2acronym(np.asarray(atlas_ids[peak_channels], dtype=np.int64))
beryl = brain_regions.acronym2acronym(allen, mapping="Beryl").astype(str)
```

iii. The notes justify `label >= 1` as the data paper's well-isolated criterion, reproducing 75,708 good units in the complete release.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial windows begin 0.5 s before `stimOn_times` and end 1.5 s after it. Absolute spike times are sliced by these bounds and converted to offsets from the window beginning before binning.

ii.
```python
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
ends_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_END
rel_t = np.asarray(times[lo:hi], dtype=np.float64)[keep] - beg
```

iii. The common IBL clock makes stimulus-relative slicing sufficient; the chosen window follows the supplied decoding configuration.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The resolution is 20 ms, with 100 bins across two seconds. Spikes are binned once; no later temporal rebinning or smoothing is applied.

ii.
```python
BIN_SIZE = 0.020
N_BINS = int(round((OFF_END - OFF_START) / BIN_SIZE))
```

iii. The notes cite both the reference implementation and method description for the 20-ms, 100-step configuration.

## 3-a. What variables in the raw data is `input` *Time since stimulus onset* derived from?

i. It is analytically defined from `stimOn_times`, the −0.5/+1.5-s window, and 20-ms bins rather than measured from another stream.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
begins_all = trials["stimOn_times"].to_numpy(dtype=float) + OFF_START
```

iii. The notes call it the reference right-edge behavioral interpolation grid.

## 3-b. What processing is involved in computing `input` *Time since stimulus onset*?

i. It creates the right edges of all 100 bins, from −0.48 through +1.50 s, and copies that row into every trial.

ii.
```python
TIME_GRID = (OFF_START + BIN_SIZE * np.arange(1, N_BINS + 1)).astype(np.float32)
inp = np.vstack((TIME_GRID, np.full(N_BINS, block_number[i], dtype=np.float32)))
```

iii. The agent chose right edges because it interpreted the supplied behavior code as evaluating at interval right edges.

## 3-c. How is the `input` *Time since stimulus onset* aligned with the neural data?

i. Entry `j` is the right edge of neural half-open bin `j`; therefore it labels the end, not the center, of the associated spike-count interval.

ii.
```python
"time_coordinate": "right edge of each half-open neural spike-count bin, seconds from stimulus onset"
```

iii. The agent argues that this also matches the timestamps used to interpolate both behavior streams.

## 4-a. What variables in the raw data is `input` *Trial number in block* derived from?

i. It is derived from the full native sequence of `probabilityLeft` values; a changed or nonfinite value starts a new block.

ii.
```python
native_prob = trials["probabilityLeft"].to_numpy(dtype=float)
block_number = block_trial_numbers(native_prob)[trial_indices]
```

iii. Because no explicit block ID exists, the constant block prior is used to infer transitions.

## 4-b. What processing is involved in computing `input` *Trial number in block*?

i. A zero-based counter increments while adjacent native prior values match and resets otherwise. It is computed before filtering, selected at retained indices, and repeated over time.

ii.
```python
count = count + 1 if same else 0
out[i] = count
np.full(N_BINS, block_number[i], dtype=np.float32)
```

iii. Computing before filtering preserves the animal's true experimental position even when intervening bad trials are removed.

## 5-a. What variables in the raw data is `output` *Choice* derived from?

i. It comes from the trial table's `choice` column after no-choice trials are excluded.

ii.
```python
choice_native = trials["choice"].to_numpy(dtype=float)[trial_indices]
```

iii. The agent treats native −1 and +1 as the two task choices and excludes native zero.

## 5-b. What processing is involved in computing `output` *Choice*?

i. The code maps native −1 to category 0 and everything retained (normally +1) to category 1, then repeats the category across 100 bins. It labels these categories left and right respectively.

ii.
```python
choice = np.where(choice_native == -1, 0, 1).astype(np.int64)
np.full(N_BINS, choice[i], dtype=np.int64)
```

iii. The notes state “native −1 (left) → 0; +1 (right) → 1” to satisfy the requested left=0/right=1 convention.

## 6-a. What variables in the raw data is `output` *Prior probability of left* derived from?

i. It comes directly from the trial table's `probabilityLeft` column.

ii.
```python
prior_native = native_prob[trial_indices]
```

iii. The agent uses the actual block prior requested by the task rather than a model-derived subjective prior.

## 6-b. What processing is involved in computing `output` *Prior probability of left*?

i. Values close to 0.2, 0.5, and 0.8 map to categories 0, 1, and 2; unexpected values raise an error. The result is repeated across time.

ii.
```python
for value, label in ((0.2, 0), (0.5, 1), (0.8, 2)):
    prior[np.isclose(prior_native, value)] = label
if np.any(prior < 0):
    raise ValueError(...)
```

iii. This is the exact mapping specified by the decoder task.

## 7-a. What variables in the raw data is `output` *Wheel speed* derived from?

i. It derives from `_ibl_wheel.timestamps.npy` and `_ibl_wheel.position.npy`.

ii.
```python
times_path = preferred(path.glob("alf/**/_ibl_wheel.timestamps.npy"))
pos_path = preferred(path.glob("alf/**/_ibl_wheel.position.npy"))
```

iii. These are the released raw wheel object used by `SessionLoader` and the reference processing.

## 7-b. What processing is involved in computing `output` *Wheel speed*?

i. Position is interpolated to 1 kHz, differentiated with the standard order-8 20-Hz filtered velocity routine, absolutized, then linearly interpolated within each trial at bin right edges.

ii.
```python
pos_1khz, times_1khz = interpolate_position(raw_times, raw_pos, freq=1000)
velocity, _ = velocity_filtered(pos_1khz, fs=1000, corner_frequency=20, order=8)
BehaviorStream(times_1khz, np.abs(velocity), "wheel")
y = np.interp(x, t, v)
```

iii. The agent says this matches `SessionLoader.load_wheel` and explicitly emulates the reference's final-point extrapolation.

## 7-c. How is `output` *Wheel speed* thresholded into categories?

i. The flattened retained session trace is split at its one-third and two-thirds quantiles; right-sided search produces low/medium/high labels 0/1/2. If thresholds tie, stable-rank equal-frequency labels are substituted.

ii.
```python
thresholds = np.quantile(flat, [1 / 3, 2 / 3])
labels = np.searchsorted(thresholds, values, side="right").astype(np.int64)
```

iii. Session-specific tertiles avoid between-rig scale effects and create balanced categorical outputs; the fallback guarantees classes for degenerate signals.

## 7-d. How is `output` *Wheel speed* aligned with the neural data?

i. Wheel speed is interpolated at `begin + 0.02, ..., begin + 2.00`, the right edges of the corresponding neural bins.

ii.
```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
x = beg + steps
y = np.interp(x, t, v)
```

iii. Both streams use the same session clock and stimulus-derived interval bounds.

## 8-a. What variables in the raw data is `output` *Whisker motion energy* derived from?

i. It uses `<side>Camera.ROIMotionEnergy.npy` and `_ibl_<side>Camera.times.npy`, preferring left and falling back to right.

ii.
```python
for side in ("left", "right"):
    values_path = preferred(path.glob(f"alf/**/{side}Camera.ROIMotionEnergy.npy"))
    times_path = preferred(path.glob(f"alf/**/_ibl_{side}Camera.times.npy"))
```

iii. This follows the reference left-first/right-fallback convention for the whisker-pad ROI trace.

## 8-b. What processing is involved in computing `output` *Whisker motion energy*?

i. If timestamps outnumber frames, leading timestamps are discarded; invalid/nonmonotonic streams are rejected. Otherwise the released values are not filtered or normalized and are linearly interpolated at trial bin right edges.

ii.
```python
if len(times) > len(values):
    times = times[-len(values):]
if not np.all(np.diff(times) > 0):
    continue
motion_cont = interpolate_trials(motion, begins, ends)
```

iii. The notes attribute the timestamp correction to `SessionLoader` and state that no additional motion-energy preprocessing is needed.

## 8-c. How is `output` *Whisker motion energy* thresholded into categories?

i. It uses the same retained-session tertile rule and tied-quantile stable-rank fallback as wheel speed.

ii.
```python
motion_labels, motion_q, motion_method = discretize_tertiles(motion_cont)
```

iii. The agent applies one consistent low/medium/high definition to both continuous outputs.

## 8-d. How is `output` *Whisker motion energy* aligned with the neural data?

i. Motion energy is sampled on the shared clock at the right edge of every stimulus-relative neural bin.

ii.
```python
steps = BIN_SIZE * np.arange(1, N_BINS + 1)
x = beg + steps
```

iii. Complete-window coverage is required first, preventing clamping or imputation at trial boundaries.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The newest revision is preferred; excess leading camera timestamps are trimmed; malformed streams, missing files, uncovered trials, sessions with fewer than two trials, and sessions without good units are rejected rather than imputed. Unexpected priors and release-count mismatches raise errors.

ii.
```python
return sorted(paths, key=lambda p: ("#" in str(p), str(p)))[-1]
if len(times) > len(values): times = times[-len(values):]
if len(trial_indices) < 2: return None
if n_units == 0: return None
```

iii. The notes emphasize reproducibility and no output imputation; 14 sessions without motion energy and one without coverage were excluded.

## 10-a. What are the most time-consuming steps of the code?

i. Reading large spike arrays from mounted storage and serializing the 12.6-GiB pickle dominate. A sequential cold-I/O run projected to 19 minutes; four-session threading reduced the completed conversion to about 349 seconds, including 16.7 seconds serialization.

ii.
```python
times = np.load(info["times_path"], mmap_mode="r")
clusters = np.load(info["clusters_path"], mmap_mode="r")
with ThreadPoolExecutor(max_workers=4) as pool:
with outfile.open("wb") as stream:
    pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Profiling in the notes identified cold spike-array I/O, not numeric computation, as the bottleneck.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. `block_trial_numbers`, per-trial behavior interpolation, per-probe/per-trial spike binning, per-trial output assembly, and some region/index list construction remain Python loops. The main candidates are the per-trial interpolation and spike-binning loops, although each operates on variable-length slices.

ii.
```python
for i in range(1, len(probability_left)):
for i, (beg, end, ib, ie) in enumerate(zip(begins, ends, ibs, ies)):
for trial, (beg, lo, hi) in enumerate(zip(begins, left, right)):
```

iii. The agent says it already vectorized inner spike work with remapping and `bincount`; remaining trial loops keep memory bounded and handle differently sized windows.

## 10-c. What processing does the code repeat multiple times?

i. Each trial separately repeats interpolation setup, spike cluster remapping/allocation, and construction/casting of constant input/output rows. Each probe also repeats glob/revision resolution and file opening.

ii.
```python
mapped = np.full(len(raw_c), -1, dtype=np.int32)
destination[trial] = np.bincount(...)
inp = np.vstack((TIME_GRID, np.full(...)))
out = np.vstack((np.full(...), np.full(...), ...))
```

iii. The notes focus on avoiding larger repetition—full-session loading, per-spike Python work, and repeated processing of behavior-ineligible sessions—rather than claiming no repeated work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It performs a full release preflight scan of every cluster metrics file on every run; computes and retains extensive provenance/audit metadata; loads raw wheel arrays for plots even when plotting is disabled; collects unit probe names/cluster IDs; and casts each count trial from uint16 to float32. Most metadata and plot-only raw wheel values are ignored by decoder training.

ii.
```python
release_stats = release_preflight(freeze)
wheel, raw_wheel_times, raw_wheel_pos = load_wheel_speed(path)
unit_probe_names.extend(...)
unit_cluster_ids.extend(...)
neural_trials.append(counts[i].astype(np.float32))
```

iii. These costs are justified as integrity checks, provenance, optional visual auditing, and target-format compliance, though they are not needed by the downstream model itself.
