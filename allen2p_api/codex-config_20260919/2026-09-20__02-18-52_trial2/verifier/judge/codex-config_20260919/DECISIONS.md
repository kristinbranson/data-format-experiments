# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent constructs an AllenSDK `VisualBehaviorOphysProjectCache` rooted at `/app/data`, obtains its experiment table, and selects locally present, active-behavior experiments whose project code is `VisualBehavior`. It then loads each selected experiment through `get_behavior_ophys_experiment`. Thus “all” means all 168 locally cached active VisualBehavior experiments, not passive experiments or release entries whose NWB is absent locally; 165 survive conversion.

ii.
```python
def make_cache():
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)

mask = ((table["project_code"] == "VisualBehavior")
        & (table["behavior_type"] == "active_behavior")
        & table.index.isin(local_experiment_ids()))
...
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
```

iii. The notes say passive sessions cannot provide meaningful go/catch outcomes, local-ID filtering prevents downloading absent release data, and all access must remain through AllenSDK. The agent reports 168 eligible active experiments and three later exclusions for unusable pupil data.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among experiments that successfully converted. They are sorted globally, and each session receives an integer `subject_idx`.

ii.
```python
subjects = sorted(kept["mouse_id"].astype(str).unique())
subject_map = {x: i for i, x in enumerate(subjects)}
subject_idx = np.asarray([subject_map[str(x)] for x in kept["mouse_id"]], dtype=np.int64)
```

iii. The agent treats the SDK mouse ID as the canonical animal identifier and delays construction until failed experiments are removed, avoiding orphan subjects.

## 1-c. How are the data split into sessions?

i. Every selected ophys experiment is one output session. It does not explicitly group experiments by `ophys_session_id`.

ii.
```python
ids = [int(x) for x in table.index]
for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
    neural.append(n); inputs.append(i); outputs.append(o)
```

iii. The notes establish that this `VisualBehavior` subset is single-plane and its experiment/session mapping is one-to-one, so no multiplane merge is needed.

## 1-d. How are the data split into trials?

i. The SDK `trials` table defines trials. Each retained trial is the half-open interval from `start_time` to `stop_time`, converted to ophys-frame indices with `searchsorted`; lengths therefore vary.

ii.
```python
for trial_id, row in trials.iterrows():
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
    ...
    x = events[:, lo:hi]
```

iii. The agent calls the SDK trial table authoritative and says the half-open interval gives exact segmentation on the required ophys clock.

## 1-e. How are trials filtered based on quality controls?

i. It keeps go or catch trials and rejects aborted and auto-rewarded trials. It additionally requires exactly one of the four outcome flags, at least two frames, finite neural/output arrays, and at least two valid trials per experiment.

ii.
```python
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
...
if flags.sum() != 1: continue
if hi - lo < 2: continue
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)): continue
...
if len(neural_trials) < 2:
    raise ValueError(...)
```

iii. The first four predicates directly implement the requested curation; the extra checks enforce well-formed labels, finite decoder arrays, and the validator’s minimum-trial requirement.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the SDK `events["events"]` arrays: unfiltered L0 inferred calcium-event magnitudes, not dF/F.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
```

iii. The agent initially used dF/F, then changed after its critical review because the supplied paper says its neural analyses used detected calcium events to reduce slow GCaMP dynamics.

## 2-b. How is the `neural` data processed?

i. Event vectors are stacked cell-by-time, cast to float32, truncated with timestamps to their common minimum length, checked for finite values, and sliced by trial. There is no normalization, smoothing, or aggregation.

ii.
```python
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
...
x = events[:, lo:hi]
```

iii. The notes describe this as preserving the paper’s event stream while satisfying the task’s framewise decoder format.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No new neuron-level threshold is imposed. The agent relies on release QC and AllenSDK-valid ROIs, while rejecting an entire experiment if event samples are nonfinite or fewer than two trials survive.

ii.
```python
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
```

iii. The notes state that the release and SDK already exclude invalid ROIs, so an additional cell-quality cutoff would be invented rather than reference-based.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural samples retain native synchronized ophys timestamps and are segmented relative to trial start/stop; trial time zero is the first frame at or after `start_time`.

ii.
```python
lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
x = events[:, lo:hi]
```

iii. The agent says this uses the required ophys timestamp alignment and the SDK trial boundaries without interpolation of neural activity.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One output bin is one native single-plane ophys frame, approximately 32.26 ms (about 31 Hz). No temporal rebinning is applied. Dataset metadata uses the median of per-session median frame intervals.

ii.
```python
"median_frame_interval_ms": float(np.median(np.diff(ts)) * 1000),
...
median_bin = float(np.median([x["median_frame_interval_ms"] for x in infos]))
"time_bin_size": median_bin,
```

iii. The agent argues that retaining native frames preserves resolution and that the selected single-plane sessions share the nominal 31 Hz acquisition rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from task-block rows in `stimulus_presentations`, principally `image_name`, `start_time`, `end_time`, and `omitted`.

ii.
```python
stim = dataset.stimulus_presentations
...
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
```

iii. The notes favor the presentation table because it describes every flash and gray/omitted interval, rather than only the initial and changed trial images.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A global vocabulary is built as `gray` plus sorted non-omitted image names across image sets. The full session begins as gray, and frames in each non-omitted presentation interval receive that image’s integer code; ISIs and omissions remain gray.

ii.
```python
return ["gray"] + sorted(found)
...
image = np.full(n, image_to_idx["gray"], dtype=np.int16)
if not omitted and name in image_to_idx and end > start:
    image[start:end] = image_to_idx[name]
```

iii. The agent says a deterministic global vocabulary preserves cross-session semantics and an explicit gray class faithfully represents the majority ISI/omission frames.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation boundaries are mapped onto the same ophys timestamp vector as events, after which the same `[lo:hi]` trial slice is taken.

ii.
```python
start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
...
y[0] = image[lo:hi]
x = events[:, lo:hi]
```

iii. The agent validated these labels independently against raw SDK timing for sampled trials.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. Image change comes from `stimulus_presentations.is_change` and each presentation’s `start_time`, restricted to the change-detection stimulus block.

ii.
```python
if bool(row.is_change) and start < n:
    change[start] = 1
```

iii. The notes explain that true presentation changes should be positive, while catch sham changes and omissions should remain zero.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-valued session vector is constructed and a single bin—the first ophys frame at or after each true change onset—is set to one.

ii.
```python
change = np.zeros(n, dtype=np.int16)
...
change[start] = 1
```

iii. The agent interprets “right after a change” as an impulse and documents exactly one positive frame per real identity change.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is intrinsically binary: `0 = no_change`, `1 = change`; no numeric threshold is applied beyond the boolean `is_change` flag.

ii.
```python
"output_values": [..., ["no_change", "change"], ...]
```

iii. The SDK already supplies the categorical change flag, so the agent performs no continuous-value thresholding.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Change onset is converted with `searchsorted` on ophys timestamps and the result is sliced with the identical trial bounds used for events.

ii.
```python
start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
...
y[1] = change[lo:hi]
```

iii. The notes report independent boundary checks and exact agreement with the agent’s intended impulse construction.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It uses the AllenSDK filtered `running_speed` table’s `timestamps` and `speed` columns.

ii.
```python
run = dataset.running_speed
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
```

iii. The notes select this stream because it already embodies the SDK/reference encoder corrections and low-pass filtering, unlike raw running speed.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite samples are sorted, linearly interpolated to all ophys timestamps with `np.interp`, and converted to five bins using that session’s 20th, 40th, 60th, and 80th percentiles.

ii.
```python
return np.interp(t_dst, t_src[valid][order], value_src[valid][order])
...
edges = np.percentile(values, [20, 40, 60, 80])
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. The agent says interpolation is valid because clocks are synchronized, while per-session quintiles balance categories and control session-to-session calibration differences.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Each session’s aligned continuous speed distribution determines four percentile thresholds, yielding integer classes 0–4.

ii.
```python
speed_bin = percentile_bins(speed)
```

iii. The task explicitly requests five equal percentile bins; the agent chose per-session rather than global percentiles to make classes approximately balanced within each decoder session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Continuous speed is interpolated onto the full ophys grid before trial slicing, then indexed by the same trial boundaries as neural events.

ii.
```python
speed = interpolate_valid(..., timestamps)
...
y[2] = speed_bin[lo:hi]
x = events[:, lo:hi]
```

iii. The agent relies on hardware-synchronized timestamps and verified raw-to-converted samples.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It derives a diameter-equivalent value from `eye_tracking.pupil_area` and uses eye-tracking timestamps.

ii.
```python
area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
```

iii. The agent argues that area incorporates both ellipse axes and has blink/outlier frames already represented as NaN by the SDK, making it preferable to selecting width or height alone.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area is clipped below at zero and converted to an equivalent circular diameter. Finite diameter samples are sorted and linearly interpolated to ophys frames, then assigned per-session quintile labels.

ii.
```python
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
pupil_bin = percentile_bins(pupil)
```

iii. The agent describes this as blink-clean interpolation followed by calibration-robust discretization.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Four per-session percentile cutoffs (20/40/60/80) define categorical labels 0–4.

ii.
```python
edges = np.percentile(values, [20, 40, 60, 80])
return np.searchsorted(edges, values, side="right").astype(np.int16)
```

iii. Per-session bins were intended to balance labels despite eye-camera/session scale differences.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil diameter is interpolated onto `ophys_timestamps`, then sliced with the same `[lo:hi]` interval as the event matrix.

ii.
```python
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
...
y[3] = pupil_bin[lo:hi]
```

iii. The agent cites synchronized acquisition and independent sample checks as its alignment justification.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. Outcome is derived from the trial table’s mutually exclusive `hit`, `miss`, `false_alarm`, and `correct_reject` boolean columns.

ii.
```python
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
if flags.sum() != 1:
    continue
outcome = int(np.flatnonzero(flags)[0])
```

iii. These are the SDK’s canonical outcomes for valid go/catch trials; malformed or ambiguous rows are rejected.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The true flag’s position in the fixed outcome list becomes code 0–3 and is repeated across every time bin of the trial.

ii.
```python
y[4] = outcome
```

iii. Repetition allows a static target to coexist in a single `(5, T)` output matrix with four time-varying targets while preserving its static meaning.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Source timestamps/values are filtered to finite pairs; interpolation requires at least two. Events, timestamps, outcomes, trial lengths, and final arrays are validated. Experiments with unavailable/insufficient eye data or fewer than two surviving trials are caught, logged, and skipped. `np.interp` fills destinations outside valid source support with endpoint values. Event/timestamp length mismatches are truncated to their common minimum.

ii.
```python
valid = np.isfinite(t_src) & np.isfinite(value_src)
if valid.sum() < 2: raise ValueError(...)
...
try:
    result = convert_experiment(...)
except Exception as exc:
    print(f"SKIP {experiment_id}: ...")
    continue
```

iii. The agent preferred excluding three pupil-deficient experiments over fabricating pupil labels, and used strict checks so bad data could not silently propagate. Endpoint extrapolation is an implicit consequence of `np.interp`.

## 9-a. What are the most time-consuming steps of the code?

i. AllenSDK experiment loading and materializing complete neural, stimulus, running, eye, and trial objects dominate; full conversion took about 798 seconds. Plotting is additional optional work.

ii.
```python
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
events = np.stack(ds.events["events"].to_numpy())
```

iii. The notes explicitly identify SDK construction/loading as the runtime bottleneck and report the measured full-run duration.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loops over stimulus presentations and trials could partly be vectorized (bulk boundary `searchsorted`, interval assignment, and outcome extraction). Image discovery also loops by image set, though its loads are intentional. Experiment loading remains inherently session-wise.

ii.
```python
for row in stim.itertuples(): ...
for trial_id, row in trials.iterrows(): ...
```

iii. The agent prioritized readable boundary logic; its notes say I/O dominates, so vectorizing these loops would provide a smaller gain.

## 9-c. What processing does the code repeat multiple times?

i. Each representative experiment loaded by `discover_images` is loaded again during conversion. For every experiment, `task_stimuli` scans/filters the presentation table, and output arrays are constructed for the entire session before trial slicing. Percentiles process full-session aligned behavior, including frames outside retained trials.

ii.
```python
ds = cache.get_behavior_ophys_experiment(int(group.index[0]))  # discovery
...
ds = cache.get_behavior_ophys_experiment(int(experiment_id))  # conversion
```

iii. The representative loads were used to establish a global image vocabulary before output encoding; caching mitigates disk/network cost, but object construction is still repeated.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes continuous speed and pupil for every session frame even though only trial slices and categorical bins are saved; the continuous arrays survive only for optional plots. It constructs full-session image/change vectors and retains a small `plot_rows` list even when plots are disabled. `show_processing` also creates diagnostic figures that the decoder does not consume.

ii.
```python
image, change, speed_bin, pupil_bin, speed, pupil = build_continuous_outputs(...)
plot_rows = []
if len(plot_rows) < 6:
    plot_rows.append(...)
```

iii. These intermediates simplify synchronized processing and diagnostics, but continuous behavior and plots are absent from `converted_data.pkl` and therefore discarded for downstream decoding.
