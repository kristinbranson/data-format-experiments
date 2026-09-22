# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache`, built with `from_s3_cache(cache_dir=/app/data)`. The experiment-level manifest `get_ophys_experiment_table()` is the single source of truth for what exists. Three masks are applied to that table: `project_code == "VisualBehavior"` (the single-plane VISp project named in the instructions), `behavior_type == "active_behavior"` (drops the 71 passive-replay experiments), and membership in the set of experiment IDs actually present in the local cache directory. The local-ID set is obtained by globbing the filenames of `data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/*.nwb` and parsing the integer out of the stem — the files are never opened, only their names are read, so the "no direct NWB access" constraint is respected while preventing the SDK from attempting S3 downloads of the ~1,700 released-but-absent experiments. The selection yields 168 experiments / 168 sessions / 37 mice. Each selected experiment is then loaded once with `cache.get_behavior_ophys_experiment(experiment_id)`, and every stream (`ophys_timestamps`, `events`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `trials`, `metadata`) is pulled off that object.

ii.
```python
def make_cache():
    return VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)


def local_experiment_ids() -> set[int]:
    folder = DATA_DIR / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {int(p.stem.rsplit("_", 1)[1]) for p in folder.glob("*.nwb")}


def selected_table(cache):
    table = cache.get_ophys_experiment_table()
    mask = (
        (table["project_code"] == "VisualBehavior")
        & (table["behavior_type"] == "active_behavior")
        & table.index.isin(local_experiment_ids())
    )
    return table.loc[mask].sort_index()
```
```python
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
ts = np.asarray(ds.ophys_timestamps, dtype=np.float64)
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
```

iii. From CONVERSION_NOTES Step 1/Step 4: the experiment table is the SDK's canonical manifest and already encodes release QC; `get_behavior_ophys_experiment` is the documented loader. The `project_code` filter is taken verbatim from the instruction ("Collect and convert data under the 'Visual Behavior' task"). The `behavior_type` filter is justified in Step 3/Step 4 as "Passive sessions cannot supply meaningful go/catch behavioral outcomes and are excluded from this task-specific conversion." The local-ID restriction is justified as necessary because "the AllenSDK metadata is unmodified and still lists every released experiment" while only a subset is cached, and the agent explicitly notes that it inspected only filenames, never file contents.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` strings of the experiments that were successfully converted (not of the full selected table). They are sorted lexicographically to give a deterministic `subjects` list, and `subject_idx` is the index of each converted session's mouse into that list. The result is 37 mice with 2–9 sessions each.

ii.
```python
kept = table.loc[kept_ids]
subjects = sorted(kept["mouse_id"].astype(str).unique())
subject_map = {x: i for i, x in enumerate(subjects)}
subject_idx = np.asarray([subject_map[str(x)] for x in kept["mouse_id"]], dtype=np.int64)
```

iii. CONVERSION_NOTES Step 5 mapping table: "experiment `mouse_id` → `subjects`, `subject_idx`; sorted unique strings and integer lookup; session order is sorted experiment ID." `mouse_id` is the SDK's unique animal identifier; building the list from the *kept* experiments guarantees that no subject index points at a mouse with zero sessions.

## 1-c. How are the data split into sessions?

i. One converted "session" = one AllenSDK *experiment* (one imaging plane). The agent verified through the cache that in the `VisualBehavior` project each `ophys_session_id` maps one-to-one onto exactly one `ophys_experiment_id` (239 experiments / 239 session IDs locally), because this project is single-plane, so no multi-plane merging is required. Sessions are ordered by sorted experiment ID. Of the 168 active sessions selected, 165 survive conversion (3 are dropped for unusable pupil data, see 8).

ii.
```python
return table.loc[mask].sort_index()          # one row == one session
...
ids = [int(x) for x in table.index]
for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
```
Session identity is recorded in metadata:
```python
info = {
    "ophys_experiment_id": int(experiment_id),
    "ophys_session_id": int(ds.metadata["ophys_session_id"]),
    "mouse_id": str(ds.metadata["mouse_id"]),
    ...
}
```

iii. CONVERSION_NOTES Step 4: "Session vs experiment — Session is continuous recording; experiment is one plane. VisualBehavior is single-plane, so all selected ophys session IDs map one-to-one to experiments. Treat each selected experiment as one target session; no multiplane merging is needed." Step 1 flagged that "different planes have distinct timestamp grids" and that merging should only happen "when grids genuinely align"; the 1:1 check made the question moot. The exclusion of the 71 passive sessions is justified separately (see 1-a/1-e).

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. Each kept trial is the half-open wall-clock interval `[start_time, stop_time)`, converted to ophys-frame indices with `np.searchsorted(..., side="left")` on the monotonically increasing `ophys_timestamps`. Trials are therefore variable length (mean 262 frames ≈ 8.4 s at 31 Hz, range 217–389), while the bin size stays fixed at one native ophys frame. Every stream (neural + all five outputs) is sliced with the same `[lo:hi]` index pair.

ii.
```python
trials = ds.trials
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
trials = trials.loc[keep]
for trial_id, row in trials.iterrows():
    ...
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
    if hi - lo < 2:
        continue
    ...
    x = events[:, lo:hi]
```

iii. CONVERSION_NOTES Step 5 Key Decision 2: "Use `[start_time, stop_time)` and `np.searchsorted` on monotonically increasing ophys timestamps. Half-open intervals prevent double-counting boundary frames and preserve the SDK trial definition." The full window is used (rather than a fixed window around the change) so the pre-change flashes and the post-change response window are both present, which is what makes time-varying image identity and image change meaningful. Step 10 Check 5 reports that half-open boundaries were verified against adjacent raw timestamps.

## 1-e. How are trials filtered based on quality controls?

i. Filters applied, in order:
- Session level: `project_code == VisualBehavior`, `behavior_type == active_behavior`, file locally present (1-a).
- Session level: reject the session if `ophys_timestamps` has <2 samples or is non-monotonic, if any calcium event value is non-finite, or if the pupil stream has <2 finite blink-clean samples (`interpolate_valid` raises). 3 sessions were dropped for the pupil reason.
- Trial level: `(go | catch) & ~aborted & ~auto_rewarded` — exactly the exclusion the instructions demand.
- Trial level: exactly one of the four outcome booleans must be true (`flags.sum() != 1 → skip`).
- Trial level: at least 2 ophys frames in the window (`hi - lo < 2 → skip`).
- Session level: at least 2 valid trials, otherwise the whole session is raised out and skipped.
No neuron-level filtering beyond the SDK default (see 2-c). Result: 42,470 trials of 43,387 eligible (the 917 missing are the trials inside the 3 pupil-invalid sessions).

ii.
```python
keep = (trials["go"] | trials["catch"]) & ~trials["aborted"] & ~trials["auto_rewarded"]
trials = trials.loc[keep]
for trial_id, row in trials.iterrows():
    flags = np.asarray([bool(row[x]) for x in OUTCOMES])
    if flags.sum() != 1:
        continue
    ...
    if hi - lo < 2:
        continue
    ...
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):
        continue
if len(neural_trials) < 2:
    raise ValueError(f"only {len(neural_trials)} valid trials")
```
```python
if len(ts) < 2 or np.any(np.diff(ts) <= 0):
    raise ValueError("invalid ophys timestamps")
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
```

iii. CONVERSION_NOTES Step 3 Curation: "Keep active task go and catch trials; exclude aborted and auto-rewarded trials as explicitly required. Trial outcomes are the four mutually exclusive hit/miss/false_alarm/correct_reject states." Step 5 Key Decision 8: "Reject any trial with nonfinite neural/outputs or fewer than two ophys frames. Retain sessions only with at least two trials" — the 2-trial floor is required by the target format ("There needs to be at least two trials within each session"). Step 2 notes that go+catch already excludes aborted rows in this SDK table, but the explicit predicates "remain necessary and are retained" as a defensive check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The `neural` matrix is the **unfiltered L0 detected calcium-event magnitude** array, `dataset.events["events"]`, one value per cell per ophys frame. The agent's first implementation used `dff_traces.dff`; this was changed during Critical Review 1 (Step 10) to `events`. The alternative `events["filtered_events"]` column (the SDK's causal half-Gaussian-smoothed version) was explicitly considered in Step 1 and not used.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
...
x = events[:, lo:hi]
neural_trials.append(x)
```

iii. CONVERSION_NOTES Step 4 / Step 10: "SDK exposes precomputed dF/F and L0 events … the paper explicitly uses events for all neural analyses. Use unfiltered `events` magnitudes at native ophys frames. Initial dF/F choice was corrected during Critical Review 1." This is directly supported by `methods.txt` line 208: "For all analysis of neural data we used the detected calcium events … This process produces, for each cell, a set of calcium events each with a time and magnitude", and line 179: "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f." The trajectory (step 91) records the switch: "reference fidelity takes precedence."

## 2-b. How is the `neural` data processed?

i. No processing beyond (a) stacking the per-cell object-column into a dense `(n_cells, T)` float32 array in `events`-table row order, (b) truncating `ophys_timestamps` and the event array to their common length, and (c) slicing per trial. No smoothing, no z-scoring, no baseline subtraction, no rebinning, no aggregation into image-presentation intervals. The raw sparse event magnitudes are kept at the native ophys frame rate. A consequence recorded in the logs is 1,729 trials in which every cell has a zero event value for the whole trial (the validator flags each as "all neural data is zero"), and full-dataset decoding accuracy that sits at 1.04–1.10× chance for four of the five outputs.

ii.
```python
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
```
(there is no other transformation of `events` anywhere in the file)

iii. CONVERSION_NOTES Step 3: "Reference paper neural analyses use detected calcium events to remove slow GCaMP dynamics and assign activity to 750-ms image intervals. … The task requires framewise activity, so the conversion retains native event arrays on ophys timestamps rather than aggregating into presentation vectors." Step 5 Key Decision 3: "Do not re-bin or interpolate neural traces. Each time bin is one native ophys frame." Step 10/12 keep the all-zero trials on the grounds that "Fixing them would require fabricating activity, deleting valid trials, or abandoning the reference event stream", and Step 12 concludes "Changing labels or smoothing events merely to inflate accuracy would violate the requested definitions or reference stream."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No explicit neuron-level filtering is written in the conversion script. The agent relies entirely on two SDK-level controls it verified in the reference code: (1) only release-QC-passing experiments appear in `get_ophys_experiment_table()`, and (2) `get_behavior_ophys_experiment()` constructs `BehaviorOphysExperiment` with `exclude_invalid_rois=True` by default, so motion-border ROIs, classifier-rejected non-cells, duplicates, unions and bad-demixing ROIs are already gone. The only cell-count-affecting action is session-level rejection (3 pupil-invalid sessions remove 276 cells). Final count: 28,821 session-neurons, mean 174.67/session, range 6–666, all VISp.

ii. There is no neuron-filtering code; the relevant line is simply the SDK loader whose defaults do the filtering:
```python
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
events = np.stack(ds.events["events"].to_numpy()).astype(np.float32, copy=False)
```
plus a whole-session finiteness guard:
```python
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
```

iii. CONVERSION_NOTES Step 1: "Public cache experiment entries have passed Allen release QC; the SDK `BehaviorOphysExperiment` normally excludes invalid ROIs (`exclude_invalid_rois=True`). We will retain the cells exposed by the cache object rather than invent an additional cell-quality threshold." Step 3 enumerates the pipeline's own rules (motion border, classifier labels, >70% overlap duplicates, unions, bad demixing, invalid matching) as evidence that adding a hand-made threshold would deviate from the reference.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the native synchronized ophys frame clock: `ophys_timestamps` is the master time base and every other stream is resampled onto it, exactly as the instruction "Temporally align based on ophys timestamp" requires. Trial boundaries are the SDK trial's `start_time`/`stop_time` converted to frame indices with `searchsorted(..., side="left")`, giving the half-open frame window `[lo, hi)`; the first frame of every trial is the first ophys frame at or after `start_time`. Because trials are the full SDK trial window rather than a fixed offset from a point event, `off_start` and `off_end` are recorded as `None`.

ii.
```python
lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
...
x = events[:, lo:hi]
y[0] = image[lo:hi]; y[1] = change[lo:hi]; y[2] = speed_bin[lo:hi]; y[3] = pupil_bin[lo:hi]
```
```python
"temporal_alignment_event": "Native synchronized ophys frame timestamps; each trial is the half-open SDK interval [start_time, stop_time).",
"off_start": None,
"off_end": None,
```

iii. CONVERSION_NOTES Step 3/4: "All acquisition clocks were recorded on one 100-kHz synchronization board. SDK-exposed behavior/ophys timestamps are synchronized; conversion should resample behavior/stimulus streams at each exact ophys timestamp." Step 5 Key Decision 2 justifies the half-open convention as preventing double-counted boundary frames. Step 10 Check 5 reports independent verification "that every trial's first/last converted frame lies within `[start_time, stop_time)`, adjacent source frames outside do not."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. One time bin = one native ophys frame. The `VisualBehavior` project is single-plane 31 Hz imaging, so the bin is ≈32.26 ms. **No** rebinning, resampling, or interpolation is applied to the neural data; only the behavioral/stimulus streams are resampled *onto* the neural grid. `metadata["time_bin_size"]` is the median across sessions of each session's median inter-frame interval, in ms; each session's own value is also stored in `metadata["session_info"]`. Trial lengths vary (217–389 bins) but the bin duration is constant everywhere.

ii.
```python
"median_frame_interval_ms": float(np.median(np.diff(ts)) * 1000),   # per session
...
median_bin = float(np.median([x["median_frame_interval_ms"] for x in infos]))
...
"time_bin_size": median_bin,
```

iii. CONVERSION_NOTES Step 3: "Neural data time bin: Native microscope frames: 31 Hz single-plane, 11 Hz per plane multi-plane." Step 5 Key Decision 3: "Do not re-bin or interpolate neural traces. Each time bin is one native ophys frame (~32.26 ms at 31 Hz); variable trial lengths are allowed while bin duration stays fixed." Step 4 adds that the metadata "will report empirical median frame interval in ms (and session-level values)" rather than a nominal rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. Image identity is derived from `dataset.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection`, using the `image_name`, `start_time`, `end_time` and `omitted` columns. It is **not** taken from the trials table. A dedicated `"gray"` class covers every frame that is not inside a non-omitted image flash — i.e. the 500 ms inter-stimulus gray and the 5% omitted flashes. The global vocabulary is `["gray"] + sorted(all image names across both image sets)` = 17 classes; it is discovered before conversion by loading one representative experiment per `image_set`.

ii.
```python
def task_stimuli(dataset):
    stim = dataset.stimulus_presentations
    names = stim["stimulus_block_name"].fillna("").astype(str)
    return stim.loc[names.str.contains("change_detection", case=False)].sort_values("start_time")


def discover_images(cache, table) -> list[str]:
    found: set[str] = set()
    for _, group in table.groupby("image_set", dropna=False):
        ds = cache.get_behavior_ophys_experiment(int(group.index[0]))
        stim = task_stimuli(ds)
        vals = stim.loc[~stim["omitted"].fillna(False), "image_name"].dropna().astype(str)
        found.update(v for v in vals.unique() if v.lower() not in {"omitted", "nan"})
    return ["gray"] + sorted(found)
```

iii. CONVERSION_NOTES Step 4: "Current SDK table can contain several blocks; warning requires change-detection selection … Restrict image identity/change construction to block names containing `change_detection`; gray/omitted intervals receive a dedicated gray class." Step 5 Key Decision 6: "Build a deterministic global vocabulary before conversion so identical integer labels have identical meaning across sessions. `gray` is an explicit class because most frames are the inter-stimulus gray screen and instructions ask for the identity of the image on the non-grey screen." The stimulus table (rather than the trials table) is used because it is the frame-accurate record of what was actually on the monitor, including omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A full-session integer label vector is initialised to the `gray` code and then, for each non-omitted change-detection presentation, the half-open frame span `[searchsorted(ts, start_time), searchsorted(ts, end_time))` is overwritten with that image's global code. The per-trial row is a plain slice of that vector. Because a flash is 250 ms of a 750 ms cycle, ≈1/3 of frames carry an image code and 0.669 of frames end up `gray` — matching the duty-cycle prediction the agent wrote down as a sanity check. The 16 images each occupy ≈0.020–0.021 of frames.

ii.
```python
image = np.full(n, image_to_idx["gray"], dtype=np.int16)
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
    omitted = bool(row.omitted) if not np.isnan(row.omitted) else False
    name = str(row.image_name)
    if not omitted and name in image_to_idx and end > start:
        image[start:end] = image_to_idx[name]
```
```python
"output_values": [images, ...]      # ['gray', 'im000', ..., 'im106']
```

iii. CONVERSION_NOTES Step 5 mapping: "Interval lookup at each ophys frame. Non-omitted image presentation gets its image class; gray ISI and omitted presentation get `gray`." Step 9 consistency row: "Image classes … gray + 16 images … gray 0.669, each image ~0.020–0.021 … Yes; 250/750 duty cycle predicts image total ~1/3." Step 10 Check 4 reports an independent reconstruction of the image intervals for three trials passing `np.allclose`.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is built as a full-session vector on the *same* `ophys_timestamps` array that indexes the neural matrix, then sliced with the identical `[lo:hi]` index pair, so alignment is exact by construction — there is no separate resampling step that could drift. Presentation boundaries use the same `side="left"` searchsorted convention as the trial boundaries, so a frame is labelled with an image iff its timestamp lies in `[flash_start, flash_end)`.

ii.
```python
image, change, speed_bin, pupil_bin, speed, pupil = build_continuous_outputs(ds, ts, image_to_idx)
...
y = np.empty((5, hi - lo), dtype=np.int16)
y[0] = image[lo:hi]
...
x = events[:, lo:hi]
```

iii. CONVERSION_NOTES Step 4: "Sample all labels at native ophys timestamps. Use interval lookup for images/change impulses and interpolation for speed/pupil." Step 7 plot review: "Final plots show … 250-ms image/500-ms gray cadence, one-frame impulses exactly at identity transitions … No temporal discontinuity or misalignment was visible."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. From the `is_change` and `start_time` columns of the same change-detection-block `stimulus_presentations` table. The SDK's `is_change` is defined as "the first presentation of a new `image_name`", with omitted stimuli ignored and the session's first stimulus ignored — so it is true only for *real* identity changes, and is false on catch (sham-change) trials. The trials table's `change_time`/`go` columns are **not** used for this output.

ii.
```python
stim = task_stimuli(dataset)
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    ...
    if bool(row.is_change) and start < n:
        change[start] = 1
```

iii. CONVERSION_NOTES Step 5 mapping: "`stimulus_presentations.is_change`, start time → `output[1]` image change … Catch sham changes and omissions are zero; exactly one bin per real identity change." The agent read the instruction "Have value of 1 right after a change in image identity" as being about the identity change itself, and `is_change` is the SDK's own encoding of exactly that predicate, so it avoids having to re-derive the change from image names.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A single-frame impulse. A zero vector over the whole session gets a `1` written at exactly one index per real change — the first ophys frame at or after the change flash onset — and nothing else. The impulse is therefore ≈32 ms wide, it is *not* extended over the flash, the following gray, or the response window. Across the full dataset this makes the positive class 0.003 of all time bins (no_change 0.997). The agent noted in Step 12 that the resulting decoder accuracy is 0.5197 vs 0.5 chance (1.04×), below the instructions' 1.5×-chance investigation threshold, and decided not to change the definition.

ii.
```python
change = np.zeros(n, dtype=np.int16)
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    ...
    if bool(row.is_change) and start < n:
        change[start] = 1
```
```python
y[1] = change[lo:hi]
```

iii. CONVERSION_NOTES Step 4: "image-change impulses at the first ophys frame at/after each true presentation change". Step 12: "Image-change impulses are deliberately one 31-Hz frame, only 0.3% of samples, exactly as requested. Same-frame event inference has limited calcium response latency information; 0.5197 nonetheless falls in the low end of the paper's plotted change-decoder range despite a very different balanced, framewise task." The agent explicitly declined to widen the window: "Changing labels or smoothing events merely to inflate accuracy would violate the requested definitions or reference stream."

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is needed: the source `is_change` is already boolean, so the output is the two-class variable `{0: "no_change", 1: "change"}` stored as int16. The only "categorisation" choice is the width of the positive class, which is one bin (see 4-b).

ii.
```python
change = np.zeros(n, dtype=np.int16)
...
change[start] = 1
```
```python
"output_names": [..., "image_change", ...],
"output_values": [images, ["no_change", "change"], ...],
```

iii. CONVERSION_NOTES Step 5 mapping: "Binary impulse on first selected ophys frame at/after each true image-change onset; all other frames zero." The instruction itself specifies a binary variable, so no threshold was invented.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: the impulse vector is built on the session's `ophys_timestamps` and sliced with the same `[lo:hi]` pair as the neural matrix. The impulse index is `searchsorted(ts, change_flash_start, side="left")`, i.e. the first neural frame that is at or after the physical change — which is why the label is "right after" the change rather than before it.

ii.
```python
y[1] = change[lo:hi]
x = events[:, lo:hi]
```

iii. CONVERSION_NOTES Step 5 planned check: "every change impulse is the nearest ophys frame at or after presentation start", reported as passing in Step 10 Check 5. Step 7 plot review confirms "one-frame impulses exactly at identity transitions".

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `dataset.running_speed`, i.e. the SDK's *filtered* linear running speed in cm/s (columns `timestamps` and `speed`), sampled at ~60 Hz. The unfiltered `raw_running_speed` was explicitly rejected.

ii.
```python
run = dataset.running_speed
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)
```

iii. CONVERSION_NOTES Step 1: "`running_speed` is the reference filtered stream (not `raw_running_speed`)." Step 3 documents what that filtering is, from the whitepaper: "Running processing unwraps encoder voltage, removes >5.1-V artifacts, corrects wraps, suppresses wrap transients within ±0.25 s, removes z-score ≥10 transients, converts to cm/s, and applies a 10-Hz low-pass Butterworth filter. This is already embodied in SDK `running_speed`."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite source samples are dropped, the remaining samples are sorted by time, and the speed is linearly interpolated onto every ophys timestamp with `np.interp` (which clamps to the nearest endpoint outside the sampled support rather than producing NaN). The resulting per-frame speed is then quantised (5-c). Negative speeds are kept. No smoothing beyond the SDK's own 10 Hz filter, and no per-trial re-processing — the whole session is interpolated once, then sliced.

ii.
```python
def interpolate_valid(t_src, value_src, t_dst):
    t_src = np.asarray(t_src, dtype=np.float64)
    value_src = np.asarray(value_src, dtype=np.float64)
    valid = np.isfinite(t_src) & np.isfinite(value_src)
    if valid.sum() < 2:
        raise ValueError("fewer than two finite source samples")
    order = np.argsort(t_src[valid])
    return np.interp(t_dst, t_src[valid][order], value_src[valid][order])
```

iii. CONVERSION_NOTES Step 5 mapping: "Linear interpolation to ophys frames … Negative speeds retained because they are valid filtered encoder estimates." Step 4: behavioral and ophys clocks are hardware-synchronised on the 100 kHz board, so direct interpolation onto the ophys grid is legitimate.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-count percentile bins (quintiles), computed **per session** from the full-session interpolated speed vector (all frames, not only the frames inside kept trials), using the 20/40/60/80 percentiles as edges and `np.searchsorted(..., side="right")` to assign labels 0–4. The whole-dataset realised distribution across kept trials is [0.188, 0.197, 0.202, 0.208, 0.205] — close to uniform, the small deviation coming from the fact that edges are fitted on all frames but only trial frames are exported.

ii.
```python
def percentile_bins(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("nonfinite or empty values supplied to percentile_bins")
    edges = np.percentile(values, [20, 40, 60, 80])
    return np.searchsorted(edges, values, side="right").astype(np.int16)
```
```python
speed_bin = percentile_bins(speed)
...
"quintile_definition": "per-session 20/40/60/80 percentiles on aligned continuous samples",
```

iii. CONVERSION_NOTES Step 5 Key Decision 5: "Compute percentile boundaries from all finite aligned samples in each session before trial slicing … Per-session binning controls scale/calibration differences and makes approximately balanced classes." Note that the same decision text also promises "deterministic rank-based assignment where repeated values would make percentile edges non-unique" and (Step 4) to "collapse repeated edges safely" — neither of those mechanisms exists in `percentile_bins`, which uses plain percentile edges with no tie handling.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. The speed is interpolated directly *at* the ophys timestamps before any trial segmentation, so the binned vector shares the neural array's time axis element-for-element; the trial row is the same `[lo:hi]` slice. No per-trial interpolation or offset is applied.

ii.
```python
speed = interpolate_valid(run["timestamps"], run["speed"], timestamps)   # timestamps == ophys ts
speed_bin = percentile_bins(speed)
...
y[2] = speed_bin[lo:hi]
x = events[:, lo:hi]
```

iii. CONVERSION_NOTES Step 4: "Timing/rates — Ophys timestamps are authoritative; behavior streams carry synchronized timestamps … Sample all labels at native ophys timestamps." Step 10 Check 4 reports an independent recomputation of the speed interpolation and quintiles for three trials passing `np.allclose`.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `dataset.eye_tracking`, specifically the `pupil_area` column (plus `timestamps`). The area is converted to an equivalent circular diameter `2*sqrt(area/pi)`. Blink frames are handled implicitly: the SDK's `filter_on_blinks` already sets `pupil_area` (and `pupil_width`/`pupil_height`) to NaN wherever `likely_blink` is true, and `interpolate_valid` drops all non-finite samples. `pupil_width` and `pupil_height` were considered and rejected in favour of the area-derived diameter.

ii.
```python
eye = dataset.eye_tracking
if eye is None:
    raise ValueError("eye tracking unavailable")
area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
```

iii. CONVERSION_NOTES Step 4: "SDK offers pupil area, width, height; invalid/blink-derived values are NaN … Compute diameter-equivalent `2*sqrt(pupil_area/pi)` from valid area, interpolate only within valid support, then percentile-bin. This is geometrically defined and uses both ellipse axes rather than choosing width or height." Step 1: "Eye tracking derived pupil values are NaN on likely blinks."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Clamp negative areas to 0, convert area→diameter, drop non-finite (= blink / outlier) samples, sort by time, linearly interpolate onto the ophys timestamps with `np.interp` (flat extrapolation at the edges), then quantise (6-c). A session whose pupil stream has fewer than two finite samples raises and is dropped entirely rather than being given fabricated labels — this removed 3 sessions (795953296, 806456687, 833631914).

ii.
```python
area = eye["pupil_area"].to_numpy(dtype=np.float64)
diameter = 2.0 * np.sqrt(np.maximum(area, 0.0) / np.pi)
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
pupil_bin = percentile_bins(pupil)
```
```python
valid = np.isfinite(t_src) & np.isfinite(value_src)
if valid.sum() < 2:
    raise ValueError("fewer than two finite source samples")
```

iii. CONVERSION_NOTES Step 5 mapping: "Drop nonfinite/blink samples; diameter-equivalent `2*sqrt(area/pi)`; linear interpolation to ophys frames; per-session percentile quintiles 0–4. Exclude session only if insufficient valid pupil support; do not substitute raw blink values." Step 10: "Three sessions have fewer than two finite blink-clean samples and are excluded instead of receiving fabricated labels."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Exactly the same `percentile_bins` routine as running speed: per-session 20/40/60/80 percentiles of the full-session interpolated diameter, labels 0–4. The realised full-dataset distribution over kept trials is [0.160, 0.207, 0.229, 0.237, 0.166], i.e. noticeably less uniform than the running quintiles because the edges are fitted on all session frames (including the long pre/post-trial and inter-trial periods) while only trial frames are exported.

ii.
```python
pupil_bin = percentile_bins(pupil)     # same function as for speed
...
"output_values": [..., ["q1_lowest", "q2", "q3", "q4", "q5_highest"], ...],
```

iii. CONVERSION_NOTES Step 4: "Task explicitly requires five equal percentile bins. Fit per-session quintile edges from eligible sampled timepoints … Per-session binning controls scale/calibration differences and makes approximately balanced classes." Step 9 comments on the resulting distribution: "Reasonable shift after selecting trial epochs." As with running speed, the promised rank-based tie handling is not present in the code.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same mechanism as running speed — interpolated at the ophys timestamps for the whole session before segmentation, then sliced with the identical `[lo:hi]` indices, so it is frame-exact with the neural data by construction.

ii.
```python
pupil = interpolate_valid(eye["timestamps"], diameter, timestamps)
pupil_bin = percentile_bins(pupil)
...
y[3] = pupil_bin[lo:hi]
x = events[:, lo:hi]
```

iii. CONVERSION_NOTES Step 3/4: all cameras and the microscope are logged on the single 100 kHz sync board, so the SDK's eye-tracking timestamps are already in the ophys clock and can be interpolated directly. Step 10 Check 4 verified the pupil row of three trials against an independent reconstruction with `np.allclose`.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK `trials` table: `hit`, `miss`, `false_alarm`, `correct_reject`, in that fixed order. A trial is only kept if exactly one of the four is true, so there is no "other" fallback class.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
...
flags = np.asarray([bool(row[x]) for x in OUTCOMES])
if flags.sum() != 1:
    continue
outcome = int(np.flatnonzero(flags)[0])
```

iii. CONVERSION_NOTES Step 1/Step 3: "Trials are authoritatively defined by the SDK `trials` table … outcomes are hit/miss/false_alarm/correct_reject", "Trial outcomes are the four mutually exclusive hit/miss/false_alarm/correct_reject states." Step 5 mapping: "Keep only rows with exactly one valid outcome" — the exactly-one check is both the encoding and a consistency assertion on the SDK table.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The index of the true flag (0–3) is broadcast across all `T` time bins of the trial so that the static per-trial label can live in the same `(5, T)` int16 matrix as the four time-varying outputs. `output_values[4]` records the name of each code. Full-dataset, time-bin-weighted distribution: hit 0.316, miss 0.559, false_alarm 0.018, correct_reject 0.107.

ii.
```python
outcome = int(np.flatnonzero(flags)[0])
y = np.empty((5, hi - lo), dtype=np.int16)
...
y[4] = outcome
```
```python
"output_names": [..., "trial_outcome"],
"output_values": [..., OUTCOMES],
```

iii. CONVERSION_NOTES Step 5 Key Decision 4: "Repeat outcome across time. A single target array cannot mix vector and matrix dimensions, and repetition preserves its explicitly static per-trial meaning while satisfying validator/trainer shape rules." Step 10 Check 5 confirmed the outcome row is constant within every trial.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- **Stream-length mismatch** between `ophys_timestamps` and the event array: both truncated to the common length `min(ts.size, events.shape[1])`.
- **Bad ophys clock**: session rejected if <2 timestamps or non-monotonic.
- **Non-finite calcium events**: session rejected.
- **Missing / blink pupil samples**: NaN samples are dropped before interpolation; `np.interp` flat-extrapolates outside the valid support so no NaN can reach the output. If fewer than two finite samples exist, the session is rejected (3 sessions, 917 trials, 276 cells) rather than being given invented labels.
- **Missing running samples**: same `interpolate_valid` path.
- **Missing/ambiguous outcome flags**: trial skipped unless exactly one flag is set.
- **Degenerate trial windows**: trials with <2 ophys frames skipped; sessions with <2 valid trials rejected.
- **Any other per-session failure**: caught at the top level, logged as `SKIP <id>: <error>`, and the run continues.
- **All-zero event trials** (1,729 of them) are *not* treated as errors — they are kept and documented as genuine sparse-event observations.

ii.
```python
n = min(ts.size, events.shape[1])
ts, events = ts[:n], events[:, :n]
if len(ts) < 2 or np.any(np.diff(ts) <= 0):
    raise ValueError("invalid ophys timestamps")
if not np.all(np.isfinite(events)):
    raise ValueError("nonfinite calcium events")
```
```python
valid = np.isfinite(t_src) & np.isfinite(value_src)
if valid.sum() < 2:
    raise ValueError("fewer than two finite source samples")
```
```python
try:
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)
except Exception as exc:
    print(f"SKIP {experiment_id}: {type(exc).__name__}: {exc}", flush=True)
    continue
```

iii. CONVERSION_NOTES Step 5 Key Decision 8: "Interpolate only from SDK-valid pupil samples; `np.interp` uses nearest valid endpoint outside support. Reject a session if it lacks at least two finite valid pupil samples. Reject any trial with nonfinite neural/outputs or fewer than two ophys frames. Retain sessions only with at least two trials." Step 9/10: the 3 excluded sessions are named and their cost (917 trials, 276 cells) quantified; the all-zero warnings are retained because "Fixing them would require fabricating activity, deleting valid trials, or abandoning the reference event stream."

## 9-a. What are the most time-consuming steps of the code?

i. Data loading dominates. `cache.get_behavior_ophys_experiment()` (NWB deserialisation through the SDK) plus materialising the `events`, `stimulus_presentations`, `running_speed` and `eye_tracking` tables accounts for essentially all of the 3–5 s per session; the full run was 798.33 s (13.3 min) for 168 sessions, i.e. ≈4.75 s/session. Secondary costs: the Python-level loop over ~14k `stimulus_presentations` rows per session, and pickling the 8.4 GB output (the dense float32 event arrays are the payload). Per-session and total wall-clock are printed so the bottleneck is visible in the logs.

ii.
```python
started = time.perf_counter()
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
...
elapsed = time.perf_counter() - started
...
print(f"[{k}/{len(ids)}] id={experiment_id} neurons={info['n_neurons']} "
      f"trials={info['n_trials']} time={info['seconds']:.2f}s", flush=True)
...
print(f"WROTE {out} ... elapsed={time.perf_counter()-total_start:.2f}s", flush=True)
```

iii. CONVERSION_NOTES Step 6: "SDK construction/loading dominates runtime; naive repeated loading and holding continuous arrays would increase I/O and memory." Step 7 run-time table estimates "4.4–4.5 s for sample sessions → approximately 12.6 min for 168 sessions", and Step 9 reports the realised 13.31 min as within the 15-minute budget.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. Two Python loops remain:
- `build_continuous_outputs`: `for row in stim.itertuples()` over every change-detection stimulus presentation (~13,800 rows/session, ~2.3M rows overall). This is fully vectorizable — `np.searchsorted` accepts arrays, so all flash start/end indices could be computed in one call, the change impulses written with fancy indexing, and the image spans filled with a `np.repeat`/interval-fill or a single `searchsorted` lookup of each frame into the flash-start array.
- `convert_experiment`: `for trial_id, row in trials.iterrows()` over kept trials (~260/session). `iterrows()` is the slowest pandas iteration idiom; the boundary indices and outcome codes could be computed as whole columns, leaving only the array slicing in the loop.
Neither was vectorized. The agent's own efficiency notes identify only I/O as a cost and do not mention either loop.

ii.
```python
for row in stim.itertuples():
    start = int(np.searchsorted(timestamps, float(row.start_time), side="left"))
    end = int(np.searchsorted(timestamps, float(row.end_time), side="left"))
    ...
```
```python
for trial_id, row in trials.iterrows():
    flags = np.asarray([bool(row[x]) for x in OUTCOMES])
    ...
    lo = int(np.searchsorted(ts, float(row.start_time), side="left"))
    hi = int(np.searchsorted(ts, float(row.stop_time), side="left"))
```

iii. CONVERSION_NOTES Step 6 lists only "Code inefficiencies identified: SDK construction/loading dominates runtime". The implicit justification for leaving the loops scalar is that the conversion already fits inside the instructions' 15-minute budget, so the loops are not the bottleneck; that conclusion is correct but the loops themselves are never named.

## 9-c. What processing does the code repeat multiple times?

i. The main repetition is in `discover_images`: one representative experiment per `image_set` (2 image sets → 2 experiments) is fully loaded through `get_behavior_ophys_experiment` and its stimulus table filtered, purely to collect the image-name vocabulary — and those same experiments are then loaded again from scratch in the main conversion loop. `task_stimuli()` (a string filter over the whole presentations table) is likewise executed twice for those experiments. Smaller repetitions: `np.searchsorted` is called once per stimulus row and twice per trial on the same `ts` array instead of once per array; `np.percentile`/`searchsorted` run the same code path twice per session (speed and pupil). Sample mode additionally pulls `get_ophys_cells_table()`, which is unused in full mode.

ii.
```python
def discover_images(cache, table) -> list[str]:
    for _, group in table.groupby("image_set", dropna=False):
        ds = cache.get_behavior_ophys_experiment(int(group.index[0]))   # loaded again later
        stim = task_stimuli(ds)
        ...
```
```python
for k, experiment_id in enumerate(ids, 1):
    result = convert_experiment(cache, experiment_id, image_to_idx, ...)  # reloads those ids
```

iii. CONVERSION_NOTES Step 6 presents the vocabulary pass as a speed-up: "Representative sessions discover the global image vocabulary once. … Avoids redundant full-session reads." The justification for a *global* vocabulary is Step 5 Key Decision 6 ("identical integer labels have identical meaning across sessions"); the duplicate load of two sessions (~8 s, ~1% of runtime) is not acknowledged as a repeat.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several small items:
- `build_continuous_outputs` computes and returns the **continuous** `speed` and `pupil` vectors in addition to the binned versions; only the bins are exported — the continuous arrays are used solely by `plot_processing`, which runs for at most 2 sessions in `--show-processing` mode.
- `plot_rows` (trial windows, outcomes, ids for the diagnostic figure) is accumulated on every trial of every session even when `show_processing` is False.
- `np.all(np.isfinite(y))` is evaluated on `y`, an `int16` array, where it is unconditionally True; only the check on `x` can ever fire.
- `np.stack` materialises a dense float32 `(n_cells, T)` array from the *sparse* event table, and the trial slices are stored densely, producing an 8.4 GB pickle for data that is overwhelmingly zeros (1,729 trials are all-zero).
- Per-trial `np.empty((0, hi - lo))` input arrays are allocated for a decoder with no inputs (zero payload, but one object per trial).
- `info` carries `n_native_trials`, `outcome_counts` and per-session `median_frame_interval_ms` into `metadata["session_info"]` for all 165 sessions; only the frame interval feeds a downstream field, the rest is documentation.
- In `--sample` mode `get_ophys_cells_table()` is fetched only to rank sessions by cell count.

ii.
```python
return image, change, speed_bin, pupil_bin, speed, pupil   # speed/pupil only used for plots
```
```python
if len(plot_rows) < 6:
    plot_rows.append((lo, hi, outcome, int(trial_id)))     # built even when not plotting
```
```python
if not np.all(np.isfinite(y)) or not np.all(np.isfinite(x)):   # y is int16 -> always finite
    continue
```
```python
"session_info": infos,
```

iii. CONVERSION_NOTES Step 6 frames the memory strategy positively: "Neural data remains float32 and is sliced as views where possible; empty decoder inputs allocate no payload; outputs use int16. Continuous arrays are released after each serialized in-memory session result is produced." The continuous speed/pupil are kept deliberately so that `--show-processing` can "visually convince the user that every step of the conversion is correct" (a Step 6 requirement); the redundant finiteness check and the always-on `plot_rows` accumulation are not discussed anywhere.
