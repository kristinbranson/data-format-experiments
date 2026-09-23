# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/app/data')`. The agent enumerates which assets are actually present locally by globbing the filenames of the cached NWB files (filenames only — the files are never opened directly), intersects those ids with `cache.get_ophys_experiment_table()`, and then keeps only rows whose `behavior_type == 'active_behavior'`. Each selected row is loaded with `cache.get_behavior_ophys_experiment(experiment_id)`, and everything else (`ophys_timestamps`, `events`, `trials`, `stimulus_presentations`, `running_speed`, `eye_tracking`, `metadata`) is read off that SDK object. 284 experiments are cached; 202 are active; 199 survive conversion. Loading is parallelised over 4 processes, each constructing its own independent cache object. Note that the agent does **not** filter on `project_code`, so both `VisualBehavior` (single-plane) and `VisualBehaviorMultiscope` (multi-plane) experiments are included.

ii.
```python
CACHE_DIR = APP / "data"
EXPERIMENT_DIR = RELEASE_DIR / "behavior_ophys_experiments"

def cached_experiment_ids() -> list[int]:
    """Enumerate cache assets without reading their contents."""
    prefix = "behavior_ophys_experiment_"
    return sorted(int(p.stem.removeprefix(prefix)) for p in EXPERIMENT_DIR.glob(f"{prefix}*.nwb"))

def active_experiment_table(cache) -> pd.DataFrame:
    table = cache.get_ophys_experiment_table()
    ids = [x for x in cached_experiment_ids() if x in table.index]
    selected = table.loc[ids]
    selected = selected[selected["behavior_type"].eq("active_behavior")]
    return selected.sort_index()
```
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
table = active_experiment_table(cache)
...
obj = cache.get_behavior_ophys_experiment(int(experiment_id))
```
```python
def convert_experiment_worker(item):
    """Independent SDK cache/session construction for safe process parallelism."""
    experiment_id, meta_dict = item
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=str(CACHE_DIR))
    session, exclusions = convert_experiment(cache, experiment_id, pd.Series(meta_dict), keep_diagnostic=False)
```

iii. From CONVERSION_NOTES.md Step 1/Step 5: "The cache is the required entrypoint; detailed session objects must be obtained using `get_behavior_ophys_experiment`, never by opening NWB files directly." The filename glob exists so that only locally cached assets are requested ("These were enumerated by filename only; their contents were accessed exclusively through `VisualBehaviorOphysProjectCache`"), i.e. to avoid triggering S3 downloads of the ~1,936 released experiments that are not in the local cache. Passive experiments are dropped because "passive viewing has no behavioral trial outcomes and the paper explicitly did not analyze passive sessions" (Step 4 discrepancy table). Parallelism is justified in Step 6/7 as the fix for the dominant SDK-loading bottleneck.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values of the retained experiments, sorted as strings. Each converted session (= one imaging plane/experiment) carries its `mouse_id`, and `subject_idx` is the index of that mouse in the sorted subject list. Result: 38 mice.

ii.
```python
"mouse_id": str(meta.mouse_id),
...
subjects = sorted({s["mouse_id"] for s in sessions})
subject_map = {x: i for i, x in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in sessions], dtype=np.int64),
```

iii. Step 5 variable-mapping table: "`mouse_id` → `subjects`, `subject_idx`; Sorted string IDs and per-experiment index; One target session per experiment/plane." The agent cross-checked the count against the cache metadata (38 mice in the cached active subset) in Steps 2/9/10.

## 1-c. How are the data split into sessions?

i. One target "session" = one **ophys experiment (imaging plane)**, not one `ophys_session_id`. For single-plane `VisualBehavior` experiments the two are identical, but for the cached `VisualBehaviorMultiscope` experiments the same behavioural session appears once per simultaneously-imaged plane (visible in the output as one mouse, 457841, contributing 34 "sessions" with repeated trial counts 209×7, 287×7, 309×7, …). The true `ophys_session_id` / `behavior_session_id` are preserved in `metadata['session_info']`. Final count: 199 sessions.

ii.
```python
result = {
    "experiment_id": int(experiment_id),
    "ophys_session_id": int(meta.ophys_session_id),
    "behavior_session_id": int(meta.behavior_session_id),
    ...
}
```
```python
for s in sessions:                      # one entry per experiment = one target session
    sn, si, so = [], [], []
    for r in s["records"]:
        ...
    neural.append(sn); inputs.append(si); outputs.append(so)
    region_indices.append(np.full(len(s["cell_ids"]), region_map[s["region"]], dtype=np.int64))
```

iii. Step 4 discrepancy table: "Each target session will be one ophys experiment. A neural matrix cannot combine different frame clocks/regions/planes, and the paper's unit of decoding is an imaging plane. Shared behavioral trials are therefore intentionally represented once per plane." The whitepaper distinction (session = one continuous recording, experiment = one imaging plane) and the paper's per-imaging-plane decoding are cited as support.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table. Each retained trial spans its native `start_time` → `stop_time` (variable length, 7.0–12.5 s). Within that window the agent lays down a fixed 100 ms grid anchored at `start_time`, keeping only whole bins (`nbins = floor((stop-start)/0.1)`); trailing partial time is discarded. All modalities use those same bin edges/centres, so T varies per trial (70–125 bins, mean 84.2).

ii.
```python
DT = 0.100  # seconds
...
for trial_id, row in eligible.iterrows():
    start, stop = float(row.start_time), float(row.stop_time)
    nbins = int(np.floor((stop - start) / DT + 1e-9))
    if nbins < 2:
        dropped_short += 1
        continue
    edges = start + np.arange(nbins + 1, dtype=float) * DT
    centers = edges[:-1] + DT / 2.0
```

iii. Step 5 decision 4: "Use native trial `start_time` to `stop_time`; choose complete 100 ms bins whose centers lie in the interval. Trial arrays may have different T, which validator/model explicitly supports. `off_start` and `off_end` are `None` because alignment is to absolute ophys timestamps/trial boundaries rather than a fixed event window." Step 10 edge-case audit records "all bins are half-open `[edge_i,edge_{i+1})`; only complete bins fit inside trial bounds; output uses centers."

## 1-e. How are trials filtered based on quality controls?

i. Layered filtering:
* **Experiment level**: active behaviour only; experiment dropped if `eye_tracking` is absent/empty or has no `pupil_area` column (3 experiments: 795953296, 806456687, 833631914); dropped if <2 surviving trials or 0 cells.
* **Trial level (table)**: `(go | catch) & ~aborted & ~auto_rewarded`, exactly one of `hit/miss/false_alarm/correct_reject` true, finite `start_time`/`stop_time` with `stop > start`.
* **Trial level (data quality)**: <2 complete 100 ms bins → drop; running or pupil interpolation not fully covered by valid samples → drop; stimulus-consistency assertion (a `go` trial must contain exactly one `is_change` presentation, a `catch` trial exactly zero) → drop.
* Every exclusion is counted and stored in `metadata['session_info'][i]['exclusions']` / `metadata['all_experiment_exclusions']`.
* Trials whose inferred-event matrix is all zero (4.90%) are deliberately **kept**.

ii.
```python
def valid_trials(trials):
    flags = trials[OUTCOME_COLUMNS].fillna(False).astype(bool)
    contingent = trials["go"].fillna(False).astype(bool) | trials["catch"].fillna(False).astype(bool)
    non_aborted = ~trials["aborted"].fillna(False).astype(bool)
    non_auto = ~trials["auto_rewarded"].fillna(False).astype(bool)
    one_outcome = flags.sum(axis=1).eq(1)
    finite_bounds = np.isfinite(trials["start_time"]) & np.isfinite(trials["stop_time"])
    positive = trials["stop_time"].to_numpy() > trials["start_time"].to_numpy()
    mask = contingent & non_aborted & non_auto & one_outcome & finite_bounds & positive
    ...
    return trials.loc[mask].copy(), counts
```
```python
    if run is None or pupil is None:
        dropped_coverage += 1
        continue
    trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]
    image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
    is_go, is_catch = bool(row.go), bool(row.catch)
    if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
        dropped_stim_consistency += 1
        continue
...
if len(records) < 2 or len(cell_ids) == 0:
    return None, exclusions
```

iii. Step 5 decision 2: "`(go | catch) & ~aborted & ~auto_rewarded`, finite ordered time bounds, exactly one of the four contingent outcomes, and sufficient synchronized ophys/running/pupil coverage. This directly implements the task and avoids invented outcomes." Step 5 decision 8 on pupil: "Do not extrapolate outside valid eye coverage; drop affected trials." Step 9/10 on the zero-event trials: "Removing them would select on decoder input and bias behavior; replacing events with ΔF/F or smoothed visualization events would violate the reference representation. The warnings are explicitly accepted."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.events` — the SDK's **unfiltered FastLZero inferred calcium-event magnitudes** (`events['events']`), indexed by `ophys_timestamps`. dF/F is explicitly *not* used, and neither is `filtered_events`. Rows are the SDK's valid ROIs; `cell_specimen_id`s are stored in metadata.

ii.
```python
def stack_event_traces(events: pd.DataFrame, nframes: int):
    cell_ids = events.index.to_numpy(dtype=np.int64)
    traces = np.vstack(events["events"].to_numpy()).astype(np.float32, copy=False)
    if traces.shape != (len(cell_ids), nframes):
        raise ValueError(f"event shape {traces.shape} != ({len(cell_ids)}, {nframes})")
    if not np.isfinite(traces).all():
        raise ValueError("non-finite inferred event value")
    return traces, cell_ids
...
timestamps = np.asarray(obj.ophys_timestamps, dtype=float)
events, cell_ids = stack_event_traces(obj.events, len(timestamps))
```

iii. Step 4 discrepancy table: "Paper says all neural analyses used detected calcium events to remove slow GCaMP decay → Use unfiltered inferred event magnitudes, not recomputed ΔF/F or smoothed visualization events." This quotes methods.txt: "For all analysis of neural data we used the detected calcium events … thus removing the slow decay dynamics of the calcium indicator GCaMP6f". Step 3 adds that `filtered_events` is documented by the SDK as visualization smoothing only.

## 2-b. How is the `neural` data processed?

i. No re-computation of dF/F, no normalisation, no smoothing, no z-scoring. The only transformation is temporal re-binning: event magnitudes are **summed** within each half-open 100 ms bin `[edge_i, edge_{i+1})`, using `searchsorted` on `ophys_timestamps` and a trial-local cumulative sum. Output is `float32`, shape `(n_cells, T)`.

ii.
```python
def bin_events(events, timestamps, edges):
    """Sum frames in half-open bins using a small trial-local cumulative sum."""
    indices = np.searchsorted(timestamps, edges, side="left")
    lo, hi = int(indices[0]), int(indices[-1])
    local = events[:, lo:hi]
    cumulative = np.empty((events.shape[0], local.shape[1] + 1), dtype=np.float32)
    cumulative[:, 0] = 0.0
    np.cumsum(local, axis=1, dtype=np.float32, out=cumulative[:, 1:])
    rel = indices - lo
    return (cumulative[:, rel[1:]] - cumulative[:, rel[:-1]]).astype(np.float32, copy=False)
...
neural = bin_events(events, timestamps, edges)
if neural.shape[1] != nbins:
    raise AssertionError("neural bin length mismatch")
```

iii. Step 5 mapping table: "Sum inferred event magnitudes into non-overlapping 100 ms bins bounded by each retained trial … Unfiltered discrete events match paper; bin sum preserves event magnitude/count-like activity." Step 10 check 6: "Summed 100 ms unfiltered FastLZero magnitudes preserve discrete event signal and approximate the slowest frame period. Independent sums passed exactly" (45/45 `np.allclose` re-derivations from a freshly loaded SDK object, `/app/cache/sanity_checks.py`).

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron-level filtering. The agent takes exactly the rows the SDK returns in `events` (which correspond to the `cell_specimen_table`'s `roi_valid=True` ROIs), preserves SDK row order, asserts every trace length equals `len(ophys_timestamps)` and that all values are finite. No SNR/activity/event-rate threshold is imposed, and trials with all-zero events are not removed. Sessions with zero cells are dropped (none occurred).

ii.
```python
    if traces.shape != (len(cell_ids), nframes):
        raise ValueError(f"event shape {traces.shape} != ({len(cell_ids)}, {nframes})")
    if not np.isfinite(traces).all():
        raise ValueError("non-finite inferred event value")
...
if len(records) < 2 or len(cell_ids) == 0:
    print(f"  experiment {experiment_id}: excluded ({len(records)} trials, {len(cell_ids)} cells)")
    return None, exclusions
```

iii. Step 5 decision 9: "Trust SDK `cell_specimen_table`/events rows, which already contain valid ROIs … Do not filter on activity/SNR because neither paper nor decoder specification calls for it." Step 3 lists the upstream QC the Allen pipeline already applies (non-cells, unions, duplicates, motion-border ROIs, demixing failures, z-drift, epileptiform recordings), and concludes "Do not impose a new downstream cell filter absent from the reference analysis."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to **trial start** on the common synchronised clock: `edges = start_time + k·0.1 s`. Frames are assigned to bins by `np.searchsorted(ophys_timestamps, edges, side='left')`, i.e. by timestamp and never by frame index. Because the trial window is the native variable-length trial rather than a fixed window around an event, `off_start`/`off_end` are recorded as `None`. All other streams are evaluated on the centres of these same bins, so neural and outputs are aligned by construction.

ii.
```python
edges = start + np.arange(nbins + 1, dtype=float) * DT
centers = edges[:-1] + DT / 2.0
...
indices = np.searchsorted(timestamps, edges, side="left")
```
```python
"temporal_alignment_event": "Variable-length trial start on the common synchronized clock; bins follow ophys timestamps.",
"off_start": None,
"off_end": None,
```

iii. Step 4/Step 10 check 5: "all sources are queried by synchronized timestamps. Ophys timestamps define half-open 100 ms neural bins; behavior is interpolated to centers; displayed stimulus intervals are queried at the same centers. This follows the whitepaper synchronization model and satisfies the common-bin requirement" (methods.txt: all clocks recorded on one 100 kHz sync board). The `--show-processing` plots overlay native event traces against the binned trace to demonstrate no shift.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — the native ophys frame rate (~31 Hz single-plane, ~11 Hz multiscope) is rebinned to a **fixed 100 ms bin for every session and trial** (`time_bin_size = 100.0` ms). Neural events are summed within bins; behavioural signals are interpolated at bin centres; categorical labels are evaluated at bin centres. Resulting T: mean 84.2, min 70, max 125.

ii.
```python
DT = 0.100  # seconds
...
"time_bin_size": DT * 1000.0,
```

iii. Step 5 decision 3: "Fixed 100 ms bins for every experiment. This is close to the slower multiscope frame period (~91 ms), avoids pretending it has 31 Hz resolution, and retains multiple bins per 250 ms image and first 400 ms paper decoding window." Step 4 timing row: "target requires one bin size across sessions → Align all modalities by timestamps, then aggregate/interpolate to a fixed bin width; never align by raw sample index." The driver is the format requirement that "Time bins should be the same size for all trials and sessions" combined with the agent's decision to keep both the 31 Hz and 11 Hz rigs.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. From the SDK `stimulus_presentations` table (not the trials table): `image_name`, `start_time`, `end_time`, `omitted`, plus the block selector columns `active` and `stimulus_block_name` used to restrict to the active `change_detection` block.

ii.
```python
def active_stimulus_table(stim: pd.DataFrame) -> pd.DataFrame:
    mask = stim["active"].fillna(False).astype(bool) if "active" in stim else np.ones(len(stim), bool)
    if "stimulus_block_name" in stim:
        names = stim["stimulus_block_name"].fillna("").astype(str)
        mask &= names.str.contains("change_detection", case=False, regex=False)
    ans = stim.loc[mask].copy()
    return ans.sort_values("start_time")
...
stim = active_stimulus_table(obj.stimulus_presentations)
```

iii. Step 1/Step 5: "The modern stimulus table includes several blocks; the SDK warning directs users to the active `change_detection` block (or equivalently active task rows), rather than unrelated passive/movie blocks." The SDK itself emits `UpdatedStimulusPresentationTableWarning` telling the user to filter on `stimulus_block_name.str.contains('change_detection')`, which the agent followed.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. For each 100 ms bin centre the agent finds the last stimulus presentation that started at or before the centre (`searchsorted(..., side='right') - 1`) and labels the bin with that presentation's `image_name` **only if the centre also falls before its `end_time`** and the presentation is not omitted/blank. Otherwise the bin is labelled `"gray"`. A single global vocabulary `["gray"] + sorted(unique image names)` is built across the whole dataset (17 classes: gray + 16 images) and labels are mapped to `int16` codes. In the converted data, `gray` covers ~66.5% of bins and each image ~1.9–2.3%.

ii.
```python
def label_stimuli(stim, centers, edges):
    labels = np.full(len(centers), "gray", dtype=object)
    changes = np.zeros(len(centers), dtype=np.int16)
    starts = stim["start_time"].to_numpy(float)
    ends = stim["end_time"].to_numpy(float)
    idx = np.searchsorted(starts, centers, side="right") - 1
    valid_idx = idx >= 0
    rows = np.maximum(idx, 0)
    image_name = stim["image_name"].fillna("").astype(str).to_numpy()
    omitted = stim["omitted"].fillna(False).astype(bool).to_numpy() if "omitted" in stim else np.zeros(len(stim), bool)
    shown = valid_idx & (centers < ends[rows]) & (~omitted[rows]) & (image_name[rows] != "") & (image_name[rows] != "omitted")
    labels[shown] = image_name[rows[shown]]
```
```python
image_values = ["gray"] + sorted({str(x) for s in sessions for r in s["records"] for x in r["image_labels"] if x != "gray"})
image_map = {x: i for i, x in enumerate(image_values)}
...
image = np.fromiter((image_map[str(x)] for x in r["image_labels"]), dtype=np.int16, count=T)
```

iii. Step 5 decision 5: "Build a single deterministic global class list: `gray` followed by sorted image names across retained active stimulus rows. Omitted images and 500 ms inter-stimulus periods are genuinely gray and receive `gray`, not the preceding identity." The agent reads the decoder-output spec ("Image identity (of the image presented during the non-grey screen)") as meaning the label applies to the non-grey periods, with grey getting its own class; methods.txt's "250 ms image + 500 ms gray" flash cycle is cited as the physical basis.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Labels are evaluated at the centres of exactly the same 100 ms bins used to sum the neural events, within the same trial window, so row *t* of the output array corresponds to column *t* of the neural array by construction. Only stimulus rows overlapping the trial window are considered.

ii.
```python
trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
neural = bin_events(events, timestamps, edges)
...
output = np.vstack([image, r["change"], discretize(r["running"], run_edges),
                    discretize(r["pupil"], pupil_edges), outcome]).astype(np.int16)
if output.shape != (5, T) or not np.isfinite(r["neural"]).all():
    raise AssertionError("invalid final trial")
```

iii. Step 10 check 5 and check 9: all streams are queried on the synchronised clock at the same bin centres; the `--show-processing` plots draw the stimulus/image/change staircase against the binned neural trace on a common time axis, and the independent sanity checks re-derived image identity for 9 trials across 3 sessions and matched it exactly.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The `is_change` flag of the active `stimulus_presentations` table, together with that presentation's `start_time`/`end_time`. The trials-table `change_time` is deliberately not used directly; the trials-table `go`/`catch` flags are used only as a consistency assertion.

ii.
```python
is_change = stim["is_change"].fillna(False).astype(bool).to_numpy()
change_rows = np.flatnonzero(is_change & (starts < edges[-1]) & (ends > edges[0]))
```

iii. Step 5 decision 6: "Use the full stimulus presentation carrying `is_change`, not trial `change_time` independently, because this display-lag-corrected row is the newly changed image shown immediately after identity changes."

## 4-b. What processing is involved in computing `output` *Image change*?

i. Binary per-bin indicator: every bin whose centre falls inside the on-screen interval of an `is_change` presentation gets 1 (≈250 ms ⇒ 2–3 bins), everything else 0. Because `is_change` marks genuine identity changes only, catch (sham-change) trials are all-zero. The agent then asserts each retained go trial contains exactly one change presentation and each catch trial none, dropping any trial that violates this. Final distribution: 97.3% zeros / 2.7% ones.

ii.
```python
for row in change_rows:
    # `is_change` belongs to the newly shown image presentation. Mark its
    # complete on-screen interval (normally 250 ms), i.e. the period
    # immediately after identity changed, rather than an undersampled
    # mathematical impulse at onset.
    changes[(centers >= starts[row]) & (centers < ends[row])] = 1
return labels, changes, len(change_rows)
...
if (is_go and nchanges != 1) or (is_catch and nchanges != 0):
    dropped_stim_consistency += 1
    continue
```

iii. Step 5 decision 6 and the Step 8 iteration note: an initial single-bin impulse representation gave validation balanced accuracy 0.4916 (below the 0.5 chance level); "Review against the SDK and paper showed `is_change` labels the newly presented image interval and paper decoding uses post-presentation neural activity. The output was corrected to 1 across bin centers in that ~250 ms on-screen interval ('right after' identity changes)"; change accuracy then rose to 0.6496 on the sample. Steps 7–9 were rerun.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding of a continuous quantity is involved — the variable is natively binary (`no_change` = 0, `change` = 1). The only "categorisation" decision is the temporal extent that counts as a change: the changed image's full presentation interval (~250 ms / 2–3 bins) rather than a single onset bin or the full 750 ms flash+grey cycle.

ii.
```python
changes = np.zeros(len(centers), dtype=np.int16)
...
changes[(centers >= starts[row]) & (centers < ends[row])] = 1
...
"output_values": [image_values, ["no_change", "change"], ...]
```

iii. Same as 4-b: a single-bin impulse was "arbitrarily undersampled" at 100 ms resolution and decoded below chance; the presentation interval is the display-lag-corrected period "immediately after identity changed" (Step 5 decision 6, Step 8 iteration).

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: it is computed at the centres of the same 100 ms bins from the same trial-restricted stimulus table, in the same `label_stimuli` call, and stacked as row 1 of the `(5, T)` output array whose T equals the neural column count.

ii.
```python
image_labels, change, nchanges = label_stimuli(trial_stim, centers, edges)
...
output = np.vstack([image, r["change"].astype(np.int16, copy=False), ...])
```

iii. Step 10 checks 5/9 (synchronised-clock querying, half-open bins, no off-by-one) and the processing plots, which show "Each go change interval exactly overlays the newly changed image presentation."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `BehaviorOphysExperiment.running_speed` — the SDK's filtered linear running speed in cm/s with synchronised `timestamps`. The `raw_running_speed` variant is explicitly not used.

ii.
```python
running = obj.running_speed
...
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
```

iii. Step 1/Step 3: "Filtered running speed is preferred to `raw_running_speed`"; methods.txt describes the SDK pipeline (encoder unwrap, >5.1 V artifact removal, angular→linear conversion, transient removal, 10 Hz low-pass Butterworth) and "The SDK `running_speed` property supplies this processed signal."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation of the SDK speed trace onto the trial's 100 ms bin centres, after sorting, de-duplicating timestamps and dropping non-finite samples. If the trial window is not fully inside the valid sample range the whole trial is dropped rather than extrapolated. The interpolated continuous value is then discretised (see 5-c).

ii.
```python
def interpolate_finite(times, values, query):
    good = np.isfinite(times) & np.isfinite(values)
    if good.sum() < 2:
        return None
    times, values = times[good], values[good]
    order = np.argsort(times, kind="stable")
    times, values = times[order], values[order]
    unique = np.r_[True, np.diff(times) > 0]
    times, values = times[unique], values[unique]
    if len(times) < 2 or query[0] < times[0] or query[-1] > times[-1]:
        return None
    return np.interp(query, times, values).astype(np.float32)
```

iii. Step 5 mapping table: "Linearly interpolate SDK-filtered cm/s to bin centers, then discretize using pooled 20/40/60/80 percentiles over retained data." No extrapolation is allowed because "sufficient synchronized ophys/running/pupil coverage" is part of the trial-validity definition (Step 5 decision 2).

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile (quintile) bins. The four thresholds are computed **once, globally**, from all finite running samples pooled over every retained trial of every retained session (`np.quantile` at 0.2/0.4/0.6/0.8), then applied with `searchsorted(..., side='right')` to give codes 0–4. Degenerate (repeated) edges would raise a warning; none occurred. Thresholds are stored in metadata. Resulting distribution is exactly [.2,.2,.2,.2,.2].

ii.
```python
def percentile_edges(values: list[np.ndarray]) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float64, copy=False)
    edges = np.quantile(joined[np.isfinite(joined)], [0.2, 0.4, 0.6, 0.8])
    if not np.all(np.diff(edges) > 0):
        warnings.warn(f"Repeated percentile edges: {edges}")
    return edges

def discretize(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, values, side="right").astype(np.int16)
...
run_edges = percentile_edges([r["running"] for s in sessions for r in s["records"]])
```

iii. Step 5 decision 7: "Calculate thresholds from all finite, temporally aligned samples in the final retained sessions/trials, so each requested continuous output has dataset-wide equal-percentile definitions … record thresholds." This implements the instruction "discretized into five equal percentile bins" at the dataset level so that class codes mean the same thing in every session.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the trial's bin centres — the same bins used for the neural sums — so no separate alignment step is needed. Trials where the running stream does not cover the window are excluded rather than padded.

ii.
```python
centers = edges[:-1] + DT / 2.0
run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
...
neural = bin_events(events, timestamps, edges)
```

iii. Step 4/Step 10 check 5: the whitepaper's single 100 kHz synchronisation board means SDK timestamps of all streams are on a common clock, so interpolation by timestamp is the correct alignment; the processing plots overlay the raw SDK running samples with the aligned trace to verify no shift.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `BehaviorOphysExperiment.eye_tracking`, column `pupil_area` (the **processed** ellipse area, which the SDK sets to NaN on `likely_blink` frames) plus `timestamps`. Raw/blink-contaminated columns are not used. Experiments with no processed pupil stream are dropped entirely.

ii.
```python
eye = obj.eye_tracking
if eye is None or len(eye) == 0 or "pupil_area" not in eye:
    exclusions["no_eye_session"] = len(eligible)
    print(f"  experiment {experiment_id}: excluded (no processed pupil data)", flush=True)
    return None, exclusions
pupil_area = eye["pupil_area"].to_numpy(float)
```

iii. Step 4 discrepancy table: "Processed values are NaN on blink/outlier frames … Derive equivalent circular diameter `2*sqrt(pupil_area/pi)` from processed area, preserving ellipse size while respecting SDK masking." Step 5 decision 8: "Use processed pupil area, never raw blink-corrupted values."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Area → equivalent circular diameter `2·sqrt(area/π)` (a monotone transform, computed before interpolation), then the same `interpolate_finite` treatment as running: drop NaN (blink) samples, sort/de-duplicate, linearly interpolate onto bin centres, and drop the trial if the window is outside valid coverage. Internal blink gaps are therefore bridged by linear interpolation, matching the reference practice of interpolating after blink removal.

ii.
```python
pupil_area = eye["pupil_area"].to_numpy(float)
pupil_diameter = 2.0 * np.sqrt(np.maximum(pupil_area, 0.0) / np.pi)
...
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
if run is None or pupil is None:
    dropped_coverage += 1
    continue
```

iii. Step 3: "The task asks for pupil diameter, so ellipse-derived diameter should be used rather than area." Step 5 decision 8: "Convert to equivalent diameter before interpolation … Linear interpolation across internal missing frames avoids losing whole trials for short blinks; report missing fraction and sensitivity in checks." Metadata records `"pupil_definition": "2*sqrt(processed pupil_area/pi); blink/outlier samples masked upstream by SDK, internal gaps linearly interpolated"`.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: four global quintile thresholds computed by `np.quantile` over all finite pupil-diameter samples of all retained trials, applied with `searchsorted(side='right')` → codes 0–4, stored in metadata as `pupil_quintile_edges_equivalent_diameter_pixels`. Resulting distribution exactly [.2,.2,.2,.2,.2].

ii.
```python
pupil_edges = percentile_edges([r["pupil"] for s in sessions for r in s["records"]])
...
discretize(r["pupil"], pupil_edges)
...
"output_values": [..., ["Q1_smallest", "Q2", "Q3", "Q4", "Q5_largest"], ...]
```

iii. Step 5 decision 7 (as for running): dataset-wide equal-percentile definitions so a quintile label means the same thing across sessions; thresholds recorded for interpretability. Note the quintiles of the equivalent diameter are identical to quintiles of the area since the transform is monotone.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same 100 ms bin centres as the neural bins, within the same trial window; trials without full eye-tracking coverage are dropped rather than padded/extrapolated.

ii.
```python
pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
...
neural = bin_events(events, timestamps, edges)
```

iii. Same rationale as running speed — the eye camera clock is recorded on the same synchronisation board, so timestamp interpolation is the correct alignment (Step 4, Step 10 check 5). The processing plot panel overlays the processed SDK pupil samples with the aligned trace and its quintile staircase.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`. Trials not having exactly one of these true were already excluded upstream.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
OUTCOME_NAMES = ["hit", "miss", "false_alarm", "correct_reject"]

def outcome_code(row: pd.Series) -> int:
    flags = np.asarray([bool(row[x]) for x in OUTCOME_COLUMNS])
    if flags.sum() != 1:
        raise ValueError("trial does not have exactly one outcome")
    return int(np.flatnonzero(flags)[0])
```

iii. Step 4/Step 5: "The trial schema distinguishes go, catch, aborted, and auto-rewarded trials and supplies mutually interpretable outcome flags: hit, miss, false alarm, correct reject"; using the direct flags "avoids invented outcomes". The whitepaper's four contingent outcomes are cited.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The flag vector is mapped to an integer code 0–3 (hit/miss/false_alarm/correct_reject) and **broadcast as a constant across all T bins of the trial** so that it can live in the same `(5, T)` output array as the time-varying outputs. Class names are exported in `output_values`. Time-weighted distribution: [.303 hit, .571 miss, .017 FA, .108 CR].

ii.
```python
code = outcome_code(row)
...
outcome = np.full(T, r["outcome"], dtype=np.int16)
output = np.vstack([image, r["change"], discretize(r["running"], run_edges),
                    discretize(r["pupil"], pupil_edges), outcome]).astype(np.int16, copy=False)
```

iii. Step 5 mapping table: "Map to 0–3 and repeat constant across all T bins … Static per trial represented as a constant time series so all requested outputs coexist in one `(5,T)` array." The instruction lists trial outcome as "Static per-trial" while the format requires a common `(n_output, n_timepoints)` array.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled defensively at four levels, with an auditable ledger:
* Boolean trial flags are `fillna(False)` before use; time bounds must be finite and ordered.
* Non-finite inferred events or a trace/timestamp shape mismatch raise immediately (fail loud, not silent).
* Missing behavioural coverage (running or pupil not spanning the trial) → trial dropped (`dropped_coverage`); blink-NaN pupil samples are dropped before interpolation so internal gaps are bridged; trials shorter than 2 bins → dropped (`dropped_short`); go/catch vs `is_change` inconsistency → dropped (`dropped_stim_consistency`).
* Whole experiments missing the processed pupil stream → excluded (3 experiments); experiments with <2 trials or 0 cells → excluded; any unexpected exception in an experiment is caught, logged as `fatal_error`, and the run continues.
* All counts are written into `metadata['session_info'][i]['exclusions']` and `metadata['all_experiment_exclusions']`.
* Known-but-accepted issue: 2,502/51,075 trials (4.90%) have all-zero inferred events; these are kept deliberately and the validator warning is documented rather than suppressed.

ii.
```python
flags = trials[OUTCOME_COLUMNS].fillna(False).astype(bool)
...
finite_bounds = np.isfinite(trials["start_time"]) & np.isfinite(trials["stop_time"])
```
```python
    if eye is None or len(eye) == 0 or "pupil_area" not in eye:
        exclusions["no_eye_session"] = len(eligible)
        return None, exclusions
```
```python
exclusions.update({
    "dropped_coverage": dropped_coverage,
    "dropped_short": dropped_short,
    "dropped_stim_consistency": dropped_stim_consistency,
    "retained": len(records),
})
```
```python
            except Exception as exc:
                excluded[int(experiment_id)] = {"fatal_error": repr(exc)}
                print(f"  experiment {experiment_id}: ERROR {exc!r}", flush=True)
```

iii. Step 5 decision 8 and Step 10: "categorical pupil output cannot be honestly imputed from absent data" — hence dropping rather than filling; "All-zero event warnings were investigated, not suppressed: sparse inferred events and low neuron counts explain them; retained to avoid conditioning trial inclusion on neural activity." The exclusion ledger exists so that "no data is missed during conversion" can be verified arithmetically (Step 7 verified that a 39-trial session was not data loss: 1,078/1,117 rows were non-contingent or aborted).

## 9-a. What are the most time-consuming steps of the code?

i. The dominant cost is SDK/NWB object construction and reading full-session arrays in `cache.get_behavior_ophys_experiment()` (plus `obj.events`, `obj.stimulus_presentations`, `obj.running_speed`, `obj.eye_tracking`, each lazily materialised). Measured: ~3.8–5.3 s per experiment sequentially; total full run 348.33 s (5.8 min) for 202 experiments using 4 worker processes (~1.72 s/experiment wall-clock). Secondary costs are pickling the 2.74 GB output and the pooled percentile concatenation over ~4.3 M samples. The agent prints per-experiment and total timing.

ii.
```python
started = time.perf_counter()
obj = cache.get_behavior_ophys_experiment(int(experiment_id))
...
elapsed = time.perf_counter() - started
print(f"  experiment {experiment_id}: {len(cell_ids)} cells, {len(records)} trials, {elapsed:.2f}s", flush=True)
...
print(f"Total conversion time: {elapsed:.2f}s ({elapsed/len(table):.2f}s/selected experiment)", flush=True)
```

iii. Step 6: "Potential bottlenecks are SDK/NWB object construction, stacking full-session event arrays, and repeated trial-level aggregation. Full trace loading is unavoidable through the SDK." Mitigations listed: metadata selection avoids loading passive assets; four independent worker processes ("Estimated ~4x reduction of SDK loading/processing wall time"); `del obj, events; gc.collect()` after each experiment; float32/int16 storage.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-trial loop in `convert_experiment` is the main candidate:
* `interpolate_finite(running[...], ...)` and `interpolate_finite(eye[...], ...)` are called **inside** the trial loop, so the full-session running (~290 k samples) and eye (~145 k samples) arrays are converted to numpy, finite-masked, argsorted and de-duplicated once per trial — ~250 times per experiment. Interpolating once onto a session-wide grid (or hoisting the cleaning out of the loop) would eliminate almost all of this.
* `trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]` performs a full pandas boolean scan of the ~13 k-row stimulus table per trial; a `searchsorted` on sorted start times would be O(log n).
* Inside `label_stimuli`, `for row in change_rows: changes[(centers >= starts[row]) & (centers < ends[row])] = 1` loops over change rows with a full boolean mask each time (usually 1 row, so cheap).
* In `assemble`, `np.fromiter((image_map[str(x)] for x in r["image_labels"]), ...)` is a Python-level dict lookup per bin over ~4.3 M bins, and the `image_values` vocabulary is built by a comprehension over every label of every trial.
The genuinely vectorised parts are `bin_events` (searchsorted + trial-local cumsum, no neuron/bin loops) and `label_stimuli`'s label assignment.

ii.
```python
    for trial_id, row in eligible.iterrows():
        ...
        run = interpolate_finite(running["timestamps"].to_numpy(), running["speed"].to_numpy(), centers)
        pupil = interpolate_finite(eye["timestamps"].to_numpy(), pupil_diameter, centers)
        ...
        trial_stim = stim[(stim["start_time"] < edges[-1]) & (stim["end_time"] > edges[0])]
```

iii. Step 6: "Event binning uses vectorized `searchsorted` and trial-local cumulative sums rather than neuron/bin Python loops or a second full-session cumulative array. Continuous streams use vectorized interpolation." The agent treated the per-trial loop as acceptable because total runtime (5.8 min) was well inside the 15-minute budget after process parallelism, and it did not flag the per-trial re-cleaning of the full behavioural arrays.

## 9-c. What processing does the code repeat multiple times?

i. Repeated work that could have been done once:
* Per-trial re-preparation of the session's running and eye arrays (`to_numpy`, finite mask, `argsort`, unique) inside `interpolate_finite` — repeated for every trial of every experiment.
* Per-trial boolean filtering of the full stimulus table.
* Finiteness of the neural data is checked twice: once for the whole session in `stack_event_traces` and again per trial in `assemble` (`not np.isfinite(r["neural"]).all()`).
* `percentile_edges` concatenates every running (and pupil) sample of the dataset into one array, and the retained per-trial arrays are then traversed again for image-vocabulary construction and again for discretisation.
* In `--show-processing`/sequential mode the parallel path is disabled, so a re-run to produce plots re-loads experiments already processed.
Not repeated: experiments are loaded exactly once per run, and event binning uses a trial-local cumulative sum rather than recomputing a session-wide cumsum.

ii.
```python
def interpolate_finite(times, values, query):
    times = np.asarray(times, dtype=float)
    good = np.isfinite(times) & np.isfinite(values)
    ...
    order = np.argsort(times, kind="stable")          # repeated for every trial
```
```python
            if output.shape != (5, T) or not np.isfinite(r["neural"]).all():
                raise AssertionError("invalid final trial")
```

iii. The agent's stated efficiency rationale (Step 6) is about avoiding *copies* of the large neural arrays — "copying full arrays more than once would inflate memory" — and about parallelism; it does not claim that behavioural preprocessing is hoisted out of the trial loop. The duplicated finiteness check is presented as a deliberate final validation gate ("Validate data shapes and types at each step").

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Minor, mostly bookkeeping:
* `trial_ids` is built inside `convert_experiment` (`trial_ids = []` … `trial_ids.append(...)`) and **never read** — the result dict re-derives trial ids from `records`. Dead code.
* `stack_event_traces` materialises and finiteness-checks the **whole session's** event matrix even though only the retained trial windows are ever written out (full-session load is unavoidable via the SDK, but the full `np.isfinite(...).all()` pass and the `np.vstack` copy are not).
* Rich provenance that the decoder never uses is computed and pickled: per-experiment `exclusions` ledgers, `all_experiment_exclusions`, `cell_specimen_ids`, `trial_ids`, `trial_start_times`, `trial_stop_times_binned`, `native_frame_rate_hz`, `region_aliases` — useful for auditing, but it contributes to the 2.74 GB pickle.
* A `(0, T)` float32 input array is allocated per trial although `input_names` is empty and the decoder has no inputs; the reference allocates a `(0,)` array instead.
* The `diagnostic` payload (full ophys timestamps, mean event trace, raw running/eye traces) is assembled for plotting — correctly gated behind `keep_diagnostic`, so it is not paid for in the full run.

ii.
```python
    records = []
    ...
    trial_ids = []
    for trial_id, row in eligible.iterrows():
        ...
        trial_ids.append(int(trial_id))     # never used afterwards
```
```python
            si.append(np.empty((0, T), dtype=np.float32))
```
```python
        "diagnostic": {...} if keep_diagnostic else None,
```

iii. The agent's justification for the extra metadata (Step 5 decision 10) is auditability: "Include experiment IDs, cell specimen IDs, class mappings, quintile thresholds, microscope rate, region, trial IDs, and exclusions in metadata/session_info", which it then used for the Step 7/9/10 reconciliation checks. The dead `trial_ids` variable and the duplicated finiteness check are not mentioned anywhere in CONVERSION_NOTES.md.
