# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. All access goes through the AllenSDK `VisualBehaviorOphysProjectCache`, built with `from_s3_cache(cache_dir=/app/data)` (the local cache directory). The experiment-level metadata table (`get_ophys_experiment_table()`) is the master listing. The AI intersects that table with the experiment IDs actually present on local disk (obtained by globbing the `behavior_ophys_experiments/*.nwb` *file names* only — no NWB file is ever opened directly), and further restricts to `project_code == 'VisualBehavior'` (exact single-plane project, excluding the Multiscope/Task1B variants) and `passive == False`. Each surviving experiment is then loaded with `cache.get_behavior_ophys_experiment(exp_id)` inside a `ProcessPoolExecutor` worker (16 workers), and every stream (events, ophys timestamps, trials, stimulus presentations, running speed, eye tracking) is read from the returned SDK object. 168 experiments were selected; 165 were retained.

ii.
```python
def make_cache():
    cls = _import_cache_class()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return cls.from_s3_cache(cache_dir=DATA_DIR)

def local_experiment_ids(cache) -> list[int]:
    """Return active exact-VisualBehavior experiments present on local disk."""
    table = cache.get_ophys_experiment_table()
    file_ids = {
        int(path.stem.rsplit("_", 1)[1])
        for path in (DATA_DIR / "visual-behavior-ophys-1.1.0" /
                     "behavior_ophys_experiments").glob("*.nwb")
    }
    keep = (
        table.index.isin(file_ids)
        & table["project_code"].eq("VisualBehavior")
        & ~table["passive"].astype(bool)
    )
    return sorted(table.index[keep].astype(int).tolist())
```
```python
cache = make_cache()
exp = cache.get_behavior_ophys_experiment(int(experiment_id))
meta = exp.metadata
if meta["project_code"] != "VisualBehavior":
    raise ValueError(f"{experiment_id}: unexpected project {meta['project_code']}")
```
```python
workers = min(16, len(ids), max(1, (os.cpu_count() or 2) // 2))
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. From CONVERSION_NOTES Step 1/Step 4: "The reference documentation explicitly recommends interacting through SDK objects rather than the NWB representation"; the metadata cache distinguishes `VisualBehavior` from `VisualBehaviorMultiscope`, and the task asks for data "under the Visual Behavior task", so the exact `project_code == 'VisualBehavior'` cohort is used. The local-file intersection exists because the cached manifest lists every released experiment while only a subset (284 NWBs) is on disk — the glob only reads file names so the "never open NWB directly" constraint is preserved. Parallel loading was chosen because "Detailed SDK object construction includes compressed trace reads… I/O/decompression-bound on the full cache" (full conversion: 73.3 s).

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values taken from each loaded experiment's SDK metadata, stored as strings, sorted, and indexed by `subject_idx` (one entry per converted session). 37 mice result — matching the 37 mice in the VisualBehavior project metadata table.

ii.
```python
"mouse_id": str(meta["mouse_id"]),
...
subjects = sorted({s["mouse_id"] for s in raw_sessions})
subject_map = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_map[s["mouse_id"]] for s in raw_sessions],
                          dtype=np.int32),
```

iii. "experiment `mouse_id` → `subjects`, `subject_idx`: Sorted unique string IDs and integer lookup" (Step 5 mapping table). The notes cross-checked the resulting count against the project metadata (37 exact-`VisualBehavior` active mice) and against the whitepaper/paper totals (82 mice in the v1 release), attributing the difference to project/version scope.

## 1-c. How are the data split into sessions?

i. One SDK *experiment* (one imaging plane) is mapped to one decoder "session". The AI verified that for `project_code == 'VisualBehavior'` each session contains exactly one plane, so experiment ↔ `ophys_session_id` is 1:1 (confirmed independently: 239 experiments / 239 unique `ophys_session_id`, all VISp). Sessions are ordered deterministically by sorted `experiment_id`. Both `experiment_id` and `ophys_session_id` are written into `metadata['session_info']`.

ii.
```python
raw_sessions.sort(key=lambda x: x["experiment_id"])
...
"experiment_id": int(experiment_id),
"ophys_session_id": int(meta["ophys_session_id"]),
```

iii. Step 1 notes: "Sessions and experiments differ: one `ophys_session_id` may have multiple planes/experiments. The target's neuron matrix naturally maps one SDK experiment (one simultaneous imaging plane) to one decoder session; merging asynchronous multiplane traces would require resampling and is not implied by the format." Step 4: "These are single-plane VISp recordings, so session and experiment are one-to-one and no asynchronous plane merge is needed."

## 1-d. How are the data split into trials?

i. Trials come from the SDK `trials` table, sorted by `start_time`. The eligible mask is `(go | catch) & ~aborted & ~auto_rewarded`. Each eligible trial spans the **half-open** interval `[start_time, stop_time)` and is resampled onto a uniform 30 Hz grid anchored at that trial's `start_time`; grids for all trials of a session are concatenated once, all streams are interpolated on the concatenated vector, then split back per trial using cumulative lengths. Trial lengths are therefore variable (211–377 bins, 7.0–12.6 s). 42,470 trials were produced (mean 257/session).

ii.
```python
trials = exp.trials.sort_values("start_time")
eligible = (
    (trials["go"] | trials["catch"])
    & ~trials["aborted"]
    & ~trials["auto_rewarded"]
)
trials = trials.loc[eligible].copy()
if len(trials) < 2:
    raise ValueError(f"{experiment_id}: fewer than two eligible trials")

def _trial_grid(start: float, stop: float) -> np.ndarray:
    """30 Hz relative grid on the half-open SDK trial interval."""
    n = int(np.ceil((stop - start) * FS - 1e-10))
    grid = start + np.arange(max(0, n), dtype=np.float64) * DT
    return grid[grid < stop]

grids = [_trial_grid(float(r.start_time), float(r.stop_time))
         for r in trials.itertuples()]
lengths = np.asarray([len(g) for g in grids], dtype=np.int32)
offsets = np.r_[0, np.cumsum(lengths)]
all_t = np.concatenate(grids)
```

iii. Step 5: "Eligible mask `(go|catch)&~aborted&~auto_rewarded`; grid `start + arange(ceil((stop-start)*30))/30`, retaining samples `< stop`… Half-open boundaries avoid duplicated timestamps." This follows the task instruction to "Include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials" and the whitepaper definition of trial structure. Step 10 verified "Every grid point is in half-open `[start,stop)`… no empty trials or boundary duplication."

## 1-e. How are trials filtered based on quality controls?

i. Filtering happens at three levels.
- **Trial type** (per instructions): only `(go | catch) & ~aborted & ~auto_rewarded`.
- **Outcome integrity**: the four outcome booleans must be exactly one-hot for every eligible trial, else the experiment errors out.
- **Session level**: passive-replay sessions are excluded entirely (71 of 239); sessions with <2 eligible trials are rejected; sessions whose SDK eye-tracking table is empty are excluded and logged (3 sessions: 795953296, 806456687, 833631914), since pupil diameter is a required output. Final cohort: 165/168 sessions, 28,821 cells, 42,470 trials.
No filtering is applied based on neural responsiveness — 1,717 trials (4.0%) whose raw event traces are all-zero are deliberately retained.

ii.
```python
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
if not np.all(outcome_bool.sum(axis=1) == 1):
    bad = trials.index[outcome_bool.sum(axis=1) != 1].tolist()
    raise ValueError(f"{experiment_id}: nonexclusive outcomes in {bad[:5]}")
```
```python
eye = exp.eye_tracking
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")
...
except MissingRequiredData as exc:
    exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
    print(f"[{done}/{len(ids)}] {eid}: EXCLUDED ({exc})", flush=True)
    continue
```

iii. Step 4/Step 10: passive sessions are dropped because "the paper explicitly says passive imaging 'was not analyzed here'" and because "a passive example has all 351 go trials labeled miss and all 51 catch trials correct reject solely because the lick spout is retracted" — i.e. the required trial-outcome output would be an artifact of hardware, not behavior. Missing-eye sessions: "inventing pupil labels would be less defensible than this 1.8% session exclusion". All-zero-event trials: "Removing those trials would violate the requested trial cohort and bias toward neural activity"; an independent SDK reload of the first warned trial reproduced the same all-zero trace.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.events` — the SDK's raw, detected L0 calcium-event magnitude traces (one row per valid cell, `events` column), together with `exp.ophys_timestamps` for their time base. ΔF/F, `filtered_events`, and corrected/demixed/neuropil fluorescence are all deliberately *not* used.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
ophys_t = np.asarray(exp.ophys_timestamps, dtype=np.float64)
if events.shape[1] != len(ophys_t):
    raise ValueError(f"{experiment_id}: event/timestamp length mismatch")
```

iii. Step 3/Step 4/Step 5: the supplied paper states "We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f" and "For all analysis of neural data we used the detected calcium events". The AI therefore treats raw event magnitude as "the reference-matching activity", and rejects `filtered_events` as "documented as half-Gaussian visualization smoothing". It also notes ΔF/F must *not* be recomputed since the SDK version is already baseline-corrected/detrended.

## 2-b. How is the `neural` data processed?

i. Minimal processing: all valid cells of the (single) plane are stacked into a `(n_cells, n_ophys_frames)` float32 matrix, then linearly interpolated from the synchronized `ophys_timestamps` onto the concatenated 30 Hz trial grid, and split into per-trial `(n_cells, T)` float32 views. No normalization, smoothing, z-scoring, or thresholding is applied. The interpolation is hand-vectorized: the timestamp bracketing (`searchsorted`, weights) is computed once and reused for every cell, with edge clamping outside the recording.

ii.
```python
def interp_rows_shared_time(source_t, source_x, target_t):
    right = np.searchsorted(source_t, target_t, side="right")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    lo = source_x[:, left]
    hi = source_x[:, right]
    out = lo + (hi - lo) * weight[None, :]
    before = target_t <= source_t[0]
    after = target_t >= source_t[-1]
    if before.any():
        out[:, before] = source_x[:, [0]]
    if after.any():
        out[:, after] = source_x[:, [-1]]
    return out.astype(np.float32, copy=False)

neural_all = interp_rows_shared_time(ophys_t, events, all_t)
neural = [neural_all[:, offsets[i]:offsets[i + 1]] for i in range(len(lengths))]
```

iii. Step 3/Step 5: "Paper event-triggered neural and running traces were linearly interpolated to a common 30 Hz time series", so the AI applies the same linear interpolation but over whole trials. Step 6: the shared-bracket implementation replaced one `np.interp` call per cell after profiling — verified `np.allclose` against per-cell `np.interp` with max difference 4.77e-7. Step 1: "SDK cell curation is already applied… No electrophysiology-style unit quality filtering applies."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No additional neuron filtering. The AI relies entirely on the SDK/pipeline QC: only ROIs marked valid appear in `cell_specimen_table`/`events` (motion-border, duplicate/union, dendrite, small/dim ROIs and failed demixing are already removed upstream), and only experiments that passed experiment/container QC are published. Sessions with as few as 6 cells are kept. No responsiveness or SNR threshold is imposed.

ii.
```python
event_table = exp.events
cell_ids = event_table.index.to_numpy(copy=True)
events = np.vstack(event_table["events"].to_numpy()).astype(np.float32)
```
(no filtering statement exists; `validate_internal` only asserts `ncell > 0`.)

iii. Step 3 curation notes: "Published data passed experiment and container QC. ROI filtering excludes unions/duplicates, motion-border ROIs, dendrites, and ROIs that are too small, narrow, or dim… Use all SDK-valid cells without an extra ad hoc signal threshold." Step 10 adds that dropping the 4% all-zero-event trials "would select on neural response and inflate results improperly".

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment event is trial start: each trial's time grid begins exactly at the SDK `trials.start_time` and runs to `stop_time` (exclusive). Because the grid times are expressed in the same synchronized clock as `ophys_timestamps`, neural values at those times are obtained by interpolation from the ophys frames. Metadata records `temporal_alignment_event = 'trial start (AllenSDK trials.start_time)'`, `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
grid = start + np.arange(max(0, n), dtype=np.float64) * DT
...
"temporal_alignment_event": "trial start (AllenSDK trials.start_time)",
"off_start": 0.0,
"off_end": None,
```

iii. Step 5 decision 4: "Trial start is time zero; preserve each full variable-length SDK trial until `stop_time`… `off_end=None` because trial duration varies." Step 10: "Whitepaper synchronized clocks through the common 100 kHz board; converter uses SDK ophys timestamps as source coordinates". Using the whole trial (pre-change flashes + response window) is what makes the time-varying image-identity and image-change outputs meaningful.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. A single uniform bin size of 33.333 ms (30 Hz) is used for every trial and session (`metadata['time_bin_size'] = 1000/30`). This *is* a rebinning: the native single-plane ophys rate is ~31 Hz (median inter-frame interval 32.31 ms), so all neural and behavioural streams are resampled by linear interpolation from their own synchronized timestamps onto the common 30 Hz grid. No averaging/binning of spikes is performed (the data are event magnitudes, not spike times).

ii.
```python
FS = 30.0
DT = 1.0 / FS
...
"time_bin_size": 1000.0 / FS,
"sampling_rate_hz": FS,
"neural_resampling": "linear interpolation from synchronized ophys timestamps",
```

iii. Step 5 decision 3: "Resample all streams to 30 Hz (33.333 ms). The paper explicitly interpolated event-triggered neural/running traces to 30 Hz, the whitepaper reports behavior/eye at 30 Hz, and this gives one identical bin size across sessions while alignment originates from synchronized ophys timestamps. Native single-plane imaging is ~31 Hz and not perfectly identical across experiments." This also satisfies the target-format requirement that "Time bins should be the same size for all trials and sessions."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `exp.stimulus_presentations`, restricted to rows whose `stimulus_block_name` contains `change_detection` (the active task block), using the columns `image_name`, `start_time`, `end_time`, and `omitted`. The trials table is *not* used for image identity.

ii.
```python
stim = exp.stimulus_presentations
task = stim[
    stim["stimulus_block_name"].str.contains("change_detection", na=False)
].sort_values("start_time")
```

iii. Step 4: "Presentation rows cover 250 ms image flashes; gray ISI is not a separate image presentation… Use a dedicated `gray` class outside non-omitted image intervals. User wording 'image identity (of the image presented during the non-grey screen)' requires explicit gray handling rather than carrying the last image through ISI." Using the presentation table rather than the trial table gives the actual screen content flash-by-flash, including omissions.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. The time series is initialized to class 0 = `gray`. For every non-omitted task-block presentation, all 30 Hz samples falling in `[start_time, end_time)` are set to the integer code of that presentation's `image_name`. Codes come from a fixed global 17-value list (`gray` plus the 8 image-set-A and 8 image-set-B images); an unrecognized image name raises. Omitted flashes stay `gray`. Result: 66.9% gray, each of the 16 images ≈2.0–2.1%, i.e. the expected 250 ms-on / 500 ms-off duty cycle (non-gray fraction 0.331 vs 250/750 = 0.333).

ii.
```python
IMAGE_VALUES = [
    "gray", "im000", "im031", "im035", "im045", "im054", "im061",
    "im062", "im063", "im065", "im066", "im069", "im073", "im075",
    "im077", "im085", "im106",
]
IMAGE_TO_INT = {name: idx for idx, name in enumerate(IMAGE_VALUES)}
...
image_all = np.zeros(len(all_t), dtype=np.int16)
for row in task.itertuples():
    ...
    if hi > lo and not bool(row.omitted):
        idx = np.arange(lo, hi)
        valid = (all_t[idx] >= start) & (all_t[idx] < end)
        name = str(row.image_name)
        if name not in IMAGE_TO_INT:
            raise ValueError(f"{experiment_id}: unknown image {name}")
        image_all[idx[valid]] = IMAGE_TO_INT[name]
```

iii. Step 5 decision 5: "Stable lexicographic/global values are `gray`, `im000`, … `im106`. Gray is essential because the requested identity applies only during non-gray screen and occupies the 500 ms ISI/omissions." Step 10 check 8 confirmed the converted non-gray fraction (0.33084) matches the 250/750 ms cadence and that all 17 declared values are observed.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is written onto exactly the same concatenated 30 Hz grid `all_t` used for the neural interpolation and split with the same `lengths`/`offsets`, so each output column corresponds to the same timestamp as the matching neural column. Assignment uses `searchsorted` on `all_t` plus an explicit interval mask, which prevents a stimulus that falls inside a *discarded* inter-trial gap from leaking into the next retained trial.

ii.
```python
lo = int(np.searchsorted(all_t, start, side="left"))
hi = int(np.searchsorted(all_t, end, side="left"))
# Concatenated trial grids have gaps; interval condition prevents a
# stimulus in an excluded gap from contaminating the next kept trial.
if hi > lo and not bool(row.omitted):
    idx = np.arange(lo, hi)
    valid = (all_t[idx] >= start) & (all_t[idx] < end)
...
image_trials = split_vector(s["image"], lengths)
```

iii. Step 7/Step 10: the `--show-processing` plots were reviewed — "Non-gray image epochs occupy about 7–8 30 Hz bins and gray intervals about 15 bins, matching 250/500 ms… No temporal offset, boundary leakage, or discretization anomaly is visible." An independent SDK reload reconstructed the image intervals and matched with `np.allclose`.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The same task-block `stimulus_presentations` rows, using the boolean `is_change` column together with `start_time`/`end_time`. (Catch/sham-change presentations have `is_change == False`, so catch trials get no change flag.)

ii.
```python
if bool(row.is_change):
    change_all[idx[valid]] = 1
```

iii. Step 5 mapping table: "task-block `is_change`, `start_time`, `end_time` → `output[1]` image change: Set to 1 throughout each true changed-image presentation… Binary time-varying indicator of the 250 ms interval right after identity changes."

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero-initialized int16 series is set to 1 for every 30 Hz sample inside the 250 ms presentation of the changed image (≈7–8 bins), and 0 everywhere else. Only genuinely changed flashes qualify. Resulting global distribution: 97.4% no_change / 2.6% change.

ii.
```python
change_all = np.zeros(len(all_t), dtype=np.int16)
...
    # "Right after" an identity change is the changed-image flash,
    # not only an arbitrarily narrow single sample at its leading edge.
    if bool(row.is_change):
        change_all[idx[valid]] = 1
```

iii. Step 5 decision 6 and Step 10 "Issues Found and Resolved": the first implementation marked a single bin at the change onset and decoded *below* chance (0.4905). "This revealed that the impulse was too narrow for pointwise decoding of calcium activity. The definition was revised to cover the changed-image flash" — "less dependent on an arbitrary sampling phase than a one-bin impulse and gives the pointwise decoder access to the immediate post-change neural interval" — raising sample validation balanced accuracy to 0.6289.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is required: the variable is natively binary, encoded as 0 = `no_change`, 1 = `change`, with `output_values` `['no_change', 'change']`. `validate_internal` asserts the values stay within {0,1}.

ii.
```python
"output_values": [
    IMAGE_VALUES,
    ["no_change", "change"],
    ...
]
...
assert y[1].min() >= 0 and y[1].max() <= 1
```

iii. The instruction defines the variable as binary ("Have value of 1 right after a change in image identity, otherwise 0"); the only interpretive choice — the *width* of the "right after" window — is documented under 4-b.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Identically to image identity: built on the shared `all_t` 30 Hz grid with the same `searchsorted` + interval-validity logic, and split per trial with the same lengths, so column *t* of the output matrix corresponds to column *t* of the neural matrix.

ii.
```python
change_trials = split_vector(s["change"], lengths)
out_trials.append(np.vstack([
    image_trials[i], change_trials[i], run_trials[i], pupil_trials[i],
    np.full(int(length), s["outcome"][i], dtype=np.int16),
]).astype(np.int16, copy=False))
```

iii. Step 7 plot review: "Each change impulse lies on the onset of the changed image." Step 10 check 10: "image change never occurs during gray" — i.e. the change flag is exactly co-extensive with a non-gray flash, which is the intended alignment.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `exp.running_speed`, i.e. the SDK-provided filtered running speed DataFrame (`timestamps`, `speed` in cm/s). The raw encoder voltage is not re-processed and the signal is not re-filtered.

ii.
```python
running = exp.running_speed
run_all = finite_interp(running["timestamps"], running["speed"],
                        all_t, "running speed")
```

iii. Step 3/Step 4: "Reference processing unwraps voltage, removes >5.1 V artifacts, corrects wrap points…, then applies a 10 Hz low-pass Butterworth filter. The SDK's `running_speed` is this filtered signal and should be used rather than recomputation."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Non-finite samples are dropped, timestamps are sorted and de-duplicated, at least two finite samples are required, and the signal is linearly interpolated (`np.interp`, nearest-edge clamping outside the recorded range) onto the concatenated 30 Hz trial grid as float32. The continuous values are then discretized (see 5-c). No smoothing or re-filtering.

ii.
```python
def finite_interp(source_t, source_x, target_t, label: str):
    good = np.isfinite(source_t) & np.isfinite(source_x)
    if good.sum() < 2:
        raise ValueError(f"fewer than two finite {label} samples")
    t = source_t[good]; x = source_x[good]
    order = np.argsort(t, kind="stable")
    t, x = t[order], x[order]
    unique = np.r_[True, np.diff(t) > 0]
    t, x = t[unique], x[unique]
    if len(t) < 2:
        raise ValueError(f"fewer than two unique-time {label} samples")
    return np.interp(target_t, t, x).astype(np.float32)
```

iii. Step 5 mapping: "Interpolate SDK-filtered finite cm/s to grid". Step 5 decision 8: "Remove nonfinite source pairs, require at least two finite samples, then linearly interpolate; `np.interp` uses nearest finite edge only for rare trial-edge gaps." This mirrors the paper's own linear interpolation of running traces to 30 Hz.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five equal-percentile bins. After **all** sessions are extracted, the 20/40/60/80th percentiles are computed from the pooled running values of every retained trial (cohort-wide, not per session), the edges are checked for strict monotonicity, and every sample is assigned a bin 0–4 with `searchsorted(..., side='right')`. Cohort edges: [0.0428, 4.409, 22.643, 36.300] cm/s; realized bin fractions 0.200 each.

ii.
```python
def percentile_edges(values: list[np.ndarray], label: str) -> np.ndarray:
    joined = np.concatenate(values).astype(np.float32, copy=False)
    if not np.all(np.isfinite(joined)):
        raise ValueError(f"nonfinite {label} values after interpolation")
    edges = np.quantile(joined, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
    if np.any(np.diff(edges) <= 0):
        raise ValueError(f"non-unique {label} quintile edges: {edges}")
    return edges
...
run_bin = np.searchsorted(run_edges, s["running"], side="right").astype(np.int16)
```

iii. Step 5 decision 7: "Estimate global cohort thresholds from all eligible-trial resampled values (not from excluded time or passive sessions), then apply fixed thresholds to all sessions. This implements equal cohort percentiles and preserves cross-session comparability." The edges are stored in `metadata['running_speed_quintile_edges_cm_per_s']`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. It is interpolated directly onto the shared `all_t` 30 Hz grid before trial splitting, so it uses exactly the same timestamps (and the same `lengths`/`offsets` split) as the neural matrix. Cross-stream alignment rests on the SDK's hardware-synchronized timestamps.

ii.
```python
run_all = finite_interp(running["timestamps"], running["speed"], all_t, "running speed")
...
run_trials = split_vector(run_bin, lengths)
```

iii. Step 3/Step 10: "All experimental clocks were acquired on one 100 kHz synchronization board… conversion should retain SDK-synchronized times and use ophys timestamps as required." The `--show-processing` plots overlay the continuous cm/s trace and its quintile staircase against the neural trace to demonstrate no offset.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `exp.eye_tracking`, specifically the **cleaned** `pupil_width` and `pupil_height` ellipse-fit columns plus `timestamps`. Pupil diameter is defined as `2 * max(pupil_width, pupil_height)`, i.e. the full major axis, because the SDK treats these fields as ellipse semi-axes. Blink/outlier frames need no explicit masking here because the SDK sets these cleaned columns to NaN wherever `likely_blink` is true, and the interpolation drops non-finite samples.

ii.
```python
eye = exp.eye_tracking
if eye is None or eye.empty:
    raise MissingRequiredData("eye tracking table is missing or empty")
# SDK/whitepaper define width and height as ellipse half-axes and diameter
# as the major full axis. Cleaned fields are NaN for likely blinks.
pupil_diameter = 2.0 * np.maximum(
    eye["pupil_width"].to_numpy(dtype=np.float64),
    eye["pupil_height"].to_numpy(dtype=np.float64),
)
```

iii. Step 3/Step 4: "Pupil processing uses DeepLabCut ellipse fits. The pupil is assumed circular under oblique projection; the major ellipse axis reflects pupil diameter. `pupil_area` and other cleaned measures are NaN for likely blink/outlier frames… Define diameter as `2 * max(pupil_width, pupil_height)` if fields are semi-axes (verified in Step 5/tutorial docs)… Do not use raw blink-contaminated fits." (This is corroborated by the SDK's own `compute_circular_area`, which uses `max(width, height)` as the *radius*, and by `filter_on_blinks`, which NaNs `pupil_width`/`pupil_height` on likely blinks.)

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Same pipeline as running speed: build the diameter series, drop non-finite (blink/missing) samples, sort/de-duplicate timestamps, require ≥2 finite samples, linearly interpolate onto the 30 Hz grid (bridging internal blink gaps, clamping at the edges), then discretize. Sessions with a completely empty eye table are excluded rather than imputed.

ii.
```python
try:
    pupil_all = finite_interp(eye["timestamps"], pupil_diameter,
                              all_t, "pupil diameter")
except ValueError as exc:
    raise MissingRequiredData(str(exc)) from exc
```

iii. Step 5 mapping: "Diameter = `2*max(width,height)` (fields are half-axes); interpolate finite non-blink values to grid; cohort-wide quintiles. Cleaned fields are NaN on likely blinks. Linear interpolation is consistent with paper resampling and avoids a forbidden sixth/missing class."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identically to running speed: cohort-wide 20/40/60/80th percentiles of all retained-trial pupil samples, monotonicity-checked, then `searchsorted(..., side='right')` → bins 0–4. Cohort edges [75.42, 84.94, 94.18, 107.17] pixels; realized fractions 0.200 each.

ii.
```python
pupil_edges = percentile_edges([s["pupil"] for s in raw_sessions], "pupil")
...
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

iii. Step 5 decision 7 (shared with running speed): global cohort quintiles for cross-session comparability; edges saved in `metadata['pupil_diameter_quintile_edges_pixels']`. Step 9 verified each bin holds exactly 20% (to within one sample).

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Interpolated onto the same `all_t` 30 Hz grid before trial splitting, using the eye camera's own hardware-synchronized timestamps, then split with the same per-trial lengths as the neural data.

ii.
```python
pupil_trials = split_vector(pupil_bin, lengths)
out_trials.append(np.vstack([
    image_trials[i], change_trials[i], run_trials[i], pupil_trials[i], ...
```

iii. Same synchronization argument as running speed (single 100 kHz sync board; SDK returns already-synchronized timestamps). The processing plots show the continuous diameter and its quintile staircase on the same time axis as the neural trace.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the SDK trials table: `hit`, `miss`, `false_alarm`, `correct_reject`, read for the eligible (go/catch, non-aborted, non-auto-rewarded) trials only.

ii.
```python
OUTCOME_COLUMNS = ["hit", "miss", "false_alarm", "correct_reject"]
...
outcome_bool = trials[OUTCOME_COLUMNS].to_numpy(dtype=bool)
if not np.all(outcome_bool.sum(axis=1) == 1):
    bad = trials.index[outcome_bool.sum(axis=1) != 1].tolist()
    raise ValueError(f"{experiment_id}: nonexclusive outcomes in {bad[:5]}")
outcomes = outcome_bool.argmax(axis=1).astype(np.int16)
```

iii. Step 1: `Trial._get_trial_data` "Defines go/catch/outcome flags; aborted and auto-reward trials are excluded from ordinary outcome categories." Step 5: "trial flags `hit`, `miss`, `false_alarm`, `correct_reject` → `output[4]`… Map to 0–3". The one-hot assertion is the AI's own integrity check that the four labels really are mutually exclusive on the eligible cohort.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. `argmax` over the one-hot boolean matrix gives an integer class 0–3 in the fixed order hit/miss/false_alarm/correct_reject; that static value is broadcast across all T time bins of the trial so it can live in the same `(5, T)` int16 output matrix as the time-varying variables. `validate_internal` asserts the row is constant within each trial. Trial counts: 13,569 hit / 23,574 miss / 814 false alarm / 4,513 correct reject.

ii.
```python
out_trials.append(np.vstack([
    image_trials[i], change_trials[i], run_trials[i], pupil_trials[i],
    np.full(int(length), s["outcome"][i], dtype=np.int16),
]).astype(np.int16, copy=False))
...
assert np.all(y[4] == y[4, 0])
```

iii. Step 5 decision 9: "Store a single int16 `(5,T)` matrix per trial. Trial outcome is repeated through time so static and time-varying variables coexist and the validator/trainer sees a consistent output dimension." Step 12 notes the consequence: because the trainer scores every timepoint, including seconds before the animal responds, a near-chance four-class score is expected "without leaking future response information".

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handling is explicit and fail-loud rather than silently imputing:
- **Empty eye-tracking table** → `MissingRequiredData`, session excluded and recorded in `metadata['excluded_sessions']` with a reason (3 sessions).
- **Non-finite / duplicated / unsorted behavioural samples** → dropped before interpolation; <2 usable samples → session excluded.
- **Blink frames** → already NaN in the SDK cleaned columns, so dropped; internal gaps are bridged by linear interpolation; grid points outside the recorded range take the nearest finite value rather than NaN.
- **Grid/array consistency** → event/timestamp length mismatch, non-one-hot outcomes, unknown image names, <2 eligible trials, empty trial grids, and non-monotonic ophys timestamps all raise.
- **Post-hoc validation** → `validate_internal` asserts shapes, dtypes, finiteness, per-class ranges and outcome constancy for every trial.
- **All-zero neural trials** (1,717) are deliberately kept, after confirming against an independent SDK reload that the source events really are zero.
Note a robustness caveat: only `MissingRequiredData` is caught in the driver; every other per-experiment exception (including the "<2 eligible trials" case that Step 5 describes as a session filter) aborts the entire conversion via `RuntimeError`. No such case occurred (minimum retained trials = 39).

ii.
```python
class MissingRequiredData(RuntimeError):
    """A published session cannot supply one of the required decoder outputs."""
...
except MissingRequiredData as exc:
    exclusions.append({"experiment_id": int(eid), "reason": str(exc)})
    print(f"[{done}/{len(ids)}] {eid}: EXCLUDED ({exc})", flush=True)
    continue
except Exception as exc:
    raise RuntimeError(f"experiment {eid} failed") from exc
...
if not np.all(np.isfinite(joined)):
    raise ValueError(f"nonfinite {label} values after interpolation")
...
assert np.all(np.isfinite(n)) and np.all(np.isfinite(y))
```

iii. Step 5 decision 8: "Sessions/trials without sufficient finite running or pupil coverage are excluded and logged rather than filled with arbitrary constants." Step 10: "Three empty eye tables are excluded/logged rather than fabricated… Trial grids use a half-open end and `ceil` followed by `<stop`, preventing off-by-one samples." Step 9: "inventing pupil labels would be less defensible than this 1.8% session exclusion."

## 9-a. What are the most time-consuming steps of the code?

i. The AI measured per-experiment timing (`result['seconds']`, printed for every experiment) and identified SDK object construction — i.e. `get_behavior_ophys_experiment` plus decompression of the cached trace arrays — as the dominant cost (3.7–10.5 s per experiment, scaling with cell count). Secondary costs are the final assembly and writing the 7.6 GiB pickle. Mitigation: 16-way `ProcessPoolExecutor` parallelism over experiments; total full conversion 73.3 s for 168 experiments.

ii.
```python
t0 = time.perf_counter()
...
"seconds": time.perf_counter() - t0,
...
workers = min(16, len(ids), max(1, (os.cpu_count() or 2) // 2))
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(extract_session, eid): eid for eid in ids}
```

iii. Step 6: "Loading all 140k-frame traces in the parent sequentially would also underuse available CPU/I/O… Iteration 2 showed the dominant full-cache cost was instead SDK trace decompression in large experiments. With 1 TiB RAM and many CPUs available, independent workers were increased from 4 to 16 for iteration 3." Step 7 recorded the extrapolated full-run estimate (7–12 min) against the 15-minute budget, and the realized run beat it.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already removed the expensive loop: per-neuron (and per-neuron × per-trial) interpolation was replaced with a single shared timestamp-bracketing computation reused across all cells. Loops that remain and *could* still be vectorized:
- `for row in task.itertuples()` over ~4,800 stimulus presentations per session for image/change labelling — could be done with two `searchsorted` calls on arrays of `start_time`/`end_time` plus `np.repeat`/index arithmetic.
- the `_trial_grid` list comprehension over trials (grids could be built from a single `arange` with per-trial offsets).
- the per-trial assembly loop in `finalize` (`np.vstack` per trial) and `split_vector`'s per-trial slicing.
None of these dominate runtime, which is I/O-bound.

ii.
```python
for row in task.itertuples():
    start = float(row.start_time)
    end = float(row.end_time)
    lo = int(np.searchsorted(all_t, start, side="left"))
    hi = int(np.searchsorted(all_t, end, side="left"))
    ...
```
```python
for i, length in enumerate(lengths):
    out_trials.append(np.vstack([...]))
    in_trials.append(np.empty((0, int(length)), dtype=np.float32))
```

iii. Step 6: "Naively calling interpolation separately for every neuron × trial repeats binary searches and Python overhead… Step 9 iteration 1 revealed that even one `np.interp` call per cell redundantly searched identical timestamps and was replaced with shared vectorized brackets/weights (verified `np.allclose`, maximum difference 4.77e-7)." Step 7: "Concatenated trial grids / one interpolation per stream… Avoids roughly 229 × 231 separate trial-neuron interpolation calls in this sample." The remaining loops are left in place implicitly because the profile showed loading dominates.

## 9-c. What processing does the code repeat multiple times?

i. Very little per-session work is repeated — each stream is loaded once and interpolated once, and trial slicing reuses cached offsets. What *is* repeated:
- `make_cache()` (i.e. `VisualBehaviorOphysProjectCache.from_s3_cache`, which re-reads the manifest and re-imports the SDK) runs once in the parent **and once inside every worker task**, i.e. 168 additional times, as a consequence of process-based parallelism.
- `cache.get_ophys_experiment_table()` is called in `local_experiment_ids` and again in `main` when `--sample` is used.
- `exp.metadata` project_code is re-checked per experiment even though the table was already filtered on `project_code`.
- `searchsorted` over `all_t` is redone for each of the ~4,800 stimulus presentations per session (see 9-b).
All of these are cheap relative to trace decompression.

ii.
```python
def extract_session(experiment_id: int) -> dict:
    ...
    cache = make_cache()          # re-created in every worker task
    exp = cache.get_behavior_ophys_experiment(int(experiment_id))
    meta = exp.metadata
    if meta["project_code"] != "VisualBehavior":   # already filtered in main
        raise ValueError(...)
```
```python
ids = local_experiment_ids(cache)       # calls get_ophys_experiment_table()
if args.sample:
    table = cache.get_ophys_experiment_table().loc[ids]   # again
```

iii. Not explicitly discussed in CONVERSION_NOTES; the notes justify the parallel design generally ("Detailed SDK object construction includes compressed trace reads. It is I/O/decompression-bound on the full cache; moderate process parallelism is substantially faster"), which is the reason the cache object must be rebuilt per worker task. The redundant project-code check is presented as a defensive assertion ("The implementation validates project/trials/outcomes").

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Nothing expensive is computed and thrown away; the extras are small:
- Per-session bookkeeping that the decoder never reads — `trial_ids`, `trial_starts`, `trial_stops`, `cell_ids` (only its length is used), `image_set`, `experience_level`, `source_trial_count`, `go_count`, `catch_count`, `outcome_counts`, `ophys_frame_rate` — retained only for `metadata['session_info']`/provenance.
- The `2.0 *` factor in the pupil diameter is discarded by the quintile discretization (percentile bins are invariant to a positive scale factor); only the `max(width, height)` part actually affects the labels.
- Continuous running/pupil traces are interpolated at full float precision and then collapsed to 5 integer bins; the continuous values are not stored in the output.
- `validate_internal` re-walks every trial of every session after assembly (cheap but purely defensive), and `--show-processing` re-splits raw vectors for plotting.
- Storage-wise, the 99.7%-sparse event matrices are written as dense float32 (7.6 GiB pickle); a sparse or lower-precision representation would carry the same information.

ii.
```python
session_info.append({k: s[k] for k in [
    "experiment_id", "ophys_session_id", "mouse_id", "region",
    "session_type", "image_set", "experience_level", "source_trial_count",
    "eligible_trial_count", "go_count", "catch_count", "outcome_counts",
    "ophys_frame_rate",
]})
```
```python
pupil_diameter = 2.0 * np.maximum(...)   # scale factor irrelevant after quintiles
...
pupil_bin = np.searchsorted(pupil_edges, s["pupil"], side="right").astype(np.int16)
```

iii. Step 5 decision 11: "Available but intentionally unused variables: ΔF/F, corrected/demixed/neuropil fluorescence, smoothed events, ROI masks, motion correction, lick/reward arrays, omissions, reaction time/latency, engagement, genotype, sex, depth, and projections are preserved in source but are neither requested decoder inputs nor outputs. Relevant session metadata will record IDs, experience, image set, cell count, and trial counts." Step 5 decision 10: "avoid retaining full fluorescence or images" — the design intent was explicitly to compute only what the target format needs.
