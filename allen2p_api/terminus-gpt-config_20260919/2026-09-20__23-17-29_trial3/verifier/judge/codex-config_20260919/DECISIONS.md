# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI constructs an AllenSDK `VisualBehaviorOphysProjectCache` at `/app/data`, gets the experiment table, discovers locally cached experiment IDs from filenames without opening NWB contents, restricts them to active/non-passive experiments, and loads each selected experiment through `get_behavior_ophys_experiment`. Full mode processes 202 active experiments, in up to four worker processes.

ii.
```python
cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir=DATA_DIR)
table = cache.get_ophys_experiment_table()
ids = local_experiment_ids(table)
local = table.loc[ids]
active = local[(local["behavior_type"] == "active_behavior") & (~local["passive"].astype(bool))]
...
exp = cache.get_behavior_ophys_experiment(int(eid))
```

iii. The notes say scientific reads must use the required SDK, passive sessions do not contain the requested active task, and filesystem inspection is used only to avoid fetching experiments absent from the supplied local cache.

## 1-b. How are the data split into subjects?

i. Subjects are unique string-valued `mouse_id`s among retained experiments, sorted globally; `subject_idx` maps every experiment-level session to that list.

ii.
```python
subjects = sorted({s["mouse_id"] for s in prepared})
subject_map = {x: i for i, x in enumerate(subjects)}
"subject_idx": np.array([subject_map[s["mouse_id"]] for s in prepared], dtype=np.int16),
```

iii. The AI identifies `mouse_id` as the SDK animal identifier and reports that all 38 active-subset mice remain after QC.

## 1-c. How are the data split into sessions?

i. One output session is one `ophys_experiment_id`, i.e. one imaging plane. Simultaneous planes sharing an `ophys_session_id` are not merged.

ii.
```python
for i, eid in enumerate(ids, 1):
    sess = prepare_experiment(cache, eid, active.loc[eid])
...
"session_unit": "ophys experiment (one imaging plane)",
```

iii. The notes justify plane-level sessions because the paper summarizes decoders over imaging planes, each plane has its own cells/timestamps, and merging multiscope planes would require artificial cross-plane interpolation.

## 1-d. How are the data split into trials?

i. Trials come from `exp.trials`; each retained row is represented from SDK `start_time` through `stop_time` on bin centers `start + 0.05 + 0.1*k`, producing variable-length trial arrays.

ii.
```python
trials = exp.trials.copy()
...
def trial_grid(start, stop):
    n = int(np.floor((float(stop) - float(start)) / BIN_SEC + 1e-9))
    return float(start) + BIN_SEC * (np.arange(n, dtype=np.float64) + 0.5)
```

iii. The SDK trial table is treated as authoritative. Full trial windows retain pre- and post-change activity and permit time-varying stimulus/behavior outputs.

## 1-e. How are trials filtered based on quality controls?

i. The AI keeps go or catch trials, removes aborted and auto-rewarded trials, requires exactly one canonical outcome, a nonempty grid fully supported by ophys timestamps, and complete finite running and pupil values after interpolation. Experiments with fewer than two retained trials are dropped.

ii.
```python
eligible = ((trials["go"].fillna(False) | trials["catch"].fillna(False))
            & ~trials["aborted"].fillna(False)
            & ~trials["auto_rewarded"].fillna(False))
...
if sum(outcome_flags) != 1: continue
if not np.isfinite(run).all(): continue
if not np.isfinite(pup).all(): continue
...
if len(sess["records"]) >= 2: prepared.append(sess)
```

iii. The first exclusions directly implement the task. Complete-stream QC avoids inventing output categories for missing values; exclusion counts are recorded. The notes report 48,112 of 51,992 eligible trials and 199 of 202 experiments retained.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity comes from SDK-released `exp.dff_traces["dff"]`, ordered to match `exp.cell_specimen_table`, with `exp.ophys_timestamps` as its source time axis.

ii.
```python
cells = experiment.cell_specimen_table
dff = experiment.dff_traces
if not cells.index.equals(dff.index):
    dff = dff.loc[cells.index]
traces = np.stack(dff["dff"].to_numpy()).astype(np.float32, copy=False)
```

iii. The AI says released dF/F already embodies the Allen processing pipeline and is a dense, documented, framewise neural signal suitable for this decoder.

## 2-b. How is the `neural` data processed?

i. SDK dF/F is validated for shape/finiteness and linearly interpolated from native ophys timestamps onto every trial's 100 ms bin centers. It is stored as float32; no normalization or filtering is added.

ii.
```python
neural = linear_sample_matrix(ophys_t, dff, grid).astype(np.float32, copy=False)
...
return matrix[:, lo] * (1.0 - alpha)[None, :] + matrix[:, hi] * alpha[None, :]
```

iii. The common grid satisfies the same-bin-size requirement across native rates of about 10.7–31 Hz. The AI chose 10 Hz to avoid materially upsampling the slowest planes.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only SDK-curated cells present in `cell_specimen_table`/`dff_traces` are retained. An experiment fails if trace ordering/count, trace length versus timestamps, or finiteness is invalid; there is no additional cell-level criterion.

ii.
```python
if traces.shape[0] != len(cells):
    raise ValueError("cell table and dF/F row count differ")
if not np.isfinite(traces).all():
    raise ValueError("nonfinite values in released dF/F traces")
if dff.shape[1] != ophys_t.size:
    raise ValueError("dF/F trace length differs from ophys timestamps")
```

iii. The notes state the SDK table already represents cells passing released ROI curation, so redoing segmentation/filtering would diverge from the reference pipeline.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. It is aligned to trial start: 100 ms centers are anchored 50 ms after `start_time`, expressed on the absolute ophys clock, and dF/F is interpolated to those times until `stop_time`.

ii.
```python
grid = trial_grid(tr.start_time, tr.stop_time)
neural = linear_sample_matrix(ophys_t, dff, grid)
...
"temporal_alignment_event": "trial start; 100 ms bin centers represented in absolute ophys timestamp coordinates",
```

iii. The AI says this preserves the full SDK trial and guarantees every stream is compared at identical timestamps.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Every session uses a 100 ms grid (10 Hz). This is temporal resampling by linear interpolation, not aggregation/averaging within bins.

ii.
```python
BIN_SEC = 0.100
...
"time_bin_size": 100.0,
```

iii. The notes explicitly corrected an earlier native-rate plan because the target requires a common bin size; 100 ms is just below the slowest native plane rate.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. It is derived from change-detection rows of `exp.stimulus_presentations`, principally `image_name`, `start_time`, `end_time`, and `omitted`.

ii.
```python
stim = exp.stimulus_presentations
stim = stim[stim["stimulus_block_name"].str.contains("change_detection", na=False)]
...
name = str(row.image_name)
omitted = bool(row.omitted) if pd.notna(row.omitted) else False
```

iii. The AI uses presentation intervals rather than only trial initial/change names so that identity reflects what is actually visible and excludes unrelated stimulus blocks.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. Each grid center within a non-omitted presentation interval receives its image name; all other centers are `none/gray`. Real names are globally sorted and integer-coded after reserving code 0 for gray/no image.

ii.
```python
mask = (grid >= float(row.start_time)) & (grid < float(row.end_time))
image[mask] = name
...
image_values = ["none/gray"] + real_images
image_to_code = {name: i for i, name in enumerate(image_values)}
```

iii. The notes cite the task's 250 ms image/500 ms gray cycle and argue that “image presented during the non-grey screen” requires an explicit gray category outside visible intervals.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Presentation interval membership is evaluated directly at the same 100 ms `grid` used to interpolate neural data.

ii.
```python
neural = linear_sample_matrix(ophys_t, dff, grid)
image_names, changes = visible_images_and_changes(stim, grid)
```

iii. A shared absolute grid makes image and neural columns one-to-one; independent source checks in the notes passed.

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. It uses `stimulus_presentations.start_time`, `is_change`, and `omitted`, after selecting the change-detection stimulus block.

ii.
```python
is_change = bool(row.is_change) if pd.notna(row.is_change) else False
if is_change and not omitted:
    k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
```

iii. The AI states SDK `is_change` distinguishes real change onsets from catch/sham events and agrees with trial `change_time` in its checks.

## 4-b. What processing is involved in computing `output` *Image change*?

i. A zero vector is created and the single 100 ms bin containing each real, non-omitted change onset is set to one.

ii.
```python
change = np.zeros(grid.size, dtype=np.int8)
...
if 0 <= k < grid.size:
    change[k] = 1
```

iii. The rationale interprets “right after a change” as an onset event, rather than labeling the full changed-image interval or all post-change time.

## 4-c. How is `output` *Image change* thresholded into categories?

i. It is already binary: 0 means no change onset and 1 means a real change onset. No numeric threshold is estimated.

ii.
```python
"output_values": [
    ...,
    ["no_change", "change"],
]
```

iii. The source flag is categorical, so direct binary encoding is appropriate; catch and omitted presentations remain zero.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The stimulus onset is converted to the index of the containing 100 ms trial-grid bin, the same grid used for neural interpolation.

ii.
```python
k = int(np.floor((float(row.start_time) - (grid[0] - BIN_SEC / 2)) / BIN_SEC))
```

iii. The notes report exact agreement in independent raw-to-converted checks.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. It comes from SDK `exp.running_speed`, using `timestamps` and `speed`.

ii.
```python
running = exp.running_speed.sort_values("timestamps")
run = interp_vector(running["timestamps"], running["speed"], grid)
```

iii. This is the SDK-processed wheel-speed stream used by the paper/reference ecosystem.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Finite speed samples are linearly interpolated without extrapolation to the 100 ms grid. Global 0/20/40/60/80/100 percentiles over retained bins define five categories.

ii.
```python
out[inside] = np.interp(target_t[inside], st, sv).astype(np.float32)
...
edges = np.percentile(x, [0, 20, 40, 60, 80, 100])
return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
```

iii. The paper also interpolates running to a common timebase; global percentile edges provide consistent and approximately balanced classes across sessions.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Four global interior percentile edges split speed into quintiles 0–4; ties go to the upper bin. Non-increasing edges cause an error.

ii.
```python
if np.any(np.diff(edges) <= 0):
    raise ValueError(...)
return np.searchsorted(edges[1:-1], x, side="right").astype(np.int16)
```

iii. This directly implements five equal percentile bins, with one mapping shared by the full dataset.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Speed is interpolated by timestamp to the identical per-trial grid used for dF/F.

ii.
```python
run = interp_vector(running["timestamps"], running["speed"], grid)
neural = linear_sample_matrix(ophys_t, dff, grid)
```

iii. Timestamp interpolation avoids row-index assumptions and makes output length equal neural time length.

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. It uses `eye_tracking.timestamps`, `pupil_width`, `pupil_height`, and `likely_blink`.

ii.
```python
pupil = np.sqrt(
    eye["pupil_width"].to_numpy(float) * eye["pupil_height"].to_numpy(float)
)
blink = eye["likely_blink"].fillna(True).to_numpy(bool)
pupil[blink] = np.nan
```

iii. The geometric mean of ellipse axes is presented as an orientation-invariant equivalent-area diameter; blink fits are considered invalid.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. The ellipse-derived diameter is blink-masked, linearly interpolated to the trial grid only across gaps no longer than 0.5 s and without extrapolation, then globally percentile-binned.

ii.
```python
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
...
pupil_edges = percentile_edges([r["pupil"] for s in prepared for r in s["records"]])
```

iii. The 0.5 s guard prevents interpolation across long blink/missing periods; global quintiles fulfill the requested categorical output.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Global 20th-percentile boundaries split all retained aligned diameters into codes 0–4, just as for running speed.

ii.
```python
edges = np.percentile(x, [0, 20, 40, 60, 80, 100])
...
discretize(rec["pupil"], pupil_edges)
```

iii. This yields five pooled equal-occupancy bins and consistent codes across experiments.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Pupil is timestamp-interpolated to the same 100 ms trial grid as neural activity; trials with any unresolved pupil sample are discarded.

ii.
```python
pup = interp_vector(eye["timestamps"], pupil, grid, max_gap=MAX_PUPIL_GAP_SEC)
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1
    continue
```

iii. The AI prefers exclusion to inventing a missing category inconsistent with the specified five bins.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. It is derived from the trial-table booleans `hit`, `miss`, `false_alarm`, and `correct_reject`.

ii.
```python
OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]
outcome_flags = [bool(tr[x]) if pd.notna(tr[x]) else False for x in OUTCOMES]
```

iii. These are the SDK's canonical mutually exclusive outcomes for valid go/catch trials.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. Rows without exactly one true outcome are excluded. The true flag's position gives code 0–3, and that static code is repeated at every time bin.

ii.
```python
if sum(outcome_flags) != 1:
    exclusion["ambiguous_outcome"] += 1
    continue
...
np.full(T, rec["outcome"], dtype=np.int16)
```

iii. Repetition makes the mixed static/time-varying outputs a uniform `(5,T)` array while preserving a constant per-trial label.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI validates shapes, ordering, finite dF/F, time support, unique outcomes, and percentile edges. It masks blinks, permits pupil interpolation only over short gaps, rejects trials with unresolved running/pupil or unsupported grids, catches experiment-level exceptions, records failures/exclusion reasons, and drops sessions with fewer than two trials.

ii.
```python
if not np.isfinite(run).all():
    exclusion["missing_running"] += 1; continue
if not np.isfinite(pup).all():
    exclusion["missing_pupil"] += 1; continue
...
except Exception as exc:
    failed.append((eid, f"{type(exc).__name__}: {exc}"))
```

iii. The stated policy is to avoid extrapolation or fabricated categories and make every loss auditable. Three experiments were ultimately excluded for unusable pupil streams.

## 9-a. What are the most time-consuming steps of the code?

i. Loading each experiment through AllenSDK and interpolating/duplicating dense neural matrices for tens of thousands of trials dominate. The AI parallelizes experiment preparation with four processes; final pickle serialization and later loading are also costly because the output is about 2.47 GiB.

ii.
```python
with ProcessPoolExecutor(max_workers=workers) as pool:
    futures = {pool.submit(prepare_experiment_worker, eid): eid for eid in ids}
...
neural = linear_sample_matrix(ophys_t, dff, grid)
```

iii. The notes identify the large dense, trial-sliced dF/F payload as expected and use process-level parallelism because experiments are independent.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The loop over trials, the loop over relevant stimulus presentations in `visible_images_and_changes`, the list comprehension mapping every image name to a code, and final nested validation/count loops could be reduced or vectorized. The neural interpolation within a trial is already vectorized over cells.

ii.
```python
for trial_id, tr in trials.iterrows():
...
for row in relevant.itertuples():
...
image = np.array([image_to_code.get(x, 0) for x in rec["image_names"]])
```

iii. The AI's notes emphasize parallelizing the more consequential experiment-level work; the remaining loops handle variable-length records and are not identified as the primary bottleneck.

## 9-c. What processing does the code repeat multiple times?

i. Each simultaneous imaging plane reloads identical behavioral trial, stimulus, running, and eye tables because every experiment is processed independently. Full dF/F is interpolated separately for every trial, and trial neural slices duplicate overlapping pre-change periods. Percentile collection traverses all retained records, then assembly traverses them again, followed by another traversal for class counts.

ii.
```python
exp = cache.get_behavior_ophys_experiment(int(eid))
...
for trial_id, tr in trials.iterrows():
    neural = linear_sample_matrix(ophys_t, dff, grid)
...
run_edges = percentile_edges([r["running"] for s in prepared for r in s["records"]])
...
for session in data["output"]:
```

iii. The plane-level session decision intentionally repeats shared behavioral processing to preserve independent plane timestamps. The multiple output passes support global mappings and validation.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It stores per-record `grid`, `trial_id`, `start_time`, and `stop_time` during preparation, but these are not placed in the final dataset (except aggregated session metadata). It also builds detailed exclusion/session info and optionally plots data that the decoder does not consume. Trial outcome is redundantly expanded across time, although this is useful for validator-compatible shapes.

ii.
```python
records.append({"trial_id": int(trial_id), "grid": grid, ...,
                "start_time": float(tr.start_time), "stop_time": float(tr.stop_time)})
...
out = np.vstack([... np.full(T, rec["outcome"], dtype=np.int16)])
```

iii. The temporary fields support diagnostics and plotting; repeated outcome values avoid a ragged mixed static/time-varying output representation. The AI therefore treats most of this as validation/formatting overhead rather than scientific processing.
