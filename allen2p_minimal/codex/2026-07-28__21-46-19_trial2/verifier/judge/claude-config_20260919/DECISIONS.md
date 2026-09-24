# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI uses the AllenSDK `VisualBehaviorOphysProjectCache.from_s3_cache(cache_dir='/app/data')` and `cache.get_ophys_experiment_table()` as the master listing. Because the supplied cache is a read-only *partial* mirror (the manifest lists 1,936 experiments but only 284 NWB files exist on disk), it globs the on-disk NWB filenames and intersects the experiment table with those IDs. It then applies four filters: experiment present locally, `project_code in {VisualBehavior, VisualBehaviorMultiscope}`, `behavior_type == 'active_behavior'`, and `targeted_structure in {VISp, VISl}`. Each surviving *experiment* (imaging plane) is loaded individually with `cache.get_behavior_ophys_experiment(exp_id)`; no grouping into multi-plane sessions is performed. This yields 202 experiments / 174 ophys sessions / 38 mice, of which 199 survive per-experiment quality skips.

ii.
```python
def get_local_experiment_ids(cache_dir: Path) -> set[int]:
    experiment_dir = cache_dir / "visual-behavior-ophys-1.1.0" / "behavior_ophys_experiments"
    return {
        int(path.stem.split("_")[-1])
        for path in experiment_dir.glob("behavior_ophys_experiment_*.nwb")
    }

def select_experiments(cache, cache_dir) -> pd.DataFrame:
    experiment_table = cache.get_ophys_experiment_table().reset_index()
    local_ids = get_local_experiment_ids(cache_dir)
    selected = experiment_table[
        (experiment_table["ophys_experiment_id"].isin(local_ids))
        & (experiment_table["project_code"].isin(["VisualBehavior", "VisualBehaviorMultiscope"]))
        & (experiment_table["behavior_type"] == "active_behavior")
        & (experiment_table["targeted_structure"].isin(["VISp", "VISl"]))
    ].copy()
    selected.sort_values(by=["mouse_id", "date_of_acquisition", "session_type",
                             "targeted_structure", "imaging_depth", "ophys_experiment_id"],
                         inplace=True)
```
```python
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
```

iii. From `CONVERSION_NOTES.md`: "The manifest ... describes many more experiments than are actually present in this workspace. The AllenSDK cache directory is also mounted read-only, so missing NWB files cannot be downloaded ... the converter intentionally restricts the cohort to the experiments that are already present locally ... This is the only reproducible choice in this environment." The trajectory confirms the AI discovered this the hard way: its first sample run failed because the SDK tried to materialize missing NWB files into a read-only mount (step ~"the AllenSDK cache is trying to materialize missing NWB files back into `/app/data`, and that path is mounted read-only"). It originally planned the paper's exact cohort (`project_code == 'VisualBehaviorMultiscope'`, Familiar only — the paper states "For neural analysis we used neurons recorded during familiar image set presentations on the multi-plane imaging rig") but abandoned it: "In the local on-disk subset provided here, that exact paper-style familiar multiscope subset is not broadly available; only a small fragment of it is present locally. Using only that fragment would have produced a distorted dataset." It therefore fell back to the task wording ("collect and convert data under the Visual Behavior task") and kept all locally available *active* Visual Behavior ophys experiments. Passive sessions are excluded because the decoder targets (trial outcome, licking-driven behaviour) are only meaningful under active behaviour.

## 1-b. How are the data split into subjects?

i. Subjects are the unique `mouse_id` values, read from each loaded dataset's `ds.metadata['mouse_id']`, sorted as strings. `subject_idx` maps each converted session to its mouse. The final dataset has 38 subjects.

ii.
```python
mouse_id=str(ds.metadata["mouse_id"]),
...
subjects = sorted({s.mouse_id for s in session_data})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
...
subject_idx.append(subject_to_idx[session.mouse_id])
```

iii. Not discussed explicitly in the notes beyond reporting "Subjects: 38". `mouse_id` is the SDK's canonical animal identifier, and the AI cross-checked cohort composition (cre line, session type, structure counts) via `summarize_selected` before converting.

## 1-c. How are the data split into sessions?

i. **One converted "session" == one AllenSDK `ophys_experiment` (one imaging plane)**, not one `ophys_session_id`. Multi-plane (Multiscope) recordings are therefore *not* merged: the 34 selected Multiscope experiments come from only 6 real recording sessions (all from a single mouse, 457841), so those 6 behavioural recordings appear up to 7 times each as independent "sessions" carrying identical trial labels but different neurons. Experiments are ordered by `mouse_id`, then `date_of_acquisition`. `brain_region_idx` is assigned per session from the plane's `targeted_structure` (VISp or VISl only, no depth).

ii.
```python
selected.sort_values(by=["mouse_id", "date_of_acquisition", "session_type",
                         "targeted_structure", "imaging_depth", "ophys_experiment_id"], inplace=True)
...
for idx, experiment_id in enumerate(selected["ophys_experiment_id"].tolist(), start=1):
    session = process_experiment(cache=cache, experiment_id=int(experiment_id), ...)
    if session is not None:
        session_data.append(session)
...
brain_region_idx.append(
    np.full(session.n_cells, brain_region_to_idx[session.targeted_structure], dtype=np.int64))
```

iii. The notes never justify keeping planes separate; they simply report "199 sessions" alongside "174 ophys sessions" in the selection summary, and `session_info` records both `ophys_experiment_id` and `behavior_session_id` per entry. The implicit rationale is that an imaging plane is the unit the paper analyses ("we report summary statistics as the mean +/- SEM over imaging planes", and the paper's decoder operates "for each imaging plane"). The AI did note "the paper ... combined V1/LM cells across depths" in `CONVERSION_NOTES.md` but did not implement that combination.

## 1-d. How are the data split into trials?

i. Trials come from the SDK `ds.trials` table. Kept trials are `(~aborted) & (~auto_rewarded) & (go | catch)`. The temporal extent of a trial is *not* `start_time`→`stop_time`; instead it is the set of active `change_detection` stimulus presentations whose `trials_id` equals that trial's index. Each presentation becomes one time bin, so trials are variable length (10–17 bins, mean 11.6 bins ≈ 8.7 s).

ii.
```python
trials = ds.trials.copy()
keep_trials = (~trials["aborted"]) & (~trials["auto_rewarded"]) & (trials["go"] | trials["catch"])
kept_trials = trials.loc[keep_trials].copy()
...
stim = make_change_detection_table(ds)
valid_trial_ids = set(int(x) for x in kept_trials.index.to_numpy())
stim = stim[stim["trials_id"].isin(valid_trial_ids)].copy()
stim.reset_index(drop=True, inplace=True)
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
    if trial_stim.empty:
        continue
    stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
```

iii. From the notes: "Trials come directly from `BehaviorOphysExperiment.trials` ... This matches the task instruction to include Go/Catch and exclude Aborted/Auto-rewarded trials." For the time axis: "the stimulus table already maps every image flash or omission back to a `trials_id`" (trajectory), and "image identity and change labels are naturally defined on these intervals". The AI explicitly fixed a bug here mid-run: "The first pass found an indexing bug in how filtered stimulus intervals were mapped back into per-trial arrays" — resolved by `reset_index(drop=True)` on the filtered stimulus table so positional indices line up with the pre-computed per-interval matrices.

## 1-e. How are trials filtered based on quality controls?

i. Trial-level: aborted, auto-rewarded, and any trial that is neither go nor catch are dropped; a kept trial with no matching stimulus intervals is skipped. Session-level: an experiment is dropped entirely if it has no valid ROI events, no eye-tracking table, an all-NaN pupil trace, or fewer than 2 kept trials. Three experiments (795953296, 806456687, 833631914) were dropped for missing eye tracking. No neuron-count minimum and no "all-zero neural activity" filter are applied (the final data contains sessions with as few as 4 neurons and 2,420 all-zero trials).

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events"); return None
if ds.eye_tracking.empty:
    print(f"Skipping {experiment_id}: no eye tracking data"); return None
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
if not np.isfinite(pupil_width).any():
    print(f"Skipping {experiment_id}: pupil width is entirely NaN"); return None
...
if kept_trials.empty:
    print(f"Skipping {experiment_id}: no kept go/catch trials"); return None
...
if trial_stim.empty:
    continue
...
if len(trial_data) < 2:
    print(f"Skipping {experiment_id}: fewer than 2 kept trials with stimulus intervals"); return None
```

iii. The notes justify the filter list ("6. During per-session loading, skip experiments with: no valid ROI events / no eye-tracking table / all-NaN pupil width / fewer than 2 kept trials after trial filtering") and explicitly defend *not* filtering sparse trials: "I left these trials in place because: they are scientifically plausible with event-based calcium data; removing them would be an undocumented extra curation step." The 2-trial minimum follows the instruction "There needs to be at least two trials within each session in order to evaluate the decoder performance."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `BehaviorOphysExperiment.events` — the SDK's detected/deconvolved calcium events (per-cell event magnitude time series on the ophys clock) — **not** `dff_traces`.

ii.
```python
event_matrix = np.vstack(ds.events["events"].values).astype(np.float32)
...
n_cells=int(event_matrix.shape[0]),
```

iii. From the notes: "This follows the paper's use of discrete calcium events rather than dF/F traces." The paper's methods (read at trajectory step 9) state: "For all analysis of neural data we used the detected calcium events ... We performed our analyses on discrete calcium events that were regressed from the raw fluorescence traces, thus removing the slow decay dynamics of the calcium indicator GCaMP6f." The AI also verified via `rg valid_roi|cell_specimen_table|events` that the SDK already excludes invalid ROIs from `events`.

## 2-b. How is the `neural` data processed?

i. For each image-presentation interval, the event magnitudes of every cell are **summed** over all ophys frames whose timestamp falls in `[interval_start, interval_end)`. The summation is vectorised with a cumulative sum along the time axis in float64. No normalisation, no z-scoring, no smoothing, no per-neuron filtering, and no merging of cells across imaging planes.

ii.
```python
def interval_reduce_sum_matrix(timestamps, matrix, starts, ends):
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate([np.zeros((matrix.shape[0], 1), dtype=np.float64),
                           np.cumsum(matrix, axis=1, dtype=np.float64)], axis=1)
    reduced = csum[:, end_idx] - csum[:, start_idx]
    return reduced.astype(np.float32)

neural_by_interval = interval_reduce_sum_matrix(
    timestamps=ophys_timestamps, matrix=event_matrix,
    starts=interval_starts, ends=interval_ends)
...
neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
```

iii. Notes: "Per interval, neural activity is: sum of event magnitudes across all ophys timestamps within that image interval." Summing (rather than averaging) is the natural aggregation for event magnitudes, which are sparse and zero at most frames. The AI observed and accepted the consequence — "The validator reports many warnings of the form `all neural data is zero` ... these are valid event-sum trials with no detected events across the recorded cells in that interval set."

## 2-c. How is the `neural` data filtered based on quality controls?

i. No neuron-level filtering is applied. The AI relies on the SDK's own ROI validity filtering, and only skips an experiment whose `events` table is empty.

ii.
```python
if len(ds.events) == 0:
    print(f"Skipping {experiment_id}: no valid ROI events")
    return None
```

iii. Notes: "The AllenSDK already excludes invalid ROIs from `events`/`cell_specimen_table`, so no additional ROI-quality filter was added." The AI verified this by grepping the SDK for `valid_roi|cell_specimen_table|events` (trajectory step 173).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the stimulus-flash grid, anchored at each trial's first image-presentation onset. Bin boundaries are successive `start_time` values of active `change_detection` stimulus presentations (the last presentation gets a boundary at `start_time + median_dt`). Events are assigned to bins using `np.searchsorted` on `ds.ophys_timestamps` with a half-open `[start, end)` convention, so the ophys timestamps determine which frames land in which bin. `metadata['temporal_alignment_event']` documents this; `off_start = 0.0`, `off_end = None`.

ii.
```python
def make_change_detection_table(ds):
    stim = ds.stimulus_presentations.copy()
    block_mask = stim["stimulus_block_name"].astype(str).str.contains("change_detection")
    active_mask = stim["active"].fillna(False).astype(bool)
    stim = stim.loc[block_mask & active_mask].copy()
    stim.sort_values("start_time", inplace=True)
    stim.reset_index(drop=False, inplace=True)
    start_times = stim["start_time"].to_numpy(dtype=np.float64)
    median_dt = float(np.median(np.diff(start_times)))
    interval_end = np.empty_like(start_times)
    interval_end[:-1] = start_times[1:]
    interval_end[-1] = start_times[-1] + median_dt
    stim["interval_end"] = interval_end
```
```python
"temporal_alignment_event": (
    "Successive image-presentation intervals defined by active "
    "change_detection stimulus onsets and assigned to trials via trials_id; "
    "modalities aggregated using ophys/running/eye timestamps within each interval."),
"off_start": 0.0,
"off_end": None,
```

iii. Notes: "take active `change_detection` stimulus presentations only; use successive stimulus onsets as bin boundaries; assign bins to trials using `stimulus_presentations.trials_id`". Rationale given: "the paper explicitly assigns behavioral events to each 750 ms image presentation interval ... this keeps a fixed task-relevant time step across sessions". The paper's methods indeed say: "By image presentation interval we refer to the 750 ms interval beginning with each image presentation. For image omissions we used the 750 ms following the time of the omission."

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. Yes — substantial rebinning. Native ophys sampling (~32 ms/frame single-plane, ~90 ms/frame Multiscope) is collapsed to one bin per image-presentation interval: `time_bin_size = 750.61 ms` (median interval duration; mean 750.67 ms). Trials are 10–17 bins (mean 11.60). This is ~23× coarser than the native single-plane frame rate. The AI validated bin uniformity by reporting median vs. mean interval duration, which agree to 0.06 ms.

ii.
```python
interval_durations = np.concatenate([s.interval_durations for s in session_data])
...
"time_bin_size": float(np.median(interval_durations) * 1000.0),
"image_interval_duration_median_s": float(np.median(interval_durations)),
"image_interval_duration_mean_s": float(np.mean(interval_durations)),
```

iii. Notes: "I did not use raw 2p frames as decoder bins. Instead, I matched the paper's image-interval analysis style ... Reasoning: the paper explicitly assigns behavioral events to each 750 ms image presentation interval; image identity and change labels are naturally defined on these intervals; this keeps a fixed task-relevant time step across sessions; it makes the dataset tractable for the supplied decoder." Sanity check reported: "Median bin duration `0.75061 s`, matching the expected flashed-image cadence" and "These agree with the whitepaper/paper task cadence of `250 ms` image plus `500 ms` gray."

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. `ds.stimulus_presentations['image_name']`, read directly per stimulus presentation (i.e. per time bin). Omitted flashes carry the literal value `"omitted"`, which is kept as a 17th category (16 real images + `omitted`). NaN names would fall back to `"unknown"`.

ii.
```python
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
```

iii. Notes: "`omitted` is retained as a category rather than being dropped, because omissions are an explicit part of the Visual Behavior task and are present in the SDK stimulus table." Deriving from the stimulus table rather than the trials table's `initial_image_name`/`change_image_name` follows from the AI's decision to bin on the flash grid — each bin corresponds to exactly one stimulus presentation, so the displayed image is read off directly.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A single global name→integer mapping is built across all sessions from the sorted set of image names, with `omitted` forced to sort last. Per-trial names are mapped to `int64` codes. `output_values[0]` stores the ordered names.

ii.
```python
image_values = sorted(
    {name for s in session_data for t in s.trials for name in t.image_names.tolist()},
    key=lambda x: (x == "omitted", x),
)
image_to_idx = {name: idx for idx, name in enumerate(image_values)}
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. A global mapping keeps codes consistent across sessions (the dataset spans image sets A and B, 16 images total). Sorting with `omitted` last keeps the 16 real images in a contiguous, deterministic block. Verified distribution in the notes/verification output: 16 images at ~0.055–0.066 each, `omitted` at 0.034, with the AI explaining the sub-5% omission rate: "omissions are disallowed for change and pre-change intervals" and aborted/auto-rewarded trials are excluded.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. It is defined on exactly the same interval grid: `stim_idx` indexes both `neural_by_interval` columns and the rows of `trial_stim`, so the image label in bin *t* is the image whose 250 ms flash starts that bin (and the label persists over the following 500 ms grey period, which belongs to the same bin).

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
trial_data.append(TrialData(
    neural=neural_by_interval[:, stim_idx].astype(np.float32, copy=False),
    image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
    ...))
```

iii. Alignment is structural rather than interpolated: because bins *are* stimulus presentations, no resampling is needed. This was the AI's stated motivation for the interval grid — "image identity and change labels are naturally defined on these intervals".

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. `ds.stimulus_presentations['is_change']`, the SDK's per-flash boolean marking the first presentation of a new image identity.

ii.
```python
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
```

iii. Notes: "Taken directly from `stimulus_presentations.is_change`." Using the SDK flag means catch (sham-change) trials correctly get all-zero change indicators, since no image identity change actually occurs on those trials.

## 4-b. What processing is involved in computing `output` *Image change*?

i. None beyond `NaN → False`, `bool → int64`. It is 1 in exactly the one 750 ms bin that begins with the changed image, 0 elsewhere.

ii.
```python
change_idx = trial.image_change.astype(np.int64)
...
"output_values": [image_values, ["no_change", "change"], ...]
```

iii. No justification is given beyond the choice of source variable; the single-bin encoding is a direct consequence of the 750 ms binning (the change flash occupies exactly one bin). The resulting class balance (7.5% change) is reported in the verification output and matches the expected ~1 change per ~11.6-bin trial.

## 4-c. How is `output` *Image change* thresholded into categories?

i. No thresholding is needed — the variable is already binary. Categories are named `["no_change", "change"]` with codes 0/1.

ii.
```python
"output_names": ["image_identity", "image_change", "running_speed_bin",
                 "pupil_diameter_bin", "trial_outcome"],
"output_values": [image_values, ["no_change", "change"], RUNNING_BIN_NAMES,
                  PUPIL_BIN_NAMES, TRIAL_OUTCOMES],
```

iii. Notes: "`0 = no_change`, `1 = change`". The instruction asked for a binary variable with "value of 1 right after a change in image identity, otherwise 0".

## 4-d. How is `output` *Image change* aligned with the neural data?

i. Same interval grid as neural data — indexed by the same `stim_idx`, so it is exactly co-registered with the neural bins with no interpolation.

ii.
```python
stim_idx = trial_stim.index.to_numpy(dtype=np.int64)
TrialData(neural=neural_by_interval[:, stim_idx], image_change=trial_stim["is_change"]...)
```

iii. See 3-c — alignment is structural because bins are stimulus presentations.

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `ds.running_speed`, using its `speed` (cm/s) and `timestamps` columns.

ii.
```python
running_df = ds.running_speed.copy()
running_t = running_df["timestamps"].to_numpy(dtype=np.float64)
running_v = running_df["speed"].to_numpy(dtype=np.float64).astype(np.float32)
```

iii. Notes: "source: `dataset.running_speed['speed']`". This is the SDK's standard locomotion interface, whose derivation from the wheel encoder is described in `methods.txt`.

## 5-b. What processing is involved in computing `output` *Running speed*?

i. The running trace is averaged over all running-encoder samples whose timestamps fall in `[interval_start, interval_end)`, again via a cumulative-sum trick. If an interval happens to contain no running samples, the value is linearly interpolated at the interval centre. No smoothing or outlier removal is applied (negative speeds are retained).

ii.
```python
def interval_reduce_mean(timestamps, values, starts, ends):
    start_idx = np.searchsorted(timestamps, starts, side="left")
    end_idx = np.searchsorted(timestamps, ends, side="left")
    csum = np.concatenate([np.array([0.0]), np.cumsum(values, dtype=np.float64)])
    counts = end_idx - start_idx
    sums = csum[end_idx] - csum[start_idx]
    out = np.empty(len(starts), dtype=np.float32)
    valid = counts > 0
    out[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    if (~valid).any():
        centers = (starts + ends) / 2.0
        out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)
    return out

running_by_interval = interval_reduce_mean(
    timestamps=running_t, values=running_v, starts=interval_starts, ends=interval_ends)
```

iii. Notes: "aggregated per image interval by the mean over samples in that interval". Averaging (rather than point sampling) is the appropriate reduction given the wide 750 ms bins and the ~60 Hz encoder sampling rate.

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Into five **exactly** equal-frequency bins (quintiles), computed globally over every time bin of every trial of every session pooled together. Ranking with `method="first"` before `qcut` breaks ties so the bins are exactly 20% each. The quintile edges are also stored in metadata.

ii.
```python
def rank_quintiles(values: np.ndarray) -> np.ndarray:
    ranks = pd.Series(values).rank(method="first")
    bins = pd.qcut(ranks, q=5, labels=False)
    return bins.to_numpy(dtype=np.int64)

all_running = np.concatenate([trial.running_raw for s in session_data for trial in s.trials])
running_bins = rank_quintiles(all_running)
...
running_quantiles = np.quantile(all_running, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
"running_bin_quantiles": [float(x) for x in running_quantiles],
```

iii. Notes: "Both variables are discretized into five equal-frequency bins across the full converted dataset by rank-based global quintiles. This guarantees five balanced categories even with ties." Ties matter here because running speed is exactly/near zero for a large fraction of bins; a plain percentile-edge `digitize` would collapse bins around 0. Verification confirms exactly `{q1 0.200, q2 0.200, q3 0.200, q4 0.200, q5 0.200}`.

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Running is reduced onto the identical interval grid before trials are cut, so the per-trial slice `running_by_interval[stim_idx]` is element-for-element aligned with `neural_by_interval[:, stim_idx]`. The global quintile codes are then sliced back into per-trial vectors with a running cursor that walks the trials in exactly the same order used to build the pooled array.

ii.
```python
running_raw=running_by_interval[stim_idx].astype(np.float32, copy=False),
...
run_idx = running_bins[running_cursor : running_cursor + t]
running_cursor += t
```

iii. Not explicitly justified; alignment is guaranteed by construction because all modalities are reduced onto the same `interval_starts`/`interval_ends` arrays using their own native timestamps (the hardware clocks are synced by the SDK).

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `ds.eye_tracking['pupil_width']` (with `ds.eye_tracking['timestamps']`). Blink frames are not explicitly removed because the SDK has already set them to NaN.

ii.
```python
if ds.eye_tracking.empty:
    print(f"Skipping {experiment_id}: no eye tracking data"); return None
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
if not np.isfinite(pupil_width).any():
    print(f"Skipping {experiment_id}: pupil width is entirely NaN"); return None
...
eye_df = ds.eye_tracking.copy()
eye_t = eye_df["timestamps"].to_numpy(dtype=np.float64)
```

iii. Notes: "this is already blink-filtered by the AllenSDK (`likely_blink` rows are NaN)" and "Why `pupil_width`: the task requested pupil diameter; `pupil_width` is the direct diameter-like quantity exposed by the SDK." The AI verified this by reading `allensdk/.../eye_tracking_table.py`, which calls `filter_on_blinks(eye_tracking_data)` in `from_nwb` — that function sets `pupil_width`/`pupil_height`/`pupil_area` to NaN wherever `likely_blink` is True.

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. (1) NaN samples (blinks/outliers) are filled by linear interpolation over eye-tracking time; (2) the filled trace is averaged over each image interval; (3) the pooled per-bin values across the whole dataset are discretised into five exactly equal-frequency rank quintiles, identically to running speed.

ii.
```python
def fill_nan_by_time(timestamps, values):
    values = values.astype(np.float32, copy=True)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError("Series has no finite values")
    if finite.all():
        return values
    values[~finite] = np.interp(timestamps[~finite], timestamps[finite], values[finite]).astype(np.float32)
    return values

pupil_filled = fill_nan_by_time(timestamps=eye_t, values=eye_df["pupil_width"].to_numpy(dtype=np.float64))
pupil_by_interval = interval_reduce_mean(timestamps=eye_t, values=pupil_filled,
                                         starts=interval_starts, ends=interval_ends)
...
all_pupil = np.concatenate([trial.pupil_raw for s in session_data for trial in s.trials])
pupil_bins = rank_quintiles(all_pupil)
```

iii. Notes: "NaNs are linearly interpolated over eye-tracking time within each session; interval value is the mean interpolated pupil width in that interval." Interpolating (rather than propagating NaN or mapping to a sentinel bin) keeps every bin usable and avoids contaminating one quintile with missing data.

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Five exactly equal-frequency global rank quintiles over all pooled bins (`rank_quintiles`), named `q1`–`q5`; the 0/20/40/60/80/100 percentile values are recorded in metadata.

ii.
```python
pupil_bins = rank_quintiles(all_pupil)
pupil_quantiles = np.quantile(all_pupil, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
"pupil_bin_quantiles": [float(x) for x in pupil_quantiles],
PUPIL_BIN_NAMES = [f"q{i}" for i in range(1, 6)]
```

iii. Same rationale as running speed — the instruction asks for "five equal percentile bins", and a single global binning keeps the category meaning constant across sessions. Verification confirms exactly 0.200 per bin.

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Identical mechanism to running speed: reduced onto the same interval grid using eye-tracking timestamps, sliced with the same `stim_idx`, and quintile codes re-sliced per trial with a `pupil_cursor` walking the trials in the same order as the pooled concatenation.

ii.
```python
pupil_raw=pupil_by_interval[stim_idx].astype(np.float32, copy=False),
...
pupil_idx = pupil_bins[pupil_cursor : pupil_cursor + t]
pupil_cursor += t
```

iii. See 5-d.

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the trials table — `hit`, `miss`, `false_alarm`, `correct_reject` — checked in that fixed priority order. If none is True the converter raises rather than silently assigning a fallback.

ii.
```python
TRIAL_OUTCOMES = ["hit", "miss", "false_alarm", "correct_reject"]

def infer_trial_outcome(trial_row: pd.Series) -> str:
    if bool(trial_row["hit"]): return "hit"
    if bool(trial_row["miss"]): return "miss"
    if bool(trial_row["false_alarm"]): return "false_alarm"
    if bool(trial_row["correct_reject"]): return "correct_reject"
    raise ValueError(f"Could not infer trial outcome for trial {trial_row.name}")
```

iii. Notes list the four categories as the trial-outcome classes. These are the SDK's canonical change-detection outcome labels and are exhaustive for the retained go/catch, non-aborted, non-auto-rewarded trials — which the AI evidently relied on, since it raises instead of defaulting. The full run completed without triggering the exception, empirically confirming exhaustiveness on this cohort.

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The outcome string is mapped to a fixed integer code 0–3 via `TRIAL_OUTCOMES` order and broadcast (`np.full`) across every time bin of the trial, so that a per-trial static variable can occupy a row of the same `(5, T)` output matrix as the time-varying outputs.

ii.
```python
outcome_to_idx = {name: idx for idx, name in enumerate(TRIAL_OUTCOMES)}
...
outcome_idx = np.full(t, outcome_to_idx[trial.trial_outcome], dtype=np.int64)
session_output.append(np.vstack([image_idx, change_idx, run_idx, pupil_idx, outcome_idx]).astype(np.int64))
```

iii. Notes: "The per-trial outcome is repeated across all bins in that trial so it can live in the same `(d_output, T)` array as the time-varying outputs." Resulting distribution (verification output): hit 0.303, miss 0.571, false_alarm 0.017, correct_reject 0.108.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. Handled cases:
- Missing eye tracking / all-NaN pupil / empty events / <2 usable trials → the whole experiment is skipped with a printed reason (3 experiments skipped for missing eye tracking).
- NaN pupil samples (blinks, tracking dropouts) → linearly interpolated over time.
- A kept trial with no matching stimulus presentations → skipped.
- An interval containing no running/eye samples → value interpolated at the interval centre instead of producing NaN or a divide-by-zero.
- Missing `image_name` → `"unknown"`; missing `is_change` → `False`; missing `active` flag → `False`.
- The last stimulus presentation of a session, which has no successor onset, gets a synthetic `interval_end` at `start_time + median_dt`.

Not handled: there is no `try/except` around `process_experiment`, so any unanticipated per-experiment failure (including the `ValueError` from `infer_trial_outcome`) aborts the entire conversion; and `running_speed` is not NaN-filled before the cumulative sum, so a single NaN sample would poison every subsequent interval in that session (this did not occur in practice on this cohort).

ii.
```python
active_mask = stim["active"].fillna(False).astype(bool)
...
interval_end[-1] = start_times[-1] + median_dt
...
image_names=trial_stim["image_name"].fillna("unknown").astype(str).to_numpy(),
image_change=trial_stim["is_change"].fillna(False).astype(bool).to_numpy(),
...
if (~valid).any():
    centers = (starts + ends) / 2.0
    out[~valid] = np.interp(centers[~valid], timestamps, values).astype(np.float32)
```

iii. The skips are enumerated and justified in `CONVERSION_NOTES.md` ("6. During per-session loading, skip experiments with: ..."), and each skipped experiment is listed by ID. Sparse/zero neural trials were deliberately retained: "they are scientifically plausible with event-based calcium data; removing them would be an undocumented extra curation step." The AI also ran `verify_data_format` inside the converter so that a malformed result fails loudly rather than being written to disk.

## 9-a. What are the most time-consuming steps of the code?

i. (1) By far the dominant cost is `cache.get_behavior_ophys_experiment()` — reading 202 NWB files (the AI measured roughly 180 GB of I/O during the full pass, which ran for well over 20 minutes). (2) Within each experiment, `np.cumsum(event_matrix, axis=1, dtype=np.float64)` allocates and fills an `(n_cells, n_ophys_frames+1)` float64 array for the *entire* session, which for a 666-neuron, ~140k-frame session is ~750 MB. (3) The per-trial `stim["trials_id"] == trial_id` scan is O(n_trials × n_intervals) per session.

ii.
```python
ds = cache.get_behavior_ophys_experiment(int(experiment_id))
...
csum = np.concatenate([np.zeros((matrix.shape[0], 1), dtype=np.float64),
                       np.cumsum(matrix, axis=1, dtype=np.float64)], axis=1)
...
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
```

iii. The AI repeatedly diagnosed the bottleneck as I/O-plus-CPU inside the NWB reads: "The full pass is still inside the bulk NWB reads, which is expected for the 200-session cohort"; "It has already streamed through roughly 180 GB of NWB reads and is still CPU-active, so this looks like a long but legitimate full pass rather than a deadlock." It chose to let it run rather than restart, so no optimisation was attempted.

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy aggregation is already vectorised (cumulative-sum interval reduction for neural, running, and pupil — no Python loop over time bins or neurons). What remains loop-bound:
- `for trial_id, trial_row in kept_trials.iterrows()` with an inner full-column boolean comparison `stim["trials_id"] == trial_id`; a single `stim.groupby("trials_id").indices` would replace the quadratic scan.
- `image_idx = np.array([image_to_idx[name] for name in trial.image_names])` — a per-bin Python loop; `pd.Categorical(...).codes` or a vectorised `np.searchsorted` against the sorted name list would do this in one shot for the whole dataset.
- The per-trial `np.vstack` of five one-row arrays in `build_decoder_dataset`.
- `for idx, row in selected.iterrows()` in `choose_sample_experiments`.

None of these is a real bottleneck relative to NWB loading.

ii.
```python
for trial_id, trial_row in kept_trials.iterrows():
    trial_stim = stim[stim["trials_id"] == trial_id].copy()
...
image_idx = np.array([image_to_idx[name] for name in trial.image_names], dtype=np.int64)
```

iii. The AI never discusses vectorisation trade-offs in its notes. Its implementation choices show it prioritised vectorising the per-frame aggregation (the only step that scales with the number of ophys frames × neurons) and left the per-trial bookkeeping as straightforward loops.

## 9-c. What processing does the code repeat multiple times?

i.
- `ds.eye_tracking` is accessed three separate times in `process_experiment` (emptiness check, pupil-width finiteness check, then `.copy()`); each access goes through the SDK's lazy property.
- Redundant defensive `.copy()` calls on `ds.trials`, `ds.stimulus_presentations`, `ds.running_speed`, `ds.eye_tracking`, and on each `trial_stim` slice.
- The per-trial linear scan over the whole `stim` table repeats the same comparison once per trial.
- `verify_data_format` is run inside the converter, and then again by `train_decoder.py` on the same pickle.
- The interval reductions are computed for *all* kept intervals, then re-indexed per trial; and the whole-session cumulative sum is computed even though only the kept intervals are needed.
- `np.quantile` recomputes bin boundaries for metadata after `rank_quintiles` has already effectively determined them.

ii.
```python
if ds.eye_tracking.empty: ...
pupil_width = ds.eye_tracking["pupil_width"].to_numpy(dtype=np.float64)
...
eye_df = ds.eye_tracking.copy()
...
valid, errors, warnings_list = verify_data_format(data)   # also run again by train_decoder.py
...
running_bins = rank_quintiles(all_running)
running_quantiles = np.quantile(all_running, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]).tolist()
```

iii. Not discussed. Each experiment is loaded exactly once, which is the expensive thing; the repeats above are all cheap relative to NWB I/O.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i.
- `interval_duration_all` is built inside the main assembly loop and then never read — pure dead code.
- `import math` is unused.
- `TrialData.trial_id`, `SessionData.trial_count_before_filter`, and `SessionData.trial_count_after_filter` are populated but never consumed anywhere.
- An empty `np.empty((0, t), dtype=np.float32)` input array is allocated for every one of the 51,075 trials, although the task specifies no decoder inputs.
- The whole-session `np.cumsum` covers ophys frames outside any kept trial (aborted/auto-rewarded trial periods, the 5-minute grey screens, the movie block).
- Neural/running/pupil values are computed for intervals belonging to trials that are later dropped (`trial_stim.empty`) and for experiments later skipped for having <2 trials.
- `choose_sample_experiments`, `make_sample_dataset`, `summarize_selected`, `omission_count`, and the extra `sample_data.pkl` write are bookkeeping/QA artefacts not used by the decoder.

ii.
```python
interval_duration_all = []
...
            interval_duration_all.append(t)   # never read
...
session_input.append(np.empty((0, t), dtype=np.float32))
...
    trial_count_before_filter=int(len(trials)),
    trial_count_after_filter=int(len(trial_data)),
```

iii. Not discussed. The sample-dataset machinery is deliberate and documented ("I'm regenerating `sample_data.pkl` in a deliberately small stratified mode so that the sample logs, verification, and decoder training all refer to the same file"); the dead variables and unused dataclass fields appear to be leftovers from iteration.
