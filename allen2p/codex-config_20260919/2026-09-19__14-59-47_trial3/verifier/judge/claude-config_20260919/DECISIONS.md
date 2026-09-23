# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI bypassed the AllenSDK object model entirely and read the released NWB (HDF5) files directly with `h5py`. It enumerates every `behavior_ophys_experiment_*.nwb` under `/app/data/visual-behavior-ophys-1.1.0/behavior_ophys_experiments/` (284 files locally) and joins them to `project_metadata/ophys_experiment_table.csv` on `ophys_experiment_id` for mouse/region/session metadata. It did **not** filter by `project_code`, so both `VisualBehavior` (239 files) and `VisualBehaviorMultiscope` (45 files) experiments enter the candidate pool. Each candidate is then screened: it must be present in the metadata table, have `behavior_type == "active_behavior"` (82 passive-viewing files dropped), contain an `acquisition/EyeTracking` group (3 more files dropped), and contain the required datasets (`processing/ophys/event_detection/{data,timestamps}`, `processing/running/speed/{data,timestamps}`, `intervals/trials`). 199 experiments survive, covering 38 mice and 48,909 trials. Only the specific HDF5 datasets needed are read; masks, projections and segmentation images are never touched.

ii.
```python
APP_ROOT = Path("/app")
DATA_ROOT = APP_ROOT / "data" / "visual-behavior-ophys-1.1.0"
NWB_ROOT = DATA_ROOT / "behavior_ophys_experiments"
METADATA_PATH = DATA_ROOT / "project_metadata" / "ophys_experiment_table.csv"

def choose_experiments(sample: bool):
    metadata = pd.read_csv(METADATA_PATH).set_index("ophys_experiment_id", drop=False)
    paths = sorted(NWB_ROOT.glob("behavior_ophys_experiment_*.nwb"))
    available = {experiment_id_from_path(path): path for path in paths}
    ...
    for eid in candidate_ids:
        row = metadata.loc[eid]
        if row["behavior_type"] != "active_behavior":
            excluded.append({"ophys_experiment_id": eid, "reason": "passive viewing"})
            continue
        with h5py.File(path, "r") as nwb:
            if "EyeTracking" not in nwb["acquisition"]:
                excluded.append({"ophys_experiment_id": eid, "reason": "missing eye tracking"})
                continue
            required = [
                "processing/ophys/event_detection/data",
                "processing/ophys/event_detection/timestamps",
                "processing/running/speed/data",
                "processing/running/speed/timestamps",
                "intervals/trials",
            ]
            absent = [key for key in required if key not in nwb]
            ...
        selected.append((eid, path))
```

iii. From CONVERSION_NOTES Step 1/Step 6: "The supplied ophys NWBs contain synchronized streams, so use `BehaviorOphysExperiment.from_nwb` semantics (or direct NWB reads that reproduce them) rather than reconstructing synchronization." and "Loading AllenSDK session objects would materialize projections, masks, tables, and pandas object arrays that are irrelevant to conversion… Direct `h5py` access; one sequential read per required stream/session." Step 4 justifies the active-only rule ("Passive outcomes are largely forced misses/correct rejects and cannot represent animal choice"; the paper "used all active sessions across experience levels and expressly excluded passive") and the eye-tracking rule ("Exclude these 3 experiments because pupil diameter is a required decoder output"). Step 10 reports independent re-reads of raw NWBs reproducing converted values with `np.allclose`, i.e. the direct-HDF5 route was validated against the data rather than assumed.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are the `mouse_id` column of the experiment metadata table, stringified. The unique set over the *retained* sessions is sorted numerically to form `subjects`, and `subject_idx` maps each session to its mouse. 38 mice are retained.

ii.
```python
"mouse_id": str(int(meta["mouse_id"])),
...
subjects = sorted({result.subject for result in results}, key=int)
subject_lookup = {name: i for i, name in enumerate(subjects)}
...
"subject_idx": np.asarray([subject_lookup[result.subject] for result in results], dtype=np.int32),
```

iii. Step 5 Key Decision / mapping table: "Metadata `mouse_id` → `subjects`, `subject_idx`; String IDs, sorted unique list, per-experiment index… Planned cohort retains all 38 mice." Step 2 independently counted 38 mice in the supplied NWBs and Step 9 confirms the converted file also contains 38.

## 1-c. How are the data split into sessions?

i. One decoder "session" = one NWB file = one *ophys experiment* = one imaging plane. Sessions are never merged. For the 45 multiscope files this means a single physical recording session (one `ophys_session_id` with up to 7 simultaneously imaged planes) is emitted as up to 7 separate decoder sessions that share identical behavioural trials and identical output labels but disjoint neuron sets. This is visible in the verification log, where runs of identical trial counts appear (e.g. `208` seven times, `268` seven times, `307` seven times). `ophys_session_id` is retained in metadata but is not used to group.

ii.
```python
for position, (eid, path) in enumerate(selected):
    result = process_session(eid=eid, path=path, meta=metadata.loc[eid], ...)
    results.append(result)
...
data = {
    "neural": [result.neural for result in results],   # one entry per experiment/plane
    ...
}
# metadata records, but does not group by, the parent session
"ophys_session_id": int(meta["ophys_session_id"]),
"session_definition": "one released ophys experiment/imaging plane",
```

iii. Step 4: "Local subset: 284 experiments/247 recordings/38 mice… Use each imaging plane/experiment as a decoder session, consistent with the paper's imaging-plane sampling unit." Step 2 acknowledges the consequence explicitly: "The difference is seven multiscope sessions represented by multiple simultaneous imaging planes; each NWB is a distinct neural population and is therefore a natural target-format session, although behavioral trials repeat across simultaneous planes." The paper's decoding analysis does report results per imaging plane ("we averaged the performance of all samples from the same imaging plane"), which the AI cites as precedent.

## 1-d. How are the data split into trials?

i. Trials come from the native NWB `intervals/trials` table. A trial spans `start_time → stop_time` (variable duration, 7.0–12.5 s). Eligible trials are exactly those with `go | catch`, which the code asserts is disjoint from `aborted | auto_rewarded`. Each eligible trial is then converted to a fixed-rate grid of complete 1/30 s bins whose centres are `start + (k+0.5)/30`, `k = 0 … floor((stop-start)*30)-1`; the incomplete final bin is dropped. Retained trials have 210–376 timepoints (mean 253.5).

ii.
```python
go = trials["go"][:].astype(bool)
catch = trials["catch"][:].astype(bool)
aborted = trials["aborted"][:].astype(bool)
auto_rewarded = trials["auto_rewarded"][:].astype(bool)
eligible = go | catch
if np.any(eligible & (aborted | auto_rewarded)):
    raise AssertionError(f"{eid}: eligible trial is aborted/auto-rewarded")
starts_all = trials["start_time"][:].astype(np.float64)
stops_all  = trials["stop_time"][:].astype(np.float64)
candidate_indices = np.flatnonzero(eligible)

def trial_centers(start: float, stop: float) -> np.ndarray:
    n_bins = int(np.floor((stop - start) * TARGET_HZ + 1.0e-9))
    if n_bins < 1:
        return np.empty(0, dtype=np.float64)
    return start + (np.arange(n_bins, dtype=np.float64) + 0.5) * BIN_SIZE_S
```

iii. Step 4: "Trust NWB start/stop/change times and retain exactly `go | catch`; no performance/engagement filter was requested." Step 5: "Native variable-duration trials (7.02–12.56 s) become 210–376 timepoints"; "create complete 1/30-s bins whose centers are `start + (k+0.5)/30`" so that "only complete bins are included" and edge ambiguity is removed. This directly follows the Decoder Task instruction to "include both the 'Go' and 'Catch' trials, but exclude the 'Aborted' and 'Auto-rewarded' trials."

## 1-e. How are trials filtered based on quality controls?

i. Four layers of curation:
1. **Session level** — passive-viewing experiments are dropped wholesale (82 files); experiments with no eye-tracking stream are dropped (3 files); experiments missing any required dataset are dropped.
2. **Trial category** — only `go | catch` (51,075 candidates across the 199 retained experiments).
3. **Pupil validity** — a trial is discarded if its `[start_time, stop_time)` window overlaps any *residual* pupil-invalid interval, i.e. a NaN run longer than 30 eye-camera frames (~1 s) or a run touching the recording edge. This removes 2,166 trials (4.2%), leaving 48,909.
4. **Degenerate cases** — a retained trial must contain ≥2 bins; a session must retain ≥2 trials, otherwise `process_session` raises.

There is **no** activity-based, engagement-based, or performance-based trial filter.

ii.
```python
invalid_trials = trials_overlapping_invalid_runs(
    candidate_starts, candidate_stops, eye_t, residual_runs
)
kept_indices = candidate_indices[~invalid_trials]
removed_indices = candidate_indices[invalid_trials]
if len(kept_indices) < 2:
    raise RuntimeError(f"{eid}: only {len(kept_indices)} valid trials after pupil filtering")

per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
if any(len(x) < 2 for x in per_trial_t):
    raise AssertionError(f"{eid}: retained trial has fewer than two bins")

def trials_overlapping_invalid_runs(starts, stops, sample_times, residual_runs):
    bad = np.zeros(len(starts), dtype=bool)
    half_step = 0.5 * float(np.median(np.diff(sample_times)))
    for first, last_exclusive in residual_runs:
        invalid_start = sample_times[first] - half_step
        invalid_stop = sample_times[last_exclusive - 1] + half_step
        bad |= (starts < invalid_stop) & (stops > invalid_start)
    return bad
```

iii. Step 5 Key Decision 4: "Short blink/outlier gaps (≤1 s) are linearly interpolated only between valid processed samples; any go/catch trial intersecting a longer invalid interval is removed. The 1-s rule limits interpolation to brief blink-scale gaps while retaining about 95.8% of otherwise eligible trials." Key Decision 9: "Pupil uses no raw outlier values and no global/zero fill. Sessions without the stream are excluded; trials with long invalid periods are excluded." Step 4 justifies the passive exclusion (paper "expressly excluded passive"; passive outcomes "cannot represent animal choice"). Step 10 Check 6 re-derived the removed trial-ID sets independently from the raw NWBs and reports exact agreement plus exact global accounting (51,075 − 2,166 = 48,909).

## 2-a. What variables in the raw data is the `neural` data derived from?

i. `processing/ophys/event_detection/data` (time × cell detected calcium-event magnitudes) together with `processing/ophys/event_detection/timestamps`. Neither dF/F nor the SDK's smoothed `filtered_events` is used. The cell set is the released `cell_specimen_table`, which the code asserts is entirely `valid_roi == True` and dimensionally consistent with the event matrix (29,168 session-neurons total).

ii.
```python
ophys_t = nwb["processing/ophys/event_detection/timestamps"][:].astype(np.float64)
event_dataset = nwb["processing/ophys/event_detection/data"]
if event_dataset.shape[0] != len(ophys_t):
    raise AssertionError(f"{eid}: event/timestamp mismatch")
nneurons = int(event_dataset.shape[1])
events = read_float32_dataset(event_dataset)
...
cell_ids = nwb["processing/ophys/image_segmentation/cell_specimen_table/cell_specimen_id"][:]
valid_rois = nwb["processing/ophys/image_segmentation/cell_specimen_table/valid_roi"][:].astype(bool)
if len(cell_ids) != nneurons or not np.all(valid_rois):
    raise AssertionError(f"{eid}: event/cell table mismatch or invalid ROI present")
```

iii. Step 4 discrepancy table: "Study used detected calcium events to remove slow GCaMP decay → Use raw released event magnitudes, not DFF and not the visualization-only half-Gaussian filtered events." Step 5 Key Decision 2: "Use released raw detected calcium event magnitudes. DFF is less appropriate because the paper explicitly used inferred events; SDK `filtered_events` adds smoothing intended for visualization only." This tracks `methods.txt` line 208: "For all analysis of neural data we used the detected calcium events… thus removing the slow decay dynamics of the calcium indicator GCaMP6f." Step 1 also notes dF/F "does not need to be recomputed" — it is stored — but the AI still preferred events.

## 2-b. How is the `neural` data processed?

i. Essentially no transformation beyond resampling. The full-session event matrix is read as float32, checked finite, and linearly interpolated (per neuron, vectorised over all neurons at once) from the native ophys timestamps onto the concatenated 30 Hz trial-bin-centre grid. It is then sliced per trial and transposed to (n_neurons, n_timepoints), C-contiguous float32. No smoothing, no z-scoring, no baseline subtraction, no neuron selection, no cross-plane merging (each plane is its own session).

ii.
```python
def interpolate_matrix(source_t, source_values, target_t):
    right = np.searchsorted(source_t, target_t, side="left")
    right = np.clip(right, 1, len(source_t) - 1)
    left = right - 1
    denom = source_t[right] - source_t[left]
    weight = ((target_t - source_t[left]) / denom).astype(np.float32)
    result = source_values[left] * (1.0 - weight[:, None]) + source_values[right] * weight[:, None]
    return np.asarray(result, dtype=np.float32)

neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
```

iii. Step 3: "For event-triggered analyses, paper code isolated events at native ophys timestamps and linearly interpolated to a common 30-Hz relative timebase. Running speed was processed the same way. This is the clearest reference precedent for a common decoder bin size." Step 5 Key Decision 2/3 and the mapping table: "Select valid released cells; linearly interpolate raw event magnitudes at 30-Hz trial-bin centers; transpose to neuron × time; float32… Raw detected events match the paper; every trial matrix has the experiment's complete valid neural population." Step 6: the interpolation is deliberately vectorised over "all retained trial samples" in a single concatenated pass.

## 2-c. How is the `neural` data filtered based on quality controls?

i. No quality filter is applied beyond what the Allen release already did. The code *asserts* that every ROI in the cell-specimen table is `valid_roi == True` (which the AI verified holds for all 42,147 local experiment-cell rows) and that the cell table length equals the event-matrix width. No activity threshold, SNR threshold, or silent-cell removal is applied; trials whose neural matrix is entirely zero (2,429 of them, flagged as warnings by the verifier) are deliberately retained.

ii.
```python
valid_rois = nwb["processing/ophys/image_segmentation/cell_specimen_table/valid_roi"][:].astype(bool)
if len(cell_ids) != nneurons or not np.all(valid_rois):
    raise AssertionError(f"{eid}: event/cell table mismatch or invalid ROI present")
if not np.all(np.isfinite(events)):
    raise ValueError(f"{eid}: event data contain NaN/Inf")
```

iii. Step 1: "Default cell curation is `valid_roi == True`; there is no electrophysiology unit-quality filtering because this is two-photon calcium imaging." Step 4: "Preserve all released valid cells; add no unreferenced activity threshold." Step 10 Check 1 defends keeping the all-zero trials: "FastLZero event traces are sparse, and selecting trials based on neural activity would bias the dataset and violate go/catch trial selection. DFF substitution or visualization smoothing would contradict the paper." Three of the warning cases were independently reconstructed from the raw event arrays and confirmed genuinely silent.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Alignment is to the **native trial start** on the synchronized ophys clock. All streams (events, running, pupil, stimulus) are placed on one common grid of bin centres generated from each trial's `start_time`, so alignment is enforced by construction: the same index range `[lo, hi)` slices the neural matrix and every output row. The code asserts every retained trial's grid lies strictly inside the ophys timestamp range (no extrapolation). Metadata records `temporal_alignment_event = "native trial start on the synchronized ophys clock"`, `off_start = 0.0`, `off_end = None` (variable trial length).

ii.
```python
per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
offsets = np.cumsum([0] + [len(x) for x in per_trial_t])
target_t = np.concatenate(per_trial_t)
...
if target_t[0] < ophys_t[0] or target_t[-1] > ophys_t[-1]:
    raise AssertionError(f"{eid}: retained trials outside ophys bounds")
neural_aligned = interpolate_matrix(ophys_t, events, target_t)
...
for trial_position, raw_index in enumerate(kept_indices):
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
    output_trial[0] = image_codes[lo:hi]   # same slice for every stream
```
```python
"temporal_alignment_event": "native trial start on the synchronized ophys clock",
"off_start": 0.0,
"off_end": None,
```

iii. Step 1/Step 3: "All clocks were hardware synchronized through one 100-kHz digital I/O board"; "Ophys frame timestamps are the required master timebase." Step 5 Key Decision 3: "Source alignment remains the synchronized ophys timestamp axis." Key Decision 11 explains `off_start`/`off_end`. Step 4 records the cross-check that trial `change_time` agrees exactly (atol 1e-9) with the stimulus-table `is_change` onsets across all 202 active experiments, i.e. the trial clock and the stimulus clock are the same clock.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 33.3333 ms (30 Hz), identical for every trial and session. Yes — every session is rebinned. Native rates are ~31 Hz (single-plane, median 32.30 ms) and ~10.7 Hz (multiplane, median up to 93.22 ms); both are linearly *interpolated* onto the common 30 Hz grid (a slight downsample for single-plane, a ~3× upsample for multiplane). Bins are defined by centres `start + (k+0.5)/30`; partial trailing bins are discarded.

ii.
```python
TARGET_HZ = 30.0
BIN_SIZE_S = 1.0 / TARGET_HZ
...
"time_bin_size": 1000.0 / TARGET_HZ,
"time_bin_size_units": "ms",
"target_sampling_rate_hz": TARGET_HZ,
```

iii. Step 4: "Native frame interval differs by rig (32.29–93.22 ms)… Paper linearly interpolated calcium events and running onto a common 30-Hz event-relative series → Generate a 30-Hz grid within each native trial, anchored to synchronized trial start… This retains ophys-based alignment and provides identical bins across rigs." Step 5 Key Decision 3: "Use 30 Hz (`33.333333 ms`) because target bins must match across sessions and the paper linearly interpolated both events and running to 30 Hz." This is also required by the target-format rule "Time bins should be the same size for all trials and sessions", which the AI cites.

## 3-a. What variables in the raw data is `output` *Image identity* derived from?

i. The natural-image **stimulus presentation interval table** (`intervals/<image>_presentations`), using `image_name`, `start_time`, `stop_time`, `omitted` and `active`. It is *not* derived from the trials table's `initial_image_name`/`change_image_name`. The presentation table is located by excluding `trials`, `natural_movie_one_presentations` and `spontaneous_presentations` and requiring exactly one remaining table with an `image_name` column.

ii.
```python
presentations = presentation_group(nwb)
presentation_starts = presentations["start_time"][:].astype(np.float64)
presentation_stops  = presentations["stop_time"][:].astype(np.float64)
active = presentations["active"][:].astype(bool)
omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
image_names = decode_strings(presentations["image_name"][:])
unknown = sorted(set(image_names[active & ~omitted]) - set(IMAGE_TO_CODE))
if unknown:
    raise ValueError(f"{eid}: unknown image identities {unknown}")
```

iii. Step 5 mapping table: "Active image-presentation `image_name`, `start_time`, `stop_time`, `omitted` → `output[...][0, :]`: at each bin center label one of 16 global image identities only while an image is physically on screen." Step 4: "Presentation table has exact start/stop and image identity… Active images median 250.2 ms and gray gap median 500.4 ms… Label image only on `[start_time, stop_time)`; label all gaps and omissions as `gray`." The intent is to use the authoritative record of what was on the monitor rather than a per-trial summary.

## 3-b. What processing is involved in computing `output` *Image identity*?

i. A 17-value global codebook is hard-coded: `gray` (code 0) plus the 16 natural images that span image sets A and B (`im000 … im106`), so codes are comparable across sessions and image sets. For each 30 Hz bin centre the code finds the last presentation whose `start_time` ≤ centre, and labels the bin with that image only if the presentation is `active`, not `omitted`, and the centre is still `< stop_time`; otherwise the bin is `gray`. Result: 66.9% of bins are `gray` and each of the 16 images occupies 1.9–2.2%. The code asserts no unexpected image name appears.

ii.
```python
IMAGE_VALUES = ["gray", "im000", "im031", "im035", "im045", "im054", "im061", "im062",
                "im063", "im065", "im066", "im069", "im073", "im075", "im077", "im085", "im106"]
IMAGE_TO_CODE = {name: i for i, name in enumerate(IMAGE_VALUES)}
...
image_codes = np.zeros(len(target_t), dtype=np.int8)          # default = gray
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1
valid_pidx = pidx >= 0
shown = np.zeros(len(target_t), dtype=bool)
shown[valid_pidx] = (
    active[pidx[valid_pidx]]
    & ~omitted[pidx[valid_pidx]]
    & (target_t[valid_pidx] < presentation_stops[pidx[valid_pidx]])
)
shown_indices = np.flatnonzero(shown)
image_codes[shown_indices] = np.asarray(
    [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]], dtype=np.int8)
```

iii. Step 5 Key Decision 6: "Use 17 global identity values (16 named images plus gray). Omissions mean no image was presented, so they are gray rather than a seventeenth image. Natural-movie/spontaneous blocks are outside active trials." Step 9 cross-checks the resulting distribution against the task structure: "250-ms image + 500-ms gray predicts ~1/3 image, ~2/3 gray" vs achieved "gray 0.669; each image 0.019–0.022". Step 10 Check 4 reconstructed the row independently from the raw presentation table for a specific trial and matched exactly.

## 3-c. How is `output` *Image identity* aligned with the neural data?

i. Identity is evaluated at exactly the same 30 Hz bin centres used for the neural interpolation, over the same concatenated `target_t`, and sliced with the identical `[lo, hi)` indices. Alignment is therefore structural, not re-derived. The half-open `[start_time, stop_time)` convention means a bin centre that coincides with a flash offset is already gray.

ii.
```python
pidx = np.searchsorted(presentation_starts, target_t, side="right") - 1   # same target_t as neural
...
output_trial[0] = image_codes[lo:hi]      # same lo:hi as neural_aligned[lo:hi]
```

iii. Step 5 Key Decision 3 ("Bin centers ensure only complete bins are included. Source alignment remains the synchronized ophys timestamp axis") and Step 10's alignment plot review: "image periods and gray periods alternate without shift"; "Native ophys intervals cluster near 32.3 ms and the target line is 33.33 ms; raw event dots and interpolated traces coincide without shifts."

## 4-a. What variables in the raw data is `output` *Image change* derived from?

i. The trials table's `change_time` gated by the `go` flag. The stimulus table's `is_change` (with `start_time`) is used as an independent cross-check, not as the primary source. Catch trials (`is_sham_change`) never produce a pulse.

ii.
```python
change_times_all = trials["change_time"][:].astype(np.float64)
...
presentation_change = np.nan_to_num(presentations["is_change"][:], nan=0.0).astype(bool)
true_change_times = presentation_starts[active & presentation_change]
go_change_times = change_times_all[kept_indices[go[kept_indices]]]
if len(go_change_times):
    nearest = np.min(np.abs(go_change_times[:, None] - true_change_times[None, :]), axis=1)
    if not np.allclose(nearest, 0.0, rtol=0.0, atol=1.0e-9):
        raise AssertionError(f"{eid}: trial and presentation change times disagree")
```

iii. Step 4: "Stimulus table provides exact `is_change` and `is_sham_change` onset times… Across all 202 active experiments, every go change and catch sham change matches trial `change_time` exactly; no boundary violations → Set image-change=1 only for the single 30-Hz bin containing a true `is_change` onset; catch sham changes remain 0."

## 4-b. What processing is involved in computing `output` *Image change*?

i. For each retained **go** trial, the code finds the first bin centre at or after `change_time` and sets exactly **one** bin to 1; everything else is 0. No window, no smoothing, no extension over the flash or the following gray. Catch trials get an all-zero row. Globally this yields 42,791 positive bins out of 12,454,785 (0.34%), i.e. one positive bin per go trial in a trial averaging 253 bins. The converter asserts that no trial contains more than one pulse.

ii.
```python
change_codes = np.zeros(len(target_t), dtype=np.int8)
for trial_position, raw_index in enumerate(kept_indices):
    if not go[raw_index]:
        continue
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
    if local_index >= hi - lo:
        raise AssertionError(f"{eid}: go change falls after its trial grid")
    change_codes[lo + local_index] = 1
...
if int(outputs[1].sum()) > 1:
    raise AssertionError("A trial contains more than one change pulse")
```

iii. Step 5 Key Decision 7: "Mark one sample—the first sample at or after a true change onset. Do not mark sham catch changes, trial onset, image-to-gray offset, gray-to-repeat onset, or omissions." The AI read the instruction "Have value of 1 right after a change in image identity" literally as an instantaneous event. In Step 12 it acknowledges the consequence — "the harder required label being positive for only one 33-ms bin rather than a selected 400-ms change-vs-previous-repeat example" — but concludes "No converter change is justified" after its label-reconstruction audits passed.

## 4-c. How is `output` *Image change* thresholded into categories?

i. Binary, two values `["no change", "change"]`. Category 1 is assigned only for real identity changes on go trials; catch/sham changes, omissions, repeat flashes and trial onsets are all category 0. No numeric threshold is involved — the categorisation is the go-flag gate plus the single-bin onset rule of 4-b.

ii.
```python
"output_values": [ IMAGE_VALUES, ["no change", "change"], QUINTILE_VALUES, QUINTILE_VALUES, OUTCOME_VALUES ],
...
if not go[raw_index]:
    continue                      # catch/sham trials stay all-zero
change_codes[lo + local_index] = 1
...
limits = (16, 1, 4, 4, 3)
if any(outputs[i].min() < 0 or outputs[i].max() > limits[i] for i in range(5)):
    raise AssertionError("Categorical output outside codebook")
```

iii. Step 5 mapping table: "Catch `is_sham_change` is intentionally 0 because identity does not change." Step 10 all-session check: "change-pulse count was exactly one for hit/miss and zero for false alarm/correct reject" — i.e. the binary label is verified to track the go/catch structure exactly.

## 4-d. How is `output` *Image change* aligned with the neural data?

i. The pulse index is found by `searchsorted` on the *same* per-trial slice of the 30 Hz bin-centre grid used for the neural data, so the positive bin is the first neural bin whose centre is at or after the true change onset (worst-case lag 33 ms). The row is stored with the same `[lo, hi)` slice.

ii.
```python
local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
change_codes[lo + local_index] = 1
...
output_trial[1] = change_codes[lo:hi]
```

iii. Step 12 debugging item 2: "`/app/cache/critical_review2_alignment.png` overlays mean raw detected events, image identity, exact one-bin change labels, running quintile, pupil quintile, and raw synchronized change onset. Visual review shows the hit change pulse begins at the first bin center at/after the raw change onset; catch trials correctly contain no true-change pulse."

## 5-a. What variables in the raw data is `output` *Running speed* derived from?

i. `processing/running/speed/data` and `processing/running/speed/timestamps` — the Allen-processed, wrap/transient-corrected, low-pass-filtered linear speed in cm/s. The unfiltered `speed_unfiltered` stream is deliberately not used, and the speed is not re-filtered or clipped.

ii.
```python
running_t = nwb["processing/running/speed/timestamps"][:].astype(np.float64)
running   = nwb["processing/running/speed/data"][:].astype(np.float64)
running_aligned = interpolate_vector(running_t, running, target_t)
```

iii. Step 1: "Running speed should use the already processed `running_speed`, not `raw_running_speed`." Step 4: "SDK prose calls it 10-Hz low-pass; current code constructs a 3rd-order Butterworth with `Wn=4, fs=60`… Use stored `processing/running/speed`; do not re-filter, avoiding version-label ambiguity." Step 5 mapping note: "Do not clamp legitimate negative filtered values or re-filter."

## 5-b. What processing is involved in computing `output` *Running speed*?

i. Linear interpolation from the running clock onto the 30 Hz trial-bin-centre grid (`np.interp` after dropping any non-finite source samples), with a hard error if the target grid falls outside the finite source range. No smoothing, no clipping of negative speeds, no unit conversion. The interpolated values are then quintile-coded (see 5-c).

ii.
```python
def interpolate_vector(source_t, source_values, target_t):
    finite = np.isfinite(source_t) & np.isfinite(source_values)
    if finite.sum() < 2:
        raise ValueError("Insufficient finite points for interpolation")
    valid_t = source_t[finite]; valid_v = source_values[finite]
    if target_t[0] < valid_t[0] or target_t[-1] > valid_t[-1]:
        raise ValueError("Target timestamps outside finite behavioral stream")
    return np.interp(target_t, valid_t, valid_v).astype(np.float32)
```

iii. Step 3: "For event-triggered analyses, paper code isolated events at native ophys timestamps and linearly interpolated to a common 30-Hz relative timebase. Running speed was processed the same way." Step 5 Key Decision 9: "Running has no NaNs… any unexpected nonfinite neural/running/output value is a hard error."

## 5-c. How is `output` *Running speed* thresholded into categories?

i. Five quintile bins. The 20/40/60/80th percentiles are computed **per session**, over the concatenated samples of that session's retained trials only, and applied with `searchsorted(..., side="right")` to give codes 0–4. Each session's physical-unit edges (cm/s) are saved in `metadata['session_info'][i]['running_quintile_edges_cm_per_s']`. The resulting global distribution is exactly [0.200, 0.200, 0.200, 0.200, 0.200].

ii.
```python
def quantile_codes(values):
    if not np.all(np.isfinite(values)):
        raise ValueError("Nonfinite value passed to quantile binning")
    edges = np.quantile(values.astype(np.float64), [0.2, 0.4, 0.6, 0.8])
    codes = np.searchsorted(edges, values, side="right").astype(np.int8)
    if codes.min() < 0 or codes.max() > 4:
        raise AssertionError("Quintile code outside 0..4")
    return codes, edges

running_codes, running_edges = quantile_codes(running_aligned)   # running_aligned = this session's kept-trial samples
```

iii. Step 5 Key Decision 5: "Compute running and pupil quintile edges separately within each session, using only kept-trial samples. Pupil pixels are rig/animal specific, and session-local quantiles remove calibration offsets; applying the same rule to running yields comparable behavioral-state ranks and balanced targets. Store each session's physical-unit edges in metadata." Step 5 sanity-check plan: "Verify empirical running/pupil output frequencies are near 20% per class."

## 5-d. How is `output` *Running speed* aligned with the neural data?

i. Interpolated at the identical `target_t` bin centres and sliced with the identical `[lo, hi)` indices as the neural matrix, so there is no independent alignment step and no possible offset.

ii.
```python
running_aligned = interpolate_vector(running_t, running, target_t)   # same target_t
...
output_trial[2] = running_codes[lo:hi]                                # same lo:hi
```

iii. Step 3/Step 4: all clocks are hardware-synchronized on one 100-kHz IO board, so the running timestamps and ophys timestamps live on the same clock and direct interpolation is valid. Step 7 plot review: "running and pupil are smooth and synchronized; quintile thresholds are ordered."

## 6-a. What variables in the raw data is `output` *Pupil diameter* derived from?

i. `acquisition/EyeTracking/pupil_tracking/width` and `.../height` (the fitted pupil-ellipse semi-axes, already NaN on `likely_blink` frames in the released NWB) plus `acquisition/EyeTracking/eye_tracking/timestamps`. Diameter is defined as `2 × max(width, height)` in pixels, i.e. the major axis. Pupil *area* and the un-filtered `area_raw`/`*_raw` variants are not used.

ii.
```python
eye_t = nwb["acquisition/EyeTracking/eye_tracking/timestamps"][:].astype(np.float64)
pupil_width  = nwb["acquisition/EyeTracking/pupil_tracking/width"][:].astype(np.float64)
pupil_height = nwb["acquisition/EyeTracking/pupil_tracking/height"][:].astype(np.float64)
if not (len(eye_t) == len(pupil_width) == len(pupil_height)):
    raise AssertionError(f"{eid}: pupil stream length mismatch")
pupil_raw = 2.0 * np.maximum(pupil_width, pupil_height)
```

iii. Step 4: "SDK stores ellipse half-axes and processed pupil area; area = pi * max(width,height)^2. Direct equality check confirms area uses the major half-axis; blink frames are NaN… Whitepaper says major ellipse axis reflects pupil diameter → Define diameter as `2 * max(pupil_width, pupil_height)` in pixels, preserving the requested physical variable and blink filtering." Step 3: "Processed pupil values exclude failed/outlier fits: missing pupil/eye fits or area z-score >3, plus two frames on each side, are marked likely blinks and set to NaN. Raw outlier values should not be substituted."

## 6-b. What processing is involved in computing `output` *Pupil diameter*?

i. Three stages. (1) **Gap repair**: contiguous NaN runs are located; a run is linearly interpolated only if it is *internal* (finite samples on both sides) and ≤30 eye-camera frames (~1 s). (2) **Residual gaps**: longer runs and runs touching the recording edge are left as NaN and instead used to *exclude* overlapping trials (see 1-e); the code then asserts the aligned pupil is fully finite. (3) **Resampling**: linear interpolation onto the 30 Hz bin centres, then quintile coding. No extrapolation, no zero/mean fill, no use of raw outlier values.

ii.
```python
def fill_short_internal_gaps(values, timestamps, max_gap_frames):
    filled = np.asarray(values, dtype=np.float64).copy()
    invalid = ~np.isfinite(filled)
    changes = np.diff(np.r_[False, invalid, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1); ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        bounded = start > 0 and end < n and np.isfinite(filled[start-1]) and np.isfinite(filled[end])
        if bounded and end - start <= max_gap_frames:
            filled[start:end] = np.interp(timestamps[start:end],
                                          [timestamps[start-1], timestamps[end]],
                                          [filled[start-1], filled[end]])
            filled_runs.append((int(start), int(end)))
        else:
            residual_runs.append((int(start), int(end)))
    return filled, filled_runs, residual_runs

pupil_filled, short_runs, residual_runs = fill_short_internal_gaps(pupil_raw, eye_t, MAX_PUPIL_GAP_FRAMES)
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
if not np.all(np.isfinite(pupil_aligned)):
    raise ValueError(f"{eid}: nonfinite pupil after valid-trial selection")
```

iii. Step 5 Key Decision 4: "Short blink/outlier gaps (≤1 s) are linearly interpolated only between valid processed samples; any go/catch trial intersecting a longer invalid interval is removed. The 1-s rule limits interpolation to brief blink-scale gaps while retaining about 95.8% of otherwise eligible trials in a pre-conversion scan." Mapping table: "Gaps >1 s are invalid periods; trials overlapping them are excluded rather than fabricating long stretches. Edge gaps are not extrapolated."

## 6-c. How is `output` *Pupil diameter* thresholded into categories?

i. Identical machinery to running speed: five per-session quintiles (20/40/60/80th percentiles of that session's retained-trial samples), codes 0–4, edges stored in metadata as `pupil_diameter_quintile_edges_pixels`. Because all NaNs have been removed by the trial filter, no missing-data code is needed and the achieved distribution is exactly [0.200]×5.

ii.
```python
pupil_codes, pupil_edges = quantile_codes(pupil_aligned)
...
"pupil_diameter_quintile_edges_pixels": pupil_edges.tolist(),
"percentile_scope": "within session across retained trial timepoints",
```

iii. Step 5 Key Decision 5: "Pupil pixels are rig/animal specific, and session-local quantiles remove calibration offsets." Step 9 consistency table records the achieved per-quintile fractions; Step 10 Check 8 reports "running/pupil bins are each 20.0% per quintile".

## 6-d. How is `output` *Pupil diameter* aligned with the neural data?

i. Same construction as running speed: interpolated at the shared `target_t` bin centres, sliced with the shared `[lo, hi)` indices. The blink-gap logic operates on the native eye clock *before* resampling, so it does not perturb alignment; the half-sample padding (`half_step`) used when testing trial overlap is the only timing tolerance introduced.

ii.
```python
pupil_aligned = interpolate_vector(eye_t, pupil_filled, target_t)
...
output_trial[3] = pupil_codes[lo:hi]
...
half_step = 0.5 * float(np.median(np.diff(sample_times)))
invalid_start = sample_times[first] - half_step
invalid_stop  = sample_times[last_exclusive - 1] + half_step
```

iii. Step 3: eye tracking is hardware-synchronized at 30 Hz with the ophys clock, so interpolation to the ophys-anchored grid is valid. Step 7 plot review: "pupil raw/filled traces agree outside invalid samples" and "running and pupil are smooth and synchronized."

## 7-a. What variables in the raw data is `output` *Trial outcome* derived from?

i. The four mutually exclusive boolean columns of the NWB trials table: `hit`, `miss`, `false_alarm`, `correct_reject`. The code asserts that, restricted to eligible (`go | catch`) trials, exactly one of the four is true for every trial.

ii.
```python
OUTCOME_COLUMNS = ("hit", "miss", "false_alarm", "correct_reject")
...
outcome_flags = np.vstack([trials[name][:].astype(bool) for name in OUTCOME_COLUMNS])
if np.any(outcome_flags[:, eligible].sum(axis=0) != 1):
    raise AssertionError(f"{eid}: retained outcomes are not mutually exclusive/exhaustive")
```

iii. Step 1: "`Trial._get_trial_data` defines mutually exclusive go/catch/aborted/auto-rewarded and hit/miss/false-alarm/correct-reject logic." Step 3 Curation: "Outcomes are hit, miss, false alarm, and correct rejection. Trial times and category flags should be accepted from the synchronized released NWB rather than re-derived." Step 4: "every retained row has exactly one of four outcomes."

## 7-b. What processing is involved in computing `output` *Trial outcome*?

i. The single true flag is converted to an integer code (hit=0, miss=1, false_alarm=2, correct_reject=3) via the column order, and that constant is broadcast across every timepoint of the trial so that the static label can share the `(5, T)` output matrix with the four time-varying rows. The retained distribution is hit 15,209 / miss 27,582 / false alarm 886 / correct reject 5,232 trials (time-weighted fractions 0.307 / 0.568 / 0.017 / 0.108).

ii.
```python
outcome = int(np.flatnonzero(outcome_flags[:, raw_index])[0])
output_trial = np.empty((5, hi - lo), dtype=np.int16)
...
output_trial[4].fill(outcome)
...
"output_values": [..., OUTCOME_VALUES]        # ["hit","miss","false_alarm","correct_reject"]
```

iii. Step 5 Key Decision 8: "Repeat outcome along time to support a single output matrix and allow the validator/decoder to consume mixed static and time-varying outputs. Codes are hit=0, miss=1, false alarm=2, correct reject=3." Step 10's all-session check verified "output outcome was constant" within every trial and that the four counts sum to 48,909.

## 8. How are minor mistakes in the data, e.g. missing data, handled?

i. The policy is "repair only what can be repaired locally; otherwise exclude; never impute":
- **Missing eye-tracking stream** (3 experiments) → experiment excluded.
- **Missing required NWB datasets / missing metadata row** → experiment excluded, with the reason recorded in `metadata['excluded_sessions']`.
- **Passive experiments** → excluded (outcomes are not genuine behaviour).
- **Pupil NaNs (blinks/failed fits)** → internal runs ≤30 eye frames linearly interpolated; longer or edge-touching runs are never filled, and every trial overlapping them is dropped (2,166 trials). Counts of both categories are logged per session.
- **Omitted stimulus flashes** → labelled `gray`, and `omitted`/`is_change` NaNs are coerced with `np.nan_to_num`.
- **Partial trailing time bin** → dropped by the `floor` in `trial_centers`, so no padding or truncation artefacts.
- **Anything unexpected** (non-finite events, non-finite running, non-finite aligned pupil, ROI table mismatch, non-exclusive outcomes, change time outside its trial, trials outside the ophys range, >1 change pulse, <2 bins, <2 trials) → raises immediately rather than being silently repaired. There is no `try/except` around `process_session` in `main`, so any such condition aborts the whole conversion.

ii.
```python
if row["behavior_type"] != "active_behavior":
    excluded.append({"ophys_experiment_id": eid, "reason": "passive viewing"}); continue
if "EyeTracking" not in nwb["acquisition"]:
    excluded.append({"ophys_experiment_id": eid, "reason": "missing eye tracking"}); continue
absent = [key for key in required if key not in nwb]
if absent:
    excluded.append({"ophys_experiment_id": eid, "reason": f"missing required datasets: {absent}"}); continue
...
if not np.all(np.isfinite(events)):
    raise ValueError(f"{eid}: event data contain NaN/Inf")
if not np.all(np.isfinite(pupil_aligned)):
    raise ValueError(f"{eid}: nonfinite pupil after valid-trial selection")
if len(kept_indices) < 2:
    raise RuntimeError(f"{eid}: only {len(kept_indices)} valid trials after pupil filtering")
omitted = np.nan_to_num(presentations["omitted"][:], nan=0.0).astype(bool)
...
"short_pupil_gaps_interpolated": int(len(short_runs)),
"long_or_edge_pupil_gaps": int(len(residual_runs)),
"pupil_invalid_fraction_raw": float(np.mean(~np.isfinite(pupil_raw))),
"removed_raw_trial_ids": trials["id"][:][removed_indices].astype(int).tolist(),
```

iii. Step 5 Key Decision 9: "Running has no NaNs. Pupil uses no raw outlier values and no global/zero fill. Sessions without the stream are excluded; trials with long invalid periods are excluded; any unexpected nonfinite neural/running/output value is a hard error." Step 4 adds that there is "no valid imputation for absent recordings". Step 10 Check 6 independently re-derived every short/residual gap and every removed trial ID for all 199 sessions and reports exact agreement.

## 9-a. What are the most time-consuming steps of the code?

i. The AI's stated and measured bottleneck is HDF5 I/O: reading the full-session event-detection matrix (48,284–149,508 frames × 4–666 cells, stored float64 on disk) for each experiment, plus decompression of the multi-GiB NWBs. Secondary costs are the vectorised `interpolate_matrix` over `n_timepoints × n_neurons` and the final pickle write. Measured: 88.09 s total for 199 sessions (81.40 s processing, 6.69 s serialisation) producing a 7.35 GiB file; sample sessions took 0.42–0.48 s each. Sessions are processed strictly sequentially to bound memory.

ii.
```python
def read_float32_dataset(dataset: h5py.Dataset) -> np.ndarray:
    destination = np.empty(dataset.shape, dtype=np.float32)
    dataset.read_direct(destination)          # whole-session event matrix
    return destination
...
start_clock = time.perf_counter()
...
elapsed = time.perf_counter() - start_clock
print(f"[{eid}] {nneurons} neurons, {len(candidate_indices)} candidate -> "
      f"{len(neural_trials)} trials, {sum(x.shape[1] for x in neural_trials):,} bins in {elapsed:.2f}s")
...
print(f"Timing: processing+plots {write_start - all_start:.2f}s; pickle write {write_elapsed:.2f}s; ...")
```

iii. Step 6: "Loading AllenSDK session objects would materialize projections, masks, tables, and pandas object arrays that are irrelevant to conversion. Per-neuron interpolation loops and repeated trial-level HDF5 reads would also multiply I/O. A full converted neural payload is estimated at ~7.5 GiB, so redundant copies matter." Step 7 Run Time Estimates predicted 2–4 minutes for the full run; Step 9 records the actual 88 s, "substantially below the estimate and optimization threshold."

## 9-b. What loops in the code could have been vectorized to improve efficiency?

i. The heavy work is already vectorised (one `searchsorted`+interpolation over the whole concatenated session grid for neurons, running and pupil; one vectorised presentation lookup for image identity; one vectorised trial-overlap mask). The loops that remain and could in principle be vectorised are all cheap:
- `for trial_position, raw_index in enumerate(kept_indices)` marking the change pulse — a single `searchsorted` of all go `change_time`s against `target_t` plus an offset add would replace it.
- The second `for trial_position, raw_index in enumerate(kept_indices)` that assembles per-trial arrays (it duplicates the same enumeration).
- `per_trial_t = [trial_centers(...) for i in kept_indices]` — could be built with one `np.repeat`/`np.concatenate` construction.
- `for start, end in zip(starts, ends)` in `fill_short_internal_gaps` and `for first, last_exclusive in residual_runs` in `trials_overlapping_invalid_runs` — gap runs are few, but both are Python-level loops over runs.
- The list comprehension `[IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]]` — a precomputed per-presentation code array indexed by `pidx` would be a pure gather.
- `decode_strings` decodes every presentation name in Python even though only the codebook lookup is needed.

ii.
```python
for trial_position, raw_index in enumerate(kept_indices):      # loop 1: change pulse
    if not go[raw_index]: continue
    local_index = int(np.searchsorted(target_t[lo:hi], change_times_all[raw_index], side="left"))
    change_codes[lo + local_index] = 1
...
for trial_position, raw_index in enumerate(kept_indices):      # loop 2: assembly (same enumeration)
    lo, hi = offsets[trial_position], offsets[trial_position + 1]
    neural_trial = np.ascontiguousarray(neural_aligned[lo:hi].T, dtype=np.float32)
    ...
per_trial_t = [trial_centers(starts_all[i], stops_all[i]) for i in kept_indices]
image_codes[shown_indices] = np.asarray(
    [IMAGE_TO_CODE[name] for name in image_names[pidx[shown_indices]]], dtype=np.int8)
```

iii. Step 6 "Code speedups added": "Direct `h5py` access; one sequential read per required stream/session; float32 `read_direct` for event matrices; vectorized timestamp search/interpolation across all retained trial samples; one concatenated alignment followed by contiguous trial slicing; compact int16 outputs; and no image masks/projections." The AI did not flag the residual per-trial loops, implicitly because the measured runtime (88 s) was far under the 15-minute budget.

## 9-c. What processing does the code repeat multiple times?

i. Little of substance, but several small repeats exist:
- `trials["id"][:]` is read from HDF5 twice in consecutive dictionary entries.
- `kept_indices` is enumerated twice (change-pulse pass, then assembly pass), and `offsets[...]` recomputed in both.
- The go-change cross-check recomputes change onsets that were already located for the pulse, and does so with an O(n_go × n_change) broadcast difference matrix.
- `np.nan_to_num` is applied separately to `omitted` and `is_change`, and `presentation_starts` is searched once for identity and again (implicitly) for the change cross-check.
- `validate_converted` re-walks every trial of every session, re-checking finiteness and ranges that `process_session` already asserted.
- Each experiment is opened once in `choose_experiments` (to test for required datasets) and a second time in `process_session`.

ii.
```python
"removed_raw_trial_ids": trials["id"][:][removed_indices].astype(int).tolist(),
"retained_raw_trial_ids": trials["id"][:][kept_indices].astype(int).tolist(),
...
nearest = np.min(np.abs(go_change_times[:, None] - true_change_times[None, :]), axis=1)
...
with h5py.File(path, "r") as nwb:        # in choose_experiments
    if "EyeTracking" not in nwb["acquisition"]: ...
...
with h5py.File(path, "r") as nwb:        # again in process_session
```

iii. Step 6: "one sequential read per required stream/session" — the intent was explicitly to avoid repeated I/O, and the repeats above are on tiny (trial-table / presentation-table) datasets or are deliberate verification passes. Step 10 frames the duplicate verification as intentional: the independent checks "did not import conversion code" and exist to convince the reader the conversion is right.

## 9-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several items are computed or read and then not used by the decoder:
- **Whole-session event matrix**: `read_float32_dataset` materialises every ophys frame (up to 149,508 × 666 floats) although only the frames bracketing the retained trial windows are needed; trials cover roughly a fifth of a session, so most of the decompressed array is discarded. `np.all(np.isfinite(events))` then makes a second full pass over it.
- **Change-time verification matrix**: the `(n_go × n_change)` pairwise difference is purely a sanity check and is thrown away.
- **Diagnostic bundle** (full `ophys_t`, `pupil_raw`, `pupil_filled`, event samples) retained per session when `--show-processing` is on.
- **Metadata never consumed downstream**: per-session `cell_specimen_ids` (29,168 integers), retained/removed raw trial ID lists, gap counts, quintile edges, `excluded_sessions`, equipment/cre-line/experience fields.
- **`pupil_height`** is read in full only to take a per-sample `max` with `width`.
- **Empty input arrays**: a distinct `(0, T)` float32 array is allocated for each of the 48,909 trials even though `input_names` is empty.
- **Upsampling multiplane data** from ~10.7 Hz to 30 Hz roughly triples the stored samples for those 45 sessions with linearly interpolated values that add no information, inflating the 7.35 GiB payload.
- **`validate_converted`** re-verifies the entire dataset after assembly.

ii.
```python
events = read_float32_dataset(event_dataset)          # entire session, ~80% unused
if not np.all(np.isfinite(events)):                   # second full pass
    raise ValueError(f"{eid}: event data contain NaN/Inf")
...
nearest = np.min(np.abs(go_change_times[:, None] - true_change_times[None, :]), axis=1)   # discarded
...
"cell_specimen_ids": cell_ids.tolist(),
"retained_raw_trial_ids": trials["id"][:][kept_indices].astype(int).tolist(),
...
empty_input = np.empty((0, hi - lo), dtype=np.float32)   # one per trial
input_trials.append(empty_input)
```

iii. The AI does not list these as waste; Step 6 instead argues the opposite — that the design already avoids the main waste ("Direct `h5py` access… no image masks/projections… Sessions are processed sequentially to avoid simultaneous decompression of multi-GiB NWBs and unbounded memory"), and Step 10 explicitly defends the redundant verification work as necessary evidence of correctness. The `(0, T)` input shape is justified in Step 5: "Empty float32 array with shape `(0, n_timepoints)`… Preserves target nesting/time shape without leaking any requested output into decoder inputs."
