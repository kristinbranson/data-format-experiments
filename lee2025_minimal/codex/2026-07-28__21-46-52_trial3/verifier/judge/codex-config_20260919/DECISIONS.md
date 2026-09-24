# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI hard-codes the seven animal IDs, opens each animal's MATLAB v7.3/HDF5 `.mat` file with `h5py`, and dereferences the session-level `envs`, `blocked`, `position`, and `trace` datasets. In the default mode it processes every stored session and also makes a sample dataset.

ii.
```python
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
    with h5py.File(path, "r") as h5:
        n_sessions = h5["envs"].shape[1]
        session_indices = list(range(n_sessions))
        for session_index in session_indices:
            neural_trials, input_trials, output_trials, region_idx, summary = convert_session(...)
```

iii. The trajectory says the AI switched to direct HDF5 access because it was faster for metadata and verified that the raw files contain the paper's full 207 sessions and headline counts.

## 1-b. How are the data split into subjects?

i. One named `.mat` file is treated as one mouse; its position in `ANIMALS` supplies `subject_idx`.

ii.
```python
ANIMALS = ["QLAK-CA1-08", "QLAK-CA1-30", "QLAK-CA1-50", "QLAK-CA1-51",
           "QLAK-CA1-56", "QLAK-CA1-74", "QLAK-CA1-75"]
for subject_idx, animal in enumerate(ANIMALS):
    path = data_dir / f"{animal}.mat"
```

iii. The AI found that the source is organized per animal and explicitly checked that these seven files reproduce the paper's 5,413 unique-neuron count.

## 1-c. How are the data split into sessions?

i. Each referenced entry along the file's session dimension becomes one output session, in source order.

ii.
```python
n_sessions = h5["envs"].shape[1]
for session_index in list(range(n_sessions)):
    ...
    dataset["neural"].append(neural_trials)
```

iii. The trajectory records that the AI resolved the expected count by discovering that one animal has 21 sessions rather than 31, giving exactly 207 total, and chose to preserve source order.

## 1-d. How are the data split into trials?

i. Sessions are divided into consecutive 1,800-frame (one-minute at 30 Hz) raw windows. Unlike the reference, the final incomplete window is retained initially. Movement filtering and pooling later make trial time-series lengths variable.

ii.
```python
for trial_idx, start in enumerate(range(0, position.shape[0], TRIAL_FRAMES)):
    end = min(start + TRIAL_FRAMES, position.shape[0])
    pos_trial = position[start:end]
    trace_trial = trace[start:end]
```

iii. The AI observed that recordings are slightly shorter than nominally 40 minutes and explicitly chose to keep the shorter final chunk instead of dropping data.

## 1-e. How are trials filtered based on quality controls?

i. It drops windows with no movement, fewer than three moving frames, fewer than 10 pooled samples, or all-zero pooled neural activity. It raises an error if fewer than two trials survive in a session.

ii.
```python
if not np.any(trial_movement): continue
if pos_trial.shape[0] < TEMPORAL_BIN_FRAMES: continue
if trace_pooled.shape[0] < MIN_POOLED_TIMEPOINTS_PER_TRIAL: continue
if np.all(neural_trial == 0): continue
if len(neural_trials) < 2: raise ValueError(...)
```

iii. After verification warned about a nearly empty, flat trial, the AI added these filters to remove “low-information edge cases” and eliminate verifier warnings.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from the session-referenced raw `trace` arrays, described by the AI as pre-extracted binary calcium rise-event traces.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
```

iii. The trajectory says the AI inspected the repository and confirmed that `trace` contains the paper's binary rise-event vectors.

## 2-b. How is the `neural` data processed?

i. Invalid position rows are removed from trace too; absent cells are removed; remaining non-finite trace values become zero. Within each raw trial, only moving frames are retained, each cell is Gaussian-smoothed with sigma three frames, non-overlapping groups of three frames are averaged, and the result is transposed to neuron-by-time `float32`.

ii.
```python
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
trace_trial = trace_trial[trial_movement]
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
neural_trial = trace_pooled.T.astype(np.float32)
```

iii. The AI chose to bake in the paper repository's position-decoder preprocessing—movement selection, smoothing, and three-frame temporal binning—because sample decoder accuracy suggested this was “the right scale.”

## 2-c. How is the `neural` data filtered based on quality controls?

i. Cells whose first sample is NaN are treated as unregistered and removed. Other NaN/inf values are replaced by zero, and trials whose final neural matrix is entirely zero are discarded.

ii.
```python
registered_mask = ~np.isnan(trace[0])
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
if np.all(neural_trial == 0):
    continue
```

iii. The AI intended to keep all cells registered in a session, preserving the published 69,744 cell-session instances, while excluding cells absent that day and unusable trials.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. There is no event alignment. Raw windows begin at session start in consecutive one-minute increments; neural and behavior use identical window and movement masks.

ii.
```python
"temporal_alignment_event": "No event alignment; contiguous 1-minute windows from session start."
```

iii. The AI found no stimulus-alignment event and treated these as artificial windows of a continuous free-exploration session.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The saved resolution is nominally 100 ms: native 30 Hz samples are average-pooled in non-overlapping three-moving-frame groups. Because stationary frames are removed first, consecutive saved bins need not represent contiguous 100 ms of wall-clock time.

ii.
```python
TEMPORAL_BIN_FRAMES = 3
TEMPORAL_BIN_MS = 1000.0 * TEMPORAL_BIN_FRAMES / FPS
trace_pooled = avg_pool_2d(trace_smoothed, TEMPORAL_BIN_FRAMES)
```

iii. The AI identified a three-frame temporal bin in the paper's own decoder and decided to embed that 10 Hz representation in the converted data rather than retain native 30 Hz samples.

## 3-a. What variables in the raw data is `input` *Environment geometry* derived from?

i. Geometry is derived from the session's `blocked` indices. `envs` is dereferenced for naming and statistics, but not used to construct the decoder input.

ii.
```python
raw_blocked = normalize_blocked(deref_numeric(h5, h5["blocked"][0, session_index]))
env_open = open_mask_from_blocked(raw_blocked)
```

iii. The AI discovered that asymmetric geometries can have session-specific orientations, so it treated `blocked` as authoritative instead of using a geometry-name template.

## 3-b. What processing is involved in computing `input` *Environment geometry*?

i. A length-nine mask is initialized as open (`1`), with blocked indices set to `0`; it is flattened and copied as a static vector for every retained trial.

ii.
```python
open_mask = np.ones(9, dtype=np.float32)
for idx in blocked_indices:
    open_mask[idx] = 0.0
input_trials.append(env_open_flat.copy())
```

iii. The AI wanted an explicit per-session 3×3 open/blocked representation that retains session orientation; its notes define `1=open` and `0=blocked`.

## 4-a. What variables in the raw data is `output` *Mouse position* derived from?

i. Output is derived from each session's two-column `position` array.

ii.
```python
position = deref_array(h5, h5["position"][session_index, 0], dtype=np.float32)
```

iii. The AI identified these as stored, session-aligned DLC trajectories.

## 4-b. What processing is involved in computing `output` *Mouse position*?

i. Rows containing missing coordinates are removed; speed is computed, smoothed, and thresholded so only movement samples remain. Coordinates are scaled by one third of the session-wide maximum coordinate, averaged in groups of three, floored, clipped to the 3×3 grid, and any labels in blocked cells are snapped to the nearest open cell.

ii.
```python
movement_mask = compute_movement_mask(position)
arena_extent = float(np.nanmax(position) + POSITION_BUFFER)
pos_scaled = pos_trial / (arena_extent / POSITION_BINS_PER_AXIS)
pos_pooled = avg_pool_2d(pos_scaled, TEMPORAL_BIN_FRAMES)
bin_xy = np.clip(np.floor(pos_pooled).astype(np.int64), 0, 2)
flat_bins, n_remapped = remap_invalid_bins(bin_xy, env_open_flat)
```

iii. The AI sought to match the repository's within-session decoder: movement above 5 cm/s, session-level normalization, and temporal pooling. It added nearest-open-bin remapping after observing pooled positions landing in blocked bins.

## 4-c. How is `output` *Mouse position* thresholded into categories?

i. Each pooled coordinate is floored into an integer 0–2. The pair is flattened as `first_axis * 3 + second_axis`, producing classes 0–8, followed by nearest-open-bin correction where needed.

ii.
```python
bin_xy = np.floor(pos_pooled).astype(np.int64)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
flat = bin_xy[:, 0] * POSITION_BINS_PER_AXIS + bin_xy[:, 1]
```

iii. The AI chose a coarse 3×3 output consistent with the requested nine classes and used the session common scale it inferred from the paper code.

## 4-d. How is `output` *Mouse position* aligned with the neural data?

i. Position and trace are checked for equal raw lengths, filtered by the same valid-position and movement masks, sliced with the same trial boundaries, and pooled with the same three-frame groups. A defensive truncation equalizes pooled lengths if necessary.

ii.
```python
position = position[valid_position_rows]
trace = trace[valid_position_rows]
pos_trial = pos_trial[trial_movement]
trace_trial = trace_trial[trial_movement]
if pos_pooled.shape[0] != trace_pooled.shape[0]:
    min_len = min(pos_pooled.shape[0], trace_pooled.shape[0])
```

iii. The AI noted that behavioral and neural streams were already session-aligned and deliberately applied identical masks and pooling to preserve samplewise correspondence.

## 5. How are minor mistakes in the data, e.g. missing data, handled?

i. The code validates shapes and alignment, deletes rows with missing position, removes cells marked absent by a NaN first sample, converts remaining trace NaN/inf to zero, rejects empty/short/zero trials and sessions with fewer than two trials, clips out-of-range bins, remaps blocked-bin labels, and raises errors for impossible session states.

ii.
```python
valid_position_rows = ~np.isnan(position).any(axis=1)
position = position[valid_position_rows]
trace = trace[valid_position_rows]
trace = np.nan_to_num(trace[:, registered_mask], nan=0.0, posinf=0.0, neginf=0.0)
bin_xy = np.clip(bin_xy, 0, POSITION_BINS_PER_AXIS - 1)
```

iii. The trajectory shows iterative defensive handling driven by source inspection and verifier warnings, with the goal of retaining registered cells and producing warning-free decoder inputs.

## 6-a. What are the most time-consuming steps of the code?

i. Reading large HDF5 trace/position arrays, smoothing and pooling all 207 sessions, serializing the roughly 3.3 GB full pickle, and then redundantly building the sample dataset are the main costs.

ii.
```python
trace = deref_array(h5, h5["trace"][session_index, 0], dtype=np.float32)
trace_smoothed = gaussian_filter1d(trace_trial, sigma=TRACE_SMOOTH_SIGMA_FRAMES, axis=0)
pickle.dump(full_data, f, protocol=pickle.HIGHEST_PROTOCOL)
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The AI repeatedly described the full conversion as I/O-bound and reported that walking 207 sessions and writing both pickles took several minutes.

## 6-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-invalid-sample nearest-bin loop can be vectorized with broadcasting; trial slicing and pooling could also be batched before variable movement filtering, though session/file iteration is naturally sequential and trials become ragged.

ii.
```python
for idx in np.where(invalid)[0]:
    diffs = valid_bins - bin_xy[idx]
    nearest = valid_bins[np.argmin(np.linalg.norm(diffs, axis=1))]
```

iii. The trajectory does not discuss vectorizing loops; it focused on avoiding slow full-file loads through HDF5 session-by-session access.

## 6-c. What processing does the code repeat multiple times?

i. A normal run builds the full dataset and then rereads/reprocesses the first animal to build the sample dataset. Environment names and position-bin names are also regenerated for summaries, and several full traversals compute aggregate trial lengths and class counts.

ii.
```python
full_data, full_summary = build_dataset(args.data_dir, sample_only=False)
...
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
...
flat_outputs = np.concatenate([trial.reshape(-1) for session in dataset["output"] for trial in session])
```

iii. The AI intentionally produced both full and sample artifacts for validation, but did not justify rereading sample sessions instead of extracting them from the full in-memory result.

## 6-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It dereferences environment names and `SFPs` counts for checks, computes extensive conversion summaries, remapping counters, trial-length distributions and output histograms, and creates a sample pickle during a default full run. These are useful validation artifacts but are not consumed by decoder training; most summary intermediates are discarded unless `--stats-json` is supplied.

ii.
```python
env_name = deref_string(h5, h5["envs"][0, session_index])
source_stats["raw_unique_neurons"] += int(h5["SFPs"].shape[1])
trial_lengths = np.array([trial.shape[1] for session in dataset["neural"] for trial in session])
sample_data, sample_summary = build_dataset(args.data_dir, sample_only=True)
```

iii. The AI used these operations to verify headline paper counts, inspect class balance, and leave reproducible sample artifacts; it viewed them as sanity checks rather than decoder features.
