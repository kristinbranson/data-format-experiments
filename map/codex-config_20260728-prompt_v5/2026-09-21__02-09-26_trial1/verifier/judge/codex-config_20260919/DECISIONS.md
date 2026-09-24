# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The converter sorts every `/app/data/sub-*/*.nwb` path, opens each file once with `h5py`, processes all valid sessions in full mode, and assembles their trial lists.

ii.
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
for session_path in chosen_paths:
    result = process_session(session_path, make_plot=plots_remaining > 0)
```

iii. The notes identify 174 NWB files, one per session, across 28 subject folders. Sorting makes traversal deterministic; full mode retained 173 QC-valid sessions.

## 1-b. How are the data split into subjects?

i. Subject identity is the parent folder name (for example `sub-440956`). Unique subjects are retained in first-session order and each session receives an integer index.

ii.
```python
subject_id = session_path.parent.name
subjects_order = list(OrderedDict((sess["subject_id"], None)
                                  for sess in processed_sessions).keys())
```

iii. The agent chose folder identifiers for stability and documented that the archive has 28 `sub-<mouse_id>` folders.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session; its filename stem is the session ID. Output session order follows sorted paths, excluding sessions that fail QC.

ii.
```python
session_id = session_path.stem
"session_ids": [sess["session_id"] for sess in processed_sessions]
```

iii. The notes state that the archive layout is one NWB file per session and that the one session without good classified units is excluded.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`; the code requires exactly one `go_start` per row and uses retained row indices to slice all trial-level streams.

ii.
```python
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
if go_start.shape[0] != n_trials_raw:
    raise ValueError(f"{session_id}: expected one go cue per trial.")
keep_idx = np.flatnonzero(keep_trials)
```

iii. The NWB trial table is the explicit trial definition; the event-count check prevents silent misalignment.

## 1-e. How are trials filtered based on quality controls?

i. A trial is kept only if its entire `[-2.5,+1.5]` window lies in the intersection of observation support over all good units, it has valid go/response/sample events, and its binned neural tensor is not entirely zero. Sessions with fewer than two surviving trials are dropped. Early-lick, ignore, miss, and photostim trials are deliberately retained.

ii.
```python
common_good_trials = ((go_start + OFF_START_S) >= common_obs_start) & \
                     ((go_start + OFF_END_S) <= common_obs_end)
keep_trials = common_good_trials & event_valid
neural_supported = np.any(neural != 0, axis=(1, 2))
```

iii. The agent found behavioral tables extending beyond common neural support and a residual all-zero trial. It avoided `is_good_trials` because dimensions/semantics were inconsistent, and retained behavioral categories required by the decoder.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, restricted by `units/classification`, and aligned using behavioral `go_start_times`.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
neural = bin_spikes_for_session(nwb["units/spike_times"],
    np.asarray(nwb["units/spike_times_index"][:]), good_unit_idx, keep_go_start)
```

iii. The notes identify classifier-good units as the NWB equivalent of the paper's classifier QC lists and spike times as the raw neural representation.

## 2-b. How is the `neural` data processed?

i. For each good unit, absolute go-aligned edges are searched in sorted spike times; adjacent cumulative indices are differenced and divided by 0.05 s to produce firing rates in Hz, stored as `float16`. No smoothing or normalization is applied.

ii.
```python
counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS + 1)
fr[:, out_i, :] = (np.diff(counts, axis=1) / BIN_SIZE_S).astype(np.float16)
```

iii. This follows the reference `sliding_histogram(..., rate=True)` concept while using the task-mandated bins; `float16` was chosen to reduce the very large output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose `classification` equals `good` are retained. Sessions with no good units are dropped; no individual metric thresholds are applied.

ii.
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
if good_unit_idx.size == 0:
    return None
```

iii. The agent regarded this field as the published classifier verdict and obtained 69,453 retained units in 173 sessions, matching the local-release reference count.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative bin edges are added to each absolute go-cue onset, and session-absolute spikes are counted between those edges.

ii.
```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
```

iii. The notes state that NWB spikes and events share a clock, so no offset correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The data have 80 non-overlapping 50-ms bins over `[-2.5,1.5)` seconds. Raw spike timestamps are binned once; no later rebinning is performed.

ii.
```python
BIN_SIZE_S = 0.05
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1) * BIN_SIZE_S
```

iii. These values directly implement the decoder instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, trial start boundaries, go-cue onset, and the common bin centers. The last sample event within a trial and before go is selected.

ii.
```python
sample_start, sample_valid = get_last_events_within(
    sample_start_times, trial_start, go_start)
sample_start_rel = sample_start[keep_idx] - keep_go_start
```

iii. Early licks can replay the sample epoch, so the agent used trial-specific event timing instead of a fixed offset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The selected tone onset is expressed relative to go, then subtracted from each go-relative neural-bin center.

ii.
```python
def make_tone_time_matrix(sample_start_rel):
    return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)
```

iii. This gives seconds elapsed from tone onset at every bin center.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same 80 go-relative bin centers used by the neural edge grid.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
tone_time = make_tone_time_matrix(sample_start_rel)
```

iii. Both quantities use the same go-cue clock and bin grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It comes from absolute `photostim_start_times` and `photostim_stop_times`, bounded to each trial, then expressed relative to go.

ii.
```python
stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)
stim_start_rel = stim_start[keep_idx] - keep_go_start
```

iii. The notes say event timestamps preserve the same content as the reference's aligned stimulation on/off times.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Each bin is one when its center is at or after a finite onset and before its offset; otherwise it is zero.

ii.
```python
active = (BIN_CENTERS_REL[None, :] >= start) & \
         (BIN_CENTERS_REL[None, :] < stop)
mat[valid] = active.astype(np.float32)
```

iii. A binary time series is required, and missing event pairs naturally remain all-zero.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute event times are shifted by each trial's go cue and compared to the common neural bin centers.

ii.
```python
stim_start_rel = stim_start[keep_idx] - keep_go_start
photostim_on = make_photostim_matrix(stim_start_rel, stim_stop_rel)
```

iii. This uses the same global clock and go-cue origin as spike binning.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived directly from `left_lick_times` and `right_lick_times`, using go onset and the response endpoint `min(go+1.5 s, trial_stop)`.

ii.
```python
response_stop = np.minimum(go_start + 1.5, trial_stop)
choice = derive_choice_labels(left_lick_times, right_lick_times,
                              keep_go_start, keep_response_stop)
```

iii. Sampled data showed perfect agreement with instruction-plus-outcome. Direct licks were chosen because later-session `go_stop_times` sometimes mark only a 50-ms cue rather than the response window.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The first lick in the response window determines 0 left or 1 right; no lick is 2, and if both sides occur the earlier event wins. The label is broadcast over all 80 bins.

ii.
```python
choice = np.full(go_start.shape, 2, dtype=np.int64)
choice[left_only] = 0; choice[right_only] = 1
choice[both] = (right_first[both] < left_first[both]).astype(np.int64)
choice_2d = np.broadcast_to(choice[:, None], (choice.shape[0], N_BINS))
```

iii. This implements the paper's 1.5-s answer period and provides the requested no-lick category.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial table's `outcome` strings.

ii.
```python
outcome_raw = read_str_array(trials["outcome"])[keep_idx]
```

iii. The raw categories exactly match the requested output.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. `ignore`, `miss`, and `hit` are mapped to 0, 1, and 2 and broadcast over time.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_raw], dtype=np.int64)
```

iii. Broadcasting lets all outputs share one `(4,80)` trial array.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It is read from the trial table's `early_lick` column.

ii.
```python
early_raw = read_str_array(trials["early_lick"])[keep_idx]
```

iii. The archive explicitly stores the required flag, and these trials are retained because it is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` and `early` map to 0 and 1, then the per-trial label is broadcast across 80 bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int64)
```

iii. This is the requested categorical encoding.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses column 1 (y), column 2 (likelihood), and timestamps from `Camera0_side_TongueTracking`, plus go times and session-level y quantiles.

ii.
```python
tongue_y = tongue_data[:, 1]
tongue_lik = tongue_data[:, 2]
tongue_bins, tongue_y_last, tongue_visible = align_tongue_bins(
    tongue_times, tongue_y, tongue_lik, keep_go_start, q40, q60)
```

iii. The agent identified this as the available side-camera tongue measure and used likelihood to distinguish visibility.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Session frames with likelihood at least 0.9 define raw-frame 40th/60th percentiles. For each trial/bin the code selects the last timestamped frame before the bin end; it is visible only if it lies inside the bin and passes finite/likelihood checks. Its y is categorized, otherwise class 3 is assigned.

ii.
```python
VISIBILITY_THRESHOLD = 0.9
q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(...) - 1
out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
```

iii. The agent argued likelihood is strongly bimodal and that taking the last frame follows the reference marker-alignment idea. It interpreted “over the session” literally as all visible raw frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible values below q40 are 0; q40 through q60 inclusive are 1; above q60 are 2; invalid/invisible bins are 3.

ii.
```python
out[visible & (last_y < q40)] = 0
out[visible & (last_y >= q40) & (last_y <= q60)] = 1
out[visible & (last_y > q60)] = 2
```

iii. The category boundaries and per-session scope implement the explicit decoder specification; class 3 represents no visible frame.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames and spikes share absolute timestamps. Each neural interval is converted to absolute go-aligned start/end times, and the last camera frame inside it is selected.

ii.
```python
abs_starts = go_times[:, None] + BIN_EDGES_REL[:-1][None, :]
abs_ends = go_times[:, None] + BIN_EDGES_REL[1:][None, :]
in_bin = (idx_last >= 0) & (last_times >= abs_starts)
```

iii. The common clock and identical bin boundaries provide direct alignment without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no good units or invalid/empty common observation support are dropped; trials with missing sample/go support, incomplete neural windows, or all-zero neural data are dropped; absent photostimulation becomes zeros; missing/invisible tongue becomes category 3; missing unit anatomy falls back to insertion location and then `unknown`.

ii.
```python
if good_unit_idx.size == 0: return None
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
unit_region_names.append(region or "unknown")
```

iii. The notes distinguish unrecoverable neural/event absence (exclude) from meaningful absence such as invisible tongue or no stimulation (explicit representation/fallback).

## 10-a. What are the most time-consuming steps of the code?

i. Per-unit spike binning is the dominant measured session cost; NWB reads and final pickling also contribute. Tongue alignment and behavioral labels are much smaller. Full conversion took 184.30 s.

ii.
```python
t_spike = now()
neural = bin_spikes_for_session(...)
print(f"  Neural binning: {now() - t_spike:.2f}s")
```

iii. The notes predicted spike binning and tongue alignment as bottlenecks; full logs show neural binning commonly around 0.8–1.8 s versus 0.05–0.09 s for tongue alignment.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. A loop remains over ragged units for spike searches, loops build per-trial input/output/neural lists, and loops resolve per-unit region names. Trial/time dimensions of spike binning and tongue lookup are already vectorized.

ii.
```python
for out_i, unit_i in enumerate(good_unit_idx):
    spikes = np.asarray(spike_times_dataset[unit_starts[unit_i]:spike_index[unit_i]])
input_trials = [np.vstack((tone_time[i], photostim_on[i]))
                for i in range(keep_idx.shape[0])]
```

iii. The agent explicitly vectorized search over all trial edges per unit and tongue `searchsorted`; ragged spike arrays make a single fully vectorized unit search impractical. List construction is required by the target format, though it could be mechanically optimized.

## 10-c. What processing does the code repeat multiple times?

i. It repeatedly converts trial-table string datasets (`outcome`, `early_lick`) and several index datasets, and constructs per-trial arrays from already computed session matrices. Otherwise each NWB file is opened once and global bin grids are reused.

ii.
```python
outcome_raw = read_str_array(trials["outcome"])[keep_idx]
early_raw = read_str_array(trials["early_lick"])[keep_idx]
neural_trials = [neural[i].astype(np.float16, copy=False)
                 for i in range(keep_idx.shape[0])]
```

iii. The agent described the conversion as a single pass and reused precomputed edges/centers. The remaining repetition is lightweight formatting or separate source fields, not repeated expensive analysis.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. In normal conversion it reads `go_stop` but does not use it for choice, records the unused shape of `is_good_trials`, computes `keep_response_stop` beyond label derivation, and prepares extensive stats. Plot-only mode additionally builds figures and intermediates not stored in the dataset. Region metadata is retained but not used by the supplied decoder training.

ii.
```python
go_stop = np.asarray(nwb["acquisition/BehavioralEvents/go_stop_times/timestamps"][:])
is_good_trials_shape = tuple(nwb["units/is_good_trials"].shape)
if make_plot:
    make_processing_plot(...)
```

iii. `go_stop` was inspected because the agent discovered its inconsistent semantics, while `is_good_trials` was retained only for audit metadata. Diagnostic plotting is optional and justified for validation, but discarded by downstream decoder analyses.
