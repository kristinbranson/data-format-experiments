# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all session files by globbing `/app/data/sub-*/*.nwb`, sorting the paths, and opening each NWB file once with `h5py`. Within each file it reads the trial table from `intervals/trials`, event streams from `acquisition/BehavioralEvents`, tracking from `acquisition/BehavioralTimeSeries`, and unit data from `units`.

ii. 
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
...
for session_path in chosen_paths:
    result = process_session(session_path, make_plot=plots_remaining > 0)
```

```python
with h5py.File(session_path, "r") as nwb:
    trials = nwb["intervals/trials"]
    go_start = np.asarray(
        nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:],
        dtype=np.float64,
    )
    tongue_data = np.asarray(
        nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][:],
        dtype=np.float64,
    )
```

iii. In `CONVERSION_NOTES.md`, the AI justified this as using the local NWB archive as the authoritative raw format and mapping NWB trial, event, tracking, and unit fields onto the decoder format.

## 1-b. How are the data split into subjects (mice)?

i. The AI treats the parent folder name of each NWB file, such as `sub-440956`, as the subject identifier. Subjects are assembled in first-seen order across retained sessions, and `subject_idx` indexes each session into that ordered subject list.

ii. 
```python
with h5py.File(session_path, "r") as nwb:
    subject_id = session_path.parent.name
```

```python
subjects_order = list(OrderedDict((sess["subject_id"], None) for sess in processed_sessions).keys())
subject_to_idx = {subject: i for i, subject in enumerate(subjects_order)}
...
"subjects": subjects_order,
"subject_idx": np.array(
    [subject_to_idx[sess["subject_id"]] for sess in processed_sessions], dtype=np.int64
),
```

iii. The notes say this was chosen for stability and because the NWB directory structure is already grouped by subject.

## 1-c. How are the data split into sessions?

i. The AI uses one NWB file as one session. Session identity is the filename stem, and the output session order is the sorted file order after dropping skipped sessions.

ii. 
```python
all_paths = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = session_path.stem
...
return {
    "session_id": session_id,
    ...
}
```

iii. The notes state that the archive is organized as one NWB file per session, so no extra grouping is needed.

## 1-d. How are the data split into trials?

i. The AI reads trial boundaries from `intervals/trials`, assumes one go cue per trial, and keeps trial-level arrays indexed by the trial table row. It also requires a valid sample-start event before the go cue within the same trial.

ii. 
```python
trials = nwb["intervals/trials"]
trial_start = np.asarray(trials["start_time"][:], dtype=np.float64)
trial_stop = np.asarray(trials["stop_time"][:], dtype=np.float64)
...
go_start = np.asarray(
    nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:],
    dtype=np.float64,
)
if go_start.shape[0] != n_trials_raw:
    raise ValueError(f"{session_id}: expected one go cue per trial.")
```

```python
sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)
```

iii. The AI justified using the NWB trial table directly and treating the go cue as the per-trial anchor, while adding sample-event validity checks because replayed sample epochs make tone timing trial-specific.

## 1-e. How are trials filtered based on quality controls?

i. The AI does not use the reference solution’s `free_water` filter. Instead it keeps trials only if the full decoding window `[go-2.5 s, go+1.5 s)` lies inside the shared `units/obs_intervals` support across retained good units, a valid sample event exists before go, and the binned neural tensor is not all zero. Sessions with fewer than 2 surviving trials are dropped.

ii. 
```python
common_obs_start, common_obs_end = get_common_observation_window(
    np.asarray(nwb["units/obs_intervals"][:], dtype=np.float64),
    np.asarray(nwb["units/obs_intervals_index"][:], dtype=np.int64),
    good_unit_idx.astype(np.int64),
)
...
common_good_trials = (
    np.isfinite(go_start)
    & ((go_start + OFF_START_S) >= common_obs_start)
    & ((go_start + OFF_END_S) <= common_obs_end)
)
...
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
keep_trials = common_good_trials & event_valid
```

```python
neural_supported = np.any(neural != 0, axis=(1, 2))
...
if keep_idx.shape[0] < 2:
    print(f"  Skipping {session_id}: fewer than 2 neural-supported trials after filtering.")
    return None
```

iii. In the notes, the AI argued that some NWB sessions contain behavioral trials beyond usable spike support, so it preferred explicit observation-window checks and an all-zero-neural fallback drop over `is_good_trials` or paper-specific behavioral trial masks.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. The neural data is derived from `units/spike_times` for units whose `units/classification` is `"good"`, with per-trial go-cue times from `BehavioralEvents/go_start_times` used to define the bin edges.

ii. 
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
```

```python
neural = bin_spikes_for_session(
    nwb["units/spike_times"],
    np.asarray(nwb["units/spike_times_index"][:], dtype=np.int64),
    good_unit_idx.astype(np.int64),
    keep_go_start,
)
```

iii. The notes say the NWB `classification == "good"` field is the closest equivalent to the paper’s QC-approved unit list and that spike times are the only raw neural representation available.

## 2-b. How is the `neural` data processed?

i. The AI bins spike times into 50 ms go-cue-aligned bins and converts counts to firing rates in Hz. It uses `np.searchsorted` over flattened absolute bin edges for all trials, loops over units, and stores the result as `float16`.

ii. 
```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
...
fr = np.empty((go_times.shape[0], good_unit_idx.shape[0], N_BINS), dtype=np.float16)
for out_i, unit_i in enumerate(good_unit_idx):
    spikes = np.asarray(
        spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]],
        dtype=np.float64,
    )
    counts = np.searchsorted(spikes, abs_edges.ravel(), side="left").reshape(
        go_times.shape[0], N_BINS + 1
    )
    fr[:, out_i, :] = (np.diff(counts, axis=1) / BIN_SIZE_S).astype(np.float16)
```

iii. The AI justified this as preserving the reference go-cue alignment and firing-rate representation while adapting the bin width and trial window to the decoder task.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The AI keeps only units with `classification == "good"` and drops an entire session if no such units remain. It does not apply extra metric thresholds.

ii. 
```python
classification = read_str_array(nwb["units/classification"])
good_unit_idx = np.flatnonzero(classification == "good")
...
if good_unit_idx.size == 0:
    print(f"  Skipping {session_id}: no good units in NWB classification.")
    return None
```

iii. The notes explicitly say this mirrors the paper’s QC classifier verdict as stored in NWB and avoids ad hoc thresholds on other per-unit QC metrics.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. The AI aligns neural data to go-cue onset by adding a fixed relative bin grid to each trial’s absolute go-cue timestamp. No extra clock correction or interpolation is applied.

ii. 
```python
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
```

```python
abs_edges = go_times[:, None] + BIN_EDGES_REL[None, :]
```

iii. The notes say all raw timestamps share the same NWB time base, so alignment only requires expressing each trial window relative to the go cue.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted neural data uses 80 non-overlapping bins of width 50 ms spanning `[-2.5, 1.5)` seconds relative to go cue. There is no additional smoothing or rebinning beyond this binning step.

ii. 
```python
BIN_SIZE_S = 0.05
OFF_START_S = -2.5
OFF_END_S = 1.5
N_BINS = int(round((OFF_END_S - OFF_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = OFF_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
```

iii. The notes say this is the task-driven change from the reference code’s original analysis settings.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `BehavioralEvents/sample_start_times` together with trial `start_time` and per-trial `go_start`. For each kept trial, the AI picks the last sample-start event inside `[trial_start, go_start]`.

ii. 
```python
sample_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/sample_start_times/timestamps"][:],
    dtype=np.float64,
)
sample_start, sample_valid = get_last_events_within(sample_start_times, trial_start, go_start)
```

iii. The notes say this was necessary because early licks can replay the sample epoch, so tone onset must be chosen per trial from the raw event times.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. After converting the chosen sample start to time relative to go cue, the AI computes the time from tone onset at each neural bin center as `bin_center - sample_start_rel`.

ii. 
```python
sample_start_rel = sample_start[keep_idx] - keep_go_start
...
def make_tone_time_matrix(sample_start_rel: np.ndarray) -> np.ndarray:
    return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)
```

iii. The notes describe this as a direct transformation from the go-cue-aligned bin grid into “seconds since tone onset.”

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is computed on exactly the same 50 ms go-cue-centered bin centers used for the neural data, so each input timepoint corresponds to the same bin as the firing-rate matrix.

ii. 
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
...
return (BIN_CENTERS_REL[None, :] - sample_start_rel[:, None]).astype(np.float32)
```

iii. The AI’s notes say all input and neural streams are expressed on the shared go-cue-aligned grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. The AI derives photostimulation from the absolute event streams `BehavioralEvents/photostim_start_times` and `BehavioralEvents/photostim_stop_times`, restricted to each trial interval, rather than from the trial-table `photostim_onset` and `photostim_duration` fields.

ii. 
```python
photostim_start_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_start_times/timestamps"][:],
    dtype=np.float64,
)
photostim_stop_times = np.asarray(
    nwb["acquisition/BehavioralEvents/photostim_stop_times/timestamps"][:],
    dtype=np.float64,
)
stim_start, _ = get_first_events_within(photostim_start_times, trial_start, trial_stop)
stim_stop, _ = get_last_events_within(photostim_stop_times, trial_start, trial_stop)
```

iii. The notes justify this as reconstructing the same stimulation interval from absolute timestamps relative to the go cue, which the AI viewed as equivalent to the reference logic.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The AI converts the per-trial stim start and stop times into go-cue-relative times and then marks a bin as `1` when its center lies within `[stim_start_rel, stim_stop_rel)`, else `0`.

ii. 
```python
stim_start_rel = stim_start[keep_idx] - keep_go_start
stim_stop_rel = stim_stop[keep_idx] - keep_go_start
```

```python
def make_photostim_matrix(
    stim_start_rel: np.ndarray,
    stim_stop_rel: np.ndarray,
) -> np.ndarray:
    mat = np.zeros((stim_start_rel.shape[0], N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_start_rel) & np.isfinite(stim_stop_rel)
    if np.any(valid):
        start = stim_start_rel[valid][:, None]
        stop = stim_stop_rel[valid][:, None]
        active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
        mat[valid] = active.astype(np.float32)
    return mat
```

iii. The AI’s notes say the decoder needs a time-varying binary stimulation signal, so it represented the interval directly on the bin grid.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation is aligned by subtracting each trial’s go cue from the stim timestamps and comparing the resulting relative interval against the same neural bin centers.

ii. 
```python
stim_start_rel = stim_start[keep_idx] - keep_go_start
stim_stop_rel = stim_stop[keep_idx] - keep_go_start
...
active = (BIN_CENTERS_REL[None, :] >= start) & (BIN_CENTERS_REL[None, :] < stop)
```

iii. The notes describe this as matching the go-cue-relative representation used for spikes and other task variables.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The AI derives choice from the raw lick event streams `left_lick_times` and `right_lick_times`, using the first lick in the post-go answer window. It does not use `trial_instruction` plus `outcome` to infer choice.

ii. 
```python
left_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/left_lick_times/timestamps"][:],
    dtype=np.float64,
)
right_lick_times = np.asarray(
    nwb["acquisition/BehavioralEvents/right_lick_times/timestamps"][:],
    dtype=np.float64,
)
choice = derive_choice_labels(
    left_lick_times,
    right_lick_times,
    keep_go_start,
    keep_response_stop,
)
```

```python
def derive_choice_labels(
    left_lick_times: np.ndarray,
    right_lick_times: np.ndarray,
    go_start: np.ndarray,
    go_stop: np.ndarray,
) -> np.ndarray:
    left_first, left_valid = get_first_events_within(left_lick_times, go_start, go_stop)
    right_first, right_valid = get_first_events_within(right_lick_times, go_start, go_stop)
    ...
```

iii. The notes say this was chosen because it uses the most direct behavioral measurement and avoids inconsistent `go_stop_times`, so the AI instead used `[go_start, min(go_start + 1.5 s, trial_stop)]` as the response window.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The AI encodes choice as `0` for left, `1` for right, and `2` for no lick. When both lick sides occur in the answer window, it uses the earlier one. The per-trial label is then broadcast across all 80 bins.

ii. 
```python
choice = np.full(go_start.shape, 2, dtype=np.int64)  # 2 = no lick
left_only = left_valid & ~right_valid
right_only = right_valid & ~left_valid
both = left_valid & right_valid
choice[left_only] = 0
choice[right_only] = 1
choice[both] = (right_first[both] < left_first[both]).astype(np.int64)
```

```python
choice_2d = np.broadcast_to(choice[:, None], (choice.shape[0], N_BINS))
...
np.vstack((choice_2d[i], outcome_2d[i], early_2d[i], tongue_bins[i])).astype(
    np.int8, copy=False
)
```

iii. The notes justify the 1.5 s answer window as matching the paper-defined response period and say time-broadcasting was used so all outputs share one `(n_output, T)` array shape.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` column.

ii. 
```python
outcome_raw = read_str_array(trials["outcome"])[keep_idx]
```

iii. The notes treat the NWB trial table as the authoritative source for the three requested outcome categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The AI maps `ignore`, `miss`, and `hit` to integer classes `0`, `1`, and `2`, then broadcasts the result across all time bins for each trial.

ii. 
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome = np.array([outcome_map[x] for x in outcome_raw], dtype=np.int64)
...
outcome_2d = np.broadcast_to(outcome[:, None], (outcome.shape[0], N_BINS))
```

iii. The notes say the decoder task needs categorical outputs and that per-trial outputs were repeated across time to fit the common output array shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is taken directly from the trial-table `early_lick` column.

ii. 
```python
early_raw = read_str_array(trials["early_lick"])[keep_idx]
```

iii. The notes say these trials were kept because `early_lick` is itself a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The AI maps `no early` to `0` and `early` to `1`, then broadcasts the label across all bins in the trial.

ii. 
```python
early_map = {"no early": 0, "early": 1}
early = np.array([early_map[x] for x in early_raw], dtype=np.int64)
...
early_2d = np.broadcast_to(early[:, None], (early.shape[0], N_BINS))
```

iii. The notes say this preserves early-lick information explicitly instead of using it only as an exclusion mask.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `BehavioralTimeSeries/Camera0_side_TongueTracking`: column 1 of the `data` array is tongue `y`, column 2 is the tracking likelihood, and `timestamps` provides frame times.

ii. 
```python
tongue_data = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"][:],
    dtype=np.float64,
)
tongue_times = np.asarray(
    nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/timestamps"][:],
    dtype=np.float64,
)
tongue_y = tongue_data[:, 1]
tongue_lik = tongue_data[:, 2]
```

iii. The notes describe this as the available side-camera tongue tracking stream and use its `y` coordinate plus likelihood for discretization.

## 8-b. How is `output` *Tongue y-position* processed?

i. The AI keeps only frames with likelihood at least `0.9`, computes session-wide 40th and 60th percentiles from all visible raw `tongue_y` frames, and then classifies each neural bin using the last visible tongue frame before that bin’s end. Bins without a visible frame get class `3`.

ii. 
```python
VISIBILITY_THRESHOLD = 0.9
...
session_visible = (
    np.isfinite(tongue_y)
    & np.isfinite(tongue_lik)
    & (tongue_lik >= VISIBILITY_THRESHOLD)
)
if np.any(session_visible):
    q40, q60 = np.quantile(tongue_y[session_visible], [0.4, 0.6])
```

```python
idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS
) - 1
...
visible = (
    in_bin
    & np.isfinite(last_y)
    & np.isfinite(last_lik)
    & (last_lik >= VISIBILITY_THRESHOLD)
)
```

iii. The notes justify the high threshold by saying tongue likelihood is strongly bimodal and that a strict cutoff avoids false visibility. They also say percentile thresholds should be computed “over the session” and interpreted this literally as over visible session frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI thresholds visible bins using session-level `q40` and `q60` percentiles, assigning `0` if `y < q40`, `1` if `q40 <= y <= q60`, `2` if `y > q60`, and `3` if no visible frame is available for that bin.

ii. 
```python
out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
if np.any(visible):
    out[visible & (last_y < q40)] = 0
    out[visible & (last_y >= q40) & (last_y <= q60)] = 1
    out[visible & (last_y > q60)] = 2
```

iii. The notes say this was meant to match the requested 40th/60th percentile discretization while preserving a dedicated “not visible” class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI aligns tongue output to the same go-cue-centered 50 ms bins as the neural data. For each bin it finds the last camera frame occurring before the bin end and accepts it only if it still lies within that bin.

ii. 
```python
abs_starts = go_times[:, None] + BIN_EDGES_REL[:-1][None, :]
abs_ends = go_times[:, None] + BIN_EDGES_REL[1:][None, :]
...
idx_last = np.searchsorted(frame_times, abs_ends.ravel(), side="left").reshape(
    go_times.shape[0], N_BINS
) - 1
...
in_bin = (idx_last >= 0) & (last_times >= abs_starts)
```

iii. The notes say the camera timestamps share the NWB session clock with spikes and task events, so no extra interpolation or offset correction is needed beyond putting them on the shared go-cue grid.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles missing or problematic data by dropping sessions with no QC-passing units, dropping sessions with invalid shared observation support, filtering trials with missing sample/go support or all-zero neural tensors, representing missing tongue visibility as class `3`, and using fallback insertion-region labels when `anno_name` is missing.

ii. 
```python
if good_unit_idx.size == 0:
    print(f"  Skipping {session_id}: no good units in NWB classification.")
    return None
...
if not np.isfinite(common_obs_start) or not np.isfinite(common_obs_end):
    print(f"  Skipping {session_id}: invalid good-unit observation interval.")
    return None
```

```python
event_valid = sample_valid & np.isfinite(go_start) & np.isfinite(response_stop)
...
neural_supported = np.any(neural != 0, axis=(1, 2))
```

```python
if not region:
    start_idx = unit_electrode_start[unit_i]
    if start_idx < unit_electrodes.shape[0]:
        region = insertion_regions[unit_electrodes[start_idx]]
...
out = np.full((go_times.shape[0], N_BINS), 3, dtype=np.int64)
```

iii. The notes frame this as conservative handling: exclude trials or sessions when no trustworthy neural support exists, and use explicit “not visible” or fallback labels when a measurement is missing but the rest of the trial remains usable.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identified spike binning and tongue alignment as the main runtime costs within a session, plus reading large NWB arrays from disk. Optional plotting also adds work when enabled.

ii. 
```python
t_spike = now()
neural = bin_spikes_for_session(...)
print(f"  Neural binning: {now() - t_spike:.2f}s")
...
t_tongue = now()
...
tongue_bins, tongue_y_last, tongue_visible = align_tongue_bins(...)
print(f"  Tongue alignment: {now() - t_tongue:.2f}s")
```

iii. In the notes, the AI explicitly called spike binning and tongue alignment the expected bottlenecks and recorded per-session timing for those blocks.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The AI already vectorized most trial-wise operations. The main remaining explicit loops are the per-unit spike binning loop and the per-good-unit loop that assigns region names. Trial stacking into Python lists is also still loop-based.

ii. 
```python
for out_i, unit_i in enumerate(good_unit_idx):
    spikes = np.asarray(
        spike_times_dataset[unit_starts[unit_i] : spike_index[unit_i]],
        dtype=np.float64,
    )
    ...
```

```python
for unit_i in good_unit_idx:
    region = normalize_region_name(anno_name[unit_i])
    if not region:
        ...
    unit_region_names.append(region or "unknown")
```

iii. The notes say spike binning and tongue alignment were intentionally vectorized where possible, while the remaining ragged per-unit operations were left as loops.

## 10-c. What processing does the code repeat multiple times?

i. The AI does not intentionally do a second pass over the full dataset. Each session is processed once, but some computations are repeated at the trial level when the script repacks per-trial arrays into Python lists for the target format and, if plotting is enabled, builds additional plot-specific summaries.

ii. 
```python
input_trials = [
    np.vstack((tone_time[i], photostim_on[i])).astype(np.float32, copy=False)
    for i in range(keep_idx.shape[0])
]
output_trials = [
    np.vstack((choice_2d[i], outcome_2d[i], early_2d[i], tongue_bins[i])).astype(
        np.int8, copy=False
    )
    for i in range(keep_idx.shape[0])
]
neural_trials = [neural[i].astype(np.float16, copy=False) for i in range(keep_idx.shape[0])]
```

iii. The notes claim the conversion is effectively a single pass and that per-session statistics and optional plots were added for validation rather than as a second processing pipeline.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI includes optional diagnostic plotting and extra bookkeeping that are not needed for the final converted dataset used by downstream decoder training. It records detailed `session_stats`, imports `matplotlib`, constructs plotting inputs such as `tongue_y_last` and `tongue_visible`, and can save per-session PNGs when `--show-processing` is requested.

ii. 
```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

```python
if make_plot:
    plot_name = Path(f"/app/processing_{session_id}.png")
    make_processing_plot(
        session_id=session_id,
        session_stats=session_stats,
        neural_trials=neural_trials,
        input_trials=input_trials,
        output_trials=output_trials,
        ...
    )
```

```python
"session_info": [sess["stats"] for sess in processed_sessions],
```

iii. The notes justify these extras as sanity checks and user-facing validation artifacts rather than part of the core conversion.
