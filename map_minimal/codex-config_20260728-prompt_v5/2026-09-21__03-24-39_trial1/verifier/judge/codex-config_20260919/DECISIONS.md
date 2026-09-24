# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers sorted NWB files with a glob, opens each once with `h5py`, and reads subject, units, trials, behavioral events, electrode metadata, and tongue tracking. It does not retain all data: later filters restrict sessions, units, and trials.

ii.
```python
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
```
```python
with h5py.File(path, "r") as f:
    subject = decode_scalar(f["general/subject/subject_id"])
    trials = f["intervals/trials"]
```

iii. The trajectory says the agent inspected the NWB structure and chose direct NWB fields. It later restricted the data because it estimated an all-unit export at roughly 12 GB and wanted the provided decoder to remain trainable.

## 1-b. How are the data split into subjects?

i. The subject ID is read from each NWB file. A first-seen mapping creates `subjects`, and `subject_idx` records the corresponding subject for every retained session.

ii.
```python
subject = decode_scalar(f["general/subject/subject_id"])
```
```python
if subject not in subject_to_index:
    subject_to_index[subject] = len(data["subjects"])
    data["subjects"].append(subject)
subject_idx.append(subject_to_index[subject])
```

iii. The agent treated the NWB `subject_id` as the canonical identifier. It inspected subject/session counts across all files before implementation.

## 1-c. How are the data split into sessions?

i. One NWB file is treated as one session. A file becomes an output session only if it has good ALM units, passes behavioral session QC, and retains at least one nonzero-spike trial.

ii.
```python
for path in sorted(glob.glob(DATA_GLOB)):
    session = process_session(path, region_to_index)
    if session is None:
        continue
    data["neural"].append(session["neural"])
```

iii. The agent recognized the one-file-per-session layout, but intentionally constrained sessions to ALM and applied paper-derived performance criteria to reduce size and make the dataset region-homogeneous and trainable.

## 1-d. How are the data split into trials?

i. Trial-indexed columns come from `intervals/trials`; the code assumes their row order and length match `go_start_times`. Retained indices are used consistently to slice neural, input, and output arrays, but no explicit length assertion is made.

ii.
```python
n_trials = len(trials["id"])
go_times = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()]
kept_trial_indices = np.flatnonzero(keep_trials)
```

iii. The trajectory describes the NWB trial fields as straightforward and uses the published trial table rather than reconstructing trials. It inspected trial events but did not document an explicit one-go-cue-per-row validation.

## 1-e. How are trials filtered based on quality controls?

i. First, whole sessions must have control performance above 0.65 and at least 50 correct control trials on each side, computed on non-early, non-photostim instructed trials. Within retained sessions, all trial types are initially retained, then any trial with no spikes from the retained ALM population anywhere in the aligned window is removed. The code neither uses `obs_intervals` nor excludes `free_water`, and it does not enforce the required two-trial minimum.

ii.
```python
if not behavior_metrics["passes_filter"]:
    return None
```
```python
keep_trials = np.any(rates != 0, axis=(0, 2))
if not np.any(keep_trials):
    return None
```

iii. The agent used session criteria it found in the methods paper, while keeping early-lick and outcome classes because they are decoder targets. After validation warned about all-zero trials, it interpreted them as post-recording behavioral tails and added the nonzero-ALM-spike filter.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, with `classification` and electrode-derived region labels selecting units. `go_start_times` defines trial windows.

ii.
```python
classification = decode_array(f["units/classification"])
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
```

iii. The agent found `classification == good` explicitly stored and chose electrode-location metadata for region labels after inspecting both annotation and electrode fields.

## 2-b. How is the `neural` data processed?

i. For each retained unit, spikes are assigned to trial windows and 50 ms bins with `searchsorted`, `floor`, and `np.add.at`; counts are divided by 0.05 to produce Hz and cast to `float16`. There is no smoothing or normalization. Because the assignment selects only the most recent window start, overlapping trial windows would be mishandled.

ii.
```python
counts = assign_events_to_windows(spikes, window_starts, window_ends, BIN_WIDTH, N_BINS)
rates[out_idx] = (counts / BIN_WIDTH).astype(np.float16)
```

iii. The trajectory sought to mirror go-cue alignment and 50 ms binning. The agent also chose compact types to control a multi-gigabyte output.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units must have `classification == "good"` and be in left or right ALM. Sessions with no such units are dropped. Other brain regions and otherwise good units are discarded; per-trial unit QC is not used.

ii.
```python
REGIONS_TO_KEEP = {"left ALM", "right ALM"}
keep_mask = (classification == "good") & np.isin(unit_regions, list(REGIONS_TO_KEEP))
```

iii. The agent identified `classification` as the relevant QC verdict. It added the ALM restriction for size, region homogeneity, behavioral relevance, and decoder practicality, not because the task requested it.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue timestamps are combined with relative bin edges to form each trial window; spike timestamps share that clock. However, the window is actually -2.525 to +1.525 seconds because edges are centered around centers spanning -2.5 to +1.5.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
window_ends = go_times + BIN_EDGES[-1]
```

iii. The agent explicitly chose go-cue alignment after inspecting behavioral events and considered the NWB timestamps directly compatible.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The declared width and stride are both 50 ms, so spikes are rebinned from event times to firing rates. The implementation creates 81 centers from -2.5 through +1.5 seconds inclusive and 82 edges from -2.525 through +1.525, rather than 80 bins over the requested interval.

ii.
```python
BIN_WIDTH = 0.05
BIN_STRIDE = 0.05
BIN_CENTERS = compute_bin_centers(BEGIN_TIME, END_TIME, BIN_STRIDE)
BIN_EDGES = np.concatenate([BIN_CENTERS - BIN_WIDTH / 2.0,
                            [BIN_CENTERS[-1] + BIN_WIDTH / 2.0]])
```

iii. The agent intended 50 ms go-cue bins, as required, but its center-count helper introduced the off-by-one/window-extension error.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Despite its name, it is derived only from the fixed go-cue-relative `BIN_CENTERS`. No tone/sample onset variable is read or used.

ii.
```python
input_trials.append(np.vstack([BIN_CENTERS, stim_series]))
```

iii. The trajectory investigated sample and go events and noted repeated sample tones, but the final implementation omitted tone onset entirely. No justification for that omission appears.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. There is no tone-relative computation; the same vector `[-2.5, ..., 1.5]` is copied into every trial, making it time from go cue rather than tone onset.

ii.
```python
np.vstack([BIN_CENTERS, stim_series]).astype(np.float32, copy=False)
```

iii. The code contradicts both the input name and the agent's earlier inspection of tone timing. The trajectory offers no rationale.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Its 81 entries correspond one-for-one to the neural bin centers, but values use go-relative center coordinates rather than elapsed time from the last tone.

ii.
```python
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
input_trials.append(np.vstack([BIN_CENTERS, stim_series]))
```

iii. The shared `N_BINS` was chosen for structural alignment; semantic tone alignment was not implemented.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial `start_time`, `photostim_onset`, `photostim_duration`, and the corresponding go time.

ii.
```python
onset_abs = trial_start + float(photostim_onset)
offset_abs = onset_abs + float(photostim_duration)
rel_on = onset_abs - go_time
```

iii. The agent compared table timing against photostimulation event streams and confirmed the table onset is relative to trial start.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Nonstimulated (`"N/A"`) trials receive zeros. Otherwise, bin centers at or after onset and before offset receive 1.

ii.
```python
if photostim_onset == "N/A":
    return stim
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. The agent deliberately represented stimulation as a time-varying binary series as requested.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset and offset are converted to go-relative times and compared with the same nominal centers used for neural bins. It therefore shares the code's 81-bin, half-bin-shifted grid.

ii.
```python
rel_on = onset_abs - go_time
stim[(BIN_CENTERS >= rel_on) & (BIN_CENTERS < rel_off)] = 1.0
```

iii. The agent verified absolute/table timing and chose direct go-relative alignment.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `left_lick_times`, `right_lick_times`, and each trial's go cue, rather than from instruction and outcome.

ii.
```python
choice = first_choice_after_go(left_licks, right_licks, go_times[trial_idx])
```

iii. The agent explicitly investigated lick events for miss and ignore trials to resolve the choice definition.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. It finds the first left and right licks in `[go, go+1.5)`, labels the earlier one left (0) or right (1), and uses no lick (2) if neither occurs. The static result is repeated over all bins.

ii.
```python
if np.isfinite(first_left) and first_left < first_right:
    return 0
if np.isfinite(first_right):
    return 1
return 2
```

iii. The trajectory shows empirical inspection of response licks. This is a reasonable direct measurement of actual choice, although the reference inferred it from instruction and outcome.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from `intervals/trials/outcome`.

ii.
```python
outcome = decode_array(trials["outcome"])
```

iii. The agent found the requested categories already present in the trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to ignore=0, miss=1, hit=2, then the per-trial label is repeated across all time bins.

ii.
```python
outcome_label = {"ignore": 0, "miss": 1, "hit": 2}[outcome[trial_idx]]
np.full(N_BINS, outcome_label, dtype=np.int64)
```

iii. This coding follows the requested category order and the trainer's common time-varying array shape.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from `intervals/trials/early_lick`.

ii.
```python
early = decode_array(trials["early_lick"])
```

iii. The agent identified the explicit trial-table flag and kept early trials because early lick is a decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `"no early"` maps to 0 and `"early"` to 1, repeated across all bins.

ii.
```python
early_label = {"no early": 0, "early": 1}[early[trial_idx]]
np.full(N_BINS, early_label, dtype=np.int64)
```

iii. The mapping follows the requested no/yes order.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses side-camera tongue tracking timestamps, data column 1 for y, column 2 for likelihood, and go-cue times.

ii.
```python
track = np.asarray(f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data"])
tongue_bins, p40, p60 = build_tongue_bins(track_times, track[:, 1], track[:, 2], go_times)
```

iii. The agent searched the code and data for tongue visibility handling and inspected likelihood distributions before selecting these channels.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood greater than 0.9 are retained. Session thresholds are percentiles of raw visible-frame y values. Each visible frame is assigned to a trial/bin and its class is written there; if multiple frames share a 50 ms bin, later writes overwrite earlier ones rather than averaging. Unwritten bins remain not visible.

ii.
```python
visible = track_prob > VISIBILITY_THRESHOLD
p40, p60 = np.percentile(track_y[visible], [40, 60])
flat[trial_idx * N_BINS + bin_idx] = classes
```

iii. The agent found likelihood values near-binary and selected 0.9 as a visibility threshold. It did not justify raw-frame percentiles or last-frame overwrite versus the reference's bin means.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Visible y below the session raw-frame 40th percentile is 0, strictly above the 60th is 2, and all values in between (including boundaries) are 1; missing/unassigned bins are 3.

ii.
```python
classes = np.full(len(ys), 1, dtype=np.int64)
classes[ys < p40] = 0
classes[ys > p60] = 2
```

iii. The 40/60 split and per-session scope come from the task. The agent chose percentiles over all visible frames, recorded that choice in metadata, and used an explicit not-visible class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera frames are placed into windows using go-cue-relative edges and floored into the same nominal 50 ms indices. As with spikes, assigning each timestamp to only the most recent window can fail if trial windows overlap; it also uses the extended -2.525/+1.525 window.

ii.
```python
window_starts = go_times + BIN_EDGES[0]
trial_idx = np.searchsorted(window_starts, track_times, side="right") - 1
bin_idx = np.floor((times - window_starts[trial_idx]) / BIN_WIDTH).astype(np.int64)
```

iii. The agent intended common go-cue windows for camera and neural data and relied on their shared NWB clock.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Byte/object strings are decoded defensively. Sessions with no retained units, failed session QC, or no nonzero trials return `None`. Nonstim trials become all-zero stimulation; no visible tongue becomes class 3 and NaN thresholds. All-zero ALM trials are dropped. There is no explicit go/trial length check, `obs_intervals` handling, free-water exclusion, or two-trial safeguard.

ii.
```python
if keep_units.size == 0:
    return None
if not np.any(keep_trials):
    return None
```
```python
tongue_bins = np.full((len(go_times), N_BINS), 3, dtype=np.int64)
```

iii. The all-zero trial policy was added after validator warnings and justified as removing behavioral tails without electrophysiology. Other handling is mostly defensive or categorical rather than imputation.

## 10-a. What are the most time-consuming steps of the code?

i. Loading large NWB arrays, per-unit spike binning across every trial, accumulating the multi-GB in-memory result, pickle writing/loading, and decoder verification/training dominate. Tongue processing also scans all frames.

ii.
```python
for out_idx, unit_idx in enumerate(keep_units):
    counts = assign_events_to_windows(...)
```

iii. The trajectory measured saturated CPU and memory growth, described conversion as compute-bound, and reported a roughly 2.3 GB ALM-only pickle; training reached about 5.9 GB RSS.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike loop remains; ragged spike trains make full vectorization difficult, though work might be parallelized. The per-retained-trial construction of inputs/outputs and list comprehensions for unit regions and brain indices could be vectorized or batched. Within each unit, trials are already processed together.

ii.
```python
for out_idx, unit_idx in enumerate(keep_units):
    ...
for trial_idx in kept_trial_indices:
    ...
```

iii. The agent monitored performance but kept the straightforward loops once runtime and memory appeared bounded. No explicit vectorization analysis was provided.

## 10-c. What processing does the code repeat multiple times?

i. Every retained trial separately constructs stimulation and repeated static output rows. `BIN_CENTERS` is repeatedly stacked, and static choice/outcome/early labels are expanded to 81 values. Each session separately parses electrode location JSON and creates region labels; each unit separately searches spikes against all windows.

ii.
```python
for trial_idx in kept_trial_indices:
    input_trials.append(np.vstack([BIN_CENTERS, stim_series]))
    output_trials.append(np.vstack([np.full(N_BINS, choice), ...]))
```

iii. The agent favored the uniform `(variables, time)` representation expected by the validator and did not discuss eliminating repeated construction.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It computes tongue bins and thresholds for all original trials before dropping zero-spike trials. It bins all retained-unit spikes before determining which trials survive. Session QC computes and stores several metrics used only for filtering/metadata. Static outputs and the fixed time vector are redundantly materialized at every timepoint/trial, increasing storage although the trainer may consume that shape.

ii.
```python
tongue_bins, p40, p60 = build_tongue_bins(..., go_times)
rates = np.empty((keep_units.size, n_trials, N_BINS), dtype=np.float16)
keep_trials = np.any(rates != 0, axis=(0, 2))
```

iii. The trajectory prioritized a trainable, validator-compatible artifact and later filtered dead trials, but did not refactor earlier computations to avoid work subsequently discarded.
