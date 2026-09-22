# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI finds every NWB with a sorted glob over `/app/data/sub-*/*.nwb`, opens each file once with `h5py`, converts one session at a time, and accumulates usable sessions. Full mode processes all paths; sample mode stops after two usable sessions.

ii.
```python
paths = sorted(glob.glob(str(DATA_ROOT / "sub-*" / "*.nwb")))
...
for path in paths:
    result = convert_session(path, make_plot=make_plot)
    if result is None:
        continue
    converted_sessions.append(result)
```

iii. The AI justified native NWB/HDF5 loading as retrieving the same units, spikes, trials/events, tracking, QC, and anatomy as the paper's MATLAB exports while preserving absolute clocks. Sorting makes output reproducible, and one-session-at-a-time processing limits working memory.

## 1-b. How are the data split into subjects?

i. Subject membership is taken from the parent `sub-*` directory, checked against `general/subject/subject_id`, then unique subject strings are sorted and sessions receive indices into that list.

ii.
```python
subject_id = Path(path).parent.name
source_subject = decode_scalar(nwb["general/subject/subject_id"][()])
if source_subject not in {subject_id, subject_id.removeprefix("sub-")}:
    raise ValueError(...)
...
subjects = sorted({x["subject"] for x in converted_sessions})
subject_idx = np.asarray([subject_to_idx[x["subject"]] for x in converted_sessions])
```

iii. The path and NWB identifier are treated as redundant provenance and cross-validated. The notes report 28 subjects, matching the source and papers.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Its filename stem supplies `session_id`; each converted file contributes one element to every session-level output list.

ii.
```python
session_id = Path(path).stem.replace("_behavior+ecephys+ogen", "").replace(
    "_behavior+ecephys", ""
)
...
"neural": [x["neural"] for x in converted_sessions],
```

iii. This follows the dataset's one-file-per-session organization. One of 174 NWBs is skipped because it has no classifier-good units, leaving 173 sessions.

## 1-d. How are the data split into trials?

i. Trial rows come from `intervals/trials`. The AI maps each good unit's ragged `obs_intervals` to unique trial rows by matching both start and stop times, validates one go cue per trial, and uses the resulting source indices consistently across all streams.

ii.
```python
starts = trial_group["start_time"][:]
stops = trial_group["stop_time"][:]
...
matches = np.flatnonzero(
    np.isclose(starts, obs_start, atol=1e-8, rtol=0)
    & np.isclose(stops, obs_stop, atol=1e-8, rtol=0)
)
observed_trial_indices[j] = matches[0]
```

iii. The notes explain that nine files have `is_good_trials` columns only for the contiguous subset represented by `obs_intervals`; exact interval mapping avoids padding or misindexing those trials.

## 1-e. How are trials filtered based on quality controls?

i. The AI retains only observed trials for which every classifier-good unit has `is_good_trials=True`, requires the entire requested window to lie in the overall recording span, and then removes any trial with no spike in any retained neuron/bin. It keeps behavioral categories including early, ignore, stimulated, auto-water, and free-water trials unless these neural rules remove them.

ii.
```python
all_units_valid = np.all(unit_trial_validity[good_units, :], axis=0)
after_unit_qc = observed_trial_indices[all_units_valid]
full_window = ((go_times[after_unit_qc] + TIME_EDGES[0] >= recording_start - 1e-9)
               & (go_times[after_unit_qc] + TIME_EDGES[-1] <= recording_stop + 1e-9))
...
neural_data_present = np.any(rates > 0, axis=(1, 2))
trial_indices = trial_indices[neural_data_present]
```

iii. It argued that mandatory decoder classes require retaining early/ignore/stimulated trials, that NWB validity fields should be honored, and that four-second population-silent windows after spike-stream truncation represent missing recordings rather than physiology. This produced 90,363 trials. It explicitly chose not to exclude free-water categorically.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data comes from ragged `units/spike_times` and `spike_times_index`, with `BehavioralEvents/go_start_times/timestamps` defining trial-aligned edges. `units/classification` chooses neurons.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
spike_index = nwb["units/spike_times_index"][:].astype(np.int64)
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
```

iii. The AI identified raw spike timestamps as the native neural signal and the common absolute NWB clock as sufficient for alignment.

## 2-b. How is the `neural` data processed?

i. For each retained neuron, spikes are counted in adjacent half-open 50-ms bins via `searchsorted`, divided by 0.05 to obtain Hz, and stored as float32 without smoothing, normalization, or baseline subtraction.

ii.
```python
edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
rates[:, out_unit, :] = np.diff(edge_positions, axis=1) / BIN_WIDTH_S
```

iii. This matches the reference histogram's rate conversion, while the requested 50-ms nonoverlapping bins override the papers' other window/stride choices.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units whose NWB `classification` equals `good` are retained; no ad hoc thresholds are applied to individual QC metrics. A session with no such unit is skipped.

ii.
```python
classifications = decode_array(nwb["units/classification"])
good_indices = np.flatnonzero(classifications == "good")
...
if len(good_units) == 0:
    return None
```

iii. The notes identify this as the authors' final classifier verdict and report 69,453 good units in 173 usable sessions; the unlabeled session is not assigned fabricated labels.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Relative bin edges from -2.5 to +1.5 seconds are added to each selected trial's absolute go-cue timestamp, and absolute spikes are counted within those edges.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
absolute_edges = go_times[trial_indices, None] + TIME_EDGES[None, :]
```

iii. The AI justified this by confirming that spikes and behavior share the NWB session clock, so no extra offset correction or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The output has 80 adjacent 50-ms bins spanning -2.5 through +1.5 seconds around go. Raw point-process spikes are binned directly; there is no subsequent temporal rebinning.

ii.
```python
TIME_EDGES = np.linspace(-2.5, 1.5, 81, dtype=np.float64)
TIME_CENTERS = (TIME_EDGES[:-1] + TIME_EDGES[1:]) / 2
BIN_WIDTH_S = 0.05
```

iii. The width and window are direct task requirements and yield a uniform 80 time points for every trial.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `sample_start_times`, `go_start_times`, trial start times, and the shared bin-center grid. The last sample/tone onset before each selected go cue is used.

ii.
```python
positions = np.searchsorted(sample_starts, selected_go, side="left") - 1
tones = sample_starts[positions]
```

iii. Early licks can replay sample/delay epochs, so the AI chose the final tone leading into the final delay/go rather than the first sample event.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. Each absolute neural bin center is subtracted from its trial's selected tone onset, producing a continuous signed, unit-slope time ramp in seconds.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. The AI states that this preserves protocol-dependent timing and repeated epochs while directly satisfying the continuous-input specification.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at exactly the same absolute centers as the neural 50-ms bins, so input column `k` corresponds to neural bin `k`.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
time_from_tone = absolute_centers - tone_onsets[:, None]
```

iii. Shared go-relative centers eliminate resampling ambiguity; independent checks in the notes found every row advances by exactly 0.05 seconds.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses absolute `BehavioralEvents/photostim_start_times/timestamps` and `photostim_stop_times/timestamps`; trial-table `photostim_onset` is used as a count consistency check.

ii.
```python
photo_starts = event_root["photostim_start_times/timestamps"][:]
photo_stops = event_root["photostim_stop_times/timestamps"][:]
```

iii. The paired event intervals give actual timing and avoid inferring stimulation from power or a trial flag. The event count is checked against non-`N/A` trial rows.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Cumulative counts of starts and stops at every bin center are compared; a center is 1 when more intervals have started than stopped, otherwise 0.

ii.
```python
started = np.searchsorted(photo_starts, absolute_centers, side="right")
stopped = np.searchsorted(photo_stops, absolute_centers, side="right")
photostim_on = (started > stopped).astype(np.float32)
```

iii. The AI validated paired, positive-duration intervals and used real event membership to create the required binary time series.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Absolute photostimulation intervals are sampled at the same go-aligned absolute bin centers used by the neural data.

ii.
```python
absolute_centers = go_times[trial_indices, None] + TIME_CENTERS[None, :]
started = np.searchsorted(photo_starts, absolute_centers, side="right")
```

iii. All streams share the NWB clock, and the notes report raw interval membership checks against converted values.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from trial-table `trial_instruction` and `outcome`, rather than directly from lick-event timestamps.

ii.
```python
instruction = decode_array(trials["trial_instruction"])[trial_indices]
outcome_text = decode_array(trials["outcome"])[trial_indices]
```

iii. The AI reasoned that hit means the instructed side, miss means the opposite side, and ignore means no response; it treated trial outcome as authoritative over rare lick-event conflicts.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded left=0, right=1, no lick=2 and broadcast across all 80 bins.

ii.
```python
actual_choice = np.full(len(trial_indices), 2, dtype=np.int64)
actual_choice[hit & (instruction == "left")] = 0
actual_choice[hit & (instruction == "right")] = 1
actual_choice[miss & (instruction == "left")] = 1
actual_choice[miss & (instruction == "right")] = 0
outputs[:, 0, :] = actual_choice[:, None]
```

iii. This implements actual lick side and preserves no-response as its required third class. Broadcasting allows trial-level and time-varying outputs to share one dense array.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` strings.

ii.
```python
outcome_text = decode_array(trials["outcome"])[trial_indices]
```

iii. The source already contains exactly the requested ignore, miss, and hit categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings are mapped in output-value order to ignore=0, miss=1, hit=2, then broadcast over 80 bins.

ii.
```python
outcome_map = {name: idx for idx, name in enumerate(OUTCOME_VALUES)}
outcomes = np.asarray([outcome_map[x] for x in outcome_text], dtype=np.int64)
outputs[:, 1, :] = outcomes[:, None]
```

iii. The fixed mapping follows the requested category order; broadcasting is a representation choice for a per-trial label.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` field.

ii.
```python
early_text = decode_array(trials["early_lick"])[trial_indices]
```

iii. The source explicitly flags early licking, so the AI did not infer it from lick timestamps.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` becomes 0 and `early` becomes 1, then the value is broadcast over all bins.

ii.
```python
early = (early_text == "early").astype(np.int64)
outputs[:, 2, :] = early[:, None]
```

iii. This directly implements the requested no/yes categorical output while retaining early-lick trials that the papers often excluded for other analyses.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It uses side-camera tongue tracking timestamps and the three data columns x, y, and likelihood. Y is the output quantity; x and y jointly support a velocity-outlier calculation; likelihood determines visibility.

ii.
```python
tracking = nwb["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
timestamps = tracking["timestamps"][:]
marker = tracking["data"][:]
x, y, likelihood = marker[:, 0], marker[:, 1].copy(), marker[:, 2]
```

iii. The AI chose the reference pipeline's side-camera marker stream and used its absolute timestamps for alignment.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames are visible at likelihood at least 0.9. Speeds between adjacent visible frames are computed; visible frames above mean+5 SD are linearly interpolated. Session p40/p60 are then calculated from corrected visible raw-frame y values. For each neural bin center, the nearest camera frame is chosen if within 1.5 median frame intervals; invisible/uncovered centers become class 3.

ii.
```python
visible = np.isfinite(x) & np.isfinite(y) & (likelihood >= 0.9)
cutoff = np.mean(reference_speeds) + 5 * np.std(reference_speeds)
y[target] = np.interp(timestamps[target], timestamps[interpolation_basis],
                      y[interpolation_basis])
p40, p60 = np.percentile(y[visible], [40, 60])
...
nearest = np.where(choose_left, left_clipped, right_clipped)
sampled_visible = covered & visible[nearest]
```

iii. The AI cited the method paper's five-SD marker correction, selected 0.9 because likelihood was strongly bimodal, and argued that raw visible-frame percentiles most literally mean “over the session.” It rejected the paper's occlusion mean-imputation because the task explicitly requires a not-visible class.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session raw visible-frame p40/p60 thresholds define 0 for y<p40, 1 for p40<=y<=p60, 2 for y>p60, and 3 for no sufficiently confident/covered nearest frame.

ii.
```python
classes = np.full(len(flat_centers), 3, dtype=np.int64)
classes[sampled_visible & (sampled_y < p40)] = 0
classes[sampled_visible & (sampled_y >= p40) & (sampled_y <= p60)] = 1
classes[sampled_visible & (sampled_y > p60)] = 2
```

iii. The AI followed the specified boundary semantics and stored thresholds in metadata, expecting an approximately 40/20/40 split conditional on visible raw frames.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses the nearest camera sample to each absolute neural-bin center, accepting it only when within 1.5 median camera-frame intervals. Thus one frame, rather than a mean over the 50-ms interval, represents each bin.

ii.
```python
flat_centers = absolute_centers.ravel()
right = np.searchsorted(timestamps, flat_centers, side="left")
...
covered = np.abs(timestamps[nearest] - flat_centers) <= 1.5 * median_dt
```

iii. It justified this using the reference marker alignment's last/nearest-frame style at camera cadence and the shared absolute clock; raw spot checks found no temporal shift.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI raises on malformed or inconsistent arrays, skips sessions without classifier-good units, maps partial trial tables through `obs_intervals`, applies `is_good_trials` and recording-window validity, removes population-all-zero neural trials as truncated data, and maps uncovered/low-confidence tongue samples to not visible. It does not impute neural data; it does interpolate high-velocity visible tongue outliers.

ii.
```python
if len(good_units) == 0:
    return None
...
neural_data_present = np.any(rates > 0, axis=(1, 2))
...
classes = np.full(len(flat_centers), 3, dtype=np.int64)
```

iii. The AI emphasized not fabricating missing neural observations and distinguishing unavailable tongue measurements with an explicit class. It added defensive cross-field, range, shape, and raw-data checks after finding truncated spike streams despite nominal validity flags.

## 10-a. What are the most time-consuming steps of the code?

i. The AI identifies reading large NWB spike vectors, per-unit spike binning over all trial edges, accumulating the roughly 11.2-GiB dense neural payload, and serializing the final pickle as the principal costs. Full conversion took 194.5 seconds, including 14.52 seconds to write the pickle.

ii.
```python
all_spikes = nwb["units/spike_times"][:]
for out_unit, source_unit in enumerate(good_units):
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
...
pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. Its timing and scaling analysis tied runtime to the amount of neural data read, searched, retained, and written; it remained well below the instruction's 15-minute threshold.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The remaining substantive loop is per unit in spike binning; all trial edges are already searched together. The AI also loops over sessions (necessary for file boundaries), over trial intervals when mapping them, over converted trials in summary reporting, and over a few outputs. Interval matching and summary loops could be further vectorized, while ragged per-unit spike trains resist a simple single-array search.

ii.
```python
for out_unit, source_unit in enumerate(good_units):
    spikes = all_spikes[previous[source_unit] : spike_index[source_unit]]
    edge_positions = np.searchsorted(spikes, absolute_edges, side="left")
...
for session_inputs, session_outputs in zip(data["input"], data["output"]):
    for x, y in zip(session_inputs, session_outputs):
```

iii. The notes say it deliberately eliminated Python loops over trials/bins and random per-unit HDF5 reads by loading one concatenated spike vector and vectorizing event, tracking, output, and validity calculations.

## 10-c. What processing does the code repeat multiple times?

i. Source conversion itself is single-pass per session, but go times and some HDF5 datasets are fetched in several helper functions, converted arrays are traversed again for summary statistics, and optional checking/plotting independently reconstructs some source relationships. The fixed time grid is built only once.

ii.
```python
go_times = nwb["acquisition/BehavioralEvents/go_start_times/timestamps"][:]
# independently in map_observed_trials, final_tone_onsets, construct_inputs,
# bin_spikes, plotting, and metadata construction
```

iii. The AI characterized the conversion as one pass and considered the repeated validations worthwhile; it did not claim costly core quantities such as firing rates were recomputed.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Normal conversion computes detailed validation counts and metadata not consumed by the decoder, including per-session provenance/distributions. Optional `--show-processing` rereads spikes and creates diagnostic figures. It also computes tongue x/speed solely to modify tongue y through a reference-inspired outlier correction. These diagnostics are useful for validation but not decoder features; most are retained in metadata or files rather than silently discarded.

ii.
```python
displacement = np.hypot(np.diff(x), np.diff(y))
...
info = {"n_trials_original": ..., "retained_choice_counts": ...}
...
if make_plot:
    plot_processing(...)
```

iii. The AI justified this work as scientific sanity checking, provenance, and matching the method paper's marker cleaning. It deliberately omitted many unused source variables from decoder inputs/outputs.
