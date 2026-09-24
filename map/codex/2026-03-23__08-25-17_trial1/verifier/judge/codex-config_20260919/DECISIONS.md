# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent discovers every `/app/data/sub-*/*.nwb` file, sorts the paths, and processes each file once with `h5py`. It reads the trials, units, behavioral events, and tongue camera datasets directly from the NWB/HDF5 hierarchy.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
```

iii. The notes say direct `h5py` avoided PyNWB object-construction overhead and warning spam. The full run found 174 raw files and retained 173 sessions after unit QC.

## 1-b. How are the data split into subjects?

i. The subject ID is taken from the parent directory name, unique IDs are sorted, and each session receives an index into that list.

ii.
```python
subject_id = file_path.parent.name.replace("sub-", "")
subjects = sorted({r.subject_id for r in results})
"subject_idx": np.array([subject_to_idx[r.subject_id] for r in results])
```

iii. The agent regarded the `sub-<id>` folder as equivalent to `subject.subject_id`; its checks found 28 subjects.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. Output session order is sorted file order; sessions with no good units are skipped.

ii.
```python
for idx, file_path in enumerate(target_files, start=1):
    result = process_session(file_path, ...)
    if result is not None:
        results.append(result)
```

iii. The notes identify one NWB file per session and explain that excluding the single zero-good-unit file reconciles 174 raw files with 173 analyzable sessions.

## 1-d. How are the data split into trials?

i. Trial rows and go events are read in corresponding order, but only trials whose entire go-aligned window lies inside the minimum start/maximum stop of all good-unit observation intervals are selected. All-zero neural trials are subsequently removed.

ii.
```python
full_window_mask = ((go_times_all + WINDOW_START_S >= session_obs_start)
                    & (go_times_all + WINDOW_END_S <= session_obs_stop))
selected_trial_idx = np.flatnonzero(full_window_mask)
...
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. The agent found that ephys-backed trials were not always a prefix of behavior trials and chose full-window support as a stricter validity rule. This left 51,346 trials, versus 90,860 in the reference.

## 1-e. How are trials filtered based on quality controls?

i. It keeps trials with full-window session-level observation support, applies `is_good_trials` only when its column count happens to equal the selected count (otherwise checks each unit's observation bounds), zeros invalid unit-trials, and drops trials that are all zero. It does not explicitly remove `free_water` trials or require at least two surviving trials.

ii.
```python
uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
...
fr[invalid_trials] = 0.0
...
neural_session = neural_session[nonzero_trial_mask]
```

iii. The notes justify this as preventing unsupported neural windows and correcting an assumed-prefix bug. They deliberately retain early-lick, ignore, miss, hit, and photostimulation trials because those are targets or inputs.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural values come from ragged `units/spike_times`, indexed by `spike_times_index`, for units whose `classification` is `good`; go-cue timestamps define trial-relative bin edges.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The agent chose the classifier verdict as the paper's intended spike-sorting QC population.

## 2-b. How is the `neural` data processed?

i. For each good unit, `searchsorted` converts spikes to nonoverlapping bin counts; division by 0.05 converts counts to Hz. Invalid unit-trials are set to zero. The stored array is `float16`; there is no smoothing or normalization.

ii.
```python
edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
counts = np.diff(edge_indices, axis=1).astype(np.float32)
fr = counts / bin_width
neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. The notes say vectorized edge lookup improved speed and float16 reduced the output size while preserving 20-Hz increments and verifier compatibility.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Units are retained only for `classification == "good"`; a session with none is dropped. Additionally, the code zeros a unit's entire trial when `is_good_trials` is false or its full window falls outside that unit's observation interval.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    return None
...
fr[invalid_trials] = 0.0
```

iii. The agent says classifier-approved units best match the paper and reports 69,453 such units. It treats per-unit trial support as necessary to avoid representing absent recording as real activity.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Absolute go-cue timestamps are added to the common relative edge grid; absolute spike times are counted between those edges.

ii.
```python
go_times = go_times_all[selected_trial_idx]
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The notes state that all NWB streams share an absolute clock, so no additional offset or interpolation is needed.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. There are 80 nonoverlapping 50-ms bins from -2.5 to +1.5 seconds around the go cue. Raw spikes are newly binned at this resolution.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S)
```

iii. The agent explicitly departed from the papers' other windows/strides because the decoder instructions require 50-ms bins.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It uses `sample_start_times`, trial start/stop times, and go-cue times. It selects the first sample event within each trial, with a fixed -1.85-s fallback when none exists.

ii.
```python
events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
sample_onset_abs[trial] = events[0] if len(events) else go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. The notes call the earliest sample event the most faithful representation of replay-induced timing shifts, though the reference instead uses the last tone before go.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. It subtracts go time from the chosen sample onset, then subtracts that relative onset from every go-relative bin center.

ii.
```python
sample_onset_rel_go = sample_onset_abs - go_times
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The agent describes this as seconds elapsed since tone onset and documents how many trials needed the fixed fallback.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Values are evaluated at the exact same 80 go-relative bin centers used by neural binning.

ii.
```python
bin_centers_rel = edges[:-1] + BIN_WIDTH_S / 2.0
inp[0] = bin_centers_rel - sample_onset_rel_go[trial]
```

iii. The common go-cue clock and common bin centers are the stated alignment mechanism.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. It uses trial-table `photostim_onset`, `photostim_duration`, `start_time`, and the trial go time.

ii.
```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. The agent spot-checked trial-table timing against the behavioral photostim event stream.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For stimulated trials it emits one where a bin center is in `[on, off)` and zero otherwise; `N/A` trials remain all zero.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    inp[1] = ((bin_centers_rel >= stim_rel_on) &
              (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The time-varying binary representation was chosen because the task asks whether stimulation is on at every time point.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Trial-relative onset/offset are expressed relative to the same go cue and compared with the neural bin centers.

ii.
```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
```

iii. The shared go-relative time axis is the justification.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. The agent uses left/right lick event timestamps, trial boundaries, go time, and instruction. It takes the first post-go lick, otherwise the first lick anywhere in the trial, otherwise the instructed side.

ii.
```python
if len(left_post) or len(right_post):
    return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
...
return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. It wanted actual lick direction on responded trials but considered the binary-only fallback unavoidable for ignore trials. The reference instead derives choice from instruction and outcome and provides a third `no lick` category.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded `left=0`, `right=1`, repeated across all 80 bins. There is no no-lick category.

ii.
```python
out[0] = choice_trials[trial]
...
["left", "right"]
```

iii. The agent documented the fallback hierarchy and reported its source counts; it claimed the target specification required binary choice, despite explicitly listing “no lick.”

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. It comes directly from the trial-table `outcome` field.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
```

iii. The raw labels exactly match the requested categories.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Strings map to `ignore=0`, `miss=1`, `hit=2` and the value is repeated across time.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
out[1] = outcome_trials[trial]
```

iii. The mapping follows the requested order and makes the per-trial value compatible with the common output matrix.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. It comes directly from the trial-table `early_lick` field.

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
```

iii. The agent retains early-lick trials specifically because this field is a requested decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. `no early` maps to 0 and `early` to 1, repeated across all bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
out[2] = early_trials[trial]
```

iii. This directly implements the requested no/yes categories.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It reads `Camera0_side_TongueTracking` data column 1 and its timestamps. Column 2 likelihood is also read but only placed in optional plot payload; it does not affect conversion.

ii.
```python
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The notes say likelihood was retained only for QC diagnostics to stay close to a reference marker-alignment routine.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. For every trial/bin it takes the last available camera frame. Empty bins begin as NaN. Percentiles are computed over finite aligned values from retained trials, then values are categorized; no likelihood filtering or averaging occurs.

ii.
```python
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
```

iii. The agent says the source method's marker alignment takes the last frame in a time step, and applies the percentiles after session-level alignment and binning.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles divide values into 0 below 40th, 1 inclusive from 40th through 60th, and 2 above 60th. NaNs remain 0, so the required class 3 (`not visible`) is absent.

ii.
```python
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. The agent documents only three categories and reports roughly 40/20/40 balance; it does not justify omitting the explicitly requested not-visible class.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Absolute camera timestamps are searched against each trial's absolute go-cue-plus-bin edges, so tongue and neural arrays share bins; the last frame in each interval represents the bin.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

iii. The notes state that shared absolute timestamps guarantee go-cue alignment without interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Sessions with no recorded trials or good units are skipped; missing tone events use a fixed -1.85-s fallback; unsupported unit-trials are zeroed; all-zero trials are removed; empty tongue bins become NaN but are ultimately mislabeled class 0; empty good-unit region labels and sessions with no tongue values raise errors.

ii.
```python
if len(good_unit_idx) == 0: return None
sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
fr[invalid_trials] = 0.0
if len(valid_values) == 0: raise ValueError(...)
```

iii. The notes emphasize avoiding fabricated neural activity and record fallback/drop counts. They do not recognize the empty-tongue-to-low-category behavior as an error.

## 10-a. What are the most time-consuming steps of the code?

i. Large HDF5 reads (all spikes and camera data), per-unit spike `searchsorted`, accumulating/pickling the multi-gigabyte result, and optional plotting are the principal costs. The reported full conversion was about 4.6 minutes and produced a 3.7-GB pickle.

ii.
```python
spike_times = units["spike_times"][:].astype(np.float64)
for unit_pos, unit_idx in enumerate(good_unit_idx):
    edge_indices = np.searchsorted(spikes, flat_edges, side="left")
pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
```

iii. The agent says direct HDF5, vectorized trial edges per unit, and float16 storage were introduced specifically to reduce these costs.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The separate trial loops for tone selection, input construction, choice inference, and output assembly could largely be vectorized. The unit loop is harder because spike trains are ragged, though trials and edges within it are already vectorized.

ii.
```python
for trial in range(n_trials):  # tone
for trial in range(n_trials):  # inputs
for trial in range(n_trials):  # choice
for trial in range(n_trials):  # outputs
for unit_pos, unit_idx in enumerate(good_unit_idx):  # spikes
```

iii. The notes focus on eliminating per-spike/per-trial work in neural and tongue alignment, but leave several small per-trial loops for clarity and event-specific branching.

## 10-c. What processing does the code repeat multiple times?

i. It makes four passes over trials for tone, inputs, choice, and output packaging; repeatedly allocates per-trial arrays/lists; and performs trial-window event searches for multiple event streams. In plotting mode it again slices example-trial events and recomputes its stimulation window.

ii.
```python
for trial in range(n_trials): ...
input_trials.append(inp)
...
for trial in range(n_trials):
    output_trials.append(out)
```

iii. The agent does not identify these as material repeated computation; it presents the converter as a single pass over each file.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. It reads `delay_start_times` and tongue likelihood primarily for optional plots, collects extensive per-session stats not stored in the final dataset, and reads/uses lick events solely for a choice rule that the reference can derive from instruction/outcome. It also builds optional plot payloads and PNGs when requested.

ii.
```python
delay_start_times = h5[...]["delay_start_times"]["timestamps"][:]
tongue_likelihood = tongue_values[:, 2]
stats = {"choice_source_counts": ..., "tongue_thresholds": ...}
```

iii. These values support diagnostics and sanity checks documented in the notes, but most do not enter `converted_data.pkl` and are not needed by decoder training.
