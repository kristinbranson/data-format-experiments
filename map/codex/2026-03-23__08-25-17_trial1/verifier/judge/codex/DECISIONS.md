# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The agent loads all sessions by globbing every NWB file under `data/sub-*/*.nwb`, then opens each file with `h5py`. Within each session it reads the `units`, `intervals/trials`, `acquisition/BehavioralEvents`, and `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking` groups needed for neural, task, licking, and tongue-tracking data.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
    sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
    left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. `CONVERSION_NOTES.md` says the agent deliberately used direct NWB/HDF5 reads instead of PyNWB because the dataset is organized as one NWB per session and `h5py` avoided object-construction overhead. The notes also say this was meant to preserve the reference processing while fitting the released NWB format.

## 1-b. How are the data split into subjects?

i. Subjects are inferred from the parent directory name of each NWB file. The code strips the `sub-` prefix and later builds the unique `subjects` list plus `subject_idx` from the per-session `subject_id`.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The notes explicitly map subject identity to the NWB folder name `sub-<subject_id>` and state that subject order follows converted session order through `subject_idx`.

## 1-c. How are the data split into sessions?

i. Each NWB file is treated as one session. The session identifier is the file stem, and the top-level dataset stores one entry per kept session in `neural`, `input`, `output`, and `brain_region_idx`.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

return {
    "neural": [r.neural for r in results],
    "input": [r.inputs for r in results],
    "output": [r.outputs for r in results],
    "brain_region_idx": brain_region_idx,
}
```

iii. The notes describe the raw data organization as “one NWB file per session” and repeatedly compare converted counts at the session level.

## 1-d. How are the data split into trials?

i. Trials are identified from the NWB trial table and go-cue timestamps. The agent first finds behavioral-trial indices whose full `[-2.5, +1.5] s` neural extraction window lies inside the session observation interval, then subsets all trial-level arrays to those indices. After neural construction, it drops any trial whose neural tensor is all zeros.

ii.
```python
def select_trial_indices(go_times_all, good_unit_obs_intervals):
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    return np.flatnonzero(full_window_mask), session_obs_start, session_obs_stop

selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(...)
trial_start = trial_start_all[selected_trial_idx]
trial_stop = trial_stop_all[selected_trial_idx]
go_times = go_times_all[selected_trial_idx]

nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
```

iii. The trajectory and Step 10 notes say the agent originally assumed the first `N` behavioral trials matched `units/is_good_trials`, found that this was wrong in many sessions, and replaced it with the current go-time plus `obs_intervals` trial-selection rule.

## 1-e. How are trials filtered based on quality controls?

i. The agent keeps only trials with a fully supported neural window, optionally applies `units/is_good_trials` when its shape matches the selected-trial set, and drops trials with completely zero neural activity. It explicitly does not apply the reference `regular_trial_mask` that would remove early-lick, no-response, or photostimulation trials, because those variables are decoder targets or inputs.

ii.
```python
is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
uses_direct_is_good_trials = is_good_trials_raw.shape[1] == len(selected_trial_idx)
if uses_direct_is_good_trials:
    is_good_trials = is_good_trials_raw.copy()
else:
    is_good_trials = np.ones((len(good_unit_idx), len(selected_trial_idx)), dtype=bool)

if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs

nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. In Step 5 the notes say the agent intentionally retained early-lick, ignore, and photostimulation trials because those are required decoder outputs/inputs, while enforcing a stricter raw neural-validity filter instead.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural activity is derived from `units/spike_times` for units where `units/classification == "good"`. Trial validity is informed by `units/is_good_trials` and `units/obs_intervals`, and unit metadata comes from `units/anno_name`.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
brain_region_names = decode_str_array(units["anno_name"][good_unit_idx])
unit_obs_intervals = units["obs_intervals"][good_unit_idx].astype(np.float64)
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
```

iii. The notes repeatedly justify this as the NWB equivalent of the reference code’s classifier-approved good-unit population.

## 2-b. How is the `neural` data processed?

i. The code bins spikes into non-overlapping 50 ms bins over `[-2.5, 1.5]` s relative to each trial’s go cue, converts counts to firing rates in Hz, and stores the result as `float16` arrays with shape `(n_trials, n_units, n_bins)` before splitting into per-trial matrices.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)

for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. The notes say the agent intentionally deviated from the paper’s original 40 ms / 3.4 ms preprocessing because the task instructions explicitly required 50 ms bins.

## 2-c. How is the `neural` data filtered based on quality controls?

i. The neural stream keeps only units labeled `good`, skips sessions with zero such units, rejects kept units with empty `anno_name`, and zeroes out unit-by-trial firing rates for trials unsupported by `is_good_trials` or `obs_intervals`.

ii.
```python
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None

brain_region_names = decode_str_array(units["anno_name"][good_unit_idx])
if np.any(brain_region_names == ""):
    raise ValueError(f"{session_id}: found kept good units with empty anno_name")

if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. Step 5 and the README say the agent chose `classification == good` as the primary NWB QC rule, treating it as the closest available analogue to the reference classifier output.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. All neural bins are aligned to each trial’s go-cue onset. Absolute spike-bin edges are constructed by adding the fixed relative bin edges to each trial’s `go_start_times` timestamp.

ii.
```python
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The notes explicitly state that the reference papers and code are go-cue aligned, and the task instructions required go-cue alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The converted data use 50 ms bins and therefore re-bin the raw spike times into 80 non-overlapping bins across the 4 s window. No further temporal smoothing or overlapping-window rebinning is applied.

ii.
```python
BIN_WIDTH_S = 0.05
edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
centers = edges[:-1] + BIN_WIDTH_S / 2.0
```

iii. The notes emphasize that this was a deliberate task-driven deviation from the paper’s original analysis windows.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. It is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps`, plus the trial `start_time`, `stop_time`, and `go_start_times` needed to assign the relevant sample-start event to each trial.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
sample_onset_rel_go = sample_onset_abs - go_times
```

iii. Step 5 says the agent mapped “earliest sample-start event within trial window” to tone onset because that was the most faithful raw representation available in NWB.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial the code takes the first sample-start event within the trial. If none exists, it falls back to a hard-coded expected onset `-1.85 s` relative to go cue. It then computes time since tone onset for every neural bin center as `bin_center_rel_go - sample_onset_rel_go`.

ii.
```python
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85

for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO

inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes justify the earliest-event rule by saying early-lick replay can shift tone timing earlier, and the fallback exists only for missing-event edge cases.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It is evaluated at the same go-aligned 50 ms neural bin centers, so each trial’s time-from-tone vector has the same 80 time points as the neural matrix.

ii.
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The notes and processing plots describe this as a shared go-cue-referenced time axis for neural, inputs, and outputs.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from the trial-table fields `photostim_onset` and `photostim_duration`, together with trial start time and go-cue time to convert those values into the go-aligned frame.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])

stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. The notes say the agent spot-checked these trial-table fields against photostim event timestamps and found them consistent, so it used the simpler per-trial table representation.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For non-`N/A` trials, the code converts onset and duration into go-relative start and stop times and writes a binary `1/0` vector over the 50 ms bin centers indicating whether photostimulation is on in each bin.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. Step 5 says the agent chose a time-varying on/off series because the task requested “whether photostimulation is on at every time point.”

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. It is written directly on the same go-aligned 50 ms bin centers used for neural firing rates.

ii.
```python
inp = np.zeros((2, n_bins), dtype=np.float32)
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. The notes call this a shared go-cue alignment across all streams.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived primarily from `left_lick_times` and `right_lick_times`, with `trial_instruction` used as a fallback when no lick is found.

ii.
```python
left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)
trial_instruction = trial_instruction_all[selected_trial_idx]
```

iii. Step 5 says the agent preferred actual lick timing as the closest raw choice variable, but needed a fallback for ignore trials because the decoder spec still required a binary per-trial choice label.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. The code looks for the first post-go lick; if absent, it falls back to the first lick anywhere in the trial; if there is still no lick, it uses the instructed trial direction. It encodes left as `0` and right as `1`, then repeats that label across all time bins.

ii.
```python
if len(left_post) or len(right_post):
    return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")

if len(left_any) or len(right_any):
    return (0, "any_lick") if first_left < first_right else (1, "any_lick")

return (0 if instruction == "left" else 1, "instruction_fallback")

out[0] = choice_trials[trial]
```

iii. The notes describe this explicit fallback hierarchy as an ambiguity forced by the task format rather than by the reference analysis.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is taken directly from the trial-table `outcome` field.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
outcome = outcome_all[selected_trial_idx]
```

iii. The notes say this uses the exact raw trial labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Outcome strings are mapped as `ignore -> 0`, `miss -> 1`, `hit -> 2`, then repeated across all 80 time bins for each trial.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
out[1] = outcome_trials[trial]
```

iii. The notes describe this as a direct task-spec mapping from raw labels to categorical decoder targets.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question text does not match the implemented converter: there is no distance-to-reward-zone output. For the implemented `outcome` output, the code aligns it by broadcasting the per-trial categorical outcome label across the same 80 go-aligned time bins as the neural data.

ii.
```python
out = np.empty((4, n_bins), dtype=np.int8)
out[1] = outcome_trials[trial]
```

iii. The notes consistently describe `outcome` as a per-trial label repeated over the shared go-cue-aligned decoder time axis.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from the trial-table `early_lick` field.

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
early_lick = early_lick_all[selected_trial_idx]
```

iii. The notes explicitly map the NWB trial label to the decoder target.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The code maps `no early -> 0` and `early -> 1`, then repeats the resulting per-trial categorical label across the 80 time bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
out[2] = early_trials[trial]
```

iii. The notes justify retaining early-lick trials because early lick itself is one of the requested decoder outputs.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. It is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 of the `data` array, together with that series’ timestamps.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The notes say this matches the reference marker-alignment code path that uses tongue-tracking data from the side camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The code bins tongue `y` into the same go-aligned 50 ms windows as the neural data and, within each bin, takes the last observed frame rather than an average. Missing bins remain `NaN` at this stage.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1

binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
```

iii. The notes explicitly cite the reference `align_markers_between_lims` behavior of taking the last frame in each time step as the reason for this choice.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. After alignment, the code computes the 40th and 60th percentiles of all finite binned tongue-`y` values from that session. Values below the 40th percentile become `0`, values from the 40th through 60th percentile become `1`, and values above the 60th percentile become `2`.

ii.
```python
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. Step 5 says this was chosen to match the decoder task’s required per-session discretization.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue position is aligned by building absolute bin edges from each trial’s go cue and then sampling the last tongue-tracking frame inside each neural bin, so the resulting tongue array has the same 80 go-aligned time bins as the neural matrix.

ii.
```python
tongue_y_binned, tongue_valid = bin_tongue_y(
    tongue_timestamps=tongue_timestamps,
    tongue_y=tongue_y,
    go_times=go_times,
    bin_edges_rel=bin_edges_rel,
)
out[3] = tongue_discrete[trial]
```

iii. The notes describe this as a shared go-cue alignment with last-frame binning chosen to mirror the reference marker code.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The code handles several edge cases explicitly: sessions with zero recorded trials or zero good units are skipped; missing sample-start events fall back to a fixed `-1.85 s` onset; no valid tongue values for a session raises an error; photostim `N/A` is treated as no stimulation; and all-zero neural trials are dropped. Missing tongue bins are left `NaN` during alignment but then implicitly become category `0` when discretized because the output array is initialized with zeros.

ii.
```python
if n_recorded_trials == 0:
    return None
if len(good_unit_idx) == 0:
    return None

if len(events):
    sample_onset_abs[trial] = events[0]
else:
    sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO

if len(valid_values) == 0:
    raise ValueError(f"{session_id}: no valid tongue_y values after alignment")

tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
```

iii. The notes justify the sample-onset fallback and session skips, and the trajectory documents that the agent considered these “sensible defaults.” The implicit `NaN -> 0` tongue handling is not called out in the notes; it follows from the implementation.

## 10-a. What are the most time-consuming steps of the code?

i. The dominant cost is per-session neural binning from raw spike times, especially the loop over all good units with `searchsorted` on every trial/bin edge. The full conversion log also shows that large sessions with many units and trials take the longest. Loading full tongue traces and other NWB arrays is secondary.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
```

iii. Step 6 identifies direct HDF5 reads and vectorized spike binning as the key speedups, which implies those were the main bottlenecks the agent was optimizing.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several trial-wise Python loops remain vectorizable: the loop that finds per-trial sample onset, the loop that constructs `input_trials`, the loop that infers choice per trial, and the loop that assembles `output_trials`. The per-unit neural loop was already partially vectorized internally, but not across units.

ii.
```python
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]

for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)

for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)

for trial in range(n_trials):
    out = np.empty((4, n_bins), dtype=np.int8)
```

iii. The notes say the agent focused its vectorization effort on the spike-binning path and accepted some remaining Python loops because they were not the main runtime bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. The code repeats `searchsorted`-based slicing for left and right licks inside every trial, repeats list-based masking when dropping zero-neural trials, and recomputes per-trial task/event slices that are only later used for plotting. It also repeatedly copies per-trial constant labels into full-length time series for `choice`, `outcome`, and `early_lick`.

ii.
```python
left_start = np.searchsorted(left_lick_times, trial_start, side="left")
left_go = np.searchsorted(left_lick_times, go_time, side="left")
left_stop = np.searchsorted(left_lick_times, trial_stop, side="right")

input_trials = [trial for keep, trial in zip(nonzero_trial_mask, input_trials) if keep]
output_trials = [trial for keep, trial in zip(nonzero_trial_mask, output_trials) if keep]

sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
delay_slice_starts, delay_slice_ends = event_slices_for_trials(delay_start_times, trial_start, trial_stop)
```

iii. This follows the code structure rather than a stated design goal; the notes mainly discuss correctness and only call out neural binning as the main optimized path.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Even when `--show-processing` is off, the code still computes `delay_slice_starts`/`delay_slice_ends`, loads `tongue_likelihood`, returns `tongue_valid` from `bin_tongue_y`, and prepares some stats/plot-related values that are not needed for the saved decoder dataset. It also materializes time-varying copies of per-trial categorical outputs, which downstream analyses may not need if they only use per-trial labels.

ii.
```python
delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)
tongue_likelihood = tongue_values[:, 2]
tongue_y_binned, tongue_valid = bin_tongue_y(...)

output_trials = []
for trial in range(n_trials):
    out = np.empty((4, n_bins), dtype=np.int8)
    out[0] = choice_trials[trial]
    out[1] = outcome_trials[trial]
    out[2] = early_trials[trial]
```

iii. The notes frame plots and diagnostics as validation aids, but from the saved pickle’s perspective these computations are not essential to the final decoder inputs.
