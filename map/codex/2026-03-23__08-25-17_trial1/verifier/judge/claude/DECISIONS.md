# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by globbing `data/sub-*/*.nwb`, yielding one HDF5 file per session. Each file is opened with `h5py` (not PyNWB) and the relevant groups (`units`, `intervals/trials`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`) are read into numpy arrays. The AI processes sessions sequentially in a for-loop, constructing a `SessionResult` per session, then assembles them into the final dictionary.

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

# In process_session():
with h5py.File(file_path, "r") as h5:
    subject_id, session_id = get_session_identity(file_path)
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    # ... reads trial_start_all, trial_stop_all, go_times_all, etc.
```

iii. The AI chose `h5py` over PyNWB for speed, noting that "Full PyNWB object loading produces avoidable overhead and warning spam." This is consistent with the data being in NWB/HDF5 format. The reference code loads `.mat` files exported from DataJoint, but the underlying data is the same MAP dataset.

## 1-b. How are the data split into subjects?

i. Subjects are identified from the parent directory name of each NWB file (e.g., `sub-440956`). The subject ID is extracted by stripping the `sub-` prefix. Unique subjects are collected and sorted alphabetically to form the `subjects` list, with `subject_idx` mapping each session to its subject.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

# In build_dataset():
subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. This is the natural way to extract subject identity from the DANDI-structured NWB dataset. The resulting 28 subjects match the paper's reported count.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The session ID is the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Sessions with zero `classification == good` units are skipped, yielding 173 sessions matching the paper.

ii.
```python
# In process_session():
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
```

iii. The AI documented that 174 raw NWB files exist, but one session (`sub-440958_ses-20190216T162508`) has zero good units, so after excluding it, 173 sessions remain, consistent with the paper's "69,943 good units recorded across 173 behavioral sessions."

## 1-d. How are the data split into trials?

i. Trials are read from the `intervals/trials` table in each NWB file. The AI extracts all trial start/stop times, behavioral labels (instruction, early_lick, outcome, photostim), and go cue times. A subset of these trials is then selected based on whether the full neural extraction window [-2.5s, +1.5s] relative to the go cue falls within the session's observation interval.

ii.
```python
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)

selected_trial_idx, session_obs_start, session_obs_stop = select_trial_indices(
    go_times_all=go_times_all,
    good_unit_obs_intervals=unit_obs_intervals,
)
```

iii. The AI justifies this by noting that some trials occur outside the neural recording window, so including them would yield meaningless neural data. The CONVERSION_NOTES document that many sessions have offset or shortened valid-trial blocks.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two trial filters: (1) a session-level observation-interval filter that retains only trials whose full [-2.5, +1.5]s go-aligned neural window lies within the union of good-unit observation intervals, and (2) a post-hoc filter that drops trials where ALL neurons have zero firing rate across all time bins. The AI does NOT apply the reference code's `get_regular_trial_mask` (which excludes early-lick, auto-water, free-water, no-response, and photostimulation trials). Additionally, per-unit `is_good_trials` masks are used to zero out neural activity for specific neuron-trial combinations.

ii.
```python
def select_trial_indices(go_times_all, good_unit_obs_intervals):
    session_obs_start = float(np.min(good_unit_obs_intervals[:, 0]))
    session_obs_stop = float(np.max(good_unit_obs_intervals[:, 1]))
    full_window_mask = (
        (go_times_all + WINDOW_START_S >= session_obs_start)
        & (go_times_all + WINDOW_END_S <= session_obs_stop)
    )
    trial_idx = np.flatnonzero(full_window_mask)
    return trial_idx, session_obs_start, session_obs_stop

# Post-hoc zero-trial filter:
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    # ... also filters all input/output arrays
```

iii. The AI reasoned that early lick, outcome, and photostimulation should be retained since they are decoder inputs/outputs. The zero-firing-rate filter was added as a robustness measure. The result is 51,346 total trials across 173 sessions (mean 296.8/session), significantly fewer than the raw ~95,000 behavioral trials.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged array of spike timestamps) for units where `units/classification == "good"`. The `spike_times_index` array is used to index into the ragged spike_times array for each unit.

ii.
```python
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)

# For each unit:
spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
```

iii. This directly uses the raw spike times from Kilosort2 spike sorting, filtered by the classifier-based QC that labels units as "good." This is consistent with the reference approach.

## 2-b. How is the `neural` data processed?

i. Raw spike times are binned into non-overlapping 50ms bins aligned to the go cue, covering [-2.5s, +1.5s]. Spike counts are converted to firing rates (Hz) by dividing by the bin width (0.05s). The result is stored as float16.

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

iii. The AI chose 50ms non-overlapping bins as explicitly required by the task instructions ("Use 50-ms-width bins for computing firing rates"). The reference code uses `bw=0.04, stride=0.0034` (40ms Gaussian-like bins with 3.4ms stride), but the task instructions override this.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are included. When `is_good_trials` column count matches the selected trials, per-unit-trial validity masks are used to zero out firing rates for invalid neuron-trial combinations. When it doesn't match, per-unit observation intervals are used to determine valid trials. Finally, trials with all-zero neural activity are dropped entirely.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")

# Per-unit validity:
if uses_direct_is_good_trials:
    invalid_trials = ~is_good_trials[unit_pos]
else:
    obs_start = unit_obs_intervals[unit_pos, 0]
    obs_stop = unit_obs_intervals[unit_pos, 1]
    valid_obs = (go_times + WINDOW_START_S >= obs_start) & (go_times + WINDOW_END_S <= obs_stop)
    invalid_trials = ~valid_obs
if np.any(invalid_trials):
    fr[invalid_trials] = 0.0
```

iii. The AI notes that `classification == good` in the NWB files represents the classifier-based QC described in the paper (69,943 good units). Invalid neuron-trial pairs are zeroed rather than excluded, preserving the matrix shape.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, the go cue timestamp from `acquisition/BehavioralEvents/go_start_times` is used as the reference time (t=0). Bin edges are computed as offsets from the go cue: `abs_edges = go_times[:, None] + bin_edges_rel[None, :]`.

ii.
```python
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
# ...
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The instructions state "Temporally align based on Go cue onset" and the reference code treats spike times as already relative to go cue time. The AI correctly uses go_start_times as the alignment anchor.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50ms (0.05s), producing 80 time bins over the [-2.5s, +1.5s] window. This is a direct binning of raw spike times into non-overlapping 50ms bins -- no temporal rebinning from an intermediate resolution is applied.

ii.
```python
BIN_WIDTH_S = 0.05
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5

def bin_edges_and_centers():
    edges = np.arange(WINDOW_START_S, WINDOW_END_S + BIN_WIDTH_S * 0.5, BIN_WIDTH_S, dtype=np.float64)
    centers = edges[:-1] + BIN_WIDTH_S / 2.0
    return edges, centers
```

iii. The task instructions specify "Use 50-ms-width bins for computing firing rates." The reference code uses 40ms bins with 3.4ms stride, but the AI correctly follows the task instructions rather than the reference preprocessing parameters.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The tone onset time is derived from `acquisition/BehavioralEvents/sample_start_times/timestamps`, which records the absolute timestamps of sample epoch (tone) onsets. The go cue time from `go_start_times` is used to compute relative timing.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
# ...
sample_onset_abs[trial] = events[0]  # earliest sample_start_times event within trial
sample_onset_rel_go = sample_onset_abs - go_times
```

iii. The reference code uses `task_sample_time` from the .mat files, which contains the same information in a different format. The AI correctly identifies the sample onset as the tone onset.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each trial, the earliest `sample_start_times` event within the trial window is identified as the tone onset. The tone onset is then expressed relative to the go cue. For each time bin, the input value is `bin_center - sample_onset_rel_go`, giving a continuous time-from-tone-onset value. If no sample event is found within a trial (rare), a fallback of -1.85s relative to go cue is used.

ii.
```python
EXPECTED_SAMPLE_ONSET_REL_GO = -1.85

for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
        sample_onset_fallbacks += 1

sample_onset_rel_go = sample_onset_abs - go_times

# Input construction:
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. The AI notes that the expected sample onset relative to go cue is approximately -1.85s based on task timing (3 tones of 150ms + 2 gaps of 100ms = 650ms sample, followed by 1.2s delay). Early-lick trials can have different timing due to epoch replay.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone-onset is computed at the same bin centers as the neural data (50ms bins from -2.5s to +1.5s relative to go cue), so it is inherently aligned. Both share the same temporal reference frame (go cue = 0).

ii.
```python
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
# bin_centers_rel are the same centers used for neural binning
```

iii. Since both neural data and this input are computed on the same go-cue-aligned 50ms bin centers, no additional alignment step is needed.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation timing is derived from the trial-table columns `intervals/trials/photostim_onset` and `intervals/trials/photostim_duration`. The onset is relative to trial start time.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])
```

iii. The reference code stores stimulation information in `task_stimulation` with columns `[laser_power, stim_type, laser_on_time, laser_off_time]` relative to go cue. The AI uses the NWB trial-table equivalent. Non-stimulation trials have `photostim_onset == "N/A"`.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. For each trial, if `photostim_onset` is not "N/A", the stimulation onset is converted to go-cue-relative time by: `stim_rel_on = trial_start + photostim_onset - go_time`. A binary time series is created where bins within `[stim_rel_on, stim_rel_on + duration)` are set to 1, and all others to 0.

ii.
```python
if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
    inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
    photostim_trial_count += 1
```

iii. The AI correctly creates a time-varying binary signal as required by the instructions ("Whether photostimulation is on at every time point"). The paper states photostimulation occurs during the late delay epoch (last 0.5s before go cue).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary vector is evaluated at the same 50ms bin centers as the neural data, using go-cue-relative coordinates. The onset is converted from trial-start-relative to go-cue-relative before comparison.

ii.
```python
# stim_rel_on is relative to go cue
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. Since bin_centers_rel are the same time points used for neural binning, the photostimulation signal is inherently aligned.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `acquisition/BehavioralEvents/left_lick_times/timestamps` and `right_lick_times/timestamps` (the raw lick event times). As a fallback for trials with no licks, `intervals/trials/trial_instruction` is used.

ii.
```python
left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)
# ...
choice_val, source = infer_choice_for_trial(
    trial_start=trial_start[trial], trial_stop=trial_stop[trial],
    go_time=go_times[trial], instruction=str(trial_instruction[trial]),
    left_lick_times=left_lick_times, right_lick_times=right_lick_times,
)
```

iii. The reference code has `lick_directions` (0=lick right, 1=lick left) directly per trial. The AI reconstructs this from raw lick event timestamps, which is a valid approach but more complex.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. A hierarchical fallback is used: (1) first post-go-cue lick determines choice (left lick first = 0, right lick first = 1); (2) if no post-go licks, first lick anywhere in the trial is used; (3) if no licks at all (ignore trials), the trial instruction is used (left instruction = 0, right = 1). The choice is encoded as a per-trial scalar broadcast across all time bins.

ii.
```python
def infer_choice_for_trial(...) -> tuple[int, str]:
    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    # ... fallback to any_lick, then instruction_fallback
    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. The AI notes this fallback is necessary because "ignore trials have no post-go lick but the decoder spec still requires a binary choice output." The encoding matches the instructions: left=0, right=1.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from `intervals/trials/outcome`, which contains string values like "hit", "miss", and "ignore".

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
```

iii. The reference code uses `correctness` (1=correct, 0=error, -1=no response). The NWB `outcome` field provides the same information with different labels.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The string outcome labels are mapped to integers: ignore -> 0, miss -> 1, hit -> 2. The result is broadcast as a per-trial scalar across all time bins.

ii.
```python
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
# In output construction:
out[1] = outcome_trials[trial]  # broadcast across all bins
```

iii. This matches the instructions: "Outcome (ignore = 0, miss = 1, hit = 2, per-trial)."

## 6-c. How is `output` *Outcome* aligned with the neural data?

i. Outcome is a per-trial scalar value that is broadcast (repeated) across all 80 time bins for each trial. Since it doesn't vary within a trial, alignment is trivial -- it shares the same trial structure as the neural data.

ii.
```python
out = np.empty((4, n_bins), dtype=np.int8)
out[1] = outcome_trials[trial]  # scalar broadcast to all bins
```

iii. No temporal alignment is needed for per-trial variables.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`, which contains string values like "early" or "no early".

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
```

iii. The reference code uses `early_lick_trials` (0 or 1). The NWB field provides string equivalents.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The string labels are mapped to integers: "no early" -> 0, "early" -> 1. The result is broadcast as a per-trial scalar across all time bins.

ii.
```python
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
out[2] = early_trials[trial]
```

iii. This matches the instructions: "Early lick (no = 0, yes = 1, per-trial)."

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking/data` (column index 1 = tongue_y) and its associated `timestamps`.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. The reference code uses `tracking/camera_0_side/tongue_y` from the .mat files. The NWB stores the same data in a standardized structure.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The continuous tongue_y time series is first aligned to each trial's go cue by converting absolute timestamps to go-cue-relative time. Within each 50ms bin, the last available tongue_y sample is taken (matching the reference code's `align_markers_between_lims` approach). The aligned values are then discretized into three categories using per-session percentiles.

ii.
```python
def bin_tongue_y(tongue_timestamps, tongue_y, go_times, bin_edges_rel):
    abs_edges = go_times[:, None] + bin_edges_rel[None, :]
    start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
    end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
    clipped_end = np.clip(end_idx, 0, len(tongue_y) - 1)
    valid = end_idx >= start_idx
    binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
    binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)
    return binned, valid
```

iii. The AI explicitly followed the reference marker alignment convention: "Reference marker alignment uses the last frame within each time step rather than averaging."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session, the 40th and 60th percentiles of all valid (non-NaN) binned tongue_y values are computed. Values below the 40th percentile get category 0, between 40th and 60th percentile get category 1, and above 60th percentile get category 2.

ii.
```python
valid_values = tongue_y_binned[np.isfinite(tongue_y_binned)]
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. This matches the instructions: "0: < 40th percentile, 1: 40th to 60th percentile, 2: > 60th percentile of y-position over the session." The percentiles are computed over the session's binned tongue_y values (pooled across all trials and time bins).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Tongue y is aligned to the go cue using absolute timestamps, then binned into the same 50ms bins as the neural data. The `bin_tongue_y` function uses `go_times[:, None] + bin_edges_rel[None, :]` to compute absolute bin edges, matching the neural binning approach.

ii.
```python
# Same bin edges as neural data:
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
# bin_edges_rel comes from bin_edges_and_centers() using WINDOW_START_S and BIN_WIDTH_S
```

iii. Since the same go-cue-aligned bin edges are used for both neural data and tongue_y, the signals are temporally aligned.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several edge cases are handled:
- Missing sample onset events: fallback to expected timing (-1.85s relative to go cue).
- Missing tongue data in a bin: `bin_tongue_y` returns NaN for bins with no tongue samples; these get category 0 in discretization (since NaN < p40 is False, they fall into the default zeros).
- Invalid neuron-trial combinations: firing rates are zeroed using `is_good_trials` or `obs_intervals`.
- All-zero neural trials: dropped entirely after neural binning.
- Sessions with zero good units: skipped.
- Mismatched `is_good_trials` column count: fallback to obs_intervals-based validity.

ii.
```python
# Missing sample onset:
if len(events):
    sample_onset_abs[trial] = events[0]
else:
    sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
    sample_onset_fallbacks += 1

# Tongue NaN handling:
binned = np.full(end_idx.shape, np.nan, dtype=np.float32)
binned[valid] = tongue_y[clipped_end[valid]].astype(np.float32)

# Zero-neural-trial dropping:
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. The AI documented these edge cases in CONVERSION_NOTES.md and verified them with spot checks across multiple sessions, including sessions with large trial offsets.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is the neural spike binning loop, which iterates over each good unit and uses `np.searchsorted` across all trial/bin edges to count spikes. With ~400 good units per session and ~300 trials each with 80 bins, this is the computational bottleneck. Reading the HDF5 file is the second most time-consuming step.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
```

iii. The AI noted the full conversion takes about 1.6s per session (~4.6 minutes for 173 sessions), which is reasonable. The vectorized `searchsorted` approach is much faster than a naive Python triple loop.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several Python for-loops could potentially be vectorized:
- The per-trial sample onset loop (lines 314-320) could be vectorized by finding the first sample event per trial using searchsorted.
- The per-trial input construction loop (lines 327-336) could be vectorized using broadcasting.
- The per-trial choice inference loop (lines 341-351) could be partially vectorized.
- The per-trial output construction loop (lines 375-381) could be vectorized.

ii.
```python
# Sample onset loop:
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO

# Input construction loop:
for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
    # ... photostim logic

# Choice inference loop:
for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
```

iii. While these loops add to runtime, they are not the primary bottleneck compared to the spike binning. The AI prioritized vectorizing the spike binning (the dominant cost) and kept simpler logic in Python loops.

## 10-c. What processing does the code repeat multiple times?

i. The code computes absolute bin edges twice: once in `bin_tongue_y` (`go_times[:, None] + bin_edges_rel[None, :]`) and once in the neural binning section (`abs_edges = go_times[:, None] + bin_edges_rel[None, :]`). The `event_slices_for_trials` function is called twice (for sample and delay events). The `decode_str_array` function is called multiple times for different string arrays.

ii.
```python
# In bin_tongue_y:
abs_edges = go_times[:, None] + bin_edges_rel[None, :]

# In neural binning (same computation):
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
```

iii. The repeated computations are relatively cheap compared to the spike binning and don't significantly impact performance.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several pieces of data are loaded and processed but not directly used in the final output:
- `tongue_likelihood` (column 2 of tongue tracking) is loaded but only used for diagnostic plotting, not for filtering or output.
- `delay_start_times` are loaded and sliced per trial but only used in processing plots, not in any input/output.
- The `is_good_trials` per-unit masking zeros out neural activity for some neuron-trial pairs, but the zeroed values are still included in the output (rather than being marked as missing).
- Sample and delay event slices are computed for all trials but only used in plots for a single example trial.
- The `plot_payload` data structures are constructed even when not needed (though gated by `show_processing` flag).

ii.
```python
tongue_likelihood = tongue_values[:, 2]  # loaded but only used in plot_payload
delay_start_times = h5["acquisition"]["BehavioralEvents"]["delay_start_times"]["timestamps"][:].astype(np.float64)  # only used for plots
```

iii. The unnecessary processing is minor in terms of computational cost. The tongue_likelihood and delay_start_times loading adds negligible overhead.
