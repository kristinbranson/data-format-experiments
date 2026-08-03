# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB session files by globbing `data/sub-*/*.nwb` and sorts them. Each file is opened with `h5py` directly (not pynwb) for performance. All trial, unit, and behavioral data are read from the HDF5 groups within each file (`intervals/trials`, `units`, `acquisition/BehavioralEvents`, `acquisition/BehavioralTimeSeries`).

ii.
```python
def get_session_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.glob("sub-*/*.nwb"))

# In process_session:
with h5py.File(file_path, "r") as h5:
    units = h5["units"]
    trials = h5["intervals"]["trials"]
    go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
    sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)
    tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
```

iii. The AI chose h5py over pynwb for speed, noting in CONVERSION_NOTES that "direct h5py reads instead of PyNWB objects removed heavy object construction and warning spam." The glob pattern matches the DANDI directory layout (one NWB file per session under `sub-<id>/` folders).

## 1-b. How are the data split into subjects?

i. Subject IDs are extracted from the parent folder name of each NWB file (e.g., `sub-440956` → `"440956"`), not from the NWB metadata fields. Unique subjects are sorted and indexed.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id

# In build_dataset:
subjects = sorted({r.subject_id for r in results})
subject_to_idx = {subject: idx for idx, subject in enumerate(subjects)}
```

iii. The AI derives subject identity from the folder naming convention rather than reading `nwb.subject.subject_id`. Both yield the same numeric IDs (28 subjects), since the folder names follow the pattern `sub-<subject_id>`.

## 1-c. How are the data split into sessions?

i. One NWB file equals one session. Session ID is the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). After excluding the one session with zero good units, 173 sessions remain.

ii.
```python
def get_session_identity(file_path: Path) -> tuple[str, str]:
    subject_id = file_path.parent.name.replace("sub-", "")
    session_id = file_path.stem
    return subject_id, session_id
```

iii. The AI documents that 174 raw NWB files become 173 analyzable sessions after excluding the one zero-good-unit session (`sub-440958_ses-20190216T162508`), matching the paper's reported 173 sessions.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The AI reads all trial metadata columns and go-cue timestamps, then applies filtering to select a valid subset.

ii.
```python
trials = h5["intervals"]["trials"]
n_behavior_trials = int(len(trials["id"]))
trial_start_all = trials["start_time"][:].astype(np.float64)
trial_stop_all = trials["stop_time"][:].astype(np.float64)
go_times_all = h5["acquisition"]["BehavioralEvents"]["go_start_times"]["timestamps"][:n_behavior_trials].astype(np.float64)
```

iii. The AI reads the full behavioral trial table and then filters down to valid trials based on neural observation windows (see 1-e).

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two filtering stages: (1) Trials are selected whose full [-2.5, +1.5] s go-aligned neural window lies within the session-wide observation interval (min start to max stop of good-unit `obs_intervals`). (2) After neural binning, trials where all neural activity is zero across all units and bins are dropped. No `free_water` filter is applied. This yields 51,346 total trials across 173 sessions (~297/session).

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

# Post-hoc zero-trial dropping:
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
if dropped_zero_trials:
    neural_session = neural_session[nonzero_trial_mask]
    # ... also filter all other trial arrays
```

iii. The AI's CONVERSION_NOTES state: "Do not assume `units/is_good_trials.shape[1]` means 'use the first N behavioral trials'. In several sessions the ephys-backed trials form an offset block relative to the full behavioral trial table; use raw go times plus `obs_intervals` to select valid trials." The AI also notes using "the corrected converter [that] selects trials using the raw neural support window instead of assuming the first N behavioral trials line up with `units/is_good_trials`."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times for each unit) with indexing via `units/spike_times_index`. Only units with `classification == 'good'` are used.

ii.
```python
spike_times = units["spike_times"][:].astype(np.float64)
spike_times_index = units["spike_times_index"][:]
spike_starts, spike_ends = get_ragged_row_bounds(spike_times_index)

classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
```

iii. Same raw data as the reference — spike times are the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50 ms non-overlapping bins spanning [-2.5, 1.5] s relative to the go cue, giving 80 bins. Spike counts per bin are computed via `np.searchsorted` and `np.diff`, then divided by bin width (0.05 s) to get firing rates in Hz. The result is stored as `float16`. Additionally, the AI uses `is_good_trials` (a per-unit × per-trial boolean) to zero out firing rates for trials where a unit's data is flagged as invalid.

ii.
```python
neural_session = np.zeros((n_trials, n_units, n_bins), dtype=np.float16)
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)

for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
    counts = np.diff(edge_indices, axis=1).astype(np.float32)
    fr = counts / bin_width
    if uses_direct_is_good_trials:
        invalid_trials = ~is_good_trials[unit_pos]
    else:
        # ... fallback obs_intervals check
        invalid_trials = ~valid_obs
    if np.any(invalid_trials):
        fr[invalid_trials] = 0.0
    neural_session[:, unit_pos, :] = fr.astype(np.float16)
```

iii. The AI noted using float16 to "reduce memory/pickle size while remaining valid floating-point input for the decoder." It also uses `is_good_trials` for per-unit per-trial validity masking, which is not present in the reference code.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` (lowercased) are kept. Sessions with zero good units are skipped. This retains 69,453 good units across 173 sessions.

ii.
```python
classification = np.char.lower(decode_str_array(units["classification"][:]))
good_unit_idx = np.flatnonzero(classification == "good")
if len(good_unit_idx) == 0:
    print(f"Skipping {session_id}: zero good units")
    return None
```

iii. The AI's CONVERSION_NOTES confirm: "Use units with `classification == good` as the primary QC filter, because the papers and code describe classifier-selected good units as the analysis set."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue. Absolute bin edges are computed as `go_times[:, None] + bin_edges_rel[None, :]`, and spikes are binned against these edges using `searchsorted`.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
flat_edges = abs_edges.reshape(-1)
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)
```

iii. Go cue alignment is consistent with the instructions and reference papers.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins spanning [-2.5, 1.5] s relative to the go cue, giving 80 bins per trial. No rebinning — spike times are binned directly from raw timestamps.

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

iii. Matches the instructions exactly (50 ms bins, -2.5 to 1.5 s).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. Derived from `sample_start_times` (the tone onset events) and `go_start_times`. For each trial, the AI takes the **earliest** (first) `sample_start_times` event within the trial window, rather than the last one before the go cue.

ii.
```python
sample_start_times = h5["acquisition"]["BehavioralEvents"]["sample_start_times"]["timestamps"][:].astype(np.float64)

sample_slice_starts, sample_slice_ends = event_slices_for_trials(sample_start_times, trial_start, trial_stop)
for trial in range(n_trials):
    events = sample_start_times[sample_slice_starts[trial]:sample_slice_ends[trial]]
    if len(events):
        sample_onset_abs[trial] = events[0]  # FIRST event
    else:
        sample_onset_abs[trial] = go_times[trial] + EXPECTED_SAMPLE_ONSET_REL_GO
```

iii. The AI's CONVERSION_NOTES state: "Use the earliest sample-start event in each trial as tone onset. This preserves replay-induced timing shifts visible in early-lick trials and is the most faithful raw-data representation of 'time from tone onset'." A fallback value of go_time + (-1.85) is used when no sample event is found.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin, the value is computed as `bin_center_rel_go - sample_onset_rel_go`, which gives time in seconds since the tone onset.

ii.
```python
sample_onset_rel_go = sample_onset_abs - go_times

# In the per-trial loop:
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Mathematically equivalent to the reference computation `CENTERS + (go - tone)`, just organized differently.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin centers are used for both the neural data and the time-from-tone input, ensuring alignment.

ii.
```python
bin_edges_rel, bin_centers_rel = bin_edges_and_centers()
inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)
```

iii. Both neural and input data share the same go-cue-relative time grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` columns in the trials table, with `start_time` and go cue used to compute relative timing.

ii.
```python
photostim_onset_all = decode_str_array(trials["photostim_onset"][:])
photostim_duration_all = decode_str_array(trials["photostim_duration"][:])

if str(photostim_onset[trial]) != "N/A":
    stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
    stim_rel_off = stim_rel_on + float(photostim_duration[trial])
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A binary time series: 1 during [onset, onset+duration) relative to go cue, 0 elsewhere. Non-stimulated trials (marked `'N/A'`) stay all zeros.

ii.
```python
inp[1] = ((bin_centers_rel >= stim_rel_on) & (bin_centers_rel < stim_rel_off)).astype(np.float32)
```

iii. Same logic as the reference.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Onset and offset are expressed relative to the go cue, matching the neural bin centers.

ii.
```python
stim_rel_on = trial_start[trial] + float(photostim_onset[trial]) - go_times[trial]
```

iii. Same alignment approach as the reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is inferred from actual lick events: `left_lick_times` and `right_lick_times` from `BehavioralEvents`, plus `trial_instruction` as a fallback. The AI uses a priority hierarchy: (1) first post-go lick direction, (2) first lick anywhere in the trial, (3) trial instruction for no-lick trials.

ii.
```python
left_lick_times = h5["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][:].astype(np.float64)
right_lick_times = h5["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][:].astype(np.float64)

def infer_choice_for_trial(trial_start, trial_stop, go_time, instruction, left_lick_times, right_lick_times):
    left_post = left_lick_times[left_go:left_stop]
    right_post = right_lick_times[right_go:right_stop]
    if len(left_post) or len(right_post):
        first_left = left_post[0] if len(left_post) else np.inf
        first_right = right_post[0] if len(right_post) else np.inf
        return (0, "post_go_lick") if first_left < first_right else (1, "post_go_lick")
    # ... fallbacks ...
    return (0 if instruction == "left" else 1, "instruction_fallback")
```

iii. The AI's CONVERSION_NOTES describe a "choice fallback rule: first post-go lick; else first lick anywhere in trial; else trial instruction for no-lick ignore trials." This differs from the reference which derives choice from `trial_instruction × outcome`.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is encoded as left=0, right=1 with only 2 classes. For ignore trials (no lick), the AI falls back to the trial instruction direction rather than creating a "no lick" category. The value is repeated across all 80 bins.

ii.
```python
choice_trials[trial] = choice_val  # 0 or 1

out[0] = choice_trials[trial]

# In build_dataset:
"output_values": [
    ["left", "right"],  # only 2 classes
    ...
]
```

iii. The AI's approach gives ignore trials a fabricated choice direction based on instruction, rather than a separate "no lick" class as in the reference.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. From the `outcome` column of the trials table, which contains `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_all = np.char.lower(decode_str_array(trials["outcome"][:]))
outcome_map = {"ignore": 0, "miss": 1, "hit": 2}
outcome_trials = np.array([outcome_map[str(x)] for x in outcome], dtype=np.int8)
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Mapped to ignore=0, miss=1, hit=2 and repeated across all 80 bins.

ii.
```python
out[1] = outcome_trials[trial]
```

iii. Same encoding as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table (`'early'` / `'no early'`).

ii.
```python
early_lick_all = np.char.lower(decode_str_array(trials["early_lick"][:]))
early_map = {"no early": 0, "early": 1}
early_trials = np.array([early_map[str(x)] for x in early_lick], dtype=np.int8)
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to no=0, yes=1 and repeated across all 80 bins.

ii.
```python
out[2] = early_trials[trial]
```

iii. Same encoding as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, specifically column 1 (tongue_y) of the `(n_frames, 3)` data array with explicit timestamps.

ii.
```python
tongue_group = h5["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]
tongue_values = tongue_group["data"][:].astype(np.float64)
tongue_y = tongue_values[:, 1]
tongue_likelihood = tongue_values[:, 2]
tongue_timestamps = tongue_group["timestamps"][:].astype(np.float64)
```

iii. Same source as the reference. The AI reads tongue_likelihood but does not use it for filtering (see 8-b).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI takes the **last frame** within each 50 ms bin (not the average of visible frames). No likelihood-based filtering is applied — all frames are used regardless of tracking confidence. Session-wide percentiles (40th/60th) are computed over all finite binned values. Discretization produces 3 classes: 0 (below 40th), 1 (40th to 60th inclusive), 2 (above 60th). There is no "not visible" class.

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

# Discretization:
tongue_p40 = float(np.percentile(valid_values, 40))
tongue_p60 = float(np.percentile(valid_values, 60))
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2
```

iii. The AI's CONVERSION_NOTES describe: "Tongue alignment uses last-frame-within-bin alignment, which matches the marker-alignment convention in the reference code" and "Use tongue_likelihood only for QC diagnostics, not thresholding, to stay close to reference code."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles computed over all finite binned tongue_y values. Discretized into 3 classes: 0 (< p40), 1 (>= p40 and <= p60), 2 (> p60). The boundary between class 0 and 1 uses `>=` (inclusive), and between 1 and 2 uses `>` (exclusive at p60 for class 1). No fourth "not visible" class exists.

ii.
```python
tongue_discrete = np.zeros_like(tongue_y_binned, dtype=np.int8)
tongue_discrete[(tongue_y_binned >= tongue_p40) & (tongue_y_binned <= tongue_p60)] = 1
tongue_discrete[tongue_y_binned > tongue_p60] = 2

# output_values:
["lt_p40", "p40_to_p60", "gt_p60"]  # 3 classes only
```

iii. The reference uses `np.digitize` with edges [p40, p60], giving strict boundaries and 4 classes including "not visible" for bins with no visible tongue frame.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Camera timestamps are on the same session clock as spikes and go cues. Bin edges are computed as `go_times + bin_edges_rel`, and searchsorted finds the frame indices for each bin boundary. The last frame in each bin is selected.

ii.
```python
abs_edges = go_times[:, None] + bin_edges_rel[None, :]
start_idx = np.searchsorted(tongue_timestamps, abs_edges[:, :-1], side="left")
end_idx = np.searchsorted(tongue_timestamps, abs_edges[:, 1:], side="left") - 1
```

iii. Same go-cue-relative alignment as the neural data.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases: (1) Sessions with zero good units are skipped entirely. (2) Trials outside the neural observation window are excluded by the window-based filter. (3) Trials with all-zero neural activity post-binning are dropped. Tongue bins without frames get NaN which becomes class 0 after discretization (not a special class). No free_water filtering is applied.

ii.
```python
if len(good_unit_idx) == 0:
    return None

# Window-based trial filter
full_window_mask = (go_times_all + WINDOW_START_S >= session_obs_start) & (go_times_all + WINDOW_END_S <= session_obs_stop)

# Post-hoc zero-trial drop
nonzero_trial_mask = np.any(neural_session > 0, axis=(1, 2))
```

iii. The AI chose not to filter free_water trials and instead relies on the window-based and all-zero filters to handle edge cases.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB/HDF5 files dominates. The AI reports full conversion taking about 4-5 minutes for 173 sessions, with ~1.6 s per session. Loading the full `spike_times` array and tongue tracking data are the main per-session costs.

ii.
```python
with h5py.File(file_path, "r") as h5:
    spike_times = units["spike_times"][:].astype(np.float64)
    tongue_values = tongue_group["data"][:].astype(np.float64)
```

iii. The AI noted switching from pynwb to h5py as a speedup, and the per-session timing is comparable to the reference.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit neural binning loop (`for unit_pos, unit_idx in enumerate(good_unit_idx)`) iterates over each unit individually. The per-trial input construction loop and the choice inference loop (with per-trial lick event lookups) could potentially be vectorized.

ii.
```python
for unit_pos, unit_idx in enumerate(good_unit_idx):
    spikes = spike_times[spike_starts[unit_idx]:spike_ends[unit_idx]]
    edge_indices = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_bins + 1)

for trial in range(n_trials):
    inp = np.zeros((2, n_bins), dtype=np.float32)
    inp[0] = (bin_centers_rel - sample_onset_rel_go[trial]).astype(np.float32)

for trial in range(n_trials):
    choice_val, source = infer_choice_for_trial(...)
```

iii. The per-unit loop is inherent to ragged spike storage. The input construction and choice loops could be vectorized but are not performance bottlenecks.

## 10-c. What processing does the code repeat multiple times?

i. The `decode_str_array` function is called multiple times on different columns. Otherwise, no significant computation is repeated — each NWB file is opened once and all quantities are computed in a single pass.

ii. N/A

iii. The single-pass design avoids redundant work.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes and stores `is_good_trials` per-unit per-trial masking and uses it to zero out invalid neural data, which is extra processing beyond what the reference does. It also computes detailed statistics (`stats` dict) and optional plot payloads that are not part of the final pickle output.

ii.
```python
is_good_trials_raw = units["is_good_trials"][good_unit_idx, :n_recorded_trials].astype(bool)
# ...
stats = {
    "session_id": session_id,
    "selected_trial_start_index": int(selected_trial_idx[0]),
    # ... many statistics
}
```

iii. The stats dictionary is useful for debugging but adds computation that doesn't affect the output data.
