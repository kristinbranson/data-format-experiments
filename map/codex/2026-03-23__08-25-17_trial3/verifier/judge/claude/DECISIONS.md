# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads data from NWB files organized under `data/sub-*/`. It first scans for all NWB files using `sorted(DATA_DIR.glob("sub-*/*.nwb"))`, then pre-filters to only sessions with at least one `classification == "good"` unit. Each file is opened with `h5py` (not `pynwb`) and processed in `process_session()`. Trials, units, behavioral events, and tongue tracking are all read from within each HDF5 file.

ii.
```python
def get_nwb_files(sample_only: bool) -> list[Path]:
    files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
    valid = []
    for path in files:
        with h5py.File(path, "r") as f:
            good = decode_str_array(f["units/classification"]) == "good"
            if np.any(good):
                valid.append(path)
    if sample_only:
        return valid[:SAMPLE_SESSION_COUNT]
    return valid
```

```python
with h5py.File(path, "r") as f:
    trials = f["intervals/trials"]
    ...
    classification = decode_str_array(f["units/classification"])
    ...
```

iii. The AI chose `h5py` over `pynwb` for lower overhead and explicit access to ragged arrays. The pre-filtering step opens every file twice (once to check for good units, once to process), but ensures only valid sessions enter the pipeline.

## 1-b. How are the data split into subjects?

i. Subject IDs are derived from the parent folder name of each NWB file (e.g., `sub-440956`), not from `nwb.subject.subject_id`. The `subject_id` is stored as the folder name string. At assembly time, subjects are accumulated in order of first appearance.

ii.
```python
session_id = path.stem
subject_id = path.parent.name
```

```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The folder name encodes the subject ID in DANDI convention. The AI uses this directly rather than reading `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. Sessions are identified by the file stem (e.g., `sub-440956_ses-20190207T120657_behavior+ecephys+ogen`). Session order follows sorted file paths. Only sessions with at least one good unit are included (173 of 174).

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
```

```python
session_id = path.stem
```

iii. The AI notes in CONVERSION_NOTES that one session (`sub-440958_ses-20190216T162508`) has zero good units and is excluded, yielding 173 sessions matching the papers.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The trial count is verified to match the number of go cue events. Trials are then filtered based on obs_intervals and nonzero spike activity.

ii.
```python
n_trials_raw = len(trials["id"])
...
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
```

iii. The AI checks that the number of go cue timestamps equals the number of trials, ensuring a one-to-one mapping.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered in two stages: (1) `obs_intervals`-based validity check — trials must fall within a unit's observed recording intervals, and (2) removal of trials where all good units have zero spikes in the trial window. No behavioral quality filter (e.g., no `free_water` exclusion) is applied.

ii.
```python
def compute_valid_trial_mask(obs_intervals, go_times):
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)
```

```python
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
    ...
```

iii. The AI uses obs_intervals to exclude trials outside the recording window, then removes any remaining all-zero trials as a defensive check. Unlike the reference, the AI does NOT filter out `free_water` trials explicitly. The AI's approach checks whether the full [-2.5, 1.5] window is covered by any obs_interval, which is a stricter geometric containment check rather than the reference's approach of matching trial start times.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (the sorted spike times) and `units/spike_times_index` (the ragged index). Only units with `classification == "good"` contribute. Go cue times from `BehavioralEvents/go_start_times` are used for alignment.

ii.
```python
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
firing_rates = bin_spikes_to_firing_rates(
    spike_times_flat=spike_times_flat,
    spike_times_index=spike_times_index,
    good_unit_indices=good_unit_indices,
    trial_edges_abs=trial_edges_abs,
)
```

iii. Same source variables as the reference.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 80 non-overlapping 50 ms bins spanning [-2.5, 1.5] s relative to the go cue. For each good unit, spike counts per bin are computed via `np.searchsorted` and then divided by the bin width to convert to firing rates in Hz. Results are stored as `float16`.

ii.
```python
def bin_spikes_to_firing_rates(...):
    ...
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates
```

iii. The processing matches the reference approach (searchsorted binning, divide by bin width). The key difference is using `float16` instead of `float32`, which loses precision.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept. A session with no good units raises a ValueError (pre-filtered in `get_nwb_files`). This retains 69,453 of 272,227 units.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

iii. Matches the reference approach of using the QC classifier verdict.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as offsets from each trial's go cue time. The absolute times of the bin edges are `go_times[:, None] + REL_EDGES[None, :]`, and spikes are binned against these edges.

ii.
```python
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. Same approach as the reference — both use go-cue-centered alignment.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins, 80 bins spanning [-2.5, 1.5] s. No rebinning is applied — spikes are binned directly at 50 ms resolution. Bin edges are computed using `np.linspace`.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
```

iii. Matches the task instructions. The reference uses `T_START + BIN * np.arange(N_BINS + 1)` which produces equivalent edges.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI uses a **fixed canonical offset** of -1.85 s relative to the go cue, derived from task structure (0.65 s sample epoch + 1.2 s delay = 1.85 s before go). It does NOT read `sample_start_times` from the NWB file.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
...
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The AI argued that raw NWB sample-event streams contain replay-related extra events and are not reliable one-to-one trial markers, so using the canonical `-1.85 s` tone onset is more consistent with the task definition. This means time_from_tone_onset is identical for every trial.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The bin centers (relative to go cue) are shifted by subtracting the fixed tone onset offset. Since the offset is constant, the result is the same for every trial: `REL_CENTERS - (-1.85)`. The result is stored as `float16`.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. No per-trial variation in time_from_tone_onset.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. It uses the same bin center grid as the neural data (`REL_CENTERS`), so alignment is inherent.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. Same grid ensures alignment.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, plus `start_time` and go times to convert to go-cue-relative coordinates.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Onset and duration strings are parsed (with `'N/A'` becoming NaN). Onset relative to go is computed as `onset_trial - (go - start)`. A bin is set to 1 if its center falls between onset and offset (onset + duration). Non-stim trials stay all zeros. Result stored as `float16`.

ii.
```python
def build_photostim_matrix(...):
    onset_trial = parse_optional_float_array(photostim_onset_str)
    duration = parse_optional_float_array(photostim_duration_str)
    go_minus_start = go_times - start_times
    onset_rel_go = onset_trial - go_minus_start
    offset_rel_go = onset_rel_go + duration
    stim = np.zeros((len(go_times), N_BINS), dtype=np.float16)
    valid = np.isfinite(onset_rel_go) & np.isfinite(offset_rel_go)
    for trial_idx in np.where(valid)[0]:
        mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
        stim[trial_idx, mask] = 1.0
    return stim
```

iii. Matches the reference approach. Both convert photostim onset from trial-start-relative to go-cue-relative coordinates, then create a binary time series.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. The photostim binary vector is evaluated at the same `REL_CENTERS` as the neural data, so alignment is inherent.

ii.
```python
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```

iii. Same approach as reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction` (left/right) and `outcome` (hit/miss/ignore). For ignore trials, the AI additionally reads `left_lick_times` and `right_lick_times` from BehavioralEvents to attempt to determine which side the animal licked.

ii.
```python
left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
choice_code = build_choice_array(
    trial_instruction=trial_instruction,
    outcome_code=outcome_code,
    start_times=start_times,
    go_times=go_times,
    stop_times=stop_times,
    left_lick_times=left_lick_times,
    right_lick_times=right_lick_times,
)
```

iii. The AI differs from the reference here. The reference assigns ignore trials a separate "no lick" class (code 2), while the AI attempts to find a lick direction for ignore trials using a fallback chain.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice = instructed side. For miss trials, choice = opposite of instructed side. For ignore trials, the AI uses a multi-step fallback: (1) first post-go lick direction, (2) first lick anywhere in trial, (3) instructed side. Choice is coded as left=0, right=1 with only two values (no "no lick" class). Output values list is `["left", "right"]`.

ii.
```python
def lick_choice_with_fallback(left_times, right_times, start_time, go_time, stop_time, instructed_choice):
    left_post = left_times[(left_times >= go_time) & (left_times <= stop_time)]
    right_post = right_times[(right_times >= go_time) & (right_times <= stop_time)]
    post_choice = first_side(left_post, right_post)
    if post_choice is not None:
        return post_choice
    left_any = left_times[(left_times >= start_time) & (left_times <= stop_time)]
    right_any = right_times[(right_times >= start_time) & (right_times <= stop_time)]
    any_choice = first_side(left_any, right_any)
    if any_choice is not None:
        return any_choice
    return instructed_choice
```

```python
"output_values": [
    ["left", "right"],
    ...
]
```

iii. The AI chose not to create a "no lick" class for ignore trials, instead always assigning a left/right value. This is a significant departure from the reference, which uses three classes (left=0, right=1, no_lick=2).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. Same source as reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. Maps `ignore -> 0`, `miss -> 1`, `hit -> 2`. Per-trial value repeated across all 80 bins.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
...
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
```

iii. Matches the reference approach.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. Same source as reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Maps `"no early" -> 0`, `"early" -> 1`. Per-trial value repeated across all 80 bins.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
...
np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
```

iii. Matches the reference approach.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(x, y, likelihood)` columns, plus timestamps.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. Same source as reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a multi-step cleaning pipeline: (1) compute frame-to-frame speed, (2) identify outliers exceeding mean + 5*std, (3) interpolate outlier positions from neighbors, (4) replace low-likelihood frames (likelihood < 0.9) with the mean y of visible frames. Then the cleaned y is aligned to bin centers using last-frame-carried-forward (`searchsorted` with `side="right"` minus 1). Session-wide percentiles (40th, 60th) are computed over ALL aligned tongue-y values (across all trials and bins), and the signal is discretized into 3 classes.

ii.
```python
def clean_tongue_tracking(x, y, likelihood):
    ...
    speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
    ...
    x[outlier_mask] = np.interp(...)
    y[outlier_mask] = np.interp(...)
    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    y[occluded_mask] = mean_y
    return y, diagnostics

def align_tongue_y(timestamps, cleaned_y, go_times):
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The AI's tongue processing differs significantly from the reference in several ways: (1) uses likelihood threshold of 0.9 vs reference's 0.5, (2) imputes low-likelihood frames with mean-y rather than setting to NaN, (3) does NOT have a "not visible" class (always produces 3 classes, never class 3), (4) computes percentiles over all aligned bin values (including imputed ones) rather than over session-wide 50ms bin means of only visible frames, (5) alignment uses last-frame-carried-forward on the cleaned (imputed) data.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles are computed from all aligned tongue-y values (flattened across all trials and bins). Values below 40th percentile get class 0, between 40th-60th get class 1, above 60th get class 2. There is no "not visible" class.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The reference uses `np.nanpercentile` on bin means of visible-only frames, creating 4 classes (0, 1, 2, 3=not visible). The AI always assigns one of 3 classes since low-likelihood frames were already imputed.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The cleaned tongue-y values are aligned to neural bin centers using last-frame-carried-forward: for each bin center, the index of the last camera frame before that time is found via `searchsorted(..., side="right") - 1`.

ii.
```python
def align_tongue_y(timestamps, cleaned_y, go_times):
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The reference averages all frames within each 50ms bin and sets bins with no visible frames to NaN/class 3. The AI takes a single sample per bin center instead of averaging, and never produces empty bins since all frames have imputed values.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three main cases: (1) Session with no good units — raises ValueError, pre-filtered in `get_nwb_files`. (2) Trials outside obs_intervals — filtered by `compute_valid_trial_mask`. (3) All-zero neural trials after binning — removed by `nonzero_trial_mask`. (4) Low-likelihood tongue frames — imputed with mean visible y-value. (5) `photostim_onset == "N/A"` — parsed as NaN, resulting in no stim for those trials.

ii.
```python
# Session with no good units
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")

# Trials outside obs_intervals
valid_trial_mask = compute_valid_trial_mask(...)

# All-zero neural trials
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))

# Low-likelihood tongue
y[occluded_mask] = mean_y
```

iii. The AI handles missing data through filtering (sessions, trials) or imputation (tongue). The reference uses similar approaches for sessions and trials, but handles tongue differently (NaN + separate class).

## 10-a. What are the most time-consuming steps of the code?

i. Full conversion took 7.24 minutes for 173 sessions. Per-session, the dominant costs are reading the HDF5 file and spike binning. The AI reports binning time separately in diagnostics.

ii.
```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
neural_time_s = time.perf_counter() - t_neural
```

iii. Similar to reference — I/O and per-unit searchsorted dominate.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop iterates over each good unit, running one `searchsorted` per unit. This is necessary because each unit has a different number of spikes (ragged storage). The photostim matrix construction also loops over stimulated trials but this is a small fraction.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
```

```python
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0
```

iii. Same inherent limitation as reference — ragged spike arrays prevent full vectorization.

## 10-c. What processing does the code repeat multiple times?

i. The AI opens each NWB file twice: once in `get_nwb_files` to check for good units, and once in `process_session` to actually process the data. The `photostim_onset` strings are parsed twice (once in `process_session` for building the stim matrix, once after for diagnostics).

ii.
```python
# First pass in get_nwb_files:
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"

# Second pass in main loop:
for i, path in enumerate(files, start=1):
    result = process_session(path)
```

iii. The double-open is a minor inefficiency. The reference avoids this by checking inside `process_session` and returning `None` for dropped sessions.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The tongue cleaning pipeline computes speed-based outlier detection and interpolation, which adds processing that the reference does not do and that may not improve decoder performance. The diagnostics dictionary collects extensive per-session metadata that is not saved to the final pickle. The `clean_tongue_tracking` function processes x-coordinates even though only y is used downstream.

ii.
```python
speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
...
x[outlier_mask] = np.interp(...)  # x is cleaned but never used
```

iii. The x-coordinate cleaning and speed computation add overhead for data that is ultimately discarded. The diagnostic collection is useful for debugging but not part of the output.
