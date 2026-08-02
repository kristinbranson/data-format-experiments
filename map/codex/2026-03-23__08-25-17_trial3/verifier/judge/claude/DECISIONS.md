# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads raw NWB files directly using `h5py`. It discovers all NWB files by globbing `data/sub-*/*.nwb`, sorts them, and then pre-filters to only those sessions that have at least one unit with `classification == "good"`. Each NWB file is opened individually in `process_session()`, and trial, unit, and behavioral data are extracted from the HDF5 groups (`intervals/trials`, `units/`, `acquisition/BehavioralEvents/`, `acquisition/BehavioralTimeSeries/`).

ii.
```python
DATA_DIR = Path("data")

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

iii. The AI documented that the NWB files are a DANDI-style dataset with 174 files across 28 subjects. One session (`sub-440958_ses-20190216T162508`) has zero good units and is excluded, leaving 173 sessions matching the paper's reported count. The AI chose h5py over higher-level NWB wrappers for lower overhead and explicit access to ragged arrays.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified by the parent directory name of each NWB file (e.g., `sub-440956`). During dataset assembly in `build_dataset()`, unique subject IDs are collected and each session is assigned a subject index.

ii.
```python
def process_session(path: Path) -> SessionResult:
    ...
    subject_id = path.parent.name
    ...

def build_dataset(results: list[SessionResult]) -> dict:
    ...
    for session_idx, result in enumerate(results):
        if result.subject_id not in subject_to_idx:
            subject_to_idx[result.subject_id] = len(subjects)
            subjects.append(result.subject_id)
        data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The AI noted 28 subject folders in the dataset, matching the paper's report of 28 mice.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI processes sessions sequentially, one file at a time. The session ID is the NWB file stem (filename without extension).

ii.
```python
def process_session(path: Path) -> SessionResult:
    ...
    session_id = path.stem
    ...

def main() -> None:
    ...
    files = get_nwb_files(sample_only=sample_only)
    ...
    for i, path in enumerate(files, start=1):
        result = process_session(path)
        results.append(result)
```

iii. The AI documented that 174 NWB files exist but one is excluded for having zero good units, yielding 173 analyzed sessions matching the paper.

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` table in each NWB file. The AI reads trial start/stop times, go cue times from `acquisition/BehavioralEvents/go_start_times/timestamps`, and trial-level metadata (instruction, outcome, early_lick, photostim). Trials are then filtered (see 1-e) before processing.

ii.
```python
trials = f["intervals/trials"]
n_trials_raw = len(trials["id"])
...
start_times_all = trials["start_time"][()].astype(np.float64)
stop_times_all = trials["stop_time"][()].astype(np.float64)
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
```

iii. The AI verified that go cue count matches trial count in each session and uses go cue times as the primary alignment anchor.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two stages of trial filtering:
1. **Observation interval filter**: Uses the first good unit's `obs_intervals` to check whether each trial's full neural window `[go - 2.5, go + 1.5]` falls within the recording period.
2. **Zero-spike removal**: After spike binning, any trial with zero spikes across all good units and all time bins is removed.

The AI does NOT apply the reference code's "regular trial" filter (which excludes early lick, auto water, free water, no-response, and stimulation trials), because these variables are required decoder inputs/outputs.

ii.
```python
def compute_valid_trial_mask(obs_intervals, go_times):
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)

# In process_session:
obs_intervals = get_ragged_row(
    f["units/obs_intervals"],
    obs_intervals_index,
    int(good_unit_indices[0]),  # Only first good unit
)
valid_trial_mask = compute_valid_trial_mask(obs_intervals=obs_intervals, go_times=go_times_all)
...
# After binning:
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
```

iii. The AI justified retaining stimulation, ignore, miss, and early-lick trials because they are required by the decoder task (photostimulation is an input, outcome/early_lick/choice are outputs). The obs_intervals filter was added to prevent all-zero neural trials, and the zero-spike removal was added as a defensive post-binning check.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged array of spike timestamps per unit) and `units/classification` (to identify good units). The anatomical labels come from `units/anno_name`.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
...
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
```

iii. The AI documented that `units/spike_times` contains the raw spike timestamps and `classification` is the NWB-native equivalent of the reference code's external QC classifier files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50-ms non-overlapping bins aligned to the go cue, spanning [-2.5, 1.5] seconds (80 bins). Spike counts are converted to firing rates by dividing by the bin width (0.05 s). The result is stored as float16.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))  # 80
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)

def bin_spikes_to_firing_rates(spike_times_flat, spike_times_index, good_unit_indices, trial_edges_abs):
    ...
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates
```

iii. The AI noted that the reference code uses 40 ms bins with 3.4 ms stride, but the instructions explicitly require 50 ms bins. The AI uses `searchsorted` for efficient vectorized binning per unit.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are retained. No additional firing rate thresholds, ISI violation filters, or other quality metrics are applied beyond what the classifier already captures.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

iii. The AI documented that the NWB `classification` field stores results of the region-specific logistic-regression classifiers trained from manual curation labels, as described in the spike sorting QC white paper.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, absolute bin edges are computed as `go_time + relative_edges`, where relative edges span [-2.5, 1.5] seconds. Spikes falling within each absolute bin are counted.

ii.
```python
go_times = go_times_all[valid_trial_mask]
...
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
firing_rates = bin_spikes_to_firing_rates(
    spike_times_flat=spike_times_flat,
    spike_times_index=spike_times_index,
    good_unit_indices=good_unit_indices,
    trial_edges_abs=trial_edges_abs,
)
```

iii. The AI documented that go cue alignment is consistent with both the reference code (which subtracts go cue time from all event timestamps) and the instructions (which specify "Go cue onset" alignment).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms (0.05 s) per bin, with 80 non-overlapping bins spanning 4 seconds [-2.5, 1.5]. No temporal rebinning is applied; spikes are binned directly from raw spike times into the final bin size.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))  # 80
```

iii. The AI explicitly noted this deviates from the reference code's 40 ms / 3.4 ms stride, but matches the instruction requirement of "50-ms-width bins."

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is NOT derived from any raw data variable. It is computed entirely from task structure constants: the canonical tone onset at -1.85 s relative to go cue (derived from 0.65 s sample epoch + 1.2 s delay), and the bin centers of the go-aligned time grid.

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
...
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The AI justified using the canonical timing because raw NWB sample-event streams contain replay-related extra events (from early licks triggering epoch replays) and are not reliable one-to-one trial markers.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. For each bin center, the time from tone onset is computed as `bin_center - (-1.85) = bin_center + 1.85`. This produces a continuously increasing time series that is identical for every trial, ranging from approximately -0.65 to 3.35 seconds.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The AI noted that the time-from-tone-onset is the same for all trials because the task structure is fixed (tone always occurs at the same time relative to go cue).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone-onset uses the same bin centers (`REL_CENTERS`) as the neural data, ensuring perfect temporal alignment. Both are defined on the same go-cue-centered time grid.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
# Used for both neural binning edges and input time computation
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. No separate alignment step is needed because both signals share the same temporal reference frame.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` (onset time relative to trial start), `intervals/trials/photostim_duration`, `intervals/trials/start_time`, and `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

iii. The AI documented that photostim onset/duration are stored as strings in the NWB trial table, with "N/A" for non-stimulated trials.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostimulation onset is converted from trial-start-relative coordinates to go-cue-relative coordinates by subtracting (go_time - start_time). A binary time series is created where bins are set to 1 if the bin center falls within [onset_rel_go, onset_rel_go + duration).

ii.
```python
def build_photostim_matrix(photostim_onset_str, photostim_duration_str, start_times, go_times):
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

iii. The AI documented that N/A values (non-stimulated trials) are parsed as NaN and result in all-zero photostim vectors, which is correct behavior.

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. Photostimulation uses the same go-cue-centered bin centers as the neural data. The conversion from trial-start coordinates to go-centered coordinates ensures temporal consistency.

ii.
```python
# Same REL_CENTERS used for both neural bin edges and photostim binary vector
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```

iii. The AI documented this alignment as consistent with the reference code, which also subtracts go cue time from stimulation times.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction` (left/right), `intervals/trials/outcome` (hit/miss/ignore), `acquisition/BehavioralEvents/left_lick_times/timestamps`, and `acquisition/BehavioralEvents/right_lick_times/timestamps`.

ii.
```python
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
```

iii. The AI documented that these variables collectively determine the animal's choice for each trial type.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. Choice is determined by trial outcome:
- **Hit trials**: Choice = instructed side (mouse licked correctly)
- **Miss trials**: Choice = opposite of instructed side (mouse licked wrong side)
- **Ignore trials**: First checks for post-go licks, then any licks in the trial window, then falls back to instructed side

The result is encoded as left=0, right=1, constant across all time bins.

ii.
```python
def build_choice_array(trial_instruction, outcome_code, start_times, go_times, stop_times,
                       left_lick_times, right_lick_times):
    choice = np.zeros(len(trial_instruction), dtype=np.int16)
    instructed = np.where(trial_instruction == "left", 0, 1).astype(np.int16)
    hit_mask = outcome_code == 2
    miss_mask = outcome_code == 1
    ignore_mask = outcome_code == 0
    choice[hit_mask] = instructed[hit_mask]
    choice[miss_mask] = 1 - instructed[miss_mask]
    ignore_trials = np.where(ignore_mask)[0]
    for trial_idx in ignore_trials:
        choice[trial_idx] = lick_choice_with_fallback(...)
    return choice
```

iii. The AI documented that most ignore trials (13,770/14,095) have no post-go lick, requiring a fallback. The AI uses instructed side as the ultimate fallback, which is a reasonable default for trials where the mouse didn't respond.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived directly from `intervals/trials/outcome`, which contains string values "hit", "miss", or "ignore".

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. The AI mapped outcome strings to integers matching the instruction specification: ignore=0, miss=1, hit=2.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The outcome string is directly mapped to an integer code with no further processing. The mapping is: ignore -> 0, miss -> 1, hit -> 2. The value is constant across all time bins for each trial.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
...
output_trial = np.vstack([
    ...
    np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
    ...
])
```

iii. No justification needed beyond matching the instruction encoding.

## 6-c. How is `output` *Distance to reward zone* aligned with the neural data?

i. This question does not apply to this dataset. There is no "Distance to reward zone" variable. The Outcome output is a per-trial constant that is broadcast across all time bins, so no temporal alignment is needed beyond the trial-level assignment.

ii. N/A

iii. N/A

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived directly from `intervals/trials/early_lick`, which contains string values "early" or "no early".

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. The mapping matches the instruction specification: no=0, yes=1.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The early lick string is directly mapped to an integer code with no further processing. The value is constant across all time bins for each trial.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
...
output_trial = np.vstack([
    ...
    np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
    ...
])
```

iii. No additional justification needed.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains columns (x, y, likelihood) with associated timestamps.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The AI documented that all 174 sessions have tongue tracking data from the side-view camera.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a multi-step cleaning pipeline:
1. **Velocity outlier detection**: Frame-to-frame speed is computed; frames with speed > mean + 5*std are flagged as outliers.
2. **Outlier interpolation**: Outlier x/y positions are linearly interpolated from neighboring good frames.
3. **Likelihood thresholding**: Frames with likelihood < 0.9 are treated as occluded; their y-values are replaced with the mean y of visible frames.

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9

def clean_tongue_tracking(x, y, likelihood):
    speed = np.zeros_like(x)
    if len(x) > 1:
        speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
    # Interpolate outliers
    frame_idx = np.arange(len(x), dtype=np.float64)
    keep_mask = ~outlier_mask
    if np.sum(keep_mask) >= 2:
        x[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], x[keep_mask])
        y[outlier_mask] = np.interp(frame_idx[outlier_mask], frame_idx[keep_mask], y[keep_mask])
    # Handle low-likelihood (occluded) frames
    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    y[~visible_mask] = mean_y
    return y, diagnostics
```

iii. The AI justified this cleaning as following the method paper's description: "Marker outliers were identified using a five-sigma frame-to-frame velocity rule and imputed from nearby frames" and "Tongue position was imputed to its mean when occluded."

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session percentiles are computed from all aligned tongue-y values in the session. The thresholds are the 40th and 60th percentiles. Values are classified as:
- 0: y < 40th percentile
- 1: 40th percentile <= y <= 60th percentile
- 2: y > 60th percentile

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. The AI documented replacing an initial `np.digitize` approach with explicit threshold logic to handle edge cases where q40 == q60. The resulting distribution is [0.082, 0.834, 0.085], indicating that most tongue y-values cluster near the mean (likely due to occlusion imputation), causing the 40th and 60th percentiles to be very close together.

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Cleaned tongue y-values are aligned to go-cue-centered bin centers using a nearest-preceding-frame (last-frame-carried-forward) approach via `searchsorted`.

ii.
```python
def align_tongue_y(timestamps, cleaned_y, go_times):
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The AI justified this as mirroring the reference code's alignment approach rather than using linear interpolation.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. The AI handles several types of missing/problematic data:
- **Missing photostimulation**: "N/A" strings are parsed as NaN and produce all-zero photostim vectors.
- **Low-likelihood tongue tracking**: Frames below 0.9 likelihood threshold have y replaced with mean visible y.
- **Velocity outliers in tracking**: Frames exceeding 5-sigma speed threshold are interpolated from neighbors.
- **Zero-spike trials**: Trials with no spikes across all good units and all bins are removed post-binning.
- **Out-of-recording trials**: Trials whose neural window falls outside obs_intervals are excluded.
- **Session with no good units**: The single zero-good-unit session is excluded during file discovery.

ii.
```python
# Missing photostim
def parse_optional_float_array(strings):
    out = np.full(strings.shape, np.nan, dtype=np.float64)
    for i, value in enumerate(strings):
        if value == "N/A":
            continue
        out[i] = float(value)
    return out

# Zero-spike trial removal
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
```

iii. The AI documented each missing-data handling decision in CONVERSION_NOTES.md and ran sanity checks to verify correct handling.

## 10-a. What are the most time-consuming steps of the code?

i. The most time-consuming step is spike binning (`bin_spikes_to_firing_rates`), which loops over all good units in a session and uses `searchsorted` to count spikes in each bin for each trial. The AI reports per-session timing and found binning takes the majority of processing time.

ii.
```python
def bin_spikes_to_firing_rates(...):
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
```

iii. The AI documented 0.6-0.8s per session for sample sessions, with full conversion taking 7.24 minutes for 173 sessions.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. Several loops could potentially be vectorized:
1. **Spike binning loop over units**: The outer loop over `good_unit_indices` could potentially be restructured, though the ragged nature of spike times per unit makes full vectorization difficult.
2. **Photostim matrix loop over valid trials**: The loop `for trial_idx in np.where(valid)[0]` builds the binary stim vector trial-by-trial.
3. **Ignore-trial choice loop**: The `for trial_idx in ignore_trials` loop in `build_choice_array` processes each ignore trial individually.
4. **Output trial assembly loop**: The `for trial_idx in range(n_trials)` loop constructs input/output arrays trial by trial.

ii.
```python
# Photostim loop (could be vectorized with broadcasting)
for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
    stim[trial_idx, mask] = 1.0

# Ignore-trial choice loop
for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)
```

iii. The AI acknowledged these as potential inefficiencies but noted the overall runtime (7.24 min) was within acceptable bounds.

## 10-c. What processing does the code repeat multiple times?

i. The photostim onset parsing is computed twice:
1. In `build_photostim_matrix()` during main processing
2. Again after session processing for diagnostic statistics (photostim_onsets_rel_go)

ii.
```python
# First time, in build_photostim_matrix:
onset_trial = parse_optional_float_array(photostim_onset_str)
...
# Second time, after session processing for diagnostics:
onset_trial = parse_optional_float_array(photostim_onset_str)
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
photostim_onsets_rel_go = onset_rel_go[np.isfinite(onset_rel_go)]
```

iii. This duplication exists to populate diagnostic data for plotting.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. Several processing steps produce data not used in the final pickle:
1. **Tongue x cleaning**: The `clean_tongue_tracking` function processes both x and y coordinates, but only y is used in the output.
2. **Diagnostic data**: Extensive diagnostic dictionaries are built per session (tracking previews, photostim onset distributions, etc.) that are only used for optional `--show-processing` plots but are computed regardless.
3. **Session-level statistics**: Various counts and timing statistics are computed and printed but not stored in the output.

ii.
```python
# Tongue x is cleaned but never used downstream
cleaned_tongue_y, tongue_clean_diag = clean_tongue_tracking(
    x=tongue_x,   # x is processed internally but only y is returned/used
    y=tongue_y,
    likelihood=tongue_likelihood,
)

# Diagnostics always computed even without --show-processing
diagnostics = {
    "tracking_preview_time": tongue_timestamps[:preview_n] - go_times[0],
    "tracking_preview_raw": tongue_y[:preview_n],
    ...
}
```

iii. The AI designed the cleaning function to take x for velocity computation (speed = sqrt(dx^2 + dy^2)) but only returns y. The diagnostics are always computed for completeness but add minor overhead.
