# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI discovers all NWB files by globbing `data/sub-*/*.nwb`, sorts them, and pre-filters by opening each with h5py to check if any unit has `classification == "good"`. Valid sessions are then processed one at a time via `process_session()`. The AI uses `h5py` directly rather than the higher-level `pynwb` library.

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
for i, path in enumerate(files, start=1):
    result = process_session(path)
    results.append(result)
```

iii. The AI's CONVERSION_NOTES.md documents that NWB files are the raw session-level source, and that h5py was chosen over higher-level wrappers for "lower overhead and explicit access to ragged arrays." The pre-filtering step excludes the one session with zero good units before any processing begins.

## 1-b. How are the data split into subjects?

i. The AI uses the parent directory name (e.g., `sub-440956`) as the subject identifier, rather than reading the subject metadata from inside the NWB file.

ii.
```python
subject_id = path.parent.name
```

```python
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The AI documents using "exact subject IDs (`sub-xxxxx`)" and mapping each session to its subject index. The subject IDs are derived from the folder structure rather than `nwb.subject.subject_id`.

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. The AI uses the file stem (filename without extension) as the session ID. Sessions are processed in sorted file-path order.

ii.
```python
session_id = path.stem
```

iii. The AI notes that "each session is a single NWB file inside its subject folder" in CONVERSION_NOTES.md Step 2.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`intervals/trials`). The AI verifies that the number of go cue events matches the trial count.

ii.
```python
trials = f["intervals/trials"]
n_trials_raw = len(trials["id"])
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
if len(go_times_all) != n_trials_raw:
    raise ValueError(...)
```

iii. The AI's CONVERSION_NOTES.md notes that `go_start_times` matches trial count exactly in all sessions.

## 1-e. How are trials filtered based on quality controls?

i. The AI applies two filters: (1) an `obs_intervals`-based coverage check that verifies the entire go-aligned window `[-2.5, 1.5)` falls within the recording intervals of the first good unit, and (2) a post-binning removal of trials where all good units have zero spikes. Notably, the AI does **not** filter out `free_water` trials.

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

iii. The AI's CONVERSION_NOTES.md documents including stimulation, early-lick, ignore, and miss trials because they are required decoder inputs/outputs. The zero-spike removal is documented as a "defensive guard against edge-case recording coverage mismatches."

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike times) and `units/spike_times_index` (ragged index). Only units with `classification == "good"` are used.

ii.
```python
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
```

iii. The AI documents using spike times as the only neural representation in the NWB files.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms go-aligned bins using `np.searchsorted`, then divided by bin width to get firing rates in Hz. The AI stores firing rates as `float16` rather than `float32`.

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

iii. The AI notes using "Vectorized spike binning / compact dtypes" and "search-sorted spike binning on per-unit spike vectors."

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `units/classification == "good"` are kept. Sessions with zero good units are excluded during the pre-filtering step.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
if len(good_unit_indices) == 0:
    raise ValueError(f"Session {session_id} has no good units")
```

iii. The AI documents that this "matches the paper session count and avoids the single raw session with zero analyzable units."

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Trial bin edges are computed as go cue time + relative edges, so spikes are binned relative to the go cue onset.

ii.
```python
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The AI documents go-cue-centered alignment consistent with the reference code and papers.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50 ms non-overlapping bins spanning [-2.5, 1.5) relative to go cue, giving 80 time bins per trial. No rebinning is applied — spikes are binned directly from spike times.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))
REL_EDGES = np.linspace(REL_START_S, REL_END_S, N_BINS + 1, dtype=np.float64)
```

iii. The AI documents this as an intentional deviation from the reference 40ms/3.4ms preprocessing, as required by the task instructions.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. The AI does **not** use any raw data variable for tone onset. Instead, it uses a fixed canonical offset of -1.85 s relative to the go cue, derived from the task structure (0.65s sample epoch + 1.2s delay).

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
```

iii. The AI's CONVERSION_NOTES.md Step 5 states: "Raw NWB sample-event streams contain replay-related extra events and are not reliable one-to-one trial markers; using the canonical -1.85 s tone onset is more consistent with the task definition."

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_center - (-1.85)` for every bin. Since the canonical offset is fixed, this produces the **same** time-from-tone vector for every trial.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. Because the tone onset is fixed at -1.85s relative to go for all trials, the input is identical across trials. This ignores the variation due to early-lick replays of the sample epoch.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The time-from-tone values are computed at the same bin centers used for neural data, so they are inherently aligned.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. Same bin grid as neural data.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, along with `start_time` and go cue times for coordinate conversion.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. Photostim onset (relative to trial start) is converted to go-relative coordinates. A binary 0/1 time series is created where bin centers falling within [onset, offset) are set to 1.

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

iii. The AI documents converting photostim from trial-start coordinates to go-centered coordinates.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Photostim onset/offset are expressed relative to the go cue, matching the neural bin centers. The binary mask is applied at the same REL_CENTERS used for neural binning.

ii. Same as 4-b code.

iii. The AI's metadata documents: "Binary per-bin input from trial-table photostim onset/duration converted to go coordinates."

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `trial_instruction`, `outcome`, and additionally `left_lick_times` and `right_lick_times` from BehavioralEvents. The AI uses actual lick event times for ignore trials, unlike the reference which assigns a "no lick" category.

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

iii. The AI's CONVERSION_NOTES.md Step 5 documents: "for ignore, use first post-go lick side if present, else first lick side anywhere in trial if present, else fall back to instructed side."

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice = instructed side. For miss trials, choice = opposite of instructed side. For ignore trials, the AI uses a multi-level lick fallback: first checks post-go licks, then any lick in trial, then falls back to instructed side. The output has only 2 values (left=0, right=1) — there is **no "no lick"** category.

ii.
```python
def build_choice_array(...):
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

```python
"output_values": [
    ["left", "right"],       # only 2 classes, no "no lick"
    ...
]
```

iii. The AI documents this as handling the "ignore-trial choice edge case" where most ignore trials have no post-go lick.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome comes directly from the `outcome` column of the trials table, which stores `'ignore'`, `'miss'`, and `'hit'`.

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. Same approach as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. The three strings are mapped to codes: ignore=0, miss=1, hit=2. The value is repeated across all 80 time bins.

ii.
```python
output_trial = np.vstack([
    ...
    np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16),
    ...
])
```

iii. Same mapping as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which stores `'no early'` and `'early'`.

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. Same approach as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. The two strings are mapped to codes: no early=0, early=1. The value is repeated across all 80 time bins.

ii.
```python
output_trial = np.vstack([
    ...
    np.full(N_BINS, early_code[trial_idx], dtype=np.int16),
    ...
])
```

iii. Same mapping as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains `(x, y, likelihood)` columns with corresponding timestamps.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. The AI applies a multi-step cleaning pipeline: (1) 5-sigma velocity outlier detection and interpolation of outlier frames, (2) frames with likelihood < 0.9 are set to the mean visible y-value (not NaN), (3) the cleaned y values are aligned to bin centers using last-frame-carried-forward (searchsorted). This differs significantly from the reference, which simply sets low-likelihood frames to NaN and averages within bins.

ii.
```python
def clean_tongue_tracking(x, y, likelihood):
    speed = np.zeros_like(x)
    if len(x) > 1:
        speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
    ...
    x[outlier_mask] = np.interp(...)
    y[outlier_mask] = np.interp(...)
    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    occluded_mask = ~visible_mask
    y[occluded_mask] = mean_y
    return y, diagnostics
```

```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9  # vs reference 0.5
```

iii. The AI's CONVERSION_NOTES.md documents following method-paper-inspired velocity-outlier interpolation and setting low-likelihood frames to mean visible y. The likelihood threshold of 0.9 is notably higher than the reference's 0.5.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. The AI computes per-session 40th and 60th percentiles from **all aligned tongue y values** (flattened across all trials and time bins) and discretizes into 3 classes: 0 (< q40), 1 (q40 to q60), 2 (> q60). There is **no "not visible" class** (class 3).

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

```python
"output_values": [
    ...
    ["lt_40pct", "40_to_60pct", "gt_60pct"],  # only 3 classes, no "not visible"
]
```

iii. The AI documents using "per-session 40th/60th percentile discretization" but does not include a "not visible" class because all frames have been imputed (low-likelihood frames set to mean y).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. The AI uses a last-frame-carried-forward approach: for each bin center, it finds the last camera frame at or before that time using `searchsorted`, and uses the cleaned y value from that frame.

ii.
```python
def align_tongue_y(timestamps, cleaned_y, go_times):
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The AI documents this as "last-frame-carried-forward alignment to bin centers."

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Three cases are handled: (1) Sessions with zero good units are excluded during pre-filtering. (2) Trials outside obs_intervals or with all-zero spikes are removed. (3) Tongue tracking: velocity outliers are interpolated, and low-likelihood frames are imputed to mean visible y. Unlike the reference, `free_water` trials are not explicitly filtered. Unlike the reference, there is no "not visible" class for tongue — all frames are imputed.

ii.
```python
# Pre-filtering
good = decode_str_array(f["units/classification"]) == "good"
if np.any(good):
    valid.append(path)

# Zero-spike removal
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))

# Tongue imputation
y[occluded_mask] = mean_y
```

iii. The AI's CONVERSION_NOTES.md documents these choices across Steps 5, 6, and 10.

## 10-a. What are the most time-consuming steps of the code?

i. According to the AI's timing output, the full conversion took ~7.24 minutes for 173 sessions. Per-session, the spike binning step is timed separately (reported in diagnostics). NWB file I/O and spike binning are the dominant costs.

ii.
```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
neural_time_s = time.perf_counter() - t_neural
```

iii. The AI's CONVERSION_NOTES.md notes a per-session time of ~0.7s and total of ~7.24 minutes.

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop iterates over each good unit. The per-trial loop for photostim matrix construction iterates over stimulated trials. The per-trial loop for building input/output arrays could be vectorized.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    ...

for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & ...
    stim[trial_idx, mask] = 1.0

for trial_idx in range(n_trials):
    input_trial = np.vstack([input_time[trial_idx], input_stim[trial_idx]]).astype(np.float16)
    ...
```

iii. The AI notes using "search-sorted spike binning on per-unit spike vectors" as a speedup, but the per-unit loop remains because spike times are ragged.

## 10-c. What processing does the code repeat multiple times?

i. The AI opens each NWB file twice: once during `get_nwb_files()` to check for good units, and again during `process_session()`. The `parse_optional_float_array` for photostim onset is called once for the photostim matrix and again later for diagnostics. The `go_minus_start` computation is also repeated.

ii.
```python
# First open in get_nwb_files:
with h5py.File(path, "r") as f:
    good = decode_str_array(f["units/classification"]) == "good"

# Second open in process_session:
with h5py.File(path, "r") as f:
    classification = decode_str_array(f["units/classification"])
```

```python
# Repeated photostim parsing for diagnostics
onset_trial = parse_optional_float_array(photostim_onset_str)
go_minus_start = go_times - start_times
onset_rel_go = onset_trial - go_minus_start
```

iii. The duplicate file opening is noted as a design choice for cleaner pre-filtering logic.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI computes velocity-based outlier detection and interpolation for tongue tracking, which adds complexity. The `clean_tongue_tracking` function computes `x` outlier interpolation even though only `y` is used downstream. The diagnostics dictionary collects extensive metadata that is not saved to the output pickle. The `tongue_x` cleaning is fully wasted.

ii.
```python
def clean_tongue_tracking(x, y, likelihood):
    ...
    x[outlier_mask] = np.interp(...)  # x is cleaned but never used in output
    y[outlier_mask] = np.interp(...)
```

iii. The AI documents the cleaning as "method-paper-inspired" but the x-coordinate processing is discarded.
