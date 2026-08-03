# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. Data are loaded from NWB files stored in a DANDI-style directory structure (`data/sub-<subject>/*.nwb`). The script discovers all NWB files by globbing `data/sub-*/*.nwb`, opens each with `h5py`, and reads trial tables, unit tables, behavioral events, and behavioral time series directly from HDF5 groups. Each session (one NWB file) is processed sequentially in `process_session()`.

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

iii. The AI chose to load NWB files directly with h5py rather than using PyNWB or higher-level wrappers, for "lower overhead and explicit access to ragged arrays" (CONVERSION_NOTES Step 6). This is a reasonable choice since the reference code loaded from author-exported MATLAB structures, and the NWB files are the raw source available in this context.

## 1-b. How are the data split into subjects (mice)?

i. Subjects are identified from the parent directory name of each NWB file (e.g., `sub-440956`). A unique list of subjects is built incrementally as sessions are processed, and each session is assigned a `subject_idx` into that list.

ii.
```python
subject_id = path.parent.name
# ...
if result.subject_id not in subject_to_idx:
    subject_to_idx[result.subject_id] = len(subjects)
    subjects.append(result.subject_id)
data["subject_idx"][session_idx] = subject_to_idx[result.subject_id]
```

iii. The AI noted that there are 28 subject folders matching the 28 mice reported in the papers (CONVERSION_NOTES Step 2, Step 9).

## 1-c. How are the data split into sessions?

i. Each NWB file corresponds to one session. Sessions are sorted by file path. The single session with zero good units (`sub-440958_ses-20190216T162508_behavior+ecephys+ogen.nwb`) is excluded, yielding 173 sessions matching the paper count.

ii.
```python
files = sorted(DATA_DIR.glob("sub-*/*.nwb"))
valid = []
for path in files:
    with h5py.File(path, "r") as f:
        good = decode_str_array(f["units/classification"]) == "good"
        if np.any(good):
            valid.append(path)
```

iii. CONVERSION_NOTES Step 4 documents that 174 raw NWB files exist, but one has zero good units. Excluding it yields 173, matching the paper's "173 behavioral sessions."

## 1-d. How are the data split into trials?

i. Trials are defined by the `intervals/trials` table in each NWB file. Go cue times are read from `acquisition/BehavioralEvents/go_start_times/timestamps`. A validity check ensures the number of go cues matches the number of trials. Trial start/stop times come from the trial table.

ii.
```python
trials = f["intervals/trials"]
n_trials_raw = len(trials["id"])
start_times_all = trials["start_time"][()].astype(np.float64)
stop_times_all = trials["stop_time"][()].astype(np.float64)
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
```

iii. The AI verified that go_start_times count matches trial count exactly for all sessions (CONVERSION_NOTES Step 2).

## 1-e. How are trials filtered based on quality controls?

i. Two trial filters are applied: (1) an `obs_intervals` coverage filter that keeps only trials where the full [-2.5, 1.5] s go-aligned window falls within recorded observation intervals of the first good unit, and (2) a post-binning filter that removes trials with all-zero firing rates across all good units. No filtering based on early lick, outcome type, photostimulation, auto-water, or free-water is applied.

ii.
```python
def compute_valid_trial_mask(obs_intervals, go_times):
    trial_start = go_times + REL_START_S
    trial_end = go_times + REL_END_S
    starts = obs_intervals[:, 0][None, :]
    ends = obs_intervals[:, 1][None, :]
    return np.any((trial_start[:, None] >= starts) & (trial_end[:, None] <= ends), axis=1)

# Post-binning zero-trial removal:
nonzero_trial_mask = np.any(firing_rates != 0, axis=(0, 2))
if not np.all(nonzero_trial_mask):
    firing_rates = firing_rates[:, nonzero_trial_mask, :]
    # ... filter all other arrays similarly
```

iii. The AI decided to retain stimulation, early-lick, ignore, and miss trials because they are required decoder inputs/outputs. The reference code's `get_regular_trial_mask` excludes these for specific analyses, but the decoder task requires them. The obs_intervals filter and zero-spike filter are documented in CONVERSION_NOTES Steps 5, 6, 9, and 10.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` (ragged spike time arrays), `units/spike_times_index` (ragged array index), and `units/classification` (to select "good" units).

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
spike_times_flat = f["units/spike_times"][()]
spike_times_index = f["units/spike_times_index"][()]
```

iii. The AI documented that NWB `classification == "good"` is the closest NWB-native analogue of the reference code's external `goodunits` classifier files (CONVERSION_NOTES Steps 1, 4, 5).

## 2-b. How is the `neural` data processed?

i. Spike times for each good unit are binned into 50 ms non-overlapping bins spanning [-2.5, 1.5] s relative to go cue (80 bins total). Spike counts per bin are divided by bin width (0.05 s) to yield firing rates in Hz. Results are stored as float16.

ii.
```python
def bin_spikes_to_firing_rates(spike_times_flat, spike_times_index, good_unit_indices, trial_edges_abs):
    # ...
    for i, unit_idx in enumerate(good_unit_indices):
        spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
        edge_idx = np.searchsorted(spikes, flat_edges, side="left").reshape(n_trials, n_edges)
        counts = np.diff(edge_idx, axis=1)
        firing_rates[i] = (counts / BIN_SIZE_S).astype(np.float16)
    return firing_rates
```

iii. The AI noted that the reference method paper uses 40 ms bins with 3.4 ms stride, but the decoder task explicitly requires 50 ms bins (CONVERSION_NOTES Steps 3, 5). The searchsorted approach is vectorized per unit across all trials simultaneously.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == "good"` are included. No additional filtering based on firing rate thresholds, ISI violations, or other quality metrics is applied beyond the NWB classification field.

ii.
```python
classification = decode_str_array(f["units/classification"])
good_mask = classification == "good"
good_unit_indices = np.flatnonzero(good_mask)
```

iii. The AI documented that the reference code relies on externally generated QC files (classifier-based good-unit labels) rather than ad hoc thresholds, and that NWB `classification == "good"` is the NWB-native equivalent (CONVERSION_NOTES Step 1, Step 10 Check 3).

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Neural data is aligned to the go cue onset. For each trial, bin edges are computed as `go_time + REL_EDGES` where REL_EDGES spans from -2.5 to 1.5 s in 50 ms steps.

ii.
```python
go_times_all = f["acquisition/BehavioralEvents/go_start_times/timestamps"][()].astype(np.float64)
trial_edges_abs = go_times[:, None] + REL_EDGES[None, :]
```

iii. The instructions specify alignment to go cue onset, and the reference code also uses go-cue-centered alignment throughout (CONVERSION_NOTES Steps 3, 4).

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. The temporal resolution is 50 ms (0.05 s), producing 80 bins over the 4 s window [-2.5, 1.5]. No rebinning is applied; spikes are binned directly into 50 ms bins from raw spike times. The reference code uses 40 ms bins with 3.4 ms stride, but the instructions explicitly require 50 ms bins.

ii.
```python
BIN_SIZE_S = 0.05
REL_START_S = -2.5
REL_END_S = 1.5
N_BINS = int(round((REL_END_S - REL_START_S) / BIN_SIZE_S))  # 80
```

iii. The AI documented this as a "required deviation from reference bins" (CONVERSION_NOTES Step 5, Key Decision 12).

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. This input is derived from the canonical task structure rather than from raw NWB event variables. The tone onset is defined as a fixed offset of -1.85 s relative to go cue, based on the task timing (sample epoch = 0.65 s + delay epoch = 1.2 s).

ii.
```python
TONE_ONSET_REL_GO_S = -1.85
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The AI reasoned that raw NWB sample-event streams contain replay-related extra events from early licks and are not reliable one-to-one trial markers. Using the canonical -1.85 s offset is "more consistent with the task definition" (CONVERSION_NOTES Step 5, Key Decision 5).

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The time from tone onset is computed as `bin_center - (-1.85)` = `bin_center + 1.85` for each of the 80 bin centers. This produces a continuous time-varying signal that is the same for every trial, ranging from approximately -0.6 to 3.3 seconds.

ii.
```python
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. The AI noted the range [-0.6, 3.3] matches expectations from the go-aligned window (CONVERSION_NOTES Step 7, Step 9).

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. Both use the same bin centers (`REL_CENTERS`), which are go-cue-aligned. The time-from-tone-onset vector shares the same 80 time points as the neural firing rates, so they are inherently aligned.

ii.
```python
REL_CENTERS = (REL_EDGES[:-1] + REL_EDGES[1:]) / 2.0
input_time = np.tile((REL_CENTERS - TONE_ONSET_REL_GO_S).astype(np.float16), (n_trials, 1))
```

iii. Alignment is implicit through the shared bin center array.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. Photostimulation is derived from `intervals/trials/photostim_onset` (onset time relative to trial start), `intervals/trials/photostim_duration`, `intervals/trials/start_time`, and go cue times from `acquisition/BehavioralEvents/go_start_times/timestamps`.

ii.
```python
photostim_onset_str = decode_str_array(trials["photostim_onset"])[valid_trial_mask]
photostim_duration_str = decode_str_array(trials["photostim_duration"])[valid_trial_mask]
```

iii. The AI documented that photostimulation onset/duration are stored as strings in the NWB trial table, with "N/A" for non-stimulation trials (CONVERSION_NOTES Steps 2, 4).

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. The photostim onset (relative to trial start) is converted to go-cue-relative coordinates by subtracting `(go_time - start_time)`. A binary 0/1 time series is constructed where bin centers falling within [onset_rel_go, onset_rel_go + duration) are marked as 1. Non-stimulation trials (N/A values) result in all-zero vectors.

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

iii. The AI verified this matches the reference code's approach of subtracting go cue time from stimulation times (CONVERSION_NOTES Step 4). The photostim timing check passed in sanity checks (Step 10).

## 4-c. How is the `input` *Photostimulation* aligned with the neural data?

i. The photostimulation binary vector uses the same `REL_CENTERS` (go-cue-aligned bin centers) as the neural data, ensuring alignment.

ii.
```python
mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & (REL_CENTERS < offset_rel_go[trial_idx])
```

iii. Alignment is implicit through the shared go-cue-relative bin centers.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is derived from `intervals/trials/trial_instruction` (left/right), `intervals/trials/outcome` (hit/miss/ignore), `acquisition/BehavioralEvents/left_lick_times/timestamps`, and `acquisition/BehavioralEvents/right_lick_times/timestamps`.

ii.
```python
trial_instruction = decode_str_array(trials["trial_instruction"])[valid_trial_mask]
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
left_lick_times = f["acquisition/BehavioralEvents/left_lick_times/timestamps"][()].astype(np.float64)
right_lick_times = f["acquisition/BehavioralEvents/right_lick_times/timestamps"][()].astype(np.float64)
```

iii. The AI documented the need for different choice derivation strategies depending on trial outcome (CONVERSION_NOTES Step 5, Key Decision 8).

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For hit trials, choice = instructed side (left=0, right=1). For miss trials, choice = opposite of instructed side. For ignore trials, the code first checks for post-go licks within [go_time, stop_time]; if none, checks for any lick within [start_time, stop_time]; if none, falls back to instructed side. The result is a per-trial scalar replicated across all 80 time bins.

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

iii. The AI documented that most ignore trials have no post-go lick (13,770/14,095), so the fallback to instructed side is used frequently (CONVERSION_NOTES Step 5).

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Outcome is derived from `intervals/trials/outcome`, which contains string values "hit", "miss", or "ignore".

ii.
```python
outcome_str = decode_str_array(trials["outcome"])[valid_trial_mask]
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
```

iii. Directly read from the NWB trial table.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String values are mapped to integers: ignore=0, miss=1, hit=2. The result is a per-trial scalar replicated across all 80 time bins.

ii.
```python
outcome_code = np.array([{"ignore": 0, "miss": 1, "hit": 2}[x] for x in outcome_str], dtype=np.int16)
# Per-trial replication:
np.full(N_BINS, outcome_code[trial_idx], dtype=np.int16)
```

iii. Matches the decoder task specification exactly (ignore=0, miss=1, hit=2).

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. Early lick is derived from `intervals/trials/early_lick`, which contains string values "early" or "no early".

ii.
```python
early_lick_str = decode_str_array(trials["early_lick"])[valid_trial_mask]
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
```

iii. Directly read from the NWB trial table.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. String values are mapped to integers: "no early"=0, "early"=1. The result is a per-trial scalar replicated across all 80 time bins.

ii.
```python
early_code = np.array([{"no early": 0, "early": 1}[x] for x in early_lick_str], dtype=np.int16)
np.full(N_BINS, early_code[trial_idx], dtype=np.int16)
```

iii. Matches the decoder task specification (no=0, yes=1).

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. Tongue y-position is derived from `acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking`, which contains columns (x, y, likelihood), and the associated `timestamps` array.

ii.
```python
tongue_ts = f["acquisition/BehavioralTimeSeries/Camera0_side_TongueTracking"]
tongue_data = tongue_ts["data"][()]
tongue_x = tongue_data[:, 0]
tongue_y = tongue_data[:, 1]
tongue_likelihood = tongue_data[:, 2]
tongue_timestamps = tongue_ts["timestamps"][()].astype(np.float64)
```

iii. The AI documented that all 174 sessions contain tongue tracking data at 300 Hz (CONVERSION_NOTES Step 2).

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Processing involves: (1) velocity-based outlier detection using 5-sigma threshold on frame-to-frame speed, with linear interpolation for outlier frames; (2) low-likelihood frame imputation where tongue y is set to the mean of visible frames (likelihood >= 0.9); (3) alignment to go-cue-relative bin centers using last-frame-carried-forward (searchsorted).

ii.
```python
def clean_tongue_tracking(x, y, likelihood):
    speed = np.zeros_like(x)
    if len(x) > 1:
        speed[1:] = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    speed_threshold = float(np.nanmean(speed) + 5.0 * np.nanstd(speed))
    outlier_mask = ~np.isfinite(x) | ~np.isfinite(y) | (speed > speed_threshold)
    # ... interpolate outliers ...
    visible_mask = np.isfinite(likelihood) & (likelihood >= TONGUE_LIKELIHOOD_THRESHOLD)
    mean_y = float(np.nanmean(y[visible_mask])) if np.any(visible_mask) else float(np.nanmean(y))
    occluded_mask = ~visible_mask
    y[occluded_mask] = mean_y
    return y, diagnostics
```

iii. The AI noted this mirrors the reference method paper's preprocessing: "Marker outliers were identified using a five-sigma frame-to-frame velocity rule and imputed from nearby frames. Tongue position was imputed to its mean when occluded before the response epoch" (CONVERSION_NOTES Step 3). The reference paper imputes the mean only before the response epoch, but the AI imputes it globally for all low-likelihood frames.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Per-session 40th and 60th percentiles are computed over all aligned tongue y values in the session. Values < 40th percentile map to 0, values in [40th, 60th] percentile map to 1, values > 60th percentile map to 2.

ii.
```python
q40, q60 = np.percentile(aligned_tongue_y.reshape(-1), [40, 60])
tongue_disc = np.ones(aligned_tongue_y.shape, dtype=np.int16)
tongue_disc[aligned_tongue_y < q40] = 0
tongue_disc[aligned_tongue_y > q60] = 2
```

iii. Matches the decoder task specification. The AI initially used `np.digitize` but switched to explicit threshold logic to handle edge cases where q40 == q60 (CONVERSION_NOTES Step 6).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. Cleaned tongue y values are aligned to the same go-cue-relative bin centers as neural data using `searchsorted` (last-frame-carried-forward). For each bin center, the last tongue tracking frame before that time is used.

ii.
```python
def align_tongue_y(timestamps, cleaned_y, go_times):
    abs_centers = go_times[:, None] + REL_CENTERS[None, :]
    idx = np.searchsorted(timestamps, abs_centers, side="right") - 1
    idx = np.clip(idx, 0, len(timestamps) - 1)
    return cleaned_y[idx]
```

iii. The AI described this as "last-frame-carried-forward alignment to bin centers" which mirrors the reference alignment approach (CONVERSION_NOTES Step 5).

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several strategies are used: (1) Photostim "N/A" values are parsed as NaN and result in all-zero photostim vectors. (2) Trials outside recording observation intervals are filtered out. (3) Trials with all-zero neural activity across all good units are removed as a defensive guard. (4) Tongue tracking outliers (velocity > 5-sigma or non-finite) are interpolated from neighboring frames. (5) Low-likelihood tongue frames are imputed to the session mean of visible frames. (6) If no lick is detected in ignore trials, the instructed side is used as fallback for choice. (7) The single session with zero good units is excluded entirely.

ii. Key code snippets shown above in relevant sections.

iii. The AI documented each edge case handling in CONVERSION_NOTES Steps 5, 6, and 10.

## 10-a. What are the most time-consuming steps of the code?

i. Spike binning is the most time-consuming step, accounting for the majority of per-session processing time. The conversion logs show binning times of 0.2-2.9 s per session, with total session processing times of 0.5-9.6 s. Full conversion completed in 7.24 minutes for 173 sessions.

ii.
```python
t_neural = time.perf_counter()
firing_rates = bin_spikes_to_firing_rates(...)
neural_time_s = time.perf_counter() - t_neural
```

iii. The AI tracked timing per session and noted the full conversion was well under the 15-minute optimization threshold (CONVERSION_NOTES Step 7, Step 9).

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The spike binning loop iterates over units (`for i, unit_idx in enumerate(good_unit_indices)`), which could potentially be vectorized but would require complex ragged-array handling. The `build_choice_array` function loops over ignore trials for the lick-fallback logic. The `build_photostim_matrix` loops over stimulated trials. The `decode_str_array` function loops over elements to decode bytes.

ii.
```python
for i, unit_idx in enumerate(good_unit_indices):
    spikes = spike_times_flat[start_index[unit_idx]: spike_times_index[unit_idx]]
    # ...

for trial_idx in ignore_trials:
    choice[trial_idx] = lick_choice_with_fallback(...)

for trial_idx in np.where(valid)[0]:
    mask = (REL_CENTERS >= onset_rel_go[trial_idx]) & ...
```

iii. The unit loop in spike binning is the most impactful; however, since spike times are ragged arrays (different lengths per unit), full vectorization is non-trivial. The ignore-trial and photostim loops operate on small subsets and are not bottlenecks.

## 10-c. What processing does the code repeat multiple times?

i. The `parse_optional_float_array` for photostim onset is called twice: once in `build_photostim_matrix()` during the main processing, and once again after the nonzero trial filter when computing diagnostics for plotting. The `REL_CENTERS - TONE_ONSET_REL_GO_S` computation is tiled per trial but could be computed once and broadcast.

ii.
```python
# In build_photostim_matrix:
onset_trial = parse_optional_float_array(photostim_onset_str)
# Later in diagnostics:
onset_trial = parse_optional_float_array(photostim_onset_str)
```

iii. This duplication has minimal performance impact since it operates on small arrays (one value per trial).

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The code cleans tongue x-coordinates (velocity outlier interpolation) even though only tongue y is used for the output. The full tongue tracking data (x, y, likelihood) is loaded and processed through the cleaning pipeline, but x is only used for velocity computation (outlier detection) and then discarded. The tongue x-cleaning (interpolation) is unnecessary since x values are never used in the final output. Additionally, the code stores extensive diagnostics dictionaries per session that are only used for plotting and are discarded before the final pickle.

ii.
```python
def clean_tongue_tracking(x, y, likelihood):
    x = np.asarray(x, dtype=np.float64).copy()
    y = np.asarray(y, dtype=np.float64).copy()
    # ... x is cleaned/interpolated but only y is returned
    return y, diagnostics
```

iii. The tongue x processing is needed for the velocity-based outlier detection, but the interpolation of x values themselves is unnecessary since they are not used downstream.
