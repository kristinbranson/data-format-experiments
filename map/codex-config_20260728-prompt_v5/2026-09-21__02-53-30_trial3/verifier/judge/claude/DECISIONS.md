# Decisions

## 1-a. How are **all the data** for all subjects, sessions, and trials loaded in?

i. The AI loads all NWB files from `/app/data/sub-*/*.nwb` using `h5py` (not `pynwb`). It iterates over sorted glob results, processing each file with `process_session()`. One hardcoded session (`EXCLUDED_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen"`) is excluded before iteration begins in `list_valid_session_paths()`.

ii.
```python
def list_valid_session_paths() -> list[str]:
    all_paths = sorted(glob.glob("/app/data/sub-*/*.nwb"))
    valid_paths = []
    for path in all_paths:
        session_name = session_name_from_path(path)
        if session_name == EXCLUDED_SESSION:
            continue
        valid_paths.append(path)
    return valid_paths
```

```python
with h5py.File(path, "r") as h5file:
    subject_id = get_subject_id(h5file, path)
    classification = decode_string_array(h5file["units"]["classification"])
    ...
```

iii. The AI uses `h5py` for efficiency instead of `pynwb`. The excluded session was identified as having zero classifier-good units (all `classification == nan`), matching the paper's 173-session set. The AI documents this decision in CONVERSION_NOTES.md Step 4.

## 1-b. How are the data split into subjects?

i. Subject ID is read from `h5file["general"]["subject"]["subject_id"]` for each session. Unique subjects are collected in insertion order during assembly, and each session is mapped to its subject index.

ii.
```python
def get_subject_id(h5file: h5py.File, path: str) -> str:
    try:
        subject = h5file["general"]["subject"]["subject_id"][()]
        if isinstance(subject, bytes):
            return subject.decode("utf-8")
        return str(subject)
    except Exception:
        return os.path.basename(os.path.dirname(path)).replace("sub-", "")
```

```python
subject = session["subject_id"]
if subject not in subject_to_idx:
    subject_to_idx[subject] = len(subjects)
    subjects.append(subject)
subject_idx[session_idx] = subject_to_idx[subject]
```

iii. The AI extracts subject_id from the NWB subject field, with a fallback to parsing the directory name. This yields numeric subject IDs (e.g., '440956'). The subjects list is ordered by first appearance rather than sorted.

## 1-c. How are the data split into sessions?

i. One NWB file corresponds to one session. The session name is derived from the filename (minus `.nwb` extension). Sessions are processed in sorted file order.

ii.
```python
def session_name_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in (".nwb",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base
```

iii. The NWB dataset stores one session per file, so no splitting logic is needed. The AI correctly identifies this structure.

## 1-d. How are the data split into trials?

i. Trials come from the NWB trials table (`h5file["intervals"]["trials"]`). The number of trials is taken as the minimum length across all trial-level arrays (start_time, stop_time, auto_water, free_water, early_lick, outcome, trial_instruction, photostim_onset, photostim_duration, go_times).

ii.
```python
ntrials = min(
    len(trial_start_times),
    len(trial_stop_times),
    len(auto_water),
    len(free_water),
    len(early_lick),
    len(outcome),
    len(trial_instruction),
    len(photostim_onset_trial_rel),
    len(photostim_duration),
    len(go_times_abs),
)
```

iii. The AI takes a defensive approach by using the minimum length of all trial arrays, though in practice these arrays should all have the same length in well-formed NWB files.

## 1-e. How are trials filtered based on quality controls?

i. Trials are filtered based on multiple criteria: (1) `auto_water` trials excluded, (2) `free_water` trials excluded, (3) trials with missing tone onset excluded, (4) trials without common neural coverage across all good units (via `is_good_trials`), (5) trials whose decoder window falls outside the spike time span, and (6) trials with all-zero neural activity after binning. A session is dropped if fewer than 2 trials survive.

ii.
```python
keep_trials = (
    (~auto_water)
    & (~free_water)
    & np.isfinite(tone_onsets_abs)
    & neural_coverage_mask
    & spike_support_mask
)
```

```python
nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. The AI documents in CONVERSION_NOTES.md that auto_water and free_water are excluded because "they are not requested outputs, they are explicitly excluded in the reference analyses, and they can confound choice/outcome interpretation." The neural coverage checks were added after discovering all-zero neural trials during sample validation.

## 2-a. What variables in the raw data is the `neural` data derived from?

i. Neural data is derived from `units/spike_times` and `units/spike_times_index` for units with `classification == 'good'`. Go cue times from `BehavioralEvents/go_start_times` define the alignment.

ii.
```python
spike_times_all = np.asarray(h5file["units"]["spike_times"][()], dtype=np.float64)
spike_times_index = np.asarray(h5file["units"]["spike_times_index"][()], dtype=np.int64)
```

iii. Same source variables as the reference solution.

## 2-b. How is the `neural` data processed?

i. Spike times are binned into 50ms non-overlapping bins spanning -2.5s to +1.5s relative to go cue using `np.searchsorted`. Counts are divided by bin width (0.05s) to get firing rates in Hz. Rates are stored as `float16`.

ii.
```python
def bin_good_unit_spikes(...) -> list[np.ndarray]:
    neural_stack = np.empty((n_units, n_trials, N_BINS), dtype=np.float16)
    bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
    for out_idx, unit_idx in enumerate(good_unit_indices):
        ...
        insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
        counts = np.diff(insertion_idx, axis=1)
        neural_stack[out_idx] = (counts.astype(np.float32) / BIN_SIZE_S).astype(np.float16)
    return [neural_stack[:, trial_idx, :].copy() for trial_idx in range(n_trials)]
```

iii. The AI uses `float16` to reduce memory and pickle size. No smoothing or normalization is applied.

## 2-c. How is the `neural` data filtered based on quality controls?

i. Only units with `classification == 'good'` are retained. Sessions with zero such units are dropped. The AI hardcodes the exclusion of one known problematic session.

ii.
```python
classification = decode_string_array(h5file["units"]["classification"])
good_unit_indices = np.flatnonzero(classification == "good")
if len(good_unit_indices) == 0:
    return None
```

```python
EXCLUDED_SESSION = "sub-440958_ses-20190216T162508_behavior+ecephys+ogen"
```

iii. The AI correctly identifies `classification` (not `unit_quality`) as the field matching the paper's classifier-based QC. The hardcoded session exclusion is redundant with the zero-good-units check but makes the intent explicit.

## 2-d. How is the per-trial `neural` data aligned to the event described in the `instructions`?

i. Bin edges are computed as go-cue-absolute times by adding relative bin edges to each trial's go cue time. Spikes are binned against these absolute edges using `searchsorted`.

ii.
```python
bin_edges_abs = go_times_abs[:, None] + BIN_EDGES_REL[None, :]
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
    counts = np.diff(insertion_idx, axis=1)
```

iii. All NWB times share one global clock, so adding relative offsets to the go cue time gives absolute bin edges directly.

## 2-e. What is the temporal resolution (time bin size) of the converted data? Is any temporal rebinning applied?

i. 50ms non-overlapping bins, 80 bins total from -2.5s to +1.5s relative to go cue. No rebinning is applied; spikes are binned directly from spike times.

ii.
```python
WINDOW_START_S = -2.5
WINDOW_END_S = 1.5
BIN_SIZE_S = 0.05
N_BINS = int(round((WINDOW_END_S - WINDOW_START_S) / BIN_SIZE_S))
BIN_EDGES_REL = WINDOW_START_S + np.arange(N_BINS + 1, dtype=np.float64) * BIN_SIZE_S
```

iii. Matches the instruction requirements exactly.

## 3-a. What variables in the raw data is `input` *Time from tone onset in seconds* derived from?

i. From `sample_start_times` (absolute timestamps of tone/sample events), `trial_start_times`, and `go_start_times`. The tone for each trial is the last `sample_start` event between the trial's start time and its go cue.

ii.
```python
def build_tone_onsets_abs(trial_start_times, go_times_abs, sample_start_times_abs):
    start_idx = np.searchsorted(sample_start_times_abs, trial_start_times, side="left")
    end_idx = np.searchsorted(sample_start_times_abs, go_times_abs, side="left")
    tone_onsets_abs = np.full(go_times_abs.shape, np.nan, dtype=np.float64)
    for trial_idx in range(len(go_times_abs)):
        if end_idx[trial_idx] > start_idx[trial_idx]:
            tone_onsets_abs[trial_idx] = sample_start_times_abs[end_idx[trial_idx] - 1]
    return tone_onsets_abs
```

iii. The AI constrains the search to events between trial start and go cue, and returns NaN if no sample event is found. This handles early-lick replay correctly by taking the last sample event.

## 3-b. What processing is involved in computing `input` *Time from tone onset in seconds*?

i. The tone onset is expressed relative to the go cue (`tone_rel_s = tone_onset_abs - go_abs`), then subtracted from bin centers to get time from tone onset at each bin.

ii.
```python
tone_rel_s = tone_onsets_abs[kept_trial_indices] - go_keep
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
```

iii. This is algebraically equivalent to the reference's `CENTERS + (go - tone)`. Since `tone_rel_s = tone - go`, we get `BIN_CENTERS - (tone - go) = BIN_CENTERS + (go - tone)`.

## 3-c. How is the `input` *Time from tone onset in seconds* aligned with the neural data?

i. The same bin center grid (relative to go cue) is used for both neural binning and the time-from-tone computation, ensuring alignment.

ii.
```python
BIN_CENTERS_REL = BIN_EDGES_REL[:-1] + BIN_SIZE_S / 2.0
...
time_from_tone = BIN_CENTERS_REL[None, :] - tone_rel_s[:, None]
```

iii. Alignment is inherent in using the same temporal grid.

## 4-a. What variables in the raw data is `input` *Photostimulation* derived from?

i. From `photostim_onset` and `photostim_duration` in the trials table, plus `start_time` and go cue times for converting to go-relative timing.

ii.
```python
photostim_onset_trial_rel = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_onset"])
photostim_duration = decode_optional_float_array(h5file["intervals"]["trials"]["photostim_duration"])
...
stim_on_rel_s[valid_stim] = (
    trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim]
)
stim_off_rel_s = stim_on_rel_s + photostim_duration_keep
```

iii. Same source variables as the reference.

## 4-b. What processing is involved in computing `input` *Photostimulation*?

i. A bin is marked as 1 if it overlaps the stimulation period (bin_start < stim_off AND bin_end > stim_on), and 0 otherwise. This is an overlap-based criterion rather than a bin-center criterion.

ii.
```python
def build_photostim_binary(stim_on_rel_s, stim_off_rel_s):
    bin_starts = BIN_EDGES_REL[:-1][None, :]
    bin_ends = BIN_EDGES_REL[1:][None, :]
    photostim = np.zeros((len(stim_on_rel_s), N_BINS), dtype=np.float32)
    valid = np.isfinite(stim_on_rel_s) & np.isfinite(stim_off_rel_s)
    if np.any(valid):
        on = stim_on_rel_s[valid][:, None]
        off = stim_off_rel_s[valid][:, None]
        photostim[valid] = ((bin_starts < off) & (bin_ends > on)).astype(np.float32)
    return photostim
```

iii. The AI uses an overlap criterion (any part of the bin overlaps with stimulation), while the reference uses a center-in-range criterion. This can differ at bin boundaries.

## 4-c. How is `input` *Photostimulation* aligned with the neural data?

i. Stimulation onset/offset are expressed relative to the go cue, and compared against the bin edges which are also relative to the go cue.

ii.
```python
stim_on_rel_s[valid_stim] = (
    trial_start_keep[valid_stim] + photostim_onset_keep[valid_stim] - go_keep[valid_stim]
)
```

iii. Same alignment approach as the reference.

## 5-a. What variables in the raw data is `output` *Lick direction choice* derived from?

i. Choice is reconstructed from the actual lick events: `left_lick_times` and `right_lick_times` from `BehavioralEvents`. The first lick within a 1.5s response window after the go cue determines the choice direction.

ii.
```python
left_lick_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["left_lick_times"]["timestamps"][()], ...)
right_lick_times_abs = np.asarray(
    h5file["acquisition"]["BehavioralEvents"]["right_lick_times"]["timestamps"][()], ...)
...
choice_labels = reconstruct_choice_labels(go_keep, left_lick_times_abs, right_lick_times_abs)
```

iii. The AI chose to reconstruct choice from actual lick events rather than inferring from `trial_instruction` + `outcome`. The AI's CONVERSION_NOTES.md states this "uses actual behavior instead of inferred correctness" and reports >99.6% agreement with instruction/outcome labels.

## 5-b. What processing is involved in computing `output` *Lick direction choice*?

i. For each trial, the first left and right lick times within [go, go+1.5s) are found. The earliest lick determines the direction (left=0, right=1). If neither side has a lick in the window, it's coded as "no lick" (2). The value is constant across all time bins.

ii.
```python
def reconstruct_choice_labels(go_times_abs, left_lick_times_abs, right_lick_times_abs):
    labels = np.empty(len(go_times_abs), dtype=object)
    for trial_idx, go_time in enumerate(go_times_abs):
        window_end = go_time + RESPONSE_WINDOW_S
        left_idx = np.searchsorted(left_lick_times_abs, go_time, side="left")
        right_idx = np.searchsorted(right_lick_times_abs, go_time, side="left")
        left_time = np.inf
        right_time = np.inf
        if left_idx < len(left_lick_times_abs) and left_lick_times_abs[left_idx] < window_end:
            left_time = left_lick_times_abs[left_idx]
        if right_idx < len(right_lick_times_abs) and right_lick_times_abs[right_idx] < window_end:
            right_time = right_lick_times_abs[right_idx]
        if left_time == np.inf and right_time == np.inf:
            labels[trial_idx] = "no lick"
        elif left_time <= right_time:
            labels[trial_idx] = "left"
        else:
            labels[trial_idx] = "right"
    return labels
```

iii. The AI's approach directly measures the behavioral response rather than inferring it from trial metadata. This gives the same result in the vast majority of cases.

## 6-a. What variables in the raw data is `output` *Outcome* derived from?

i. Directly from the `outcome` column of the trials table, which contains strings 'ignore', 'miss', 'hit'.

ii.
```python
outcome = decode_string_array(h5file["intervals"]["trials"]["outcome"])
```

iii. Same source as the reference.

## 6-b. What processing is involved in computing `output` *Outcome*?

i. String labels are mapped to integers: ignore=0, miss=1, hit=2. The value is constant across all time bins per trial.

ii.
```python
OUTCOME_TO_INT = {"ignore": 0, "miss": 1, "hit": 2}
trial_outcome_int = np.asarray([OUTCOME_TO_INT[x] for x in outcome_keep], dtype=np.int8)
outputs_2d[:, 1, :] = trial_outcome_int[:, None]
```

iii. Same mapping as the reference.

## 7-a. What variables in the raw data is `output` *Early lick* derived from?

i. From the `early_lick` column of the trials table, which contains 'early' and 'no early'.

ii.
```python
early_lick = decode_string_array(h5file["intervals"]["trials"]["early_lick"])
```

iii. Same source as the reference.

## 7-b. What processing is involved in computing `output` *Early lick*?

i. Mapped to integers: 'no early'=0, 'early'=1. Constant across time bins.

ii.
```python
EARLY_TO_INT = {"no early": 0, "early": 1}
trial_early_int = np.asarray([EARLY_TO_INT[x] for x in early_lick_keep], dtype=np.int8)
outputs_2d[:, 2, :] = trial_early_int[:, None]
```

iii. Same mapping as the reference.

## 8-a. What variables in the raw data is `output` *Tongue y-position* derived from?

i. From `Camera0_side_TongueTracking` in `BehavioralTimeSeries`, which contains (x, y, likelihood) data and timestamps.

ii.
```python
tongue_data = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["data"][()], ...)
tongue_timestamps_abs = np.asarray(
    h5file["acquisition"]["BehavioralTimeSeries"]["Camera0_side_TongueTracking"]["timestamps"][()], ...)
tongue_y_raw = tongue_data[:, 1]
tongue_likelihood_raw = tongue_data[:, 2]
```

iii. Same source as the reference.

## 8-b. What processing is involved in computing `output` *Tongue y-position*?

i. Frames with likelihood >= 0.9 are considered visible. The 40th and 60th percentiles of the visible raw tongue y values over the entire session define the class boundaries. For per-trial binning, the last frame before each bin's end time is used as the bin's value (not the mean of all frames in the bin). If no visible frame exists or the sampled frame has likelihood < 0.9, the bin gets class 3 (not visible).

ii.
```python
TONGUE_LIKELIHOOD_THRESHOLD = 0.9
...
visible_global = (
    np.isfinite(tongue_y_raw)
    & np.isfinite(tongue_likelihood_raw)
    & (tongue_likelihood_raw >= TONGUE_LIKELIHOOD_THRESHOLD)
)
tongue_q40, tongue_q60 = np.quantile(tongue_y_raw[visible_global], [0.4, 0.6])
```

```python
def align_tongue_to_bins(...):
    last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
    ...
    classes[visible & (sampled_y < visible_q40)] = 0
    classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
    classes[visible & (sampled_y > visible_q60)] = 2
```

iii. The AI chose a 0.9 likelihood threshold based on the bimodal distribution of likelihood values, arguing it's defensible for the "not visible" class. Percentiles are computed over raw visible frames rather than bin means. The last-frame-in-bin approach differs from the reference's bin-mean approach.

## 8-c. How is `output` *Tongue y-position* thresholded into categories?

i. Session-wide 40th and 60th percentiles of visible tongue y frames define boundaries. Values below the 40th percentile get class 0, between 40th and 60th (inclusive) get class 1, above the 60th percentile get class 2, and not-visible bins get class 3.

ii.
```python
classes[visible & (sampled_y < visible_q40)] = 0
classes[visible & (sampled_y >= visible_q40) & (sampled_y <= visible_q60)] = 1
classes[visible & (sampled_y > visible_q60)] = 2
```

iii. The boundary handling differs slightly from the reference: the AI includes the 60th percentile value in class 1 (middle), while the reference's `np.digitize` places it in class 2 (above).

## 8-d. How is `output` *Tongue y-position* aligned with the neural data?

i. For each bin defined by the go-cue-aligned grid, the last camera frame before the bin's end time is used. If that frame's timestamp falls within the bin and its likelihood meets the threshold, it provides the bin's tongue y value.

ii.
```python
def align_tongue_to_bins(...):
    bin_starts_abs = go_times_abs[:, None] + BIN_EDGES_REL[:-1][None, :]
    bin_ends_abs = go_times_abs[:, None] + BIN_EDGES_REL[1:][None, :]
    last_idx = np.searchsorted(tongue_timestamps_abs, bin_ends_abs, side="left") - 1
    ...
    last_times = tongue_timestamps_abs[chosen]
    flat_starts = bin_starts_abs.ravel()[flat_valid]
    keep = last_times >= flat_starts
```

iii. This "last frame" approach differs from the reference's "mean of all frames in the bin" approach. The reference computes bin means by assigning each frame to a bin via floor division and averaging, while the AI samples a single frame per bin.

## 9. How are minor mistakes in the data, e.g. missing data, handled?

i. Several cases are handled:
- Session with all NaN classifications: excluded (hardcoded and by zero-good-units check)
- Trials without neural coverage: excluded via `is_good_trials` intersection
- Trials outside spike time span: excluded via spike support mask
- Trials with all-zero neural activity: excluded post-hoc
- Trials with missing tone onset: excluded (tone_onsets_abs is NaN)
- Tongue frames with low likelihood: treated as not visible

ii.
```python
keep_trials = (
    (~auto_water)
    & (~free_water)
    & np.isfinite(tone_onsets_abs)
    & neural_coverage_mask
    & spike_support_mask
)
...
nonzero_neural_mask = np.asarray([np.any(trial) for trial in neural_trials], dtype=bool)
```

iii. The AI took a comprehensive approach to handling data quality issues, documented in CONVERSION_NOTES.md with specific counts of trials affected by each filter.

## 10-a. What are the most time-consuming steps of the code?

i. Reading NWB files via h5py and per-unit spike binning dominate. The full conversion completed in ~2.14 minutes for 173 sessions.

ii. N/A (timing is reported in conversion output)

iii. From CONVERSION_NOTES.md: "Per-session spike binning remains the dominant cost because every classifier-good unit must be aligned to every kept trial."

## 10-b. What loops in the code could have been vectorized to improve efficiency?

i. The per-unit spike binning loop iterates over each good unit individually. The choice reconstruction loop iterates per trial. The tongue alignment in `align_tongue_to_bins` is vectorized.

ii.
```python
for out_idx, unit_idx in enumerate(good_unit_indices):
    ...
    insertion_idx = np.searchsorted(spikes, bin_edges_abs, side="left")
```

```python
for trial_idx, go_time in enumerate(go_times_abs):
    ...  # reconstruct_choice_labels
```

iii. The per-unit loop is inherent to the ragged spike time storage. The choice reconstruction loop could potentially be vectorized but is not a bottleneck.

## 10-c. What processing does the code repeat multiple times?

i. Nothing significant is recomputed. Each NWB file is opened once and processed in a single pass.

ii. N/A

iii. The AI's code processes each session independently in a single loop.

## 10-d. What unnecessary processing does the code do that is discarded in downstream analyses?

i. The AI stores extensive intermediate data in the session result dictionary for plotting purposes (tongue_timestamps_abs, tongue_y_raw, tongue_likelihood_raw, tongue_y_visible_raw, go_times_abs, tone_rel_s, stim_on_rel_s, stim_off_rel_s, etc.). This data is used for `--show-processing` plots but is not included in the final pickle. Additionally, trial-level string labels (choice_labels, trial_outcome_labels, trial_early_labels) and distribution counts are computed for logging but not saved to the output.

ii.
```python
return {
    ...
    "tongue_timestamps_abs": tongue_timestamps_abs,
    "tongue_y_raw": tongue_y_raw,
    "tongue_likelihood_raw": tongue_likelihood_raw,
    "tongue_y_visible_raw": tongue_y_raw[visible_global],
    "tongue_classes": tongue_classes,
    "go_times_abs": go_keep,
    "tone_rel_s": tone_rel_s,
    "stim_on_rel_s": stim_on_rel_s,
    "stim_off_rel_s": stim_off_rel_s,
    ...
}
```

iii. This extra data is retained in memory for the entire conversion but only used if `--show-processing` is enabled.
